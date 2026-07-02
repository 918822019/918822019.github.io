"""
Qwen3.5-0.8B 模型蒸馏量化脚本 v2

改进版本：
1. 正确的蒸馏流程：原始模型（teacher）vs 量化模型（student）
2. 更好的FSQ量化器实现
3. 改进的蒸馏损失计算
4. 更详细的输出和结果保存
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 模型路径
model_path = str(Path(__file__).resolve().parents[3] / "data" / "models" / "Qwen3.5-0.8B")

# 输出目录
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# 模型配置
MODEL_CONFIG = {
    "hidden_size": 1024,
    "intermediate_size": 3584,
    "vocab_size": 248320,
    "num_hidden_layers": 24,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "rms_norm_eps": 1e-06,
}


class FSQQuantizer(nn.Module):
    """
    FSQ 量化器 v2
    
    改进：
    1. 更稳定的归一化
    2. 支持分组量化
    3. 更好的梯度传递
    4. 支持 Per-channel Salience（借鉴 AWQ 思想）
    """
    
    def __init__(self, levels: int = 16, group_size: int = 128, use_salience: bool = True):
        super().__init__()
        self.levels = levels
        self.group_size = group_size
        self.min_val = None
        self.max_val = None
        self._last_indices = None
        self.use_salience = use_salience
        self.salience_scale = None  # Per-channel 显著性缩放因子
    
    def fit(self, x: torch.Tensor):
        """计算归一化参数，包括 Per-channel Salience"""
        # 计算 Per-channel Salience（借鉴 AWQ 思想）
        if self.use_salience and x.dim() >= 2:
            # 计算每个通道的激活显著性
            # 使用每个通道的绝对值平均作为显著性
            channel_salience = torch.mean(torch.abs(x), dim=0)  # (hidden_size,)
            
            # 避免除零
            channel_salience = torch.clamp(channel_salience, min=1e-8)
            
            # 归一化到 [0.5, 2.0] 范围，避免极端值
            salience_min = channel_salience.min()
            salience_max = channel_salience.max()
            if salience_max - salience_min > 1e-8:
                self.salience_scale = 0.5 + 1.5 * (channel_salience - salience_min) / (salience_max - salience_min)
            else:
                self.salience_scale = torch.ones_like(channel_salience)
            
            # 应用 Salience 缩放
            x_scaled = x * self.salience_scale.unsqueeze(0)
        else:
            self.salience_scale = None
            x_scaled = x
        
        # 使用分组统计
        if x_scaled.dim() >= 2:
            # 对每个分组计算min/max
            x_flat = x_scaled.reshape(-1, self.group_size) if x_scaled.numel() > self.group_size else x_scaled.reshape(1, -1)
            self.min_val = x_flat.min(dim=1, keepdim=True)[0].detach()
            self.max_val = x_flat.max(dim=1, keepdim=True)[0].detach()
        else:
            self.min_val = x_scaled.min().detach()
            self.max_val = x_scaled.max().detach()
        
        # 避免除零
        range_val = self.max_val - self.min_val
        if isinstance(range_val, torch.Tensor):
            range_val = torch.clamp(range_val, min=1e-8)
        else:
            range_val = max(range_val, 1e-8)
        self.max_val = self.min_val + range_val
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """量化（使用 STE 保持梯度流，支持 Per-channel Salience）"""
        if self.min_val is None or self.max_val is None:
            self.fit(x)
        
        # 保存原始形状
        original_shape = x.shape
        
        # 应用 Per-channel Salience 缩放（如果启用）
        if self.use_salience and self.salience_scale is not None:
            # 量化前：W' = W * s
            x_scaled_by_salience = x * self.salience_scale.unsqueeze(0)
        else:
            x_scaled_by_salience = x
        
        # 展平以便处理
        x_flat = x_scaled_by_salience.reshape(-1)
        
        # 归一化到 [0, 1]
        if isinstance(self.min_val, torch.Tensor) and self.min_val.dim() > 0:
            # 分组归一化
            x_groups = x_flat.reshape(-1, self.group_size)
            min_vals = self.min_val[:x_groups.shape[0]]
            max_vals = self.max_val[:x_groups.shape[0]]
            x_normalized = (x_groups - min_vals) / (max_vals - min_vals)
            x_normalized = x_normalized.reshape(-1)
        else:
            # 全局归一化
            x_normalized = (x_flat - self.min_val) / (self.max_val - self.min_val)
        
        # 映射到 [0, levels-1] 并取整
        levels = float(self.levels)
        x_level_scaled = x_normalized * (levels - 1)
        x_rounded = torch.round(x_level_scaled)
        
        # 记录量化索引
        self._last_indices = x_rounded.detach().long()
        
        # 反归一化
        x_normalized_back = x_rounded / (levels - 1)
        
        if isinstance(self.min_val, torch.Tensor) and self.min_val.dim() > 0:
            # 分组反归一化
            x_norm_groups = x_normalized_back.reshape(-1, self.group_size)
            min_vals = self.min_val[:x_norm_groups.shape[0]]
            max_vals = self.max_val[:x_norm_groups.shape[0]]
            x_quantized_groups = x_norm_groups * (max_vals - min_vals) + min_vals
            x_quantized_salience = x_quantized_groups.reshape(-1)
        else:
            # 全局反归一化
            x_quantized_salience = x_normalized_back * (self.max_val - self.min_val) + self.min_val
        
        # 恢复原始形状
        x_quantized_salience = x_quantized_salience.reshape(original_shape)
        
        # 反量化后：\hat{W} = \hat{W'} / s
        if self.use_salience and self.salience_scale is not None:
            x_quantized = x_quantized_salience / self.salience_scale.unsqueeze(0)
        else:
            x_quantized = x_quantized_salience
        
        # 使用 Straight-Through Estimator 传递梯度
        x_quantized = x + (x_quantized - x).detach()
        
        return x_quantized
    
    def compute_utilization(self) -> Dict[str, float]:
        """计算码本利用率"""
        if self._last_indices is None:
            return {'utilization': 0.0, 'used_levels': 0, 'total_levels': self.levels}
        
        flat_indices = self._last_indices.reshape(-1)
        level_counts = torch.zeros(self.levels, dtype=torch.long)
        for idx in flat_indices:
            if 0 <= idx < self.levels:
                level_counts[idx] += 1
        
        used_levels = (level_counts > 0).sum().item()
        utilization = used_levels / self.levels
        
        return {
            'utilization': utilization,
            'used_levels': used_levels,
            'total_levels': self.levels
        }


class DynamicLossScheduler:
    """
    动态 Loss 退火策略（改进版）
    
    训练阶段：
    - Warmup (前 20%): MSE=1.0, KL=0.0  # 更长的Warmup，让scale和min_val稳定
    - Alignment (20-60%): MSE=0.3, KL=0.7
    - Fine-tune (60-100%): MSE=0.1, KL=0.9
    """
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
    
    def get_weights(self) -> Tuple[float, float]:
        """获取当前阶段的 MSE 和 KL 权重"""
        progress = self.current_step / self.total_steps
        
        if progress < 0.2:
            # Warmup 阶段（延长到20%，让FSQ参数稳定）
            mse_weight = 1.0
            kl_weight = 0.0
        elif progress < 0.6:
            # Alignment 阶段
            mse_weight = 0.3
            kl_weight = 0.7
        else:
            # Fine-tune 阶段
            mse_weight = 0.1
            kl_weight = 0.9
        
        return mse_weight, kl_weight
    
    def step(self):
        """更新步数"""
        self.current_step += 1


class AdaptiveTemperature:
    """
    自适应温度缩放
    
    根据 FP16 Logits 的熵动态计算 T：
    T_adaptive = max(1.0, H(p_teacher) / H_target)
    """
    
    def __init__(self, target_entropy: float = 3.0, min_temperature: float = 1.0):
        self.target_entropy = target_entropy
        self.min_temperature = min_temperature
    
    def compute_temperature(self, logits: torch.Tensor) -> float:
        """根据 Logits 的熵计算自适应温度"""
        # 计算概率分布
        probs = F.softmax(logits, dim=-1)
        
        # 计算熵 H(p) = -sum(p * log(p))
        epsilon = 1e-8
        entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)
        
        # 计算平均熵
        avg_entropy = entropy.mean().item()
        
        # 计算自适应温度
        temperature = max(self.min_temperature, avg_entropy / self.target_entropy)
        
        return temperature


class LogitsKLDistillationLoss(nn.Module):
    """
    Logits KL 蒸馏损失
    
    关键技巧：
    1. Temperature Scaling（T=2.0~4.0）
    2. Adaptive Temperature：根据 Logits 熵动态调整
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        use_adaptive_temperature: bool = True,
        target_entropy: float = 3.0
    ):
        super().__init__()
        self.temperature = temperature
        self.use_adaptive_temperature = use_adaptive_temperature
        self.adaptive_temp = AdaptiveTemperature(target_entropy=target_entropy)
    
    def forward(
        self,
        logits_original: torch.Tensor,
        logits_quantized: torch.Tensor,
        mse_weight: float = 0.5,
        kl_weight: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        计算蒸馏损失
        
        Args:
            logits_original: 原始模型的输出 logits
            logits_quantized: 量化模型的输出 logits
            mse_weight: MSE 损失的权重
            kl_weight: KL 散度的权重
            
        Returns:
            total_loss: 总损失
            mse_loss: MSE 损失
            kl_loss: KL 散度损失
            temperature: 使用的温度
        """
        # 计算自适应温度
        if self.use_adaptive_temperature:
            temperature = self.adaptive_temp.compute_temperature(logits_original)
        else:
            temperature = self.temperature
        
        # MSE 损失
        mse_loss = F.mse_loss(logits_quantized, logits_original)
        
        # Temperature Scaling
        logits_original_scaled = logits_original / temperature
        logits_quantized_scaled = logits_quantized / temperature
        
        # KL 散度
        probs_original = F.softmax(logits_original_scaled, dim=-1)
        log_probs_quantized = F.log_softmax(logits_quantized_scaled, dim=-1)
        
        kl_loss = F.kl_div(
            log_probs_quantized,
            probs_original,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # 总损失
        total_loss = mse_weight * mse_loss + kl_weight * kl_loss
        
        return total_loss, mse_loss, kl_loss, temperature


class ExpertWithFSQ(nn.Module):
    """
    带 FSQ 量化的 Expert
    
    改进：
    1. 支持分组量化
    2. 更好的权重初始化
    3. 支持 Per-channel Salience（借鉴 AWQ 思想）
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        fsq_levels: int = 16,
        group_size: int = 128,
        use_salience: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Expert 权重
        self.gate_proj = nn.Parameter(torch.randn(intermediate_size, hidden_size))
        self.up_proj = nn.Parameter(torch.randn(intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.randn(hidden_size, intermediate_size))
        
        # FSQ 量化器（支持 Per-channel Salience）
        self.fsq_gate = FSQQuantizer(levels=fsq_levels, group_size=group_size, use_salience=use_salience)
        self.fsq_up = FSQQuantizer(levels=fsq_levels, group_size=group_size, use_salience=use_salience)
        self.fsq_down = FSQQuantizer(levels=fsq_levels, group_size=group_size, use_salience=use_salience)
        
    def forward(self, x: torch.Tensor, use_fsq: bool = True) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入激活值 (batch_size, seq_len, hidden_size)
            use_fsq: 是否使用FSQ量化
            
        Returns:
            output: 输出张量 (batch_size, seq_len, hidden_size)
        """
        if use_fsq:
            # 量化权重
            gate_quantized = self.fsq_gate(self.gate_proj)
            up_quantized = self.fsq_up(self.up_proj)
            down_quantized = self.fsq_down(self.down_proj)
        else:
            # 使用原始权重
            gate_quantized = self.gate_proj
            up_quantized = self.up_proj
            down_quantized = self.down_proj
        
        # Expert 前向传播
        gate = F.silu(F.linear(x, gate_quantized))
        up = F.linear(x, up_quantized)
        intermediate = gate * up
        output = F.linear(intermediate, down_quantized)
        
        return output


def load_model_weights() -> Dict[str, torch.Tensor]:
    """
    加载0.8B模型权重
    
    Returns:
        weights: 权重字典
    """
    print("加载0.8B模型权重...")
    
    # 模型文件路径
    model_file = os.path.join(model_path, "model.safetensors-00001-of-00001.safetensors")
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"模型文件不存在: {model_file}")
    
    # 加载权重
    weights = {}
    with safe_open(model_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # 转换为Float32，因为SVD分解不支持BFloat16
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            weights[key] = tensor
    
    print(f"加载了 {len(weights)} 个张量")
    
    # 打印统计信息
    total_params = sum(p.numel() for p in weights.values())
    total_size_mb = sum(p.numel() * p.element_size() for p in weights.values()) / 1024 / 1024
    
    print(f"总参数量: {total_params:,}")
    print(f"总大小: {total_size_mb:.2f} MB")
    
    return weights


def create_synthetic_data(
    batch_size: int = 2,
    seq_len: int = 64,
    hidden_size: int = 1024
) -> torch.Tensor:
    """
    创建合成数据
    
    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        hidden_size: 隐藏层大小
        
    Returns:
        data: 合成数据
    """
    # 使用正态分布创建合成数据
    data = torch.randn(batch_size, seq_len, hidden_size)
    
    # 添加一些结构（模拟真实的激活值）
    for i in range(batch_size):
        for j in range(seq_len):
            # 添加位置编码
            pos_encoding = torch.sin(torch.arange(hidden_size) * 0.01 * j)
            data[i, j] += pos_encoding * 0.1
    
    return data


def train_distillation_v2():
    """
    训练蒸馏量化 v2
    
    改进：
    1. 正确的teacher-student蒸馏流程
    2. 更好的损失计算
    3. 更详细的结果保存
    """
    print("=" * 70)
    print("Qwen3.5-0.8B 模型蒸馏量化训练 v2")
    print("=" * 70)
    
    # 检查GPU可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 加载权重
    weights = load_model_weights()
    
    # 获取一个MoE专家层的权重
    gate_key = None
    up_key = None
    down_key = None
    
    for key in weights.keys():
        if "gate_proj" in key and "layers.0" in key:
            gate_key = key
        elif "up_proj" in key and "layers.0" in key:
            up_key = key
        elif "down_proj" in key and "layers.0" in key:
            down_key = key
    
    if gate_key is None or up_key is None or down_key is None:
        print("未找到gate_proj, up_proj, down_proj权重")
        print("使用随机权重进行测试...")
        hidden_size = MODEL_CONFIG["hidden_size"]
        intermediate_size = MODEL_CONFIG["intermediate_size"]
    else:
        print(f"\n找到权重:")
        print(f"  gate_proj: {gate_key} - {weights[gate_key].shape}")
        print(f"  up_proj: {up_key} - {weights[up_key].shape}")
        print(f"  down_proj: {down_key} - {weights[down_key].shape}")
        
        hidden_size = weights[gate_key].shape[1]
        intermediate_size = weights[gate_key].shape[0]
    
    print(f"\n模型参数:")
    print(f"  hidden_size: {hidden_size}")
    print(f"  intermediate_size: {intermediate_size}")
    
    # 创建teacher模型（原始权重，不量化）
    teacher = ExpertWithFSQ(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        fsq_levels=16,
        group_size=128,
        use_salience=True  # 启用 Per-channel Salience
    ).to(device)
    
    # 创建student模型（量化模型）
    student = ExpertWithFSQ(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        fsq_levels=16,
        group_size=128,
        use_salience=True  # 启用 Per-channel Salience
    ).to(device)
    
    # 加载权重到teacher
    if gate_key and up_key and down_key:
        teacher.gate_proj.data = weights[gate_key].to(device)
        teacher.up_proj.data = weights[up_key].to(device)
        teacher.down_proj.data = weights[down_key].to(device)
    
    # 复制权重到student
    student.gate_proj.data = teacher.gate_proj.data.clone()
    student.up_proj.data = teacher.up_proj.data.clone()
    student.down_proj.data = teacher.down_proj.data.clone()
    
    # 拟合FSQ（teacher不量化，student量化）
    student.fsq_gate.fit(student.gate_proj)
    student.fsq_up.fit(student.up_proj)
    student.fsq_down.fit(student.down_proj)
    
    # 创建损失函数
    loss_fn = LogitsKLDistillationLoss(
        temperature=2.0,
        use_adaptive_temperature=True,
        target_entropy=3.0
    ).to(device)
    
    # 创建优化器（只优化student）
    optimizer = torch.optim.AdamW(
        list(student.parameters()),
        lr=1e-4,
        weight_decay=0.01
    )
    
    # 创建动态Loss调度器
    total_steps = 1000
    loss_scheduler = DynamicLossScheduler(total_steps)
    
    # 训练循环
    print(f"\n开始训练（{total_steps} 步）...")
    
    losses = []
    temperatures = []
    quantization_errors = []
    
    start_time = time.time()
    
    for step in range(total_steps):
        # 创建合成数据
        x = create_synthetic_data(
            batch_size=2,
            seq_len=64,
            hidden_size=hidden_size
        ).to(device)
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 前向传播（teacher - 原始权重，不量化）
        with torch.no_grad():
            output_teacher = teacher(x, use_fsq=False)
        
        # 前向传播（student - 量化权重）
        output_student = student(x, use_fsq=True)
        
        # 获取当前阶段的权重
        mse_weight, kl_weight = loss_scheduler.get_weights()
        
        # 计算损失
        total_loss, mse_loss, kl_loss, temperature = loss_fn(
            output_teacher,
            output_student,
            mse_weight=mse_weight,
            kl_weight=kl_weight
        )
        
        # 反向传播
        total_loss.backward()
        
        # 梯度裁剪：防止FSQ参数梯度爆炸
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 重新拟合FSQ
        student.fsq_gate.fit(student.gate_proj)
        student.fsq_up.fit(student.up_proj)
        student.fsq_down.fit(student.down_proj)
        
        # 更新调度器
        loss_scheduler.step()
        
        # 记录损失
        losses.append({
            'step': step,
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'kl_loss': kl_loss.item(),
            'mse_weight': mse_weight,
            'kl_weight': kl_weight
        })
        
        # 记录温度
        temperatures.append(temperature)
        
        # 计算量化误差
        with torch.no_grad():
            gate_error = torch.norm(teacher.gate_proj - student.fsq_gate(student.gate_proj)) / torch.norm(teacher.gate_proj)
            up_error = torch.norm(teacher.up_proj - student.fsq_up(student.up_proj)) / torch.norm(teacher.up_proj)
            down_error = torch.norm(teacher.down_proj - student.fsq_down(student.down_proj)) / torch.norm(teacher.down_proj)
            
            quantization_errors.append({
                'step': step,
                'gate_error': gate_error.item(),
                'up_error': up_error.item(),
                'down_error': down_error.item()
            })
        
        # 打印进度
        if (step + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step + 1}/{total_steps}: "
                  f"Loss={total_loss.item():.6f}, "
                  f"MSE={mse_loss.item():.6f}, "
                  f"KL={kl_loss.item():.6f}, "
                  f"T={temperature:.2f}, "
                  f"Weights=({mse_weight:.1f}, {kl_weight:.1f}), "
                  f"GateErr={gate_error.item()*100:.2f}%, "
                  f"Time={elapsed:.1f}s")
    
    # 计算最终误差
    print("\n" + "=" * 70)
    print("训练完成")
    print("=" * 70)
    
    # 测试量化误差
    with torch.no_grad():
        gate_original = teacher.gate_proj
        up_original = teacher.up_proj
        down_original = teacher.down_proj
        
        gate_quantized = student.fsq_gate(gate_original)
        up_quantized = student.fsq_up(up_original)
        down_quantized = student.fsq_down(down_original)
        
        gate_error = torch.norm(gate_original - gate_quantized) / torch.norm(gate_original)
        up_error = torch.norm(up_original - up_quantized) / torch.norm(up_original)
        down_error = torch.norm(down_original - down_quantized) / torch.norm(down_original)
        
        print(f"\n量化误差:")
        print(f"  gate_proj: {gate_error.item()*100:.2f}%")
        print(f"  up_proj: {up_error.item()*100:.2f}%")
        print(f"  down_proj: {down_error.item()*100:.2f}%")
    
    # 计算压缩比
    bits_per_element = np.log2(16)  # FSQ-16
    original_bits = (gate_original.numel() + up_original.numel() + down_original.numel()) * 32
    compressed_bits = (gate_original.numel() + up_original.numel() + down_original.numel()) * bits_per_element
    compression_ratio = original_bits / compressed_bits
    
    print(f"\n压缩比: {compression_ratio:.2f}x")
    print(f"每元素位数: {bits_per_element:.2f}")
    
    # 统计温度使用情况
    avg_temp = np.mean(temperatures)
    std_temp = np.std(temperatures)
    print(f"\n自适应温度统计:")
    print(f"  平均温度: {avg_temp:.2f}")
    print(f"  温度标准差: {std_temp:.2f}")
    print(f"  最小温度: {min(temperatures):.2f}")
    print(f"  最大温度: {max(temperatures):.2f}")
    
    # 计算码本利用率
    gate_util = student.fsq_gate.compute_utilization()
    up_util = student.fsq_up.compute_utilization()
    down_util = student.fsq_down.compute_utilization()
    
    print(f"\n码本利用率:")
    print(f"  gate_proj: {gate_util['utilization']*100:.2f}% ({gate_util['used_levels']}/{gate_util['total_levels']})")
    print(f"  up_proj: {up_util['utilization']*100:.2f}% ({up_util['used_levels']}/{up_util['total_levels']})")
    print(f"  down_proj: {down_util['utilization']*100:.2f}% ({down_util['used_levels']}/{down_util['total_levels']})")
    
    # 保存结果
    output_file = output_dir / "distill_quantize_0_8b_v2_results.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Qwen3.5-0.8B 蒸馏量化训练结果 v2\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("模型参数:\n")
        f.write(f"  hidden_size: {hidden_size}\n")
        f.write(f"  intermediate_size: {intermediate_size}\n\n")
        
        f.write("训练配置:\n")
        f.write(f"  总步数: {total_steps}\n")
        f.write(f"  FSQ 级别: 16\n")
        f.write(f"  使用自适应温度: 是\n")
        f.write(f"  目标熵: 3.0\n\n")
        
        f.write("Loss 退火策略:\n")
        f.write("  Warmup (前 10%): MSE=1.0, KL=0.0\n")
        f.write("  Alignment (10-60%): MSE=0.3, KL=0.7\n")
        f.write("  Fine-tune (60-100%): MSE=0.1, KL=0.9\n\n")
        
        f.write("量化误差:\n")
        f.write(f"  gate_proj: {gate_error.item()*100:.2f}%\n")
        f.write(f"  up_proj: {up_error.item()*100:.2f}%\n")
        f.write(f"  down_proj: {down_error.item()*100:.2f}%\n\n")
        
        f.write(f"压缩比: {compression_ratio:.2f}x\n")
        f.write(f"每元素位数: {bits_per_element:.2f}\n\n")
        
        f.write("自适应温度统计:\n")
        f.write(f"  平均温度: {avg_temp:.2f}\n")
        f.write(f"  温度标准差: {std_temp:.2f}\n")
        f.write(f"  最小温度: {min(temperatures):.2f}\n")
        f.write(f"  最大温度: {max(temperatures):.2f}\n\n")
        
        f.write("码本利用率:\n")
        f.write(f"  gate_proj: {gate_util['utilization']*100:.2f}% ({gate_util['used_levels']}/{gate_util['total_levels']})\n")
        f.write(f"  up_proj: {up_util['utilization']*100:.2f}% ({up_util['used_levels']}/{up_util['total_levels']})\n")
        f.write(f"  down_proj: {down_util['utilization']*100:.2f}% ({down_util['used_levels']}/{down_util['total_levels']})\n\n")
        
        f.write("训练损失曲线:\n")
        for loss in losses:
            f.write(f"  Step {loss['step']}: "
                    f"Total={loss['total_loss']:.6f}, "
                    f"MSE={loss['mse_loss']:.6f}, "
                    f"KL={loss['kl_loss']:.6f}\n")
    
    print(f"\n结果已保存到: {output_file}")
    
    # 保存量化后的权重
    quantized_weights = {
        'gate_proj': student.fsq_gate(student.gate_proj).cpu(),
        'up_proj': student.fsq_up(student.up_proj).cpu(),
        'down_proj': student.fsq_down(student.down_proj).cpu(),
        'fsq_gate_min': student.fsq_gate.min_val.cpu() if student.fsq_gate.min_val is not None else None,
        'fsq_gate_max': student.fsq_gate.max_val.cpu() if student.fsq_gate.max_val is not None else None,
        'fsq_up_min': student.fsq_up.min_val.cpu() if student.fsq_up.min_val is not None else None,
        'fsq_up_max': student.fsq_up.max_val.cpu() if student.fsq_up.max_val is not None else None,
        'fsq_down_min': student.fsq_down.min_val.cpu() if student.fsq_down.min_val is not None else None,
        'fsq_down_max': student.fsq_down.max_val.cpu() if student.fsq_down.max_val is not None else None,
    }
    
    weights_file = output_dir / "quantized_weights_0_8b_v2.pt"
    torch.save(quantized_weights, weights_file)
    print(f"量化权重已保存到: {weights_file}")
    
    return losses


if __name__ == "__main__":
    # 运行训练
    losses = train_distillation_v2()