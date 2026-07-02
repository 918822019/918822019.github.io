"""
Qwen3.5-0.8B 模型蒸馏量化脚本

使用FSQ+Logits KL蒸馏方案对0.8B模型进行量化压缩。

核心特性：
1. 使用0.8B模型作为参考模型（teacher model）
2. 实现FSQ量化 + Logits KL蒸馏
3. 动态Loss退火策略
4. 自适应温度缩放
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


class RMSNorm(nn.Module):
    """
    RMSNorm 实现（Qwen3.5 使用 RMSNorm 而非 LayerNorm）
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FSQQuantizer(nn.Module):
    """
    FSQ 量化器
    
    使用 Straight-Through Estimator (STE) 实现梯度传递
    """
    
    def __init__(self, levels: int = 16):
        super().__init__()
        self.levels = levels
        self.min_val = None
        self.max_val = None
        self._last_indices = None
    
    def fit(self, x: torch.Tensor):
        """计算归一化参数"""
        self.min_val = x.min().detach()
        self.max_val = x.max().detach()
        if (self.max_val - self.min_val).item() < 1e-8:
            self.max_val = self.min_val + 1e-8
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """量化（使用 STE 保持梯度流）"""
        if self.min_val is None or self.max_val is None:
            self.fit(x)
        
        # 归一化到 [0, 1]
        x_normalized = (x - self.min_val) / (self.max_val - self.min_val)
        
        # 映射到 [0, levels-1] 并取整
        levels = float(self.levels)
        x_scaled = x_normalized * (levels - 1)
        x_rounded = torch.round(x_scaled)
        
        # 记录量化索引
        self._last_indices = x_rounded.detach().long()
        
        # 反归一化
        x_normalized_back = x_rounded / (levels - 1)
        x_quantized = x_normalized_back * (self.max_val - self.min_val) + self.min_val
        
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
    动态 Loss 退火策略
    
    训练阶段：
    - Warmup (前 10%): MSE=1.0, KL=0.0
    - Alignment (10-60%): MSE=0.3, KL=0.7
    - Fine-tune (60-100%): MSE=0.1, KL=0.9
    """
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
    
    def get_weights(self) -> Tuple[float, float]:
        """获取当前阶段的 MSE 和 KL 权重"""
        progress = self.current_step / self.total_steps
        
        if progress < 0.1:
            # Warmup 阶段
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
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        fsq_levels: int = 16
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Expert 权重
        self.gate_proj = nn.Parameter(torch.randn(intermediate_size, hidden_size))
        self.up_proj = nn.Parameter(torch.randn(intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.randn(hidden_size, intermediate_size))
        
        # FSQ 量化器
        self.fsq_gate = FSQQuantizer(levels=fsq_levels)
        self.fsq_up = FSQQuantizer(levels=fsq_levels)
        self.fsq_down = FSQQuantizer(levels=fsq_levels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入激活值 (batch_size, seq_len, hidden_size)
            
        Returns:
            output: 输出张量 (batch_size, seq_len, hidden_size)
        """
        # 量化权重
        gate_quantized = self.fsq_gate(self.gate_proj)
        up_quantized = self.fsq_up(self.up_proj)
        down_quantized = self.fsq_down(self.down_proj)
        
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


def train_distillation():
    """
    训练蒸馏量化
    """
    print("=" * 70)
    print("Qwen3.5-0.8B 模型蒸馏量化训练")
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
    # 查找gate_proj, up_proj, down_proj
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
        # 使用随机权重
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
    
    # 创建带FSQ量化的Expert
    expert = ExpertWithFSQ(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        fsq_levels=16
    ).to(device)
    
    # 加载权重（如果找到）
    if gate_key and up_key and down_key:
        expert.gate_proj.data = weights[gate_key].to(device)
        expert.up_proj.data = weights[up_key].to(device)
        expert.down_proj.data = weights[down_key].to(device)
    
    # 拟合FSQ
    expert.fsq_gate.fit(expert.gate_proj)
    expert.fsq_up.fit(expert.up_proj)
    expert.fsq_down.fit(expert.down_proj)
    
    # 创建损失函数
    loss_fn = LogitsKLDistillationLoss(
        temperature=2.0,
        use_adaptive_temperature=True,
        target_entropy=3.0
    ).to(device)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        list(expert.parameters()),
        lr=1e-4,
        weight_decay=0.01
    )
    
    # 创建动态Loss调度器
    total_steps = 500
    loss_scheduler = DynamicLossScheduler(total_steps)
    
    # 训练循环
    print(f"\n开始训练（{total_steps} 步）...")
    
    losses = []
    temperatures = []
    
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
        
        # 前向传播（原始模型）
        with torch.no_grad():
            output_original = expert(x)
            # 保存原始输出，避免重复计算
            output_original_saved = output_original.clone()
        
        # 前向传播（量化模型）
        output_quantized = expert(x)
        
        # 获取当前阶段的权重
        mse_weight, kl_weight = loss_scheduler.get_weights()
        
        # 计算损失
        total_loss, mse_loss, kl_loss, temperature = loss_fn(
            output_original_saved,
            output_quantized,
            mse_weight=mse_weight,
            kl_weight=kl_weight
        )
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        # 重新拟合FSQ
        expert.fsq_gate.fit(expert.gate_proj)
        expert.fsq_up.fit(expert.up_proj)
        expert.fsq_down.fit(expert.down_proj)
        
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
        
        # 打印进度
        if (step + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step + 1}/{total_steps}: "
                  f"Loss={total_loss.item():.4f}, "
                  f"MSE={mse_loss.item():.4f}, "
                  f"KL={kl_loss.item():.4f}, "
                  f"T={temperature:.2f}, "
                  f"Weights=({mse_weight:.1f}, {kl_weight:.1f}), "
                  f"Time={elapsed:.1f}s")
    
    # 计算最终误差
    print("\n" + "=" * 70)
    print("训练完成")
    print("=" * 70)
    
    # 测试量化误差
    with torch.no_grad():
        # 原始权重
        gate_original = expert.gate_proj
        up_original = expert.up_proj
        down_original = expert.down_proj
        
        # 量化权重
        gate_quantized = expert.fsq_gate(gate_original)
        up_quantized = expert.fsq_up(up_original)
        down_quantized = expert.fsq_down(down_original)
        
        # 计算误差
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
    gate_util = expert.fsq_gate.compute_utilization()
    up_util = expert.fsq_up.compute_utilization()
    down_util = expert.fsq_down.compute_utilization()
    
    print(f"\n码本利用率:")
    print(f"  gate_proj: {gate_util['utilization']*100:.2f}% ({gate_util['used_levels']}/{gate_util['total_levels']})")
    print(f"  up_proj: {up_util['utilization']*100:.2f}% ({up_util['used_levels']}/{up_util['total_levels']})")
    print(f"  down_proj: {down_util['utilization']*100:.2f}% ({down_util['used_levels']}/{down_util['total_levels']})")
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "distill_quantize_0_8b_results.txt")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Qwen3.5-0.8B 蒸馏量化训练结果\n")
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
                    f"Total={loss['total_loss']:.4f}, "
                    f"MSE={loss['mse_loss']:.4f}, "
                    f"KL={loss['kl_loss']:.4f}\n")
    
    print(f"\n结果已保存到: {output_file}")
    
    # 保存量化后的权重
    quantized_weights = {
        'gate_proj': expert.fsq_gate(expert.gate_proj).cpu(),
        'up_proj': expert.fsq_up(expert.up_proj).cpu(),
        'down_proj': expert.fsq_down(expert.down_proj).cpu(),
    }
    
    weights_file = os.path.join(output_dir, "quantized_weights_0_8b.pt")
    torch.save(quantized_weights, weights_file)
    print(f"量化权重已保存到: {weights_file}")
    
    return losses


if __name__ == "__main__":
    # 运行训练
    losses = train_distillation()