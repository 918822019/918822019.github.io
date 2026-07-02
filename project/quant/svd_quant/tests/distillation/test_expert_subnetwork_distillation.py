"""
单 Expert 子网络蒸馏实现

核心思想：
- 不加载完整 35B 模型
- 只加载目标 Expert + 其前后的 LayerNorm/RMSNorm + Router
- 构造一个最小可运行子图
- 用合成数据或缓存的激活值做前向传播
- 显存需求从 70GB 降至 <8GB

优势：
1. 在任何消费级 GPU 上都能跑
2. 无需完整模型的环境依赖
3. 可以快速验证 FSQ + Logits KL 蒸馏的效果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
import json
import time
from typing import Dict, List, Tuple, Optional

# 模型路径
model_path = r"D:\918822019.github.io\project\quant\model\Qwen3.5-35B-A3B"


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


class ExpertSubnetwork(nn.Module):
    """
    单 Expert 子网络
    
    包含：
    - 前置 RMSNorm
    - Expert (gate_proj + up_proj + down_proj)
    - Router（可选）
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 256,
        expert_idx: int = 0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.expert_idx = expert_idx
        
        # 前置 RMSNorm
        self.input_norm = RMSNorm(hidden_size)
        
        # Expert 权重（3D 张量）
        # gate_up_proj: (num_experts, intermediate_size * 2, hidden_size)
        # down_proj: (num_experts, hidden_size, intermediate_size)
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size * 2, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size)
        )
        
        # Router（可选）
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
    def forward(
        self,
        x: torch.Tensor,
        use_fsq: bool = False,
        fsq_gate_up: Optional[nn.Module] = None,
        fsq_down: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, hidden_size)
            use_fsq: 是否使用 FSQ 量化
            fsq_gate_up: gate_up_proj 的 FSQ 量化器
            fsq_down: down_proj 的 FSQ 量化器
            
        Returns:
            output: 输出张量
            router_logits: Router logits（可选）
        """
        # 前置 RMSNorm
        x_normed = self.input_norm(x)
        
        # 获取当前 Expert 的权重
        gate_up_weight = self.gate_up_proj[self.expert_idx]  # (intermediate_size * 2, hidden_size)
        down_weight = self.down_proj[self.expert_idx]  # (hidden_size, intermediate_size)
        
        # 应用 FSQ 量化（如果指定）
        if use_fsq and fsq_gate_up is not None:
            gate_up_weight = fsq_gate_up(gate_up_weight)
        if use_fsq and fsq_down is not None:
            down_weight = fsq_down(down_weight)
        
        # Expert 前向传播
        # gate_up_proj: (batch_size, seq_len, intermediate_size * 2)
        gate_up = F.linear(x_normed, gate_up_weight)
        
        # Split into gate and up
        gate, up = gate_up.chunk(2, dim=-1)
        
        # SiLU 激活
        gate = F.silu(gate)
        
        # Element-wise multiply
        intermediate = gate * up  # (batch_size, seq_len, intermediate_size)
        
        # down_proj: (batch_size, seq_len, hidden_size)
        output = F.linear(intermediate, down_weight)
        
        # Router logits（可选）
        router_logits = self.router(x_normed)
        
        return output, router_logits


class FSQQuantizer(nn.Module):
    """
    FSQ 量化器（简化版本）
    """
    
    def __init__(self, levels: int = 8):
        super().__init__()
        self.levels = levels
        self.min_val = None
        self.max_val = None
    
    def fit(self, x: torch.Tensor):
        """计算归一化参数"""
        self.min_val = x.min()
        self.max_val = x.max()
        if self.max_val - self.min_val < 1e-8:
            self.max_val = self.min_val + 1e-8
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """量化"""
        if self.min_val is None or self.max_val is None:
            self.fit(x)
        
        # 归一化到 [0, 1]
        x_normalized = (x - self.min_val) / (self.max_val - self.min_val)
        
        # 映射到 [0, levels-1] 并取整
        levels = float(self.levels)
        x_scaled = x_normalized * (levels - 1)
        x_rounded = torch.round(x_scaled)
        
        # 反归一化
        x_normalized_back = x_rounded / (levels - 1)
        x_quantized = x_normalized_back * (self.max_val - self.min_val) + self.min_val
        
        return x_quantized


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
    
    其中 H_target 是目标熵（如 3.0）
    
    优势：
    - 数学专家的 Logits 通常更尖锐（需要更高 T）
    - 通用对话专家更平滑（较低 T 即可）
    - 每个 Expert、每个 Batch 自动找到最佳软化程度
    """
    
    def __init__(self, target_entropy: float = 3.0, min_temperature: float = 1.0):
        self.target_entropy = target_entropy
        self.min_temperature = min_temperature
    
    def compute_temperature(self, logits: torch.Tensor) -> float:
        """
        根据 Logits 的熵计算自适应温度
        
        Args:
            logits: FP16 模型的 Logits
            
        Returns:
            temperature: 自适应温度
        """
        # 计算概率分布
        probs = F.softmax(logits, dim=-1)
        
        # 计算熵 H(p) = -sum(p * log(p))
        # 添加 epsilon 避免 log(0)
        epsilon = 1e-8
        entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)
        
        # 计算平均熵
        avg_entropy = entropy.mean().item()
        
        # 计算自适应温度
        # T = max(1.0, H(p) / H_target)
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


def load_expert_weights() -> Dict[str, torch.Tensor]:
    """
    加载 Expert 权重
    
    Returns:
        weights: 权重字典
    """
    print("加载 Expert 权重...")
    
    # 加载索引
    with open(f"{model_path}/model.safetensors.index.json", "r") as f:
        index_data = json.load(f)
    
    weight_map = index_data["weight_map"]
    
    # 选择一个 MoE 专家层
    moe_keys = [k for k in weight_map.keys() if "experts" in k]
    if not moe_keys:
        moe_keys = [k for k in weight_map.keys() if "gate" in k or "up" in k or "down" in k]
    
    # 找到 gate_up_proj 和 down_proj
    gate_up_key = None
    down_key = None
    
    for key in moe_keys:
        if "gate_up_proj" in key:
            gate_up_key = key
        elif "down_proj" in key:
            down_key = key
    
    if gate_up_key is None or down_key is None:
        raise ValueError("未找到 gate_up_proj 或 down_proj")
    
    # 加载权重
    weights = {}
    
    for key in [gate_up_key, down_key]:
        file_name = weight_map[key]
        file_path = f"{model_path}/{file_name}"
        
        with safe_open(file_path, framework="pt", device="cpu") as f:
            weights[key] = f.get_tensor(key)
            print(f"  加载 {key}: {weights[key].shape}")
    
    return weights


def create_synthetic_data(
    batch_size: int = 4,
    seq_len: int = 128,
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
    # 假设激活值有某种模式
    for i in range(batch_size):
        for j in range(seq_len):
            # 添加位置编码
            pos_encoding = torch.sin(torch.arange(hidden_size) * 0.01 * j)
            data[i, j] += pos_encoding * 0.1
    
    return data


def train_expert_subnetwork():
    """
    训练单 Expert 子网络
    """
    print("=" * 70)
    print("单 Expert 子网络蒸馏训练")
    print("=" * 70)
    
    # 加载权重
    weights = load_expert_weights()
    
    # 获取权重形状
    gate_up_key = [k for k in weights.keys() if "gate_up_proj" in k][0]
    down_key = [k for k in weights.keys() if "down_proj" in k][0]
    
    gate_up_shape = weights[gate_up_key].shape
    down_shape = weights[down_key].shape
    
    print(f"\n权重形状:")
    print(f"  gate_up_proj: {gate_up_shape}")
    print(f"  down_proj: {down_shape}")
    
    # 创建子网络
    # 处理不同的权重形状
    if len(gate_up_shape) == 3:
        num_experts = gate_up_shape[0]
        hidden_size = gate_up_shape[2]
    else:
        num_experts = 1
        hidden_size = gate_up_shape[1]
    
    if len(down_shape) == 3:
        intermediate_size = down_shape[2]
    else:
        intermediate_size = down_shape[1]
    
    print(f"\n模型参数:")
    print(f"  num_experts: {num_experts}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  intermediate_size: {intermediate_size}")
    
    # 创建子网络
    subnetwork = ExpertSubnetwork(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        expert_idx=0  # 使用第一个专家
    )
    
    # 加载权重
    # 处理不同的权重形状，并转换为 float
    if len(weights[gate_up_key].shape) == 3:
        subnetwork.gate_up_proj.data = weights[gate_up_key].float()
    else:
        # 如果是 2D，扩展为 3D
        subnetwork.gate_up_proj.data = weights[gate_up_key].float().unsqueeze(0)
    
    if len(weights[down_key].shape) == 3:
        subnetwork.down_proj.data = weights[down_key].float()
    else:
        # 如果是 2D，扩展为 3D
        subnetwork.down_proj.data = weights[down_key].float().unsqueeze(0)
    
    # 创建 FSQ 量化器
    fsq_gate_up = FSQQuantizer(levels=16)
    fsq_down = FSQQuantizer(levels=16)
    
    # 拟合 FSQ
    fsq_gate_up.fit(subnetwork.gate_up_proj[0])
    fsq_down.fit(subnetwork.down_proj[0])
    
    # 创建损失函数（使用自适应温度）
    loss_fn = LogitsKLDistillationLoss(
        temperature=2.0,
        use_adaptive_temperature=True,
        target_entropy=3.0
    )
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        list(subnetwork.parameters()) + list(fsq_gate_up.parameters()) + list(fsq_down.parameters()),
        lr=1e-4,
        weight_decay=0.01
    )
    
    # 创建动态 Loss 调度器
    total_steps = 1000
    loss_scheduler = DynamicLossScheduler(total_steps)
    
    # 训练循环
    print(f"\n开始训练（{total_steps} 步）...")
    
    losses = []
    temperatures = []
    
    for step in range(total_steps):
        # 创建合成数据
        x = create_synthetic_data(
            batch_size=4,
            seq_len=128,
            hidden_size=hidden_size
        )
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 前向传播（原始模型）
        with torch.no_grad():
            output_original, router_logits_original = subnetwork(x, use_fsq=False)
            # 保存原始输出，避免重复计算
            output_original_saved = output_original.clone()
        
        # 前向传播（量化模型）
        output_quantized, router_logits_quantized = subnetwork(
            x, use_fsq=True, fsq_gate_up=fsq_gate_up, fsq_down=fsq_down
        )
        
        # 获取当前阶段的权重
        mse_weight, kl_weight = loss_scheduler.get_weights()
        
        # 计算损失（使用自适应温度）
        total_loss, mse_loss, kl_loss, temperature = loss_fn(
            output_original_saved,
            output_quantized,
            mse_weight=mse_weight,
            kl_weight=kl_weight
        )
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        # 重新拟合 FSQ（避免梯度问题）
        fsq_gate_up.fit(subnetwork.gate_up_proj[0])
        fsq_down.fit(subnetwork.down_proj[0])
        
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
        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{total_steps}: "
                  f"Loss={total_loss.item():.4f}, "
                  f"MSE={mse_loss.item():.4f}, "
                  f"KL={kl_loss.item():.4f}, "
                  f"T={temperature:.2f}, "
                  f"Weights=({mse_weight:.1f}, {kl_weight:.1f})")
    
    # 计算最终误差
    print("\n" + "=" * 70)
    print("训练完成")
    print("=" * 70)
    
    # 测试量化误差
    with torch.no_grad():
        # 原始权重
        gate_up_original = subnetwork.gate_up_proj[0]
        down_original = subnetwork.down_proj[0]
        
        # 量化权重
        gate_up_quantized = fsq_gate_up(gate_up_original)
        down_quantized = fsq_down(down_original)
        
        # 计算误差
        gate_up_error = torch.norm(gate_up_original - gate_up_quantized) / torch.norm(gate_up_original)
        down_error = torch.norm(down_original - down_quantized) / torch.norm(down_original)
        
        print(f"\n量化误差:")
        print(f"  gate_up_proj: {gate_up_error.item()*100:.2f}%")
        print(f"  down_proj: {down_error.item()*100:.2f}%")
    
    # 计算压缩比
    bits_per_element = np.log2(16)  # FSQ-16
    original_bits = (gate_up_original.numel() + down_original.numel()) * 32
    compressed_bits = (gate_up_original.numel() + down_original.numel()) * bits_per_element
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
    
    # 保存结果
    output_file = "expert_subnetwork_distillation_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("单 Expert 子网络蒸馏训练结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("模型参数:\n")
        f.write(f"  num_experts: {num_experts}\n")
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
        f.write(f"  gate_up_proj: {gate_up_error.item()*100:.2f}%\n")
        f.write(f"  down_proj: {down_error.item()*100:.2f}%\n\n")
        
        f.write(f"压缩比: {compression_ratio:.2f}x\n")
        f.write(f"每元素位数: {bits_per_element:.2f}\n\n")
        
        f.write("自适应温度统计:\n")
        f.write(f"  平均温度: {avg_temp:.2f}\n")
        f.write(f"  温度标准差: {std_temp:.2f}\n")
        f.write(f"  最小温度: {min(temperatures):.2f}\n")
        f.write(f"  最大温度: {max(temperatures):.2f}\n\n")
        
        f.write("训练损失曲线:\n")
        for loss in losses:
            f.write(f"  Step {loss['step']}: "
                    f"Total={loss['total_loss']:.4f}, "
                    f"MSE={loss['mse_loss']:.4f}, "
                    f"KL={loss['kl_loss']:.4f}\n")
    
    print(f"\n结果已保存到: {output_file}")
    
    return losses


if __name__ == "__main__":
    # 运行训练
    losses = train_expert_subnetwork()