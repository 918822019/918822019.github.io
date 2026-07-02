"""
Logits 缓存 + 离线蒸馏实现

核心思想：
- 用一台有足够显存的机器预先跑一遍 FP16 模型
- 将目标层的输入激活值和输出 Logits 缓存到 NVMe SSD
- 训练 FSQ 时只读取缓存文件
- 彻底解耦推理环境和训练环境

优势：
1. 支持多卡并行训练 FSQ
2. 无需在训练时加载完整模型
3. 可以反复使用缓存数据
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle

# 模型路径
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
model_path = str(_PROJECT_ROOT / "data" / "models" / "Qwen3.5-35B-A3B")

_CACHE_DIR = str(_PROJECT_ROOT / "data" / "cache" / "logits_cache")
cache_dir = Path(_CACHE_DIR)
cache_dir.mkdir(parents=True, exist_ok=True)


class LogitsCache:
    """
    Logits 缓存管理器
    
    用于缓存和加载 FP16 模型的激活值和 Logits
    
    关键设计：保存 attention_mask 和 position_ids 确保对齐
    """
    
    def __init__(self, cache_dir: str = _CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 缓存文件路径
        self.activations_path = self.cache_dir / "activations.pt"
        self.logits_path = self.cache_dir / "logits.pt"
        self.attention_mask_path = self.cache_dir / "attention_mask.pt"
        self.position_ids_path = self.cache_dir / "position_ids.pt"
        self.metadata_path = self.cache_dir / "metadata.json"
    
    def save_cache(
        self,
        activations: torch.Tensor,
        logits: torch.Tensor,
        metadata: Dict,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ):
        """
        保存缓存
        
        Args:
            activations: 输入激活值
            logits: 输出 Logits
            metadata: 元数据
            attention_mask: Attention Mask（确保对齐）
            position_ids: Position IDs（确保对齐）
        """
        print(f"保存缓存到 {self.cache_dir}...")
        
        # 保存激活值
        torch.save(activations, self.activations_path)
        print(f"  激活值: {activations.shape} -> {self.activations_path}")
        
        # 保存 Logits
        torch.save(logits, self.logits_path)
        print(f"  Logits: {logits.shape} -> {self.logits_path}")
        
        # 保存 Attention Mask（关键对齐信息）
        if attention_mask is not None:
            torch.save(attention_mask, self.attention_mask_path)
            print(f"  Attention Mask: {attention_mask.shape} -> {self.attention_mask_path}")
        
        # 保存 Position IDs（关键对齐信息）
        if position_ids is not None:
            torch.save(position_ids, self.position_ids_path)
            print(f"  Position IDs: {position_ids.shape} -> {self.position_ids_path}")
        
        # 更新元数据
        metadata["has_attention_mask"] = attention_mask is not None
        metadata["has_position_ids"] = position_ids is not None
        
        # 保存元数据
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"  元数据: {self.metadata_path}")
    
    def load_cache(self) -> Tuple[torch.Tensor, torch.Tensor, Dict, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        加载缓存
        
        Returns:
            activations: 输入激活值
            logits: 输出 Logits
            metadata: 元数据
            attention_mask: Attention Mask（可选）
            position_ids: Position IDs（可选）
        """
        print(f"从 {self.cache_dir} 加载缓存...")
        
        # 检查文件是否存在
        if not self.activations_path.exists():
            raise FileNotFoundError(f"激活值文件不存在: {self.activations_path}")
        if not self.logits_path.exists():
            raise FileNotFoundError(f"Logits 文件不存在: {self.logits_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {self.metadata_path}")
        
        # 加载激活值
        activations = torch.load(self.activations_path)
        print(f"  激活值: {activations.shape}")
        
        # 加载 Logits
        logits = torch.load(self.logits_path)
        print(f"  Logits: {logits.shape}")
        
        # 加载元数据
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"  元数据: {metadata}")
        
        # 加载 Attention Mask（可选）
        attention_mask = None
        if metadata.get("has_attention_mask", False) and self.attention_mask_path.exists():
            attention_mask = torch.load(self.attention_mask_path)
            print(f"  Attention Mask: {attention_mask.shape}")
        
        # 加载 Position IDs（可选）
        position_ids = None
        if metadata.get("has_position_ids", False) and self.position_ids_path.exists():
            position_ids = torch.load(self.position_ids_path)
            print(f"  Position IDs: {position_ids.shape}")
        
        return activations, logits, metadata, attention_mask, position_ids
    
    def cache_exists(self) -> bool:
        """检查缓存是否存在"""
        return (
            self.activations_path.exists() and
            self.logits_path.exists() and
            self.metadata_path.exists()
        )


class FSQQuantizer(nn.Module):
    """
    FSQ 量化器
    
    使用 Straight-Through Estimator (STE) 实现梯度传递
    
    探针功能：
    - compute_utilization(): 码本利用率（每维度实际使用级别数 / 总级别数）
    - utilization < 0.3: 维度坍缩警告
    - utilization > 0.7: 正常利用
    """
    
    def __init__(self, levels: int = 8):
        super().__init__()
        self.levels = levels
        self.min_val = None
        self.max_val = None
        # 记录最近一次量化使用的级别索引（用于利用率分析）
        self._last_indices = None
    
    def fit(self, x: torch.Tensor):
        """计算归一化参数（detach 避免梯度图残留）"""
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
        
        # 记录量化索引（用于码本利用率分析）
        self._last_indices = x_rounded.detach().long()
        
        # 反归一化
        x_normalized_back = x_rounded / (levels - 1)
        x_quantized = x_normalized_back * (self.max_val - self.min_val) + self.min_val
        
        # 使用 Straight-Through Estimator 传递梯度
        x_quantized = x + (x_quantized - x).detach()
        
        return x_quantized
    
    def compute_utilization(self) -> Dict[str, float]:
        """
        计算码本利用率
        
        指标：每个 FSQ 维度的实际使用级别数 / 总级别数
        
        Returns:
            stats: 码本利用率统计
                - utilization: 平均利用率（0~1）
                - used_levels: 平均使用的级别数
                - total_levels: 总级别数
                - min_utilization: 最小维度利用率
                - max_utilization: 最大维度利用率
                - collapsed_dims: 坍缩维度数（利用率 < 0.3）
        """
        if self._last_indices is None:
            return {
                'utilization': 0.0,
                'used_levels': 0,
                'total_levels': self.levels,
                'min_utilization': 0.0,
                'max_utilization': 0.0,
                'collapsed_dims': 0
            }
        
        # 展平所有索引
        flat_indices = self._last_indices.reshape(-1)
        
        # 统计每个级别的使用次数
        level_counts = torch.zeros(self.levels, dtype=torch.long)
        for idx in flat_indices:
            if 0 <= idx < self.levels:
                level_counts[idx] += 1
        
        # 计算使用了多少个不同级别
        used_levels = (level_counts > 0).sum().item()
        utilization = used_levels / self.levels
        
        # 按维度分析（如果有多个维度）
        if self._last_indices.dim() >= 2:
            # 对每个维度计算利用率
            dim_utilizations = []
            for dim_idx in range(self._last_indices.shape[0]):
                dim_flat = self._last_indices[dim_idx].reshape(-1)
                dim_level_counts = torch.zeros(self.levels, dtype=torch.long)
                for idx in dim_flat:
                    if 0 <= idx < self.levels:
                        dim_level_counts[idx] += 1
                dim_used = (dim_level_counts > 0).sum().item()
                dim_utilizations.append(dim_used / self.levels)
            
            min_util = min(dim_utilizations)
            max_util = max(dim_utilizations)
            collapsed_dims = sum(1 for u in dim_utilizations if u < 0.3)
        else:
            min_util = utilization
            max_util = utilization
            collapsed_dims = 1 if utilization < 0.3 else 0
        
        return {
            'utilization': utilization,
            'used_levels': used_levels,
            'total_levels': self.levels,
            'min_utilization': min_util,
            'max_utilization': max_util,
            'collapsed_dims': collapsed_dims
        }


class DynamicLossScheduler:
    """
    动态 Loss 退火策略
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
    
    def compute_temperature_with_stats(
        self,
        logits: torch.Tensor
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算自适应温度并返回统计信息
        
        Args:
            logits: FP16 模型的 Logits
            
        Returns:
            temperature: 自适应温度
            stats: 统计信息
        """
        # 计算概率分布
        probs = F.softmax(logits, dim=-1)
        
        # 计算熵 H(p) = -sum(p * log(p))
        epsilon = 1e-8
        entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)
        
        # 计算统计信息
        avg_entropy = entropy.mean().item()
        std_entropy = entropy.std().item()
        min_entropy = entropy.min().item()
        max_entropy = entropy.max().item()
        
        # 计算自适应温度
        temperature = max(self.min_temperature, avg_entropy / self.target_entropy)
        
        stats = {
            "avg_entropy": avg_entropy,
            "std_entropy": std_entropy,
            "min_entropy": min_entropy,
            "max_entropy": max_entropy,
            "target_entropy": self.target_entropy,
            "computed_temperature": temperature
        }
        
        return temperature, stats


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
        kl_weight: float = 0.5,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        计算蒸馏损失
        
        Args:
            logits_original: 原始模型的输出 logits
            logits_quantized: 量化模型的输出 logits
            mse_weight: MSE 损失的权重
            kl_weight: KL 散度的权重
            attention_mask: 注意力掩码，用于过滤PAD token（1=有效，0=PAD）
            
        Returns:
            total_loss: 总损失
            mse_loss: MSE 损失
            kl_loss: KL 散度损失
            temperature: 使用的温度
        """
        # 计算自适应温度
        if self.use_adaptive_temperature:
            temperature, temp_stats = self.adaptive_temp.compute_temperature_with_stats(logits_original)
        else:
            temperature = self.temperature
        
        # 如果有attention_mask，只计算有效token的损失
        if attention_mask is not None:
            # 检查logits的维度
            if logits_original.dim() == 3:
                # 3D logits: (batch_size, seq_len, vocab_size)
                # attention_mask形状: (batch_size, seq_len)
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(logits_original)
                
                # 只选择有效token
                valid_indices = mask_expanded.bool()
                
                # 提取有效token的logits
                logits_original_valid = logits_original[valid_indices].reshape(-1, logits_original.shape[-1])
                logits_quantized_valid = logits_quantized[valid_indices].reshape(-1, logits_quantized.shape[-1])
            elif logits_original.dim() == 2:
                # 2D logits: (batch_size, vocab_size)
                # attention_mask形状: (batch_size, seq_len)
                # 这种情况下，logits已经是对整个序列的聚合，不需要mask
                # 但我们可以使用mask来加权
                # 计算mask的平均值作为权重
                mask_weight = attention_mask.float().mean(dim=-1)  # (batch_size,)
                
                # 使用mask_weight来加权损失
                # 但这里我们直接使用所有logits，因为已经是聚合后的
                logits_original_valid = logits_original
                logits_quantized_valid = logits_quantized
            else:
                # 其他维度，直接使用所有logits
                logits_original_valid = logits_original
                logits_quantized_valid = logits_quantized
            
            # MSE 损失（只计算有效token）
            mse_loss = F.mse_loss(logits_quantized_valid, logits_original_valid)
            
            # Temperature Scaling
            logits_original_scaled = logits_original_valid / temperature
            logits_quantized_scaled = logits_quantized_valid / temperature
            
            # KL 散度（只计算有效token）
            probs_original = F.softmax(logits_original_scaled, dim=-1)
            log_probs_quantized = F.log_softmax(logits_quantized_scaled, dim=-1)
            
            kl_loss = F.kl_div(
                log_probs_quantized,
                probs_original,
                reduction='batchmean'
            ) * (temperature ** 2)
        else:
            # 没有mask时，计算所有token的损失
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
        vocab_size: int,
        fsq_levels: int = 16
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        
        # Expert 权重
        self.gate_up_proj = nn.Parameter(
            torch.randn(intermediate_size * 2, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.randn(hidden_size, intermediate_size)
        )
        
        # 输出投影层（映射到 vocab_size）
        self.output_proj = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # FSQ 量化器
        self.fsq_gate_up = FSQQuantizer(levels=fsq_levels)
        self.fsq_down = FSQQuantizer(levels=fsq_levels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入激活值 (batch_size, seq_len, hidden_size)
            
        Returns:
            output: 输出 Logits (batch_size, seq_len, vocab_size)
        """
        # 量化权重
        gate_up_quantized = self.fsq_gate_up(self.gate_up_proj)
        down_quantized = self.fsq_down(self.down_proj)
        
        # Expert 前向传播
        gate_up = F.linear(x, gate_up_quantized)
        gate, up = gate_up.chunk(2, dim=-1)
        gate = F.silu(gate)
        intermediate = gate * up
        output = F.linear(intermediate, down_quantized)
        
        # 映射到 Logits
        logits = self.output_proj(output)
        
        return logits


def generate_synthetic_cache(
    num_samples: int = 1000,
    hidden_size: int = 1024,
    vocab_size: int = 32000
):
    """
    生成合成缓存数据（用于测试）
    
    Args:
        num_samples: 样本数量
        hidden_size: 隐藏层大小
        vocab_size: 词表大小
    """
    print("生成合成缓存数据...")
    
    # 生成激活值
    activations = torch.randn(num_samples, hidden_size)
    
    # 生成 Logits（模拟 FP16 模型的输出）
    # 假设 Logits 有某种模式
    logits = torch.randn(num_samples, vocab_size)
    
    # 添加一些结构
    for i in range(num_samples):
        # 假设某些 token 更可能被选中
        logits[i, :100] += 1.0  # 增加前 100 个 token 的概率
        logits[i, -100:] -= 1.0  # 降低后 100 个 token 的概率
    
    # 生成 Attention Mask（关键对齐信息）
    # 注意：这里生成真实的 attention_mask，而非简化处理
    # 模拟真实场景：部分 token 可能被 mask（padding）
    seq_len = 128  # 假设序列长度为 128
    attention_mask = torch.ones(num_samples, seq_len)  # 初始全 1
    
    # 模拟 padding：随机将后 20% 的 token 设为 0
    for i in range(num_samples):
        pad_len = int(seq_len * 0.2)  # 20% padding
        if pad_len > 0:
            attention_mask[i, -pad_len:] = 0
    
    # 生成 Position IDs（关键对齐信息）
    # 注意：这里生成真实的位置编码，而非简化处理
    position_ids = torch.zeros(num_samples, seq_len, dtype=torch.long)
    for i in range(num_samples):
        # 计算有效长度（非 padding 的 token 数量）
        valid_len = int(attention_mask[i].sum().item())
        position_ids[i, :valid_len] = torch.arange(valid_len)
        # padding 的位置保持 0（或保持最后一个有效位置，取决于具体实现）
        position_ids[i, valid_len:] = valid_len - 1  # 保持最后一个有效位置
    
    # 元数据
    metadata = {
        "num_samples": num_samples,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "model_type": "qwen3_5_moe",
        "expert_idx": 0,
        "layer_idx": 13,
        "tokenizer_config": {
            "max_length": 2048,
            "padding_side": "right",
            "attention_implementation": "flash_attention_2",
            "seq_len": seq_len
        }
    }
    
    # 保存缓存
    cache = LogitsCache()
    cache.save_cache(activations, logits, metadata, attention_mask, position_ids)
    
    print(f"生成完成: {num_samples} 个样本，序列长度: {seq_len}")


def train_with_cached_logits():
    """
    使用缓存的 Logits 进行训练
    
    关键对齐要求：
    1. 使用缓存的 attention_mask 和 position_ids，而非重新生成
    2. 确保 Tokenizer 配置、Max Length、Padding Side 与缓存一致
    3. 使用相同的 Attention Implementation（如 Flash Attention 2）
    """
    print("=" * 70)
    print("使用缓存 Logits 进行离线蒸馏训练")
    print("=" * 70)
    
    # GPU 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n计算设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA 版本: {torch.version.cuda}")
    
    # 检查缓存是否存在
    cache = LogitsCache()
    
    if not cache.cache_exists():
        print("缓存不存在，生成合成数据...")
        generate_synthetic_cache(
            num_samples=1000,
            hidden_size=1024,
            vocab_size=32000
        )
    
    # 加载缓存（包含对齐信息）
    activations, logits_original, metadata, attention_mask, position_ids = cache.load_cache()
    
    # GPU 加速：将数据移至设备
    print(f"\n将数据加载到 {device}...")
    activations = activations.to(device)
    logits_original = logits_original.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if position_ids is not None:
        position_ids = position_ids.to(device)
    
    hidden_size = metadata["hidden_size"]
    vocab_size = metadata["vocab_size"]
    
    print(f"\n数据信息:")
    print(f"  激活值形状: {activations.shape}")
    print(f"  Logits 形状: {logits_original.shape}")
    print(f"  隐藏层大小: {hidden_size}")
    print(f"  词表大小: {vocab_size}")
    
    # 关键对齐检查：确保使用缓存的 attention_mask 和 position_ids
    if attention_mask is not None:
        print(f"  Attention Mask: {attention_mask.shape} [已缓存，训练时强制使用]")
    else:
        print("  警告: 未找到缓存的 Attention Mask，可能导致对齐问题")
        
    if position_ids is not None:
        print(f"  Position IDs: {position_ids.shape} [已缓存，训练时强制使用]")
    else:
        print("  警告: 未找到缓存的 Position IDs，可能导致对齐问题")
    
    # 检查对齐配置
    if "tokenizer_config" in metadata:
        tokenizer_config = metadata["tokenizer_config"]
        print(f"  Tokenizer 配置: {tokenizer_config}")
        
        # 验证关键配置
        required_configs = ["max_length", "padding_side", "attention_implementation"]
        for config_key in required_configs:
            if config_key not in tokenizer_config:
                print(f"  警告: 缓存缺少关键配置 '{config_key}'，可能导致对齐问题")
    
    # 创建 Expert 模型（GPU 加速）
    intermediate_size = hidden_size * 4  # 假设 intermediate_size = 4 * hidden_size
    
    expert = ExpertWithFSQ(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        fsq_levels=16
    ).to(device)
    
    print(f"\n模型已加载到 {device}")
    if device.type == "cuda":
        print(f"模型显存占用: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    
    # 创建损失函数（使用自适应温度）
    loss_fn = LogitsKLDistillationLoss(
        temperature=2.0,
        use_adaptive_temperature=True,
        target_entropy=3.0
    ).to(device)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        expert.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )
    
    # 创建动态 Loss 调度器
    total_steps = 500
    loss_scheduler = DynamicLossScheduler(total_steps)
    
    # GPU 训练模式
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # 启用 cuDNN 自动调优
        scaler = torch.amp.GradScaler('cuda')  # 混合精度训练
        print("\n启用混合精度训练 (FP16/FP32)")
    
    # 训练循环
    print(f"\n开始训练（{total_steps} 步）...")
    print("探针监控: 码本利用率 | MSE-KL 相关性 | T 分布")
    print("-" * 70)
    
    losses = []
    temperatures = []
    mse_history = []  # 用于计算 MSE-KL 相关性
    kl_history = []   # 用于计算 MSE-KL 相关性
    
    for step in range(total_steps):
        # 随机采样一个批次
        idx = torch.randint(0, len(activations), (32,))
        x_batch = activations[idx]
        logits_batch = logits_original[idx]
        
        # 关键对齐：使用缓存的 attention_mask 和 position_ids
        # 而非重新生成，确保 KV Cache / Attention Mask 的一致性
        batch_attention_mask = attention_mask[idx] if attention_mask is not None else None
        batch_position_ids = position_ids[idx] if position_ids is not None else None
        
        # 前向传播（GPU 混合精度）
        if device.type == "cuda":
            with torch.amp.autocast('cuda'):
                output = expert(x_batch)
                logits_quantized = output[:, :vocab_size]
                
                # 获取当前阶段的权重
                mse_weight, kl_weight = loss_scheduler.get_weights()
                
                # 计算损失（使用自适应温度，过滤PAD token）
                total_loss, mse_loss, kl_loss, temperature = loss_fn(
                    logits_batch,
                    logits_quantized,
                    mse_weight=mse_weight,
                    kl_weight=kl_weight,
                    attention_mask=batch_attention_mask
                )
            
            # 混合精度反向传播
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # CPU 模式
            output = expert(x_batch)
            logits_quantized = output[:, :vocab_size]
            
            # 获取当前阶段的权重
            mse_weight, kl_weight = loss_scheduler.get_weights()
            
            # 计算损失（使用自适应温度，过滤PAD token）
            total_loss, mse_loss, kl_loss, temperature = loss_fn(
                logits_batch,
                logits_quantized,
                mse_weight=mse_weight,
                kl_weight=kl_weight,
                attention_mask=batch_attention_mask
            )
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # 更新调度器
        loss_scheduler.step()
        
        # 记录损失和温度
        losses.append({
            'step': step,
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'kl_loss': kl_loss.item(),
            'mse_weight': mse_weight,
            'kl_weight': kl_weight
        })
        temperatures.append(temperature)
        
        # 记录 MSE 和 KL 历史（用于相关性计算）
        mse_history.append(mse_loss.item())
        kl_history.append(kl_loss.item())
        
        # 每 100 步打印详细探针信息
        if (step + 1) % 100 == 0:
            # 探针 1: 码本利用率
            gate_up_util = expert.fsq_gate_up.compute_utilization()
            down_util = expert.fsq_down.compute_utilization()
            
            # 探针 2: MSE-KL 相关性（滑动窗口 50 步）
            window = min(50, len(mse_history))
            if window >= 10:
                mse_arr = np.array(mse_history[-window:])
                kl_arr = np.array(kl_history[-window:])
                # 皮尔逊相关系数
                if np.std(mse_arr) > 1e-8 and np.std(kl_arr) > 1e-8:
                    correlation = np.corrcoef(mse_arr, kl_arr)[0, 1]
                else:
                    correlation = 0.0
            else:
                correlation = 0.0
            
            # 探针 3: T 分布统计（滑动窗口）
            temp_window = temperatures[-min(100, len(temperatures)):]
            temp_mean = np.mean(temp_window)
            temp_std = np.std(temp_window)
            
            # 确定训练阶段
            progress = (step + 1) / total_steps
            if progress < 0.1:
                phase = "Warmup"
            elif progress < 0.6:
                phase = "Alignment"
            else:
                phase = "Fine-tune"
            
            # 打印探针信息
            print(f"\nStep {step + 1}/{total_steps} [{phase}]")
            print(f"  Loss: total={total_loss.item():.4f}, MSE={mse_loss.item():.4f}, KL={kl_loss.item():.4f}")
            print(f"  Weights: MSE={mse_weight:.2f}, KL={kl_weight:.2f}")
            print(f"  T_adaptive: mean={temp_mean:.2f}, std={temp_std:.4f}")
            print(f"  MSE-KL 相关性: {correlation:.4f}", end="")
            if correlation > 0.7:
                print(" [WARNING: 正相关过高，FSQ 仍在做纯数值拟合]")
            elif correlation < 0.3:
                print(" [OK: 低相关，语义对齐生效]")
            else:
                print("")
            print(f"  码本利用率 (gate_up): {gate_up_util['utilization']:.2f} ({gate_up_util['used_levels']}/{gate_up_util['total_levels']}), 坍缩维度: {gate_up_util['collapsed_dims']}")
            print(f"  码本利用率 (down):    {down_util['utilization']:.2f} ({down_util['used_levels']}/{down_util['total_levels']}), 坍缩维度: {down_util['collapsed_dims']}")
            
            # GPU 显存监控
            if device.type == "cuda":
                gpu_mem = torch.cuda.memory_allocated(0) / 1024**2
                gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1024**2
                print(f"  GPU 显存: 已用 {gpu_mem:.1f} MB, 预留 {gpu_mem_reserved:.1f} MB")
            
            # 坍缩警告
            if gate_up_util['collapsed_dims'] > 0 or down_util['collapsed_dims'] > 0:
                print(f"  [WARNING] 检测到维度坍缩！考虑降低学习率或增加码本级别数")
    
    # 计算最终误差
    print("\n" + "=" * 70)
    print("训练完成")
    print("=" * 70)
    
    # 测试量化误差
    with torch.no_grad():
        # 原始权重
        gate_up_original = expert.gate_up_proj
        down_original = expert.down_proj
        
        # 量化权重
        gate_up_quantized = expert.fsq_gate_up(gate_up_original)
        down_quantized = expert.fsq_down(down_original)
        
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
    output_file = "logits_cache_distillation_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Logits 缓存 + 离线蒸馏训练结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("数据信息:\n")
        f.write(f"  激活值形状: {activations.shape}\n")
        f.write(f"  Logits 形状: {logits_original.shape}\n")
        f.write(f"  隐藏层大小: {hidden_size}\n")
        f.write(f"  词表大小: {vocab_size}\n\n")
        
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
    losses = train_with_cached_logits()