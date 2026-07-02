"""
Qwen3.5-0.8B 模型多比特FSQ量化实验

实验目标：
1. 测试FSQ-16 (4-bit) vs FSQ-8 (3-bit) vs FSQ-4 (2-bit)
2. 实现非均匀熵编码 (Huffman) 压缩
3. 实现残差降级 (Residual FSQ) 量化
4. 对比不同方案的压缩比和量化误差

核心发现验证：
- FSQ+KL蒸馏让原本不可用的2-bit/3-bit变得可用了
- FSQ诱导的低熵分布可以进一步压缩到2.5-3.2 BPW
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
from collections import Counter
import heapq
from dataclasses import dataclass

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_ROOT = Path(__file__).resolve().parents[3]
model_path = str(_ROOT / "data" / "models" / "Qwen3.5-0.8B")

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


@dataclass
class HuffmanNode:
    """Huffman树节点"""
    symbol: int
    freq: int
    left: Optional['HuffmanNode'] = None
    right: Optional['HuffmanNode'] = None
    
    def __lt__(self, other):
        return self.freq < other.freq


class HuffmanCoder:
    """Huffman编码器/解码器"""
    
    def __init__(self):
        self.codes = {}
        self.decoder = {}
    
    def build_tree(self, freq_map: Dict[int, int]) -> HuffmanNode:
        """构建Huffman树"""
        heap = []
        for symbol, freq in freq_map.items():
            heapq.heappush(heap, HuffmanNode(symbol, freq))
        
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            parent = HuffmanNode(-1, left.freq + right.freq, left, right)
            heapq.heappush(heap, parent)
        
        return heap[0] if heap else None
    
    def generate_codes(self, node: HuffmanNode, code: str = ""):
        """生成Huffman编码表"""
        if node is None:
            return
        
        if node.symbol != -1:
            self.codes[node.symbol] = code if code else "0"
            self.decoder[code if code else "0"] = node.symbol
            return
        
        self.generate_codes(node.left, code + "0")
        self.generate_codes(node.right, code + "1")
    
    def encode(self, data: List[int]) -> Tuple[str, int]:
        """
        编码数据
        
        Returns:
            encoded: 编码后的比特串
            total_bits: 总比特数
        """
        # 统计频率
        freq_map = Counter(data)
        
        # 构建Huffman树
        root = self.build_tree(freq_map)
        
        # 生成编码表
        self.codes = {}
        self.decoder = {}
        self.generate_codes(root)
        
        # 编码数据
        encoded = ''.join(self.codes[x] for x in data)
        
        return encoded, len(encoded)
    
    def decode(self, encoded: str, length: int) -> List[int]:
        """解码数据"""
        decoded = []
        current_code = ""
        
        for bit in encoded:
            current_code += bit
            if current_code in self.decoder:
                decoded.append(self.decoder[current_code])
                current_code = ""
        
        return decoded[:length]


class FSQQuantizer(nn.Module):
    """
    FSQ量化器
    
    支持不同级别：
    - FSQ-16 (4-bit): 16个量化级别
    - FSQ-8 (3-bit): 8个量化级别
    - FSQ-4 (2-bit): 4个量化级别
    """
    
    def __init__(self, levels: int = 16, group_size: int = 128):
        super().__init__()
        self.levels = levels
        self.group_size = group_size
        self.min_val = None
        self.max_val = None
        self._last_indices = None
    
    def fit(self, x: torch.Tensor):
        """计算归一化参数"""
        if x.dim() >= 2:
            x_flat = x.reshape(-1, self.group_size) if x.numel() > self.group_size else x.reshape(1, -1)
            self.min_val = x_flat.min(dim=1, keepdim=True)[0].detach()
            self.max_val = x_flat.max(dim=1, keepdim=True)[0].detach()
        else:
            self.min_val = x.min().detach()
            self.max_val = x.max().detach()
        
        range_val = self.max_val - self.min_val
        if isinstance(range_val, torch.Tensor):
            range_val = torch.clamp(range_val, min=1e-8)
        else:
            range_val = max(range_val, 1e-8)
        self.max_val = self.min_val + range_val
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """量化（使用STE保持梯度流）"""
        if self.min_val is None or self.max_val is None:
            self.fit(x)
        
        original_shape = x.shape
        x_flat = x.reshape(-1)
        
        if isinstance(self.min_val, torch.Tensor) and self.min_val.dim() > 0:
            x_groups = x_flat.reshape(-1, self.group_size)
            min_vals = self.min_val[:x_groups.shape[0]]
            max_vals = self.max_val[:x_groups.shape[0]]
            x_normalized = (x_groups - min_vals) / (max_vals - min_vals)
            x_normalized = x_normalized.reshape(-1)
        else:
            x_normalized = (x_flat - self.min_val) / (self.max_val - self.min_val)
        
        levels = float(self.levels)
        x_scaled = x_normalized * (levels - 1)
        x_rounded = torch.round(x_scaled)
        
        # 记录量化索引
        self._last_indices = x_rounded.detach().long()
        
        x_normalized_back = x_rounded / (levels - 1)
        
        if isinstance(self.min_val, torch.Tensor) and self.min_val.dim() > 0:
            x_norm_groups = x_normalized_back.reshape(-1, self.group_size)
            min_vals = self.min_val[:x_norm_groups.shape[0]]
            max_vals = self.max_val[:x_norm_groups.shape[0]]
            x_quantized_groups = x_norm_groups * (max_vals - min_vals) + min_vals
            x_quantized = x_quantized_groups.reshape(-1)
        else:
            x_quantized = x_normalized_back * (self.max_val - self.min_val) + self.min_val
        
        x_quantized = x_quantized.reshape(original_shape)
        x_quantized = x + (x_quantized - x).detach()
        
        return x_quantized
    
    def get_indices(self) -> Optional[torch.Tensor]:
        """获取量化索引"""
        return self._last_indices
    
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
            'total_levels': self.levels,
            'level_counts': level_counts.tolist()
        }


class ResidualFSQQuantizer(nn.Module):
    """
    残差FSQ量化器
    
    策略：
    - L1 (粗量化): 使用FSQ-8 (3-bit) 捕获主干
    - L2 (残差量化): 对残差使用FSQ-4 (2-bit) 微调
    """
    
    def __init__(
        self,
        l1_levels: int = 8,
        l2_levels: int = 4,
        group_size: int = 128,
        residual_threshold: float = 0.1
    ):
        super().__init__()
        self.l1_levels = l1_levels
        self.l2_levels = l2_levels
        self.group_size = group_size
        self.residual_threshold = residual_threshold
        
        # L1和L2量化器
        self.fsq_l1 = FSQQuantizer(levels=l1_levels, group_size=group_size)
        self.fsq_l2 = FSQQuantizer(levels=l2_levels, group_size=group_size)
        
        # 残差掩码（哪些通道需要L2）
        self.residual_mask = None
    
    def fit(self, x: torch.Tensor):
        """计算归一化参数"""
        # 先拟合L1
        self.fsq_l1.fit(x)
        
        # 计算L1量化后的残差
        x_l1 = self.fsq_l1(x)
        residual = x - x_l1.detach()
        
        # 拟合L2
        self.fsq_l2.fit(residual)
        
        # 计算残差掩码（残差大于阈值的通道）
        residual_norm = torch.norm(residual.reshape(-1, self.group_size), dim=1)
        threshold = residual_norm.mean() * self.residual_threshold
        self.residual_mask = (residual_norm > threshold).float()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """量化（L1 + 可选L2）"""
        if self.fsq_l1.min_val is None:
            self.fit(x)
        
        # L1量化
        x_l1 = self.fsq_l1(x)
        
        # 计算残差
        residual = x - x_l1
        
        # L2量化（仅对需要的通道）
        if self.residual_mask is not None and self.residual_mask.sum() > 0:
            residual_flat = residual.reshape(-1, self.group_size)
            mask_expanded = self.residual_mask[:residual_flat.shape[0]].unsqueeze(1)
            
            # 对需要L2的通道进行量化
            residual_quantized = self.fsq_l2(residual)
            residual_final = residual * (1 - mask_expanded) + residual_quantized * mask_expanded
        else:
            residual_final = residual
        
        return x_l1 + residual_final
    
    def get_compression_stats(self) -> Dict[str, float]:
        """获取压缩统计信息"""
        if self.residual_mask is None:
            return {}
        
        total_channels = self.residual_mask.numel()
        l2_channels = self.residual_mask.sum().item()
        l2_ratio = l2_channels / total_channels
        
        # 计算平均比特数
        l1_bits = np.log2(self.l1_levels)
        l2_bits = np.log2(self.l2_levels)
        avg_bits = l1_bits + l2_bits * l2_ratio
        
        return {
            'l1_bits': l1_bits,
            'l2_bits': l2_bits,
            'l2_channel_ratio': l2_ratio,
            'avg_bits_per_element': avg_bits,
            'compression_ratio': 32 / avg_bits
        }


class LogitsKLDistillationLoss(nn.Module):
    """Logits KL蒸馏损失"""
    
    def __init__(self, temperature: float = 2.0, use_adaptive: bool = True):
        super().__init__()
        self.temperature = temperature
        self.use_adaptive = use_adaptive
    
    def compute_entropy(self, logits: torch.Tensor) -> float:
        """计算熵"""
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return entropy.mean().item()
    
    def forward(
        self,
        logits_original: torch.Tensor,
        logits_quantized: torch.Tensor,
        mse_weight: float = 0.5,
        kl_weight: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """计算损失"""
        # 自适应温度
        if self.use_adaptive:
            entropy = self.compute_entropy(logits_original)
            temperature = max(1.0, entropy / 3.0)
        else:
            temperature = self.temperature
        
        # MSE损失
        mse_loss = F.mse_loss(logits_quantized, logits_original)
        
        # KL散度
        logits_original_scaled = logits_original / temperature
        logits_quantized_scaled = logits_quantized / temperature
        
        probs_original = F.softmax(logits_original_scaled, dim=-1)
        log_probs_quantized = F.log_softmax(logits_quantized_scaled, dim=-1)
        
        kl_loss = F.kl_div(
            log_probs_quantized,
            probs_original,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        total_loss = mse_weight * mse_loss + kl_weight * kl_loss
        
        return total_loss, mse_loss, kl_loss, temperature


class DynamicLossScheduler:
    """动态Loss退火策略"""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
    
    def get_weights(self) -> Tuple[float, float]:
        """获取当前阶段权重"""
        progress = self.current_step / self.total_steps
        
        if progress < 0.1:
            return 1.0, 0.0
        elif progress < 0.6:
            return 0.3, 0.7
        else:
            return 0.1, 0.9
    
    def step(self):
        self.current_step += 1


class ExpertWithFSQ(nn.Module):
    """带FSQ量化的Expert"""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        fsq_levels: int = 16,
        group_size: int = 128,
        use_residual: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.use_residual = use_residual
        
        # Expert权重
        self.gate_proj = nn.Parameter(torch.randn(intermediate_size, hidden_size))
        self.up_proj = nn.Parameter(torch.randn(intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.randn(hidden_size, intermediate_size))
        
        # FSQ量化器
        if use_residual:
            self.fsq_gate = ResidualFSQQuantizer(l1_levels=fsq_levels, group_size=group_size)
            self.fsq_up = ResidualFSQQuantizer(l1_levels=fsq_levels, group_size=group_size)
            self.fsq_down = ResidualFSQQuantizer(l1_levels=fsq_levels, group_size=group_size)
        else:
            self.fsq_gate = FSQQuantizer(levels=fsq_levels, group_size=group_size)
            self.fsq_up = FSQQuantizer(levels=fsq_levels, group_size=group_size)
            self.fsq_down = FSQQuantizer(levels=fsq_levels, group_size=group_size)
    
    def forward(self, x: torch.Tensor, use_fsq: bool = True) -> torch.Tensor:
        """前向传播"""
        if use_fsq:
            gate_quantized = self.fsq_gate(self.gate_proj)
            up_quantized = self.fsq_up(self.up_proj)
            down_quantized = self.fsq_down(self.down_proj)
        else:
            gate_quantized = self.gate_proj
            up_quantized = self.up_proj
            down_quantized = self.down_proj
        
        gate = F.silu(F.linear(x, gate_quantized))
        up = F.linear(x, up_quantized)
        intermediate = gate * up
        output = F.linear(intermediate, down_quantized)
        
        return output


def load_model_weights() -> Dict[str, torch.Tensor]:
    """加载0.8B模型权重"""
    print("加载0.8B模型权重...")
    
    model_file = os.path.join(model_path, "model.safetensors-00001-of-00001.safetensors")
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"模型文件不存在: {model_file}")
    
    weights = {}
    with safe_open(model_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            weights[key] = tensor
    
    print(f"加载了 {len(weights)} 个张量")
    
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
    """创建合成数据"""
    data = torch.randn(batch_size, seq_len, hidden_size)
    
    for i in range(batch_size):
        for j in range(seq_len):
            pos_encoding = torch.sin(torch.arange(hidden_size) * 0.01 * j)
            data[i, j] += pos_encoding * 0.1
    
    return data


def train_single_experiment(
    weights: Dict[str, torch.Tensor],
    fsq_levels: int,
    use_residual: bool = False,
    total_steps: int = 1000,
    device: torch.device = torch.device("cpu")
) -> Dict[str, any]:
    """
    单次蒸馏量化实验
    
    Args:
        weights: 模型权重
        fsq_levels: FSQ级别 (16/8/4)
        use_residual: 是否使用残差量化
        total_steps: 训练步数
        device: 设备
    
    Returns:
        results: 实验结果
    """
    experiment_name = f"FSQ-{fsq_levels} ({int(np.log2(fsq_levels))}-bit)"
    if use_residual:
        experiment_name += " + Residual"
    
    print(f"\n{'='*70}")
    print(f"实验: {experiment_name}")
    print(f"{'='*70}")
    
    # 获取权重
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
        print("未找到权重，使用随机权重测试")
        hidden_size = MODEL_CONFIG["hidden_size"]
        intermediate_size = MODEL_CONFIG["intermediate_size"]
    else:
        hidden_size = weights[gate_key].shape[1]
        intermediate_size = weights[gate_key].shape[0]
    
    # 创建teacher和student
    teacher = ExpertWithFSQ(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        fsq_levels=fsq_levels,
        use_residual=use_residual
    ).to(device)
    
    student = ExpertWithFSQ(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        fsq_levels=fsq_levels,
        use_residual=use_residual
    ).to(device)
    
    # 加载权重
    if gate_key and up_key and down_key:
        teacher.gate_proj.data = weights[gate_key].to(device)
        teacher.up_proj.data = weights[up_key].to(device)
        teacher.down_proj.data = weights[down_key].to(device)
    
    # 复制权重到student
    student.gate_proj.data = teacher.gate_proj.data.clone()
    student.up_proj.data = teacher.up_proj.data.clone()
    student.down_proj.data = teacher.down_proj.data.clone()
    
    # 拟合FSQ
    student.fsq_gate.fit(student.gate_proj)
    student.fsq_up.fit(student.up_proj)
    student.fsq_down.fit(student.down_proj)
    
    # 创建损失函数和优化器
    loss_fn = LogitsKLDistillationLoss(use_adaptive=True)
    optimizer = torch.optim.AdamW(list(student.parameters()), lr=1e-4, weight_decay=0.01)
    loss_scheduler = DynamicLossScheduler(total_steps)
    
    # 训练循环
    losses = []
    temperatures = []
    start_time = time.time()
    
    for step in range(total_steps):
        x = create_synthetic_data(
            batch_size=2,
            seq_len=64,
            hidden_size=hidden_size
        ).to(device)
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            output_teacher = teacher(x, use_fsq=False)
        
        output_student = student(x, use_fsq=True)
        
        mse_weight, kl_weight = loss_scheduler.get_weights()
        
        total_loss, mse_loss, kl_loss, temperature = loss_fn(
            output_teacher,
            output_student,
            mse_weight=mse_weight,
            kl_weight=kl_weight
        )
        
        total_loss.backward()
        optimizer.step()
        
        # 重新拟合FSQ
        student.fsq_gate.fit(student.gate_proj)
        student.fsq_up.fit(student.up_proj)
        student.fsq_down.fit(student.down_proj)
        
        loss_scheduler.step()
        
        losses.append({
            'step': step,
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'kl_loss': kl_loss.item()
        })
        temperatures.append(temperature)
        
        if (step + 1) % 200 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step + 1}/{total_steps}: "
                  f"Loss={total_loss.item():.6f}, "
                  f"MSE={mse_loss.item():.6f}, "
                  f"KL={kl_loss.item():.6f}, "
                  f"T={temperature:.2f}, "
                  f"Time={elapsed:.1f}s")
    
    # 计算最终误差
    with torch.no_grad():
        gate_error = torch.norm(teacher.gate_proj - student.fsq_gate(teacher.gate_proj)) / torch.norm(teacher.gate_proj)
        up_error = torch.norm(teacher.up_proj - student.fsq_up(teacher.up_proj)) / torch.norm(teacher.up_proj)
        down_error = torch.norm(teacher.down_proj - student.fsq_down(teacher.down_proj)) / torch.norm(teacher.down_proj)
    
    # 计算压缩比
    if use_residual:
        compression_stats = student.fsq_gate.get_compression_stats()
        bits_per_element = compression_stats.get('avg_bits_per_element', np.log2(fsq_levels))
    else:
        bits_per_element = np.log2(fsq_levels)
    
    compression_ratio = 32 / bits_per_element
    
    # 计算码本利用率
    gate_util = student.fsq_gate.compute_utilization() if not use_residual else {}
    up_util = student.fsq_up.compute_utilization() if not use_residual else {}
    down_util = student.fsq_down.compute_utilization() if not use_residual else {}
    
    # 应用Huffman编码
    huffman_bpw = None
    if not use_residual and student.fsq_gate.get_indices() is not None:
        huffman_coder = HuffmanCoder()
        indices = student.fsq_gate.get_indices().reshape(-1).tolist()
        encoded, total_bits = huffman_coder.encode(indices)
        huffman_bpw = total_bits / len(indices)
    
    # 打印结果
    print(f"\n{'='*70}")
    print(f"实验结果: {experiment_name}")
    print(f"{'='*70}")
    print(f"量化误差:")
    print(f"  gate_proj: {gate_error.item()*100:.2f}%")
    print(f"  up_proj: {up_error.item()*100:.2f}%")
    print(f"  down_proj: {down_error.item()*100:.2f}%")
    print(f"\n压缩信息:")
    print(f"  FSQ级别: {fsq_levels}")
    print(f"  每元素比特数: {bits_per_element:.2f}")
    print(f"  压缩比: {compression_ratio:.2f}x")
    
    if huffman_bpw is not None:
        huffman_compression = 32 / huffman_bpw
        print(f"  Huffman编码后BPW: {huffman_bpw:.2f}")
        print(f"  Huffman压缩比: {huffman_compression:.2f}x")
    
    if use_residual:
        stats = student.fsq_gate.get_compression_stats()
        print(f"\n残差统计:")
        print(f"  L1级别: {stats.get('l1_levels', 'N/A')}")
        print(f"  L2级别: {stats.get('l2_levels', 'N/A')}")
        print(f"  L2通道比例: {stats.get('l2_channel_ratio', 0)*100:.2f}%")
    
    if gate_util:
        print(f"\n码本利用率:")
        print(f"  gate_proj: {gate_util.get('utilization', 0)*100:.2f}% ({gate_util.get('used_levels', 0)}/{gate_util.get('total_levels', 0)})")
        print(f"  up_proj: {up_util.get('utilization', 0)*100:.2f}% ({up_util.get('used_levels', 0)}/{up_util.get('total_levels', 0)})")
        print(f"  down_proj: {down_util.get('utilization', 0)*100:.2f}% ({down_util.get('used_levels', 0)}/{down_util.get('total_levels', 0)})")
    
    return {
        'name': experiment_name,
        'fsq_levels': fsq_levels,
        'use_residual': use_residual,
        'gate_error': gate_error.item(),
        'up_error': up_error.item(),
        'down_error': down_error.item(),
        'bits_per_element': bits_per_element,
        'compression_ratio': compression_ratio,
        'huffman_bpw': huffman_bpw,
        'huffman_compression': 32 / huffman_bpw if huffman_bpw else None,
        'losses': losses,
        'temperatures': temperatures,
        'utilization': {
            'gate': gate_util,
            'up': up_util,
            'down': down_util
        },
        'compression_stats': compression_stats if use_residual else None
    }


def run_multibit_experiments():
    """运行多比特FSQ量化实验"""
    print("=" * 70)
    print("Qwen3.5-0.8B 多比特FSQ量化实验")
    print("=" * 70)
    
    # 检查GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 加载权重
    weights = load_model_weights()
    
    # 实验配置
    experiments = [
        {'fsq_levels': 16, 'use_residual': False, 'name': 'FSQ-16 (4-bit)'},
        {'fsq_levels': 8, 'use_residual': False, 'name': 'FSQ-8 (3-bit)'},
        {'fsq_levels': 4, 'use_residual': False, 'name': 'FSQ-4 (2-bit)'},
        {'fsq_levels': 8, 'use_residual': True, 'name': 'Residual FSQ-8+4'},
    ]
    
    # 运行实验
    all_results = []
    total_start_time = time.time()
    
    for exp in experiments:
        print(f"\n{'#'*70}")
        print(f"开始实验: {exp['name']}")
        print(f"{'#'*70}")
        
        result = train_single_experiment(
            weights=weights,
            fsq_levels=exp['fsq_levels'],
            use_residual=exp['use_residual'],
            total_steps=1000,
            device=device
        )
        
        all_results.append(result)
    
    total_time = time.time() - total_start_time
    
    # 生成对比报告
    print("\n" + "=" * 70)
    print("实验对比报告")
    print("=" * 70)
    
    print(f"\n总耗时: {total_time:.1f}秒")
    print(f"\n{'实验方案':<25} {'量化误差%':<15} {'BPW':<10} {'Huffman BPW':<12} {'压缩比':<10}")
    print("-" * 75)
    
    for r in all_results:
        error_str = f"({r['gate_error']*100:.1f}, {r['up_error']*100:.1f}, {r['down_error']*100:.1f})"
        huffman_str = f"{r['huffman_bpw']:.2f}" if r['huffman_bpw'] else "N/A"
        compression_str = f"{r['compression_ratio']:.1f}x"
        if r['huffman_compression']:
            compression_str += f" ({r['huffman_compression']:.1f}x)"
        
        print(f"{r['name']:<25} {error_str:<15} {r['bits_per_element']:<10.2f} {huffman_str:<12} {compression_str:<10}")
    
    # 保存详细结果
    output_file = output_dir / "multibit_fsq_experiments.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Qwen3.5-0.8B 多比特FSQ量化实验结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("实验配置:\n")
        f.write(f"  设备: {device}\n")
        f.write(f"  总耗时: {total_time:.1f}秒\n\n")
        
        for r in all_results:
            f.write(f"\n{'='*50}\n")
            f.write(f"实验: {r['name']}\n")
            f.write(f"{'='*50}\n")
            
            f.write(f"\n量化误差:\n")
            f.write(f"  gate_proj: {r['gate_error']*100:.2f}%\n")
            f.write(f"  up_proj: {r['up_error']*100:.2f}%\n")
            f.write(f"  down_proj: {r['down_error']*100:.2f}%\n")
            
            f.write(f"\n压缩信息:\n")
            f.write(f"  FSQ级别: {r['fsq_levels']}\n")
            f.write(f"  每元素比特数: {r['bits_per_element']:.2f}\n")
            f.write(f"  压缩比: {r['compression_ratio']:.2f}x\n")
            
            if r['huffman_bpw']:
                f.write(f"  Huffman编码后BPW: {r['huffman_bpw']:.2f}\n")
                f.write(f"  Huffman压缩比: {r['huffman_compression']:.2f}x\n")
            
            if r['compression_stats']:
                f.write(f"\n残差统计:\n")
                for k, v in r['compression_stats'].items():
                    f.write(f"  {k}: {v}\n")
            
            f.write(f"\n温度统计:\n")
            temps = r['temperatures']
            f.write(f"  平均: {np.mean(temps):.2f}\n")
            f.write(f"  标准差: {np.std(temps):.2f}\n")
            f.write(f"  最小: {min(temps):.2f}\n")
            f.write(f"  最大: {max(temps):.2f}\n")
            
            if r['utilization'].get('gate'):
                f.write(f"\n码本利用率:\n")
                for proj in ['gate', 'up', 'down']:
                    util = r['utilization'][proj]
                    if util:
                        f.write(f"  {proj}: {util.get('utilization', 0)*100:.2f}%\n")
            
            f.write(f"\n损失曲线 (最后10步):\n")
            for loss in r['losses'][-10:]:
                f.write(f"  Step {loss['step']}: "
                        f"Total={loss['total_loss']:.6f}, "
                        f"MSE={loss['mse_loss']:.6f}, "
                        f"KL={loss['kl_loss']:.6f}\n")
    
    print(f"\n详细结果已保存到: {output_file}")
    
    # 保存JSON格式
    json_file = output_dir / "multibit_fsq_experiments.json"
    json_data = []
    for r in all_results:
        json_data.append({
            'name': r['name'],
            'fsq_levels': r['fsq_levels'],
            'use_residual': r['use_residual'],
            'gate_error': r['gate_error'],
            'up_error': r['up_error'],
            'down_error': r['down_error'],
            'bits_per_element': r['bits_per_element'],
            'compression_ratio': r['compression_ratio'],
            'huffman_bpw': r['huffman_bpw'],
            'huffman_compression': r['huffman_compression'],
            'avg_temperature': float(np.mean(r['temperatures'])),
            'compression_stats': r['compression_stats']
        })
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"JSON结果已保存到: {json_file}")
    
    return all_results


if __name__ == "__main__":
    results = run_multibit_experiments()
