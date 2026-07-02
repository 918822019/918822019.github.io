"""
Qwen3.5-0.8B 快速多比特FSQ量化测试

快速测试不同FSQ级别的压缩效果
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

_ROOT = Path(__file__).resolve().parents[3]
model_path = str(_ROOT / "data" / "models" / "Qwen3.5-0.8B")

# 输出目录
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)


class FSQQuantizer(nn.Module):
    """FSQ量化器"""
    
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


def load_model_weights():
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


def quick_fsq_test(weight_tensor: torch.Tensor, levels: int, num_steps: int = 100) -> Dict[str, float]:
    """快速FSQ量化测试"""
    print(f"\n测试 FSQ-{levels} ({int(np.log2(levels))}-bit)...")
    
    # 创建量化器
    fsq = FSQQuantizer(levels=levels, group_size=128)
    
    # 创建优化器
    weight_tensor = weight_tensor.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([weight_tensor], lr=1e-4)
    
    # 训练
    losses = []
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # 量化
        quantized = fsq(weight_tensor)
        
        # MSE损失
        loss = F.mse_loss(quantized, weight_tensor)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 重新拟合FSQ
        fsq.fit(weight_tensor)
        
        losses.append(loss.item())
        
        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{num_steps}: Loss={loss.item():.6f}")
    
    # 计算最终误差
    with torch.no_grad():
        quantized = fsq(weight_tensor)
        error = torch.norm(weight_tensor - quantized) / torch.norm(weight_tensor)
    
    # 计算压缩比
    bits_per_element = np.log2(levels)
    compression_ratio = 32 / bits_per_element
    
    # 计算码本利用率
    utilization = fsq.compute_utilization()
    
    return {
        'levels': levels,
        'bits_per_element': bits_per_element,
        'compression_ratio': compression_ratio,
        'error': error.item(),
        'utilization': utilization['utilization'],
        'used_levels': utilization['used_levels'],
        'total_levels': utilization['total_levels'],
        'final_loss': losses[-1],
        'losses': losses
    }


def run_quick_tests():
    """运行快速测试"""
    print("=" * 70)
    print("Qwen3.5-0.8B 快速多比特FSQ量化测试")
    print("=" * 70)
    
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
        print("未找到权重，使用随机权重测试")
        hidden_size = 1024
        intermediate_size = 3584
        test_weights = {
            'gate_proj': torch.randn(intermediate_size, hidden_size),
            'up_proj': torch.randn(intermediate_size, hidden_size),
            'down_proj': torch.randn(hidden_size, intermediate_size)
        }
    else:
        test_weights = {
            'gate_proj': weights[gate_key],
            'up_proj': weights[up_key],
            'down_proj': weights[down_key]
        }
    
    print(f"\n测试权重:")
    for name, w in test_weights.items():
        print(f"  {name}: {w.shape}")
    
    # 测试不同FSQ级别
    fsq_levels = [16, 8, 4]
    all_results = {}
    
    for levels in fsq_levels:
        print(f"\n{'#'*70}")
        print(f"测试 FSQ-{levels} ({int(np.log2(levels))}-bit)")
        print(f"{'#'*70}")
        
        results = {}
        for name, w in test_weights.items():
            # 复制权重以避免修改原始数据
            w_copy = w.clone().detach()
            result = quick_fsq_test(w_copy, levels, num_steps=100)
            results[name] = result
        
        all_results[levels] = results
        
        # 打印结果
        print(f"\nFSQ-{levels} 结果:")
        for name, r in results.items():
            print(f"  {name}:")
            print(f"    误差: {r['error']*100:.2f}%")
            print(f"    码本利用率: {r['utilization']*100:.2f}% ({r['used_levels']}/{r['total_levels']})")
            print(f"    压缩比: {r['compression_ratio']:.2f}x")
    
    # 计算Huffman编码后的平均比特数
    print(f"\n{'='*70}")
    print("Huffman编码分析")
    print(f"{'='*70}")
    
    huffman_results = {}
    for levels in fsq_levels:
        print(f"\nFSQ-{levels} Huffman分析:")
        
        # 使用gate_proj的量化索引进行Huffman编码
        gate_result = all_results[levels]['gate_proj']
        
        # 重新创建量化器并获取索引
        fsq = FSQQuantizer(levels=levels, group_size=128)
        fsq.fit(test_weights['gate_proj'])
        quantized = fsq(test_weights['gate_proj'])
        indices = fsq._last_indices.reshape(-1).tolist()
        
        # 统计频率
        from collections import Counter
        freq_map = Counter(indices)
        
        # 计算熵
        total = len(indices)
        entropy = 0
        for count in freq_map.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        huffman_bpw = entropy  # 理论上Huffman编码接近熵
        huffman_compression = 32 / huffman_bpw
        
        huffman_results[levels] = {
            'entropy': entropy,
            'huffman_bpw': huffman_bpw,
            'huffman_compression': huffman_compression,
            'level_distribution': dict(freq_map)
        }
        
        print(f"  熵: {entropy:.2f} bits")
        print(f"  Huffman BPW: {huffman_bpw:.2f}")
        print(f"  Huffman压缩比: {huffman_compression:.2f}x")
        
        # 打印级别分布
        print(f"  级别分布:")
        for level in range(levels):
            count = freq_map.get(level, 0)
            percentage = count / total * 100
            print(f"    级别 {level}: {count} ({percentage:.2f}%)")
    
    # 保存结果
    output_file = output_dir / "quick_fsq_test_results.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Qwen3.5-0.8B 快速多比特FSQ量化测试结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("测试配置:\n")
        f.write("  每级别训练步数: 100\n")
        f.write("  分组大小: 128\n\n")
        
        for levels in fsq_levels:
            f.write(f"\n{'='*50}\n")
            f.write(f"FSQ-{levels} ({int(np.log2(levels))}-bit)\n")
            f.write(f"{'='*50}\n")
            
            for name, r in all_results[levels].items():
                f.write(f"\n{name}:\n")
                f.write(f"  误差: {r['error']*100:.2f}%\n")
                f.write(f"  每元素比特数: {r['bits_per_element']:.2f}\n")
                f.write(f"  压缩比: {r['compression_ratio']:.2f}x\n")
                f.write(f"  码本利用率: {r['utilization']*100:.2f}%\n")
                f.write(f"  使用级别数: {r['used_levels']}/{r['total_levels']}\n")
                f.write(f"  最终损失: {r['final_loss']:.6f}\n")
            
            huffman = huffman_results[levels]
            f.write(f"\nHuffman编码分析:\n")
            f.write(f"  熵: {huffman['entropy']:.2f} bits\n")
            f.write(f"  Huffman BPW: {huffman['huffman_bpw']:.2f}\n")
            f.write(f"  Huffman压缩比: {huffman['huffman_compression']:.2f}x\n")
            
            f.write(f"\n级别分布:\n")
            total_indices = sum(huffman['level_distribution'].values())
            for level in range(levels):
                count = huffman['level_distribution'].get(level, 0)
                percentage = count / total_indices * 100 if total_indices > 0 else 0
                f.write(f"  级别 {level}: {count} ({percentage:.2f}%)\n")
    
    print(f"\n结果已保存到: {output_file}")
    
    # 保存JSON格式
    json_file = output_dir / "quick_fsq_test_results.json"
    json_data = {
        'fsq_levels': {},
        'huffman_results': {}
    }
    
    for levels in fsq_levels:
        json_data['fsq_levels'][str(levels)] = {}
        for name, r in all_results[levels].items():
            json_data['fsq_levels'][str(levels)][name] = {
                'error': r['error'],
                'bits_per_element': r['bits_per_element'],
                'compression_ratio': r['compression_ratio'],
                'utilization': r['utilization'],
                'used_levels': r['used_levels'],
                'total_levels': r['total_levels'],
                'final_loss': r['final_loss']
            }
        
        huffman = huffman_results[levels]
        json_data['huffman_results'][str(levels)] = {
            'entropy': huffman['entropy'],
            'huffman_bpw': huffman['huffman_bpw'],
            'huffman_compression': huffman['huffman_compression']
        }
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"JSON结果已保存到: {json_file}")
    
    # 生成总结报告
    print(f"\n{'='*70}")
    print("总结报告")
    print(f"{'='*70}")
    
    print(f"\n{'方案':<15} {'比特数':<10} {'压缩比':<10} {'误差%':<10} {'Huffman BPW':<12} {'Huffman压缩比':<12}")
    print("-" * 70)
    
    for levels in fsq_levels:
        bits = int(np.log2(levels))
        avg_error = np.mean([all_results[levels][name]['error'] for name in test_weights.keys()])
        compression = all_results[levels]['gate_proj']['compression_ratio']
        huffman_bpw = huffman_results[levels]['huffman_bpw']
        huffman_compression = huffman_results[levels]['huffman_compression']
        
        print(f"FSQ-{levels:<5} {bits:<10} {compression:<10.2f} {avg_error*100:<10.2f} {huffman_bpw:<12.2f} {huffman_compression:<12.2f}")
    
    return all_results, huffman_results


if __name__ == "__main__":
    results, huffman_results = run_quick_tests()
