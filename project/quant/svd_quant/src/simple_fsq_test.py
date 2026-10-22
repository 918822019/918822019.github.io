"""
简单FSQ量化测试

直接计算不同FSQ级别的量化误差，不需要训练
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
import json
from pathlib import Path
from collections import Counter

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_ROOT = Path(__file__).resolve().parents[3]
model_path = str(_ROOT / "data" / "models" / "Qwen3.5-0.8B")

# 输出目录
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)


def fsq_quantize(x: torch.Tensor, levels: int, group_size: int = 128):
    """FSQ量化"""
    original_shape = x.shape
    x_flat = x.reshape(-1)
    
    # 分组
    if x_flat.numel() > group_size:
        x_groups = x_flat.reshape(-1, group_size)
    else:
        x_groups = x_flat.reshape(1, -1)
    
    # 计算每组的min/max
    min_vals = x_groups.min(dim=1, keepdim=True)[0]
    max_vals = x_groups.max(dim=1, keepdim=True)[0]
    
    # 避免除零
    range_vals = max_vals - min_vals
    range_vals = torch.clamp(range_vals, min=1e-8)
    max_vals = min_vals + range_vals
    
    # 归一化到 [0, 1]
    x_normalized = (x_groups - min_vals) / (max_vals - min_vals)
    
    # 量化到 [0, levels-1]
    x_scaled = x_normalized * (levels - 1)
    x_rounded = torch.round(x_scaled)
    
    # 获取量化索引
    indices = x_rounded.detach().long()
    
    # 反归一化
    x_normalized_back = x_rounded / (levels - 1)
    x_quantized_groups = x_normalized_back * (max_vals - min_vals) + min_vals
    
    # 恢复形状
    x_quantized = x_quantized_groups.reshape(-1)[:x_flat.numel()].reshape(original_shape)
    
    return x_quantized, indices


def compute_error(original: torch.Tensor, quantized: torch.Tensor):
    """计算量化误差"""
    error = torch.norm(original - quantized) / torch.norm(original)
    return error.item()


def compute_entropy(indices: torch.Tensor, levels: int):
    """计算熵"""
    flat_indices = indices.reshape(-1).tolist()
    total = len(flat_indices)
    
    # 统计频率
    freq_map = Counter(flat_indices)
    
    # 计算熵
    entropy = 0
    for count in freq_map.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy, freq_map


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


def run_simple_test():
    """运行简单测试"""
    print("=" * 70)
    print("Qwen3.5-0.8B 简单FSQ量化测试")
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
    group_size = 128
    
    all_results = {}
    
    for levels in fsq_levels:
        print(f"\n{'#'*70}")
        print(f"测试 FSQ-{levels} ({int(np.log2(levels))}-bit)")
        print(f"{'#'*70}")
        
        results = {}
        for name, w in test_weights.items():
            # 量化
            quantized, indices = fsq_quantize(w, levels, group_size)
            
            # 计算误差
            error = compute_error(w, quantized)
            
            # 计算熵和分布
            entropy, freq_map = compute_entropy(indices, levels)
            
            # 计算压缩比
            bits_per_element = np.log2(levels)
            compression_ratio = 32 / bits_per_element
            
            # 计算Huffman压缩后的BPW
            huffman_bpw = entropy
            huffman_compression = 32 / huffman_bpw
            
            # 计算码本利用率
            flat_indices = indices.reshape(-1).tolist()
            used_levels = len(set(flat_indices))
            utilization = used_levels / levels
            
            results[name] = {
                'error': error,
                'bits_per_element': bits_per_element,
                'compression_ratio': compression_ratio,
                'entropy': entropy,
                'huffman_bpw': huffman_bpw,
                'huffman_compression': huffman_compression,
                'utilization': utilization,
                'used_levels': used_levels,
                'total_levels': levels,
                'level_distribution': dict(freq_map)
            }
        
        all_results[levels] = results
        
        # 打印结果
        print(f"\nFSQ-{levels} 结果:")
        for name, r in results.items():
            print(f"  {name}:")
            print(f"    误差: {r['error']*100:.2f}%")
            print(f"    熵: {r['entropy']:.2f} bits")
            print(f"    Huffman BPW: {r['huffman_bpw']:.2f}")
            print(f"    Huffman压缩比: {r['huffman_compression']:.2f}x")
            print(f"    码本利用率: {r['utilization']*100:.2f}% ({r['used_levels']}/{r['total_levels']})")
    
    # 保存结果
    output_file = output_dir / "simple_fsq_test_results.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Qwen3.5-0.8B 简单FSQ量化测试结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("测试配置:\n")
        f.write(f"  分组大小: {group_size}\n\n")
        
        for levels in fsq_levels:
            f.write(f"\n{'='*50}\n")
            f.write(f"FSQ-{levels} ({int(np.log2(levels))}-bit)\n")
            f.write(f"{'='*50}\n")
            
            for name, r in all_results[levels].items():
                f.write(f"\n{name}:\n")
                f.write(f"  误差: {r['error']*100:.2f}%\n")
                f.write(f"  每元素比特数: {r['bits_per_element']:.2f}\n")
                f.write(f"  压缩比: {r['compression_ratio']:.2f}x\n")
                f.write(f"  熵: {r['entropy']:.2f} bits\n")
                f.write(f"  Huffman BPW: {r['huffman_bpw']:.2f}\n")
                f.write(f"  Huffman压缩比: {r['huffman_compression']:.2f}x\n")
                f.write(f"  码本利用率: {r['utilization']*100:.2f}%\n")
                f.write(f"  使用级别数: {r['used_levels']}/{r['total_levels']}\n")
                
                f.write(f"\n  级别分布:\n")
                total_indices = sum(r['level_distribution'].values())
                for level in range(levels):
                    count = r['level_distribution'].get(level, 0)
                    percentage = count / total_indices * 100 if total_indices > 0 else 0
                    f.write(f"    级别 {level}: {count} ({percentage:.2f}%)\n")
    
    print(f"\n结果已保存到: {output_file}")
    
    # 保存JSON格式
    json_file = output_dir / "simple_fsq_test_results.json"
    json_data = {}
    
    for levels in fsq_levels:
        json_data[str(levels)] = {}
        for name, r in all_results[levels].items():
            json_data[str(levels)][name] = {
                'error': r['error'],
                'bits_per_element': r['bits_per_element'],
                'compression_ratio': r['compression_ratio'],
                'entropy': r['entropy'],
                'huffman_bpw': r['huffman_bpw'],
                'huffman_compression': r['huffman_compression'],
                'utilization': r['utilization'],
                'used_levels': r['used_levels'],
                'total_levels': r['total_levels']
            }
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"JSON结果已保存到: {json_file}")
    
    # 生成总结报告
    print(f"\n{'='*70}")
    print("总结报告")
    print(f"{'='*70}")
    
    print(f"\n{'方案':<15} {'比特数':<10} {'压缩比':<10} {'误差%':<10} {'熵':<10} {'Huffman BPW':<12} {'Huffman压缩比':<12}")
    print("-" * 80)
    
    for levels in fsq_levels:
        bits = int(np.log2(levels))
        avg_error = np.mean([all_results[levels][name]['error'] for name in test_weights.keys()])
        compression = all_results[levels]['gate_proj']['compression_ratio']
        entropy = np.mean([all_results[levels][name]['entropy'] for name in test_weights.keys()])
        huffman_bpw = np.mean([all_results[levels][name]['huffman_bpw'] for name in test_weights.keys()])
        huffman_compression = np.mean([all_results[levels][name]['huffman_compression'] for name in test_weights.keys()])
        
        print(f"FSQ-{levels:<5} {bits:<10} {compression:<10.2f} {avg_error*100:<10.2f} {entropy:<10.2f} {huffman_bpw:<12.2f} {huffman_compression:<12.2f}")
    
    return all_results


if __name__ == "__main__":
    results = run_simple_test()
