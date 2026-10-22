"""分析Qwen模型不同层类型的秩特性"""

import os
import sys
import torch
import json
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

try:
    from safetensors.torch import load_file
except ImportError:
    print("需要安装safetensors")
    exit(1)


def analyze_rank_properties(tensor, name):
    """分析矩阵的秩特性"""
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    
    if tensor.dim() > 2:
        tensor = tensor.view(tensor.size(0), -1)
    
    # SVD分解
    U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
    
    # 计算能量分布
    total_energy = torch.sum(S ** 2)
    cumulative_energy = torch.cumsum(S ** 2, dim=0) / total_energy
    
    # 找到达到90%、95%、99%能量所需的秩
    rank_90 = torch.searchsorted(cumulative_energy, 0.9).item() + 1
    rank_95 = torch.searchsorted(cumulative_energy, 0.95).item() + 1
    rank_99 = torch.searchsorted(cumulative_energy, 0.99).item() + 1
    
    # 计算有效秩（能量达到50%所需的秩）
    rank_50 = torch.searchsorted(cumulative_energy, 0.5).item() + 1
    
    # 奇异值衰减率
    if len(S) > 1:
        decay_rate = (S[0] / S[-1]).item()
    else:
        decay_rate = 1.0
    
    return {
        'name': name,
        'shape': tensor.shape,
        'max_rank': min(tensor.shape),
        'rank_50': rank_50,
        'rank_90': rank_90,
        'rank_95': rank_95,
        'rank_99': rank_99,
        'decay_rate': decay_rate,
        'top_singular_values': S[:10].tolist()
    }


def main():
    model_path = r"D:\918822019.github.io\project\ContinuePretrain\Qwen3.5-0.8B"
    
    # 读取config
    config_path = Path(model_path) / "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("="*100)
    print("Qwen3.5-0.8B 模型配置")
    print("="*100)
    print(f"模型类型: {config.get('model_type', 'unknown')}")
    print(f"架构: {config.get('architectures', ['unknown'])[0]}")
    print(f"隐藏维度: {config.get('hidden_size', 'unknown')}")
    print(f"层数: {config.get('num_hidden_layers', 'unknown')}")
    print(f"注意力头数: {config.get('num_attention_heads', 'unknown')}")
    print(f"中间层大小: {config.get('intermediate_size', 'unknown')}")
    
    # 检查是否是MoE模型
    if 'num_experts' in config:
        print(f"专家数量: {config['num_experts']}")
        print(f"MoE层: {config.get('num_experts_per_tok', 'unknown')} 个/token")
        print("这是一个MoE模型!")
    else:
        print("这是一个Dense模型!")
    
    # 加载权重
    print("\n" + "="*100)
    print("加载权重...")
    print("="*100)
    
    safetensors_path = Path(model_path) / "model.safetensors-00001-of-00001.safetensors"
    state_dict = load_file(str(safetensors_path))
    
    # 分析不同类型的层
    layer_types = {
        'embedding': [],
        'attention_q': [],
        'attention_k': [],
        'attention_v': [],
        'attention_o': [],
        'mlp_gate': [],
        'mlp_up': [],
        'mlp_down': [],
        'layernorm': [],
        'other': []
    }
    
    for name in sorted(state_dict.keys()):
        if 'language_model' not in name:
            continue
        
        tensor = state_dict[name]
        if tensor.dim() < 2:  # 跳过1D张量（如bias、layernorm）
            continue
        
        # 分类
        if 'embed' in name:
            layer_types['embedding'].append((name, tensor))
        elif 'q_proj' in name:
            layer_types['attention_q'].append((name, tensor))
        elif 'k_proj' in name:
            layer_types['attention_k'].append((name, tensor))
        elif 'v_proj' in name:
            layer_types['attention_v'].append((name, tensor))
        elif 'o_proj' in name:
            layer_types['attention_o'].append((name, tensor))
        elif 'gate_proj' in name:
            layer_types['mlp_gate'].append((name, tensor))
        elif 'up_proj' in name:
            layer_types['mlp_up'].append((name, tensor))
        elif 'down_proj' in name:
            layer_types['mlp_down'].append((name, tensor))
        elif 'norm' in name:
            layer_types['layernorm'].append((name, tensor))
        else:
            layer_types['other'].append((name, tensor))
    
    # 分析每种类型的层
    print("\n" + "="*100)
    print("各层类型秩特性分析")
    print("="*100)
    
    for layer_type, layers in layer_types.items():
        if not layers:
            continue
        
        print(f"\n{layer_type} ({len(layers)}个层):")
        print("-" * 80)
        
        # 分析前3个层
        for i, (name, tensor) in enumerate(layers[:3]):
            result = analyze_rank_properties(tensor, name)
            
            print(f"\n  {name}")
            print(f"    形状: {result['shape'][0]}x{result['shape'][1]}")
            print(f"    最大秩: {result['max_rank']}")
            print(f"    达到50%能量需要的秩: {result['rank_50']} ({result['rank_50']/result['max_rank']*100:.1f}%)")
            print(f"    达到90%能量需要的秩: {result['rank_90']} ({result['rank_90']/result['max_rank']*100:.1f}%)")
            print(f"    达到95%能量需要的秩: {result['rank_95']} ({result['rank_95']/result['max_rank']*100:.1f}%)")
            print(f"    达到99%能量需要的秩: {result['rank_99']} ({result['rank_99']/result['max_rank']*100:.1f}%)")
            print(f"    奇异值衰减率: {result['decay_rate']:.2f}")
    
    # 汇总分析
    print("\n" + "="*100)
    print("汇总分析")
    print("="*100)
    
    print("\n各层类型平均秩需求（达到95%能量）:")
    for layer_type, layers in layer_types.items():
        if not layers:
            continue
        
        ranks = []
        for name, tensor in layers:
            if tensor.dim() >= 2:
                result = analyze_rank_properties(tensor, name)
                ranks.append(result['rank_95'])
        
        if ranks:
            avg_rank = np.mean(ranks)
            avg_ratio = np.mean([r / max(t.shape) for (_, t), r in zip(layers, ranks)])
            print(f"  {layer_type:<15}: 平均秩={avg_rank:.0f}, 平均比例={avg_ratio*100:.1f}%")
    
    print("\n" + "="*100)
    print("结论")
    print("="*100)


if __name__ == "__main__":
    main()
