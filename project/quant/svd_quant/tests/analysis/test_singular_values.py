"""
Qwen模型全层SVD分解特性分析

分析所有层的奇异值分布，了解哪些层更适合SVD分解
"""

import os
import sys
import torch
import json
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

try:
    from safetensors.torch import load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


def load_all_qwen_weights(model_path: str) -> dict:
    """加载Qwen模型所有权重"""
    print(f"加载模型: {model_path}")
    
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
    
    weight_map = index_data['weight_map']
    
    # 获取所有语言模型权重
    language_weights = []
    for name in weight_map.keys():
        if 'model.language_model' in name and '.weight' in name:
            language_weights.append(name)
    
    print(f"找到 {len(language_weights)} 个权重张量")
    
    safetensors_path = os.path.join(model_path, "model.safetensors-00001-of-00001.safetensors")
    
    if HAS_SAFETENSORS:
        state_dict = load_file(safetensors_path)
    else:
        state_dict = torch.load(safetensors_path, map_location='cpu')
    
    filtered = {name: state_dict[name] for name in language_weights if name in state_dict}
    print(f"加载了 {len(filtered)} 个张量")
    return filtered


def analyze_singular_values(tensor: torch.Tensor, name: str, top_k: int = 20):
    """分析奇异值分布"""
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    
    # 如果是3D，展平
    if tensor.dim() > 2:
        tensor = tensor.view(tensor.size(0), -1)
    
    # 计算SVD
    U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
    
    # 计算各比例的能量
    total_energy = torch.sum(S ** 2)
    cumulative_energy = torch.cumsum(S ** 2, dim=0) / total_energy
    
    # 找到达到不同能量阈值所需的秩
    thresholds = [0.9, 0.95, 0.99]
    ranks_for_threshold = {}
    for threshold in thresholds:
        rank = torch.searchsorted(cumulative_energy, threshold).item() + 1
        ranks_for_threshold[threshold] = rank
    
    return {
        'name': name,
        'shape': tensor.shape,
        'singular_values': S[:top_k].numpy(),
        'total_energy': total_energy.item(),
        'cumulative_energy': cumulative_energy.numpy(),
        'ranks_for_threshold': ranks_for_threshold,
        'max_rank': min(tensor.shape)
    }


def main():
    """主函数"""
    model_path = r"D:\918822019.github.io\project\ContinuePretrain\Qwen3.5-0.8B"
    
    print("="*100)
    print("Qwen模型全层SVD分解特性分析")
    print("="*100)
    
    # 加载所有权重
    state_dict = load_all_qwen_weights(model_path)
    
    # 只分析MLP层
    mlp_layers = []
    for name in sorted(state_dict.keys()):
        if 'mlp.' in name and ('gate_proj' in name or 'up_proj' in name or 'down_proj' in name):
            mlp_layers.append(name)
    
    print(f"\n找到 {len(mlp_layers)} 个MLP层")
    
    # 分析每层
    results = []
    
    print("\n" + "="*100)
    print("逐层分析（显示达到不同能量保留比例所需的秩）")
    print("="*100)
    
    for i, name in enumerate(mlp_layers):
        tensor = state_dict[name]
        result = analyze_singular_values(tensor, name)
        results.append(result)
        
        # 打印进度
        if i % 10 == 0:
            print(f"\n进度: {i}/{len(mlp_layers)}")
        
        # 打印关键信息
        if i < 6:  # 前6层详细打印
            print(f"\n层: {name} (形状: {result['shape'][0]}x{result['shape'][1]})")
            print(f"  达到90%能量需要的秩: {result['ranks_for_threshold'][0.9]}")
            print(f"  达到95%能量需要的秩: {result['ranks_for_threshold'][0.95]}")
            print(f"  达到99%能量需要的秩: {result['ranks_for_threshold'][0.99]}")
            print(f"  前20个奇异值: {result['singular_values'][:5].round(2)}...")
    
    # 统计分析
    print("\n" + "="*100)
    print("统计分析")
    print("="*100)
    
    # 按层类型分组
    gate_proj_results = [r for r in results if 'gate_proj' in r['name']]
    up_proj_results = [r for r in results if 'up_proj' in r['name']]
    down_proj_results = [r for r in results if 'down_proj' in r['name']]
    
    def print_stats(results, name):
        if not results:
            return
        
        print(f"\n{name} 层 ({len(results)}个):")
        
        for threshold in [0.9, 0.95, 0.99]:
            ranks = [r['ranks_for_threshold'][threshold] for r in results]
            print(f"  达到{threshold*100}%能量需要的秩:")
            print(f"    最小: {min(ranks)}")
            print(f"    最大: {max(ranks)}")
            print(f"    平均: {np.mean(ranks):.1f}")
            print(f"    中位数: {np.median(ranks):.1f}")
    
    print_stats(gate_proj_results, "gate_proj")
    print_stats(up_proj_results, "up_proj")
    print_stats(down_proj_results, "down_proj")
    
    # 分析当前配置的合理性
    print("\n" + "="*100)
    print("当前配置分析")
    print("="*100)
    
    from main import DecompositionConfig
    config = DecompositionConfig()
    
    print(f"\n默认配置: gate_proj_rank={config.gate_proj_rank}, up_proj_rank={config.up_proj_rank}, down_proj_rank={config.down_proj_rank}")
    
    for results, name, rank in [
        (gate_proj_results, "gate_proj", config.gate_proj_rank),
        (up_proj_results, "up_proj", config.up_proj_rank),
        (down_proj_results, "down_proj", config.down_proj_rank)
    ]:
        if not results:
            continue
        
        print(f"\n{name} (秩={rank}):")
        
        # 计算每层在该秩下能保留多少能量
        energy_at_rank = []
        for r in results:
            cum_energy = r['cumulative_energy']
            if rank < len(cum_energy):
                energy = cum_energy[rank - 1]
            else:
                energy = 1.0
            energy_at_rank.append(energy)
        
        print(f"  能量保留比例:")
        print(f"    最小: {min(energy_at_rank)*100:.2f}%")
        print(f"    最大: {max(energy_at_rank)*100:.2f}%")
        print(f"    平均: {np.mean(energy_at_rank)*100:.2f}%")
        
        # 建议
        avg_rank_for_95 = np.mean([r['ranks_for_threshold'][0.95] for r in results])
        avg_rank_for_99 = np.mean([r['ranks_for_threshold'][0.99] for r in results])
        print(f"  建议:")
        print(f"    达到95%能量平均需要秩: {avg_rank_for_95:.0f}")
        print(f"    达到99%能量平均需要秩: {avg_rank_for_99:.0f}")
    
    print("\n" + "="*100)
    print("分析完成！")
    print("="*100)


if __name__ == "__main__":
    main()
