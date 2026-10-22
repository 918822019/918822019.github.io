"""
真实Qwen模型SVD分解精度测试

使用Qwen3.5-0.8B模型的真实权重测试SVD分解精度
"""

import os
import sys
import torch
import json
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.main import SVDQuantizer, DecompositionConfig

try:
    from safetensors.torch import load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


def load_qwen_weights(model_path: str, max_layers: int = 2) -> dict:
    """加载Qwen模型权重"""
    print(f"加载模型: {model_path}")
    
    # 读取索引
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
    
    weight_map = index_data['weight_map']
    
    # 按层分组
    layer_weights = {}
    for name, file_name in weight_map.items():
        if 'model.language_model' in name:
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    layer_num = int(parts[i + 1])
                    if max_layers is None or layer_num < max_layers:
                        if layer_num not in layer_weights:
                            layer_weights[layer_num] = []
                        layer_weights[layer_num].append(name)
                    break
    
    # 加载权重
    safetensors_path = os.path.join(model_path, "model.safetensors-00001-of-00001.safetensors")
    
    if HAS_SAFETENSORS:
        state_dict = load_file(safetensors_path)
    else:
        state_dict = torch.load(safetensors_path, map_location='cpu')
    
    # 过滤
    filtered = {}
    for layer_num in sorted(layer_weights.keys()):
        for name in layer_weights[layer_num]:
            if name in state_dict:
                filtered[name] = state_dict[name]
    
    print(f"加载了 {len(filtered)} 个张量")
    return filtered


def analyze_svd_per_layer(state_dict: dict):
    """分析每层的SVD分解精度"""
    print("\n" + "="*100)
    print("逐层SVD分解精度分析")
    print("="*100)
    
    quantizer = SVDQuantizer()
    
    # 按层分组
    layers = {}
    for name, tensor in state_dict.items():
        # 提取层名（去掉.weight后缀）
        base_name = name.replace('.weight', '')
        parts = base_name.split('.')
        
        # 找到layers.X部分
        for i, part in enumerate(parts):
            if part == 'layers' and i + 1 < len(parts):
                layer_id = '.'.join(parts[:i+2])  # model.language_model.layers.X
                proj_type = parts[-1]  # gate_proj, up_proj, down_proj, q_proj等
                
                if layer_id not in layers:
                    layers[layer_id] = {}
                layers[layer_id][proj_type] = tensor
                break
    
    # 分析每层
    all_results = []
    
    for layer_id, projs in sorted(layers.items()):
        print(f"\n{'─'*100}")
        print(f"层: {layer_id}")
        print(f"{'─'*100}")
        
        for proj_name, tensor in sorted(projs.items()):
            if 'proj' not in proj_name:
                continue
                
            # 确定是否应该分解
            should_decompose = quantizer.is_moe_expert_layer(f"{layer_id}.{proj_name}.weight")
            rank = quantizer.get_rank_for_layer(f"{layer_id}.{proj_name}.weight") if should_decompose else None
            
            # 转换为float32（如果需要）
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            
            # 测试不同秩
            ranks_to_test = [8, 16, 32, 64, 128]
            min_dim = min(tensor.shape)
            ranks_to_test = [r for r in ranks_to_test if r < min_dim]
            
            errors = {}
            for rank in ranks_to_test:
                U_merged, Vh, original = quantizer.svd_decompose(tensor, rank)
                reconstructed = U_merged @ Vh
                error = quantizer.calculate_relative_error(original, reconstructed)
                errors[rank] = error
            
            # 记录结果
            result = {
                'layer': layer_id,
                'proj': proj_name,
                'shape': tensor.shape,
                'should_decompose': should_decompose,
                'rank': rank,
                'errors': errors
            }
            all_results.append(result)
            
            # 打印结果
            status = "✓ 分解" if should_decompose else "○ 跳过"
            print(f"\n  {proj_name} {status} (形状: {tensor.shape[0]}x{tensor.shape[1]})")
            
            if should_decompose:
                print(f"    使用秩: {rank}")
                print(f"    各秩误差:")
                for r, e in errors.items():
                    marker = " ← 当前" if r == rank else ""
                    print(f"      秩={r:3d}: 相对误差={e:.6f} ({e*100:.2f}%){marker}")
            else:
                print(f"    直接INT4量化（不分解）")
    
    return all_results


def print_summary(results: list):
    """打印汇总信息"""
    print("\n" + "="*100)
    print("汇总分析")
    print("="*100)
    
    decompose_results = [r for r in results if r['should_decompose']]
    skip_results = [r for r in results if not r['should_decompose']]
    
    print(f"\n分解层统计:")
    print(f"  需要分解的层: {len(decompose_results)}")
    print(f"  跳过的层: {len(skip_results)}")
    
    if decompose_results:
        print(f"\n各层SVD分解误差（使用配置的秩）:")
        
        # 按误差排序
        sorted_results = sorted(decompose_results, key=lambda x: x['errors'].get(x['rank'], 1.0))
        
        for r in sorted_results:
            error = r['errors'].get(r['rank'], 1.0)
            print(f"  {r['layer']}.{r['proj']}: 秩={r['rank']}, 误差={error:.6f} ({error*100:.2f}%)")
        
        # 统计
        errors = [r['errors'].get(r['rank'], 1.0) for r in decompose_results]
        print(f"\n误差统计:")
        print(f"  最小误差: {min(errors):.6f} ({min(errors)*100:.2f}%)")
        print(f"  最大误差: {max(errors):.6f} ({max(errors)*100:.2f}%)")
        print(f"  平均误差: {np.mean(errors):.6f} ({np.mean(errors)*100:.2f}%)")
        print(f"  中位数误差: {np.median(errors):.6f} ({np.median(errors)*100:.2f}%)")
        
        # 分析哪些层误差大
        threshold = 0.5  # 50%
        high_error = [r for r in decompose_results if r['errors'].get(r['rank'], 1.0) > threshold]
        
        if high_error:
            print(f"\n误差超过50%的层:")
            for r in high_error:
                error = r['errors'].get(r['rank'], 1.0)
                print(f"  {r['layer']}.{r['proj']}: {error*100:.2f}%")


def test_different_ranks(state_dict: dict):
    """测试不同秩配置的效果"""
    print("\n" + "="*100)
    print("不同秩配置对比测试")
    print("="*100)
    
    quantizer = SVDQuantizer()
    
    # 选择一个典型层进行测试
    test_layers = [
        'model.language_model.layers.0.mlp.gate_proj',
        'model.language_model.layers.0.mlp.down_proj',
        'model.language_model.layers.1.mlp.gate_proj',
        'model.language_model.layers.1.mlp.down_proj',
    ]
    
    configs = [
        ("低秩", DecompositionConfig(gate_proj_rank=16, up_proj_rank=16, down_proj_rank=32)),
        ("默认", DecompositionConfig(gate_proj_rank=32, up_proj_rank=32, down_proj_rank=64)),
        ("高秩", DecompositionConfig(gate_proj_rank=64, up_proj_rank=64, down_proj_rank=128)),
    ]
    
    for layer_name in test_layers:
        weight_key = f"{layer_name}.weight"
        if weight_key not in state_dict:
            continue
            
        tensor = state_dict[weight_key]
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        
        print(f"\n层: {layer_name} (形状: {tensor.shape[0]}x{tensor.shape[1]})")
        print("-" * 60)
        
        for config_name, config in configs:
            q = SVDQuantizer(config)
            rank = q.get_rank_for_layer(weight_key)
            
            U_merged, Vh, original = q.svd_decompose(tensor, rank)
            reconstructed = U_merged @ Vh
            error = q.calculate_relative_error(original, reconstructed)
            
            print(f"  {config_name}: 秩={rank}, 误差={error:.6f} ({error*100:.2f}%)")


def main():
    """主函数"""
    model_path = r"D:\918822019.github.io\project\ContinuePretrain\Qwen3.5-0.8B"
    
    # 加载模型（前2层）
    print("加载Qwen3.5-0.8B模型权重（前2层）...")
    state_dict = load_qwen_weights(model_path, max_layers=2)
    
    # 分析每层
    results = analyze_svd_per_layer(state_dict)
    
    # 打印汇总
    print_summary(results)
    
    # 测试不同秩配置
    test_different_ranks(state_dict)
    
    print("\n" + "="*100)
    print("测试完成！")
    print("="*100)


if __name__ == "__main__":
    main()
