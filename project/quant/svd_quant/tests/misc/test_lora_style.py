"""测试LoRA式低秩近似 vs 直接SVD分解"""

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
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
    
    weight_map = index_data['weight_map']
    layer_weights = {}
    for name in weight_map.keys():
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
    
    safetensors_path = os.path.join(model_path, "model.safetensors-00001-of-00001.safetensors")
    state_dict = load_file(safetensors_path)
    
    filtered = {}
    for layer_num in sorted(layer_weights.keys()):
        for name in layer_weights[layer_num]:
            if name in state_dict:
                filtered[name] = state_dict[name]
    
    return filtered


def test_lora_style_compression(tensor, rank, quantizer):
    """测试LoRA风格的压缩：W ≈ mean + low_rank_update"""
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    
    # 方法1：直接SVD分解
    U_merged, Vh, original = quantizer.svd_decompose(tensor, rank)
    reconstructed_svd = U_merged @ Vh
    error_svd = quantizer.calculate_relative_error(original, reconstructed_svd)
    
    # 方法2：LoRA风格 - 分解残差
    # 假设 W = mean + ΔW，其中ΔW是低秩的
    mean_weight = tensor.mean(dim=1, keepdim=True)  # 每行的均值
    residual = tensor - mean_weight  # 残差
    
    U_res, Vh_res, _ = quantizer.svd_decompose(residual, rank)
    reconstructed_lora = mean_weight + U_res @ Vh_res
    error_lora = quantizer.calculate_relative_error(original, reconstructed_lora)
    
    # 方法3：中心化后SVD
    centered = tensor - tensor.mean()
    U_c, Vh_c, _ = quantizer.svd_decompose(centered, rank)
    reconstructed_centered = tensor.mean() + U_c @ Vh_c
    error_centered = quantizer.calculate_relative_error(original, reconstructed_centered)
    
    return {
        'direct_svd': error_svd,
        'lora_style': error_lora,
        'centered_svd': error_centered
    }


def main():
    model_path = r"D:\918822019.github.io\project\ContinuePretrain\Qwen3.5-0.8B"
    
    print("加载Qwen模型权重...")
    state_dict = load_qwen_weights(model_path, max_layers=2)
    
    quantizer = SVDQuantizer()
    
    # 测试层
    test_layers = [
        'model.language_model.layers.0.mlp.gate_proj.weight',
        'model.language_model.layers.0.mlp.up_proj.weight',
        'model.language_model.layers.0.mlp.down_proj.weight',
        'model.language_model.layers.1.mlp.gate_proj.weight',
    ]
    
    print("\n" + "="*100)
    print("不同低秩分解方法对比")
    print("="*100)
    
    # 表头
    print(f"\n{'层名':<50} {'直接SVD':>12} {'LoRA风格':>12} {'中心化SVD':>12}")
    print("-" * 100)
    
    for weight_key in test_layers:
        tensor = state_dict[weight_key]
        
        # 测试不同秩
        for rank in [32, 64, 128, 256]:
            errors = test_lora_style_compression(tensor, rank, quantizer)
            
            short_name = weight_key.split('layers.')[-1]
            print(f"{short_name:<50} (秩={rank})")
            print(f"{'':50} {errors['direct_svd']*100:>10.2f}%  {errors['lora_style']*100:>10.2f}%  {errors['centered_svd']*100:>10.2f}%")
    
    print("\n" + "="*100)
    print("分析")
    print("="*100)


if __name__ == "__main__":
    main()
