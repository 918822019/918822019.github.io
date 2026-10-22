"""测试秩256的SVD分解精度"""

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


def main():
    model_path = r"D:\918822019.github.io\project\ContinuePretrain\Qwen3.5-0.8B"
    
    print("加载Qwen模型权重...")
    state_dict = load_qwen_weights(model_path, max_layers=2)
    
    quantizer = SVDQuantizer()
    
    # 测试MLP层
    test_layers = [
        ('model.language_model.layers.0.mlp.gate_proj.weight', 'gate_proj'),
        ('model.language_model.layers.0.mlp.up_proj.weight', 'up_proj'),
        ('model.language_model.layers.0.mlp.down_proj.weight', 'down_proj'),
        ('model.language_model.layers.1.mlp.gate_proj.weight', 'gate_proj'),
    ]
    
    print("\n" + "="*80)
    print("不同秩的SVD分解精度对比")
    print("="*80)
    
    for weight_key, proj_type in test_layers:
        tensor = state_dict[weight_key]
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        
        print(f"\n层: {weight_key}")
        print(f"形状: {tensor.shape[0]}x{tensor.shape[1]}")
        print("-" * 60)
        
        # 测试不同秩
        for rank in [16, 32, 64, 128, 256, 384, 512]:
            if rank >= min(tensor.shape):
                continue
            
            U_merged, Vh, original = quantizer.svd_decompose(tensor, rank)
            reconstructed = U_merged @ Vh
            error = quantizer.calculate_relative_error(original, reconstructed)
            
            # 计算压缩比
            original_size = tensor.shape[0] * tensor.shape[1]
            decomposed_size = tensor.shape[0] * rank + rank * tensor.shape[1]
            compression = original_size / decomposed_size
            
            print(f"  秩={rank:3d}: 误差={error*100:6.2f}%, 压缩比={compression:.2f}x")
    
    print("\n" + "="*80)
    print("结论")
    print("="*80)


if __name__ == "__main__":
    main()
