"""测试不同秩的端到端输出误差（SVD分解 + 量化 + 反量化）"""

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


def test_end_to_end_error(quantizer, tensor, rank, group_size=128):
    """测试完整的端到端误差"""
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    
    # 1. SVD分解
    U_merged, Vh, original = quantizer.svd_decompose(tensor, rank)
    
    # 2. INT8量化U_merged
    U_quantized = quantizer.symmetric_quantize_int8(U_merged, group_size)
    
    # 3. INT4量化Vh
    Vh_quantized = quantizer.symmetric_quantize_int4(Vh, group_size)
    
    # 4. 反量化
    U_dequant = quantizer.dequantize(U_quantized)
    Vh_dequant = quantizer.dequantize(Vh_quantized)
    
    # 5. 重构
    reconstructed = U_dequant @ Vh_dequant
    
    # 6. 计算误差
    original_np = original.cpu().numpy()
    error = np.linalg.norm(original_np - reconstructed) / np.linalg.norm(original_np)
    
    return error


def main():
    model_path = r"D:\918822019.github.io\project\ContinuePretrain\Qwen3.5-0.8B"
    
    print("加载Qwen模型权重...")
    state_dict = load_qwen_weights(model_path, max_layers=2)
    
    # 测试不同配置
    configs = [
        ("低秩配置", DecompositionConfig(gate_proj_rank=16, up_proj_rank=16, down_proj_rank=32, quant_group_size=128)),
        ("默认配置", DecompositionConfig(gate_proj_rank=32, up_proj_rank=32, down_proj_rank=64, quant_group_size=128)),
        ("中等秩", DecompositionConfig(gate_proj_rank=64, up_proj_rank=64, down_proj_rank=128, quant_group_size=128)),
        ("高秩配置", DecompositionConfig(gate_proj_rank=128, up_proj_rank=128, down_proj_rank=256, quant_group_size=128)),
        ("超高秩", DecompositionConfig(gate_proj_rank=256, up_proj_rank=256, down_proj_rank=384, quant_group_size=128)),
    ]
    
    # 测试层
    test_layers = [
        'model.language_model.layers.0.mlp.gate_proj.weight',
        'model.language_model.layers.0.mlp.up_proj.weight',
        'model.language_model.layers.0.mlp.down_proj.weight',
        'model.language_model.layers.1.mlp.gate_proj.weight',
    ]
    
    print("\n" + "="*100)
    print("端到端误差测试（SVD分解 + INT8/INT4量化 + 反量化）")
    print("="*100)
    
    # 表头
    print(f"\n{'层名':<55}", end="")
    for config_name, _ in configs:
        print(f"{config_name:>12}", end="")
    print()
    print("-" * 100)
    
    # 测试每层
    for weight_key in test_layers:
        tensor = state_dict[weight_key]
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        
        # 简化层名显示
        short_name = weight_key.split('layers.')[-1]
        print(f"{short_name:<55}", end="")
        
        for config_name, config in configs:
            quantizer = SVDQuantizer(config)
            
            # 获取该层的秩
            rank = quantizer.get_rank_for_layer(weight_key)
            
            # 测试端到端误差
            error = test_end_to_end_error(quantizer, tensor, rank)
            
            print(f"{error*100:>10.2f}%", end="  ")
        
        print()
    
    # 测试直接INT4量化（不分解）作为对比
    print("\n" + "="*100)
    print("对比：直接INT4量化（不进行SVD分解）")
    print("="*100)
    
    quantizer = SVDQuantizer()
    
    for weight_key in test_layers:
        tensor = state_dict[weight_key]
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        
        # 直接INT4量化
        quantized = quantizer.symmetric_quantize_int4(tensor, 128)
        dequantized = quantizer.dequantize(quantized)
        
        error = np.linalg.norm(tensor.numpy() - dequantized) / np.linalg.norm(tensor.numpy())
        
        short_name = weight_key.split('layers.')[-1]
        print(f"{short_name}: INT4量化误差 = {error*100:.2f}%")
    
    print("\n" + "="*100)
    print("结论")
    print("="*100)


if __name__ == "__main__":
    main()
