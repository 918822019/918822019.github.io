"""
简单测试脚本：使用示例数据验证SVD量化功能

本脚本用于快速验证SVD量化功能的基本正确性。
通过创建模拟的Qwen模型层，测试SVD分解和量化流程。

测试内容：
1. 单层分解测试：测试单个层的SVD分解和量化
2. 完整流程测试：测试完整的模型处理流程
3. 压缩比分析：计算原始模型与量化后模型的大小比例
4. 保存功能测试：测试二进制格式保存功能

使用示例：
    >>> python test_simple.py
    
注意事项：
    1. 本脚本使用模拟数据，用于快速验证功能
    2. 测试结果会以详细的报告形式输出
    3. 建议在运行前确保已安装所有依赖
"""

import sys
import torch
import numpy as np
from pathlib import Path

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.main import SVDQuantizer, DecompositionConfig


def create_mock_qwen_layers():
    """
    创建模拟的Qwen模型层
    模拟2层MLP，每层有gate_proj, up_proj, down_proj
    """
    print("创建模拟的Qwen模型层...")
    
    # 模拟2层，每层有3个MLP层
    state_dict = {}
    
    for layer_num in range(2):
        # gate_proj: [intermediate_size, hidden_size] = [3584, 1024]
        state_dict[f'model.language_model.layers.{layer_num}.mlp.gate_proj.weight'] = torch.randn(3584, 1024)
        
        # up_proj: [intermediate_size, hidden_size] = [3584, 1024]
        state_dict[f'model.language_model.layers.{layer_num}.mlp.up_proj.weight'] = torch.randn(3584, 1024)
        
        # down_proj: [hidden_size, intermediate_size] = [1024, 3584]
        state_dict[f'model.language_model.layers.{layer_num}.mlp.down_proj.weight'] = torch.randn(1024, 3584)
        
        # 添加一些注意力层（不应该被分解）
        state_dict[f'model.language_model.layers.{layer_num}.self_attn.q_proj.weight'] = torch.randn(1024, 1024)
        state_dict[f'model.language_model.layers.{layer_num}.self_attn.k_proj.weight'] = torch.randn(1024, 1024)
        state_dict[f'model.language_model.layers.{layer_num}.self_attn.v_proj.weight'] = torch.randn(1024, 1024)
        state_dict[f'model.language_model.layers.{layer_num}.self_attn.o_proj.weight'] = torch.randn(1024, 1024)
    
    print(f"创建了 {len(state_dict)} 个张量")
    return state_dict


def test_svd_quantization():
    """测试SVD量化流程"""
    print("\n" + "="*60)
    print("SVD量化功能测试")
    print("="*60)
    
    # 创建模拟数据
    state_dict = create_mock_qwen_layers()
    
    # 创建配置
    config = DecompositionConfig(
        gate_proj_rank=32,
        up_proj_rank=32,
        down_proj_rank=64,
        max_relative_error=1e-3,
        quant_group_size=128,
        enable_qat=False,
        qat_samples=1000
    )
    
    # 创建量化器
    quantizer = SVDQuantizer(config)
    
    # 处理模型
    results = quantizer.process_model(state_dict)
    
    # 打印统计信息
    quantizer.print_statistics()
    
    # 验证量化误差
    print("\n" + "="*60)
    print("验证量化误差")
    print("="*60)
    
    decomposed_layers = []
    non_decomposed_layers = []
    
    for layer_name, data in results.items():
        if data['is_decomposed']:
            decomposed_layers.append(layer_name)
            
            # 反量化U_merged和Vh
            U_dequant = quantizer.dequantize(data['quantized']['U_merged'])
            Vh_dequant = quantizer.dequantize(data['quantized']['Vh'])
            
            # 重构权重
            reconstructed = U_dequant @ Vh_dequant
            
            # 计算误差
            original = data['decomposed']['U_merged'] @ data['decomposed']['Vh']
            error = np.linalg.norm(original.cpu().numpy() - reconstructed) / np.linalg.norm(original.cpu().numpy())
            
            print(f"✓ {layer_name}:")
            print(f"  - 秩: {data['decomposed']['rank']}")
            print(f"  - 原始形状: {data['original_shape']}")
            print(f"  - U_merged形状: {data['decomposed']['U_merged'].shape}")
            print(f"  - Vh形状: {data['decomposed']['Vh'].shape}")
            print(f"  - 相对误差: {data['error']:.6f}")
            print(f"  - 反量化误差: {error:.6f}")
        else:
            non_decomposed_layers.append(layer_name)
            print(f"○ {layer_name}: 直接INT4量化")
    
    # 计算压缩比
    print("\n" + "="*60)
    print("压缩比分析")
    print("="*60)
    
    original_size = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
    print(f"原始模型大小: {original_size / 1024 / 1024:.2f} MB")
    
    # 估算量化后的大小
    quantized_size = 0
    for layer_name, data in results.items():
        if data['is_decomposed']:
            # INT8 (U_merged)
            U_size = data['quantized']['U_merged'].data.size
            # INT4 (Vh)
            Vh_size = data['quantized']['Vh'].data.size * 0.5
            quantized_size += U_size + Vh_size
        else:
            # INT4
            weight_size = data['quantized']['weight'].data.size * 0.5
            quantized_size += weight_size
        
        # 偏置
        if data['bias'] is not None:
            quantized_size += data['bias'].numel() * 2  # FP16
    
    print(f"量化后大小: {quantized_size / 1024 / 1024:.2f} MB")
    print(f"压缩比: {original_size / quantized_size:.2f}x")
    
    # 测试保存功能
    print("\n" + "="*60)
    print("测试保存功能")
    print("="*60)
    
    output_path = "output/test_quantized.bin"
    import os
    os.makedirs("output", exist_ok=True)
    quantizer.save_to_binary_format(results, output_path)
    print(f"✓ 量化后的模型已保存到: {output_path}")
    
    # 验证文件大小
    file_size = os.path.getsize(output_path)
    print(f"✓ 文件大小: {file_size / 1024 / 1024:.2f} MB")
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    
    return results


def test_single_layer():
    """测试单个层的分解和量化"""
    print("\n" + "="*60)
    print("单层分解测试")
    print("="*60)
    
    # 创建一个小的测试矩阵
    weight = torch.randn(100, 50)
    
    # 创建配置
    config = DecompositionConfig(
        gate_proj_rank=10,
        up_proj_rank=10,
        down_proj_rank=20,
        max_relative_error=1e-3,
        quant_group_size=64
    )
    
    # 创建量化器
    quantizer = SVDQuantizer(config)
    
    # 测试gate_proj分解
    print("\n测试gate_proj分解:")
    result = quantizer.decompose_and_quantize_layer(
        "model.layers.0.mlp.gate_proj.weight",
        weight
    )
    
    print(f"  原始形状: {result['original_shape']}")
    print(f"  是否分解: {result['is_decomposed']}")
    if result['is_decomposed']:
        print(f"  秩: {result['decomposed']['rank']}")
        print(f"  U_merged形状: {result['decomposed']['U_merged'].shape}")
        print(f"  Vh形状: {result['decomposed']['Vh'].shape}")
        print(f"  相对误差: {result['error']:.6f}")
    
    # 测试down_proj分解
    print("\n测试down_proj分解:")
    weight = torch.randn(50, 100)
    result = quantizer.decompose_and_quantize_layer(
        "model.layers.0.mlp.down_proj.weight",
        weight
    )
    
    print(f"  原始形状: {result['original_shape']}")
    print(f"  是否分解: {result['is_decomposed']}")
    if result['is_decomposed']:
        print(f"  秩: {result['decomposed']['rank']}")
        print(f"  U_merged形状: {result['decomposed']['U_merged'].shape}")
        print(f"  Vh形状: {result['decomposed']['Vh'].shape}")
        print(f"  相对误差: {result['error']:.6f}")
    
    # 测试注意力层（不应分解）
    print("\n测试注意力层（不应分解）:")
    weight = torch.randn(100, 100)
    result = quantizer.decompose_and_quantize_layer(
        "model.layers.0.self_attn.q_proj.weight",
        weight
    )
    
    print(f"  原始形状: {result['original_shape']}")
    print(f"  是否分解: {result['is_decomposed']}")
    if not result['is_decomposed']:
        print(f"  量化类型: {result['quantized']['weight'].quant_type.value}")
    
    print("\n" + "="*60)
    print("单层测试完成！")
    print("="*60)


if __name__ == "__main__":
    # 运行单层测试
    test_single_layer()
    
    # 运行完整测试
    test_svd_quantization()
