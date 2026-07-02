"""
精度分析测试脚本：深入分析SVD量化各环节的精度损失

本脚本用于详细分析SVD量化过程中各个环节的精度损失，帮助定位精度问题。

测试内容：
1. SVD分解误差分析：不同秩下的分解误差
2. INT8量化误差分析：U_merged矩阵的量化精度
3. INT4量化误差分析：Vh矩阵的量化精度
4. 组大小对精度的影响
5. 端到端误差分析

使用示例：
    >>> python test_precision_analysis.py
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.main import SVDQuantizer, DecompositionConfig, QuantizationType


def create_test_matrix(rows: int, cols: int, rank: int = None) -> torch.Tensor:
    """
    创建测试矩阵
    
    Args:
        rows: 行数
        cols: 列数
        rank: 矩阵的秩（如果指定）
        
    Returns:
        测试矩阵
    """
    if rank is None:
        return torch.randn(rows, cols)
    
    # 创建低秩矩阵
    U = torch.randn(rows, rank)
    V = torch.randn(rank, cols)
    return U @ V


def test_svd_decomposition_error():
    """测试SVD分解误差"""
    print("\n" + "="*80)
    print("1. SVD分解误差分析")
    print("="*80)
    
    quantizer = SVDQuantizer()
    
    # 测试不同矩阵大小
    test_cases = [
        (1024, 1024, "square"),
        (3584, 1024, "gate_proj形状"),
        (1024, 3584, "down_proj形状"),
    ]
    
    for rows, cols, name in test_cases:
        print(f"\n矩阵形状: ({rows}, {cols}) - {name}")
        print("-" * 60)
        
        weight = create_test_matrix(rows, cols)
        
        # 测试不同秩
        for rank in [16, 32, 64, 128]:
            if rank > min(rows, cols):
                continue
                
            U_merged, Vh, original = quantizer.svd_decompose(weight, rank)
            reconstructed = U_merged @ Vh
            error = quantizer.calculate_relative_error(original, reconstructed)
            
            # 计算压缩比
            original_size = rows * cols
            decomposed_size = rows * rank + rank * cols
            compression = original_size / decomposed_size
            
            print(f"  秩={rank:3d}: 相对误差={error:.6f}, 压缩比={compression:.2f}x")


def test_int8_quantization_error():
    """测试INT8量化误差"""
    print("\n" + "="*80)
    print("2. INT8量化误差分析")
    print("="*80)
    
    quantizer = SVDQuantizer()
    
    # 测试不同矩阵大小
    test_cases = [
        (1024, 32, "U_merged小矩阵"),
        (3584, 32, "U_merged大矩阵"),
        (100, 50, "测试矩阵"),
    ]
    
    for rows, cols, name in test_cases:
        print(f"\n矩阵形状: ({rows}, {cols}) - {name}")
        print("-" * 60)
        
        # 测试不同组大小
        for group_size in [32, 64, 128, 256]:
            tensor = torch.randn(rows, cols)
            quantized = quantizer.symmetric_quantize_int8(tensor, group_size)
            dequantized = quantizer.dequantize(quantized)
            
            error = np.linalg.norm(tensor.numpy() - dequantized) / np.linalg.norm(tensor.numpy())
            
            # 计算存储大小
            n_groups = (rows * cols + group_size - 1) // group_size
            storage = n_groups * (group_size + 1)  # 数据 + 缩放因子
            
            print(f"  组大小={group_size:3d}: 量化误差={error:.6f}, 存储={storage/1024:.1f}KB")


def test_int4_quantization_error():
    """测试INT4量化误差"""
    print("\n" + "="*80)
    print("3. INT4量化误差分析")
    print("="*80)
    
    quantizer = SVDQuantizer()
    
    # 测试不同矩阵大小
    test_cases = [
        (32, 1024, "Vh小矩阵"),
        (32, 3584, "Vh大矩阵"),
        (1024, 1024, "注意力层"),
    ]
    
    for rows, cols, name in test_cases:
        print(f"\n矩阵形状: ({rows}, {cols}) - {name}")
        print("-" * 60)
        
        # 测试不同组大小
        for group_size in [32, 64, 128, 256]:
            if rows * cols < group_size:
                continue
                
            tensor = torch.randn(rows, cols)
            quantized = quantizer.symmetric_quantize_int4(tensor, group_size)
            dequantized = quantizer.dequantize(quantized)
            
            error = np.linalg.norm(tensor.numpy() - dequantized) / np.linalg.norm(tensor.numpy())
            
            # 计算存储大小
            n_groups = (rows * cols + group_size - 1) // group_size
            storage = n_groups * (group_size // 2 + 1)  # 打包数据 + 缩放因子
            
            print(f"  组大小={group_size:3d}: 量化误差={error:.6f}, 存储={storage/1024:.1f}KB")


def test_end_to_end_error():
    """测试端到端误差"""
    print("\n" + "="*80)
    print("4. 端到端误差分析")
    print("="*80)
    
    # 测试不同配置
    configs = [
        DecompositionConfig(gate_proj_rank=16, up_proj_rank=16, down_proj_rank=32, quant_group_size=64),
        DecompositionConfig(gate_proj_rank=32, up_proj_rank=32, down_proj_rank=64, quant_group_size=128),
        DecompositionConfig(gate_proj_rank=64, up_proj_rank=64, down_proj_rank=128, quant_group_size=256),
    ]
    
    # 创建测试数据（模拟Qwen模型）
    state_dict = {
        'model.layers.0.mlp.gate_proj.weight': torch.randn(3584, 1024),
        'model.layers.0.mlp.up_proj.weight': torch.randn(3584, 1024),
        'model.layers.0.mlp.down_proj.weight': torch.randn(1024, 3584),
        'model.layers.0.self_attn.q_proj.weight': torch.randn(1024, 1024),
        'model.layers.0.self_attn.k_proj.weight': torch.randn(1024, 1024),
        'model.layers.0.self_attn.v_proj.weight': torch.randn(1024, 1024),
        'model.layers.0.self_attn.o_proj.weight': torch.randn(1024, 1024),
    }
    
    for i, config in enumerate(configs):
        print(f"\n配置 {i+1}: gate_proj_rank={config.gate_proj_rank}, up_proj_rank={config.up_proj_rank}, "
              f"down_proj_rank={config.down_proj_rank}, group_size={config.quant_group_size}")
        print("-" * 80)
        
        quantizer = SVDQuantizer(config)
        results = quantizer.process_model(state_dict)
        
        # 分析每种层的误差
        svd_errors = []
        int4_errors = []
        
        for layer_name, data in results.items():
            if data['is_decomposed']:
                # 反量化并计算误差
                U_dequant = quantizer.dequantize(data['quantized']['U_merged'])
                Vh_dequant = quantizer.dequantize(data['quantized']['Vh'])
                reconstructed = U_dequant @ Vh_dequant
                original = data['decomposed']['U_merged'] @ data['decomposed']['Vh']
                error = np.linalg.norm(original.cpu().numpy() - reconstructed) / np.linalg.norm(original.cpu().numpy())
                svd_errors.append(error)
                
                print(f"  {layer_name}: SVD+量化误差={error:.6f}, SVD分解误差={data['error']:.6f}")
            else:
                # 反量化并计算误差
                dequantized = quantizer.dequantize(data['quantized']['weight'])
                error = np.linalg.norm(state_dict[layer_name + '.weight'].numpy() - dequantized) / \
                        np.linalg.norm(state_dict[layer_name + '.weight'].numpy())
                int4_errors.append(error)
                
                print(f"  {layer_name}: INT4量化误差={error:.6f}")
        
        if svd_errors:
            print(f"\n  平均SVD+量化误差: {np.mean(svd_errors):.6f}")
        if int4_errors:
            print(f"  平均INT4量化误差: {np.mean(int4_errors):.6f}")


def test_error_accumulation():
    """测试误差累积"""
    print("\n" + "="*80)
    print("5. 误差累积分析")
    print("="*80)
    
    config = DecompositionConfig(gate_proj_rank=32, up_proj_rank=32, down_proj_rank=64, quant_group_size=128)
    quantizer = SVDQuantizer(config)
    
    # 创建测试矩阵
    rows, cols = 3584, 1024
    weight = torch.randn(rows, cols)
    
    print(f"测试矩阵形状: ({rows}, {cols})")
    print("-" * 60)
    
    # 步骤1: SVD分解
    U_merged, Vh, original = quantizer.svd_decompose(weight, 32)
    reconstructed_svd = U_merged @ Vh
    svd_error = quantizer.calculate_relative_error(original, reconstructed_svd)
    print(f"步骤1 - SVD分解误差: {svd_error:.6f}")
    
    # 步骤2: INT8量化U_merged
    U_quantized = quantizer.symmetric_quantize_int8(U_merged, 128)
    U_dequantized = quantizer.dequantize(U_quantized)
    U_error = np.linalg.norm(U_merged.numpy() - U_dequantized) / np.linalg.norm(U_merged.numpy())
    print(f"步骤2 - U_merged INT8量化误差: {U_error:.6f}")
    
    # 步骤3: INT4量化Vh
    Vh_quantized = quantizer.symmetric_quantize_int4(Vh, 128)
    Vh_dequantized = quantizer.dequantize(Vh_quantized)
    Vh_error = np.linalg.norm(Vh.numpy() - Vh_dequantized) / np.linalg.norm(Vh.numpy())
    print(f"步骤3 - Vh INT4量化误差: {Vh_error:.6f}")
    
    # 步骤4: 端到端误差
    reconstructed_final = U_dequantized @ Vh_dequantized
    final_error = np.linalg.norm(original.numpy() - reconstructed_final) / np.linalg.norm(original.numpy())
    print(f"步骤4 - 端到端误差: {final_error:.6f}")
    
    # 分析误差贡献
    print(f"\n误差贡献分析:")
    print(f"  SVD分解贡献: {svd_error/final_error*100:.1f}%")
    print(f"  量化误差贡献: {(final_error-svd_error)/final_error*100:.1f}%")


def test_group_size_impact():
    """测试组大小对精度的影响"""
    print("\n" + "="*80)
    print("6. 组大小对精度的影响")
    print("="*80)
    
    rows, cols = 3584, 1024
    weight = torch.randn(rows, cols)
    
    print(f"测试矩阵形状: ({rows}, {cols})")
    print("-" * 60)
    
    group_sizes = [16, 32, 64, 128, 256, 512]
    
    print("\nINT8量化:")
    for group_size in group_sizes:
        if rows * cols < group_size:
            continue
            
        quantizer = SVDQuantizer()
        quantized = quantizer.symmetric_quantize_int8(weight, group_size)
        dequantized = quantizer.dequantize(quantized)
        error = np.linalg.norm(weight.numpy() - dequantized) / np.linalg.norm(weight.numpy())
        print(f"  组大小={group_size:3d}: 误差={error:.6f}")
    
    print("\nINT4量化:")
    for group_size in group_sizes:
        if rows * cols < group_size or group_size % 2 != 0:
            continue
            
        quantizer = SVDQuantizer()
        quantized = quantizer.symmetric_quantize_int4(weight, group_size)
        dequantized = quantizer.dequantize(quantized)
        error = np.linalg.norm(weight.numpy() - dequantized) / np.linalg.norm(weight.numpy())
        print(f"  组大小={group_size:3d}: 误差={error:.6f}")


def main():
    """主函数"""
    print("SVD量化精度分析测试")
    print("="*80)
    
    # 运行各项测试
    test_svd_decomposition_error()
    test_int8_quantization_error()
    test_int4_quantization_error()
    test_end_to_end_error()
    test_error_accumulation()
    test_group_size_impact()
    
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)


if __name__ == "__main__":
    main()
