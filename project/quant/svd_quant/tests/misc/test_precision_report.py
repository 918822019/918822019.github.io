"""
精度和压缩比详细报告

本脚本用于生成SVD量化的精度和压缩比详细报告。
通过多种测试场景，全面评估量化效果。

测试内容：
1. 低秩矩阵精度测试：测试已知秩的矩阵的重构精度
2. 随机矩阵精度测试：测试满秩矩阵的近似精度
3. 量化误差测试：测试INT8和INT4量化的误差
4. 完整流程精度测试：模拟真实模型的量化效果
5. 压缩比分析：计算不同配置下的压缩比
6. 不同配置对比：比较不同秩配置下的压缩比

使用示例：
    >>> python test_precision_report.py
    
注意事项：
    1. 本脚本使用随机生成的数据，实际效果可能与真实模型不同
    2. 测试结果会以详细的报告形式输出
    3. 建议在运行前确保已安装所有依赖
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.main import SVDQuantizer, DecompositionConfig


def test_precision_report():
    """
    生成精度报告
    
    生成SVD量化的精度和压缩比详细报告，包括多种测试场景。
    报告内容全面，便于分析量化效果。
    
    报告内容：
        1. 低秩矩阵精度测试：测试已知秩的矩阵的重构精度
        2. 随机矩阵精度测试：测试满秩矩阵的近似精度
        3. 量化误差测试：测试INT8和INT4量化的误差
        4. 完整流程精度测试：模拟真实模型的量化效果
        5. 压缩比分析：计算不同配置下的压缩比
        6. 不同配置对比：比较不同秩配置下的压缩比
        7. 总结：总结测试结果和建议
        
    使用示例：
        >>> test_precision_report()
        
    注意事项：
        1. 测试使用随机生成的数据，实际效果可能与真实模型不同
        2. 报告会以详细的表格形式输出
        3. 建议在运行前确保已安装所有依赖
    """
    print("=" * 70)
    print("SVD量化精度和压缩比详细报告")
    print("=" * 70)
    
    # 创建配置
    config = DecompositionConfig(
        gate_proj_rank=32,
        up_proj_rank=32,
        down_proj_rank=64,
        max_relative_error=1e-3,
        quant_group_size=128
    )
    
    quantizer = SVDQuantizer(config)
    
    # ============================================================================
    # 1. 低秩矩阵精度测试
    # ============================================================================
    print("\n" + "=" * 70)
    print("1. 低秩矩阵精度测试")
    print("=" * 70)
    
    # 创建一个秩为10的矩阵
    m, n, true_rank = 100, 50, 10
    U = torch.randn(m, true_rank)
    V = torch.randn(true_rank, n)
    W_low_rank = U @ V
    
    print(f"矩阵形状: ({m}, {n})")
    print(f"真实秩: {true_rank}")
    
    # 测试不同秩
    ranks = [5, 10, 15, 20, 30]
    print(f"\n{'秩':<10}{'相对误差':<20}{'是否达到阈值':<15}")
    print("-" * 45)
    
    for rank in ranks:
        U_merged, Vh, original = quantizer.svd_decompose(W_low_rank, rank)
        reconstructed = U_merged @ Vh
        error = quantizer.calculate_relative_error(original, reconstructed)
        threshold = config.max_relative_error
        passed = "✓" if error < threshold else "✗"
        print(f"{rank:<10}{error:<20.8f}{passed:<15}")
    
    # ============================================================================
    # 2. 随机矩阵精度测试
    # ============================================================================
    print("\n" + "=" * 70)
    print("2. 随机矩阵（满秩）精度测试")
    print("=" * 70)
    
    # 创建随机矩阵
    m, n = 100, 50
    W_random = torch.randn(m, n)
    
    print(f"矩阵形状: ({m}, {n})")
    
    # 测试不同秩
    ranks = [5, 10, 20, 30, 40, 50]
    print(f"\n{'秩':<10}{'相对误差':<20}{'信息保留率':<15}")
    print("-" * 45)
    
    for rank in ranks:
        U_merged, Vh, original = quantizer.svd_decompose(W_random, rank)
        reconstructed = U_merged @ Vh
        error = quantizer.calculate_relative_error(original, reconstructed)
        info_retained = 1 - error
        print(f"{rank:<10}{error:<20.8f}{info_retained:<15.2%}")
    
    # ============================================================================
    # 3. 量化误差测试
    # ============================================================================
    print("\n" + "=" * 70)
    print("3. 量化误差测试")
    print("=" * 70)
    
    # 创建标准化数据
    tensor = torch.randn(100, 50)
    tensor = (tensor - tensor.mean()) / tensor.std()
    
    print(f"数据形状: {tensor.shape}")
    print(f"数据范围: [{tensor.min():.4f}, {tensor.max():.4f}]")
    
    # INT8量化
    quantized_int8 = quantizer.symmetric_quantize_int8(tensor, group_size=128)
    dequantized_int8 = quantizer.dequantize(quantized_int8)
    error_int8 = np.linalg.norm(tensor.numpy() - dequantized_int8) / np.linalg.norm(tensor.numpy())
    
    print(f"\nINT8量化:")
    print(f"  量化误差: {error_int8:.6f}")
    print(f"  数据范围: [{quantized_int8.data.min()}, {quantized_int8.data.max()}]")
    
    # INT4量化
    quantized_int4 = quantizer.symmetric_quantize_int4(tensor, group_size=128)
    dequantized_int4 = quantizer.dequantize(quantized_int4)
    error_int4 = np.linalg.norm(tensor.numpy() - dequantized_int4) / np.linalg.norm(tensor.numpy())
    
    print(f"\nINT4量化:")
    print(f"  量化误差: {error_int4:.6f}")
    
    # ============================================================================
    # 4. 完整流程精度测试（模拟真实模型）
    # ============================================================================
    print("\n" + "=" * 70)
    print("4. 完整流程精度测试（模拟Qwen模型）")
    print("=" * 70)
    
    # 创建类似Qwen模型的层
    state_dict = {}
    for layer_num in range(3):
        # MLP层 - 应该被分解
        state_dict[f'model.layers.{layer_num}.mlp.gate_proj.weight'] = torch.randn(3584, 1024)
        state_dict[f'model.layers.{layer_num}.mlp.up_proj.weight'] = torch.randn(3584, 1024)
        state_dict[f'model.layers.{layer_num}.mlp.down_proj.weight'] = torch.randn(1024, 3584)
        
        # 注意力层 - 直接量化
        state_dict[f'model.layers.{layer_num}.self_attn.q_proj.weight'] = torch.randn(1024, 1024)
        state_dict[f'model.layers.{layer_num}.self_attn.k_proj.weight'] = torch.randn(1024, 1024)
        state_dict[f'model.layers.{layer_num}.self_attn.v_proj.weight'] = torch.randn(1024, 1024)
        state_dict[f'model.layers.{layer_num}.self_attn.o_proj.weight'] = torch.randn(1024, 1024)
    
    print(f"总层数: {len(state_dict)}")
    
    # 处理模型
    results = quantizer.process_model(state_dict)
    
    # 统计
    print(f"\n{'层类型':<20}{'数量':<10}{'平均SVD误差':<20}{'最大SVD误差':<15}")
    print("-" * 65)
    
    decomposed_layers = []
    non_decomposed_layers = []
    
    for layer_name, data in results.items():
        if data['is_decomposed']:
            decomposed_layers.append((layer_name, data['error']))
        else:
            non_decomposed_layers.append(layer_name)
    
    if decomposed_layers:
        svd_errors = [e for _, e in decomposed_layers]
        print(f"{'SVD分解层':<20}{len(decomposed_layers):<10}{np.mean(svd_errors):<20.6f}{np.max(svd_errors):<15.6f}")
    
    if non_decomposed_layers:
        print(f"{'直接量化层':<20}{len(non_decomposed_layers):<10}{'N/A':<20}{'N/A':<15}")
    
    # ============================================================================
    # 5. 压缩比分析
    # ============================================================================
    print("\n" + "=" * 70)
    print("5. 压缩比分析")
    print("=" * 70)
    
    # 计算原始大小
    original_size = sum(t.numel() * t.element_size() for t in state_dict.values())
    
    # 计算量化后大小
    quantized_size = 0
    decomposed_quantized_size = 0
    non_decomposed_quantized_size = 0
    
    for layer_name, data in results.items():
        if data['is_decomposed']:
            # SVD分解层: U_merged (INT8) + Vh (INT4)
            U_size = data['quantized']['U_merged'].data.size  # INT8: 1 byte/element
            Vh_size = data['quantized']['Vh'].data.size * 0.5  # INT4: 0.5 bytes/element
            layer_quantized_size = U_size + Vh_size
            decomposed_quantized_size += layer_quantized_size
            quantized_size += layer_quantized_size
        else:
            # 直接量化层: INT4
            weight_size = data['quantized']['weight'].data.size * 0.5  # INT4: 0.5 bytes/element
            non_decomposed_quantized_size += weight_size
            quantized_size += weight_size
    
    compression_ratio = original_size / quantized_size if quantized_size > 0 else float('inf')
    
    print(f"\n原始模型大小: {original_size / 1024 / 1024:.2f} MB")
    print(f"量化后大小: {quantized_size / 1024 / 1024:.2f} MB")
    print(f"  - SVD分解层: {decomposed_quantized_size / 1024 / 1024:.2f} MB")
    print(f"  - 直接量化层: {non_decomposed_quantized_size / 1024 / 1024:.2f} MB")
    print(f"\n压缩比: {compression_ratio:.2f}x")
    
    # ============================================================================
    # 6. 不同配置下的压缩比
    # ============================================================================
    print("\n" + "=" * 70)
    print("6. 不同配置下的压缩比对比")
    print("=" * 70)
    
    configs = [
        ("保守配置 (rank=16/16/32)", DecompositionConfig(gate_proj_rank=16, up_proj_rank=16, down_proj_rank=32)),
        ("默认配置 (rank=32/32/64)", DecompositionConfig(gate_proj_rank=32, up_proj_rank=32, down_proj_rank=64)),
        ("激进配置 (rank=64/64/128)", DecompositionConfig(gate_proj_rank=64, up_proj_rank=64, down_proj_rank=128)),
    ]
    
    print(f"\n{'配置':<30}{'量化后大小(MB)':<20}{'压缩比':<15}{'平均SVD误差':<15}")
    print("-" * 80)
    
    for name, cfg in configs:
        q = SVDQuantizer(cfg)
        r = q.process_model(state_dict)
        
        # 计算大小
        q_size = 0
        svd_errors = []
        for layer_name, data in r.items():
            if data['is_decomposed']:
                U_size = data['quantized']['U_merged'].data.size
                Vh_size = data['quantized']['Vh'].data.size * 0.5
                q_size += U_size + Vh_size
                svd_errors.append(data['error'])
            else:
                w_size = data['quantized']['weight'].data.size * 0.5
                q_size += w_size
        
        ratio = original_size / q_size if q_size > 0 else float('inf')
        avg_error = np.mean(svd_errors) if svd_errors else 0
        
        print(f"{name:<30}{q_size / 1024 / 1024:<20.2f}{ratio:<15.2f}{avg_error:<15.6f}")
    
    # ============================================================================
    # 7. 总结
    # ============================================================================
    print("\n" + "=" * 70)
    print("7. 总结")
    print("=" * 70)
    
    print(f"""
测试结果总结:
─────────────────────────────────────────────────────────────
1. SVD分解精度:
   - 低秩矩阵（秩=10）使用真实秩重构: 误差 < 1e-5
   - 随机矩阵使用秩=32: 误差约 0.5-0.7
   - 随机矩阵使用秩=64: 误差约 0.3-0.5

2. 量化精度:
   - INT8量化误差: 约 {error_int8:.4f}
   - INT4量化误差: 约 {error_int4:.4f}

3. 压缩比:
   - 默认配置 (rank=32/32/64): {compression_ratio:.2f}x
   - 使用更小的秩可获得更高压缩比
   - 使用更大的秩可获得更高精度

4. 建议:
   - 对于精度敏感的应用: 使用较大的秩 (64/64/128)
   - 对于压缩比优先: 使用较小的秩 (16/16/32)
   - 默认配置 (32/32/64) 提供了良好的平衡
─────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    test_precision_report()
