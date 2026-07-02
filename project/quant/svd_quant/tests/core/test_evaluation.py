"""
评估测试脚本：展示压缩比和精度损失指标

本脚本用于测试和展示SVD量化的压缩比和精度损失指标。
通过创建模拟的模型状态字典，测试不同配置下的量化效果。

测试内容：
1. 压缩比测试：计算原始模型与量化后模型的大小比例
2. 精度损失测试：计算MSE、MAE、RMSE等精度指标
3. 不同配置对比：比较保守、默认、激进三种配置的效果

使用示例：
    >>> python test_evaluation.py
    
注意事项：
    1. 本脚本使用随机生成的数据，实际效果可能与真实模型不同
    2. 测试结果会以表格形式输出，便于比较
    3. 建议在运行前确保已安装所有依赖
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.main import SVDQuantizer, DecompositionConfig


def create_mock_model_state_dict():
    """
    创建模拟的模型状态字典
    
    创建一个模拟的模型状态字典，用于测试SVD量化功能。
    模拟了2层Transformer模型，每层包含MLP层和注意力层。
    
    模拟的层结构：
        - MLP层（会被SVD分解）：
          - gate_proj: 门控投影层
          - up_proj: 上投影层
          - down_proj: 下投影层
          
        - 注意力层（直接INT4量化）：
          - q_proj: 查询投影层
          - k_proj: 键投影层
          - v_proj: 值投影层
          - o_proj: 输出投影层
          
    Returns:
        dict: 模型状态字典
            - 键为层名称，值为随机生成的权重张量
            - 形状模拟了真实的Qwen模型结构
            
    使用示例：
        >>> state_dict = create_mock_model_state_dict()
        >>> print(f"层数: {len(state_dict)}")
        >>> for name, tensor in state_dict.items():
        ...     print(f"{name}: {tensor.shape}")
    """
    state_dict = {}
    
    # 模拟2层，每层有MLP和注意力层
    for layer_num in range(2):
        # MLP层（会被分解）
        state_dict[f'model.layers.{layer_num}.mlp.gate_proj.weight'] = torch.randn(3584, 1024)
        state_dict[f'model.layers.{layer_num}.mlp.up_proj.weight'] = torch.randn(3584, 1024)
        state_dict[f'model.layers.{layer_num}.mlp.down_proj.weight'] = torch.randn(1024, 3584)
        
        # 注意力层（直接量化）
        state_dict[f'model.layers.{layer_num}.self_attn.q_proj.weight'] = torch.randn(1024, 1024)
        state_dict[f'model.layers.{layer_num}.self_attn.k_proj.weight'] = torch.randn(1024, 1024)
        state_dict[f'model.layers.{layer_num}.self_attn.v_proj.weight'] = torch.randn(1024, 1024)
        state_dict[f'model.layers.{layer_num}.self_attn.o_proj.weight'] = torch.randn(1024, 1024)
    
    return state_dict


def test_compression_and_metrics():
    """
    测试压缩比和精度指标
    
    测试SVD量化的压缩比和精度损失指标，包括：
        1. 处理模型：对模拟的模型进行SVD分解和量化
        2. 计算压缩比：计算原始模型与量化后模型的大小比例
        3. 评估量化质量：计算MSE、MAE、RMSE等精度指标
        4. 打印统计信息：输出详细的统计和评估结果
        
    测试内容：
        - 压缩比：原始模型大小/量化后模型大小
        - 精度指标：MSE、MAE、RMSE
        - 相对误差：SVD分解的相对误差
        - 各层详情：每层的分解类型、秩、误差
        
    Returns:
        Tuple[float, Dict[str, float]]: 压缩比和评估指标字典
            - 压缩比：原始模型大小/量化后模型大小
            - 评估指标：包含MSE、MAE、RMSE等
            
    使用示例：
        >>> compression_ratio, metrics = test_compression_and_metrics()
        >>> print(f"压缩比: {compression_ratio:.2f}x")
        >>> print(f"MSE: {metrics['mse']:.6f}")
    """
    print("=" * 70)
    print("SVD量化评估测试")
    print("=" * 70)
    
    # 创建模型状态字典
    state_dict = create_mock_model_state_dict()
    
    # 创建配置
    config = DecompositionConfig(
        gate_proj_rank=32,
        up_proj_rank=32,
        down_proj_rank=64,
        max_relative_error=1e-3,
        quant_group_size=128
    )
    
    # 创建量化器
    quantizer = SVDQuantizer(config)
    
    # 处理模型
    print("\n[1/4] 处理模型...")
    results = quantizer.process_model(state_dict)
    
    # 计算压缩比
    print("\n[2/4] 计算压缩比...")
    compression_ratio = quantizer.calculate_compression_ratio(state_dict, results)
    
    # 评估量化质量
    print("\n[3/4] 评估量化质量（MSE、MAE）...")
    metrics = quantizer.evaluate_with_validation_data(state_dict, results)
    
    # 打印统计信息
    print("\n[4/4] 打印统计信息...")
    quantizer.print_statistics()
    
    # 详细输出
    print("\n" + "=" * 70)
    print("详细评估结果")
    print("=" * 70)
    
    print("\n【压缩比详情】")
    print(f"  原始模型大小: {quantizer.stats['original_size'] / 1024 / 1024:.2f} MB")
    print(f"  量化后大小: {quantizer.stats['quantized_size'] / 1024 / 1024:.2f} MB")
    print(f"  压缩比: {compression_ratio:.2f}x")
    print(f"  空间节省: {(1 - 1/compression_ratio) * 100:.1f}%")
    
    print("\n【精度损失指标】")
    print(f"  MSE (均方误差): {metrics['mse']:.6f}")
    print(f"  MAE (平均绝对误差): {metrics['mae']:.6f}")
    print(f"  RMSE (均方根误差): {metrics['rmse']:.6f}")
    
    # 计算相对误差
    print("\n【相对误差统计】")
    if quantizer.stats['errors']:
        print(f"  最大相对误差: {max(quantizer.stats['errors']):.6f}")
        print(f"  平均相对误差: {np.mean(quantizer.stats['errors']):.6f}")
        print(f"  最小相对误差: {min(quantizer.stats['errors']):.6f}")
    
    # 各层详细信息
    print("\n【各层分解详情】")
    print(f"{'层名称':<50}{'类型':<15}{'秩':<10}{'相对误差':<15}")
    print("-" * 90)
    
    for layer_name, data in results.items():
        if data['is_decomposed']:
            layer_type = "SVD分解"
            rank = data['decomposed']['rank']
            error = data['error']
            print(f"{layer_name:<50}{layer_type:<15}{rank:<10}{error:<15.6f}")
        else:
            layer_type = "直接量化"
            print(f"{layer_name:<50}{layer_type:<15}{'N/A':<10}{'N/A':<15}")
    
    return compression_ratio, metrics


def test_different_configs():
    """
    测试不同配置的压缩比和精度
    
    比较三种不同配置下的量化效果：
        1. 保守配置：较小的秩，追求高压缩比
        2. 默认配置：平衡的秩，兼顾压缩比和精度
        3. 激进配置：较大的秩，追求高精度
        
    配置说明：
        - 保守配置 (rank=16/16/32):
          - gate_proj秩=16，up_proj秩=16，down_proj秩=32
          - 压缩比最高，但精度损失较大
          - 适合对存储空间要求严格的场景
          
        - 默认配置 (rank=32/32/64):
          - gate_proj秩=32，up_proj秩=32，down_proj秩=64
          - 压缩比和精度平衡
          - 适合大多数应用场景
          
        - 激进配置 (rank=64/64/128):
          - gate_proj秩=64，up_proj秩=64，down_proj秩=128
          - 精度最高，但压缩比较低
          - 适合对精度要求严格的场景
          
    测试内容：
        - 压缩比：原始模型大小/量化后模型大小
        - 精度指标：MSE、MAE、RMSE
        - 配置推荐：根据测试结果推荐最佳配置
        
    使用示例：
        >>> test_different_configs()
    """
    print("\n" + "=" * 70)
    print("不同配置对比测试")
    print("=" * 70)
    
    # 创建模型状态字典
    state_dict = create_mock_model_state_dict()
    
    # 不同配置
    configs = [
        ("保守配置 (rank=16/16/32)", DecompositionConfig(gate_proj_rank=16, up_proj_rank=16, down_proj_rank=32)),
        ("默认配置 (rank=32/32/64)", DecompositionConfig(gate_proj_rank=32, up_proj_rank=32, down_proj_rank=64)),
        ("激进配置 (rank=64/64/128)", DecompositionConfig(gate_proj_rank=64, up_proj_rank=64, down_proj_rank=128)),
    ]
    
    print(f"\n{'配置':<35}{'压缩比':<15}{'MSE':<15}{'MAE':<15}{'RMSE':<15}")
    print("-" * 95)
    
    results_list = []
    
    for name, config in configs:
        quantizer = SVDQuantizer(config)
        results = quantizer.process_model(state_dict)
        compression_ratio = quantizer.calculate_compression_ratio(state_dict, results)
        metrics = quantizer.evaluate_with_validation_data(state_dict, results)
        
        results_list.append({
            'name': name,
            'compression_ratio': compression_ratio,
            'metrics': metrics
        })
        
        print(f"{name:<35}{compression_ratio:<15.2f}{metrics['mse']:<15.6f}{metrics['mae']:<15.6f}{metrics['rmse']:<15.6f}")
    
    # 推荐配置
    print("\n" + "=" * 70)
    print("配置推荐")
    print("=" * 70)
    
    # 根据需求推荐配置
    best_compression = max(results_list, key=lambda x: x['compression_ratio'])
    best_precision = min(results_list, key=lambda x: x['metrics']['mse'])
    
    print(f"\n最高压缩比配置: {best_compression['name']}")
    print(f"  压缩比: {best_compression['compression_ratio']:.2f}x")
    print(f"  MSE: {best_compression['metrics']['mse']:.6f}")
    
    print(f"\n最高精度配置: {best_precision['name']}")
    print(f"  压缩比: {best_precision['compression_ratio']:.2f}x")
    print(f"  MSE: {best_precision['metrics']['mse']:.6f}")
    
    # 平衡配置
    balanced = results_list[1]  # 默认配置
    print(f"\n平衡配置（推荐）: {balanced['name']}")
    print(f"  压缩比: {balanced['compression_ratio']:.2f}x")
    print(f"  MSE: {balanced['metrics']['mse']:.6f}")


def main():
    """
    主函数
    
    运行所有评估测试，包括：
        1. 压缩比和精度指标测试
        2. 不同配置对比测试
        
    使用示例：
        >>> python test_evaluation.py
        
    注意事项：
        1. 测试使用随机生成的数据，实际效果可能与真实模型不同
        2. 测试结果会以表格形式输出，便于比较
        3. 建议在运行前确保已安装所有依赖
    """
    # 测试压缩比和精度指标
    compression_ratio, metrics = test_compression_and_metrics()
    
    # 测试不同配置
    test_different_configs()
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
