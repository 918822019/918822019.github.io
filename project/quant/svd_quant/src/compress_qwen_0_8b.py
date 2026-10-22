"""
压缩Qwen3.5-0.8B模型的脚本

使用SVD量化工具对Qwen3.5-0.8B模型进行压缩。
"""

import os
import sys
import torch
import numpy as np
from safetensors import safe_open
from pathlib import Path

# 添加当前目录到路径，以便导入main模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import SVDQuantizer, DecompositionConfig


def load_model_weights(model_path: str) -> dict:
    """
    加载safetensors格式的模型权重
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        模型状态字典
    """
    print(f"Loading model from: {model_path}")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # 加载safetensors文件
    state_dict = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # 转换为Float32，因为SVD分解不支持BFloat16
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            state_dict[key] = tensor
    
    print(f"Loaded {len(state_dict)} tensors")
    
    # 打印一些统计信息
    total_params = sum(p.numel() for p in state_dict.values())
    total_size_mb = sum(p.numel() * p.element_size() for p in state_dict.values()) / 1024 / 1024
    
    print(f"Total parameters: {total_params:,}")
    print(f"Total size: {total_size_mb:.2f} MB")
    
    return state_dict


def main():
    """主函数"""
    
    # 模型路径
    model_path = str(Path(__file__).resolve().parents[3] / "data" / "models" / "Qwen3.5-0.8B" / "model.safetensors-00001-of-00001.safetensors")
    
    # 输出路径（使用不同的文件名以区分实验）
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "qwen_0_8b_quantized_v5.bin")
    
    # 1. 加载模型权重
    print("=" * 60)
    print("Step 1: Loading model weights")
    print("=" * 60)
    
    try:
        state_dict = load_model_weights(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 2. 创建SVD量化配置
    print("\n" + "=" * 60)
    print("Step 2: Configuring SVD quantization")
    print("=" * 60)
    
    # 使用极大秩配置，追求最高精度
    config = DecompositionConfig(
        gate_proj_rank=512,     # 极大gate_proj层的分解秩
        up_proj_rank=512,       # 极大up_proj层的分解秩
        down_proj_rank=1024,    # 极大down_proj层的分解秩
        max_relative_error=1e-3,  # 最大允许相对误差
        quant_group_size=8,     # 极小量化组大小以提高量化精度
        enable_qat=False,       # 不启用量化感知微调
        qat_samples=1000        # 量化感知微调样本数量
    )
    
    print(f"Configuration:")
    print(f"  - gate_proj rank: {config.gate_proj_rank}")
    print(f"  - up_proj rank: {config.up_proj_rank}")
    print(f"  - down_proj rank: {config.down_proj_rank}")
    print(f"  - quant group size: {config.quant_group_size}")
    print(f"  - max relative error: {config.max_relative_error}")
    
    # 3. 创建量化器并处理模型
    print("\n" + "=" * 60)
    print("Step 3: Processing model with SVD quantization")
    print("=" * 60)
    
    quantizer = SVDQuantizer(config)
    
    # 处理模型
    results = quantizer.process_model(state_dict)
    
    # 4. 计算压缩比
    print("\n" + "=" * 60)
    print("Step 4: Calculating compression ratio")
    print("=" * 60)
    
    compression_ratio = quantizer.calculate_compression_ratio(state_dict, results)
    
    # 5. 评估量化质量
    print("\n" + "=" * 60)
    print("Step 5: Evaluating quantization quality")
    print("=" * 60)
    
    metrics = quantizer.evaluate_with_validation_data(state_dict, results)
    
    # 6. 保存为二进制格式
    print("\n" + "=" * 60)
    print("Step 6: Saving quantized model")
    print("=" * 60)
    
    quantizer.save_to_binary_format(results, output_path)
    
    # 7. 打印统计信息
    print("\n" + "=" * 60)
    print("Step 7: Final statistics")
    print("=" * 60)
    
    quantizer.print_statistics()
    
    # 8. 验证量化误差
    print("\n" + "=" * 60)
    print("Step 8: Verifying quantization errors")
    print("=" * 60)
    
    # 只验证前几层，避免输出过多
    layer_count = 0
    for layer_name, data in results.items():
        if layer_count >= 5:  # 只验证前5层
            break
            
        if data['is_decomposed']:
            # 反量化U_merged和Vh
            U_dequant = quantizer.dequantize(data['quantized']['U_merged'])
            Vh_dequant = quantizer.dequantize(data['quantized']['Vh'])
            
            # 重构权重
            reconstructed = U_dequant @ Vh_dequant
            
            # 计算误差
            original = data['decomposed']['U_merged'] @ data['decomposed']['Vh']
            error = np.linalg.norm(original.cpu().numpy() - reconstructed) / np.linalg.norm(original.cpu().numpy())
            print(f"Layer {layer_name}: dequantization error = {error:.6f}")
        
        layer_count += 1
    
    # 9. 打印最终结果
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    print(f"Model: Qwen3.5-0.8B")
    print(f"Original size: {quantizer.stats['original_size'] / 1024 / 1024:.2f} MB")
    print(f"Quantized size: {quantizer.stats['quantized_size'] / 1024 / 1024:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Output file: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    # 打印精度指标
    print(f"\nQuality metrics:")
    print(f"  - MSE: {metrics['mse']:.6f}")
    print(f"  - MAE: {metrics['mae']:.6f}")
    print(f"  - RMSE: {metrics['rmse']:.6f}")
    print(f"  - Max relative error: {quantizer.stats['max_error']:.6f}")
    print(f"  - Average relative error: {quantizer.stats['avg_error']:.6f}")
    
    print("\n" + "=" * 60)
    print("Compression completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()