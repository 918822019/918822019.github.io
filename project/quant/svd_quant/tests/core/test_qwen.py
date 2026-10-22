"""
测试脚本：使用Qwen3.5-0.8B模型进行SVD量化测试

本脚本用于测试SVD量化工具在真实Qwen模型上的效果。
通过加载Qwen3.5-0.8B模型的权重，进行SVD分解和量化。

测试内容：
1. 加载Qwen模型权重：从指定路径加载模型权重
2. 分析层形状：分析模型各层的形状和类型
3. SVD量化测试：对模型进行SVD分解和量化
4. 验证量化误差：计算反量化误差
5. 计算压缩比：计算原始模型与量化后模型的大小比例

使用示例：
    >>> python test_qwen.py
    
注意事项：
    1. 需要Qwen3.5-0.8B模型文件，请确保模型路径正确
    2. 测试只加载前2层进行测试，以节省时间和内存
    3. 建议在运行前确保已安装所有依赖
    4. 测试结果会保存到output目录下
"""

import os
import sys
import torch
import json
from pathlib import Path
# 尝试导入safetensors，如果不存在则使用torch.load
try:
    from safetensors.torch import load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("Warning: safetensors not installed, using torch.load instead")

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.main import SVDQuantizer, DecompositionConfig


def load_qwen_model(model_path: str, max_layers: int = None) -> dict:
    """
    加载Qwen模型的权重
    
    Args:
        model_path: 模型路径
        max_layers: 最大加载层数（用于测试）
        
    Returns:
        模型状态字典
    """
    print(f"Loading model from: {model_path}")
    
    # 读取索引文件
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
    
    # 获取权重映射
    weight_map = index_data['weight_map']
    
    # 按层分组
    layer_weights = {}
    for name, file_name in weight_map.items():
        # 只加载语言模型的权重，跳过视觉模型
        if 'model.language_model' in name:
            # 提取层号
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
    print(f"Loading safetensors file: {safetensors_path}")
    
    # 根据是否有safetensors库选择加载方式
    if HAS_SAFETENSORS:
        state_dict = load_file(safetensors_path)
    else:
        # 尝试使用torch.load
        try:
            state_dict = torch.load(safetensors_path, map_location='cpu')
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please install safetensors: pip install safetensors")
            return {}
    
    # 过滤出我们需要的层
    filtered_state_dict = {}
    for layer_num in sorted(layer_weights.keys()):
        for name in layer_weights[layer_num]:
            if name in state_dict:
                filtered_state_dict[name] = state_dict[name]
    
    print(f"Loaded {len(filtered_state_dict)} tensors from {len(layer_weights)} layers")
    
    return filtered_state_dict


def test_svd_quantization(state_dict: dict, output_dir: str):
    """
    测试SVD量化流程
    
    Args:
        state_dict: 模型状态字典
        output_dir: 输出目录
    """
    print("\n" + "="*60)
    print("开始SVD量化测试")
    print("="*60)
    
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
    
    # 保存为二进制格式
    output_path = os.path.join(output_dir, "qwen3.5-0.8b_quantized.bin")
    quantizer.save_to_binary_format(results, output_path)
    
    # 打印统计信息
    quantizer.print_statistics()
    
    # 验证量化误差
    print("\n" + "="*60)
    print("验证量化误差")
    print("="*60)
    
    total_error = 0
    decomposed_count = 0
    
    for layer_name, data in results.items():
        if data['is_decomposed']:
            decomposed_count += 1
            # 反量化U_merged和Vh
            U_dequant = quantizer.dequantize(data['quantized']['U_merged'])
            Vh_dequant = quantizer.dequantize(data['quantized']['Vh'])
            
            # 重构权重
            reconstructed = U_dequant @ Vh_dequant
            
            # 计算误差
            original = data['decomposed']['U_merged'] @ data['decomposed']['Vh']
            error = np.linalg.norm(original.cpu().numpy() - reconstructed) / np.linalg.norm(original.cpu().numpy())
            total_error += error
            
            print(f"Layer {layer_name}: dequantization error = {error:.6f}")
    
    if decomposed_count > 0:
        avg_error = total_error / decomposed_count
        print(f"\n平均反量化误差: {avg_error:.6f}")
    
    return results


def analyze_layer_shapes(state_dict: dict):
    """
    分析各层的形状
    
    Args:
        state_dict: 模型状态字典
    """
    print("\n" + "="*60)
    print("模型层形状分析")
    print("="*60)
    
    # 按类型分组
    layer_types = {}
    for name, tensor in state_dict.items():
        # 提取层类型
        if 'gate_proj' in name:
            layer_type = 'gate_proj'
        elif 'up_proj' in name:
            layer_type = 'up_proj'
        elif 'down_proj' in name:
            layer_type = 'down_proj'
        elif 'q_proj' in name:
            layer_type = 'q_proj'
        elif 'k_proj' in name:
            layer_type = 'k_proj'
        elif 'v_proj' in name:
            layer_type = 'v_proj'
        elif 'o_proj' in name:
            layer_type = 'o_proj'
        else:
            layer_type = 'other'
        
        if layer_type not in layer_types:
            layer_types[layer_type] = []
        layer_types[layer_type].append((name, tensor.shape))
    
    # 打印每种类型的形状
    for layer_type, layers in layer_types.items():
        print(f"\n{layer_type}:")
        if layers:
            # 显示第一个和最后一个
            print(f"  First: {layers[0][0]} -> {layers[0][1]}")
            if len(layers) > 1:
                print(f"  Last:  {layers[-1][0]} -> {layers[-1][1]}")
            print(f"  Total: {len(layers)} layers")


def main():
    """主函数"""
    import numpy as np
    
    # 模型路径
    model_path = r"D:\918822019.github.io\project\ContinuePretrain\Qwen3.5-0.8B"
    
    # 输出目录
    output_dir = r"D:\918822019.github.io\project\quant\svd_quant\output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型（只加载前2层进行测试）
    print("加载模型权重（测试模式：只加载前2层）...")
    state_dict = load_qwen_model(model_path, max_layers=2)
    
    # 分析层形状
    analyze_layer_shapes(state_dict)
    
    # 测试SVD量化
    results = test_svd_quantization(state_dict, output_dir)
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    print(f"量化后的模型已保存到: {os.path.join(output_dir, 'qwen3.5-0.8b_quantized.bin')}")
    
    # 计算压缩比
    original_size = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
    print(f"原始模型大小: {original_size / 1024 / 1024:.2f} MB")
    
    # 估算量化后的大小
    # INT8: 1字节/元素，INT4: 0.5字节/元素，FP16: 2字节/元素
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


if __name__ == "__main__":
    main()
