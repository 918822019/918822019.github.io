"""
SVD量化工具完整测试套件

本测试套件使用pytest框架，全面测试SVD量化工具的各个功能。
测试覆盖了单元测试、集成测试、边界测试和精度测试。

测试结构：
1. 测试夹具 (Fixtures)：提供测试所需的数据和配置
2. 测试配置类：测试DecompositionConfig类的功能
3. 测试层识别功能：测试MoE专家层、MLP层等识别功能
4. 测试SVD分解：测试SVD分解的正确性和精度
5. 测试量化功能：测试INT8和INT4量化的功能和精度
6. 测试二进制格式保存/加载：测试二进制格式的保存和加载
7. 测试完整流程：测试完整的模型处理流程
8. 测试边界条件：测试极端情况和边界条件
9. 测试精度验证：测试量化精度和误差界限
10. 测试统计功能：测试统计信息的收集和打印
11. 集成测试：测试完整的工作流程

使用示例：
    >>> # 运行所有测试
    >>> pytest test_svd_quant.py -v
    >>> 
    >>> # 运行特定测试类
    >>> pytest test_svd_quant.py::TestSVDDecomposition -v
    >>> 
    >>> # 运行特定测试方法
    >>> pytest test_svd_quant.py::TestSVDDecomposition::test_basic_decomposition -v
    
注意事项：
    1. 测试使用随机生成的数据，实际效果可能与真实模型不同
    2. 测试会生成临时文件，测试完成后会自动清理
    3. 建议在运行前确保已安装所有依赖
    4. 测试结果会以详细的报告形式输出
"""

import os
import sys
import pytest
import numpy as np
import torch
import tempfile
import struct
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.main import (
    SVDQuantizer, 
    DecompositionConfig, 
    QuantizedTensor, 
    QuantizationType
)


# ============================================================================
# 测试夹具 (Fixtures)
# ============================================================================

@pytest.fixture
def default_config():
    """默认配置"""
    return DecompositionConfig(
        gate_proj_rank=32,
        up_proj_rank=32,
        down_proj_rank=64,
        max_relative_error=1e-3,
        quant_group_size=128
    )


@pytest.fixture
def small_config():
    """小规模配置，用于快速测试"""
    return DecompositionConfig(
        gate_proj_rank=5,
        up_proj_rank=5,
        down_proj_rank=10,
        max_relative_error=0.1,
        quant_group_size=32
    )


@pytest.fixture
def quantizer(default_config):
    """默认量化器"""
    return SVDQuantizer(default_config)


@pytest.fixture
def small_quantizer(small_config):
    """小规模量化器"""
    return SVDQuantizer(small_config)


@pytest.fixture
def low_rank_matrix():
    """
    创建低秩矩阵用于测试
    这个矩阵应该能被SVD很好地近似
    """
    # 创建一个秩为10的矩阵
    m, n, rank = 100, 50, 10
    U = torch.randn(m, rank)
    V = torch.randn(rank, n)
    W = U @ V  # 秩为10的矩阵
    return W, rank


@pytest.fixture
def random_matrix():
    """随机矩阵（满秩）"""
    return torch.randn(64, 32)


@pytest.fixture
def sample_state_dict():
    """示例状态字典，模拟MoE模型"""
    return {
        # 专家层 - 应该被分解
        'model.layers.0.experts.0.gate_proj.weight': torch.randn(256, 128),
        'model.layers.0.experts.0.up_proj.weight': torch.randn(256, 128),
        'model.layers.0.experts.0.down_proj.weight': torch.randn(128, 256),
        'model.layers.0.experts.1.gate_proj.weight': torch.randn(256, 128),
        'model.layers.0.experts.1.up_proj.weight': torch.randn(256, 128),
        'model.layers.0.experts.1.down_proj.weight': torch.randn(128, 256),
        # MLP层 - 也应该被分解
        'model.layers.0.mlp.gate_proj.weight': torch.randn(256, 128),
        'model.layers.0.mlp.up_proj.weight': torch.randn(256, 128),
        'model.layers.0.mlp.down_proj.weight': torch.randn(128, 256),
        # 注意力层 - 不应该被分解
        'model.layers.0.self_attn.q_proj.weight': torch.randn(128, 128),
        'model.layers.0.self_attn.k_proj.weight': torch.randn(128, 128),
        'model.layers.0.self_attn.v_proj.weight': torch.randn(128, 128),
        'model.layers.0.self_attn.o_proj.weight': torch.randn(128, 128),
    }


# ============================================================================
# 测试配置类
# ============================================================================

class TestDecompositionConfig:
    """测试DecompositionConfig类"""
    
    def test_default_values(self):
        """测试默认值"""
        config = DecompositionConfig()
        assert config.gate_proj_rank == 32
        assert config.up_proj_rank == 32
        assert config.down_proj_rank == 64
        assert config.max_relative_error == 1e-3
        assert config.quant_group_size == 128
        assert config.enable_qat == False
        assert config.qat_samples == 1000
    
    def test_custom_values(self):
        """测试自定义值"""
        config = DecompositionConfig(
            gate_proj_rank=16,
            up_proj_rank=16,
            down_proj_rank=32,
            max_relative_error=0.01,
            quant_group_size=64,
            enable_qat=True,
            qat_samples=500
        )
        assert config.gate_proj_rank == 16
        assert config.up_proj_rank == 16
        assert config.down_proj_rank == 32
        assert config.max_relative_error == 0.01
        assert config.quant_group_size == 64
        assert config.enable_qat == True
        assert config.qat_samples == 500


# ============================================================================
# 测试层识别功能
# ============================================================================

class TestLayerIdentification:
    """测试层识别功能"""
    
    def test_expert_layer_detection(self, quantizer):
        """测试专家层检测"""
        # 包含expert关键字的层
        assert quantizer.is_moe_expert_layer('model.layers.0.experts.0.gate_proj.weight') == True
        assert quantizer.is_moe_expert_layer('model.layers.0.experts.1.up_proj.weight') == True
        assert quantizer.is_moe_expert_layer('model.layers.0.experts.0.down_proj.weight') == True
    
    def test_moe_layer_detection(self, quantizer):
        """测试MoE层检测"""
        assert quantizer.is_moe_expert_layer('model.layers.0.moe.gate_proj.weight') == True
        assert quantizer.is_moe_expert_layer('model.layers.0.moe_experts.up_proj.weight') == True
    
    def test_mlp_layer_detection(self, quantizer):
        """测试MLP层检测（非MoE模型）"""
        assert quantizer.is_moe_expert_layer('model.layers.0.mlp.gate_proj.weight') == True
        assert quantizer.is_moe_expert_layer('model.layers.0.mlp.up_proj.weight') == True
        assert quantizer.is_moe_expert_layer('model.layers.0.mlp.down_proj.weight') == True
    
    def test_attention_layer_exclusion(self, quantizer):
        """测试注意力层排除"""
        # 注意力层应该被排除
        assert quantizer.is_moe_expert_layer('model.layers.0.self_attn.q_proj.weight') == False
        assert quantizer.is_moe_expert_layer('model.layers.0.self_attn.k_proj.weight') == False
        assert quantizer.is_moe_expert_layer('model.layers.0.self_attn.v_proj.weight') == False
        assert quantizer.is_moe_expert_layer('model.layers.0.self_attn.o_proj.weight') == False
        assert quantizer.is_moe_expert_layer('model.layers.0.attention.q_proj.weight') == False
    
    def test_shared_expert_exclusion(self, quantizer):
        """测试共享专家排除"""
        assert quantizer.is_moe_expert_layer('model.layers.0.shared_expert.gate_proj.weight') == False
        assert quantizer.is_moe_expert_layer('model.layers.0.shared_experts.up_proj.weight') == False
    
    def test_router_exclusion(self, quantizer):
        """测试路由器排除"""
        assert quantizer.is_moe_expert_layer('model.layers.0.router.weight') == False
    
    def test_proj_type_detection(self, quantizer):
        """测试投影层类型检测"""
        assert quantizer.get_proj_type('model.layers.0.mlp.gate_proj.weight') == 'gate_proj'
        assert quantizer.get_proj_type('model.layers.0.mlp.up_proj.weight') == 'up_proj'
        assert quantizer.get_proj_type('model.layers.0.mlp.down_proj.weight') == 'down_proj'
        assert quantizer.get_proj_type('model.layers.0.self_attn.q_proj.weight') is None
    
    def test_rank_assignment(self, quantizer):
        """测试秩分配"""
        assert quantizer.get_rank_for_layer('model.layers.0.mlp.gate_proj.weight') == 32
        assert quantizer.get_rank_for_layer('model.layers.0.mlp.up_proj.weight') == 32
        assert quantizer.get_rank_for_layer('model.layers.0.mlp.down_proj.weight') == 64
        # 默认秩
        assert quantizer.get_rank_for_layer('model.layers.0.experts.0.linear.weight') == 32


# ============================================================================
# 测试SVD分解
# ============================================================================

class TestSVDDecomposition:
    """测试SVD分解功能"""
    
    def test_basic_decomposition(self, quantizer, random_matrix):
        """测试基本SVD分解"""
        rank = 10
        U_merged, Vh, original = quantizer.svd_decompose(random_matrix, rank)
        
        # 检查形状
        assert U_merged.shape == (random_matrix.shape[0], rank)
        assert Vh.shape == (rank, random_matrix.shape[1])
        assert original.shape == random_matrix.shape
    
    def test_low_rank_reconstruction(self, quantizer, low_rank_matrix):
        """测试低秩矩阵的重构精度"""
        W, true_rank = low_rank_matrix
        
        # 使用真实秩进行分解
        U_merged, Vh, original = quantizer.svd_decompose(W, true_rank)
        
        # 重构
        reconstructed = U_merged @ Vh
        
        # 计算误差
        error = quantizer.calculate_relative_error(original, reconstructed)
        
        # 低秩矩阵应该有很低的误差
        assert error < 1e-5, f"低秩矩阵重构误差过大: {error}"
    
    def test_high_rank_truncation(self, quantizer, random_matrix):
        """测试高秩矩阵的截断"""
        # 使用较小的秩
        rank = 5
        U_merged, Vh, original = quantizer.svd_decompose(random_matrix, rank)
        
        # 重构
        reconstructed = U_merged @ Vh
        
        # 计算误差
        error = quantizer.calculate_relative_error(original, reconstructed)
        
        # 截断后应该有一定误差
        assert error > 0, "截断后误差应该大于0"
        # 但误差应该在合理范围内
        assert error < 1.0, f"误差过大: {error}"
    
    def test_full_rank_reconstruction(self, quantizer, random_matrix):
        """测试满秩重构（使用最大可能的秩）"""
        max_rank = min(random_matrix.shape)
        U_merged, Vh, original = quantizer.svd_decompose(random_matrix, max_rank)
        
        # 重构
        reconstructed = U_merged @ Vh
        
        # 计算误差
        error = quantizer.calculate_relative_error(original, reconstructed)
        
        # 满秩应该有非常低的误差
        assert error < 1e-5, f"满秩重构误差过大: {error}"
    
    def test_different_shapes(self, quantizer):
        """测试不同形状的矩阵"""
        shapes = [(100, 50), (50, 100), (64, 64), (128, 32), (32, 128)]
        
        for shape in shapes:
            weight = torch.randn(shape)
            rank = min(shape) // 2
            
            U_merged, Vh, original = quantizer.svd_decompose(weight, rank)
            
            assert U_merged.shape == (shape[0], rank)
            assert Vh.shape == (rank, shape[1])
    
    def test_3d_tensor_handling(self, quantizer):
        """测试3D张量处理"""
        # 模拟一个3D权重（如某些模型的形状）
        weight_3d = torch.randn(32, 16, 8)
        rank = 4
        
        U_merged, Vh, original = quantizer.svd_decompose(weight_3d, rank)
        
        # 应该被展平为2D
        assert U_merged.shape[0] == 32
        assert U_merged.shape[1] == rank
        assert Vh.shape[0] == rank


# ============================================================================
# 测试量化功能
# ============================================================================

class TestQuantization:
    """测试量化功能"""
    
    def test_int8_quantization_shape(self, quantizer):
        """测试INT8量化形状"""
        tensor = torch.randn(100, 50)
        quantized = quantizer.symmetric_quantize_int8(tensor, group_size=32)
        
        # 检查形状
        assert quantized.data.shape[0] == (100 * 50 + 31) // 32  # 组数
        assert quantized.data.shape[1] == 32  # 每组大小
        assert quantized.quant_type == QuantizationType.INT8
    
    def test_int4_quantization_shape(self, quantizer):
        """测试INT4量化形状"""
        tensor = torch.randn(100, 50)
        quantized = quantizer.symmetric_quantize_int4(tensor, group_size=32)
        
        # 检查形状
        assert quantized.data.shape[1] == 16  # 每组32个元素打包为16个INT8
        assert quantized.quant_type == QuantizationType.INT4
    
    def test_int8_quantization_range(self, quantizer):
        """测试INT8量化范围"""
        tensor = torch.randn(100, 50)
        quantized = quantizer.symmetric_quantize_int8(tensor, group_size=32)
        
        # INT8范围: [-128, 127]
        assert quantized.data.min() >= -128
        assert quantized.data.max() <= 127
    
    def test_int4_quantization_range(self, quantizer):
        """测试INT4量化范围"""
        tensor = torch.randn(100, 50)
        quantized = quantizer.symmetric_quantize_int4(tensor, group_size=32)
        
        # 打包后的UINT8值
        assert quantized.data.min() >= 0
        assert quantized.data.max() <= 255
    
    def test_quantization_scales(self, quantizer):
        """测试量化缩放因子"""
        tensor = torch.randn(100, 50)
        group_size = 32
        quantized = quantizer.symmetric_quantize_int8(tensor, group_size=group_size)
        
        # 缩放因子应该是正数
        assert all(s > 0 for s in quantized.scales)
    
    def test_dequantization_roundtrip(self, quantizer):
        """测试量化-反量化往返"""
        tensor = torch.randn(64, 32)
        
        # INT8往返
        quantized_int8 = quantizer.symmetric_quantize_int8(tensor, group_size=32)
        dequantized_int8 = quantizer.dequantize(quantized_int8)
        
        # 检查形状
        assert dequantized_int8.shape == tensor.shape
        
        # 计算误差（应该有一定误差，但不会太大）
        error = np.linalg.norm(tensor.numpy() - dequantized_int8) / np.linalg.norm(tensor.numpy())
        assert error < 0.1, f"INT8往返误差过大: {error}"
    
    def test_different_group_sizes(self, quantizer):
        """测试不同组大小"""
        tensor = torch.randn(100, 50)
        group_sizes = [16, 32, 64, 128]
        
        for group_size in group_sizes:
            quantized = quantizer.symmetric_quantize_int8(tensor, group_size=group_size)
            
            # 检查缩放因子数量
            n_elements = tensor.numel()
            n_groups = (n_elements + group_size - 1) // group_size
            assert len(quantized.scales) == n_groups
    
    def test_zero_tensor_quantization(self, quantizer):
        """测试零张量量化"""
        tensor = torch.zeros(100, 50)
        quantized = quantizer.symmetric_quantize_int8(tensor, group_size=32)
        
        # 零张量量化后应该全为0
        assert quantized.data.sum() == 0
    
    def test_constant_tensor_quantization(self, quantizer):
        """测试常数张量量化"""
        tensor = torch.ones(100, 50) * 5.0
        quantized = quantizer.symmetric_quantize_int8(tensor, group_size=32)
        
        # 反量化后应该接近原始值
        dequantized = quantizer.dequantize(quantized)
        error = np.linalg.norm(tensor.numpy() - dequantized) / np.linalg.norm(tensor.numpy())
        assert error < 0.01, f"常数张量量化误差过大: {error}"


# ============================================================================
# 测试二进制格式保存/加载
# ============================================================================

class TestBinaryFormat:
    """测试二进制格式保存和加载"""
    
    def test_save_load_roundtrip(self, quantizer, tmp_path):
        """测试保存-加载往返"""
        # 创建测试数据
        state_dict = {
            'model.layers.0.mlp.gate_proj.weight': torch.randn(64, 32),
            'model.layers.0.mlp.up_proj.weight': torch.randn(64, 32),
            'model.layers.0.mlp.down_proj.weight': torch.randn(32, 64),
            'model.layers.0.self_attn.q_proj.weight': torch.randn(32, 32),
        }
        
        # 处理
        results = quantizer.process_model(state_dict)
        
        # 保存
        output_path = str(tmp_path / "test.bin")
        quantizer.save_to_binary_format(results, output_path)
        
        # 检查文件是否存在
        assert os.path.exists(output_path)
        
        # 检查文件大小
        file_size = os.path.getsize(output_path)
        assert file_size > 0
    
    def test_binary_format_structure(self, quantizer, tmp_path):
        """测试二进制格式结构"""
        state_dict = {
            'model.layers.0.mlp.gate_proj.weight': torch.randn(64, 32),
        }
        
        results = quantizer.process_model(state_dict)
        output_path = str(tmp_path / "test.bin")
        quantizer.save_to_binary_format(results, output_path)
        
        # 读取并验证文件头
        with open(output_path, 'rb') as f:
            magic = f.read(4)
            version = struct.unpack('I', f.read(4))[0]
            num_layers = struct.unpack('I', f.read(4))[0]
            
            assert magic == b'SVDQ'
            assert version == 1
            assert num_layers == len(results)
    
    def test_save_with_bias(self, quantizer, tmp_path):
        """测试保存带偏置的层"""
        # 创建带偏置的状态字典
        state_dict = {
            'model.layers.0.mlp.gate_proj.weight': torch.randn(64, 32),
            'model.layers.0.mlp.gate_proj.bias': torch.randn(64),
        }
        
        results = quantizer.process_model(state_dict)
        output_path = str(tmp_path / "test_with_bias.bin")
        quantizer.save_to_binary_format(results, output_path)
        
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0


# ============================================================================
# 测试完整流程
# ============================================================================

class TestFullPipeline:
    """测试完整处理流程"""
    
    def test_process_model(self, quantizer, sample_state_dict):
        """测试模型处理"""
        results = quantizer.process_model(sample_state_dict)
        
        # 检查所有层都被处理
        assert len(results) == len(sample_state_dict)
        
        # 检查统计信息
        assert quantizer.stats['total_layers'] == len(sample_state_dict)
        assert quantizer.stats['decomposed_layers'] > 0
        assert quantizer.stats['quantized_layers'] == len(sample_state_dict)
    
    def test_decomposed_layers_structure(self, quantizer, sample_state_dict):
        """测试分解层的结构"""
        results = quantizer.process_model(sample_state_dict)
        
        for layer_name, data in results.items():
            if quantizer.is_moe_expert_layer(layer_name):
                # 分解层应该有这些字段
                assert 'is_decomposed' in data
                assert 'decomposed' in data
                assert 'quantized' in data
                assert 'error' in data
                
                if data['is_decomposed']:
                    assert 'U_merged' in data['decomposed']
                    assert 'Vh' in data['decomposed']
                    assert 'rank' in data['decomposed']
                    assert 'U_merged' in data['quantized']
                    assert 'Vh' in data['quantized']
    
    def test_non_decomposed_layers_structure(self, quantizer, sample_state_dict):
        """测试非分解层的结构"""
        results = quantizer.process_model(sample_state_dict)
        
        for layer_name, data in results.items():
            if not quantizer.is_moe_expert_layer(layer_name):
                # 非分解层应该有这些字段
                assert 'is_decomposed' in data
                assert 'quantized' in data
                assert data['is_decomposed'] == False
                assert 'weight' in data['quantized']
    
    def test_error_tracking(self, quantizer, sample_state_dict):
        """测试误差追踪"""
        results = quantizer.process_model(sample_state_dict)
        
        # 检查误差记录
        assert len(quantizer.stats['errors']) > 0
        
        # 检查最大误差和平均误差
        if quantizer.stats['errors']:
            assert quantizer.stats['max_error'] == max(quantizer.stats['errors'])
            expected_avg = sum(quantizer.stats['errors']) / len(quantizer.stats['errors'])
            assert abs(quantizer.stats['avg_error'] - expected_avg) < 1e-6
    
    def test_memory_cleanup(self, quantizer, sample_state_dict):
        """测试内存清理"""
        import gc
        
        # 处理模型
        results = quantizer.process_model(sample_state_dict)
        
        # 检查原始权重是否被释放（通过检查引用）
        # 注意：这只是基本检查，实际的内存释放需要更复杂的测试
        assert results is not None
        assert len(results) > 0
    
    def test_compression_ratio(self, quantizer, sample_state_dict):
        """测试压缩比"""
        # 计算原始大小
        original_size = sum(t.numel() * t.element_size() for t in sample_state_dict.values())
        
        # 处理模型
        results = quantizer.process_model(sample_state_dict)
        
        # 估算量化后大小
        quantized_size = 0
        for layer_name, data in results.items():
            if data['is_decomposed']:
                U_size = data['quantized']['U_merged'].data.size
                Vh_size = data['quantized']['Vh'].data.size * 0.5
                quantized_size += U_size + Vh_size
            else:
                weight_size = data['quantized']['weight'].data.size * 0.5
                quantized_size += weight_size
        
        # 压缩比应该大于1
        compression_ratio = original_size / quantized_size if quantized_size > 0 else float('inf')
        assert compression_ratio > 1, f"压缩比应该大于1，实际: {compression_ratio}"


# ============================================================================
# 测试边界条件
# ============================================================================

class TestEdgeCases:
    """测试边界条件"""
    
    def test_minimum_size_matrix(self, quantizer):
        """测试最小尺寸矩阵"""
        # 非常小的矩阵
        weight = torch.randn(2, 2)
        rank = 1
        
        U_merged, Vh, original = quantizer.svd_decompose(weight, rank)
        
        assert U_merged.shape == (2, 1)
        assert Vh.shape == (1, 2)
    
    def test_rank_equals_min_dimension(self, quantizer):
        """测试秩等于最小维度"""
        weight = torch.randn(50, 100)
        rank = 50  # 等于最小维度
        
        U_merged, Vh, original = quantizer.svd_decompose(weight, rank)
        
        # 重构应该非常精确
        reconstructed = U_merged @ Vh
        error = quantizer.calculate_relative_error(original, reconstructed)
        assert error < 1e-5
    
    def test_rank_exceeds_min_dimension(self, quantizer):
        """测试秩超过最小维度"""
        weight = torch.randn(50, 100)
        rank = 100  # 超过最小维度
        
        # 应该自动限制为最小维度
        U_merged, Vh, original = quantizer.svd_decompose(weight, rank)
        
        # 实际秩应该被限制
        actual_rank = U_merged.shape[1]
        assert actual_rank <= min(weight.shape)
    
    def test_very_small_group_size(self, quantizer):
        """测试非常小的组大小"""
        tensor = torch.randn(100, 50)
        group_size = 1  # 每个元素单独量化
        
        quantized = quantizer.symmetric_quantize_int8(tensor, group_size=group_size)
        
        # 每个元素应该有自己的缩放因子
        assert len(quantized.scales) == tensor.numel()
    
    def test_very_large_group_size(self, quantizer):
        """测试非常大的组大小"""
        tensor = torch.randn(100, 50)
        group_size = 10000  # 比元素数量还大
        
        quantized = quantizer.symmetric_quantize_int8(tensor, group_size=group_size)
        
        # 应该只有1个组
        assert len(quantized.scales) == 1
    
    def test_single_element_tensor(self, quantizer):
        """测试单元素张量"""
        tensor = torch.tensor([[5.0]])
        quantized = quantizer.symmetric_quantize_int8(tensor, group_size=1)
        
        assert quantized.data.size == 1
        assert len(quantized.scales) == 1
    
    def test_empty_state_dict(self, quantizer):
        """测试空状态字典"""
        state_dict = {}
        results = quantizer.process_model(state_dict)
        
        assert len(results) == 0
        assert quantizer.stats['total_layers'] == 0
    
    def test_all_attention_layers(self, quantizer):
        """测试全是注意力层"""
        state_dict = {
            'model.layers.0.self_attn.q_proj.weight': torch.randn(32, 32),
            'model.layers.0.self_attn.k_proj.weight': torch.randn(32, 32),
            'model.layers.0.self_attn.v_proj.weight': torch.randn(32, 32),
            'model.layers.0.self_attn.o_proj.weight': torch.randn(32, 32),
        }
        
        results = quantizer.process_model(state_dict)
        
        # 所有层都不应该被分解
        for layer_name, data in results.items():
            assert data['is_decomposed'] == False
    
    def test_all_expert_layers(self, quantizer):
        """测试全是专家层"""
        state_dict = {
            'model.layers.0.experts.0.gate_proj.weight': torch.randn(64, 32),
            'model.layers.0.experts.0.up_proj.weight': torch.randn(64, 32),
            'model.layers.0.experts.0.down_proj.weight': torch.randn(32, 64),
        }
        
        results = quantizer.process_model(state_dict)
        
        # 所有层都应该被分解
        for layer_name, data in results.items():
            assert data['is_decomposed'] == True


# ============================================================================
# 测试精度验证
# ============================================================================

class TestPrecision:
    """测试精度验证"""
    
    def test_low_rank_matrix_precision(self, small_quantizer):
        """测试低秩矩阵精度"""
        # 创建一个秩为5的矩阵
        m, n, rank = 100, 50, 5
        U = torch.randn(m, rank)
        V = torch.randn(rank, n)
        W = U @ V
        
        # 使用真实秩进行分解
        U_merged, Vh, original = small_quantizer.svd_decompose(W, rank)
        
        # 重构
        reconstructed = U_merged @ Vh
        
        # 计算误差
        error = small_quantizer.calculate_relative_error(original, reconstructed)
        
        # 低秩矩阵应该有非常低的误差
        assert error < 1e-5, f"低秩矩阵误差过大: {error}"
    
    def test_rank_vs_error_tradeoff(self, quantizer):
        """测试秩与误差的权衡"""
        weight = torch.randn(100, 50)
        ranks = [5, 10, 20, 30, 40, 50]
        errors = []
        
        for rank in ranks:
            U_merged, Vh, original = quantizer.svd_decompose(weight, rank)
            reconstructed = U_merged @ Vh
            error = quantizer.calculate_relative_error(original, reconstructed)
            errors.append(error)
        
        # 误差应该随着秩的增加而减小
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i+1], \
                f"秩{ranks[i]}的误差({errors[i]})应该大于秩{ranks[i+1]}的误差({errors[i+1]})"
    
    def test_quantization_error_bounds(self, quantizer):
        """测试量化误差界限"""
        # 使用标准化数据
        tensor = torch.randn(100, 50)
        tensor = (tensor - tensor.mean()) / tensor.std()
        
        # INT8量化
        quantized = quantizer.symmetric_quantize_int8(tensor, group_size=32)
        dequantized = quantizer.dequantize(quantized)
        
        # 计算误差
        error = np.linalg.norm(tensor.numpy() - dequantized) / np.linalg.norm(tensor.numpy())
        
        # INT8量化误差应该在合理范围内
        assert error < 0.05, f"INT8量化误差过大: {error}"
    
    def test_symmetric_quantization_symmetry(self, quantizer):
        """测试对称量化对称性"""
        # 创建对称分布的数据
        tensor = torch.randn(100, 50)
        
        quantized = quantizer.symmetric_quantize_int8(tensor, group_size=32)
        
        # 检查缩放因子是否合理
        for scale in quantized.scales:
            assert scale > 0
            assert scale < float('inf')


# ============================================================================
# 测试统计功能
# ============================================================================

class TestStatistics:
    """测试统计功能"""
    
    def test_statistics_initialization(self, quantizer):
        """测试统计初始化"""
        assert quantizer.stats['total_layers'] == 0
        assert quantizer.stats['decomposed_layers'] == 0
        assert quantizer.stats['quantized_layers'] == 0
        assert quantizer.stats['max_error'] == 0.0
        assert quantizer.stats['avg_error'] == 0.0
        assert quantizer.stats['errors'] == []
    
    def test_statistics_update(self, quantizer, sample_state_dict):
        """测试统计更新"""
        quantizer.process_model(sample_state_dict)
        
        assert quantizer.stats['total_layers'] > 0
        assert quantizer.stats['quantized_layers'] > 0
        assert len(quantizer.stats['errors']) > 0
    
    def test_print_statistics(self, quantizer, sample_state_dict, capsys):
        """测试打印统计"""
        quantizer.process_model(sample_state_dict)
        quantizer.print_statistics()
        
        captured = capsys.readouterr()
        assert 'SVD量化处理统计' in captured.out
        assert '总层数' in captured.out
        assert '分解层数' in captured.out


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration:
    """集成测试"""
    
    def test_full_workflow(self, tmp_path):
        """测试完整工作流程"""
        # 1. 创建配置
        config = DecompositionConfig(
            gate_proj_rank=16,
            up_proj_rank=16,
            down_proj_rank=32,
            max_relative_error=0.1,
            quant_group_size=64
        )
        
        # 2. 创建量化器
        quantizer = SVDQuantizer(config)
        
        # 3. 创建测试数据
        state_dict = {
            'model.layers.0.experts.0.gate_proj.weight': torch.randn(128, 64),
            'model.layers.0.experts.0.up_proj.weight': torch.randn(128, 64),
            'model.layers.0.experts.0.down_proj.weight': torch.randn(64, 128),
            'model.layers.0.self_attn.q_proj.weight': torch.randn(64, 64),
        }
        
        # 4. 处理模型
        results = quantizer.process_model(state_dict)
        
        # 5. 验证结果
        assert len(results) == len(state_dict)
        
        # 6. 保存到文件
        output_path = str(tmp_path / "integration_test.bin")
        quantizer.save_to_binary_format(results, output_path)
        
        # 7. 验证文件
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        
        # 8. 打印统计
        quantizer.print_statistics()
    
    def test_real_model_like_data(self, tmp_path):
        """测试类似真实模型的数据"""
        # 模拟类似Qwen模型的结构
        config = DecompositionConfig(
            gate_proj_rank=32,
            up_proj_rank=32,
            down_proj_rank=64,
            max_relative_error=0.1,
            quant_group_size=128
        )
        
        quantizer = SVDQuantizer(config)
        
        # 创建类似真实模型的状态字典
        state_dict = {}
        for layer_num in range(2):
            # MLP层
            state_dict[f'model.layers.{layer_num}.mlp.gate_proj.weight'] = torch.randn(3584, 1024)
            state_dict[f'model.layers.{layer_num}.mlp.up_proj.weight'] = torch.randn(3584, 1024)
            state_dict[f'model.layers.{layer_num}.mlp.down_proj.weight'] = torch.randn(1024, 3584)
            
            # 注意力层
            state_dict[f'model.layers.{layer_num}.self_attn.q_proj.weight'] = torch.randn(1024, 1024)
            state_dict[f'model.layers.{layer_num}.self_attn.k_proj.weight'] = torch.randn(1024, 1024)
            state_dict[f'model.layers.{layer_num}.self_attn.v_proj.weight'] = torch.randn(1024, 1024)
            state_dict[f'model.layers.{layer_num}.self_attn.o_proj.weight'] = torch.randn(1024, 1024)
        
        # 处理
        results = quantizer.process_model(state_dict)
        
        # 验证
        assert len(results) == len(state_dict)
        
        # 计算压缩比
        original_size = sum(t.numel() * t.element_size() for t in state_dict.values())
        quantized_size = 0
        for layer_name, data in results.items():
            if data['is_decomposed']:
                U_size = data['quantized']['U_merged'].data.size
                Vh_size = data['quantized']['Vh'].data.size * 0.5
                quantized_size += U_size + Vh_size
            else:
                weight_size = data['quantized']['weight'].data.size * 0.5
                quantized_size += weight_size
        
        compression_ratio = original_size / quantized_size if quantized_size > 0 else float('inf')
        
        # 压缩比应该显著大于1
        assert compression_ratio > 5, f"压缩比过小: {compression_ratio}"
        
        # 保存
        output_path = str(tmp_path / "real_model_test.bin")
        quantizer.save_to_binary_format(results, output_path)
        
        print(f"\n真实模型测试结果:")
        print(f"  原始大小: {original_size / 1024 / 1024:.2f} MB")
        print(f"  量化后大小: {quantized_size / 1024 / 1024:.2f} MB")
        print(f"  压缩比: {compression_ratio:.2f}x")


# ============================================================================
# 运行测试
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
