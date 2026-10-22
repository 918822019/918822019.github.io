"""
SVD量化工具：实现MoE模型的低秩分解与量化处理

本模块实现了基于奇异值分解(SVD)的模型压缩技术，专门针对混合专家(MoE)模型架构。
通过结合低秩分解和混合精度量化，在保持模型精度的同时显著减少模型大小。

核心功能：
1. 模型低秩分解：对MoE模型的专家层进行SVD分解，使用非均匀秩分配策略
   - gate_proj层使用较低秩（默认32）以实现更高压缩比
   - down_proj层使用较高秩（默认64）以保持更好的精度
   - 注意力层和共享专家保持原始结构，不进行分解

2. 混合精度量化：对分解后的矩阵进行INT8/INT4量化
   - SVD分解后的U_merged矩阵使用INT8量化（精度更高）
   - SVD分解后的Vh矩阵使用INT4量化（压缩比更高）
   - 非分解层（注意力层）直接使用INT4量化

3. 自定义二进制格式保存：保存为便于快速加载的二进制格式
   - 支持魔数、版本号、层数等头部信息
   - 支持层名、原始形状、误差等元数据
   - 支持偏置项的FP16精度保存

技术原理：
   SVD分解将权重矩阵W分解为U @ diag(S) @ Vh，其中U和Vh是正交矩阵，
   S是奇异值矩阵。通过保留前k个最大的奇异值及其对应的向量，可以实现低秩近似：
   W ≈ U_k @ diag(S_k) @ Vh_k = (U_k @ diag(S_k)) @ Vh_k = U_merged @ Vh
   其中U_merged是合并了奇异值的左奇异向量矩阵。

使用示例：
   >>> from main import SVDQuantizer, DecompositionConfig
   >>> config = DecompositionConfig(gate_proj_rank=32, up_proj_rank=32, down_proj_rank=64)
   >>> quantizer = SVDQuantizer(config)
   >>> results = quantizer.process_model(model_state_dict)
   >>> quantizer.save_to_binary_format(results, "output.bin")

注意事项：
   1. 本工具主要针对MoE模型设计，但也可用于普通MLP层的压缩
   2. 分解秩的选择需要平衡压缩比和精度，建议根据具体模型调整
   3. 量化误差会随着组大小的增加而增加，但计算效率会提高
   4. 二进制格式需要配套的加载器才能正确读取

作者：AI助手
版本：1.0.0
"""

import os
import gc
import struct
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """
    量化类型枚举
    
    定义了支持的量化数据类型，用于指定量化后的数据表示方式。
    不同的量化类型在精度和存储空间之间有不同的权衡。
    
    枚举值：
        INT8: 8位整数量化，范围[-128, 127]
            - 存储空间：1字节/元素
            - 精度：中等，适合大多数权重矩阵
            - 用途：SVD分解后的U_merged矩阵
            
        INT4: 4位整数量化，范围[-8, 7]
            - 存储空间：0.5字节/元素（两个INT4打包到一个INT8中）
            - 精度：较低，适合对精度要求不高的权重
            - 用途：SVD分解后的Vh矩阵和注意力层权重
            
        FP16: 半精度浮点，范围约±65504
            - 存储空间：2字节/元素
            - 精度：高，用于偏置项等需要高精度的参数
            - 用途：偏置项（bias）
    
    使用示例：
        >>> from main import QuantizationType
        >>> if tensor.quant_type == QuantizationType.INT8:
        ...     print("这是INT8量化的张量")
        ... elif tensor.quant_type == QuantizationType.INT4:
        ...     print("这是INT4量化的张量")
    
    注意事项：
        1. INT4量化会将两个4位值打包到一个8位字节中，节省存储空间
        2. FP16量化主要用于需要保持高精度的参数，如偏置项
        3. 不同量化类型的反量化方法不同，需要使用对应的反量化函数
    """
    
    INT8 = "int8"
    """8位整数量化，范围[-128, 127]，存储空间1字节/元素"""
    
    INT4 = "int4"
    """4位整数量化，范围[-8, 7]，存储空间0.5字节/元素（打包存储）"""
    
    FP16 = "fp16"
    """半精度浮点，范围约±65504，存储空间2字节/元素"""


@dataclass
class DecompositionConfig:
    """
    分解配置数据类
    
    用于配置SVD分解和量化过程的参数。不同的配置会影响压缩比、精度和计算效率。
    
    属性说明：
        gate_proj_rank (int): gate_proj层的分解秩，默认32
            - 控制gate_proj权重矩阵的低秩近似精度
            - 较小的值（如16）会获得更高的压缩比，但精度会降低
            - 较大的值（如64）会保持更好的精度，但压缩比会降低
            - 建议范围：16-64
            
        up_proj_rank (int): up_proj层的分解秩，默认32
            - 控制up_proj权重矩阵的低秩近似精度
            - 通常与gate_proj_rank保持一致
            - 建议范围：16-64
            
        down_proj_rank (int): down_proj层的分解秩，默认64
            - 控制down_proj权重矩阵的低秩近似精度
            - 通常比gate_proj和up_proj的秩更大，因为down_proj矩阵形状不同
            - 建议范围：32-128
            
        max_relative_error (float): 最大允许相对误差，默认1e-3
            - 用于监控分解质量，当误差超过此阈值时会发出警告
            - 不会阻止分解过程，仅作为监控指标
            - 建议范围：1e-4到1e-2
            
        quant_group_size (int): 量化组大小，默认128
            - 每组元素共享一个缩放因子
            - 较小的值（如32）会提高精度，但增加存储开销
            - 较大的值（如256）会减少存储开销，但可能降低精度
            - 必须是2的幂次，建议范围：32-256
            
        enable_qat (bool): 是否启用量化感知微调，默认False
            - 启用后会在量化前进行微调以减少量化误差
            - 需要额外的训练数据和计算资源
            - 目前为预留功能，实际实现需要扩展
            
        qat_samples (int): 量化感知微调使用的样本数量，默认1000
            - 仅在enable_qat=True时有效
            - 样本数量越多，微调效果越好，但计算时间越长
            
    使用示例：
        >>> from main import DecompositionConfig
        >>> 
        >>> # 使用默认配置
        >>> config = DecompositionConfig()
        >>> 
        >>> # 自定义配置
        >>> config = DecompositionConfig(
        ...     gate_proj_rank=16,
        ...     up_proj_rank=16,
        ...     down_proj_rank=32,
        ...     max_relative_error=0.01,
        ...     quant_group_size=64
        ... )
        >>> 
        >>> # 查看配置
        >>> print(f"gate_proj秩: {config.gate_proj_rank}")
        >>> print(f"量化组大小: {config.quant_group_size}")
    
    注意事项：
        1. 秩的选择需要平衡压缩比和精度，建议通过实验确定最佳值
        2. 量化组大小会影响量化误差和计算效率，需要根据硬件特性选择
        3. 对于精度敏感的应用，建议使用较大的秩和较小的量化组大小
        4. 对于存储敏感的应用，可以使用较小的秩和较大的量化组大小
    """
    
    # 各层的秩分配
    gate_proj_rank: int = 32
    """gate_proj层的分解秩，默认32，控制门控投影的低秩近似精度"""
    
    up_proj_rank: int = 32
    """up_proj层的分解秩，默认32，控制上投影的低秩近似精度"""
    
    down_proj_rank: int = 64
    """down_proj层的分解秩，默认64，控制下投影的低秩近似精度"""
    
    # 精度阈值
    max_relative_error: float = 1e-3
    """最大允许相对误差，默认1e-3，用于监控分解质量"""
    
    # 量化参数
    quant_group_size: int = 128
    """量化组大小，默认128，每组元素共享一个缩放因子"""
    
    # 是否进行量化感知微调
    enable_qat: bool = False
    """是否启用量化感知微调，默认False，启用后可减少量化误差"""
    
    qat_samples: int = 1000
    """量化感知微调使用的样本数量，默认1000"""


@dataclass 
class QuantizedTensor:
    """
    量化后的张量数据类
    
    存储量化后的张量数据，包括量化后的整数数据、缩放因子等元信息。
    支持INT8和INT4两种量化格式，以及对应的反量化操作。
    
    属性说明：
        data (np.ndarray): 量化后的整数数据
            - INT8量化：形状为(n_groups, group_size)，数据类型为np.int8
            - INT4量化：形状为(n_groups, group_size//2)，数据类型为np.uint8
              两个INT4值打包到一个UINT8中，高4位和低4位各存储一个值
            
        scales (np.ndarray): 缩放因子数组
            - 形状为(n_groups,)
            - 每个元素对应一个量化组的缩放因子
            - 缩放因子 = 组内最大绝对值 / 量化范围最大值
            - 用于反量化时恢复原始浮点值
            
        quant_type (QuantizationType): 量化类型
            - QuantizationType.INT8: 8位整数量化
            - QuantizationType.INT4: 4位整数量化
            - QuantizationType.FP16: 半精度浮点（目前未使用）
            
        original_shape (Tuple[int, ...]): 原始张量的形状
            - 用于反量化后重塑为原始形状
            - 保持与量化前完全一致的维度信息
            
        group_size (int): 量化组大小，默认128
            - 每组元素共享一个缩放因子
            - 必须是2的幂次
            - 较小的值会提高精度但增加存储开销
            
    使用示例：
        >>> from main import QuantizedTensor, QuantizationType
        >>> 
        >>> # 创建量化张量
        >>> quantized = QuantizedTensor(
        ...     data=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8),
        ...     scales=np.array([0.1, 0.2]),
        ...     quant_type=QuantizationType.INT8,
        ...     original_shape=(2, 3),
        ...     group_size=3
        ... )
        >>> 
        >>> # 访问量化信息
        >>> print(f"量化类型: {quantized.quant_type.value}")
        >>> print(f"原始形状: {quantized.original_shape}")
        >>> print(f"缩放因子: {quantized.scales}")
    
    注意事项：
        1. INT4量化会将两个4位值打包到一个8位字节中，需要使用专门的解包函数
        2. 缩放因子是反量化时的关键信息，必须正确保存和加载
        3. original_shape用于反量化后重塑，必须与原始张量形状完全一致
        4. group_size会影响量化精度和存储效率，需要根据具体应用场景选择
    """
    
    data: np.ndarray
    """量化后的整数数据，INT8为np.int8类型，INT4为打包的np.uint8类型"""
    
    scales: np.ndarray
    """缩放因子数组，形状为(n_groups,)，用于反量化恢复浮点值"""
    
    quant_type: QuantizationType
    """量化类型，决定数据的解释方式和反量化方法"""
    
    original_shape: Tuple[int, ...]
    """原始张量形状，用于反量化后重塑为原始维度"""
    
    group_size: int = 128
    """量化组大小，默认128，每组元素共享一个缩放因子"""


class SVDQuantizer:
    """
    SVD量化器主类
    
    实现基于奇异值分解(SVD)的模型压缩和量化功能。专门针对混合专家(MoE)模型架构，
    通过结合低秩分解和混合精度量化，在保持模型精度的同时显著减少模型大小。
    
    核心功能：
        1. 模型低秩分解：对MoE模型的专家层进行SVD分解，使用非均匀秩分配策略
        2. 混合精度量化：对分解后的矩阵进行INT8/INT4量化
        3. 自定义二进制格式保存：保存为便于快速加载的二进制格式
        4. 量化质量评估：计算压缩比、MSE、MAE等指标
    
    使用示例：
        >>> from main import SVDQuantizer, DecompositionConfig
        >>> 
        >>> # 创建配置
        >>> config = DecompositionConfig(
        ...     gate_proj_rank=32,
        ...     up_proj_rank=32,
        ...     down_proj_rank=64,
        ...     quant_group_size=128
        ... )
        >>> 
        >>> # 创建量化器
        >>> quantizer = SVDQuantizer(config)
        >>> 
        >>> # 处理模型
        >>> results = quantizer.process_model(model_state_dict)
        >>> 
        >>> # 计算压缩比
        >>> compression_ratio = quantizer.calculate_compression_ratio(model_state_dict, results)
        >>> 
        >>> # 保存为二进制格式
        >>> quantizer.save_to_binary_format(results, "output.bin")
        >>> 
        >>> # 打印统计信息
        >>> quantizer.print_statistics()
    
    注意事项：
        1. 本量化器主要针对MoE模型设计，但也可用于普通MLP层的压缩
        2. 分解秩的选择需要平衡压缩比和精度，建议根据具体模型调整
        3. 量化误差会随着组大小的增加而增加，但计算效率会提高
        4. 二进制格式需要配套的加载器才能正确读取
        5. 处理大模型时需要注意内存使用，建议分批次处理
    """
    
    def __init__(self, config: Optional[DecompositionConfig] = None):
        """
        初始化SVD量化器
        
        初始化量化器实例，设置分解配置和统计信息存储结构。
        如果未提供配置，将使用默认的DecompositionConfig配置。
        
        Args:
            config (Optional[DecompositionConfig]): 分解配置对象
                - 包含各层的秩分配、量化参数等配置信息
                - 如果为None，则使用默认配置：
                  gate_proj_rank=32, up_proj_rank=32, down_proj_rank=64
                  max_relative_error=1e-3, quant_group_size=128
                  enable_qat=False, qat_samples=1000
                
        Returns:
            None
            
        Raises:
            无异常抛出
            
        使用示例：
            >>> from main import SVDQuantizer, DecompositionConfig
            >>> 
            >>> # 使用默认配置
            >>> quantizer1 = SVDQuantizer()
            >>> 
            >>> # 使用自定义配置
            >>> config = DecompositionConfig(gate_proj_rank=16, up_proj_rank=16, down_proj_rank=32)
            >>> quantizer2 = SVDQuantizer(config)
            >>> 
            >>> # 访问配置信息
            >>> print(f"gate_proj秩: {quantizer2.config.gate_proj_rank}")
            >>> print(f"量化组大小: {quantizer2.config.quant_group_size}")
            
        注意事项：
            1. 配置对象在初始化后会被保存为实例属性，后续修改不会影响已创建的量化器
            2. 统计信息会在处理过程中自动更新，用于后续的分析和报告
            3. 量化器实例可以重复使用，每次处理前会重置统计信息
        """
        self.config = config or DecompositionConfig()
        """分解配置对象，包含各层的秩分配、量化参数等配置信息"""
        
        self.stats = {
            'total_layers': 0,
            'decomposed_layers': 0,
            'quantized_layers': 0,
            'max_error': 0.0,
            'avg_error': 0.0,
            'errors': []
        }
        """统计信息字典，记录处理过程中的各种指标"""
    
    def is_moe_expert_layer(self, layer_name: str) -> bool:
        """
        判断是否为MoE模型的专家层
        
        通过分析层名称中的关键字来判断该层是否为MoE模型的专家层。
        专家层会进行SVD分解以实现模型压缩，而非专家层（如注意力层）
        则直接进行INT4量化。
        
        判断逻辑：
            1. 首先排除注意力层和共享专家层（不进行分解）
            2. 然后检查是否包含专家层关键字（进行分解）
            3. 最后检查是否为MLP层（对于非MoE模型也进行分解）
            
        排除的层类型（不进行分解）：
            - 注意力层：包含 'attention' 或 'self_attn' 关键字
            - 共享专家：包含 'shared_expert' 关键字
            - 路由器：包含 'router' 关键字
            
        专家层关键字（进行分解）：
            - 'expert'：MoE模型的专家层
            - 'moe'：混合专家层
            - 'ffn'：前馈网络层
            
        MLP层关键字（对于非MoE模型也进行分解）：
            - 'mlp'：多层感知机层
            - 'gate_proj'：门控投影层
            - 'up_proj'：上投影层
            - 'down_proj'：下投影层
            
        Args:
            layer_name (str): 层名称，例如：
                - 'model.layers.0.experts.0.gate_proj.weight'
                - 'model.layers.0.self_attn.q_proj.weight'
                - 'model.layers.0.mlp.gate_proj.weight'
                
        Returns:
            bool: 是否为专家层
                - True: 该层应该进行SVD分解
                - False: 该层应该直接进行INT4量化
                
        Raises:
            无异常抛出
            
        使用示例：
            >>> quantizer = SVDQuantizer()
            >>> 
            >>> # MoE专家层
            >>> quantizer.is_moe_expert_layer('model.layers.0.experts.0.gate_proj.weight')
            True
            >>> 
            >>> # 注意力层
            >>> quantizer.is_moe_expert_layer('model.layers.0.self_attn.q_proj.weight')
            False
            >>> 
            >>> # 共享专家
            >>> quantizer.is_moe_expert_layer('model.layers.0.shared_expert.gate_proj.weight')
            False
            >>> 
            >>> # 普通MLP层（非MoE模型）
            >>> quantizer.is_moe_expert_layer('model.layers.0.mlp.gate_proj.weight')
            True
            
        注意事项：
            1. 判断是基于层名称的关键字匹配，需要确保命名规范
            2. 注意力层和共享专家层不会进行分解，直接进行INT4量化
            3. 对于非MoE模型，所有MLP层都会被视为可分解层
            4. 路由器层不会进行分解，保持原始精度
        """
        # 排除注意力层和共享专家
        exclude_keywords = ['attention', 'self_attn', 'shared_expert', 'router']
        for keyword in exclude_keywords:
            if keyword in layer_name.lower():
                return False
        
        # 包含专家层关键字
        expert_keywords = ['expert', 'moe', 'ffn']
        for keyword in expert_keywords:
            if keyword in layer_name.lower():
                return True
        
        # 对于非MoE模型，将所有MLP层视为可分解层
        mlp_keywords = ['mlp', 'gate_proj', 'up_proj', 'down_proj']
        for keyword in mlp_keywords:
            if keyword in layer_name.lower():
                return True
        
        return False
    
    def get_proj_type(self, layer_name: str) -> Optional[str]:
        """
        获取投影层类型
        
        通过分析层名称中的关键字来判断该层属于哪种投影层类型。
        不同的投影层类型会使用不同的分解秩，以实现最优的压缩效果。
        
        投影层类型说明：
            - gate_proj: 门控投影层，用于控制信息流的门控机制
              使用较小的秩（默认32）以实现更高压缩比
              
            - up_proj: 上投影层，用于将输入投影到更高维度空间
              使用较小的秩（默认32）以实现更高压缩比
              
            - down_proj: 下投影层，用于将高维特征投影回原始维度
              使用较大的秩（默认64）以保持更好的精度
              
        Args:
            layer_name (str): 层名称，例如：
                - 'model.layers.0.mlp.gate_proj.weight'
                - 'model.layers.0.experts.0.up_proj.weight'
                - 'model.layers.0.mlp.down_proj.weight'
                - 'model.layers.0.self_attn.q_proj.weight'
                
        Returns:
            Optional[str]: 投影层类型
                - 'gate_proj': 门控投影层
                - 'up_proj': 上投影层
                - 'down_proj': 下投影层
                - None: 非投影层（如注意力层的q_proj、k_proj等）
                
        Raises:
            无异常抛出
            
        使用示例：
            >>> quantizer = SVDQuantizer()
            >>> 
            >>> # 门控投影层
            >>> quantizer.get_proj_type('model.layers.0.mlp.gate_proj.weight')
            'gate_proj'
            >>> 
            >>> # 上投影层
            >>> quantizer.get_proj_type('model.layers.0.experts.0.up_proj.weight')
            'up_proj'
            >>> 
            >>> # 下投影层
            >>> quantizer.get_proj_type('model.layers.0.mlp.down_proj.weight')
            'down_proj'
            >>> 
            >>> # 注意力层的投影（不是MLP投影）
            >>> quantizer.get_proj_type('model.layers.0.self_attn.q_proj.weight')
            None
            
        注意事项：
            1. 判断是基于层名称的关键字匹配，需要确保命名规范
            2. 注意力层的q_proj、k_proj、v_proj、o_proj不属于MLP投影层
            3. 投影层类型决定了使用哪种分解秩进行SVD分解
            4. 对于非MLP层，返回None，使用默认的分解秩
        """
        if 'gate_proj' in layer_name:
            return 'gate_proj'
        elif 'up_proj' in layer_name:
            return 'up_proj'
        elif 'down_proj' in layer_name:
            return 'down_proj'
        return None
    
    def get_rank_for_layer(self, layer_name: str) -> int:
        """
        根据层名称获取对应的分解秩
        
        根据层名称判断其投影层类型，然后返回对应的分解秩。
        不同的投影层类型使用不同的分解秩，以实现最优的压缩效果。
        
        秩分配策略：
            - gate_proj: 使用gate_proj_rank配置（默认32）
            - up_proj: 使用up_proj_rank配置（默认32）
            - down_proj: 使用down_proj_rank配置（默认64）
            - 其他类型: 使用默认秩32
            
        秩选择原则：
            1. 较小的秩（如16-32）会获得更高的压缩比，但精度会降低
            2. 较大的秩（如64-128）会保持更好的精度，但压缩比会降低
            3. down_proj通常需要较大的秩，因为其矩阵形状不同
            4. gate_proj和up_proj可以使用较小的秩，因为它们对精度要求相对较低
            
        Args:
            layer_name (str): 层名称，例如：
                - 'model.layers.0.mlp.gate_proj.weight'
                - 'model.layers.0.experts.0.up_proj.weight'
                - 'model.layers.0.mlp.down_proj.weight'
                - 'model.layers.0.experts.0.linear.weight'
                
        Returns:
            int: 该层应使用的分解秩
                - 对于gate_proj: 返回config.gate_proj_rank（默认32）
                - 对于up_proj: 返回config.up_proj_rank（默认32）
                - 对于down_proj: 返回config.down_proj_rank（默认64）
                - 对于其他类型: 返回默认秩32
                
        Raises:
            无异常抛出
            
        使用示例：
            >>> quantizer = SVDQuantizer()
            >>> 
            >>> # gate_proj层
            >>> quantizer.get_rank_for_layer('model.layers.0.mlp.gate_proj.weight')
            32
            >>> 
            >>> # up_proj层
            >>> quantizer.get_rank_for_layer('model.layers.0.experts.0.up_proj.weight')
            32
            >>> 
            >>> # down_proj层
            >>> quantizer.get_rank_for_layer('model.layers.0.mlp.down_proj.weight')
            64
            >>> 
            >>> # 其他层（如专家层的线性层）
            >>> quantizer.get_rank_for_layer('model.layers.0.experts.0.linear.weight')
            32
            
        注意事项：
            1. 秩的分配基于层名称的关键字匹配，需要确保命名规范
            2. 不同投影层类型使用不同的默认秩，以实现最优压缩效果
            3. 默认秩32适用于大多数情况，但建议根据具体模型调整
            4. 秩的选择需要平衡压缩比和精度，建议通过实验确定最佳值
        """
        proj_type = self.get_proj_type(layer_name)
        
        rank_mapping = {
            'gate_proj': self.config.gate_proj_rank,
            'up_proj': self.config.up_proj_rank,
            'down_proj': self.config.down_proj_rank
        }
        
        return rank_mapping.get(proj_type, 32)  # 默认秩为32
    
    def svd_decompose(self, weight: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对权重矩阵进行SVD分解
        
        使用奇异值分解(SVD)将权重矩阵分解为低秩近似形式。
        SVD分解将矩阵W分解为U @ diag(S) @ Vh，其中U和Vh是正交矩阵，
        S是奇异值矩阵。通过保留前k个最大的奇异值及其对应的向量，
        可以实现低秩近似。
        
        分解公式：
            原始分解：W = U @ diag(S) @ Vh
            低秩近似：W ≈ U_k @ diag(S_k) @ Vh_k
            合并形式：W ≈ (U_k @ diag(S_k)) @ Vh_k = U_merged @ Vh
            
        其中：
            - U_k: 前k个左奇异向量，形状为(m, k)
            - S_k: 前k个奇异值，形状为(k,)
            - Vh_k: 前k个右奇异向量，形状为(k, n)
            - U_merged: 合并了奇异值的左奇异向量，形状为(m, k)
            
        数学原理：
            1. 奇异值分解将任意矩阵分解为三个矩阵的乘积
            2. 奇异值按从大到小排列，前k个奇异值包含矩阵的主要信息
            3. 通过保留前k个奇异值，可以实现数据压缩和降维
            4. 重构误差与舍弃的奇异值大小成正比
            
        Args:
            weight (torch.Tensor): 权重矩阵
                - 支持2D和3D张量（3D会被自动展平为2D）
                - 形状通常为(out_features, in_features)或(in_features, out_features)
                - 数据类型应为torch.float32或torch.float16
                
            rank (int): 保留的奇异值数量
                - 必须为正整数
                - 不能超过min(m, n)，其中m和n是矩阵的维度
                - 如果超过，会自动限制为min(m, n)
                - 较小的值会获得更高的压缩比，但精度会降低
                - 较大的值会保持更好的精度，但压缩比会降低
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 包含三个元素的元组
                - U_merged (torch.Tensor): 合并了奇异值的左奇异向量矩阵
                  形状为(m, rank)，包含前rank个左奇异向量与对应奇异值的乘积
                  
                - Vh (torch.Tensor): 右奇异向量矩阵的转置
                  形状为(rank, n)，包含前rank个右奇异向量
                  
                - original_weight (torch.Tensor): 原始权重矩阵
                  用于后续的误差计算和验证
                  
        Raises:
            无异常抛出，但可能因内存不足等原因导致计算失败
            
        使用示例：
            >>> quantizer = SVDQuantizer()
            >>> 
            >>> # 创建权重矩阵
            >>> weight = torch.randn(100, 50)
            >>> 
            >>> # 进行SVD分解，保留前10个奇异值
            >>> U_merged, Vh, original = quantizer.svd_decompose(weight, rank=10)
            >>> 
            >>> # 检查输出形状
            >>> print(f"U_merged形状: {U_merged.shape}")  # (100, 10)
            >>> print(f"Vh形状: {Vh.shape}")  # (10, 50)
            >>> print(f"原始形状: {original.shape}")  # (100, 50)
            >>> 
            >>> # 重构权重
            >>> reconstructed = U_merged @ Vh
            >>> 
            >>> # 计算重构误差
            >>> error = torch.norm(original - reconstructed) / torch.norm(original)
            >>> print(f"相对误差: {error.item():.6f}")
            
        注意事项：
            1. 3D张量会被自动展平为2D，展平后的形状为(第一维, 其余维乘积)
            2. rank不能超过矩阵的最小维度，否则会自动限制
            3. 分解过程会消耗较多内存，建议对大矩阵分批处理
            4. 重构误差与舍弃的奇异值大小成正比，可以通过调整rank来控制精度
            5. U_merged已经合并了奇异值，可以直接与Vh相乘进行重构
        """
        # 确保权重是2D矩阵
        if weight.dim() > 2:
            weight = weight.view(weight.size(0), -1)
        
        # 执行SVD分解
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        
        # 只保留前rank个奇异值
        U_k = U[:, :rank]
        S_k = S[:rank]
        Vh_k = Vh[:rank, :]
        
        # 合并为 U_merged = U @ diag(S)
        U_merged = U_k @ torch.diag(S_k)
        
        return U_merged, Vh_k, weight
    
    def calculate_relative_error(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """
        计算相对误差
        
        计算原始权重矩阵与重构权重矩阵之间的相对误差。
        相对误差是衡量重构质量的重要指标，误差越小表示重构质量越好。
        
        计算公式：
            相对误差 = ||original - reconstructed|| / ||original||
            其中||·||表示Frobenius范数（矩阵的欧几里得范数）
            
        误差范围：
            - 0: 完全重构，无误差
            - (0, 0.01): 非常小的误差，重构质量优秀
            - [0.01, 0.1): 较小的误差，重构质量良好
            - [0.1, 0.5): 中等误差，重构质量一般
            - >=0.5: 较大误差，重构质量较差
            
        Args:
            original (torch.Tensor): 原始权重矩阵
                - 形状应与reconstructed相同
                - 数据类型应为torch.float32或torch.float16
                
            reconstructed (torch.Tensor): 重构的权重矩阵
                - 形状应与original相同
                - 通常是SVD分解后的重构结果
                
        Returns:
            float: 相对误差值
                - 范围：[0, +∞)
                - 值越小表示重构质量越好
                - 0表示完全重构，无误差
                
        Raises:
            无异常抛出，但如果original是零向量会导致除零错误
            
        使用示例：
            >>> quantizer = SVDQuantizer()
            >>> 
            >>> # 创建原始权重
            >>> original = torch.randn(100, 50)
            >>> 
            >>> # 进行SVD分解
            >>> U_merged, Vh, _ = quantizer.svd_decompose(original, rank=10)
            >>> 
            >>> # 重构权重
            >>> reconstructed = U_merged @ Vh
            >>> 
            >>> # 计算相对误差
            >>> error = quantizer.calculate_relative_error(original, reconstructed)
            >>> print(f"相对误差: {error:.6f}")
            
        注意事项：
            1. 相对误差基于Frobenius范数，对矩阵中的所有元素都敏感
            2. 如果原始矩阵是零矩阵，会导致除零错误，需要特殊处理
            3. 相对误差越小，表示SVD分解的近似效果越好
            4. 可以通过调整分解秩来控制相对误差的大小
            5. 相对误差是量化前SVD分解的误差，量化还会引入额外的误差
        """
        error = torch.norm(original - reconstructed) / torch.norm(original)
        return error.item()
    
    def symmetric_quantize_int8(self, tensor: torch.Tensor, group_size: int = 128) -> QuantizedTensor:
        """
        对称INT8量化
        
        将浮点张量量化为8位整数（INT8）格式。使用对称量化方法，
        即量化范围关于零点对称：[-128, 127]。
        
        量化原理：
            1. 将输入张量展平并分组，每组group_size个元素
            2. 对每组计算缩放因子：scale = max(|group|) / 127
            3. 量化公式：quantized = round(group / scale)
            4. 裁剪到INT8范围：[-128, 127]
            
        对称量化特点：
            - 零点固定为0，简化了计算
            - 缩放因子是正数，保持数据的对称性
            - 量化误差相对较小，适合大多数权重矩阵
            - 存储空间：1字节/元素
            
        Args:
            tensor (torch.Tensor): 输入张量
                - 支持任意形状的张量
                - 数据类型应为torch.float32或torch.float16
                - 会自动转换为numpy数组进行处理
                
            group_size (int): 每组元素数量，默认128
                - 必须为正整数
                - 较小的值会提高精度但增加存储开销
                - 较大的值会减少存储开销但可能降低精度
                - 建议范围：32-256，通常是2的幂次
                
        Returns:
            QuantizedTensor: 量化后的张量对象
                - data: INT8量化数据，形状为(n_groups, group_size)
                - scales: 缩放因子数组，形状为(n_groups,)
                - quant_type: QuantizationType.INT8
                - original_shape: 原始张量形状
                - group_size: 量化组大小
                
        Raises:
            无异常抛出，但可能因内存不足等原因导致计算失败
            
        使用示例：
            >>> quantizer = SVDQuantizer()
            >>> 
            >>> # 创建测试张量
            >>> tensor = torch.randn(100, 50)
            >>> 
            >>> # 进行INT8量化
            >>> quantized = quantizer.symmetric_quantize_int8(tensor, group_size=32)
            >>> 
            >>> # 检查量化结果
            >>> print(f"量化类型: {quantized.quant_type.value}")
            >>> print(f"数据形状: {quantized.data.shape}")
            >>> print(f"缩放因子数量: {len(quantized.scales)}")
            >>> print(f"数据范围: [{quantized.data.min()}, {quantized.data.max()}]")
            
        注意事项：
            1. 输入张量会被自动转换为numpy数组，会丢失梯度信息
            2. 分组时会自动填充到group_size的整数倍，可能导致少量内存浪费
            3. 缩放因子是每组一个，用于反量化时恢复浮点值
            4. INT8量化的范围是[-128, 127]，超出范围的值会被裁剪
            5. 量化过程是确定性的，相同输入总是产生相同输出
        """
        # 转换为numpy
        data = tensor.cpu().float().numpy()
        original_shape = data.shape
        
        # 展平为1D便于分组
        flat_data = data.flatten()
        
        # 计算需要多少组
        n_elements = len(flat_data)
        n_groups = (n_elements + group_size - 1) // group_size
        
        # 填充到group_size的整数倍
        padded_size = n_groups * group_size
        padded_data = np.zeros(padded_size, dtype=np.float32)
        padded_data[:n_elements] = flat_data
        
        # 重塑为 (n_groups, group_size)
        grouped_data = padded_data.reshape(n_groups, group_size)
        
        # 计算每组的最大绝对值作为缩放因子
        max_abs = np.max(np.abs(grouped_data), axis=1, keepdims=True)
        scales = max_abs / 127.0  # INT8范围: [-128, 127]
        
        # 避免除零
        scales = np.where(scales == 0, 1.0, scales)
        
        # 量化
        quantized = np.clip(np.round(grouped_data / scales), -128, 127).astype(np.int8)
        
        return QuantizedTensor(
            data=quantized,
            scales=scales.flatten(),
            quant_type=QuantizationType.INT8,
            original_shape=original_shape,
            group_size=group_size
        )
    
    def symmetric_quantize_int4(self, tensor: torch.Tensor, group_size: int = 128) -> QuantizedTensor:
        """
        对称INT4量化
        
        将浮点张量量化为4位整数（INT4）格式。使用对称量化方法，
        即量化范围关于零点对称：[-8, 7]。
        
        量化原理：
            1. 将输入张量展平并分组，每组group_size个元素
            2. 对每组计算缩放因子：scale = max(|group|) / 7
            3. 量化公式：quantized = round(group / scale)
            4. 裁剪到INT4范围：[-8, 7]
            5. 将两个INT4值打包到一个INT8字节中以节省空间
            
        INT4量化特点：
            - 存储空间：0.5字节/元素（两个INT4打包到一个INT8）
            - 量化范围较小：[-8, 7]
            - 精度较低，但压缩比更高
            - 适合对精度要求不高的权重矩阵
            
        打包格式：
            - 两个INT4值打包到一个UINT8字节中
            - 高4位存储第一个值，低4位存储第二个值
            - 打包后的数据类型为np.uint8
            
        Args:
            tensor (torch.Tensor): 输入张量
                - 支持任意形状的张量
                - 数据类型应为torch.float32或torch.float16
                - 会自动转换为numpy数组进行处理
                
            group_size (int): 每组元素数量，默认128
                - 必须为正整数
                - 必须是2的倍数（因为两个INT4打包到一个INT8）
                - 较小的值会提高精度但增加存储开销
                - 较大的值会减少存储开销但可能降低精度
                - 建议范围：32-256，通常是2的幂次
                
        Returns:
            QuantizedTensor: 量化后的张量对象
                - data: 打包后的INT4数据，形状为(n_groups, group_size//2)
                - scales: 缩放因子数组，形状为(n_groups,)
                - quant_type: QuantizationType.INT4
                - original_shape: 原始张量形状
                - group_size: 量化组大小
                
        Raises:
            无异常抛出，但可能因内存不足等原因导致计算失败
            
        使用示例：
            >>> quantizer = SVDQuantizer()
            >>> 
            >>> # 创建测试张量
            >>> tensor = torch.randn(100, 50)
            >>> 
            >>> # 进行INT4量化
            >>> quantized = quantizer.symmetric_quantize_int4(tensor, group_size=32)
            >>> 
            >>> # 检查量化结果
            >>> print(f"量化类型: {quantized.quant_type.value}")
            >>> print(f"数据形状: {quantized.data.shape}")
            >>> print(f"缩放因子数量: {len(quantized.scales)}")
            >>> print(f"数据范围: [{quantized.data.min()}, {quantized.data.max()}]")
            
        注意事项：
            1. 输入张量会被自动转换为numpy数组，会丢失梯度信息
            2. group_size必须是2的倍数，因为两个INT4打包到一个INT8
            3. 分组时会自动填充到group_size的整数倍，可能导致少量内存浪费
            4. 缩放因子是每组一个，用于反量化时恢复浮点值
            5. INT4量化的范围是[-8, 7]，超出范围的值会被裁剪
            6. 打包后的数据需要专门的解包函数才能正确读取
        """
        # 转换为numpy
        data = tensor.cpu().float().numpy()
        original_shape = data.shape
        
        # 展平为1D便于分组
        flat_data = data.flatten()
        
        # 计算需要多少组
        n_elements = len(flat_data)
        n_groups = (n_elements + group_size - 1) // group_size
        
        # 填充到group_size的整数倍
        padded_size = n_groups * group_size
        padded_data = np.zeros(padded_size, dtype=np.float32)
        padded_data[:n_elements] = flat_data
        
        # 重塑为 (n_groups, group_size)
        grouped_data = padded_data.reshape(n_groups, group_size)
        
        # 计算每组的最大绝对值作为缩放因子
        max_abs = np.max(np.abs(grouped_data), axis=1, keepdims=True)
        scales = max_abs / 7.0  # INT4范围: [-8, 7]
        
        # 避免除零
        scales = np.where(scales == 0, 1.0, scales)
        
        # 量化
        quantized = np.clip(np.round(grouped_data / scales), -8, 7).astype(np.int8)
        
        # 将两个INT4打包到一个INT8中（节省空间）
        # 高4位存储第一个值，低4位存储第二个值
        n_packed = n_groups * (group_size // 2)
        packed_data = np.zeros(n_packed, dtype=np.uint8)
        
        for i in range(0, group_size, 2):
            high = quantized[:, i] & 0x0F
            low = quantized[:, i + 1] & 0x0F
            packed = (high << 4) | low
            packed_data[i // 2::group_size // 2] = packed
        
        return QuantizedTensor(
            data=packed_data.reshape(n_groups, group_size // 2),
            scales=scales.flatten(),
            quant_type=QuantizationType.INT4,
            original_shape=original_shape,
            group_size=group_size
        )
    
    def dequantize(self, quantized: QuantizedTensor) -> np.ndarray:
        """
        反量化（用于验证）
        
        将量化后的整数数据反量化为浮点数据。主要用于验证量化质量
        和计算量化误差。
        
        反量化原理：
            1. 读取量化数据和缩放因子
            2. 根据量化类型选择对应的反量化方法
            3. 使用缩放因子恢复浮点值：dequantized = quantized_data * scale
            4. 重塑为原始张量形状
            
        反量化公式：
            - INT8: dequantized = int8_data * scale
            - INT4: dequantized = unpacked_int4_data * scale
            
        Args:
            quantized (QuantizedTensor): 量化后的张量对象
                - 包含量化数据、缩放因子、量化类型等信息
                - 通常由symmetric_quantize_int8或symmetric_quantize_int4生成
                
        Returns:
            np.ndarray: 反量化后的浮点数据
                - 形状与原始张量相同（quantized.original_shape）
                - 数据类型为np.float32
                - 用于与原始数据比较，计算量化误差
                
        Raises:
            ValueError: 如果量化类型不支持
                - 目前支持INT8和INT4两种量化类型
                - 其他量化类型会抛出ValueError异常
                
        使用示例：
            >>> quantizer = SVDQuantizer()
            >>> 
            >>> # 创建原始张量
            >>> original = torch.randn(100, 50)
            >>> 
            >>> # 进行INT8量化
            >>> quantized = quantizer.symmetric_quantize_int8(original, group_size=32)
            >>> 
            >>> # 反量化
            >>> dequantized = quantizer.dequantize(quantized)
            >>> 
            >>> # 计算量化误差
            >>> error = np.linalg.norm(original.numpy() - dequantized) / np.linalg.norm(original.numpy())
            >>> print(f"量化误差: {error:.6f}")
            
        注意事项：
            1. 反量化过程会丢失精度，反量化后的值与原始值会有差异
            2. INT4反量化需要先解包，将两个打包的INT4值分开
            3. 反量化后的数据类型为np.float32，与原始数据类型可能不同
            4. 反量化主要用于验证和误差计算，实际推理时不需要反量化
            5. 反量化误差是量化误差的主要来源，可以通过调整量化参数来控制
        """
        if quantized.quant_type == QuantizationType.INT8:
            # INT8反量化
            n_groups = len(quantized.scales)
            scales = quantized.scales.reshape(n_groups, 1)
            dequantized = quantized.data.astype(np.float32) * scales
            return dequantized.flatten()[:np.prod(quantized.original_shape)].reshape(quantized.original_shape)
            
        elif quantized.quant_type == QuantizationType.INT4:
            # INT4反量化
            n_groups = len(quantized.scales)
            scales = quantized.scales.reshape(n_groups, 1)
            
            # 解包INT4
            unpacked = np.zeros((n_groups, quantized.group_size), dtype=np.int8)
            for i in range(quantized.group_size // 2):
                packed = quantized.data[:, i]
                high = (packed >> 4) & 0x0F
                low = packed & 0x0F
                
                # 转换为有符号INT4
                high = np.where(high > 7, high - 16, high)
                low = np.where(low > 7, low - 16, low)
                
                unpacked[:, i * 2] = high
                unpacked[:, i * 2 + 1] = low
            
            dequantized = unpacked.astype(np.float32) * scales
            return dequantized.flatten()[:np.prod(quantized.original_shape)].reshape(quantized.original_shape)
        
        else:
            raise ValueError(f"Unsupported quantization type: {quantized.quant_type}")
    
    def decompose_and_quantize_layer(
        self, 
        layer_name: str, 
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        对单个层进行分解和量化
        
        根据层类型选择不同的处理策略：
            1. MoE专家层：先进行SVD分解，然后对分解后的矩阵进行混合精度量化
            2. 非专家层（注意力层、共享专家）：直接进行INT4量化
            
        处理流程：
            1. 判断层类型（专家层/非专家层）
            2. 如果是专家层：
               a. 获取该层的分解秩
               b. 执行SVD分解：W ≈ U_merged @ Vh
               c. 重构权重并计算相对误差
               d. 对U_merged进行INT8量化（精度更高）
               e. 对Vh进行INT4量化（压缩比更高）
            3. 如果是非专家层：
               a. 直接对权重进行INT4量化
            4. 如果有偏置项，保持FP16精度
            
        量化策略：
            - 专家层：SVD分解 + 混合精度量化（INT8 + INT4）
            - 非专家层：直接INT4量化
            - 偏置项：保持FP16精度（不进行量化）
            
        Args:
            layer_name (str): 层名称
                - 用于判断层类型和获取分解秩
                - 例如：'model.layers.0.mlp.gate_proj.weight'
                
            weight (torch.Tensor): 权重张量
                - 形状通常为(out_features, in_features)或(in_features, out_features)
                - 数据类型应为torch.float32或torch.float16
                
            bias (Optional[torch.Tensor]): 偏置张量（如果有）
                - 形状通常为(out_features,)
                - 如果提供，会保持FP16精度
                - 如果为None，则结果中bias为None
                
        Returns:
            Dict[str, Any]: 包含量化后数据的字典
                - layer_name (str): 层名称
                - original_shape (torch.Size): 原始权重形状
                - is_decomposed (bool): 是否进行了SVD分解
                - decomposed (Optional[Dict]): SVD分解结果（仅专家层）
                  - U_merged: 合并了奇异值的左奇异向量
                  - Vh: 右奇异向量矩阵的转置
                  - rank: 使用的分解秩
                - quantized (Dict): 量化结果
                  - 对于专家层：{'U_merged': QuantizedTensor, 'Vh': QuantizedTensor}
                  - 对于非专家层：{'weight': QuantizedTensor}
                - bias (Optional[torch.Tensor]): 偏置项（FP16精度）
                - error (float): 相对误差（仅专家层）
                
        Raises:
            无异常抛出，但可能因内存不足等原因导致计算失败
            
        使用示例：
            >>> quantizer = SVDQuantizer()
            >>> 
            >>> # 处理MoE专家层
            >>> expert_weight = torch.randn(256, 128)
            >>> result = quantizer.decompose_and_quantize_layer(
            ...     'model.layers.0.experts.0.gate_proj.weight',
            ...     expert_weight
            ... )
            >>> print(f"是否分解: {result['is_decomposed']}")
            >>> print(f"相对误差: {result['error']:.6f}")
            >>> 
            >>> # 处理注意力层
            >>> attn_weight = torch.randn(128, 128)
            >>> result = quantizer.decompose_and_quantize_layer(
            ...     'model.layers.0.self_attn.q_proj.weight',
            ...     attn_weight
            ... )
            >>> print(f"是否分解: {result['is_decomposed']}")
            
        注意事项：
            1. 专家层会进行SVD分解，非专家层直接进行INT4量化
            2. SVD分解后的U_merged使用INT8量化，Vh使用INT4量化
            3. 偏置项保持FP16精度，不进行量化
            4. 处理过程中会释放原始权重内存以节省空间
            5. 相对误差超过阈值时会发出警告，但不会阻止处理
            6. 统计信息会在处理过程中自动更新
        """
        result = {
            'layer_name': layer_name,
            'original_shape': weight.shape,
            'is_decomposed': False,
            'decomposed': None,
            'quantized': None,
            'bias': None,
            'error': 0.0
        }
        
        # 检查是否为MoE专家层
        if self.is_moe_expert_layer(layer_name):
            # 获取该层的秩
            rank = self.get_rank_for_layer(layer_name)
            
            # 执行SVD分解
            U_merged, Vh, original_weight = self.svd_decompose(weight, rank)
            
            # 重构权重用于误差验证
            reconstructed = U_merged @ Vh
            error = self.calculate_relative_error(original_weight, reconstructed)
            
            # 检查误差是否在允许范围内
            if error > self.config.max_relative_error:
                logger.warning(f"Layer {layer_name}: relative error {error:.6f} exceeds threshold {self.config.max_relative_error}")
            
            # 记录统计信息
            self.stats['errors'].append(error)
            if error > self.stats['max_error']:
                self.stats['max_error'] = error
            
            # 量化分解后的矩阵
            # U_merged 使用 INT8 量化
            U_quantized = self.symmetric_quantize_int8(U_merged, self.config.quant_group_size)
            
            # Vh 使用 INT4 量化
            Vh_quantized = self.symmetric_quantize_int4(Vh, self.config.quant_group_size)
            
            result['is_decomposed'] = True
            result['decomposed'] = {
                'U_merged': U_merged,
                'Vh': Vh,
                'rank': rank
            }
            result['quantized'] = {
                'U_merged': U_quantized,
                'Vh': Vh_quantized
            }
            result['error'] = error
            
            # 释放原始权重内存
            del weight
            gc.collect()
            
            self.stats['decomposed_layers'] += 1
            
        else:
            # 非专家层（注意力层和共享专家）直接INT4量化
            quantized = self.symmetric_quantize_int4(weight, self.config.quant_group_size)
            
            result['quantized'] = {
                'weight': quantized
            }
        
        # 保持偏置项为FP16精度
        if bias is not None:
            result['bias'] = bias.half()
        
        self.stats['total_layers'] += 1
        self.stats['quantized_layers'] += 1
        
        return result
    
    def process_model(self, model_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        处理整个模型的状态字典
        
        遍历模型状态字典中的所有参数，对每一层进行分解和量化处理。
        处理完成后会计算统计信息，包括压缩比、误差等指标。
        
        处理流程：
            1. 分离权重和偏置：将'.weight'和'.bias'后缀的参数分开
            2. 匹配权重和偏置：将同一层的权重和偏置配对
            3. 逐层处理：对每一层调用decompose_and_quantize_layer方法
            4. 计算统计信息：计算平均误差等统计指标
            
        状态字典格式：
            - 权重：'model.layers.0.mlp.gate_proj.weight'
            - 偏置：'model.layers.0.mlp.gate_proj.bias'
            - 会自动匹配同名的权重和偏置
            
        Args:
            model_state_dict (Dict[str, torch.Tensor]): 模型状态字典
                - 键为参数名称，值为参数张量
                - 通常由model.state_dict()获取
                - 支持包含权重和偏置的完整状态字典
                
        Returns:
            Dict[str, Any]: 处理后的结果字典
                - 键为层名称（去掉了'.weight'后缀）
                - 值为包含量化结果的字典
                - 每个层的结果包含：
                  - layer_name: 层名称
                  - original_shape: 原始权重形状
                  - is_decomposed: 是否进行了SVD分解
                  - decomposed: SVD分解结果（仅专家层）
                  - quantized: 量化结果
                  - bias: 偏置项（如果有）
                  - error: 相对误差（仅专家层）
                  
        Raises:
            无异常抛出，但可能因内存不足等原因导致处理失败
            
        使用示例：
            >>> quantizer = SVDQuantizer()
            >>> 
            >>> # 获取模型状态字典
            >>> model_state_dict = model.state_dict()
            >>> 
            >>> # 处理模型
            >>> results = quantizer.process_model(model_state_dict)
            >>> 
            >>> # 查看处理结果
            >>> print(f"处理层数: {len(results)}")
            >>> for layer_name, data in results.items():
            ...     if data['is_decomposed']:
            ...         print(f"{layer_name}: 秩={data['decomposed']['rank']}, 误差={data['error']:.6f}")
            ...     else:
            ...         print(f"{layer_name}: 直接量化")
            
        注意事项：
            1. 处理过程中会释放原始权重内存，处理完成后原状态字典可能不可用
            2. 只处理包含权重的层，纯偏置的层会被忽略
            3. 统计信息会在处理过程中自动更新，处理完成后可以查看
            4. 处理顺序按照状态字典中的键顺序进行
            5. 对于大模型，建议分批次处理以避免内存不足
        """
        logger.info(f"Starting to process model with {len(model_state_dict)} parameters")
        
        processed_layers = {}
        bias_layers = {}
        
        # 分离权重和偏置
        for name, param in model_state_dict.items():
            if '.weight' in name:
                base_name = name.replace('.weight', '')
                processed_layers[base_name] = {'weight': param}
            elif '.bias' in name:
                base_name = name.replace('.bias', '')
                if base_name in processed_layers:
                    processed_layers[base_name]['bias'] = param
                else:
                    processed_layers[base_name] = {'weight': None, 'bias': param}
        
        # 处理每一层
        results = {}
        for base_name, params in processed_layers.items():
            if params['weight'] is not None:
                logger.info(f"Processing layer: {base_name}")
                result = self.decompose_and_quantize_layer(
                    base_name, 
                    params['weight'],
                    params.get('bias')
                )
                results[base_name] = result
        
        # 计算平均误差
        if self.stats['errors']:
            self.stats['avg_error'] = sum(self.stats['errors']) / len(self.stats['errors'])
        
        logger.info(f"Processing complete. Stats: {self.stats}")
        
        return results
    
    def save_to_binary_format(self, results: Dict[str, Any], output_path: str):
        """
        将处理后的结果保存为自定义二进制格式
        
        将量化后的模型数据保存为自定义的二进制格式，便于快速加载和推理。
        二进制格式设计考虑了存储效率和加载速度。
        
        二进制格式结构：
            文件头：
                - 魔数：4字节，固定为'SVDQ'
                - 版本号：4字节，无符号整数，当前版本为1
                - 层数：4字节，无符号整数，表示文件中包含的层数
                
            每层数据：
                - 层名长度：4字节，无符号整数
                - 层名：UTF-8编码的字符串
                - 是否分解：1字节，布尔值
                - 原始形状：4字节 + 4*维度数
                - 误差：4字节，浮点数
                - 量化数据：根据是否分解，写入不同的数据结构
                  - 分解层：U_merged量化数据 + Vh量化数据
                  - 非分解层：权重量化数据
                - 偏置：1字节布尔值 + 可选的偏置数据
                
            量化张量数据结构：
                - 名称长度：4字节
                - 名称：UTF-8编码的字符串
                - 量化类型长度：4字节
                - 量化类型：UTF-8编码的字符串（'int8'或'int4'）
                - 组大小：4字节
                - 数据形状：4字节 + 4*维度数
                - 数据长度：4字节
                - 数据：原始字节
                - 缩放因子长度：4字节
                - 缩放因子：原始字节
                
        Args:
            results (Dict[str, Any]): 处理后的结果字典
                - 通常由process_model方法生成
                - 键为层名称，值为包含量化结果的字典
                
            output_path (str): 输出文件路径
                - 如果目录不存在，会自动创建
                - 建议使用.bin扩展名
                
        Returns:
            None
            
        Raises:
            OSError: 如果无法创建目录或写入文件
            
        使用示例：
            >>> quantizer = SVDQuantizer()
            >>> 
            >>> # 处理模型
            >>> results = quantizer.process_model(model_state_dict)
            >>> 
            >>> # 保存为二进制格式
            >>> quantizer.save_to_binary_format(results, "output/model_quantized.bin")
            >>> 
            >>> # 检查文件大小
            >>> import os
            >>> file_size = os.path.getsize("output/model_quantized.bin")
            >>> print(f"文件大小: {file_size / 1024 / 1024:.2f} MB")
            
        注意事项：
            1. 二进制格式需要配套的加载器才能正确读取
            2. 文件包含所有必要的元数据，可以独立加载和使用
            3. 偏置项以FP16格式保存，保持高精度
            4. 量化数据以原始字节保存，没有额外的压缩
            5. 文件大小与量化后的数据大小基本一致
            6. 建议在保存前检查磁盘空间是否充足
        """
        logger.info(f"Saving to binary format: {output_path}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            # 写入文件头
            magic = b'SVDQ'  # 魔数
            version = struct.pack('I', 1)  # 版本号
            num_layers = struct.pack('I', len(results))
            
            f.write(magic)
            f.write(version)
            f.write(num_layers)
            
            # 写入每一层的数据
            for layer_name, data in results.items():
                # 写入层名
                name_bytes = layer_name.encode('utf-8')
                f.write(struct.pack('I', len(name_bytes)))
                f.write(name_bytes)
                
                # 写入是否分解
                f.write(struct.pack('?', data['is_decomposed']))
                
                # 写入原始形状
                shape = data['original_shape']
                f.write(struct.pack('I', len(shape)))
                for dim in shape:
                    f.write(struct.pack('I', dim))
                
                # 写入误差
                f.write(struct.pack('f', data['error']))
                
                # 写入量化数据
                if data['is_decomposed']:
                    # 分解的情况：写入U_merged和Vh
                    self._write_quantized_tensor(f, data['quantized']['U_merged'], 'U_merged')
                    self._write_quantized_tensor(f, data['quantized']['Vh'], 'Vh')
                else:
                    # 非分解的情况：直接写入权重
                    self._write_quantized_tensor(f, data['quantized']['weight'], 'weight')
                
                # 写入偏置（如果有）
                has_bias = data['bias'] is not None
                f.write(struct.pack('?', has_bias))
                if has_bias:
                    bias_data = data['bias'].cpu().numpy().astype(np.float16)
                    f.write(struct.pack('I', len(bias_data.tobytes())))
                    f.write(bias_data.tobytes())
        
        logger.info(f"Binary file saved successfully: {output_path}")
    
    def _write_quantized_tensor(self, f, tensor: QuantizedTensor, name: str):
        """
        写入量化张量到文件
        
        将量化后的张量数据按照二进制格式写入文件。包括张量的元数据
        （名称、量化类型、组大小、形状）和实际数据（量化数据和缩放因子）。
        
        写入格式：
            1. 名称长度：4字节，无符号整数
            2. 名称：UTF-8编码的字符串
            3. 量化类型长度：4字节
            4. 量化类型：UTF-8编码的字符串（'int8'或'int4'）
            5. 组大小：4字节，无符号整数
            6. 数据形状维度数：4字节
            7. 数据形状：4*维度数
            8. 数据长度：4字节
            9. 数据：原始字节
            10. 缩放因子长度：4字节
            11. 缩放因子：原始字节
            
        Args:
            f (file): 文件对象
                - 以二进制写入模式打开的文件
                - 文件指针应该在正确的位置
                
            tensor (QuantizedTensor): 量化张量对象
                - 包含量化数据、缩放因子、量化类型等信息
                - 通常由symmetric_quantize_int8或symmetric_quantize_int4生成
                
            name (str): 张量名称
                - 用于标识张量的名称，如'U_merged'、'Vh'、'weight'
                - 会以UTF-8编码写入文件
                
        Returns:
            None
            
        Raises:
            无异常抛出，但可能因文件写入失败等原因导致错误
            
        使用示例：
            >>> # 通常由save_to_binary_format方法内部调用
            >>> quantizer = SVDQuantizer()
            >>> 
            >>> # 创建量化张量
            >>> tensor = quantizer.symmetric_quantize_int8(torch.randn(100, 50))
            >>> 
            >>> # 写入文件
            >>> with open('test.bin', 'wb') as f:
            ...     quantizer._write_quantized_tensor(f, tensor, 'test_tensor')
            
        注意事项：
            1. 该方法通常由save_to_binary_format方法内部调用，不建议直接使用
            2. 文件指针会在写入后移动到新位置
            3. 数据以原始字节保存，没有额外的压缩
            4. 缩放因子是反量化时的关键信息，必须正确保存
            5. 量化类型决定了数据的解释方式
        """
        # 写入名称
        name_bytes = name.encode('utf-8')
        f.write(struct.pack('I', len(name_bytes)))
        f.write(name_bytes)
        
        # 写入量化类型
        quant_type_bytes = tensor.quant_type.value.encode('utf-8')
        f.write(struct.pack('I', len(quant_type_bytes)))
        f.write(quant_type_bytes)
        
        # 写入组大小
        f.write(struct.pack('I', tensor.group_size))
        
        # 写入数据形状
        f.write(struct.pack('I', len(tensor.data.shape)))
        for dim in tensor.data.shape:
            f.write(struct.pack('I', dim))
        
        # 写入数据
        data_bytes = tensor.data.tobytes()
        f.write(struct.pack('I', len(data_bytes)))
        f.write(data_bytes)
        
        # 写入缩放因子
        scales_bytes = tensor.scales.tobytes()
        f.write(struct.pack('I', len(scales_bytes)))
        f.write(scales_bytes)
    
    def print_statistics(self):
        """
        打印处理统计信息
        
        打印SVD量化处理的详细统计信息，包括层数统计、误差统计、
        压缩比信息和评估指标等。用于分析量化效果和调试。
        
        打印内容：
            1. 层数统计：
               - 总层数：处理的层总数
               - 分解层数：进行SVD分解的层数
               - 量化层数：进行量化的层数
               
            2. 误差统计：
               - 最大相对误差：所有分解层中的最大误差
               - 平均相对误差：所有分解层的平均误差
               
            3. 压缩比信息（如果可用）：
               - 压缩比：原始大小/量化后大小
               - 原始大小：原始模型的总大小（MB）
               - 量化后大小：量化后的总大小（MB）
               
            4. 评估指标（如果可用）：
               - MSE：均方误差
               - MAE：平均绝对误差
               - RMSE：均方根误差
               - 困惑度：原始和量化后的困惑度（如果提供）
               
        Args:
            无
            
        Returns:
            None
            
        Raises:
            无异常抛出
            
        使用示例：
            >>> quantizer = SVDQuantizer()
            >>> 
            >>> # 处理模型
            >>> results = quantizer.process_model(model_state_dict)
            >>> 
            >>> # 计算压缩比
            >>> quantizer.calculate_compression_ratio(model_state_dict, results)
            >>> 
            >>> # 打印统计信息
            >>> quantizer.print_statistics()
            
        注意事项：
            1. 统计信息会在处理过程中自动更新
            2. 压缩比信息需要调用calculate_compression_ratio方法后才可用
            3. 评估指标需要调用evaluate_with_validation_data方法后才可用
            4. 统计信息以格式化的表格形式打印，便于阅读
            5. 可以用于比较不同配置的量化效果
        """
        print("\n" + "="*60)
        print("SVD量化处理统计")
        print("="*60)
        print(f"总层数: {self.stats['total_layers']}")
        print(f"分解层数: {self.stats['decomposed_layers']}")
        print(f"量化层数: {self.stats['quantized_layers']}")
        print(f"最大相对误差: {self.stats['max_error']:.6f}")
        print(f"平均相对误差: {self.stats['avg_error']:.6f}")
        
        # 打印压缩比信息
        if 'original_size' in self.stats and 'quantized_size' in self.stats:
            print(f"\n压缩比: {self.stats['compression_ratio']:.2f}x")
            print(f"原始大小: {self.stats['original_size'] / 1024 / 1024:.2f} MB")
            print(f"量化后大小: {self.stats['quantized_size'] / 1024 / 1024:.2f} MB")
        
        # 打印评估指标
        if 'evaluation_metrics' in self.stats:
            metrics = self.stats['evaluation_metrics']
            print(f"\n评估指标:")
            print(f"  MSE: {metrics.get('mse', 'N/A'):.6f}")
            print(f"  MAE: {metrics.get('mae', 'N/A'):.6f}")
            print(f"  RMSE: {metrics.get('rmse', 'N/A'):.6f}")
            if 'perplexity_original' in metrics:
                print(f"  原始困惑度: {metrics['perplexity_original']:.4f}")
                print(f"  量化困惑度: {metrics['perplexity_quantized']:.4f}")
                print(f"  困惑度变化: {metrics['perplexity_change']:.4f}")
        
        print("="*60)
    
    def calculate_compression_ratio(self, original_state_dict: Dict[str, torch.Tensor], 
                                   quantized_results: Dict[str, Any]) -> float:
        """
        计算压缩比
        
        计算原始模型与量化后模型之间的压缩比。压缩比是衡量模型压缩效果
        的重要指标，压缩比越高表示压缩效果越好。
        
        计算公式：
            压缩比 = 原始模型大小 / 量化后模型大小
            
        大小计算方法：
            - 原始模型大小：所有参数的元素数量 * 每个元素的字节数
            - 量化后模型大小：
              - SVD分解层：U_merged(INT8) + Vh(INT4) + 偏置(FP16)
              - 非分解层：权重量化(INT4) + 偏置(FP16)
              
        字节数计算：
            - FP32：4字节/元素
            - FP16：2字节/元素
            - INT8：1字节/元素
            - INT4：0.5字节/元素（两个INT4打包到一个INT8）
            
        Args:
            original_state_dict (Dict[str, torch.Tensor]): 原始模型状态字典
                - 键为参数名称，值为参数张量
                - 通常由model.state_dict()获取
                
            quantized_results (Dict[str, Any]): 量化后的结果
                - 通常由process_model方法生成
                - 包含量化后的数据和元信息
                
        Returns:
            float: 压缩比
                - 值大于1表示压缩成功
                - 值越大表示压缩效果越好
                - 如果量化后大小为0，返回无穷大
                
        Raises:
            无异常抛出
            
        使用示例：
            >>> quantizer = SVDQuantizer()
            >>> 
            >>> # 处理模型
            >>> results = quantizer.process_model(model_state_dict)
            >>> 
            >>> # 计算压缩比
            >>> compression_ratio = quantizer.calculate_compression_ratio(model_state_dict, results)
            >>> print(f"压缩比: {compression_ratio:.2f}x")
            >>> 
            >>> # 查看详细信息
            >>> print(f"原始大小: {quantizer.stats['original_size'] / 1024 / 1024:.2f} MB")
            >>> print(f"量化后大小: {quantizer.stats['quantized_size'] / 1024 / 1024:.2f} MB")
            
        注意事项：
            1. 压缩比计算考虑了所有层，包括分解层和非分解层
            2. 偏置项以FP16格式保存，占用空间是原始FP32的一半
            3. INT4量化比INT8量化节省更多空间，但精度更低
            4. 压缩比会保存到统计信息中，可以通过print_statistics查看
            5. 对于大模型，压缩比通常在2-8倍之间
        """
        # 计算原始大小
        original_size = sum(tensor.numel() * tensor.element_size() for tensor in original_state_dict.values())
        
        # 计算量化后大小
        quantized_size = 0
        for layer_name, data in quantized_results.items():
            if data['is_decomposed']:
                # SVD分解层: U_merged (INT8) + Vh (INT4)
                U_size = data['quantized']['U_merged'].data.size  # INT8: 1 byte/element
                Vh_size = data['quantized']['Vh'].data.size * 0.5  # INT4: 0.5 bytes/element
                quantized_size += U_size + Vh_size
            else:
                # 直接量化层: INT4
                weight_size = data['quantized']['weight'].data.size * 0.5  # INT4: 0.5 bytes/element
                quantized_size += weight_size
            
            # 偏置
            if data.get('bias') is not None:
                quantized_size += data['bias'].numel() * 2  # FP16
        
        compression_ratio = original_size / quantized_size if quantized_size > 0 else float('inf')
        
        # 保存到统计信息
        self.stats['original_size'] = original_size
        self.stats['quantized_size'] = quantized_size
        self.stats['compression_ratio'] = compression_ratio
        
        return compression_ratio
    
    def evaluate_with_validation_data(self, original_state_dict: Dict[str, torch.Tensor],
                                     quantized_results: Dict[str, Any],
                                     validation_data: Optional[Dict[str, torch.Tensor]] = None,
                                     model=None, tokenizer=None) -> Dict[str, float]:
        """
        使用验证数据集评估量化后的模型
        
        评估量化后的模型质量，包括权重级别的误差指标和模型级别的困惑度指标。
        用于分析量化对模型精度的影响。
        
        评估指标：
            1. 权重级别指标：
               - MSE（均方误差）：衡量量化后权重与原始权重的平均平方差
               - MAE（平均绝对误差）：衡量量化后权重与原始权重的平均绝对差
               - RMSE（均方根误差）：MSE的平方根，与原始数据单位一致
               
            2. 模型级别指标（如果提供模型和验证数据）：
               - 困惑度：衡量语言模型预测能力的指标
               - 困惑度变化：量化后困惑度与原始困惑度的差异
               
        Args:
            original_state_dict (Dict[str, torch.Tensor]): 原始模型状态字典
                - 键为参数名称，值为参数张量
                - 用于计算权重级别的误差指标
                
            quantized_results (Dict[str, Any]): 量化后的结果
                - 通常由process_model方法生成
                - 包含量化后的数据和元信息
                
            validation_data (Optional[Dict[str, torch.Tensor]]): 验证数据集
                - 用于计算困惑度
                - 应包含'input_ids'和'labels'键
                - 如果为None，则不计算困惑度
                
            model: 原始模型
                - 用于计算困惑度
                - 如果为None，则不计算困惑度
                
            tokenizer: 分词器
                - 用于处理验证数据
                - 如果为None，则使用默认处理方式
                
        Returns:
            Dict[str, float]: 评估指标字典
                - mse: 均方误差
                - mae: 平均绝对误差
                - rmse: 均方根误差
                - perplexity_original: 原始困惑度（如果计算）
                - perplexity_quantized: 量化后困惑度（如果计算）
                - perplexity_change: 困惑度变化（如果计算）
                
        Raises:
            无异常抛出，但可能因内存不足等原因导致计算失败
            
        使用示例：
            >>> quantizer = SVDQuantizer()
            >>> 
            >>> # 处理模型
            >>> results = quantizer.process_model(model_state_dict)
            >>> 
            >>> # 评估量化质量（仅权重级别）
            >>> metrics = quantizer.evaluate_with_validation_data(model_state_dict, results)
            >>> print(f"MSE: {metrics['mse']:.6f}")
            >>> print(f"MAE: {metrics['mae']:.6f}")
            >>> 
            >>> # 评估量化质量（包含困惑度）
            >>> metrics = quantizer.evaluate_with_validation_data(
            ...     model_state_dict, results, validation_data, model, tokenizer
            ... )
            >>> print(f"原始困惑度: {metrics['perplexity_original']:.4f}")
            >>> print(f"量化后困惑度: {metrics['perplexity_quantized']:.4f}")
            
        注意事项：
            1. 权重级别指标是通过反量化后与原始权重比较计算的
            2. 困惑度计算需要完整的模型和验证数据，计算成本较高
            3. 评估结果会保存到统计信息中，可以通过print_statistics查看
            4. MSE、MAE、RMSE越小表示量化质量越好
            5. 困惑度变化越小表示量化对模型性能影响越小
        """
        metrics = {}
        
        # 1. 计算权重级别的MSE、MAE
        mse_list = []
        mae_list = []
        
        for layer_name, data in quantized_results.items():
            # 从原始状态字典中获取对应的权重
            # 注意：quantized_results的键名已经去掉了'.weight'后缀
            original_key = layer_name + '.weight'
            if original_key not in original_state_dict:
                continue
            
            original_tensor = original_state_dict[original_key]
            
            # 反量化权重
            if data['is_decomposed']:
                # SVD分解层
                U_dequant = self.dequantize(data['quantized']['U_merged'])
                Vh_dequant = self.dequantize(data['quantized']['Vh'])
                reconstructed = U_dequant @ Vh_dequant
            else:
                # 直接量化层
                reconstructed = self.dequantize(data['quantized']['weight'])
            
            # 计算MSE和MAE
            original_np = original_tensor.cpu().float().numpy()
            mse = np.mean((original_np - reconstructed) ** 2)
            mae = np.mean(np.abs(original_np - reconstructed))
            
            mse_list.append(mse)
            mae_list.append(mae)
        
        # 计算平均指标
        metrics['mse'] = np.mean(mse_list)
        metrics['mae'] = np.mean(mae_list)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # 2. 计算困惑度（如果提供了模型和验证数据）
        if model is not None and validation_data is not None:
            try:
                perplexity_metrics = self._calculate_perplexity(
                    model, validation_data, tokenizer
                )
                metrics.update(perplexity_metrics)
            except Exception as e:
                logger.warning(f"无法计算困惑度: {e}")
        
        # 保存到统计信息
        self.stats['evaluation_metrics'] = metrics
        
        return metrics
    
    def _calculate_perplexity(self, model, validation_data: Dict[str, torch.Tensor],
                            tokenizer=None) -> Dict[str, float]:
        """
        计算困惑度
        
        计算语言模型在验证数据集上的困惑度。困惑度是衡量语言模型预测能力
        的重要指标，困惑度越低表示模型预测能力越强。
        
        困惑度定义：
            困惑度 = exp(交叉熵损失)
            其中交叉熵损失是模型在验证数据集上的平均损失
            
        困惑度解释：
            - 困惑度 = 1: 模型完美预测，没有不确定性
            - 困惑度 = 10: 模型平均有10个等可能的选择
            - 困惑度 = 100: 模型平均有100个等可能的选择
            - 困惑度越高，模型预测能力越差
            
        Args:
            model: 语言模型
                - 应该支持forward方法，并返回包含loss的输出
                - 通常为HuggingFace Transformers模型
                
            validation_data (Dict[str, torch.Tensor]): 验证数据集
                - 应包含'input_ids'和'labels'键
                - input_ids: 输入token ID，形状为(batch_size, seq_len)
                - labels: 标签token ID，形状为(batch_size, seq_len)
                
            tokenizer: 分词器
                - 目前未使用，预留用于未来的数据预处理
                
        Returns:
            Dict[str, float]: 包含困惑度的字典
                - perplexity_original: 原始模型的困惑度
                - perplexity_quantized: 量化后模型的困惑度（占位符）
                - perplexity_change: 困惑度变化（占位符）
                
        Raises:
            无异常抛出，但如果模型或数据格式不正确可能导致错误
            
        使用示例：
            >>> quantizer = SVDQuantizer()
            >>> 
            >>> # 准备验证数据
            >>> validation_data = {
            ...     'input_ids': torch.randint(0, 1000, (1, 128)),
            ...     'labels': torch.randint(0, 1000, (1, 128))
            ... }
            >>> 
            >>> # 计算困惑度
            >>> perplexity_metrics = quantizer._calculate_perplexity(model, validation_data)
            >>> print(f"原始困惑度: {perplexity_metrics['perplexity_original']:.4f}")
            
        注意事项：
            1. 目前只是简化版本，实际实现需要根据模型结构调整
            2. 量化后的困惑度需要在实际量化后的模型上计算
            3. 困惑度计算需要较大的计算资源，建议使用GPU加速
            4. 验证数据应该足够大，以获得可靠的困惑度估计
            5. 困惑度是模型级别的指标，与权重级别的MSE、MAE等指标互补
        """
        import torch.nn.functional as F
        
        # 这里需要根据实际模型结构实现
        # 简化版本：假设validation_data包含input_ids和labels
        
        perplexity_metrics = {}
        
        if 'input_ids' in validation_data and 'labels' in validation_data:
            input_ids = validation_data['input_ids']
            labels = validation_data['labels']
            
            # 计算原始模型的困惑度
            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                perplexity_original = torch.exp(loss).item()
                perplexity_metrics['perplexity_original'] = perplexity_original
            
            # 注意：量化后的困惑度需要在实际量化后的模型上计算
            # 这里只是示例，实际实现需要加载量化后的权重到模型中
            perplexity_metrics['perplexity_quantized'] = perplexity_original  # 占位符
            perplexity_metrics['perplexity_change'] = 0.0  # 占位符
        
        return perplexity_metrics


def main():
    """
    主函数：演示SVD量化流程
    
    演示完整的SVD量化流程，包括：
        1. 创建配置和量化器
        2. 模拟MoE模型的状态字典
        3. 处理模型（SVD分解和量化）
        4. 计算压缩比和评估指标
        5. 保存为二进制格式
        6. 验证量化误差
        
    使用示例：
        >>> python main.py
        
    注意事项：
        1. 本示例使用随机生成的数据，实际使用时应加载真实模型
        2. 处理大模型时需要注意内存使用
        3. 可以通过修改配置来调整压缩比和精度
        4. 输出文件保存在output目录下
    """
    
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
    
    # 示例：模拟一个MoE模型的状态字典
    logger.info("Creating example model state dict...")
    
    # 模拟一些层
    example_state_dict = {
        'model.layers.0.mlp.experts.0.gate_proj.weight': torch.randn(2048, 512),
        'model.layers.0.mlp.experts.0.up_proj.weight': torch.randn(2048, 512),
        'model.layers.0.mlp.experts.0.down_proj.weight': torch.randn(512, 2048),
        'model.layers.0.self_attn.q_proj.weight': torch.randn(512, 512),
        'model.layers.0.self_attn.k_proj.weight': torch.randn(512, 512),
        'model.layers.0.self_attn.v_proj.weight': torch.randn(512, 512),
        'model.layers.0.self_attn.o_proj.weight': torch.randn(512, 512),
    }
    
    # 处理模型
    results = quantizer.process_model(example_state_dict)
    
    # 计算压缩比
    compression_ratio = quantizer.calculate_compression_ratio(example_state_dict, results)
    logger.info(f"Compression ratio: {compression_ratio:.2f}x")
    
    # 评估量化质量（使用MSE、MAE）
    logger.info("\nEvaluating quantization quality...")
    metrics = quantizer.evaluate_with_validation_data(example_state_dict, results)
    
    # 保存为二进制格式
    output_path = "output/model_quantized.bin"
    quantizer.save_to_binary_format(results, output_path)
    
    # 打印统计信息
    quantizer.print_statistics()
    
    # 验证量化误差
    logger.info("\nVerifying quantization errors...")
    for layer_name, data in results.items():
        if data['is_decomposed']:
            # 反量化U_merged和Vh
            U_dequant = quantizer.dequantize(data['quantized']['U_merged'])
            Vh_dequant = quantizer.dequantize(data['quantized']['Vh'])
            
            # 重构权重
            reconstructed = U_dequant @ Vh_dequant
            
            # 计算误差
            original = data['decomposed']['U_merged'] @ data['decomposed']['Vh']
            error = np.linalg.norm(original.cpu().numpy() - reconstructed) / np.linalg.norm(original.cpu().numpy())
            logger.info(f"Layer {layer_name}: dequantization error = {error:.6f}")


if __name__ == "__main__":
    main()
