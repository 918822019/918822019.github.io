"""
SVD量化工具包

基于奇异值分解(SVD)的模型压缩和量化工具，专门针对混合专家(MoE)模型架构设计。
"""

from .main import SVDQuantizer, DecompositionConfig, QuantizationType

__version__ = "1.0.0"
__author__ = "AI助手"

__all__ = [
    'SVDQuantizer',
    'DecompositionConfig', 
    'QuantizationType'
]
