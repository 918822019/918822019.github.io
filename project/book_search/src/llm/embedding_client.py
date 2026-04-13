"""
Embedding 客户端模块
负责文本向量化 embedding 的调用和管理
"""

import numpy as np
from typing import List, Optional, Union

from .env import config


class EmbeddingClient:
    """Embedding 客户端类，统一管理文本向量化调用"""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化 Embedding 客户端
        
        Args:
            model_name: 模型名称（默认从配置读取）
            api_key: API 密钥（默认从配置读取）
            base_url: API 基础 URL（默认从配置读取）
        """
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.api_key = api_key or config.EMBEDDING_API_KEY
        self.base_url = base_url or config.EMBEDDING_BASE_URL
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """初始化具体的 Embedding 客户端"""
        # TODO: 根据配置初始化不同的 Embedding 客户端
        # 例如：OpenAI Embeddings, Qwen Embeddings, BGE 等
        pass

    def embed(self, text: str) -> List[float]:
        """
        将单个文本转换为向量
        
        Args:
            text: 输入文本
            
        Returns:
            文本的向量表示
        """
        # TODO: 实现实际的 embedding 调用逻辑
        # 这里返回一个示例向量
        return [0.0] * 768

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量将文本转换为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表
        """
        return [self.embed(text) for text in texts]

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            余弦相似度值 (0-1)
        """
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def search_similar(self, query_embedding: List[float],
                       candidates: List[List[float]],
                       top_k: int = 5) -> List[tuple]:
        """
        在候选向量中搜索最相似的向量
        
        Args:
            query_embedding: 查询向量
            candidates: 候选向量列表
            top_k: 返回前 k 个结果
            
        Returns:
            按相似度排序的 (索引, 相似度) 元组列表
        """
        similarities = [
            (idx, self.cosine_similarity(query_embedding, cand))
            for idx, cand in enumerate(candidates)
        ]

        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]
