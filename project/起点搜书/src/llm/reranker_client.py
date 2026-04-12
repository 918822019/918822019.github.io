"""
Reranker 客户端模块
负责重排序模型的调用和管理
"""

from typing import List, Tuple, Optional


class RerankerClient:
    """Reranker 客户端类，统一管理重排序模型调用"""
    
    def __init__(self, model_name: str = "default", api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化 Reranker 客户端
        
        Args:
            model_name: 模型名称
            api_key: API 密钥
            base_url: API 基础 URL
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化具体的 Reranker 客户端"""
        # TODO: 根据配置初始化不同的 Reranker 客户端
        # 例如：BGE Reranker, Cohere Rerank 等
        pass
    
    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        对文档列表进行重排序
        
        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回前 k 个结果，None 则返回全部
            
        Returns:
            按相关性排序的 (原始索引, 相关性分数) 元组列表
        """
        # TODO: 实现实际的 reranker 调用逻辑
        # 这里返回示例结果
        scores = [(idx, 0.5) for idx in range(len(documents))]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            scores = scores[:top_k]
        
        return scores
    
    def rerank_with_metadata(self, query: str, 
                            documents: List[dict], 
                            text_field: str = "text",
                            top_k: Optional[int] = None) -> List[Tuple[int, float, dict]]:
        """
        对带元数据的文档列表进行重排序
        
        Args:
            query: 查询文本
            documents: 待排序的文档列表（包含元数据）
            text_field: 文档中文本字段的名称
            top_k: 返回前 k 个结果
            
        Returns:
            按相关性排序的 (原始索引, 相关性分数, 文档元数据) 元组列表
        """
        texts = [doc.get(text_field, "") for doc in documents]
        scores = self.rerank(query, texts, top_k)
        
        result = []
        for idx, score in scores:
            result.append((idx, score, documents[idx]))
        
        return result
