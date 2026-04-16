"""
Embedding 客户端模块
负责文本向量化 embedding 的调用和管理
"""

import json
import numpy as np
from typing import Any, List, Optional, Tuple, Union
from urllib import error, request

from .env import config


class EmbeddingClient:
    """Embedding 客户端类，统一管理文本向量化调用"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
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
        self.client = {
            "base_url": self.base_url.rstrip("/"),
            "model": self.model_name,
        }

    def _post_embeddings(
        self,
        input_data: Union[str, List[str]],
        **kwargs: Any,
    ) -> List[List[float]]:
        """调用 OpenAI 兼容 embeddings 接口。"""
        if not self.api_key:
            raise ValueError("EMBEDDING_API_KEY 未设置，无法调用 embedding 模型。")

        payload = {
            "model": self.model_name,
            "input": input_data,
        }
        payload.update(kwargs)

        body = json.dumps(payload).encode("utf-8")
        api_url = f"{self.base_url.rstrip('/')}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        req = request.Request(
            api_url,
            data=body,
            headers=headers,
            method="POST",
        )
        timeout = kwargs.get("timeout", config.REQUEST_TIMEOUT)
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                resp_data = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Embedding 调用失败: HTTP {exc.code}, detail={detail}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"Embedding 调用失败: {exc.reason}") from exc

        try:
            data = json.loads(resp_data)
            vectors = [item["embedding"] for item in data["data"]]
            return vectors
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Embedding 返回解析失败: {resp_data}") from exc

    def embed(self, text: str) -> List[float]:
        """
        将单个文本转换为向量

        Args:
            text: 输入文本

        Returns:
            文本的向量表示
        """
        vectors = self._post_embeddings(input_data=text)
        if not vectors:
            raise RuntimeError("Embedding 返回为空")
        return vectors[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量将文本转换为向量

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        if not texts:
            return []
        return self._post_embeddings(input_data=texts)

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

    def search_similar(
        self,
        query_embedding: List[float],
        candidates: List[List[float]],
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
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
