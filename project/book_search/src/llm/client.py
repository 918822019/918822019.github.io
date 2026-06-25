"""LLM / Embedding / Reranker 客户端整合模块。"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib import error, request

import numpy as np

from src.config import config


class LLMClient:
    """LLM 客户端类，统一管理大语言模型调用"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model_name = model_name or config.LLM_MODEL_NAME
        self.api_key = api_key or config.LLM_API_KEY
        self.base_url = base_url or config.LLM_BASE_URL
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """初始化 LLM 客户端配置字典。"""
        self.client = {
            "base_url": self.base_url.rstrip("/"),
            "model": self.model_name,
        }

    def _post_chat_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """发送对话补全请求，返回模型回复文本。"""
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未设置，无法调用模型。")

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.pop("temperature", 0.2),
        }
        payload.update(kwargs)

        body = json.dumps(payload).encode("utf-8")
        api_url = f"{self.base_url.rstrip('/')}/chat/completions"
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
                f"LLM 调用失败: HTTP {exc.code}, detail={detail}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"LLM 调用失败: {exc.reason}") from exc

        try:
            data = json.loads(resp_data)
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            raise RuntimeError(f"LLM 返回解析失败: {resp_data}") from exc

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """根据 prompt 生成回复，支持可选的 system prompt。"""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._post_chat_completion(messages=messages, **kwargs)

    def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """直接传入消息列表进行对话。"""
        return self._post_chat_completion(messages=messages, **kwargs)

    def generate_with_context(
        self,
        query: str,
        context: str,
        **kwargs: Any,
    ) -> str:
        """基于上下文信息生成回答。"""
        prompt = f"""基于以下信息回答问题：

相关信息：
{context}

问题：{query}

请根据上述信息给出回答："""

        return self.generate(prompt, **kwargs)


class EmbeddingClient:
    """Embedding 客户端类，统一管理文本向量化调用"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.api_key = api_key or config.EMBEDDING_API_KEY
        self.base_url = base_url or config.EMBEDDING_BASE_URL
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """初始化 Embedding 客户端配置字典。"""
        self.client = {
            "base_url": self.base_url.rstrip("/"),
            "model": self.model_name,
        }

    def _post_embeddings(
        self,
        input_data: Union[str, List[str]],
        **kwargs: Any,
    ) -> List[List[float]]:
        """发送 embedding 请求，返回向量列表。"""
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
        except Exception as exc:
            raise RuntimeError(f"Embedding 返回解析失败: {resp_data}") from exc

    def embed(self, text: str) -> List[float]:
        """对单段文本生成向量。"""
        vectors = self._post_embeddings(input_data=text)
        if not vectors:
            raise RuntimeError("Embedding 返回为空")
        return vectors[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本向量。"""
        if not texts:
            return []
        return self._post_embeddings(input_data=texts)

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度。"""
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
        """在候选向量中搜索与查询向量最相似的 top_k 个。"""
        similarities = [
            (idx, self.cosine_similarity(query_embedding, cand))
            for idx, cand in enumerate(candidates)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class RerankerClient:
    """Reranker 客户端类，统一管理重排序模型调用"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model_name = model_name or config.RERANKER_MODEL_NAME
        self.api_key = api_key or config.RERANKER_API_KEY
        self.base_url = base_url or config.RERANKER_BASE_URL
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """初始化 Reranker 客户端（当前为桩实现）。"""
        pass

    def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """对文档列表按与 query 的相关性重排序。"""
        scores = [(idx, 0.5) for idx in range(len(documents))]
        scores.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            scores = scores[:top_k]
        return scores


