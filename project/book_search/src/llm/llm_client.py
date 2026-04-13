"""
LLM 客户端模块
负责大语言模型的调用和管理
"""

import json
from typing import Optional
from urllib import error, request

from .env import config


class LLMClient:
    """LLM 客户端类，统一管理大语言模型调用"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        初始化 LLM 客户端

        Args:
            model_name: 模型名称（默认从配置读取）
            api_key: API 密钥（默认从配置读取）
            base_url: API 基础 URL（默认从配置读取）
        """
        self.model_name = model_name or config.LLM_MODEL_NAME
        self.api_key = api_key or config.LLM_API_KEY
        self.base_url = base_url or config.LLM_BASE_URL
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """初始化具体的 LLM 客户端（预留）"""
        self.client = {
            "base_url": self.base_url.rstrip("/"),
            "model": self.model_name,
        }

    def _post_chat_completion(self, messages: list, **kwargs) -> str:
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未设置，无法调用模型。")

        payload = {
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

        req = request.Request(api_url, data=body, headers=headers, method="POST")
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
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LLM 返回解析失败: {resp_data}") from exc

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> str:
        """
        生成文本响应

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            **kwargs: 其他参数

        Returns:
            生成的文本
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._post_chat_completion(messages=messages, **kwargs)

    def chat(self, messages: list, **kwargs) -> str:
        """
        对话式调用

        Args:
            messages: 消息列表，格式为 [{"role": "user/assistant/system", "content": "..."}]
            **kwargs: 其他参数

        Returns:
            助手的回复
        """
        return self._post_chat_completion(messages=messages, **kwargs)

    def generate_with_context(self, query: str, context: str, **kwargs) -> str:
        """
        基于上下文生成回答（适用于 RAG 场景）

        Args:
            query: 用户查询
            context: 相关上下文信息
            **kwargs: 其他参数

        Returns:
            基于上下文的回答
        """
        prompt = f"""基于以下信息回答问题：

相关信息：
{context}

问题：{query}

请根据上述信息给出回答："""

        return self.generate(prompt, **kwargs)
