"""
环境配置模块
统一管理 API 调用所需的配置信息
"""

import os
from typing import Optional


class EnvConfig:
    """环境配置类，管理所有 API 相关的配置"""

    # ==================== LLM 配置 ====================
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")

    # ==================== Embedding 配置 ====================
    EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-ada-002")

    # ==================== Reranker 配置 ====================
    RERANKER_BASE_URL = os.getenv("RERANKER_BASE_URL", "")
    RERANKER_API_KEY = os.getenv("RERANKER_API_KEY", "")
    RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "bge-reranker-base")

    # ==================== 通用配置 ====================
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

    @classmethod
    def get_llm_config(cls) -> dict:
        """获取 LLM 配置"""
        return {
            "base_url": cls.LLM_BASE_URL,
            "api_key": cls.LLM_API_KEY,
            "model_name": cls.LLM_MODEL_NAME,
        }

    @classmethod
    def get_embedding_config(cls) -> dict:
        """获取 Embedding 配置"""
        return {
            "base_url": cls.EMBEDDING_BASE_URL,
            "api_key": cls.EMBEDDING_API_KEY,
            "model_name": cls.EMBEDDING_MODEL_NAME,
        }

    @classmethod
    def get_reranker_config(cls) -> dict:
        """获取 Reranker 配置"""
        return {
            "base_url": cls.RERANKER_BASE_URL,
            "api_key": cls.RERANKER_API_KEY,
            "model_name": cls.RERANKER_MODEL_NAME,
        }

    @classmethod
    def validate_config(cls) -> bool:
        """验证必要的配置是否已设置"""
        required_configs = []

        if not cls.LLM_API_KEY:
            required_configs.append("LLM_API_KEY")

        if not cls.EMBEDDING_API_KEY:
            required_configs.append("EMBEDDING_API_KEY")

        if required_configs:
            print(f"警告: 以下配置未设置: {', '.join(required_configs)}")
            return False

        return True


# 创建全局配置实例
config = EnvConfig()
