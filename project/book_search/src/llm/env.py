"""
环境配置模块
统一管理 API 调用所需的配置信息
"""

import os
from pathlib import Path
from typing import Optional


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    """解析单行 .env 内容，返回 (key, value)。"""
    raw = line.strip()
    if not raw or raw.startswith("#") or "=" not in raw:
        return None

    key, value = raw.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None

    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        value = value[1:-1]
    return key, value


def _load_dotenv_files() -> None:
    """加载常见 .env 文件到进程环境（不覆盖已存在变量）。"""
    current = Path(__file__).resolve()
    llm_dir = current.parent
    project_root = current.parents[2]

    dotenv_candidates = [
        project_root / ".env",
        project_root / ".env.local",
        llm_dir / ".env",
        llm_dir / ".env.local",
    ]

    for dotenv_file in dotenv_candidates:
        if not dotenv_file.exists():
            continue
        for line in dotenv_file.read_text(encoding="utf-8").splitlines():
            parsed = _parse_dotenv_line(line)
            if not parsed:
                continue
            key, value = parsed
            os.environ.setdefault(key, value)


_load_dotenv_files()


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
