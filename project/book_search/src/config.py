"""
配置管理模块
从 .env 加载密钥，从 config.yaml 加载参数（环境变量优先）
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


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
    src_dir = current.parent
    project_root = current.parents[1]
    llm_dir = src_dir / "llm"

    dotenv_candidates = [
        project_root / ".env",
        project_root / ".env.local",
        llm_dir / ".env",
        llm_dir / ".env.local",
        src_dir / ".env",
        src_dir / ".env.local",
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


def _find_config_yaml() -> Path | None:
    """在项目根目录查找 config.yaml。"""
    current = Path(__file__).resolve()
    root = current.parents[1]
    path = root / "config.yaml"
    return path if path.exists() else None


def _load_yaml() -> dict[str, Any]:
    """加载 config.yaml，若不存在或出错则返回空字典。"""
    path = _find_config_yaml()
    if not path:
        return {}
    try:
        import yaml
    except ModuleNotFoundError:
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class _ConfigView:
    """嵌套配置视图，支持属性访问字典层级。"""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        val = self._data.get(name)
        if isinstance(val, dict):
            return _ConfigView(val)
        return val

    def get(self, name: str, default: Any = None) -> Any:
        try:
            return getattr(self, name)
        except AttributeError:
            return default


class Config(_ConfigView):
    """应用程序统一配置入口，合并 YAML 配置与环境变量。"""

    def __init__(self) -> None:
        raw = _load_yaml()
        super().__init__(raw)
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """环境变量覆盖 YAML 中的对应值（.env 优先级最高）。"""

        # ---- LLM ----
        self._data.setdefault("llm", {})
        self._data["llm"]["api_key"] = os.getenv(
            "LLM_API_KEY", self._data["llm"].get("api_key", "")
        )
        self._data["llm"]["base_url"] = os.getenv(
            "LLM_BASE_URL",
            self._data["llm"].get("base_url", "https://api.openai.com/v1"),
        )
        self._data["llm"]["model_name"] = os.getenv(
            "LLM_MODEL_NAME", self._data["llm"].get("model_name", "gpt-3.5-turbo")
        )

        # ---- Embedding ----
        self._data.setdefault("embedding", {})
        self._data["embedding"]["api_key"] = os.getenv(
            "EMBEDDING_API_KEY", self._data["embedding"].get("api_key", "")
        )
        self._data["embedding"]["base_url"] = os.getenv(
            "EMBEDDING_BASE_URL",
            self._data["embedding"].get("base_url", "https://api.openai.com/v1"),
        )
        self._data["embedding"]["model_name"] = os.getenv(
            "EMBEDDING_MODEL_NAME",
            self._data["embedding"].get("model_name", "text-embedding-ada-002"),
        )

        # ---- Reranker ----
        self._data.setdefault("reranker", {})
        self._data["reranker"]["api_key"] = os.getenv(
            "RERANKER_API_KEY", self._data["reranker"].get("api_key", "")
        )
        self._data["reranker"]["base_url"] = os.getenv(
            "RERANKER_BASE_URL", self._data["reranker"].get("base_url", "")
        )
        self._data["reranker"]["model_name"] = os.getenv(
            "RERANKER_MODEL_NAME",
            self._data["reranker"].get("model_name", "bge-reranker-base"),
        )

        # ---- Request ----
        self._data.setdefault("request", {})
        self._data["request"]["timeout"] = int(
            os.getenv(
                "REQUEST_TIMEOUT",
                str(self._data["request"].get("timeout", "30")),
            )
        )
        self._data["request"]["max_retries"] = int(
            os.getenv(
                "MAX_RETRIES",
                str(self._data["request"].get("max_retries", "3")),
            )
        )

    # ---- 向后兼容：扁平大写访问器 ----

    @property
    def LLM_API_KEY(self) -> str:
        return self.llm.api_key  # type: ignore[union-attr]

    @property
    def LLM_BASE_URL(self) -> str:
        return self.llm.base_url  # type: ignore[union-attr]

    @property
    def LLM_MODEL_NAME(self) -> str:
        return self.llm.model_name  # type: ignore[union-attr]

    @property
    def EMBEDDING_API_KEY(self) -> str:
        return self.embedding.api_key  # type: ignore[union-attr]

    @property
    def EMBEDDING_BASE_URL(self) -> str:
        return self.embedding.base_url  # type: ignore[union-attr]

    @property
    def EMBEDDING_MODEL_NAME(self) -> str:
        return self.embedding.model_name  # type: ignore[union-attr]

    @property
    def RERANKER_API_KEY(self) -> str:
        return self.reranker.api_key  # type: ignore[union-attr]

    @property
    def RERANKER_BASE_URL(self) -> str:
        return self.reranker.base_url  # type: ignore[union-attr]

    @property
    def RERANKER_MODEL_NAME(self) -> str:
        return self.reranker.model_name  # type: ignore[union-attr]

    @property
    def REQUEST_TIMEOUT(self) -> int:
        return self.request.timeout  # type: ignore[union-attr]

    @property
    def MAX_RETRIES(self) -> int:
        return self.request.max_retries  # type: ignore[union-attr]

    # ---- 向后兼容：类方法代理 ----

    @classmethod
    def get_llm_config(cls) -> dict:
        return config._get_llm_config()

    @classmethod
    def get_embedding_config(cls) -> dict:
        return config._get_embedding_config()

    @classmethod
    def get_reranker_config(cls) -> dict:
        return config._get_reranker_config()

    @classmethod
    def validate_config(cls) -> bool:
        return config._validate_config()

    # ---- 实例方法（供 config.xxx() 调用） ----

    def _get_llm_config(self) -> dict:
        return {
            "base_url": self.LLM_BASE_URL,
            "api_key": self.LLM_API_KEY,
            "model_name": self.LLM_MODEL_NAME,
        }

    def _get_embedding_config(self) -> dict:
        return {
            "base_url": self.EMBEDDING_BASE_URL,
            "api_key": self.EMBEDDING_API_KEY,
            "model_name": self.EMBEDDING_MODEL_NAME,
        }

    def _get_reranker_config(self) -> dict:
        return {
            "base_url": self.RERANKER_BASE_URL,
            "api_key": self.RERANKER_API_KEY,
            "model_name": self.RERANKER_MODEL_NAME,
        }

    def _validate_config(self) -> bool:
        required = []
        if not self.LLM_API_KEY:
            required.append("LLM_API_KEY")
        if not self.EMBEDDING_API_KEY:
            required.append("EMBEDDING_API_KEY")
        if required:
            print(f"警告: 以下配置未设置: {', '.join(required)}")
            return False
        return True

    def __getattr__(self, name: str):
        """将 get_llm_config / validate_config 等转发到实例方法。"""
        if name == "get_llm_config":
            return self._get_llm_config
        if name == "get_embedding_config":
            return self._get_embedding_config
        if name == "get_reranker_config":
            return self._get_reranker_config
        if name == "validate_config":
            return self._validate_config
        return super().__getattr__(name)


# ---- 兼容旧版 EnvConfig ----
class EnvConfig:
    """向后兼容包装，供测试 test_llm.py 引用。"""
    get_llm_config = Config.get_llm_config
    get_embedding_config = Config.get_embedding_config
    get_reranker_config = Config.get_reranker_config
    validate_config = Config.validate_config


config = Config()
