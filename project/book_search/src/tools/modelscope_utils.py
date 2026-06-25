"""ModelScope 上传/下载共用工具函数。"""

from __future__ import annotations

import os


def resolve_token(cli_token: str = "") -> str:
    """优先用命令行传入的 token，否则读取环境变量。

    环境变量查找顺序：``MODELSCOPE_API_TOKEN`` → ``MODELSCOPE_TOKEN``
    """
    if cli_token and cli_token.strip():
        return cli_token.strip()

    for env_name in ("MODELSCOPE_API_TOKEN", "MODELSCOPE_TOKEN"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return ""


def login_modelscope(token: str) -> None:
    """登录 ModelScope，token 必填。

    Raises:
        RuntimeError: token 为空时抛出
    """
    if not token:
        raise RuntimeError(
            "缺少 ModelScope token。"
            "请传 --token，或设置 MODELSCOPE_API_TOKEN 环境变量。"
        )

    from modelscope.hub.api import HubApi  # type: ignore[import-untyped]

    api = HubApi()
    login_func = getattr(api, "login")
    try:
        login_func(token)
    except TypeError:
        login_func(access_token=token)
