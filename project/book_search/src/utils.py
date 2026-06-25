"""公共工具函数，消除跨模块重复代码。"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    """返回当前时间的 ISO 格式字符串（秒精度）。"""
    return datetime.now().isoformat(timespec="seconds")


def extract_json_block(text: str) -> dict[str, Any]:
    """从模型回复中提取第一个 JSON 对象。

    先尝试直接解析整个文本，失败后用正则提取 ``{...}`` 块。
    """
    payload = text.strip()
    if not payload:
        raise ValueError("模型返回为空")

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", payload)
    if not match:
        raise ValueError(f"未找到可解析 JSON: {payload[:120]}")
    return json.loads(match.group(0))


def load_books_from_path(input_path: str | Path) -> list[dict[str, Any]]:
    """从 JSON / NDJSON / SQLite 读取书籍数据。

    支持的格式：
    - ``.db``：从 ``books`` 表读取
    - ``.ndjson`` / ``.jsonl``：逐行 JSON
    - ``.json``：JSON 数组
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"输入文件不存在: {p}")

    suffix = p.suffix.lower()

    if suffix == ".db":
        conn = sqlite3.connect(str(p))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM books ORDER BY book_id").fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    if suffix in {".ndjson", ".jsonl"}:
        books: list[dict[str, Any]] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                books.append(record)
        return books

    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("输入 JSON 必须是数组，每个元素是一条小说记录")
    return [item for item in payload if isinstance(item, dict)]


def normalize_inline_text(value: Any) -> str:
    """将多行文本压缩为单行（替换全角空格、合并空白）。"""
    if value is None:
        return ""
    return " ".join(str(value).replace("\u3000", " ").split())
