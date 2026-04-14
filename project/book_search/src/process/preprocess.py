"""预处理流程编排模块。

该模块负责把预处理步骤组织成可复用的 pipeline，当前已集成：
- 文本润色步骤（基于前五章正文的简介润色）
- 润色文本 embedding 步骤
- LLM 标签步骤（支持扁平标签与级联标签）

说明：
- 本模块只提供函数接口，不提供命令行入口。
- 其他服务代码可直接 import 并调用 `run_preprocess_pipeline`。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union, cast

from src.process.book_profile_polish import run_polish
from src.process.llm_tagging import CascadingTagger, LLMTagger
from src.process.polish_embedding import run_polish_embedding

TaggingMode = Literal["flat", "cascading"]


@dataclass(slots=True)
class PreprocessPipelineConfig:
    """预处理 pipeline 配置。"""

    input_path: Union[str, Path]  # 输入数据库文件路径
    output_path: Union[str, Path]  # 输出数据库文件路径
    enable_text_polish: bool = True  # 是否启用文本润色（基于前五章正文的简介润色）
    enable_polish_embedding: bool = True  # 是否启用润色文本的 embedding 生成
    enable_llm_tagging: bool = True  # 是否启用 LLM 标签生成
    polish_model_name: Optional[str] = (
        None  # 文本润色使用的模型名称，None 则使用默认模型
    )
    embedding_model_name: Optional[str] = (
        None  # embedding 生成使用的模型名称，None 则使用默认模型
    )
    tagging_mode: TaggingMode = (
        "flat"  # 标签生成模式："flat"（扁平标签）或 "cascading"（级联标签）
    )
    model_name: Optional[str] = None  # LLM 标签生成使用的模型名称，None 则使用默认模型
    max_tags: int = 8  # 每本书最多生成的标签数量（仅 flat 模式有效）
    sleep_seconds: float = 0.0  # 每次 API 调用后的休眠时间（秒），用于控制请求频率
    overwrite: bool = False  # 是否覆盖已存在的数据
    limit: int = 0  # 处理的最大书籍数量，0 表示处理全部书籍
    incremental_tagging: bool = True  # 是否启用标签增量模式（默认跳过已打标记录）


def _book_id(book: dict[str, Any]) -> int | None:
    """从记录中提取 book_id。"""
    raw_id = book.get("book_id")
    if raw_id is None:
        return None
    try:
        return int(raw_id)
    except (TypeError, ValueError):
        return None


def _load_existing_tagging_result(
    output_path: Path,
) -> dict[int, dict[str, Any]]:
    """读取已有标签输出，构建 book_id -> record 映射。"""
    if not output_path.exists():
        return {}

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}

    if not isinstance(payload, list):
        return {}

    payload_list = cast(list[Any], payload)
    result: dict[int, dict[str, Any]] = {}
    for item in payload_list:
        if not isinstance(item, dict):
            continue
        record = cast(dict[str, Any], item)
        bid = _book_id(record)
        if bid is None:
            continue
        result[bid] = record
    return result


def _merge_existing_tags(
    current_books: list[dict[str, Any]],
    existing_books_by_id: dict[int, dict[str, Any]],
) -> int:
    """把已有输出里的 tags/cascaded_tags 合并到当前数据，便于增量跳过。"""
    merged = 0
    for book in current_books:
        bid = _book_id(book)
        if bid is None:
            continue

        old = existing_books_by_id.get(bid)
        if not old:
            continue

        if "tags" in old and "tags" not in book:
            book["tags"] = old["tags"]
            merged += 1

        if "cascaded_tags" in old and "cascaded_tags" not in book:
            book["cascaded_tags"] = old["cascaded_tags"]
            merged += 1

    return merged


def _prepare_books_for_incremental_tagging(
    input_path: Path,
    output_path: Path,
) -> tuple[list[dict[str, Any]], int]:
    """加载输入数据并回填旧标签，供增量打标使用。"""
    current_books = _load_books_for_tagging(input_path)
    existing_books_by_id = _load_existing_tagging_result(output_path)
    merged = _merge_existing_tags(current_books, existing_books_by_id)
    return current_books, merged


def _load_books_for_tagging(input_path: Path) -> list[dict[str, Any]]:
    """从 JSON / NDJSON / SQLite 读取书籍数据（仅用于 pipeline 内部）。"""
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".db":
        conn = sqlite3.connect(str(input_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM books ORDER BY book_id").fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    if suffix in {".ndjson", ".jsonl"}:
        books: list[dict[str, Any]] = []
        for line in input_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                books.append(cast(dict[str, Any], record))
        return books

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("输入 JSON 必须是数组，每个元素是一条小说记录")
    payload_list = cast(list[Any], payload)
    output_books: list[dict[str, Any]] = []
    for item in payload_list:
        if isinstance(item, dict):
            output_books.append(cast(dict[str, Any], item))
    return output_books


def run_preprocess_pipeline(
    config: PreprocessPipelineConfig,
) -> dict[str, Any]:
    """执行预处理 pipeline。

    Args:
            config: pipeline 配置对象

    Returns:
            流程统计信息
    """

    # 输入数据
    input_path = Path(config.input_path)
    output_path = Path(config.output_path)

    steps: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "steps": steps,
    }

    if config.enable_text_polish:
        # 开始进行润色
        polish_stats = run_polish(
            db_path=input_path,
            model_name=config.polish_model_name,
            limit=max(config.limit, 0),
            sleep_seconds=max(config.sleep_seconds, 0.0),
            overwrite=config.overwrite,
        )
        steps.append(
            {
                "name": "text_polish",
                "enabled": True,
                "stats": polish_stats,
            }
        )
    else:
        steps.append(
            {
                "name": "text_polish",
                "enabled": False,
                "stats": None,
            }
        )

    if config.enable_polish_embedding:
        # 开始润色的embedding生成入库
        embed_stats = run_polish_embedding(
            db_path=input_path,
            model_name=config.embedding_model_name,
            limit=max(config.limit, 0),
            sleep_seconds=max(config.sleep_seconds, 0.0),
            overwrite=config.overwrite,
        )
        steps.append(
            {
                "name": "polish_embedding",
                "enabled": True,
                "stats": embed_stats,
            }
        )
    else:
        steps.append(
            {
                "name": "polish_embedding",
                "enabled": False,
                "stats": None,
            }
        )
    # 进入llm标签打标逻辑
    if not config.enable_llm_tagging:
        steps.append(
            {
                "name": "llm_tagging",
                "enabled": False,
                "stats": None,
            }
        )
        return result

    if config.tagging_mode == "cascading":
        tagger = CascadingTagger(
            model_name=config.model_name,
            sleep_seconds=max(config.sleep_seconds, 0.0),
        )
        merged = 0
        if config.incremental_tagging and not config.overwrite:
            books, merged = _prepare_books_for_incremental_tagging(
                input_path=input_path,
                output_path=output_path,
            )
            tagger.books = books
            stats = tagger.run(output_path=output_path)
        else:
            stats = tagger.run(
                input_path=input_path,
                output_path=output_path,
            )
    else:
        merged = 0
        tagger = LLMTagger(
            model_name=config.model_name,
            max_tags=max(config.max_tags, 1),
            sleep_seconds=max(config.sleep_seconds, 0.0),
            overwrite=config.overwrite,
            limit=max(config.limit, 0),
        )
        if config.incremental_tagging and not config.overwrite:
            books, merged = _prepare_books_for_incremental_tagging(
                input_path=input_path,
                output_path=output_path,
            )
            tagger.books = books
            stats = tagger.run(output_path=output_path)
        else:
            stats = tagger.run(
                input_path=input_path,
                output_path=output_path,
            )

    stats = {
        **stats,
        "merged_existing_tag_fields": merged,
    }

    steps.append(
        {
            "name": "llm_tagging",
            "enabled": True,
            "mode": config.tagging_mode,
            "stats": stats,
        }
    )
    return result
