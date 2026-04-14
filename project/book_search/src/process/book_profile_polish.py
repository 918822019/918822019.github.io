"""小说书名与简介润色处理模块。"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.llm.llm_client import LLMClient


POLISH_TABLE = "book_polish"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def extract_json_block(text: str) -> dict[str, Any]:
    """从模型回复中提取第一个 JSON 对象。"""
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


def ensure_polish_table(conn: sqlite3.Connection) -> None:
    """确保润色结果表存在。"""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {POLISH_TABLE} (
            book_id INTEGER PRIMARY KEY,
            source_title TEXT,
            source_intro TEXT,
            polished_title TEXT NOT NULL,
            polished_intro TEXT NOT NULL,
            model_name TEXT,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (book_id) REFERENCES books(book_id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()


def build_prompt(title: str, intro: str) -> str:
    """构建润色提示词。"""
    return f"""
请润色下面小说的书名和简介，让表达更自然、更有吸引力。

要求：
1. 只返回 JSON，不要输出额外文字。
2. JSON 格式必须为：{{"polished_title": "...", "polished_intro": "..."}}。
3. 不要改变核心设定、人物关系、时代背景和关键事实。
4. 书名长度建议 2-16 字；简介长度建议 40-180 字。
5. 避免夸张营销词、避免低俗词、避免加入不存在的信息。

原始信息：
- 书名：{title or '未知书名'}
- 简介：{intro or '（无简介）'}
""".strip()


def polish_one_book(llm: LLMClient, title: str, intro: str) -> tuple[str, str]:
    """调用 LLM 润色单本书。"""
    system_prompt = "你是中文小说文案编辑，擅长在不改动事实的前提下优化标题和简介。"
    response = llm.generate(
        prompt=build_prompt(title=title, intro=intro),
        system_prompt=system_prompt,
        temperature=0.3,
    )
    payload = extract_json_block(response)

    polished_title = str(payload.get("polished_title", "")).strip()
    polished_intro = str(payload.get("polished_intro", "")).strip()

    if not polished_title:
        polished_title = title.strip() or "未知书名"
    if not polished_intro:
        polished_intro = intro.strip()

    return polished_title, polished_intro


def run_polish(
    db_path: Path,
    model_name: str | None,
    limit: int,
    sleep_seconds: float,
    overwrite: bool,
) -> dict[str, int]:
    """执行批量润色。"""
    if not db_path.exists():
        raise FileNotFoundError(f"数据库不存在: {db_path}")

    llm = LLMClient(model_name=model_name)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        ensure_polish_table(conn)

        total = int(conn.execute("SELECT COUNT(*) AS c FROM books").fetchone()["c"])
        changed = 0
        skipped = 0
        failed = 0
        processed = 0

        rows = conn.execute(
            """
            SELECT book_id, title, intro
            FROM books
            ORDER BY book_id
            """
        ).fetchall()

        for idx, row in enumerate(rows, start=1):
            if limit > 0 and processed >= limit:
                break

            book_id = int(row["book_id"])
            title = str(row["title"] or "").strip()
            intro = str(row["intro"] or "").strip()

            exists = conn.execute(
                f"SELECT 1 FROM {POLISH_TABLE} WHERE book_id = ?",
                (book_id,),
            ).fetchone()
            if exists and not overwrite:
                skipped += 1
                continue

            try:
                polished_title, polished_intro = polish_one_book(
                    llm=llm,
                    title=title,
                    intro=intro,
                )
                conn.execute(
                    f"""
                    INSERT INTO {POLISH_TABLE} (
                        book_id,
                        source_title,
                        source_intro,
                        polished_title,
                        polished_intro,
                        model_name,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(book_id) DO UPDATE SET
                        source_title = excluded.source_title,
                        source_intro = excluded.source_intro,
                        polished_title = excluded.polished_title,
                        polished_intro = excluded.polished_intro,
                        model_name = excluded.model_name,
                        updated_at = excluded.updated_at
                    """,
                    (
                        book_id,
                        title,
                        intro,
                        polished_title,
                        polished_intro,
                        llm.model_name,
                        now_iso(),
                    ),
                )
                conn.commit()

                changed += 1
                processed += 1
                print(f"[{idx}/{total}] 已润色: {title or '未知书名'}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                processed += 1
                print(f"[{idx}/{total}] 处理失败: {title or '未知书名'} | error={exc}")

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        return {
            "total": total,
            "processed": processed,
            "changed": changed,
            "skipped": skipped,
            "failed": failed,
        }
    finally:
        conn.close()
