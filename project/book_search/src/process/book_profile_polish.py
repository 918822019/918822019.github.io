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
CHAPTER_PREVIEW_COUNT = 5
CHAPTER_PREVIEW_MAX_CHARS = 1200


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


def normalize_text(text: str, max_chars: int | None = None) -> str:
    """规范化文本，避免提示词中出现过长或过碎的内容。"""
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if max_chars is not None and max_chars > 0 and len(normalized) > max_chars:
        return f"{normalized[:max_chars].rstrip()}..."
    return normalized


def fetch_chapter_previews(
    conn: sqlite3.Connection,
    book_id: int,
    limit: int = CHAPTER_PREVIEW_COUNT,
) -> list[dict[str, str]]:
    """读取前若干章正文预览，用于辅助简介润色。"""
    rows = conn.execute(
        """
        SELECT chapter_id, chapter_name, content
        FROM chapters
        WHERE book_id = ?
          AND is_content_fetched = 1
          AND COALESCE(TRIM(content), '') <> ''
        ORDER BY chapter_id
        LIMIT ?
        """,
        (book_id, limit),
    ).fetchall()

    previews: list[dict[str, str]] = []
    for row in rows:
        chapter_name = normalize_text(str(row["chapter_name"] or "")) or "未命名章节"
        content = normalize_text(
            str(row["content"] or ""),
            max_chars=CHAPTER_PREVIEW_MAX_CHARS,
        )
        if not content:
            continue
        previews.append(
            {
                "chapter_name": chapter_name,
                "content": content,
            }
        )
    return previews


def format_chapter_previews(chapter_previews: list[dict[str, str]]) -> str:
    """将章节预览拼接成提示词文本。"""
    if not chapter_previews:
        return "（未提供章节正文，仅可参考原简介与书名）"

    lines: list[str] = []
    for idx, chapter in enumerate(chapter_previews, start=1):
        lines.append(f"第{idx}章：{chapter['chapter_name']}")
        lines.append(chapter["content"])
    return "\n".join(lines)


def build_prompt(
    title: str,
    intro: str,
    chapter_previews: list[dict[str, str]],
) -> str:
    """构建润色提示词。"""
    chapter_context = format_chapter_previews(chapter_previews)
    return f"""
【角色设定】
你是一位资深的出版编辑，擅长在不改变作者原意的前提下，提炼小说的核心钩子。你需要基于已有的正文内容反向优化简介，确保简介与正文前五章的基调、人设、冲突完全一致。

【核心任务】
根据提供的信息，润色小说的简介（polished_intro），并清理书名的空白（polished_title）。

【正文依据（必读）】
{chapter_context}

【待处理信息】
- 书名：{title or '未知书名'}
- 原简介：{intro or '（无简介）'}

【执行约束】
1. **输出格式**：仅输出标准 JSON，严禁包含任何解释性前缀、后缀或 Markdown 代码块标记。
2. **JSON 结构**：{{"polished_title": "...", "polished_intro": "..."}}
3. **事实锁定**：严禁修改世界观、角色姓名、身份关系及已发生的关键剧情事实。
4. **书名处理**：仅做全角/半角空格清理，严禁修改书名文字本身。
5. **简介字数**：严格控制在 60-180 字之间。
6. **内容边界**：
   - 必须基于前五章已出现的冲突或线索（避免写飞）。
   - 严禁出现营销体（如“爆款”、“震撼来袭”）、严禁低俗擦边、严禁剧透五章后的结局。
7. **文风一致性**：保持与正文前五章相同的叙事人称（如第三人称/第一人称）和文字气质。

【请直接返回 JSON】:
""".strip()


def polish_one_book(
    llm: LLMClient,
    title: str,
    intro: str,
    chapter_previews: list[dict[str, str]],
) -> tuple[str, str]:
    """调用 LLM 润色单本书。"""
    system_prompt = "你是中文小说文案编辑，擅长在不改动事实的前提下优化标题和简介。"
    response = llm.generate(
        prompt=build_prompt(
            title=title,
            intro=intro,
            chapter_previews=chapter_previews,
        ),
        system_prompt=system_prompt,
        temperature=0.3,
    )
    payload = extract_json_block(response)

    polished_title = normalize_text(str(payload.get("polished_title", "")))
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
            chapter_previews = fetch_chapter_previews(
                conn=conn,
                book_id=book_id,
            )

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
                    chapter_previews=chapter_previews,
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
