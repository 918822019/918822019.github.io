"""基于润色后的小说资料生成 embedding 并入库。"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from src.llm.embedding_client import EmbeddingClient
from src.process.book_profile_polish import POLISH_TABLE, ensure_polish_table

EMBED_TABLE = "book_polish_embedding"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_embedding_table(conn: sqlite3.Connection) -> None:
    """确保 embedding 结果表存在。"""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {EMBED_TABLE} (
            book_id INTEGER PRIMARY KEY,
            text_content TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            embedding_dim INTEGER NOT NULL,
            model_name TEXT,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (book_id) REFERENCES books(book_id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()


def build_embedding_text(polished_title: str, polished_intro: str) -> str:
    """构建用于 embedding 的文本。"""
    return f"书名：{polished_title.strip()}\n简介：{polished_intro.strip()}".strip()


def run_polish_embedding(
    db_path: Path,
    model_name: str | None,
    limit: int,
    sleep_seconds: float,
    overwrite: bool,
) -> dict[str, int]:
    """基于 book_polish 表生成 embedding 并写入。"""
    if not db_path.exists():
        raise FileNotFoundError(f"数据库不存在: {db_path}")

    client = EmbeddingClient(model_name=model_name)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        ensure_polish_table(conn)
        ensure_embedding_table(conn)

        rows = conn.execute(
            f"""
            SELECT book_id, polished_title, polished_intro
            FROM {POLISH_TABLE}
            ORDER BY book_id
            """
        ).fetchall()

        total = len(rows)
        processed = 0
        changed = 0
        skipped = 0
        failed = 0

        for idx, row in enumerate(rows, start=1):
            if limit > 0 and processed >= limit:
                break

            book_id = int(row["book_id"])
            polished_title = str(row["polished_title"] or "").strip()
            polished_intro = str(row["polished_intro"] or "").strip()
            text_content = build_embedding_text(polished_title, polished_intro)

            exists = conn.execute(
                f"SELECT 1 FROM {EMBED_TABLE} WHERE book_id = ?",
                (book_id,),
            ).fetchone()
            if exists and not overwrite:
                skipped += 1
                continue

            try:
                embedding = client.embed(text_content)
                conn.execute(
                    f"""
                    INSERT INTO {EMBED_TABLE} (
                        book_id,
                        text_content,
                        embedding_json,
                        embedding_dim,
                        model_name,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(book_id) DO UPDATE SET
                        text_content = excluded.text_content,
                        embedding_json = excluded.embedding_json,
                        embedding_dim = excluded.embedding_dim,
                        model_name = excluded.model_name,
                        updated_at = excluded.updated_at
                    """,
                    (
                        book_id,
                        text_content,
                        json.dumps(embedding, ensure_ascii=False),
                        len(embedding),
                        client.model_name,
                        now_iso(),
                    ),
                )
                conn.commit()

                changed += 1
                processed += 1
                print(f"[{idx}/{total}] 已写入 embedding: book_id={book_id}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                processed += 1
                print(
                    f"[{idx}/{total}] embedding 失败: book_id={book_id} | error={exc}"
                )

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
