"""Embedding 生成模块"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from src.config import config
from src.llm.client import EmbeddingClient
from src.utils import now_iso

from src.process.polish._db import EMBED_TABLE, POLISH_TABLE, ensure_embedding_table, ensure_polish_table
from src.process.polish._faiss import (
    _get_index_ids,
    _invalidate_id_cache,
    _require_faiss,
    _upsert_vector,
    get_faiss_index_path,
    load_or_create_faiss_index,
    resolve_existing_db_path,
    resolve_data_db_path,
)

DATA_DIR_NAME = config.data.dir


def build_embedding_text(polished_title: str, polished_intro: str) -> str:
    """构建用于生成 embedding 的文本"""
    return f"书名：{polished_title.strip()}\n简介：{polished_intro.strip()}".strip()


def run_polish_embedding(
    db_path: Path,
    model_name: str | None,
    limit: int,
    sleep_seconds: float,
    overwrite: bool,
) -> dict[str, int]:
    """批量生成润色文本的 embedding 并入库"""
    active_db_path = resolve_existing_db_path(db_path)
    if not active_db_path.exists():
        raise FileNotFoundError(
            "数据库不存在: "
            f"{resolve_data_db_path(db_path)} "
            f"(也未找到旧路径: {db_path})"
        )
    f = _require_faiss()

    client = EmbeddingClient(model_name=model_name)
    conn = sqlite3.connect(str(active_db_path))
    conn.row_factory = sqlite3.Row
    index_path = get_faiss_index_path(active_db_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

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
        index_dirty = False

        # 预加载 Faiss 索引（如果存在），避免每轮重复加载
        index = load_or_create_faiss_index(index_path, dim=0) if index_path.exists() else None
        if index is not None:
            embedded_ids = _get_index_ids(index)
        else:
            embedded_ids = set()

        if not overwrite:
            # 从 DB 中读取已存在的 book_id，避免重复查库
            existing_rows = conn.execute(
                f"SELECT book_id FROM {EMBED_TABLE}"
            ).fetchall()
            db_embedded_ids = {int(r["book_id"]) for r in existing_rows}
            embedded_ids.update(db_embedded_ids)

        for idx, row in enumerate(rows, start=1):
            if limit > 0 and processed >= limit:
                break

            book_id = int(row["book_id"])
            polished_title = str(row["polished_title"] or "").strip()
            polished_intro = str(row["polished_intro"] or "").strip()
            text_content = build_embedding_text(polished_title, polished_intro)

            if not overwrite and book_id in embedded_ids:
                skipped += 1
                continue

            try:
                embedding = client.embed(text_content)
                if index is None:
                    index = load_or_create_faiss_index(index_path, dim=len(embedding))
                _upsert_vector(
                    index=index,
                    book_id=book_id,
                    embedding=embedding,
                    overwrite=overwrite,
                )
                _invalidate_id_cache(index)

                conn.execute(
                    f"""
                    INSERT INTO {EMBED_TABLE} (
                        book_id,
                        text_content,
                        embedding_dim,
                        model_name,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(book_id) DO UPDATE SET
                        text_content = excluded.text_content,
                        embedding_dim = excluded.embedding_dim,
                        model_name = excluded.model_name,
                        updated_at = excluded.updated_at
                    """,
                    (
                        book_id,
                        text_content,
                        len(embedding),
                        client.model_name,
                        now_iso(),
                    ),
                )
                conn.commit()
                index_dirty = True

                changed += 1
                processed += 1
                print(f"[{idx}/{total}] 已写入 embedding: book_id={book_id}")
            except Exception as exc:
                failed += 1
                processed += 1
                print(
                    f"[{idx}/{total}] embedding 失败: "
                    f"book_id={book_id} | error={exc}"
                )

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        if index_dirty and index is not None:
            f.write_index(index, str(index_path))

        return {
            "total": total,
            "processed": processed,
            "changed": changed,
            "skipped": skipped,
            "failed": failed,
        }
    finally:
        conn.close()



