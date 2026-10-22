"""语义搜索模块"""

from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from src.llm.client import EmbeddingClient

from src.process.polish._db import EMBED_TABLE, POLISH_TABLE
from src.process.polish._faiss import (
    get_faiss_index_path,
    load_or_create_faiss_index,
    resolve_existing_db_path,
    resolve_data_db_path,
)


def _load_metadata_by_ids(
    conn: sqlite3.Connection, ordered_book_ids: list[int]
) -> dict[int, sqlite3.Row]:
    """根据 book_id 列表从数据库批量加载元数据"""
    if not ordered_book_ids:
        return {}

    placeholders = ",".join("?" for _ in ordered_book_ids)
    rows = conn.execute(
        f"""
        SELECT
            e.book_id,
            e.embedding_dim,
            e.model_name,
            b.title AS source_title,
            p.polished_title,
            p.polished_intro
        FROM {EMBED_TABLE} e
        LEFT JOIN books b ON b.book_id = e.book_id
        LEFT JOIN book_polish p ON p.book_id = e.book_id
        WHERE e.book_id IN ({placeholders})
        """,
        ordered_book_ids,
    ).fetchall()
    return {int(row["book_id"]): row for row in rows}


def search_books_by_polish_embedding(
    db_path: Path,
    query: str,
    model_name: str | None,
    top_k: int,
) -> list[dict[str, Any]]:
    """基于润色 embedding 的语义搜索"""
    active_db_path = resolve_existing_db_path(db_path)
    if not active_db_path.exists():
        raise FileNotFoundError(
            "数据库不存在: "
            f"{resolve_data_db_path(db_path)} "
            f"(也未找到旧路径: {db_path})"
        )
    query_text = query.strip()
    if not query_text:
        raise ValueError("query 不能为空")

    client = EmbeddingClient(model_name=model_name)
    query_embedding = np.asarray(client.embed(query_text), dtype=np.float32).reshape(
        1, -1
    )
    index_path = get_faiss_index_path(active_db_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Faiss 索引不存在: {index_path}")

    index = load_or_create_faiss_index(index_path)
    if index.ntotal == 0:
        return []
    if index.d != query_embedding.shape[1]:
        raise ValueError(
            "查询向量维度与索引不一致: "
            f"query_dim={query_embedding.shape[1]}, index_dim={index.d}"
        )
    np_query = query_embedding.astype(np.float32, copy=True)
    if np_query.size == 0:
        return []
    faiss = importlib.import_module("faiss")
    faiss.normalize_L2(np_query)
    distances, indices = index.search(np_query, max(top_k, 1))
    candidate_ids = [int(x) for x in indices[0].tolist() if int(x) >= 0]
    if not candidate_ids:
        return []

    conn = sqlite3.connect(str(active_db_path))
    conn.row_factory = sqlite3.Row

    try:
        metadata_by_id = _load_metadata_by_ids(conn, candidate_ids)
    finally:
        conn.close()

    scored: list[dict[str, Any]] = []
    for i, book_id in enumerate(candidate_ids):
        row = metadata_by_id.get(book_id)
        if row is None:
            continue
        scored.append(
            {
                "book_id": book_id,
                "score": float(distances[0][i]),
                "source_title": str(row["source_title"] or "").strip(),
                "polished_title": str(row["polished_title"] or "").strip(),
                "polished_intro": str(row["polished_intro"] or "").strip(),
                "embedding_model": str(row["model_name"] or "").strip(),
                "embedding_dim": int(row["embedding_dim"] or 0),
            }
        )

    return scored



