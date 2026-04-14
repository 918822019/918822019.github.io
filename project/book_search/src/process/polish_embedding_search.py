"""基于润色 embedding 的相似书检索模块。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.llm.embedding_client import EmbeddingClient
from src.process.polish_embedding import EMBED_TABLE


def _load_embedding_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """读取 embedding 表并补齐展示字段。

    这里 LEFT JOIN books / book_polish 是为了在输出中携带更友好的信息：
    - 展示原始书名（books.title）
    - 展示润色书名（book_polish.polished_title）
    """
    return conn.execute(
        f"""
        SELECT
            e.book_id,
            e.text_content,
            e.embedding_json,
            e.embedding_dim,
            e.model_name,
            b.title AS source_title,
            p.polished_title,
            p.polished_intro
        FROM {EMBED_TABLE} e
        LEFT JOIN books b ON b.book_id = e.book_id
        LEFT JOIN book_polish p ON p.book_id = e.book_id
        ORDER BY e.book_id
        """
    ).fetchall()


def search_books_by_polish_embedding(
    db_path: Path,
    query: str,
    model_name: str | None,
    top_k: int,
) -> list[dict[str, Any]]:
    """基于润色 embedding 执行相似书检索。

    流程：
    1) 对用户查询生成 embedding
    2) 读取本地 embedding 表
    3) 计算余弦相似度并排序
    """
    if not db_path.exists():
        raise FileNotFoundError(f"数据库不存在: {db_path}")

    query_text = query.strip()
    if not query_text:
        raise ValueError("query 不能为空")

    client = EmbeddingClient(model_name=model_name)
    query_embedding = client.embed(query_text)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        rows = _load_embedding_rows(conn)
    finally:
        conn.close()

    scored: list[dict[str, Any]] = []
    for row in rows:
        # embedding 以 JSON 字符串落库，这里反序列化为向量。
        candidate = json.loads(str(row["embedding_json"]))
        score = client.cosine_similarity(query_embedding, candidate)

        scored.append(
            {
                "book_id": int(row["book_id"]),
                "score": float(score),
                "source_title": str(row["source_title"] or "").strip(),
                "polished_title": str(row["polished_title"] or "").strip(),
                "polished_intro": str(row["polished_intro"] or "").strip(),
                "embedding_model": str(row["model_name"] or "").strip(),
                "embedding_dim": int(row["embedding_dim"] or 0),
            }
        )

    # 按相似度降序取前 top_k
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[: max(top_k, 1)]
