"""基于润色后的小说资料生成 embedding 并入库。"""

from __future__ import annotations

import importlib
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.llm.embedding_client import EmbeddingClient
from src.process.book_profile_polish import POLISH_TABLE, ensure_polish_table

EMBED_TABLE = "book_polish_embedding"
FAISS_INDEX_SUFFIX = ".polish_embedding.faiss"
DATA_DIR_NAME = "data"
_faiss_module: Any = None


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _require_faiss() -> Any:
    global _faiss_module
    if _faiss_module is not None:
        return _faiss_module
    try:
        _faiss_module = importlib.import_module("faiss")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "未安装 faiss-cpu，请先安装依赖：pip install faiss-cpu"
        ) from exc
    return _faiss_module


def get_faiss_index_path(db_path: Path) -> Path:
    """根据数据库路径推导 Faiss 索引文件路径。"""
    return db_path.with_name(f"{db_path.stem}{FAISS_INDEX_SUFFIX}")


def resolve_data_db_path(db_path: Path) -> Path:
    """将数据库路径统一映射到 data 根目录。"""
    if db_path.parent.name == DATA_DIR_NAME:
        return db_path

    parts = list(db_path.parts)
    if DATA_DIR_NAME in parts:
        data_idx = parts.index(DATA_DIR_NAME)
        data_root = Path(*parts[: data_idx + 1])
        return data_root / db_path.name

    return db_path.parent / DATA_DIR_NAME / db_path.name


def resolve_existing_db_path(db_path: Path) -> Path:
    """优先使用 data 根目录数据库；若不存在则回退旧路径。"""
    data_db_path = resolve_data_db_path(db_path)
    if data_db_path.exists():
        return data_db_path
    return db_path


def _create_index(dim: int) -> Any:
    f = _require_faiss()
    return f.IndexIDMap2(f.IndexFlatIP(dim))


def load_or_create_faiss_index(index_path: Path, dim: int | None = None) -> Any:
    """读取已有索引；若不存在则按给定维度创建。"""
    f = _require_faiss()
    if index_path.exists():
        index = f.read_index(str(index_path))
        if not isinstance(index, f.IndexIDMap2):
            raise RuntimeError("Faiss 索引类型不受支持，期望 IndexIDMap2")
        if dim is not None and index.d != dim:
            raise ValueError(
                f"Faiss 索引维度不一致: index_dim={index.d}, current_dim={dim}"
            )
        return index

    if dim is None:
        raise ValueError("索引不存在时必须提供 embedding 维度")
    return _create_index(dim)


def _get_index_ids(index: Any) -> set[int]:
    f = _require_faiss()
    if index.ntotal == 0:
        return set()
    id_array = f.vector_to_array(index.id_map)
    return {int(x) for x in id_array.tolist()}


def _upsert_vector(
    index: Any,
    book_id: int,
    embedding: list[float],
    overwrite: bool,
) -> None:
    f = _require_faiss()
    ids = _get_index_ids(index)
    if book_id in ids:
        if not overwrite:
            return
        index.remove_ids(np.asarray([book_id], dtype=np.int64))

    vec = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
    f.normalize_L2(vec)
    index.add_with_ids(vec, np.asarray([book_id], dtype=np.int64))


def ensure_embedding_table(conn: sqlite3.Connection) -> None:
    """确保 embedding 元数据表存在。"""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {EMBED_TABLE} (
            book_id INTEGER PRIMARY KEY,
            text_content TEXT NOT NULL,
            embedding_dim INTEGER NOT NULL,
            model_name TEXT,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (book_id) REFERENCES books(book_id) ON DELETE CASCADE
        )
        """
    )
    columns = {
        row[1] for row in conn.execute(f"PRAGMA table_info({EMBED_TABLE})").fetchall()
    }
    if "text_content" not in columns:
        conn.execute(f"ALTER TABLE {EMBED_TABLE} ADD COLUMN text_content TEXT")
    if "embedding_dim" not in columns:
        conn.execute(f"ALTER TABLE {EMBED_TABLE} ADD COLUMN embedding_dim INTEGER")
    if "model_name" not in columns:
        conn.execute(f"ALTER TABLE {EMBED_TABLE} ADD COLUMN model_name TEXT")
    if "updated_at" not in columns:
        conn.execute(f"ALTER TABLE {EMBED_TABLE} ADD COLUMN updated_at TEXT")
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
    """基于 book_polish 表生成 embedding 并写入 Faiss + SQLite 元数据。"""

    # 读取数据
    active_db_path = resolve_existing_db_path(db_path)
    if not active_db_path.exists():
        raise FileNotFoundError(
            "数据库不存在: "
            f"{resolve_data_db_path(db_path)} "
            f"(也未找到旧路径: {db_path})"
        )
    f = _require_faiss()

    # 初始化 embedding 客户端
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
        index: Any = None
        index_dirty = False

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
            if index is not None and book_id in _get_index_ids(index):
                exists = True
            if exists and not overwrite:
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
            except Exception as exc:  # noqa: BLE001
                failed += 1
                processed += 1
                print(
                    (
                        f"[{idx}/{total}] embedding 失败: "
                        f"book_id={book_id} | error={exc}"
                    )
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
