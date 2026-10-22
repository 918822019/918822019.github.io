"""数据库表管理内部模块"""

from __future__ import annotations

import sqlite3

from src.config import config

POLISH_TABLE = config.polish.polish_table
EMBED_TABLE = config.polish.embed_table
CHAPTER_PREVIEW_COUNT = config.polish.chapter_preview_count
CHAPTER_PREVIEW_MAX_CHARS = config.polish.chapter_preview_max_chars
FAISS_INDEX_SUFFIX = config.data.faiss_index_suffix
DATA_DIR_NAME = config.data.dir


def ensure_polish_table(conn: sqlite3.Connection) -> None:
    """创建书评润色结果表（如不存在）"""
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


def ensure_embedding_table(conn: sqlite3.Connection) -> None:
    """创建 embedding 结果表（如不存在），并补全缺失列"""
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
