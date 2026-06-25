"""小说简介润色、Embedding 生成与语义搜索整合模块。"""

from src.process.polish._db import (
    CHAPTER_PREVIEW_COUNT,
    CHAPTER_PREVIEW_MAX_CHARS,
    DATA_DIR_NAME,
    EMBED_TABLE,
    FAISS_INDEX_SUFFIX,
    POLISH_TABLE,
    ensure_embedding_table,
    ensure_polish_table,
)
from src.process.polish._faiss import (
    _require_faiss,
    _upsert_vector,
    get_faiss_index_path,
    load_or_create_faiss_index,
    resolve_data_db_path,
    resolve_existing_db_path,
)
from src.process.polish.core import (
    build_prompt,
    fetch_chapter_previews,
    format_chapter_previews,
    normalize_text,
    polish_one_book,
    run_polish,
)
from src.process.polish.embedding import build_embedding_text, run_polish_embedding
from src.process.polish.search import search_books_by_polish_embedding

__all__ = [
    "POLISH_TABLE",
    "EMBED_TABLE",
    "CHAPTER_PREVIEW_COUNT",
    "CHAPTER_PREVIEW_MAX_CHARS",
    "FAISS_INDEX_SUFFIX",
    "DATA_DIR_NAME",
    "ensure_polish_table",
    "ensure_embedding_table",
    "get_faiss_index_path",
    "load_or_create_faiss_index",
    "resolve_data_db_path",
    "resolve_existing_db_path",
    "_require_faiss",
    "_upsert_vector",
    "normalize_text",
    "fetch_chapter_previews",
    "format_chapter_previews",
    "build_prompt",
    "polish_one_book",
    "run_polish",
    "build_embedding_text",
    "run_polish_embedding",
    "search_books_by_polish_embedding",
]
