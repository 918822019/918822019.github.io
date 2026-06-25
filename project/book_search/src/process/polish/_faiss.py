"""Faiss 索引操作内部工具模块"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np

from src.config import config

FAISS_INDEX_SUFFIX = config.data.faiss_index_suffix
DATA_DIR_NAME = config.data.dir

_faiss_module: Any = None


def _require_faiss() -> Any:
    """懒加载 faiss 模块，未安装时抛出友好提示"""
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
    """根据 db 路径计算对应的 Faiss 索引文件路径"""
    return db_path.with_name(f"{db_path.stem}{FAISS_INDEX_SUFFIX}")


def resolve_data_db_path(db_path: Path) -> Path:
    """将 db 路径修正到 data/ 子目录下的对应路径"""
    if db_path.parent.name == DATA_DIR_NAME:
        return db_path
    parts = list(db_path.parts)
    if DATA_DIR_NAME in parts:
        data_idx = parts.index(DATA_DIR_NAME)
        data_root = Path(*parts[: data_idx + 1])
        return data_root / db_path.name
    return db_path.parent / DATA_DIR_NAME / db_path.name


def resolve_existing_db_path(db_path: Path) -> Path:
    """优先返回 data/ 下的数据库路径，不存在则返回原路径"""
    data_db_path = resolve_data_db_path(db_path)
    if data_db_path.exists():
        return data_db_path
    return db_path


def _create_index(dim: int) -> Any:
    """创建带 ID 映射的 Faiss 内积索引"""
    f = _require_faiss()
    return f.IndexIDMap2(f.IndexFlatIP(dim))


def load_or_create_faiss_index(index_path: Path, dim: int | None = None) -> Any:
    """加载已有 Faiss 索引或创建新索引"""
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
    """获取 Faiss 索引中所有已存在的 ID"""
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
    """向 Faiss 索引中插入或更新向量"""
    f = _require_faiss()
    ids = _get_index_ids(index)
    if book_id in ids:
        if not overwrite:
            return
        index.remove_ids(np.asarray([book_id], dtype=np.int64))
    vec = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
    f.normalize_L2(vec)
    index.add_with_ids(vec, np.asarray([book_id], dtype=np.int64))
