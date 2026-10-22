"""全线统计 — 一键查看数据库、向量化、打标、文件系统状态.

用法：
    python -m src.tools.stats
    python -m src.tools.stats --db-path data/books.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def _fmt(n: int) -> str:
    return f"{n:,}"


def _pct(part: int, total: int) -> str:
    if total == 0:
        return "-"
    return f"{part / total * 100:.1f}%"


def _size_mb(path: Path) -> str:
    if not path.exists():
        return "-"
    return f"{path.stat().st_size / (1024 * 1024):.1f} MB"


def _heading(text: str) -> None:
    print()
    print(f"{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def _kv(key: str, value: str) -> None:
    print(f"  {key:<20} {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="全线统计")
    parser.add_argument(
        "--db-path",
        default="../../data/book_search/books.db",
        help="SQLite 数据库路径（默认 data/books.db）",
    )
    parser.add_argument(
        "--tagged-json",
        default="data/books_tagged.json",
        help="打标结果 JSON 路径（默认 data/books_tagged.json）",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    tagged_path = Path(args.tagged_json)
    data_dir = db_path.parent
    shards_dir = data_dir / "shards"
    faiss_path = db_path.with_name(f"{db_path.stem}.polish_embedding.faiss")
    faiss_exists = faiss_path.exists()

    print()
    print("全线统计")
    print("=" * 60)

    # ---- 1. 数据库基础 ----
    _heading("书籍数据")

    if not db_path.exists():
        _kv("数据库", "文件不存在")
        return

    _kv("数据库", f"{db_path.name} ({_size_mb(db_path)})")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        book_count = conn.execute("SELECT COUNT(*) AS c FROM books").fetchone()["c"]
        chapter_count = conn.execute("SELECT COUNT(*) AS c FROM chapters").fetchone()["c"]
        fetched = conn.execute("SELECT COUNT(*) AS c FROM chapters WHERE is_content_fetched = 1").fetchone()["c"]

        _kv("书籍总数", _fmt(book_count))
        _kv("章节数", _fmt(chapter_count))
        _kv("已抓取正文", f"{_fmt(fetched)} ({_pct(fetched, chapter_count)})")
        _kv("未抓取正文", _fmt(chapter_count - fetched))

        # ---- 2. 文本润色 ----
        _heading("文本润色")

        try:
            polish_count = conn.execute("SELECT COUNT(*) AS c FROM book_polish").fetchone()["c"]
            _kv("已润色", f"{_fmt(polish_count)} ({_pct(polish_count, book_count)})")
            _kv("未润色", _fmt(book_count - polish_count))
            models = conn.execute("SELECT DISTINCT model_name FROM book_polish WHERE model_name IS NOT NULL").fetchall()
            if models:
                _kv("润色模型", ", ".join(m["model_name"] for m in models))
        except sqlite3.OperationalError:
            _kv("润色表", "不存在")
            polish_count = 0

        # ---- 3. 向量化 ----
        _heading("向量化")

        try:
            embed_count = conn.execute("SELECT COUNT(*) AS c FROM book_polish_embedding").fetchone()["c"]
            _kv("已向量化", f"{_fmt(embed_count)} ({_pct(embed_count, book_count)})")
            _kv("未向量化", _fmt(book_count - embed_count))
            models = conn.execute("SELECT DISTINCT model_name FROM book_polish_embedding WHERE model_name IS NOT NULL").fetchall()
            if models:
                _kv("向量化模型", ", ".join(m["model_name"] for m in models))
        except sqlite3.OperationalError:
            _kv("向量化表", "不存在")
            embed_count = 0

    finally:
        conn.close()

    # ---- 4. Faiss 索引 ----
    _heading("Faiss 索引")

    _kv("索引文件", f"{'[存在]' if faiss_exists else '[不存在]'} {_size_mb(faiss_path)}")
    if faiss_exists:
        try:
            import faiss
            index = faiss.read_index(str(faiss_path))
            _kv("类型", type(index).__name__)
            _kv("向量数", _fmt(index.ntotal))
            _kv("维度", str(index.d))
        except Exception as e:
            _kv("读取失败", str(e))

    # ---- 5. LLM 打标 ----
    _heading("LLM 打标")

    tagged_exists = tagged_path.exists()
    _kv("打标文件", f"{'[存在]' if tagged_exists else '[不存在]'} {_size_mb(tagged_path)}")
    if tagged_exists:
        try:
            with open(tagged_path, encoding="utf-8") as f:
                data = json.load(f)
            n = len(data) if isinstance(data, list) else 0
            _kv("已打标书籍", _fmt(n))
            if isinstance(data, list) and n > 0:
                s = data[0]
                if "tags" in s:
                    _kv("标签模式", "flat")
                    tags = s.get("tags", [])
                    _kv("样例标签", ", ".join(tags[:5]) + ("..." if len(tags) > 5 else ""))
                elif "cascaded_tags" in s:
                    _kv("标签模式", "cascading")
                    _kv("样例标签", str(list(s["cascaded_tags"].keys())))
                if s.get("model_name"):
                    _kv("打标模型", s["model_name"])
        except Exception as e:
            _kv("读取失败", str(e))

    # ---- 6. 分片 ----
    _heading("数据分片")

    shards_exist = shards_dir.exists()
    _kv("分片目录", "[存在]" if shards_exist else "[不存在]")
    if shards_exist:
        shard_files = sorted(shards_dir.glob("*.db"))
        index_json = shards_dir / "index.json"
        _kv("分片文件数", _fmt(len(shard_files)))
        _kv("索引文件", "[存在]" if index_json.exists() else "[不存在]")
        if shard_files:
            total_size = sum(f.stat().st_size for f in shard_files)
            _kv("分片总大小", f"{total_size / (1024 * 1024):.1f} MB")
        if index_json.exists():
            try:
                with open(index_json, encoding="utf-8") as f:
                    idx = json.load(f)
                if isinstance(idx, dict):
                    _kv("Shard 范围", f"{idx.get('start', '?')}-{idx.get('end', '?')}")
                    _kv("Shard 大小", str(idx.get("shard_size", "?")))
            except Exception:
                pass

    # ---- 7. 数据目录 ----
    _heading("数据目录")
    if data_dir.exists():
        for item in sorted(data_dir.iterdir()):
            if item.is_file():
                _kv(item.name, _size_mb(item))
            elif item.is_dir():
                subs = list(item.rglob("*"))
                total = sum(f.stat().st_size for f in subs if f.is_file())
                _kv(f"{item.name}/", f"{total / (1024 * 1024):.1f} MB ({len(subs)} items)")

    print()


if __name__ == "__main__":
    main()
