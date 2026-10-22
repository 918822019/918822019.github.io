#!/usr/bin/env python3
"""统一 CLI 入口，合并原 polish/embed/search 三个独立脚本。

用法：
    cd project/book_search
    python tools/cli.py polish   --db-path data/books.db --limit 100
    python tools/cli.py embed    --db-path data/books.db --limit 100
    python tools/cli.py search   --db-path data/books.db --query "玄幻小说" --top-k 5
    python tools/cli.py stats    --db-path data/books.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """添加通用的 --db-path 参数。"""
    parser.add_argument(
        "--db-path",
        default="../../data/book_search/books.db",
        help="SQLite 数据库路径（默认 data/books.db）",
    )


def cmd_polish(args: argparse.Namespace) -> None:
    """执行文本润色。"""
    from src.process.polish import POLISH_TABLE, run_polish

    stats = run_polish(
        db_path=Path(args.db_path),
        model_name=args.model,
        limit=max(args.limit, 0),
        sleep_seconds=max(args.sleep, 0.0),
        overwrite=args.overwrite,
    )
    print("\n=== 润色任务完成 ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"  结果表: {POLISH_TABLE}")


def cmd_embed(args: argparse.Namespace) -> None:
    """执行向量化入库。"""
    from src.process.polish import EMBED_TABLE, run_polish_embedding

    stats = run_polish_embedding(
        db_path=Path(args.db_path),
        model_name=args.model,
        limit=max(args.limit, 0),
        sleep_seconds=max(args.sleep, 0.0),
        overwrite=args.overwrite,
    )
    print("\n=== Embedding 任务完成 ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"  结果表: {EMBED_TABLE}")


def cmd_search(args: argparse.Namespace) -> None:
    """基于 embedding 检索相似书籍。"""
    from src.process.polish import search_books_by_polish_embedding

    results = search_books_by_polish_embedding(
        db_path=Path(args.db_path),
        query=args.query,
        model_name=args.model,
        top_k=max(args.top_k, 1),
    )
    print(
        json.dumps(
            {"query": args.query, "results": results},
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_stats(args: argparse.Namespace) -> None:
    """查看数据库统计信息。"""
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"❌ 数据库不存在: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        book_count = conn.execute("SELECT COUNT(*) AS c FROM books").fetchone()["c"]
        chapter_count = conn.execute("SELECT COUNT(*) AS c FROM chapters").fetchone()["c"]
        fetched = conn.execute(
            "SELECT COUNT(*) AS c FROM chapters WHERE is_content_fetched = 1"
        ).fetchone()["c"]

        # 检查润色表
        try:
            polish_count = conn.execute("SELECT COUNT(*) AS c FROM book_polish").fetchone()["c"]
        except sqlite3.OperationalError:
            polish_count = 0

        # 检查向量化表
        try:
            embed_count = conn.execute(
                "SELECT COUNT(*) AS c FROM book_polish_embedding"
            ).fetchone()["c"]
        except sqlite3.OperationalError:
            embed_count = 0

        size_mb = db_path.stat().st_size / (1024 * 1024)

        print("\n📊 数据库统计")
        print("=" * 50)
        print(f"  书籍总数:       {book_count}")
        print(f"  章节数:         {chapter_count}")
        print(f"  已抓取正文:     {fetched} ({fetched / max(chapter_count, 1) * 100:.1f}%)")
        print(f"  未抓取正文:     {chapter_count - fetched}")
        print(f"  已润色:         {polish_count}")
        print(f"  已向量化:       {embed_count}")
        print(f"  数据库大小:     {size_mb:.1f} MB")
        print("=" * 50)
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="book_search 统一 CLI 工具",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- polish ---
    p_polish = sub.add_parser("polish", help="基于前五章正文润色小说简介")
    _add_common_args(p_polish)
    p_polish.add_argument("--model", default=None, help="覆盖默认模型名")
    p_polish.add_argument("--limit", type=int, default=0, help="最多处理多少本，0=全部")
    p_polish.add_argument("--sleep", type=float, default=0.0, help="每本之间等待秒数")
    p_polish.add_argument("--overwrite", action="store_true", help="覆盖已润色结果")
    p_polish.set_defaults(func=cmd_polish)

    # --- embed ---
    p_embed = sub.add_parser("embed", help="为润色结果生成 embedding 并入库")
    _add_common_args(p_embed)
    p_embed.add_argument("--model", default=None, help="覆盖默认 embedding 模型名")
    p_embed.add_argument("--limit", type=int, default=0, help="最多处理多少本，0=全部")
    p_embed.add_argument("--sleep", type=float, default=0.0, help="每本之间等待秒数")
    p_embed.add_argument("--overwrite", action="store_true", help="覆盖已有 embedding")
    p_embed.set_defaults(func=cmd_embed)

    # --- search ---
    p_search = sub.add_parser("search", help="基于润色 embedding 检索相似书籍")
    _add_common_args(p_search)
    p_search.add_argument("--query", required=True, help="检索查询文本")
    p_search.add_argument("--model", default=None, help="覆盖查询 embedding 模型名")
    p_search.add_argument("--top-k", type=int, default=10, help="返回结果数量")
    p_search.set_defaults(func=cmd_search)

    # --- stats ---
    p_stats = sub.add_parser("stats", help="查看数据库统计信息")
    _add_common_args(p_stats)
    p_stats.set_defaults(func=cmd_stats)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
