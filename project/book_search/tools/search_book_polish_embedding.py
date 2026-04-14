#!/usr/bin/env python3
"""使用润色后的 embedding 做相似书检索。

示例：
    cd project/book_search
    python tools/search_book_polish_embedding.py \
        --db-path data/books.db \
        --query "轻松搞笑修仙文" \
        --top-k 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.process.polish_embedding_search import search_books_by_polish_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于润色 embedding 检索相似小说")
    parser.add_argument(
        "--db-path",
        default="data/books.db",
        help="SQLite 元数据库路径（Faiss 索引同目录，建议位于 data 根目录）",
    )
    parser.add_argument("--query", required=True, help="检索查询文本")
    parser.add_argument("--model", default=None, help="可选，覆盖查询 embedding 模型名")
    parser.add_argument("--top-k", type=int, default=10, help="返回结果数量")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = search_books_by_polish_embedding(
        db_path=Path(args.db_path),
        query=args.query,
        model_name=args.model,
        top_k=max(args.top_k, 1),
    )

    # 用 JSON 输出，便于后续脚本化处理或接入前端。
    print(
        json.dumps(
            {"query": args.query, "results": results}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
