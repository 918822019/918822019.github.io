#!/usr/bin/env python3
"""基于润色后的书名与简介生成 embedding 并入库。"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.process.polish_embedding import EMBED_TABLE, run_polish_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为 book_polish 生成 embedding 并写入 Faiss（元数据写入 SQLite）"
    )
    parser.add_argument(
        "--db-path",
        default="data/books.db",
        help="SQLite 元数据库路径（建议位于 data 根目录）",
    )
    parser.add_argument("--model", default=None, help="可选，覆盖默认 embedding 模型名")
    parser.add_argument(
        "--limit", type=int, default=0, help="最多处理多少本，0 表示不限制"
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="每本之间等待秒数")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在 embedding")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = run_polish_embedding(
        db_path=Path(args.db_path),
        model_name=args.model,
        limit=max(args.limit, 0),
        sleep_seconds=max(args.sleep, 0.0),
        overwrite=args.overwrite,
    )

    print("\n=== Embedding 任务完成 ===")
    print(f"总记录数: {stats['total']}")
    print(f"本次处理数: {stats['processed']}")
    print(f"成功写入数: {stats['changed']}")
    print(f"跳过数: {stats['skipped']}")
    print(f"失败数: {stats['failed']}")
    print(f"结果表: {EMBED_TABLE}")


if __name__ == "__main__":
    main()
