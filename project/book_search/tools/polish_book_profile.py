#!/usr/bin/env python3
"""基于书名、简介和前五章正文润色小说简介，并写入 SQLite 的 book_polish 表。"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.process.book_profile_polish import POLISH_TABLE, run_polish


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于前五章正文批量润色小说简介并写入 SQLite"
    )
    parser.add_argument(
        "--db-path",
        default="data/books.db",
        help="SQLite 数据库路径",
    )
    parser.add_argument("--model", default=None, help="可选，覆盖默认模型名")
    parser.add_argument(
        "--limit", type=int, default=0, help="最多处理多少本，0 表示不限制"
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="每本之间等待秒数")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已润色结果")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = run_polish(
        db_path=Path(args.db_path),
        model_name=args.model,
        limit=max(args.limit, 0),
        sleep_seconds=max(args.sleep, 0.0),
        overwrite=args.overwrite,
    )
    print("\n=== 润色任务完成 ===")
    print(f"总记录数: {stats['total']}")
    print(f"本次处理数: {stats['processed']}")
    print(f"成功写入数: {stats['changed']}")
    print(f"跳过数: {stats['skipped']}")
    print(f"失败数: {stats['failed']}")
    print(f"结果表: {POLISH_TABLE}")


if __name__ == "__main__":
    main()
