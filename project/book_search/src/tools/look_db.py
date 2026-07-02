#!/usr/bin/env python3
"""数据库快速查看工具，打印各表结构和前几行数据。

用法：
    cd project/book_search
    python tools/look_db.py --db-path data/books.db
    python tools/look_db.py --db-path data/books.db --limit 10
"""

from __future__ import annotations

import argparse
import sqlite3


def main() -> None:
    parser = argparse.ArgumentParser(description="查看 SQLite 数据库表结构和样本数据")
    parser.add_argument(
        "--db-path",
        default="../../data/book_search/books.db",
        help="SQLite 数据库路径（默认 data/books.db）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="每个表预览的行数（默认 5）",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        print(f"数据库中找到 {len(tables)} 个表：")
        for table in tables:
            table_name = table[0]
            print(f"\n--- 表: {table_name} ---")

            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [(row[1], row[2]) for row in cursor.fetchall()]
            print(f"  列: {', '.join(f'{name}({typ})' for name, typ in columns)}")

            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total = cursor.fetchone()[0]
            print(f"  总行数: {total}")

            cursor.execute(f"SELECT * FROM {table_name} LIMIT {args.limit}")
            rows = cursor.fetchall()
            col_names = [description[0] for description in cursor.description]

            for i, row in enumerate(rows, 1):
                print(f"  [{i}] {dict(zip(col_names, row))}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
