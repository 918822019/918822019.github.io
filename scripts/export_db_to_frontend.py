#!/usr/bin/env python3
"""
将抓取到的 SQLite 数据库导出为 frontend/data 下的 NDJSON / 样本 / manifest，便于前端观察完整拉取的数据。

示例：
  python3 scripts/export_db_to_frontend.py \
    --db-path project/起点搜书/data/books.db \
    --output-dir frontend/data \
    --sample-size 200
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def export_db(db_path: str, output_dir: str, sample_size: int) -> None:
    db = Path(db_path).expanduser().resolve()
    if not db.exists():
        raise SystemExit(f"数据库文件不存在: {db}")

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    books_nd = out / "books_full.ndjson"
    chapters_nd = out / "chapters_full.ndjson"
    books_sample = out / "books_sample.json"
    chapters_sample = out / "chapters_sample.json"
    manifest = out / "manifest.json"

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        books_count = int(
            conn.execute("SELECT COUNT(*) AS c FROM books").fetchone()["c"]
        )
        chapters_count = int(
            conn.execute("SELECT COUNT(*) AS c FROM chapters").fetchone()["c"]
        )

        # export books (ndjson) + sample
        sample_books: list[dict[str, Any]] = []
        with books_nd.open("w", encoding="utf-8") as bf:
            for i, row in enumerate(
                conn.execute("SELECT * FROM books ORDER BY book_id")
            ):
                d = row_to_dict(row)
                bf.write(json.dumps(d, ensure_ascii=False) + "\n")
                if i < sample_size:
                    sample_books.append(d)

        # export chapters (ndjson) + sample
        sample_chapters: list[dict[str, Any]] = []
        with chapters_nd.open("w", encoding="utf-8") as cf:
            for i, row in enumerate(
                conn.execute("SELECT * FROM chapters ORDER BY book_id, chapter_id")
            ):
                d = row_to_dict(row)
                cf.write(json.dumps(d, ensure_ascii=False) + "\n")
                if i < sample_size:
                    sample_chapters.append(d)

        books_sample.write_text(
            json.dumps(sample_books, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        chapters_sample.write_text(
            json.dumps(sample_chapters, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        manifest_payload = {
            "generated_at": now_iso(),
            "source_db": str(db),
            "output_dir": str(out),
            "books_count": books_count,
            "chapters_count": chapters_count,
            "sample_size": sample_size,
            "files": {
                "books_ndjson": books_nd.name,
                "chapters_ndjson": chapters_nd.name,
                "books_sample": books_sample.name,
                "chapters_sample": chapters_sample.name,
            },
        }
        manifest.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print("导出完成:", manifest)
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="导出 SQLite 到 frontend/data 供前端观察使用"
    )
    p.add_argument("--db-path", default="project/起点搜书/data/books.db")
    p.add_argument("--output-dir", default="frontend/data")
    p.add_argument("--sample-size", type=int, default=100, help="生成样本数量")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    export_db(args.db_path, args.output_dir, args.sample_size)


if __name__ == "__main__":
    main()
