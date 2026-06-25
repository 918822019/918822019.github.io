from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

from src.process.pipeline import (
    _load_existing_tagging_result,
    _merge_existing_tags,
)


class PreprocessIncrementalTests(unittest.TestCase):
    def test_load_existing_tagging_result_builds_book_id_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "books_tagged.json"
            output_path.write_text(
                json.dumps(
                    [
                        {"book_id": 1, "tags": ["都市"]},
                        {"book_id": "2", "cascaded_tags": {"genre": "玄幻"}},
                        {"book_id": "bad", "tags": ["无效"]},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            indexed = _load_existing_tagging_result(output_path)

            self.assertEqual(set(indexed.keys()), {1, 2})
            self.assertEqual(indexed[1]["tags"], ["都市"])
            self.assertEqual(indexed[2]["cascaded_tags"]["genre"], "玄幻")

    def test_merge_existing_tags_only_fills_missing_fields(self) -> None:
        current_books = [
            {"book_id": 1, "title": "A"},
            {"book_id": 2, "title": "B", "tags": ["已有标签"]},
            {"book_id": 3, "title": "C"},
        ]
        existing_books_by_id = {
            1: {
                "book_id": 1,
                "tags": ["都市"],
                "cascaded_tags": {"genre": "都市"},
            },
            2: {
                "book_id": 2,
                "tags": ["玄幻"],
                "cascaded_tags": {"genre": "玄幻"},
            },
        }

        merged = _merge_existing_tags(current_books, existing_books_by_id)

        self.assertEqual(merged, 3)
        self.assertEqual(current_books[0]["tags"], ["都市"])
        cascaded_0 = cast(dict[str, Any], current_books[0]["cascaded_tags"])
        self.assertEqual(
            cascaded_0.get("genre"),
            "都市",
        )
        self.assertEqual(current_books[1]["tags"], ["已有标签"])
        cascaded_1 = cast(dict[str, Any], current_books[1]["cascaded_tags"])
        self.assertEqual(
            cascaded_1.get("genre"),
            "玄幻",
        )
        self.assertNotIn("tags", current_books[2])


if __name__ == "__main__":
    unittest.main()
