from __future__ import annotations

import sqlite3
import unittest

from src.process.polish import (
    CHAPTER_PREVIEW_MAX_CHARS,
    build_prompt,
    fetch_chapter_previews,
    polish_one_book,
)


class FakeLLMClient:
    def __init__(self, response: str):
        self.response = response

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
    ) -> str:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        self.last_temperature = temperature
        return self.response


class BookProfilePolishTests(unittest.TestCase):
    def test_fetch_chapter_previews_returns_first_five_with_trimmed_content(
        self,
    ) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE chapters (
                book_id INTEGER NOT NULL,
                chapter_id INTEGER NOT NULL,
                chapter_name TEXT NOT NULL,
                content TEXT,
                is_content_fetched INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (book_id, chapter_id)
            )
            """
        )

        for chapter_id in range(1, 8):
            conn.execute(
                """
                INSERT INTO chapters (
                    book_id,
                    chapter_id,
                    chapter_name,
                    content,
                    is_content_fetched
                )
                VALUES (?, ?, ?, ?, 1)
                """,
                (
                    1,
                    chapter_id,
                    f"第{chapter_id}章",
                    "A" * (CHAPTER_PREVIEW_MAX_CHARS + 50),
                ),
            )

        previews = fetch_chapter_previews(conn=conn, book_id=1)

        self.assertEqual(len(previews), 5)
        self.assertEqual(previews[0]["chapter_name"], "第1章")
        self.assertTrue(previews[0]["content"].endswith("..."))

    def test_build_prompt_includes_chapter_context(self) -> None:
        prompt = build_prompt(
            title="测试书名",
            intro="旧简介",
            chapter_previews=[
                {"chapter_name": "楔子", "content": "主角在雨夜醒来。"},
                {"chapter_name": "入局", "content": "他被卷入一场纷争。"},
            ],
        )

        self.assertIn("测试书名", prompt)
        self.assertIn("旧简介", prompt)
        self.assertIn("楔子", prompt)
        self.assertIn("主角在雨夜醒来", prompt)

    def test_polish_one_book_falls_back_to_original_title(self) -> None:
        llm = FakeLLMClient(
            "{"
            '"polished_title": "", '
            '"polished_intro": "雨夜醒来的少年，被迫踏入旧王朝与新秩序交错的漩涡。"'
            "}"
        )

        polished_title, polished_intro = polish_one_book(
            llm=llm,
            title="原书名",
            intro="原简介",
            chapter_previews=[{"chapter_name": "第一章", "content": "剧情正文"}],
        )

        self.assertEqual(polished_title, "原书名")
        self.assertIn("雨夜醒来的少年", polished_intro)
        self.assertIn("第一章", llm.last_prompt)


if __name__ == "__main__":
    unittest.main()
