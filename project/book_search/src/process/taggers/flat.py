"""扁平标签器 LLMTagger，直接生成标签列表"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional, Union

from src.llm.client import LLMClient
from src.utils import extract_json_block, load_books_from_path

from src.process.taggers._utils import PLACEHOLDER_TAGS, _extract_book_fields


class LLMTagger:
    """封装的标签器类，用于批量为小说数据生成标签。

    该类封装了完整的标签生成流程：
    1. 加载小说数据
    2. 判断是否需要重新打标签
    3. 调用 LLM 生成标签
    4. 规范化和过滤标签
    5. 保存结果

    Attributes:
        max_tags: 每本书最多保留的标签数量
        sleep_seconds: 每次 API 调用后的等待时间（秒），用于限流
        overwrite: 是否覆盖已有的标签
        limit: 最多处理的书籍数量，0 表示不限制
        books: 加载的小说数据列表
        llm: LLM 客户端实例
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        model_name: Optional[str] = None,
        max_tags: int = 8,
        sleep_seconds: float = 0.0,
        overwrite: bool = False,
        limit: int = 0,
    ) -> None:
        """初始化标签器"""
        self.max_tags = max_tags
        self.sleep_seconds = sleep_seconds
        self.overwrite = overwrite
        self.limit = max(limit, 0)
        self.books: list[dict[str, Any]] = []

        self.llm = llm if llm is not None else LLMClient(model_name=model_name)

    @staticmethod
    def _normalize_tags(raw_tags: Any, max_tags: int = 8) -> list[str]:
        """规范化标签列表，保留出现顺序并去重"""
        if not isinstance(raw_tags, list):
            return []

        tags: list[str] = []
        seen = set()
        for tag in raw_tags:
            if not isinstance(tag, str):
                continue
            value = tag.strip().replace("，", " ")
            if not value or value in PLACEHOLDER_TAGS:
                continue
            if value not in seen:
                seen.add(value)
                tags.append(value)
            if len(tags) >= max_tags:
                break
        return tags

    def _book_needs_tagging(self, book: dict[str, Any]) -> bool:
        """判断一本书是否需要重新打标签"""
        if self.overwrite:
            return True

        existing = book.get("tags", [])
        if not existing:
            return True

        normalized = self._normalize_tags(existing)
        return len(normalized) == 0

    def _build_prompt(self, book: dict[str, Any]) -> str:
        """构建发送给 LLM 的提示词"""
        fields = _extract_book_fields(book)
        name = fields["name"]
        author = fields["author"]
        description = fields["description"]
        category = fields["category"]
        serial_status = fields["serial_status"]

        return f"""
请根据下列小说信息打标签。

要求：
1. 只返回 JSON，不要输出任何额外文字。
2. JSON 格式必须是：{{"tags": ["标签1", "标签2", ...]}}。
3. 标签数量 4-8 个，尽量覆盖题材、时代、风格、受众、状态。
4. 标签必须简洁（2-6 个字），避免重复。
5. 信息不足时可输出 "未知" 类标签，但尽量推断。

小说信息：
- 书名：{name}
- 作者：{author}
- 原始分类：{category or '（未知）'}
- 连载状态：{serial_status or '（未知）'}
- 简介：{description or '（无简介）'}
""".strip()

    def tag_single_book(self, book: dict[str, Any]) -> list[str]:
        """对单本书调用 LLM 并返回规范化标签列表"""
        system_prompt = "你是中文网络小说标签专家，擅长给小说打规范化标签。"
        prompt = self._build_prompt(book)
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,
        )
        parsed = extract_json_block(response)
        return self._normalize_tags(parsed.get("tags", []), max_tags=self.max_tags)

    def load_books(self, input_path: Union[str, Path]) -> None:
        """从 JSON / NDJSON / SQLite 文件加载小说数据"""
        self.books = load_books_from_path(input_path)

    def save_books(self, output_path: Union[str, Path]) -> None:
        """将小说数据保存到 JSON 文件"""
        p = Path(output_path)
        p.write_text(
            json.dumps(self.books, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def run(
        self,
        input_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> dict[str, int]:
        """执行批量打标签流程"""
        if input_path:
            self.load_books(input_path)

        total = len(self.books)
        changed = 0
        processed = 0
        skipped = 0
        failed = 0

        for idx, book in enumerate(self.books, start=1):
            if not isinstance(book, dict):
                skipped += 1
                continue

            if not self._book_needs_tagging(book):
                skipped += 1
                continue

            if self.limit > 0 and processed >= self.limit:
                break

            try:
                tags = self.tag_single_book(book)
                if tags:
                    book["tags"] = tags
                    changed += 1
                else:
                    failed += 1
                processed += 1
                name = _extract_book_fields(book)["name"] or "未知书名"
                print(f"[{idx}/{total}] 已处理: {name} -> {book.get('tags', [])}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                processed += 1
                name = _extract_book_fields(book)["name"] or "未知书名"
                print(f"[{idx}/{total}] 处理失败: {name} | error={exc}")

            if self.sleep_seconds > 0:
                time.sleep(self.sleep_seconds)

        if output_path:
            self.save_books(output_path)

        stats = {
            "total": total,
            "processed": processed,
            "changed": changed,
            "skipped": skipped,
            "failed": failed,
        }

        print("\n=== 标签任务完成 ===")
        print(f"输入记录数: {total}")
        print(f"本次处理数: {processed}")
        print(f"更新标签数: {changed}")
        print(f"跳过数: {skipped}")
        print(f"失败数: {failed}")

        return stats
