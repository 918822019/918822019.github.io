"""级联标签器 CascadingTagger，按照四级分类体系逐步打标"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional, Union

from src.llm.client import LLMClient
from src.utils import extract_json_block, load_books_from_path

from src.process.taggers._utils import (
    GENRE_CATEGORIES,
    PLOT_TYPES_BY_GENRE,
    _extract_book_fields,
)


class CascadingTagger:
    """级联标签器，按照四级分类体系逐步打标。

    分级策略：
    1. 第一级：题材分类（最粗粒度）
    2. 第二级：情节类型（进入对应分支）
    3. 第三级：角色特征标签
    4. 第四级：风格与情感标签

    Attributes:
        llm: LLM 客户端实例
        sleep_seconds: 每次 API 调用后的等待时间（秒）
        books: 加载的小说数据列表
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        model_name: Optional[str] = None,
        sleep_seconds: float = 0.5,
    ) -> None:
        """初始化级联标签器"""
        self.sleep_seconds = sleep_seconds
        self.books: list[dict[str, Any]] = []
        self.llm = llm if llm is not None else LLMClient(model_name=model_name)

    def load_books(self, input_path: Union[str, Path]) -> None:
        """从 JSON / NDJSON / SQLite 文件加载小说数据"""
        self.books = load_books_from_path(input_path)

    def save_books(self, output_path: Union[str, Path]) -> None:
        """将小说数据保存到 JSON 文件"""
        p = Path(output_path)
        p.write_text(
            json.dumps(self.books, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _call_llm_with_retry(
        self, prompt: str, system_prompt: str, max_retries: int = 2
    ) -> dict[str, Any]:
        """调用 LLM 并解析 JSON，支持重试"""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = self.llm.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.1,
                )
                return extract_json_block(response)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(1)
                    continue
        raise last_error

    def _classify_genre(self, book: dict[str, Any]) -> str:
        """第一级：题材分类"""
        fields = _extract_book_fields(book)
        name = fields["name"]
        author = fields["author"]
        description = fields["description"]
        first_500_chars = fields["first_500_chars"]
        category = fields["category"]
        serial_status = fields["serial_status"]

        prompt = f"""
请判断这本小说的题材分类。

可选分类：{', '.join(GENRE_CATEGORIES)}

要求：
1. 只返回 JSON 格式：{{"genre": "分类名称"}}
2. 必须从上述可选分类中选择一个
3. 如果无法确定，选择"其他"

小说信息：
- 书名：{name}
- 作者：{author}
- 原始分类：{category or '（未知）'}
- 连载状态：{serial_status or '（未知）'}
- 简介：{description or '（无简介）'}
- 前500字：{first_500_chars[:500]}
""".strip()

        system_prompt = "你是中文网络小说分类专家，擅长判断小说的题材类型。"
        result = self._call_llm_with_retry(prompt, system_prompt)
        genre = result.get("genre", "其他")

        if genre not in GENRE_CATEGORIES:
            genre = "其他"

        return genre

    def _classify_plot_type(self, book: dict[str, Any], genre: str) -> str:
        """第二级：情节类型分类"""
        fields = _extract_book_fields(book)
        name = fields["name"]
        description = fields["description"]
        first_500_chars = fields["first_500_chars"]

        plot_options = PLOT_TYPES_BY_GENRE.get(genre, ["综合"])

        prompt = f"""
已知这本小说的题材是「{genre}」，请判断其情节类型。

可选情节类型：{', '.join(plot_options)}

要求：
1. 只返回 JSON 格式：{{"plot_type": "类型名称"}}
2. 必须从上述可选类型中选择一个
3. 如果都不符合，选择第一个选项

小说信息：
- 书名：{name}
- 简介：{description or '（无简介）'}
- 前500字：{first_500_chars[:500]}
""".strip()

        system_prompt = f"你是{genre}小说专家，擅长判断小说的情节类型。"
        result = self._call_llm_with_retry(prompt, system_prompt)
        plot_type = result.get("plot_type", plot_options[0])

        if plot_type not in plot_options:
            plot_type = plot_options[0]

        return plot_type

    def _extract_character_features(self, book: dict[str, Any]) -> dict[str, str]:
        """第三级：提取角色特征标签"""
        fields = _extract_book_fields(book)
        name = fields["name"]
        description = fields["description"]
        first_500_chars = fields["first_500_chars"]

        prompt = f"""
请分析这本小说的角色特征。

要求：
1. 只返回 JSON 格式
2. 字段说明：
   - has_system: "有系统" 或 "无系统"
   - has_rebirth: "有重生" 或 "无重生"
   - protagonist_gender: "男主" 或 "女主" 或 "群像"
   - initial_power: "弱小" 或 "普通" 或 "强大" 或 "无敌"

小说信息：
- 书名：{name}
- 简介：{description or '（无简介）'}
- 前500字：{first_500_chars[:500]}
""".strip()

        system_prompt = "你是小说角色分析专家，擅长提取主角的特征信息。"
        result = self._call_llm_with_retry(prompt, system_prompt)

        return {
            "has_system": result.get("has_system", "未知"),
            "has_rebirth": result.get("has_rebirth", "未知"),
            "protagonist_gender": result.get("protagonist_gender", "未知"),
            "initial_power": result.get("initial_power", "未知"),
        }

    def _extract_style_and_emotion(self, book: dict[str, Any]) -> dict[str, str]:
        """第四级：提取风格与情感标签"""
        fields = _extract_book_fields(book)
        name = fields["name"]
        description = fields["description"]
        first_500_chars = fields["first_500_chars"]

        prompt = f"""
请分析这本小说的整体风格和情感基调。

要求：
1. 只返回 JSON 格式
2. 字段说明：
   - style: 从以下选择一个："轻松"、"压抑"、"烧脑"、"热血"、"平淡"
   - emotion: 从以下选择一个："甜"、"虐"、"无感情线"、"单箭头"、"多角恋"

小说信息：
- 书名：{name}
- 简介：{description or '（无简介）'}
- 前500字：{first_500_chars[:500]}
""".strip()

        system_prompt = "你是小说风格分析专家，擅长判断作品的整体基调和情感走向。"
        result = self._call_llm_with_retry(prompt, system_prompt)

        return {
            "style": result.get("style", "未知"),
            "emotion": result.get("emotion", "未知"),
        }

    def tag_single_book_cascading(self, book: dict[str, Any]) -> dict[str, Any]:
        """对单本书执行完整的四级级联标签流程"""
        genre = self._classify_genre(book)
        print(f"  [1/4] 题材分类: {genre}")

        plot_type = self._classify_plot_type(book, genre)
        print(f"  [2/4] 情节类型: {plot_type}")

        character_features = self._extract_character_features(book)
        print(f"  [3/4] 角色特征: {character_features}")

        style_emotion = self._extract_style_and_emotion(book)
        print(f"  [4/4] 风格情感: {style_emotion}")

        cascaded_tags = {
            "genre": genre,
            "plot_type": plot_type,
            "character_features": character_features,
            "style_and_emotion": style_emotion,
        }

        flat_tags = [
            genre,
            plot_type,
            character_features["has_system"],
            character_features["has_rebirth"],
            character_features["protagonist_gender"],
            character_features["initial_power"],
            style_emotion["style"],
            style_emotion["emotion"],
        ]
        flat_tags = [tag for tag in flat_tags if tag != "未知"]

        return {
            "cascaded_tags": cascaded_tags,
            "flat_tags": flat_tags,
        }

    def run(
        self,
        input_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> dict[str, int]:
        """执行批量级联标签流程"""
        if input_path:
            self.load_books(input_path)

        total = len(self.books)
        processed = 0
        skipped = 0
        failed = 0

        for idx, book in enumerate(self.books, start=1):
            if not isinstance(book, dict):
                skipped += 1
                continue

            if book.get("cascaded_tags") and not book.get("overwrite_cascaded"):
                skipped += 1
                continue

            try:
                name = _extract_book_fields(book)["name"] or "未知书名"
                print(f"\n[{idx}/{total}] 开始处理: {name}")

                result = self.tag_single_book_cascading(book)

                book["cascaded_tags"] = result["cascaded_tags"]
                book["tags"] = result["flat_tags"]
                book.pop("overwrite_cascaded", None)

                processed += 1
                print(f"[{idx}/{total}] 完成: {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                name = _extract_book_fields(book)["name"] or "未知书名"
                print(f"[{idx}/{total}] 处理失败: {name} | error={exc}")

            if self.sleep_seconds > 0:
                time.sleep(self.sleep_seconds)

        if output_path:
            self.save_books(output_path)

        stats = {
            "total": total,
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
        }

        print("\n=== 级联标签任务完成 ===")
        print(f"总记录数: {total}")
        print(f"成功处理: {processed}")
        print(f"跳过数: {skipped}")
        print(f"失败数: {failed}")

        return stats
