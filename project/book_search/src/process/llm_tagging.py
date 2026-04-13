"""使用 LLM 对小说数据进行批量打标签。

本模块提供两种标签器：
1. `LLMTagger`: 通用标签器，直接生成扁平化标签列表
2. `CascadingTagger`: 级联标签器，按照四级分类体系逐步打标
   - 第一级：题材分类（都市/玄幻/科幻/古代/悬疑/其他）
   - 第二级：情节类型（根据第一级结果进入对应分支）
   - 第三级：角色特征标签（系统、重生、性别、初始实力等）
   - 第四级：风格与情感标签（轻松/压抑、甜/虐等）

主要功能：
- 从 JSON 文件加载小说数据
- 调用 LLM 为每本小说生成标签（支持级联和扁平两种方式）
- 规范化标签格式（去重、过滤占位符）
- 支持增量更新（跳过已有有效标签的书籍）
- 保存带标签的结果到 JSON 文件

使用示例：
    ```python
    # 方式1: 使用通用标签器
    from project.book_search.src.process.llm_tagging import LLMTagger

    tagger = LLMTagger(model_name="my-model", max_tags=6, sleep_seconds=0.5)
    tagger.load_books("data/books.json")
    stats = tagger.run(output_path="data/books_tagged.json")
    print(stats)

    # 方式2: 使用级联标签器（推荐，结构化更好）
    from project.book_search.src.process.llm_tagging import CascadingTagger

    cascading_tagger = CascadingTagger(model_name="my-model", sleep_seconds=0.5)
    cascading_tagger.load_books("data/books.json")
    stats = cascading_tagger.run(output_path="data/books_cascaded.json")
    print(stats)
    ```
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Optional, Union

from src.llm.llm_client import LLMClient

# 需要过滤的占位符标签（不应出现在最终结果中）
PLACEHOLDER_TAGS = {"分类", "状态"}
# 需要过滤的占位符字段值（书名和作者的默认占位符）
PLACEHOLDER_FIELDS = {"书名", "作者"}

# 第一级：题材分类选项
GENRE_CATEGORIES = ["都市", "玄幻", "科幻", "古代", "悬疑", "其他"]

# 第二级：情节类型分支（根据第一级题材选择）
PLOT_TYPES_BY_GENRE = {
    "都市": ["职场", "豪门", "异能", "生活", "重生"],
    "玄幻": ["修仙", "魔法", "武侠", "克苏鲁", "游戏异界"],
    "科幻": ["星际", "末世", "赛博朋克", "时间旅行", "人工智能"],
    "古代": ["宫斗", "权谋", "穿越", "种田", "江湖"],
    "悬疑": ["推理", "惊悚", "犯罪", "灵异", "探险"],
    "其他": ["综合", "特殊设定"],
}

# 第三级：角色特征标签问题
CHARACTER_FEATURES = [
    "有无系统？",
    "有无重生？",
    "主角性别？（男/女/群像）",
    "主角初始实力？（弱小/普通/强大/无敌）",
]

# 第四级：风格与情感标签问题
STYLE_AND_EMOTION = [
    "整体风格？（轻松/压抑/烧脑/热血/平淡）",
    "感情线？（甜/虐/无感情线/单箭头/多角恋）",
]


class CascadingTagger:
    """级联标签器，按照四级分类体系逐步打标。

    分级策略：
    1. 第一级：题材分类（最粗粒度）
       - 输入：前500字 + 简介
       - 输出：都市/玄幻/科幻/古代/悬疑/其他

    2. 第二级：情节类型（进入对应分支）
       - 根据第一级结果，从对应的子类别中选择
       - 例如：都市 -> 职场/豪门/异能/生活/重生

    3. 第三级：角色特征标签
       - 判断：有无系统？有无重生？主角性别？主角初始实力？

    4. 第四级：风格与情感标签
       - 判断：轻松/压抑/烧脑？甜/虐/无感情线？

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
        """初始化级联标签器。

        Args:
            llm: 可选的 LLM 客户端实例
            model_name: 模型名称，当 llm 为 None 时使用
            sleep_seconds: 每次 API 调用后的等待时间（秒），默认 0.5
        """
        self.sleep_seconds = sleep_seconds
        self.books: list[dict[str, Any]] = []
        self.llm = llm if llm is not None else LLMClient(model_name=model_name)

    def load_books(self, input_path: Union[str, Path]) -> None:
        """从 JSON 文件加载小说数据。"""
        p = Path(input_path)
        if not p.exists():
            raise FileNotFoundError(f"输入文件不存在: {p}")
        books = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(books, list):
            raise ValueError("输入 JSON 必须是数组，每个元素是一条小说记录")
        self.books = books

    def save_books(self, output_path: Union[str, Path]) -> None:
        """将小说数据保存到 JSON 文件。"""
        p = Path(output_path)
        p.write_text(json.dumps(self.books, ensure_ascii=False, indent=2), encoding="utf-8")

    def _call_llm_with_retry(self, prompt: str, system_prompt: str, max_retries: int = 2) -> dict[str, Any]:
        """调用 LLM 并解析 JSON，支持重试。

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            max_retries: 最大重试次数

        Returns:
            解析后的 JSON 字典

        Raises:
            Exception: 当所有重试都失败时
        """
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = self.llm.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.1,
                )
                return LLMTagger._extract_json_block(response)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(1)  # 重试前等待1秒
                    continue
        raise last_error

    def _classify_genre(self, book: dict[str, Any]) -> str:
        """第一级：题材分类。

        Args:
            book: 小说数据字典

        Returns:
            题材分类结果（都市/玄幻/科幻/古代/悬疑/其他）
        """
        name = str(book.get("name", "")).strip()
        author = str(book.get("author", "")).strip()
        description = str(book.get("description", "")).strip()
        first_500_chars = str(book.get("first_500_chars", description[:500])).strip()

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
- 简介：{description or '（无简介）'}
- 前500字：{first_500_chars[:500]}
""".strip()

        system_prompt = "你是中文网络小说分类专家，擅长判断小说的题材类型。"
        result = self._call_llm_with_retry(prompt, system_prompt)
        genre = result.get("genre", "其他")

        # 确保返回值在合法范围内
        if genre not in GENRE_CATEGORIES:
            genre = "其他"

        return genre

    def _classify_plot_type(self, book: dict[str, Any], genre: str) -> str:
        """第二级：情节类型分类。

        Args:
            book: 小说数据字典
            genre: 第一级的题材分类结果

        Returns:
            情节类型（根据题材不同而不同）
        """
        name = str(book.get("name", "")).strip()
        description = str(book.get("description", "")).strip()
        first_500_chars = str(book.get("first_500_chars", description[:500])).strip()

        # 获取该题材对应的情节类型选项
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

        # 确保返回值在合法范围内
        if plot_type not in plot_options:
            plot_type = plot_options[0]

        return plot_type

    def _extract_character_features(self, book: dict[str, Any]) -> dict[str, str]:
        """第三级：提取角色特征标签。

        Args:
            book: 小说数据字典

        Returns:
            角色特征字典，包含：has_system, has_rebirth, protagonist_gender, initial_power
        """
        name = str(book.get("name", "")).strip()
        description = str(book.get("description", "")).strip()
        first_500_chars = str(book.get("first_500_chars", description[:500])).strip()

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
        """第四级：提取风格与情感标签。

        Args:
            book: 小说数据字典

        Returns:
            风格与情感字典，包含：style, emotion
        """
        name = str(book.get("name", "")).strip()
        description = str(book.get("description", "")).strip()
        first_500_chars = str(book.get("first_500_chars", description[:500])).strip()

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
        """对单本书执行完整的四级级联标签流程。

        Args:
            book: 小说数据字典

        Returns:
            包含所有层级标签的字典
        """
        # 第一级：题材分类
        genre = self._classify_genre(book)
        print(f"  [1/4] 题材分类: {genre}")

        # 第二级：情节类型
        plot_type = self._classify_plot_type(book, genre)
        print(f"  [2/4] 情节类型: {plot_type}")

        # 第三级：角色特征
        character_features = self._extract_character_features(book)
        print(f"  [3/4] 角色特征: {character_features}")

        # 第四级：风格与情感
        style_emotion = self._extract_style_and_emotion(book)
        print(f"  [4/4] 风格情感: {style_emotion}")

        # 构建结构化标签
        cascaded_tags = {
            "genre": genre,
            "plot_type": plot_type,
            "character_features": character_features,
            "style_and_emotion": style_emotion,
        }

        # 同时生成扁平化标签列表（用于兼容原有系统）
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
        # 过滤掉"未知"值
        flat_tags = [tag for tag in flat_tags if tag != "未知"]

        return {
            "cascaded_tags": cascaded_tags,
            "flat_tags": flat_tags,
        }

    def run(self, input_path: Optional[Union[str, Path]] = None, output_path: Optional[Union[str, Path]] = None) -> dict[str, int]:
        """执行批量级联标签流程。

        Args:
            input_path: 可选的输入文件路径
            output_path: 可选的输出文件路径

        Returns:
            统计信息字典
        """
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

            # 检查是否已有级联标签
            if book.get("cascaded_tags") and not book.get("overwrite_cascaded"):
                skipped += 1
                continue

            try:
                name = book.get("name", "未知书名")
                print(f"\n[{idx}/{total}] 开始处理: {name}")

                result = self.tag_single_book_cascading(book)

                # 保存级联标签和扁平标签
                book["cascaded_tags"] = result["cascaded_tags"]
                book["tags"] = result["flat_tags"]  # 兼容原有 tags 字段
                book.pop("overwrite_cascaded", None)  # 移除标记

                processed += 1
                print(f"[{idx}/{total}] 完成: {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                name = book.get("name", "未知书名")
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
        """初始化标签器。

        Args:
            llm: 可选的 LLM 客户端实例，用于依赖注入和测试
            model_name: 模型名称，当 llm 为 None 时使用
            max_tags: 每本书最多保留的标签数量，默认 8
            sleep_seconds: 每次 API 调用后的等待时间（秒），默认 0（不限流）
            overwrite: 是否覆盖已有标签，默认 False（只处理无标签的书籍）
            limit: 最多处理的书籍数量，0 表示不限制，默认 0
        """
        self.max_tags = max_tags
        self.sleep_seconds = sleep_seconds
        self.overwrite = overwrite
        self.limit = max(limit, 0)
        self.books: list[dict[str, Any]] = []

        # 支持注入自定义 LLMClient，便于测试和替换实现
        # 如果未提供，则创建默认的 LLMClient 实例
        self.llm = llm if llm is not None else LLMClient(model_name=model_name)

    @staticmethod
    def _extract_json_block(text: str) -> dict[str, Any]:
        """从模型回复中提取第一个 JSON 对象。

        尝试直接解析整个文本，如果失败则使用正则表达式提取 JSON 块。

        Args:
            text: LLM 返回的原始文本

        Returns:
            解析后的 JSON 字典

        Raises:
            ValueError: 当文本为空或未找到可解析的 JSON 时
        """
        text = text.strip()
        if not text:
            raise ValueError("模型返回为空")

        # 首先尝试直接解析整个文本
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 如果直接解析失败，使用正则表达式提取 JSON 对象
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError(f"未找到可解析 JSON: {text[:120]}")
        return json.loads(match.group(0))

    @staticmethod
    def _normalize_tags(raw_tags: Any, max_tags: int = 8) -> list[str]:
        """规范化标签列表，保留出现顺序并去重。

        处理步骤：
        1. 检查输入是否为列表
        2. 过滤非字符串类型的标签
        3. 清理空白字符和中文逗号
        4. 过滤空值和占位符标签
        5. 去重并限制最大数量

        Args:
            raw_tags: 原始标签列表（可能包含无效数据）
            max_tags: 最多保留的标签数量

        Returns:
            规范化后的标签列表
        """
        if not isinstance(raw_tags, list):
            return []

        tags: list[str] = []
        seen = set()  # 用于去重
        for tag in raw_tags:
            # 跳过非字符串类型
            if not isinstance(tag, str):
                continue
            # 清理空白字符，将中文逗号替换为空格
            value = tag.strip().replace("，", " ")
            # 过滤空值和占位符标签
            if not value or value in PLACEHOLDER_TAGS:
                continue
            # 去重并添加到结果列表
            if value not in seen:
                seen.add(value)
                tags.append(value)
            # 达到最大数量后停止
            if len(tags) >= max_tags:
                break
        return tags

    def _book_needs_tagging(self, book: dict[str, Any]) -> bool:
        """判断一本书是否需要重新打标签。

        判断逻辑：
        - 如果设置了 overwrite=True，总是需要重新打标签
        - 如果没有 tags 字段或 tags 为空，需要打标签
        - 如果 tags 经过规范化后为空，需要重新打标签

        Args:
            book: 小说数据字典

        Returns:
            True 表示需要打标签，False 表示可以跳过
        """
        if self.overwrite:
            return True

        existing = book.get("tags", [])
        if not existing:
            return True

        # 检查现有标签是否有效（规范化后不为空）
        normalized = self._normalize_tags(existing)
        return len(normalized) == 0

    def _build_prompt(self, book: dict[str, Any]) -> str:
        """构建发送给 LLM 的提示词。

        从书籍数据中提取书名、作者、简介，并格式化为结构化的提示词。
        如果字段值为占位符，则替换为默认值。

        Args:
            book: 小说数据字典，应包含 name、author、description 字段

        Returns:
            格式化后的提示词字符串
        """
        name = str(book.get("name", "")).strip()
        author = str(book.get("author", "")).strip()
        description = str(book.get("description", "")).strip()

        # 如果字段值是占位符，替换为默认值
        if name in PLACEHOLDER_FIELDS:
            name = "未知书名"
        if author in PLACEHOLDER_FIELDS:
            author = "未知作者"

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
- 简介：{description or '（无简介）'}
""".strip()

    def tag_single_book(self, book: dict[str, Any]) -> list[str]:
        """对单本书调用 LLM 并返回规范化标签列表。

        这是核心的标签生成方法，执行以下步骤：
        1. 构建提示词
        2. 调用 LLM 生成标签
        3. 提取 JSON 响应
        4. 规范化标签

        Args:
            book: 小说数据字典

        Returns:
            规范化后的标签列表
        """
        system_prompt = "你是中文网络小说标签专家，擅长给小说打规范化标签。"
        prompt = self._build_prompt(book)
        # 调用 LLM，使用低温度以保证结果稳定性
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,
        )
        # 从响应中提取 JSON 并规范化标签
        parsed = self._extract_json_block(response)
        return self._normalize_tags(parsed.get("tags", []), max_tags=self.max_tags)

    def load_books(self, input_path: Union[str, Path]) -> None:
        """从 JSON 文件加载小说数据。

        Args:
            input_path: 输入 JSON 文件路径

        Raises:
            FileNotFoundError: 当文件不存在时
            ValueError: 当 JSON 不是数组格式时
        """
        p = Path(input_path)
        if not p.exists():
            raise FileNotFoundError(f"输入文件不存在: {p}")
        books = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(books, list):
            raise ValueError("输入 JSON 必须是数组，每个元素是一条小说记录")
        self.books = books

    def save_books(self, output_path: Union[str, Path]) -> None:
        """将小说数据保存到 JSON 文件。

        Args:
            output_path: 输出 JSON 文件路径
        """
        p = Path(output_path)
        p.write_text(json.dumps(self.books, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(self, input_path: Optional[Union[str, Path]] = None, output_path: Optional[Union[str, Path]] = None) -> dict[str, int]:
        """执行批量打标签流程。

        这是主要的入口方法，遍历所有书籍并为需要打标签的书籍调用 LLM。
        支持增量更新：跳过已有有效标签的书籍（除非设置 overwrite=True）。

        Args:
            input_path: 可选的输入文件路径，如果提供会先加载数据
            output_path: 可选的输出文件路径，如果提供会在结束时保存结果

        Returns:
            统计信息字典，包含以下键：
            - total: 总记录数
            - processed: 本次处理的记录数
            - changed: 成功更新标签的记录数
            - skipped: 跳过的记录数
            - failed: 失败的记录数
        """
        if input_path:
            self.load_books(input_path)

        total = len(self.books)
        changed = 0  # 成功更新标签的数量
        processed = 0  # 已处理的记录数
        skipped = 0  # 跳过的记录数
        failed = 0  # 失败的记录数

        # 遍历所有书籍
        for idx, book in enumerate(self.books, start=1):
            # 跳过非字典类型的记录
            if not isinstance(book, dict):
                skipped += 1
                continue

            # 检查是否需要打标签
            if not self._book_needs_tagging(book):
                skipped += 1
                continue

            # 检查是否达到处理上限
            if self.limit > 0 and processed >= self.limit:
                break

            try:
                # 调用 LLM 生成标签
                tags = self.tag_single_book(book)
                if tags:
                    book["tags"] = tags
                    changed += 1
                else:
                    failed += 1
                processed += 1
                name = book.get("name", "未知书名")
                print(f"[{idx}/{total}] 已处理: {name} -> {book.get('tags', [])}")
            except Exception as exc:  # noqa: BLE001
                # 捕获所有异常，记录失败但不中断流程
                failed += 1
                processed += 1
                name = book.get("name", "未知书名")
                print(f"[{idx}/{total}] 处理失败: {name} | error={exc}")

            # 限流：如果设置了等待时间，则在每次调用后暂停
            if self.sleep_seconds > 0:
                time.sleep(self.sleep_seconds)

        # 如果指定了输出路径，保存结果
        if output_path:
            self.save_books(output_path)

        # 构建统计信息
        stats = {
            "total": total,
            "processed": processed,
            "changed": changed,
            "skipped": skipped,
            "failed": failed,
        }

        # 打印统计报告
        print("\n=== 标签任务完成 ===")
        print(f"输入记录数: {total}")
        print(f"本次处理数: {processed}")
        print(f"更新标签数: {changed}")
        print(f"跳过数: {skipped}")
        print(f"失败数: {failed}")

        return stats
    print(f"输入记录数: {total}")
    print(f"本次处理数: {processed}")
    print(f"更新标签数: {changed}")
    print(f"跳过数: {skipped}")
    print(f"失败数: {failed}")
    print(f"输出文件: {output_path}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数并返回命名空间对象。

    Returns:
        包含所有命令行参数的命名空间对象
    """
    parser = argparse.ArgumentParser(description="使用 LLM 对小说数据批量打标签")
    parser.add_argument("--input", default="data/books.json", help="输入 JSON 文件路径")
    parser.add_argument(
        "--output",
        default="data/books_tagged.json",
        help="输出 JSON 文件路径",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 tags")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多处理多少本，0 表示不限制",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="每本之间的等待秒数",
    )
    parser.add_argument(
        "--max-tags",
        type=int,
        default=8,
        help="每本最多保留多少标签",
    )
    parser.add_argument("--model", default=None, help="覆盖默认模型名")
    return parser.parse_args()


def main() -> None:
    """主函数：解析命令行参数并执行标签任务。

    这是命令行入口点，将命令行参数转换为 LLMTagger 的配置并执行。
    """
    args = parse_args()
    # 创建标签器实例
    tagger = LLMTagger(
        model_name=args.model,
        max_tags=max(args.max_tags, 1),
        sleep_seconds=max(args.sleep, 0.0),
        overwrite=args.overwrite,
        limit=max(args.limit, 0),
    )
    # 执行标签任务
    tagger.run(
        input_path=Path(args.input),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
