"""标签生成共享工具函数与常量"""

from __future__ import annotations

from typing import Any

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


def _pick_first_non_empty(book: dict[str, Any], keys: tuple[str, ...]) -> str:
    """从多个候选字段中取第一个非空字符串值"""
    for key in keys:
        value = str(book.get(key, "")).strip()
        if value:
            return value
    return ""


def _extract_book_fields(book: dict[str, Any]) -> dict[str, str]:
    """提取并兼容不同数据结构下的关键字段"""
    name = _pick_first_non_empty(book, ("name", "title"))
    author = _pick_first_non_empty(book, ("author",))
    description = _pick_first_non_empty(book, ("description", "intro"))
    first_500_chars = _pick_first_non_empty(
        book,
        ("first_500_chars", "excerpt", "content_preview"),
    )
    if not first_500_chars:
        first_500_chars = description[:500]

    category = _pick_first_non_empty(book, ("category",))
    serial_status = _pick_first_non_empty(book, ("serial_status", "status"))

    if name in PLACEHOLDER_FIELDS:
        name = "未知书名"
    if author in PLACEHOLDER_FIELDS:
        author = "未知作者"

    return {
        "name": name,
        "author": author,
        "description": description,
        "first_500_chars": first_500_chars,
        "category": category,
        "serial_status": serial_status,
    }
