from typing import Dict, Any


def generate_reason(book: Dict[str, Any], query: str) -> str:
    # 模板生成推荐理由，可扩展为LLM生成
    return f"因为该书标签包含'{query}'，且作者为{book['author']}，内容简介：{book['description'][:30]}..."
