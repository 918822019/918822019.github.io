"""标签生成模块，提供扁平标签与级联标签两种方式。"""

from src.process.taggers.flat import LLMTagger
from src.process.taggers.cascading import CascadingTagger

__all__ = ["LLMTagger", "CascadingTagger"]
