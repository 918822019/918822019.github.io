"""预处理流程编排模块。

该模块负责把预处理步骤组织成可复用的 pipeline，当前已集成：
- 文本润色步骤（书名+简介）
- 润色文本 embedding 步骤
- LLM 标签步骤（支持扁平标签与级联标签）

说明：
- 本模块只提供函数接口，不提供命令行入口。
- 其他服务代码可直接 import 并调用 `run_preprocess_pipeline`。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

from src.process.book_profile_polish import run_polish
from src.process.llm_tagging import CascadingTagger, LLMTagger
from src.process.polish_embedding import run_polish_embedding

TaggingMode = Literal["flat", "cascading"]


@dataclass(slots=True)
class PreprocessPipelineConfig:
    """预处理 pipeline 配置。"""

    input_path: Union[str, Path]
    output_path: Union[str, Path]
    enable_text_polish: bool = True
    enable_polish_embedding: bool = True
    enable_llm_tagging: bool = True
    polish_model_name: Optional[str] = None
    embedding_model_name: Optional[str] = None
    tagging_mode: TaggingMode = "flat"
    model_name: Optional[str] = None
    max_tags: int = 8
    sleep_seconds: float = 0.0
    overwrite: bool = False
    limit: int = 0


def run_preprocess_pipeline(config: PreprocessPipelineConfig) -> dict[str, object]:
    """执行预处理 pipeline。

    Args:
            config: pipeline 配置对象

    Returns:
            流程统计信息
    """
    input_path = Path(config.input_path)
    output_path = Path(config.output_path)

    result: dict[str, object] = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "steps": [],
    }

    if config.enable_text_polish:
        polish_stats = run_polish(
            db_path=input_path,
            model_name=config.polish_model_name,
            limit=max(config.limit, 0),
            sleep_seconds=max(config.sleep_seconds, 0.0),
            overwrite=config.overwrite,
        )
        result["steps"].append(
            {
                "name": "text_polish",
                "enabled": True,
                "stats": polish_stats,
            }
        )
    else:
        result["steps"].append(
            {
                "name": "text_polish",
                "enabled": False,
                "stats": None,
            }
        )

    if config.enable_polish_embedding:
        embed_stats = run_polish_embedding(
            db_path=input_path,
            model_name=config.embedding_model_name,
            limit=max(config.limit, 0),
            sleep_seconds=max(config.sleep_seconds, 0.0),
            overwrite=config.overwrite,
        )
        result["steps"].append(
            {
                "name": "polish_embedding",
                "enabled": True,
                "stats": embed_stats,
            }
        )
    else:
        result["steps"].append(
            {
                "name": "polish_embedding",
                "enabled": False,
                "stats": None,
            }
        )

    if not config.enable_llm_tagging:
        result["steps"].append(
            {
                "name": "llm_tagging",
                "enabled": False,
                "stats": None,
            }
        )
        return result

    if config.tagging_mode == "cascading":
        tagger = CascadingTagger(
            model_name=config.model_name,
            sleep_seconds=max(config.sleep_seconds, 0.0),
        )
        stats = tagger.run(
            input_path=input_path,
            output_path=output_path,
        )
    else:
        tagger = LLMTagger(
            model_name=config.model_name,
            max_tags=max(config.max_tags, 1),
            sleep_seconds=max(config.sleep_seconds, 0.0),
            overwrite=config.overwrite,
            limit=max(config.limit, 0),
        )
        stats = tagger.run(
            input_path=input_path,
            output_path=output_path,
        )

    result["steps"].append(
        {
            "name": "llm_tagging",
            "enabled": True,
            "mode": config.tagging_mode,
            "stats": stats,
        }
    )
    return result
