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

    input_path: Union[str, Path]  # 输入数据库文件路径
    output_path: Union[str, Path]  # 输出数据库文件路径
    enable_text_polish: bool = True  # 是否启用文本润色（书名+简介）
    enable_polish_embedding: bool = True  # 是否启用润色文本的 embedding 生成
    enable_llm_tagging: bool = True  # 是否启用 LLM 标签生成
    polish_model_name: Optional[str] = None  # 文本润色使用的模型名称，None 则使用默认模型
    embedding_model_name: Optional[str] = None  # embedding 生成使用的模型名称，None 则使用默认模型
    tagging_mode: TaggingMode = "flat"  # 标签生成模式："flat"（扁平标签）或 "cascading"（级联标签）
    model_name: Optional[str] = None  # LLM 标签生成使用的模型名称，None 则使用默认模型
    max_tags: int = 8  # 每本书最多生成的标签数量（仅 flat 模式有效）
    sleep_seconds: float = 0.0  # 每次 API 调用后的休眠时间（秒），用于控制请求频率
    overwrite: bool = False  # 是否覆盖已存在的数据
    limit: int = 0  # 处理的最大书籍数量，0 表示处理全部书籍


def run_preprocess_pipeline(config: PreprocessPipelineConfig) -> dict[str, object]:
    """执行预处理 pipeline。

    Args:
            config: pipeline 配置对象

    Returns:
            流程统计信息
    """

    # 输入数据
    input_path = Path(config.input_path)
    output_path = Path(config.output_path)

    result: dict[str, object] = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "steps": [],
    }

    if config.enable_text_polish:
        # 开始进行润色
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
        # 开始润色的embedding生成入库
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
    # 进入llm标签打标逻辑
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
