"""数据处理模块包。

本包包含小说数据的各种预处理流程，包括:
- polish/: 使用 LLM 润色书名和简介、生成 embedding、语义搜索
- taggers/: 使用 LLM 自动打标（支持扁平标签和四级级联分类）
- pipeline: 预处理流程编排，整合多个步骤

使用方法:
    # 润色 + embedding + 搜索
    from src.process.polish import run_polish, run_polish_embedding, search_books_by_polish_embedding
    
    # 打标签
    from src.process.taggers import LLMTagger, CascadingTagger
    
    # 完整的预处理 pipeline
    from src.process.pipeline import run_preprocess_pipeline, PreprocessPipelineConfig
"""
