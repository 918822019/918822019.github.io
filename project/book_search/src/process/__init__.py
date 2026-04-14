"""数据处理模块包。

本包包含小说数据的各种预处理流程，包括:
- book_profile_polish: 使用 LLM 润色书名和简介
- llm_tagging: 使用 LLM 自动打标（支持扁平标签和四级级联分类）
- polish_embedding: 基于润色后的文本生成 embedding
- polish_embedding_search: 基于 embedding 的语义搜索
- preprocess: 预处理流程编排，整合多个步骤

使用方法:
    # 单独使用某个功能
    from src.process.book_profile_polish import run_polish
    from src.process.llm_tagging import LLMTagger, CascadingTagger
    from src.process.polish_embedding import run_polish_embedding
    
    # 使用完整的预处理 pipeline
    from src.process.preprocess import run_preprocess_pipeline, PreprocessPipelineConfig
    
    config = PreprocessPipelineConfig(
        input_path="data/books.db",
        output_path="data/books_tagged.json",
        enable_text_polish=True,
        enable_polish_embedding=True,
        enable_llm_tagging=True,
        tagging_mode="cascading"  # 使用级联标签
    )
    stats = run_preprocess_pipeline(config)
    print(stats)

工作流程建议:
    1. 数据准备：确保 books.db 中有基础数据
    2. 文本润色：运行 polish_book_profile.py 优化书名和简介
    3. Embedding生成：运行 embed_book_polish.py 生成向量
    4. LLM打标：使用 llm_tagging 模块进行自动分类
    5. 语义搜索：使用 search_book_polish_embedding.py 进行搜索
"""
