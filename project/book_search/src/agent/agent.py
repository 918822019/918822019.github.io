"""Agent 基类模块，整合 LLM、Embedding 和 Reranker 基础操作"""

from typing import List, Optional, Tuple

from src.llm.client import EmbeddingClient, LLMClient, RerankerClient


class Agent:
    """智能代理类，整合 LLM、Embedding 和 Reranker 基础操作"""

    DEFAULT_PARALLEL_REWRITE_MODES = ["expansion", "clarification", "hyde"]

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        reranker_client: Optional[RerankerClient] = None,
    ):
        """初始化智能代理，可注入自定义客户端用于测试"""
        self.llm_client = llm_client or LLMClient()
        self.embedding_client = embedding_client or EmbeddingClient()
        self.reranker_client = reranker_client or RerankerClient()

    def process_query(self, query: str, context: Optional[str] = None) -> str:
        """处理用户查询，有上下文时走带语境的生成，否则直接生成"""
        if context:
            return self.llm_client.generate_with_context(query, context)
        return self.llm_client.generate(query)

    def embed_text(self, text: str) -> List[float]:
        """对单段文本生成向量表示"""
        return self.embedding_client.embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本向量"""
        return self.embedding_client.embed_batch(texts)

    def rerank_documents(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """对候选文档进行重排序"""
        return self.reranker_client.rerank(query, documents, top_k)

    def rewrite_query(
        self,
        query: str,
        mode: str = "expansion",
        context: Optional[str] = None,
    ) -> str:
        """按指定模式重写查询（扩展/澄清/分解/HyDE），返回重写后的文本"""
        rewrite_prompts = {
            "expansion": self._build_expansion_prompt,
            "clarification": self._build_clarification_prompt,
            "decomposition": self._build_decomposition_prompt,
            "hyde": self._build_hyde_prompt,
        }

        if mode not in rewrite_prompts:
            raise ValueError(
                f"不支持的重写模式: {mode}，可选: {list(rewrite_prompts.keys())}"
            )

        prompt_builder = rewrite_prompts[mode]
        prompt = prompt_builder(query, context)

        rewritten = self.llm_client.generate(prompt, temperature=0.3)
        return rewritten.strip()

    def rewrite_parallel(
        self,
        query: str,
        modes: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> List[str]:
        """并行语义上的多策略改写，失败时回退到原始查询"""
        rewrite_modes = modes or self.DEFAULT_PARALLEL_REWRITE_MODES
        rewritten_queries: List[str] = []

        for mode in rewrite_modes:
            try:
                rewritten = self.rewrite_query(
                    query, mode=mode, context=context
                ).strip()
            except Exception:
                continue

            if rewritten and rewritten not in rewritten_queries:
                rewritten_queries.append(rewritten)

        if not rewritten_queries:
            return [query]

        return rewritten_queries

    def _build_expansion_prompt(self, query: str, context: Optional[str] = None) -> str:
        """构建查询扩展的提示词，添加同义词和近义词"""
        context_info = f"\n对话历史：\n{context}" if context else ""
        return f"""你是一个查询优化专家。请对用户的问题进行扩展，添加相关的同义词、近义词和相关概念，以便更好地进行信息检索。

要求：
1. 保持原问题的核心意图不变
2. 添加 2-3 个相关的关键词或短语
3. 输出简洁，不超过 50 字
4. 只输出扩展后的查询，不要有任何解释

原始查询：{query}{context_info}

扩展后的查询："""

    def _build_clarification_prompt(
        self, query: str, context: Optional[str] = None
    ) -> str:
        """构建查询澄清的提示词，消除歧义"""
        context_info = f"\n对话历史：\n{context}" if context else ""
        return f"""你是一个查询澄清专家。请分析用户问题中的歧义或不清晰之处，并重写为更明确、具体的查询。

要求：
1. 消除代词指代不明（如"它"、"这个"等）
2. 补充隐含的背景信息
3. 使查询更加具体和可操作
4. 只输出澄清后的查询，不要有任何解释

原始查询：{query}{context_info}

澄清后的查询："""

    def _build_decomposition_prompt(
        self, query: str, context: Optional[str] = None
    ) -> str:
        """构建查询分解的提示词，拆分子问题"""
        context_info = f"\n对话历史：\n{context}" if context else ""
        return f"""你是一个问题分析专家。请将用户的复杂问题分解为 2-3 个更简单的子问题，便于分步检索。

要求：
1. 每个子问题应该是独立且完整的
2. 子问题之间应该有逻辑关联
3. 用分号分隔各个子问题
4. 只输出分解后的子问题，不要有任何解释

原始查询：{query}{context_info}

分解后的子问题："""

    def _build_hyde_prompt(self, query: str, context: Optional[str] = None) -> str:
        """构建 HyDE 假设性文档的提示词"""
        context_info = f"\n对话历史：\n{context}" if context else ""
        return f"""请根据以下问题，生成一段假设性的答案或相关文档片段。这段文本将用于向量检索，找到真正的相关文档。

要求：
1. 基于你的知识生成合理的内容
2. 长度控制在 100-200 字之间
3. 包含可能出现在真实文档中的关键词和信息
4. 只输出假设性文档，不要有任何解释

问题：{query}{context_info}

假设性文档："""
