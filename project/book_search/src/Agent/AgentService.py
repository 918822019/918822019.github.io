"""
Agent 服务模块
负责协调 LLM、Embedding 和 Reranker 客户端，实现智能代理功能
"""

from collections import defaultdict
from typing import DefaultDict, List, Optional, Tuple
from ..llm.llm_client import LLMClient
from ..llm.embedding_client import EmbeddingClient
from ..llm.reranker_client import RerankerClient


class Agent:
    """智能代理类，整合 LLM、Embedding 和 Reranker 功能"""

    DEFAULT_PARALLEL_REWRITE_MODES = ["expansion", "clarification", "hyde"]
    RETRIEVAL_STRATEGY_ALIASES = {
        "single": "single",
        "late_fusion": "late_fusion",
        "parallel": "late_fusion",
        "multi": "late_fusion",
        "multi_query": "late_fusion",
        "early_fusion": "early_fusion",
        "fusion": "early_fusion",
        "aggregated": "early_fusion",
    }

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        reranker_client: Optional[RerankerClient] = None,
    ):
        """
        初始化 Agent

        Args:
            llm_client: LLM 客户端实例（可选，未提供则自动创建）
            embedding_client: Embedding 客户端实例（可选，未提供则自动创建）
            reranker_client: Reranker 客户端实例（可选，未提供则自动创建）
        """
        self.llm_client = llm_client or LLMClient()
        self.embedding_client = embedding_client or EmbeddingClient()
        self.reranker_client = reranker_client or RerankerClient()

    def process_query(self, query: str, context: Optional[str] = None) -> str:
        """
        处理用户查询

        Args:
            query: 用户查询文本
            context: 可选的上下文信息

        Returns:
            生成的回答
        """
        if context:
            return self.llm_client.generate_with_context(query, context)
        return self.llm_client.generate(query)

    def embed_text(self, text: str) -> List[float]:
        """
        将文本转换为向量

        Args:
            text: 输入文本

        Returns:
            文本的向量表示
        """
        return self.embedding_client.embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量将文本转换为向量

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        return self.embedding_client.embed_batch(texts)

    def rerank_documents(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        对文档进行重排序

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回前 k 个结果

        Returns:
            按相关性排序的 (原始索引, 相关性分数) 元组列表
        """
        return self.reranker_client.rerank(query, documents, top_k)

    def rewrite_query(
        self,
        query: str,
        mode: str = "expansion",
        context: Optional[str] = None,
    ) -> str:
        """
        重写用户查询以优化检索效果

        Args:
            query: 原始查询文本
            mode: 重写模式
                - 'expansion': 查询扩展，添加相关词汇
                - 'clarification': 查询澄清，消除歧义
                - 'decomposition': 查询分解，拆分为子问题
                - 'hyde': HyDE 模式，生成假设性文档
            context: 可选的对话历史或上下文信息

        Returns:
            重写后的查询文本
        """
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
        """并行语义上的多策略改写，失败时回退到原始查询。"""
        rewrite_modes = modes or self.DEFAULT_PARALLEL_REWRITE_MODES
        rewritten_queries: List[str] = []

        for mode in rewrite_modes:
            try:
                rewritten = self.rewrite_query(
                    query, mode=mode, context=context
                ).strip()
            except Exception:  # noqa: BLE001
                continue

            if rewritten and rewritten not in rewritten_queries:
                rewritten_queries.append(rewritten)

        if not rewritten_queries:
            return [query]

        return rewritten_queries

    def _normalize_strategy(self, strategy: str) -> str:
        """标准化检索策略名称，并校验是否支持。"""
        normalized = self.RETRIEVAL_STRATEGY_ALIASES.get(strategy)
        if normalized is None:
            supported = sorted(self.RETRIEVAL_STRATEGY_ALIASES.keys())
            raise ValueError(f"不支持的检索策略: {strategy}，可选: {supported}")
        return normalized

    def _build_fused_query(self, query: str, rewritten_queries: List[str]) -> str:
        """
        将原始查询与多种改写结果聚合为单个查询文本
        
        用于早期融合（early fusion）策略，将多个查询版本合并为一个综合查询，
        以便在一次向量检索中利用所有语义信息。
        
        Args:
            query: 原始用户查询
            rewritten_queries: 通过不同改写模式生成的查询列表
            
        Returns:
            用换行符分隔的聚合查询文本
        """
        parts = [query]
        for rewritten in rewritten_queries:
            if rewritten and rewritten not in parts:
                parts.append(rewritten)
        return "\n".join(parts)

    def _retrieve_by_query(
        self,
        query: str,
        candidate_embeddings: List[List[float]],
        top_k: int,
    ) -> List[Tuple[int, float]]:
        """
        对单个查询执行一次向量相似度检索
        
        将查询文本转换为向量，然后在候选文档向量库中查找最相似的文档。
        
        Args:
            query: 查询文本
            candidate_embeddings: 候选文档的向量表示列表
            top_k: 返回最相似的前 k 个文档
            
        Returns:
            按相似度排序的 (文档索引, 相似度分数) 元组列表
        """
        query_embedding = self.embed_text(query)
        return self.embedding_client.search_similar(
            query_embedding, candidate_embeddings, top_k=top_k
        )

    def _reciprocal_rank_fusion(
        self,
        ranked_lists: List[List[Tuple[int, float]]],
        top_k: Optional[int] = None,
        k: int = 60,
    ) -> List[Tuple[int, float]]:
        """
        使用倒数排名融合（RRF）算法合并多路排序结果
        
        RRF 是一种无监督的排序融合方法，通过计算每个文档在多个排序列表中的
        倒数排名之和来确定最终排序。公式：score = Σ(1 / (k + rank))
        
        Args:
            ranked_lists: 多个排序结果列表，每个列表包含 (文档索引, 分数) 元组
            top_k: 返回融合后的前 k 个结果（None 表示返回全部）
            k: RRF 平滑参数，默认为 60，用于降低排名差异的影响
            
        Returns:
            融合后按分数降序排列的 (文档索引, 融合分数) 元组列表
        """
        fused_scores: DefaultDict[int, float] = defaultdict(float)

        for ranked_list in ranked_lists:
            for rank, (doc_idx, _) in enumerate(ranked_list, start=1):
                fused_scores[doc_idx] += 1.0 / (k + rank)

        fused = sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))
        if top_k is not None:
            fused = fused[:top_k]
        return fused

    def _build_expansion_prompt(self, query: str, context: Optional[str] = None) -> str:
        """构建查询扩展提示词"""
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
        """构建查询澄清提示词"""
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
        """构建查询分解提示词"""
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
        """构建 HyDE (Hypothetical Document Embeddings) 提示词"""
        context_info = f"\n对话历史：\n{context}" if context else ""
        return f"""请根据以下问题，生成一段假设性的答案或相关文档片段。这段文本将用于向量检索，找到真正的相关文档。

要求：
1. 基于你的知识生成合理的内容
2. 长度控制在 100-200 字之间
3. 包含可能出现在真实文档中的关键词和信息
4. 只输出假设性文档，不要有任何解释

问题：{query}{context_info}

假设性文档："""

    def search_and_answer(
        self,
        query: str,
        candidate_texts: List[str],
        top_k: int = 5,
        use_rewrite: bool = True,
        rewrite_mode: str = "expansion",
        rewrite_modes: Optional[List[str]] = None,
        strategy: str = "single",
        retrieval_strategy: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        """
        完整的搜索问答流程：查询重写 -> 向量检索 -> 重排序 -> 生成回答
        
        支持三种检索策略：
        1. single（单路检索）：使用单个查询（可选重写）进行向量检索
        2. early_fusion（早期融合）：将多个改写查询聚合为一个综合查询后检索
        3. late_fusion（晚期融合）：对多个查询分别检索，然后使用 RRF 融合结果
        
        流程说明：
        - 首先对所有候选文档进行向量化（批量处理）
        - 根据选择的策略执行向量检索，获取初步筛选的文档
        - 使用 Reranker 对初步筛选的文档进行精排（使用原始查询）
        - 将精排后的文档作为上下文，结合原始查询生成最终回答
        
        Args:
            query: 用户原始查询文本
            candidate_texts: 候选文档文本列表
            top_k: 重排序后保留的最相关文档数量
            use_rewrite: 是否启用查询重写功能
            rewrite_mode: 单路重写时使用的重写模式（expansion/clarification/decomposition/hyde）
            rewrite_modes: 多路重写时使用的模式列表，默认为 ["expansion", "clarification", "hyde"]
            strategy: 检索策略名称（single/late_fusion/early_fusion）
            retrieval_strategy: strategy 参数的别名，优先级更高，便于兼容外部调用
            context: 可选的对话历史上下文，用于辅助查询重写
            
        Returns:
            基于最相关文档生成的自然语言回答
        """
        if not candidate_texts:
            return self.process_query(query)

        normalized_strategy = self._normalize_strategy(retrieval_strategy or strategy)

        # 1. 获取所有候选文本的向量
        candidate_embeddings = self.embed_batch(candidate_texts)

        initial_top_k = min(len(candidate_texts), max(top_k * 2, top_k))

        if normalized_strategy == "single":
            retrieval_query = query
            if use_rewrite:
                retrieval_query = self.rewrite_query(
                    query, mode=rewrite_mode, context=context
                )
            ranked_candidates = self._retrieve_by_query(
                retrieval_query, candidate_embeddings, top_k=initial_top_k
            )
        elif normalized_strategy == "early_fusion":
            rewritten_queries = []
            if use_rewrite:
                rewritten_queries = self.rewrite_parallel(
                    query, modes=rewrite_modes, context=context
                )
            fused_query = self._build_fused_query(query, rewritten_queries)
            ranked_candidates = self._retrieve_by_query(
                fused_query, candidate_embeddings, top_k=initial_top_k
            )
        else:
            retrieval_queries = [query]
            if use_rewrite:
                parallel_queries = self.rewrite_parallel(
                    query, modes=rewrite_modes, context=context
                )
                for rewritten in parallel_queries:
                    if rewritten not in retrieval_queries:
                        retrieval_queries.append(rewritten)

            ranked_lists = [
                self._retrieve_by_query(
                    retrieval_query, candidate_embeddings, top_k=initial_top_k
                )
                for retrieval_query in retrieval_queries
            ]
            ranked_candidates = self._reciprocal_rank_fusion(
                ranked_lists, top_k=initial_top_k
            )

        # 2. 提取初步筛选的文本
        filtered_texts = [candidate_texts[idx] for idx, _ in ranked_candidates]

        # 3. 使用 Reranker 进行精排（使用原始查询）
        reranked_results = self.rerank_documents(
            query, filtered_texts, top_k=min(top_k, len(filtered_texts))
        )

        # 4. 构建上下文
        context = "\n\n".join([filtered_texts[idx] for idx, _ in reranked_results])

        # 5. 生成回答（使用原始查询）
        return self.process_query(query, context)
