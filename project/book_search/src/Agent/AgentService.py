"""
Agent 服务模块
负责协调 LLM、Embedding 和 Reranker 客户端，实现智能代理功能
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Sequence, Tuple, cast

from ..llm.embedding_client import EmbeddingClient
from ..llm.llm_client import LLMClient
from ..llm.reranker_client import RerankerClient

SearchResult = Tuple[int, float]


class Agent:
    """智能代理类，整合 LLM、Embedding 和 Reranker 功能。"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        reranker_client: Optional[RerankerClient] = None,
    ):
        """
        初始化 Agent。

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
        处理用户查询。

        Args:
            query: 用户查询文本
            context: 可选的上下文信息

        Returns:
            生成的回答
        """
        if context:
            generate_with_context = cast(
                Callable[..., str],
                self.llm_client.generate_with_context,
            )
            return generate_with_context(query, context)

        generate = cast(Callable[..., str], self.llm_client.generate)
        return generate(query)

    def embed_text(self, text: str) -> List[float]:
        """
        将文本转换为向量。

        Args:
            text: 输入文本

        Returns:
            文本的向量表示
        """
        return self.embedding_client.embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量将文本转换为向量。

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        return self.embedding_client.embed_batch(texts)

    def rerank_documents(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        对文档进行重排序。

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
        重写用户查询以优化检索效果。

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
            supported_modes = list(rewrite_prompts.keys())
            raise ValueError(f"不支持的重写模式: {mode}，可选: {supported_modes}")

        prompt_builder = rewrite_prompts[mode]
        prompt = prompt_builder(query, context)
        generate = cast(Callable[..., str], self.llm_client.generate)
        rewritten = generate(prompt, temperature=0.3)
        return rewritten.strip()

    def _build_expansion_prompt(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> str:
        """构建查询扩展提示词。"""
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
        self,
        query: str,
        context: Optional[str] = None,
    ) -> str:
        """构建查询澄清提示词。"""
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
        self,
        query: str,
        context: Optional[str] = None,
    ) -> str:
        """构建查询分解提示词。"""
        context_info = f"\n对话历史：\n{context}" if context else ""
        return f"""你是一个问题分析专家。请将用户的复杂问题分解为 2-3 个更简单的子问题，便于分步检索。

要求：
1. 每个子问题应该是独立且完整的
2. 子问题之间应该有逻辑关联
3. 用分号分隔各个子问题
4. 只输出分解后的子问题，不要有任何解释

原始查询：{query}{context_info}

分解后的子问题："""

    def _build_hyde_prompt(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> str:
        """构建 HyDE 提示词。"""
        context_info = f"\n对话历史：\n{context}" if context else ""
        return f"""请根据以下问题，生成一段假设性的答案或相关文档片段。这段文本将用于向量检索，找到真正的相关文档。

要求：
1. 基于你的知识生成合理的内容
2. 长度控制在 100-200 字之间
3. 包含可能出现在真实文档中的关键词和信息
4. 只输出假设性文档，不要有任何解释

问题：{query}{context_info}

假设性文档："""

    def _rewrite_parallel(
        self,
        query: str,
        context: Optional[str] = None,
        modes: Optional[Sequence[str]] = None,
    ) -> Dict[str, str]:
        """
        并发执行多种查询重写策略。

        通过线程池并行调用不同的重写模式（如扩展、澄清、HyDE），
        提升多路召回的效率。任一路失败时自动降级为原始查询，
        避免单个重写失败阻断整体检索流程。

        Args:
            query: 原始查询文本
            context: 可选的对话历史或上下文信息
            modes: 要执行的重写模式列表，默认为 ["expansion", "clarification", "hyde"]

        Returns:
            字典，键为重写模式，值为重写后的查询文本
        """
        selected_modes = list(modes or ["expansion", "clarification", "hyde"])
        if not selected_modes:
            return {}

        results: Dict[str, str] = {}
        max_workers = min(len(selected_modes), 3)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_mode = {
                executor.submit(
                    self.rewrite_query,
                    query,
                    mode=mode,
                    context=context,
                ): mode
                for mode in selected_modes
            }
            for future in as_completed(future_to_mode):
                mode = future_to_mode[future]
                try:
                    rewritten = future.result().strip()
                    results[mode] = rewritten or query
                except Exception:
                    results[mode] = query

        return results

    def _search_candidates(
        self,
        search_query: str,
        candidate_embeddings: List[List[float]],
        top_k: int,
    ) -> List[SearchResult]:
        """
        基于查询向量在候选向量中执行一次向量相似度召回。

        将查询文本转换为向量后，与候选文档的预计算向量进行相似度比较，
        返回最相似的 top_k 个结果。

        Args:
            search_query: 用于检索的查询文本
            candidate_embeddings: 候选文档的向量表示列表
            top_k: 返回前 k 个最相似的结果

        Returns:
            按相似度排序的 (原始索引, 相似度分数) 元组列表
        """
        if not candidate_embeddings:
            return []

        query_embedding = self.embed_text(search_query)
        search_similar = cast(
            Callable[..., List[SearchResult]],
            self.embedding_client.search_similar,
        )
        return search_similar(
            query_embedding,
            candidate_embeddings,
            top_k=top_k,
        )

    def _search_single_path(
        self,
        search_query: str,
        candidate_texts: List[str],
        top_k: int,
        candidate_embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """
        执行单路向量召回，返回供 rerank 重排序使用的候选文本列表。

        如果未提供预计算的候选向量，则先批量计算向量表示。
        召回数量会适当放大（top_k * 2），为后续重排序留出更多候选。

        Args:
            search_query: 用于检索的查询文本
            candidate_texts: 候选文档的文本列表
            top_k: 期望的最终返回数量
            candidate_embeddings: 可选的预计算候选向量，避免重复计算

        Returns:
            召回的候选文本列表（未重排序）
        """
        if not candidate_texts:
            return []

        embeddings = candidate_embeddings or self.embed_batch(candidate_texts)
        candidate_limit = min(len(candidate_texts), max(top_k, top_k * 2))
        similarities = self._search_candidates(
            search_query,
            embeddings,
            candidate_limit,
        )
        return [candidate_texts[idx] for idx, _ in similarities]

    def _search_multi_path(
        self,
        rewritten_map: Dict[str, str],
        candidate_texts: List[str],
        top_k: int,
        candidate_embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """
        对多个重写查询分别执行向量召回，再用 RRF（倒数秩融合）算法合并结果。

        通过并发执行多路检索（如扩展查询、澄清查询、HyDE 查询），
        提升召回的多样性和覆盖率。所有检索路径共享同一套候选向量，
        确保索引一致性，避免多路检索后出现索引漂移问题。

        Args:
            rewritten_map: 重写模式到重写查询的映射字典
            candidate_texts: 候选文档的文本列表
            top_k: 期望的最终返回数量
            candidate_embeddings: 可选的预计算候选向量，避免重复计算

        Returns:
            经 RRF 融合后的候选文本列表（未重排序）
        """
        if not candidate_texts or not rewritten_map:
            return []

        embeddings = candidate_embeddings or self.embed_batch(candidate_texts)
        candidate_limit = min(len(candidate_texts), max(top_k, top_k * 2))
        all_results: List[Tuple[str, List[SearchResult]]] = []

        max_workers = min(len(rewritten_map), 3)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_mode = {
                executor.submit(
                    self._search_candidates,
                    rewritten_query,
                    embeddings,
                    candidate_limit,
                ): mode
                for mode, rewritten_query in rewritten_map.items()
            }
            for future in as_completed(future_to_mode):
                mode = future_to_mode[future]
                try:
                    search_results = future.result()
                except Exception:
                    search_results = []

                if search_results:
                    all_results.append((mode, search_results))

        fused_indices = self._fuse_results_rrf(
            all_results,
            top_k=candidate_limit,
        )
        return [candidate_texts[idx] for idx in fused_indices]

    def _fuse_results_rrf(
        self,
        all_results: Sequence[Tuple[str, List[SearchResult]]],
        top_k: int,
        k_constant: int = 60,
    ) -> List[int]:
        """
        使用 RRF（Reciprocal Rank Fusion，倒数秩融合）算法融合多路检索结果。

        RRF 是一种不依赖具体相似度分数的排名融合方法，通过累加各路的
        倒数排名得分来综合评估文档的相关性。公式：RRF_score = Σ(1 / (k + rank))
        其中 k 为常数（默认 60），rank 为文档在该路检索中的排名。

        Args:
            all_results: 多路检索结果列表，每路包含 (模式名, 搜索结果列表)
            top_k: 融合后返回的前 k 个结果
            k_constant: RRF 算法的平滑常数，默认为 60

        Returns:
            按 RRF 分数降序排列的原始索引列表
        """
        rrf_scores: Dict[int, float] = defaultdict(float)

        for _, ranked_list in all_results:
            for rank, (idx, _) in enumerate(ranked_list, start=1):
                rrf_scores[idx] += 1.0 / (k_constant + rank)

        sorted_indices = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        return [idx for idx, _ in sorted_indices[:top_k]]

    def _build_response_context(
        self,
        base_context: Optional[str],
        documents: List[str],
    ) -> Optional[str]:
        """
        将外部上下文和检索到的相关文档拼接成最终的回答上下文。

        按照「外部上下文 + 检索文档」的顺序组织内容，各部分之间用双换行符分隔，
        便于 LLM 理解信息的层次结构。

        Args:
            base_context: 可选的外部上下文信息（如对话历史）
            documents: 检索并重排序后的相关文档列表

        Returns:
            拼接后的上下文字符串，如果两者都为空则返回 None
        """
        sections: List[str] = []
        if base_context:
            sections.append(base_context)
        if documents:
            sections.append("\n\n".join(documents))
        if not sections:
            return None
        return "\n\n".join(sections)

    def search_and_answer(
        self,
        query: str,
        candidate_texts: List[str],
        top_k: int = 5,
        use_rewrite: bool = True,
        rewrite_mode: str = "expansion",
        search_strategy: str = "single",
        context: Optional[str] = None,
    ) -> str:
        """
        完整的搜索问答流程：查询重写 -> 向量检索 -> 重排序 -> 生成回答。

        支持两种检索策略：
        - single（单路）：使用单一重写查询进行检索，简单高效
        - multi（多路）：并发执行多种重写策略，通过 RRF 融合结果，提升召回质量

        工作流程：
        1. 查询重写（可选）：根据策略扩展、澄清或生成假设性文档
        2. 向量召回：将查询和候选文档转换为向量，计算相似度
        3. 重排序：使用 Reranker 模型对召回结果精排
        4. 生成回答：基于最相关的文档和上下文，调用 LLM 生成最终答案

        Args:
            query: 用户查询文本
            candidate_texts: 候选文档的文本列表
            top_k: 重排序后保留的文档数量，默认为 5
            use_rewrite: 是否启用查询重写，默认为 True
            rewrite_mode: 单路检索时的查询重写模式，默认为 "expansion"
            search_strategy: 检索策略，"single"（单路）或 "multi"（多路），默认为 "single"
            context: 可选的对话历史或外部上下文信息

        Returns:
            基于最相关文档生成的自然语言回答

        Raises:
            ValueError: 当 search_strategy 不是 "single" 或 "multi" 时抛出异常
        """
        if search_strategy not in {"single", "multi"}:
            raise ValueError(
                f"不支持的检索策略: {search_strategy}，可选: ['single', 'multi']"
            )

        if not candidate_texts:
            return self.process_query(query, context)

        retrieved_texts: List[str]
        if search_strategy == "multi" and use_rewrite:
            with ThreadPoolExecutor(max_workers=2) as executor:
                rewrite_future = executor.submit(
                    self._rewrite_parallel,
                    query,
                    context,
                )
                embedding_future = executor.submit(
                    self.embed_batch,
                    candidate_texts,
                )
                rewritten_map = rewrite_future.result()
                candidate_embeddings = embedding_future.result()

            retrieved_texts = self._search_multi_path(
                rewritten_map,
                candidate_texts,
                top_k=top_k,
                candidate_embeddings=candidate_embeddings,
            )
        else:
            rewritten_query = query
            if use_rewrite:
                rewritten_query = self.rewrite_query(
                    query,
                    mode=rewrite_mode,
                    context=context,
                )

            candidate_embeddings = self.embed_batch(candidate_texts)
            retrieved_texts = self._search_single_path(
                rewritten_query,
                candidate_texts,
                top_k=top_k,
                candidate_embeddings=candidate_embeddings,
            )

        if not retrieved_texts:
            return self.process_query(query, context)

        reranked_results = self.rerank_documents(
            query,
            retrieved_texts,
            top_k=top_k,
        )
        reranked_texts = [retrieved_texts[idx] for idx, _ in reranked_results]
        answer_context = self._build_response_context(context, reranked_texts)
        return self.process_query(query, answer_context)
