"""搜索问答代理模块，在 Agent 基础上增加检索策略与多路融合能力"""

from collections import defaultdict
from typing import DefaultDict, List, Optional, Tuple

from src.agent.agent import Agent


class SearchAgent(Agent):
    """搜索问答代理，在 Agent 基础上增加检索策略与多路融合能力"""

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

    def _normalize_strategy(self, strategy: str) -> str:
        """规范化检索策略名称，支持别名映射"""
        normalized = self.RETRIEVAL_STRATEGY_ALIASES.get(strategy)
        if normalized is None:
            supported = sorted(self.RETRIEVAL_STRATEGY_ALIASES.keys())
            raise ValueError(f"不支持的检索策略: {strategy}，可选: {supported}")
        return normalized

    def _build_fused_query(self, query: str, rewritten_queries: List[str]) -> str:
        """将原始查询与多个改写结果拼接为融合查询文本"""
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
        """对单个查询做向量检索，返回 (候选索引, 分数) 列表"""
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
        """使用倒数排名融合（RRF）算法合并多路排序结果"""
        fused_scores: DefaultDict[int, float] = defaultdict(float)

        for ranked_list in ranked_lists:
            for rank, (doc_idx, _) in enumerate(ranked_list, start=1):
                fused_scores[doc_idx] += 1.0 / (k + rank)

        fused = sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))
        if top_k is not None:
            fused = fused[:top_k]
        return fused

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
        """完整的搜索问答流程：查询重写 -> 向量检索 -> 重排序 -> 生成回答"""
        if not candidate_texts:
            return self.process_query(query)

        normalized_strategy = self._normalize_strategy(retrieval_strategy or strategy)

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

        filtered_texts = [candidate_texts[idx] for idx, _ in ranked_candidates]

        reranked_results = self.rerank_documents(
            query, filtered_texts, top_k=min(top_k, len(filtered_texts))
        )

        context_text = "\n\n".join(
            [filtered_texts[idx] for idx, _ in reranked_results]
        )

        return self.process_query(query, context_text)
