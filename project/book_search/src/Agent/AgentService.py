"""
Agent 服务模块
负责协调 LLM、Embedding 和 Reranker 客户端，实现智能代理功能
"""

from typing import Optional, List
from ..llm.llm_client import LLMClient
from ..llm.embedding_client import EmbeddingClient
from ..llm.reranker_client import RerankerClient


class Agent:
    """智能代理类，整合 LLM、Embedding 和 Reranker 功能"""

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
    ) -> List[tuple]:
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
            raise ValueError(f"不支持的重写模式: {mode}，可选: {list(rewrite_prompts.keys())}")

        prompt_builder = rewrite_prompts[mode]
        prompt = prompt_builder(query, context)

        rewritten = self.llm_client.generate(prompt, temperature=0.3)
        return rewritten.strip()

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

    def _build_clarification_prompt(self, query: str, context: Optional[str] = None) -> str:
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

    def _build_decomposition_prompt(self, query: str, context: Optional[str] = None) -> str:
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
    ) -> str:
        """
        完整的搜索问答流程：rewrite -> embedding -> rerank -> generate

        Args:
            query: 用户查询
            candidate_texts: 候选文本列表
            top_k: 重排序后保留的文档数量
            use_rewrite: 是否使用查询重写
            rewrite_mode: 查询重写模式

        Returns:
            基于最相关文档生成的回答
        """
        # 0. 查询重写（可选）
        if use_rewrite:
            rewritten_query = self.rewrite_query(query, mode=rewrite_mode)
        else:
            rewritten_query = query

        # 1. 获取重写后查询的向量表示
        query_embedding = self.embed_text(rewritten_query)

        # 2. 获取所有候选文本的向量
        candidate_embeddings = self.embed_batch(candidate_texts)

        # 3. 使用余弦相似度初步筛选
        similarities = self.embedding_client.search_similar(
            query_embedding, candidate_embeddings, top_k=top_k * 2
        )

        # 4. 提取初步筛选的文本
        filtered_texts = [candidate_texts[idx] for idx, _ in similarities]

        # 5. 使用 Reranker 进行精排（使用原始查询）
        reranked_results = self.rerank_documents(query, filtered_texts, top_k=top_k)

        # 6. 构建上下文
        context = "\n\n".join([filtered_texts[idx] for idx, _ in reranked_results])

        # 7. 生成回答（使用原始查询）
        return self.process_query(query, context)