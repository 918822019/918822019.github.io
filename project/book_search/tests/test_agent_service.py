"""Agent / SearchAgent 全面单元测试。"""

# pyright: reportMissingImports=false

import os
import sys
from typing import List

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.agent import Agent, SearchAgent


class DummyLLMClient:
    def generate(self, prompt, **kwargs):
        return f"generated:{prompt}"

    def generate_with_context(self, query, context, **kwargs):
        return f"query={query}\ncontext={context}"


class DummyEmbeddingClient:
    def __init__(self, search_results=None):
        self.search_results = search_results or {}
        self.embed_calls: List[str] = []
        self.batch_calls: List[List[str]] = []

    def embed(self, text: str):
        self.embed_calls.append(text)
        return text

    def embed_batch(self, texts: List[str]):
        self.batch_calls.append(texts)
        return texts

    def search_similar(self, query_embedding, candidates, top_k=5):
        return self.search_results.get(query_embedding, [])[:top_k]


class DummyRerankerClient:
    def rerank(self, query, documents, top_k=None):
        ranked = [(idx, float(len(documents) - idx)) for idx, _ in enumerate(documents)]
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked


def make_agent() -> Agent:
    return Agent(
        llm_client=DummyLLMClient(),
        embedding_client=DummyEmbeddingClient(),
        reranker_client=DummyRerankerClient(),
    )


def make_search_agent(search_results=None) -> SearchAgent:
    return SearchAgent(
        llm_client=DummyLLMClient(),
        embedding_client=DummyEmbeddingClient(search_results=search_results),
        reranker_client=DummyRerankerClient(),
    )


# ==============================
# Agent — process_query
# ==============================

class TestAgentProcessQuery:
    def test_without_context(self):
        agent = make_agent()
        result = agent.process_query("推荐一本玄幻小说")
        assert "generated:推荐一本玄幻小说" in result

    def test_with_context(self):
        agent = make_agent()
        result = agent.process_query("推荐一本小说", context="用户喜欢废柴逆袭")
        assert "query=推荐一本小说" in result
        assert "context=用户喜欢废柴逆袭" in result

    def test_with_empty_context(self):
        agent = make_agent()
        result = agent.process_query("推荐一本小说", context="")
        assert "generated:推荐一本小说" in result


# ==============================
# Agent — embed / embed_batch / rerank
# ==============================

class TestAgentEmbed:
    def test_embed_text(self):
        agent = make_agent()
        result = agent.embed_text("斗破苍穹")
        assert result == "斗破苍穹"

    def test_embed_batch(self):
        agent = make_agent()
        texts = ["简介A", "简介B", "简介C"]
        result = agent.embed_batch(texts)
        assert result == texts

    def test_embed_batch_empty(self):
        agent = make_agent()
        assert agent.embed_batch([]) == []


class TestAgentRerank:
    def test_rerank_documents(self):
        agent = make_agent()
        docs = ["文档A", "文档B", "文档C"]
        result = agent.rerank_documents("玄幻小说", docs, top_k=2)
        assert len(result) == 2
        assert result[0] == (0, 3.0)
        assert result[1] == (1, 2.0)

    def test_rerank_top_k_larger_than_docs(self):
        agent = make_agent()
        docs = ["文档A"]
        result = agent.rerank_documents("玄幻小说", docs, top_k=5)
        assert len(result) == 1

    def test_rerank_empty_docs(self):
        agent = make_agent()
        assert agent.rerank_documents("玄幻小说", []) == []


# ==============================
# Agent — rewrite_query
# ==============================

class TestAgentRewriteQuery:
    def test_expansion_mode(self, monkeypatch):
        agent = make_agent()
        monkeypatch.setattr(agent.llm_client, "generate", lambda prompt, **kw: "玄幻 奇幻 魔幻 修仙")
        result = agent.rewrite_query("玄幻", mode="expansion")
        assert "玄幻 奇幻" in result

    def test_clarification_mode(self, monkeypatch):
        agent = make_agent()
        monkeypatch.setattr(agent.llm_client, "generate", lambda prompt, **kw: "《斗破苍穹》这本小说")
        result = agent.rewrite_query("那本书", mode="clarification")
        assert "斗破苍穹" in result

    def test_decomposition_mode(self, monkeypatch):
        agent = make_agent()
        monkeypatch.setattr(agent.llm_client, "generate", lambda prompt, **kw: "玄幻题材; 评分高; 主角有特点")
        result = agent.rewrite_query("好看的玄幻", mode="decomposition")
        assert "玄幻题材" in result

    def test_hyde_mode(self, monkeypatch):
        agent = make_agent()
        monkeypatch.setattr(agent.llm_client, "generate", lambda prompt, **kw: "一本讲述少年修炼的玄幻小说")
        result = agent.rewrite_query("玄幻", mode="hyde")
        assert "少年修炼" in result

    def test_unknown_mode_raises(self):
        agent = make_agent()
        with pytest.raises(ValueError, match="不支持的重写模式"):
            agent.rewrite_query("玄幻", mode="unknown")

    def test_default_mode_is_expansion(self, monkeypatch):
        agent = make_agent()
        monkeypatch.setattr(agent.llm_client, "generate", lambda prompt, **kw: "玄幻 奇幻")
        result = agent.rewrite_query("玄幻")
        assert result == "玄幻 奇幻"

    def test_context_passed_to_prompt(self, monkeypatch):
        agent = make_agent()
        captured: List[str] = []

        def fake_generate(prompt, **kw):
            captured.append(prompt)
            return "扩展结果"

        monkeypatch.setattr(agent.llm_client, "generate", fake_generate)
        agent.rewrite_query("玄幻", mode="expansion", context="历史对话")
        assert "历史对话" in captured[0]


# ==============================
# Agent — rewrite_parallel
# ==============================

class TestAgentRewriteParallel:
    def test_default_modes(self, monkeypatch):
        agent = make_agent()
        monkeypatch.setattr(agent.llm_client, "generate", lambda prompt, **kw: "改写结果")
        result = agent.rewrite_parallel("玄幻小说")
        assert len(result) == 1
        assert "改写结果" in result

    def test_custom_modes(self, monkeypatch):
        agent = make_agent()
        calls: List[str] = []

        def fake_generate(prompt, **kw):
            calls.append(prompt)
            return f"结果{len(calls)}"

        monkeypatch.setattr(agent.llm_client, "generate", fake_generate)
        result = agent.rewrite_parallel("玄幻", modes=["expansion", "hyde"])
        assert len(result) == 2
        assert len(calls) == 2

    def test_all_modes_fail_falls_back(self, monkeypatch):
        agent = make_agent()

        def fake_rewrite(query, mode="expansion", context=None):
            raise RuntimeError("API 不可用")

        monkeypatch.setattr(agent, "rewrite_query", fake_rewrite)
        result = agent.rewrite_parallel("原始问题", modes=["expansion", "hyde"])
        assert result == ["原始问题"]

    def test_some_modes_fail_filtered(self, monkeypatch):
        agent = make_agent()

        def fake_rewrite(query, mode="expansion", context=None):
            if mode == "expansion":
                return "扩展结果"
            raise RuntimeError("失败")

        monkeypatch.setattr(agent, "rewrite_query", fake_rewrite)
        result = agent.rewrite_parallel("原始问题", modes=["expansion", "hyde", "clarification"])
        assert result == ["扩展结果"]

    def test_empty_result_filters_empty_string(self, monkeypatch):
        agent = make_agent()

        def fake_generate(prompt, **kw):
            return ""

        monkeypatch.setattr(agent.llm_client, "generate", fake_generate)
        result = agent.rewrite_parallel("原始问题", modes=["expansion"])
        assert result == ["原始问题"]

    def test_dedup_results(self, monkeypatch):
        agent = make_agent()

        def fake_rewrite(query, mode="expansion", context=None):
            return "相同结果"

        monkeypatch.setattr(agent, "rewrite_query", fake_rewrite)
        result = agent.rewrite_parallel("原始问题", modes=["expansion", "hyde"])
        assert result == ["相同结果"]


# ==============================
# Agent — prompt builders
# ==============================

class TestAgentPromptBuilders:
    def test_expansion_prompt_contains_query(self):
        agent = make_agent()
        prompt = agent._build_expansion_prompt("玄幻小说")
        assert "玄幻小说" in prompt
        assert "扩展" in prompt

    def test_clarification_prompt_contains_query(self):
        agent = make_agent()
        prompt = agent._build_clarification_prompt("那本书")
        assert "那本书" in prompt
        assert "澄清" in prompt

    def test_decomposition_prompt_contains_query(self):
        agent = make_agent()
        prompt = agent._build_decomposition_prompt("好看的玄幻")
        assert "好看的玄幻" in prompt
        assert "分解" in prompt

    def test_hyde_prompt_contains_query(self):
        agent = make_agent()
        prompt = agent._build_hyde_prompt("玄幻")
        assert "玄幻" in prompt
        assert "假设" in prompt

    def test_prompt_with_context(self):
        agent = make_agent()
        prompt = agent._build_expansion_prompt("玄幻", context="之前聊过")
        assert "之前聊过" in prompt

    def test_prompt_without_context(self):
        agent = make_agent()
        prompt = agent._build_expansion_prompt("玄幻")
        assert "对话历史" not in prompt


# ==============================
# SearchAgent — _normalize_strategy
# ==============================

class TestSearchAgentNormalizeStrategy:
    def test_single(self):
        agent = make_search_agent()
        assert agent._normalize_strategy("single") == "single"

    @pytest.mark.parametrize("alias", ["late_fusion", "parallel", "multi", "multi_query"])
    def test_late_fusion_aliases(self, alias):
        agent = make_search_agent()
        assert agent._normalize_strategy(alias) == "late_fusion"

    @pytest.mark.parametrize("alias", ["early_fusion", "fusion", "aggregated"])
    def test_early_fusion_aliases(self, alias):
        agent = make_search_agent()
        assert agent._normalize_strategy(alias) == "early_fusion"

    def test_unknown_raises(self):
        agent = make_search_agent()
        with pytest.raises(ValueError, match="不支持的检索策略"):
            agent._normalize_strategy("unknown")


# ==============================
# SearchAgent — _build_fused_query
# ==============================

class TestSearchAgentBuildFusedQuery:
    def test_basic_concatenation(self):
        agent = make_search_agent()
        result = agent._build_fused_query("原始", ["扩展1", "扩展2"])
        assert result == "原始\n扩展1\n扩展2"

    def test_dedup_removes_duplicates(self):
        agent = make_search_agent()
        result = agent._build_fused_query("原始", ["扩展1", "原始", "扩展2"])
        assert result.split("\n") == ["原始", "扩展1", "扩展2"]

    def test_empty_rewritten(self):
        agent = make_search_agent()
        result = agent._build_fused_query("原始", [])
        assert result == "原始"

    def test_empty_strings_filtered(self):
        agent = make_search_agent()
        result = agent._build_fused_query("原始", ["", "扩展1", ""])
        assert result.split("\n") == ["原始", "扩展1"]


# ==============================
# SearchAgent — _reciprocal_rank_fusion
# ==============================

class TestSearchAgentRRF:
    def test_basic_rrf(self):
        agent = make_search_agent()
        lists = [
            [(0, 0.9), (1, 0.8)],
            [(1, 0.9), (2, 0.8)],
        ]
        result = agent._reciprocal_rank_fusion(lists)
        assert len(result) == 3
        assert result[0][0] == 1
        assert result[1][0] == 0
        assert result[2][0] == 2

    def test_rrf_with_top_k(self):
        agent = make_search_agent()
        lists = [
            [(0, 0.9), (1, 0.8), (2, 0.7)],
            [(1, 0.9), (2, 0.8), (0, 0.7)],
        ]
        result = agent._reciprocal_rank_fusion(lists, top_k=2)
        assert len(result) == 2

    def test_single_ranked_list(self):
        agent = make_search_agent()
        lists = [[(2, 0.9), (0, 0.8)]]
        result = agent._reciprocal_rank_fusion(lists)
        assert result == [(2, 1/61), (0, 1/62)]

    def test_empty_lists(self):
        agent = make_search_agent()
        result = agent._reciprocal_rank_fusion([[], []])
        assert result == []


# ==============================
# SearchAgent — _retrieve_by_query
# ==============================

class TestSearchAgentRetrieveByQuery:
    def test_delegates_to_embed_and_search(self):
        dummy_search = DummyEmbeddingClient(
            search_results={"测试查询": [(0, 0.95), (1, 0.90)]}
        )
        agent = SearchAgent(
            llm_client=DummyLLMClient(),
            embedding_client=dummy_search,
            reranker_client=DummyRerankerClient(),
        )
        candidates = [0.1, 0.2]
        result = agent._retrieve_by_query("测试查询", candidates, top_k=2)
        assert result == [(0, 0.95), (1, 0.90)]
        assert dummy_search.embed_calls[-1] == "测试查询"

    def test_top_k_limits_results(self):
        dummy_search = DummyEmbeddingClient(
            search_results={"查询": [(0, 0.9), (1, 0.8), (2, 0.7)]}
        )
        agent = SearchAgent(
            llm_client=DummyLLMClient(),
            embedding_client=dummy_search,
            reranker_client=DummyRerankerClient(),
        )
        result = agent._retrieve_by_query("查询", [0.1, 0.2, 0.3], top_k=2)
        assert len(result) == 2


# ==============================
# SearchAgent — search_and_answer
# ==============================

class TestSearchAgentSearchAndAnswer:
    def test_empty_candidates_falls_back_to_process_query(self, monkeypatch):
        agent = make_search_agent()
        monkeypatch.setattr(agent, "process_query", lambda q, ctx=None: f"fallback:{q}")
        result = agent.search_and_answer("测试问题", [])
        assert result == "fallback:测试问题"

    def test_single_strategy_without_rewrite(self, monkeypatch):
        agent = make_search_agent(
            search_results={"用户问题": [(0, 0.9)]}
        )
        def unexpected_call(*a, **kw):
            raise RuntimeError("rewrite_query should not be called")

        monkeypatch.setattr(agent, "rewrite_query", unexpected_call)
        result = agent.search_and_answer(
            "用户问题", ["文档0", "文档1"], top_k=1,
            use_rewrite=False
        )
        assert "文档0" in result

    def test_single_strategy_with_rewrite(self, monkeypatch):
        agent = make_search_agent(
            search_results={"改写": [(1, 0.9), (0, 0.8)]}
        )
        monkeypatch.setattr(agent, "rewrite_query", lambda q, **kw: "改写")
        result = agent.search_and_answer(
            "用户问题", ["文档A", "文档B"], top_k=2
        )
        assert "文档B" in result

    def test_early_fusion_strategy(self, monkeypatch):
        agent = make_search_agent(
            search_results={"原始\n改写1\n改写2": [(0, 0.9), (1, 0.8)]}
        )
        monkeypatch.setattr(agent, "rewrite_parallel", lambda q, **kw: ["改写1", "改写2"])
        result = agent.search_and_answer(
            "原始", ["文档0", "文档1"], top_k=2,
            strategy="early_fusion"
        )
        assert "文档0" in result

    def test_late_fusion_strategy(self, monkeypatch):
        agent = make_search_agent(
            search_results={
                "用户问题": [(2, 0.99)],
                "改写1": [(0, 0.98)],
                "改写2": [(1, 0.97)],
            }
        )
        monkeypatch.setattr(agent, "rewrite_parallel", lambda q, **kw: ["改写1", "改写2"])
        monkeypatch.setattr(agent, "process_query", lambda q, ctx=None: ctx or "")
        result = agent.search_and_answer(
            "用户问题", ["文档0", "文档1", "文档2"],
            top_k=2, strategy="late_fusion"
        )
        assert result

    def test_retrieval_strategy_alias(self, monkeypatch):
        agent = make_search_agent(
            search_results={"改写": [(0, 0.9)]}
        )
        monkeypatch.setattr(agent, "rewrite_query", lambda q, **kw: "改写")
        result = agent.search_and_answer(
            "用户问题", ["文档0", "文档1"], top_k=1,
            retrieval_strategy="parallel"
        )
        assert result

    def test_custom_rewrite_modes(self, monkeypatch):
        agent = make_search_agent(
            search_results={"改写": [(0, 0.9)]}
        )
        monkeypatch.setattr(agent, "rewrite_query", lambda q, **kw: "改写")
        result = agent.search_and_answer(
            "用户问题", ["文档0", "文档1"], top_k=1,
            rewrite_mode="hyde"
        )
        assert result

    def test_rejects_unknown_strategy(self):
        agent = make_search_agent()
        with pytest.raises(ValueError, match="不支持的检索策略"):
            agent.search_and_answer("用户问题", ["文档A"], strategy="unknown")

    def test_preserves_existing_behavior_single_default(self, monkeypatch):
        agent = make_search_agent(
            search_results={"改写后的查询": [(1, 0.9), (0, 0.8)]}
        )
        calls = {"rewrite": [], "rrf": 0}

        def fake_rewrite(query, mode="expansion", context=None):
            calls["rewrite"].append((query, mode, context))
            return "改写后的查询"

        def fake_rrf(*args, **kwargs):
            calls["rrf"] += 1
            return []

        monkeypatch.setattr(agent, "rewrite_query", fake_rewrite)
        monkeypatch.setattr(agent, "_reciprocal_rank_fusion", fake_rrf)

        answer = agent.search_and_answer("用户问题", ["文档A", "文档B"], top_k=1)

        assert calls["rewrite"] == [("用户问题", "expansion", None)]
        assert calls["rrf"] == 0
        assert "文档B" in answer

    def test_preserves_existing_behavior_rrf_multi(self, monkeypatch):
        agent = make_search_agent(
            search_results={
                "用户问题": [(2, 0.99), (0, 0.88)],
                "扩展查询": [(0, 0.98), (1, 0.85)],
                "澄清查询": [(1, 0.97), (0, 0.84)],
            }
        )
        captured = {}

        def fake_parallel(query, modes=None, context=None):
            return ["扩展查询", "澄清查询"]

        def fake_process(query, context=None):
            captured["context"] = context
            return context or ""

        monkeypatch.setattr(agent, "rewrite_parallel", fake_parallel)
        monkeypatch.setattr(agent, "process_query", fake_process)

        answer = agent.search_and_answer(
            "用户问题",
            ["文档0", "文档1", "文档2"],
            top_k=3,
            strategy="multi",
        )

        assert answer.split("\n\n") == ["文档0", "文档1", "文档2"]
        assert captured["context"].split("\n\n") == ["文档0", "文档1", "文档2"]
