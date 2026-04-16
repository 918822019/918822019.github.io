"""AgentService 检索策略测试。"""

# pyright: reportMissingImports=false

import os
import sys
import importlib

import pytest


sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)

Agent = importlib.import_module("book_search.src.Agent.AgentService").Agent


class DummyLLMClient:
    def generate(self, prompt, **kwargs):
        return prompt

    def generate_with_context(self, query, context, **kwargs):
        return f"query={query}\ncontext={context}"


class DummyEmbeddingClient:
    def __init__(self, search_results=None):
        self.search_results = search_results or {}

    def embed(self, text):
        return text

    def embed_batch(self, texts):
        return texts

    def search_similar(self, query_embedding, candidates, top_k=5):
        return self.search_results.get(query_embedding, [])[:top_k]


class DummyRerankerClient:
    def rerank(self, query, documents, top_k=None):
        ranked = [(idx, float(len(documents) - idx)) for idx, _ in enumerate(documents)]
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked


def build_agent(search_results=None):
    return Agent(
        llm_client=DummyLLMClient(),
        embedding_client=DummyEmbeddingClient(search_results=search_results),
        reranker_client=DummyRerankerClient(),
    )


def test_rewrite_parallel_falls_back_to_original_query(monkeypatch):
    agent = build_agent()

    def fake_rewrite(query, mode="expansion", context=None):
        if mode == "expansion":
            return ""
        raise RuntimeError("rewrite failed")

    monkeypatch.setattr(agent, "rewrite_query", fake_rewrite)

    assert agent.rewrite_parallel("原始问题", modes=["expansion", "hyde"]) == [
        "原始问题"
    ]


def test_search_and_answer_keeps_single_strategy_as_default(monkeypatch):
    agent = build_agent(
        search_results={
            "改写后的查询": [(1, 0.9), (0, 0.8)],
        }
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


def test_search_and_answer_rejects_unknown_strategy():
    agent = build_agent()

    with pytest.raises(ValueError, match="不支持的检索策略"):
        agent.search_and_answer("用户问题", ["文档A"], strategy="unknown")


def test_search_and_answer_uses_rrf_in_multi_strategy(monkeypatch):
    agent = build_agent(
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
