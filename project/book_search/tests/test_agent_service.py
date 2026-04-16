import os
import sys
from importlib import import_module

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)

Agent = import_module("book_search.src.Agent.AgentService").Agent


class DummyLLMClient:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append(("generate", prompt, kwargs))
        return f"generated:{prompt}"

    def generate_with_context(self, query, context, **kwargs):
        self.calls.append(("generate_with_context", query, context, kwargs))
        return f"answer:{query}|{context}"


class DummyEmbeddingClient:
    def __init__(self):
        self.batch_calls = 0

    def embed(self, text):
        return [float(len(text))]

    def embed_batch(self, texts):
        self.batch_calls += 1
        return [[float(index)] for index, _ in enumerate(texts)]

    def search_similar(self, query_embedding, candidates, top_k=5):
        query_value = int(query_embedding[0])
        if query_value == 10:
            ranking = [(0, 0.9), (1, 0.8), (2, 0.1)]
        elif query_value == 20:
            ranking = [(1, 0.95), (2, 0.7), (0, 0.2)]
        elif query_value == 30:
            ranking = [(2, 0.99), (1, 0.6), (0, 0.1)]
        else:
            ranking = [(2, 0.9), (0, 0.8), (1, 0.7)]
        return ranking[:top_k]


class DummyRerankerClient:
    def rerank(self, query, documents, top_k=None):
        scores = [(index, float(len(doc))) for index, doc in enumerate(documents)]
        scores.sort(key=lambda item: item[1], reverse=True)
        if top_k is not None:
            return scores[:top_k]
        return scores


def build_agent():
    return Agent(
        llm_client=DummyLLMClient(),
        embedding_client=DummyEmbeddingClient(),
        reranker_client=DummyRerankerClient(),
    )


def test_search_and_answer_keeps_single_strategy_as_default(monkeypatch):
    agent = build_agent()

    def fake_rewrite(query, mode="expansion", context=None):
        assert mode == "expansion"
        assert context is None
        return "single-query"

    monkeypatch.setattr(agent, "rewrite_query", fake_rewrite)
    monkeypatch.setattr(
        agent,
        "_search_single_path",
        lambda rewritten_query, candidate_texts, top_k, candidate_embeddings=None: [
            "short doc",
            "much longer doc",
        ],
    )

    answer = agent.search_and_answer(
        query="原始问题",
        candidate_texts=["doc1", "doc2", "doc3"],
        top_k=1,
    )

    assert answer == "answer:原始问题|much longer doc"
    assert agent.embedding_client.batch_calls == 1


def test_search_and_answer_uses_rrf_in_multi_strategy(monkeypatch):
    agent = build_agent()

    monkeypatch.setattr(
        agent,
        "_rewrite_parallel",
        lambda query, context=None, modes=None: {
            "expansion": "x" * 10,
            "clarification": "y" * 20,
            "hyde": "z" * 30,
        },
    )

    answer = agent.search_and_answer(
        query="原始问题",
        candidate_texts=["doc-a", "doc-bb", "doc-ccc"],
        top_k=2,
        search_strategy="multi",
        context="外部上下文",
    )

    assert answer == "answer:原始问题|外部上下文\n\ndoc-ccc\n\ndoc-bb"
    assert agent.embedding_client.batch_calls == 1


def test_rewrite_parallel_falls_back_to_original_query(monkeypatch):
    agent = build_agent()

    def fake_rewrite(query, mode="expansion", context=None):
        if mode == "clarification":
            raise RuntimeError("boom")
        return f"{mode}:{query}"

    monkeypatch.setattr(agent, "rewrite_query", fake_rewrite)

    rewritten_map = agent._rewrite_parallel("检索问题", context="历史")

    assert rewritten_map["expansion"] == "expansion:检索问题"
    assert rewritten_map["hyde"] == "hyde:检索问题"
    assert rewritten_map["clarification"] == "检索问题"


def test_search_and_answer_rejects_unknown_strategy():
    agent = build_agent()

    try:
        agent.search_and_answer(
            query="test",
            candidate_texts=["doc"],
            search_strategy="unknown",
        )
    except ValueError as exc:
        assert "不支持的检索策略" in str(exc)
    else:
        raise AssertionError("expected ValueError")
