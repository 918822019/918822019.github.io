import os
import sys
import importlib

import pytest

# Ensure project root is on sys.path so tests can import the package when run from the tests folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from book_search.src.llm.llm_client import LLMClient


def test_generate_builds_messages_and_calls_transport(monkeypatch):
	"""generate 应该构造消息并调用底层请求方法。"""
	client = LLMClient(model_name="test-model", api_key="dummy-key")
	called = {}

	def fake_post(messages, **kwargs):
		called["messages"] = messages
		called["kwargs"] = kwargs
		return "fake-generate"

	monkeypatch.setattr(client, "_post_chat_completion", fake_post)

	res = client.generate("hello", system_prompt="you are helpful", temperature=0.5)

	assert res == "fake-generate"
	assert called["messages"][0] == {"role": "system", "content": "you are helpful"}
	assert called["messages"][1] == {"role": "user", "content": "hello"}
	assert called["kwargs"]["temperature"] == 0.5


def test_chat_passes_messages_to_transport(monkeypatch):
	"""chat 应该原样透传 messages。"""
	client = LLMClient(model_name="chat-model", api_key="dummy-key")
	called = {}

	def fake_post(messages, **kwargs):
		called["messages"] = messages
		called["kwargs"] = kwargs
		return "fake-chat"

	monkeypatch.setattr(client, "_post_chat_completion", fake_post)

	messages = [{"role": "user", "content": "hi"}]
	res = client.chat(messages, max_tokens=32)

	assert res == "fake-chat"
	assert called["messages"] == messages
	assert called["kwargs"]["max_tokens"] == 32


def test_generate_with_context_calls_generate(monkeypatch):
    """generate_with_context 应该构造包含上下文和问题的 prompt 并调用 generate。"""
    client = LLMClient(model_name="ctx-model", api_key="dummy-key")

    called = {}

    def fake_generate(prompt, **kwargs):
        called["prompt"] = prompt
        return "fake-response"

    monkeypatch.setattr(client, "generate", fake_generate)

    res = client.generate_with_context("What is X?", "Here is relevant context.")

    assert res == "fake-response"
    assert "相关信息" in called["prompt"]
    assert "What is X?" in called["prompt"]


def test_env_get_llm_config():
    from book_search.src.llm.env import EnvConfig

    cfg = EnvConfig.get_llm_config()
    assert set(cfg.keys()) >= {"base_url", "api_key", "model_name"}


def test_env_client_reads_env(monkeypatch):
	"""从环境读取配置并构建客户端，不触发真实网络请求。"""
	monkeypatch.setenv("LLM_MODEL_NAME", "env-model")
	monkeypatch.setenv("LLM_API_KEY", "env-key")
	monkeypatch.setenv("LLM_BASE_URL", "https://example.com/api")
	monkeypatch.setenv("EMBEDDING_API_KEY", "embed-key")

	import book_search.src.llm.env as env_mod
	import book_search.src.llm.llm_client as llm_mod

	importlib.reload(env_mod)
	importlib.reload(llm_mod)

	from book_search.src.llm.env import EnvConfig
	from book_search.src.llm.llm_client import LLMClient

	cfg = EnvConfig.get_llm_config()
	assert cfg["model_name"] == "env-model"
	assert cfg["api_key"] == "env-key"
	assert cfg["base_url"] == "https://example.com/api"

	client = LLMClient()
	assert client.model_name == "env-model"
	assert client.api_key == "env-key"
	assert client.base_url == "https://example.com/api"

	assert EnvConfig.validate_config() is True


@pytest.mark.live
def test_live_generate_with_full_key_and_baseurl():
	"""可选联调测试：使用完整 key/base_url/model_name 直连真实服务。"""
	if os.getenv("RUN_LIVE_LLM_TESTS") != "1":
		pytest.skip("Set RUN_LIVE_LLM_TESTS=1 to run live LLM integration tests.")

	base_url = os.getenv("LLM_BASE_URL", "").strip()
	api_key = os.getenv("LLM_API_KEY", "").strip()
	model_name = os.getenv("LLM_MODEL_NAME", "").strip()
	missing = [
		name
		for name, value in (
			("LLM_BASE_URL", base_url),
			("LLM_API_KEY", api_key),
			("LLM_MODEL_NAME", model_name),
		)
		if not value
	]
	if missing:
		pytest.skip(f"Missing live test env vars: {', '.join(missing)}")

	client = LLMClient(model_name=model_name, api_key=api_key, base_url=base_url)
	res = client.generate("请只回复: ok", temperature=0)

	assert isinstance(res, str)
	assert res.strip() != ""
