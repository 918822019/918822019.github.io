
import os
import sys

# Ensure project root is on sys.path so tests can import the package when run from the tests folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from book_search.src.llm.llm_client import LLMClient


def test_generate_placeholder():
	"""LLMClient.generate 应该返回占位符响应并包含模型名"""
	# 使用显式的 model_name（不从 env 读取）
	client = LLMClient(model_name="test-model")
	res = client.generate("hello")
	assert "LLM 响应（待实现）" in res
	assert "test-model" in res


def test_chat_placeholder():
	"""LLMClient.chat 应该返回占位符响应并包含模型名"""
	client = LLMClient(model_name="chat-model")
	res = client.chat([{"role": "user", "content": "hi"}])
	assert "Chat 响应（待实现）" in res
	assert "chat-model" in res


def test_generate_with_context_calls_generate(monkeypatch):
	"""generate_with_context 应该构造包含上下文和问题的 prompt 并调用 generate"""
	# 为了做一个“真实”测试，从环境读取模型名
	monkeypatch.setenv("LLM_MODEL_NAME", "ctx-model")
	monkeypatch.setenv("LLM_API_KEY", "dummy-key")
	client = LLMClient()

	called = {}

	def fake_generate(prompt, **kwargs):
		called['prompt'] = prompt
		return "fake-response"

	# patch 实例的 generate 以便捕获 prompt 内容（仍然来源于真实 Env 配置）
	monkeypatch.setattr(client, "generate", fake_generate)

	res = client.generate_with_context("What is X?", "Here is relevant context.")

	assert res == "fake-response"
	assert "相关信息" in called['prompt']
	assert "What is X?" in called['prompt']


def test_env_get_llm_config():
	# 使用真实环境变量来验证返回值
	from book_search.src.llm.env import EnvConfig

	# 如果测试需要依赖环境值，使用 monkeypatch 在运行时设置（pytest fixture）
	# 这里不强制设置，测试可以在 CI 中使用已有的环境变量
	cfg = EnvConfig.get_llm_config()
	# Ensure keys exist
	assert set(cfg.keys()) >= {"base_url", "api_key", "model_name"}


def test_env_client_reads_env(monkeypatch):
	"""真实测试：从环境读取配置并构建客户端"""
	monkeypatch.setenv("LLM_MODEL_NAME", "env-model")
	monkeypatch.setenv("LLM_API_KEY", "env-key")
	monkeypatch.setenv("LLM_BASE_URL", "https://example.com/api")
	monkeypatch.setenv("EMBEDDING_API_KEY", "embed-key")

	# 模块在导入时会读取环境变量，因此在设置环境变量后需要 reload 模块以让类属性更新
	import importlib

	import book_search.src.llm.env as env_mod
	import book_search.src.llm.llm_client as llm_mod

	importlib.reload(env_mod)
	importlib.reload(llm_mod)

	from book_search.src.llm.env import EnvConfig
	from book_search.src.llm.llm_client import LLMClient

	# EnvConfig 应该能读取到我们设置的环境变量（经过 reload）
	cfg = EnvConfig.get_llm_config()
	assert cfg["model_name"] == "env-model"
	assert cfg["api_key"] == "env-key"

	# 构建客户端时不传 model_name，应从 EnvConfig 中读取（模块已 reload）
	client = LLMClient()
	res = client.generate("hello")
	assert "env-model" in client.model_name
	assert "env-model" in res

	# validate_config 需要 LLM_API_KEY 和 EMBEDDING_API_KEY，应该返回 True
	assert EnvConfig.validate_config() is True



