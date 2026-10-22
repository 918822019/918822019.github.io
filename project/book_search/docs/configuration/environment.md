# 环境变量配置

本文档详细说明系统所有环境变量的配置方式。

## 配置文件

`.env` 文件用于存放密钥等敏感配置。非敏感参数（路径、模型名、超时等）放在 `config.yaml` 中。

### .env（密钥）

建议在 `project/book_search/.env` 文件中配置环境变量：

```bash
# LLM 配置
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o-mini

# Embedding 配置
EMBEDDING_API_KEY=your-api-key
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL_NAME=text-embedding-3-small

# 请求配置
REQUEST_TIMEOUT=60
```

### config.yaml（非敏感参数）

非敏感配置项（路径、模型名、超时等）放在 `project/book_search/config.yaml` 中：

```yaml
llm:
  model_name: gpt-4o-mini

request:
  timeout: 30
  max_retries: 3

data:
  dir: data
  db_name: books.db
  faiss_index_suffix: .polish_embedding.faiss
```

环境变量中的同名配置（如 `LLM_MODEL_NAME`）会覆盖 `config.yaml` 中的值。

完整配置项列表请参考 `config.yaml` 文件本身。

## LLM 配置

| 环境变量 | 必需 | 默认值 | 说明 |
|----------|------|--------|------|
| `LLM_API_KEY` | 是 | - | LLM API 密钥 |
| `LLM_BASE_URL` | 否 | `https://api.openai.com/v1` | LLM API 地址 |
| `LLM_MODEL_NAME` | 否 | `gpt-4o-mini` | LLM 模型名称 |

### 使用示例

```bash
# OpenAI
export LLM_API_KEY="sk-..."
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL_NAME="gpt-4o-mini"

# 阿里云 Qwen
export LLM_API_KEY="your-dashscope-key"
export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export LLM_MODEL_NAME="qwen-plus"

# 本地模型
export LLM_API_KEY="not-needed"
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_MODEL_NAME="llama3"
```

## Embedding 配置

| 环境变量 | 必需 | 默认值 | 说明 |
|----------|------|--------|------|
| `EMBEDDING_API_KEY` | 是 | - | Embedding API 密钥 |
| `EMBEDDING_BASE_URL` | 否 | `https://api.openai.com/v1` | Embedding API 地址 |
| `EMBEDDING_MODEL_NAME` | 否 | `text-embedding-3-small` | Embedding 模型名称 |

### 使用示例

```bash
# OpenAI
export EMBEDDING_API_KEY="sk-..."
export EMBEDDING_BASE_URL="https://api.openai.com/v1"
export EMBEDDING_MODEL_NAME="text-embedding-3-small"

# 阿里云 Qwen Embedding
export EMBEDDING_API_KEY="your-dashscope-key"
export EMBEDDING_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export EMBEDDING_MODEL_NAME="text-embedding-v3"
```

## Reranker 配置（可选）

| 环境变量 | 必需 | 默认值 | 说明 |
|----------|------|--------|------|
| `RERANKER_API_KEY` | 否 | 同 `LLM_API_KEY` | Reranker API 密钥 |
| `RERANKER_BASE_URL` | 否 | 同 `LLM_BASE_URL` | Reranker API 地址 |
| `RERANKER_MODEL_NAME` | 否 | - | Reranker 模型名称 |

## 请求配置

| 环境变量 | 必需 | 默认值 | 说明 |
|----------|------|--------|------|
| `REQUEST_TIMEOUT` | 否 | `60` | HTTP 请求超时（秒） |

## ModelScope 配置

| 环境变量 | 必需 | 默认值 | 说明 |
|----------|------|--------|------|
| `MODELSCOPE_API_TOKEN` | 上传/下载时 | - | ModelScope API Token |
| `MODELSCOPE_TOKEN` | 备选 | - | 同上（备选变量名） |

### 获取 Token

1. 访问 [ModelScope](https://modelscope.cn)
2. 登录后进入个人中心
3. 在 API Token 页面生成 Token

## 配置优先级

环境变量的读取优先级：

```
代码传参 > 环境变量（.env） > config.yaml > 默认值
```

例如：

```python
# 代码传参优先级最高
llm = LLMClient(model_name="custom-model")

# 其次读取环境变量 LLM_MODEL_NAME
# 最后使用默认值 "gpt-4o-mini"
```

## 完整配置示例

```bash
# project/book_search/.env

# === LLM 配置 ===
LLM_API_KEY=sk-your-openai-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o-mini

# === Embedding 配置 ===
EMBEDDING_API_KEY=sk-your-openai-key
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL_NAME=text-embedding-3-small

# === Reranker 配置（可选） ===
RERANKER_API_KEY=sk-your-openai-key
RERANKER_BASE_URL=https://api.openai.com/v1
RERANKER_MODEL_NAME=text-embedding-3-small

# === 请求配置 ===
REQUEST_TIMEOUT=60

# === ModelScope 配置 ===
MODELSCOPE_API_TOKEN=your-modelscope-token
```

## 安全注意事项

!!! warning "安全提醒"

    - 不要将 `.env` 文件提交到 Git 仓库
    - `.gitignore` 中已包含 `.env` 规则
    - 生产环境使用密钥管理服务（如 Azure Key Vault）
    - 定期轮换 API 密钥
