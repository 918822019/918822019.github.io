# API 参考

本文档提供系统核心类和方法的完整 API 参考。

## 公共工具 (src.utils)

```python
from src.utils import now_iso, extract_json_block, load_books_from_path, normalize_inline_text
```

| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `now_iso` | - | `str` | 返回当前时间的 ISO 格式字符串 |
| `extract_json_block` | `text: str` | `dict` | 从模型回复中提取 JSON 对象 |
| `load_books_from_path` | `input_path: str \| Path` | `list[dict]` | 从 JSON/NDJSON/SQLite 加载书籍数据 |
| `normalize_inline_text` | `value: Any` | `str` | 将多行文本压缩为单行 |

---

## Agent 智能代理 (src.agent)

### `Agent` 类

```python
from src.agent import Agent
```

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `__init__` | `llm_client?, embedding_client?, reranker_client?` | `None` | 初始化智能代理 |
| `process_query` | `query: str, context?: str` | `str` | 处理查询生成回答 |
| `embed_text` | `text: str` | `List[float]` | 文本向量化 |
| `embed_batch` | `texts: List[str]` | `List[List[float]]` | 批量文本向量化 |
| `rerank_documents` | `query: str, documents: List[str], top_k: int` | `List[Tuple[int, float]]` | 文档重排序 |
| `rewrite_query` | `query: str, mode: str, context?: str` | `str` | 查询重写 |
| `rewrite_parallel` | `query: str, modes?: List[str], context?: str` | `List[str]` | 并行多策略查询重写 |

### `SearchAgent` 类

```python
from src.agent import SearchAgent
```

继承 `Agent` 的全部方法，增加检索问答能力。

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `search_and_answer` | `query: str, candidate_texts: List[str], top_k: int, ...` | `str` | 完整搜索问答流程 |
| `_normalize_strategy` | `strategy: str` | `str` | 规范化检索策略名 |
| `_reciprocal_rank_fusion` | `ranked_lists: list, top_k?: int, k: int` | `list` | RRF 多路排序融合 |

---

## LLMClient

```python
from src.llm.client import LLMClient
```

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `__init__` | `model_name?: str, api_key?: str, base_url?: str` | `None` | 初始化 LLM 客户端 |
| `generate` | `prompt: str, system_prompt?: str, **kwargs` | `str` | 文本生成 |
| `chat` | `messages: List[Dict], **kwargs` | `str` | 对话式调用 |
| `generate_with_context` | `query: str, context: str, **kwargs` | `str` | 基于上下文生成 |

---

## EmbeddingClient

```python
from src.llm.client import EmbeddingClient
```

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `__init__` | `model_name?: str, api_key?: str, base_url?: str` | `None` | 初始化 Embedding 客户端 |
| `embed` | `text: str` | `List[float]` | 单文本向量化 |
| `embed_batch` | `texts: List[str]` | `List[List[float]]` | 批量向量化 |
| `cosine_similarity` | `vec1: List[float], vec2: List[float]` | `float` | 余弦相似度计算 |
| `search_similar` | `query_embedding: List[float], candidates: List[List[float]], top_k: int` | `List[Tuple[int, float]]` | 相似向量搜索 |

---

## RerankerClient

```python
from src.llm.client import RerankerClient
```

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `__init__` | `model_name?: str, api_key?: str, base_url?: str` | `None` | 初始化 Reranker 客户端 |
| `rerank` | `query: str, documents: List[str], top_k: int` | `List[Tuple[int, float]]` | 文档重排序 |

---

## PreprocessPipelineConfig

```python
from src.process.pipeline import PreprocessPipelineConfig
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_path` | `Union[str, Path]` | - | 输入数据库路径 |
| `output_path` | `Union[str, Path]` | - | 标签输出路径 |
| `enable_text_polish` | `bool` | `True` | 启用文本润色 |
| `enable_polish_embedding` | `bool` | `True` | 启用向量化 |
| `enable_llm_tagging` | `bool` | `True` | 启用 LLM 打标 |
| `polish_model_name` | `Optional[str]` | `None` | 润色模型 |
| `embedding_model_name` | `Optional[str]` | `None` | Embedding 模型 |
| `tagging_mode` | `Literal["flat", "cascading"]` | `"flat"` | 标签模式 |
| `model_name` | `Optional[str]` | `None` | 打标模型 |
| `max_tags` | `int` | `8` | 最大标签数 |
| `sleep_seconds` | `float` | `0.0` | API 间隔 |
| `overwrite` | `bool` | `False` | 覆盖模式 |
| `limit` | `int` | `0` | 数量限制 |
| `incremental_tagging` | `bool` | `True` | 增量打标 |

### `run_preprocess_pipeline(config)`

```python
from src.process.pipeline import run_preprocess_pipeline

result: dict[str, Any] = run_preprocess_pipeline(config)
```

---

## BookSearchEngine

```python
from src.main import BookSearchEngine
```

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `__init__` | `db_path: str` | `None` | 初始化搜索引擎 |
| `connect` | - | `None` | 连接数据库 |
| `close` | - | `None` | 关闭连接 |
| `check_vectorization_status` | - | `dict` | 检查向量化状态 |
| `vectorize_books` | `limit: int, overwrite: bool` | `dict` | 执行向量化 |
| `load_faiss_index` | - | `None` | 加载 Faiss 索引 |
| `search_books_by_query` | `query: str, top_k: int` | `List[dict]` | 搜索书籍 |
| `answer_with_context` | `query: str, books: List[dict]` | `str` | 生成回答 |

---

## 预处理函数

### 润色与向量化 (src.process.polish)

```python
from src.process.polish import (
    run_polish, run_polish_embedding,
    ensure_polish_table, ensure_embedding_table,
    load_or_create_faiss_index, get_faiss_index_path,
    resolve_existing_db_path,
)
```

| 函数 | 说明 |
|------|------|
| `run_polish` | 执行批量润色 |
| `run_polish_embedding` | 执行向量化入库 |
| `ensure_polish_table` | 确保润色表存在 |
| `ensure_embedding_table` | 确保 embedding 表存在 |
| `load_or_create_faiss_index` | 加载或创建 Faiss 索引 |
| `get_faiss_index_path` | 推导索引文件路径 |
| `resolve_existing_db_path` | 解析数据库路径 |

### 打标签 (src.process.taggers)

```python
from src.process.taggers import LLMTagger, CascadingTagger
```

| 类/方法 | 说明 |
|---------|------|
| `LLMTagger` | 扁平标签生成器 |
| `CascadingTagger` | 级联标签生成器 |

---

## 数据爬取

```bash
python -m src.crawler.engine <command> [options]
```

| 命令 | 关键参数 | 说明 |
|------|----------|------|
| `crawl-books` | `--start, --end, --concurrency` | 抓取书籍目录 |
| `crawl-content` | `--start, --end, --concurrency, --batch-size` | 抓取章节正文 |
| `sync-all` | `--start, --end, --concurrency, --batch-size` | 完整同步 |
| `stats` | - | 查看统计 |
| `export-shards` | `--start, --end, --shard-size, --output-dir` | 导出分片 |

---

## 工具脚本

### upload_modelscope_dataset.py

```bash
python -m src.tools.upload_modelscope_dataset [options]
```

| 参数 | 说明 |
|------|------|
| `--repo-id` | ModelScope 仓库 ID |
| `--folder-path` | 上传目录 |
| `--incremental` | 增量上传模式 |
| `--dry-run` | 仅预览不上传 |
| `--run` | 执行上传 |
| `--sqlite-snapshot` | SQLite 快照策略 (auto/always/never) |
| `--include-hidden` | 包含隐藏文件 |
| `--commit-message` | 提交信息 |

### download_modelscope_dataset.py

```bash
python -m src.tools.download_modelscope_dataset [options]
```

| 参数 | 说明 |
|------|------|
| `--repo-id` | ModelScope 仓库 ID |
| `--repo-type` | 仓库类型 (dataset/model) |
| `--revision` | 分支/版本 |
| `--output-dir` | 下载目录 |
| `--allow-pattern` | 文件匹配模式 |
| `--clean-output` | 清空目标目录 |

### stats.py

```bash
python -m src.tools.stats [options]
```

输出数据库/润色/向量化/打标/分片/文件系统全线状态。

| 参数 | 说明 |
|------|------|
| `--db-path` | 数据库路径（默认 data/books.db） |
| `--tagged-json` | 打标 JSON 路径（默认 data/books_tagged.json） |


