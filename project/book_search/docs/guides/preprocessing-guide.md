# 预处理指南

本文档详细介绍如何使用预处理 Pipeline 处理书籍数据。

## 预处理流程

```mermaid
graph LR
    A[books.db] --> B[文本润色]
    B --> C[向量化]
    C --> D[LLM 打标]
    D --> E[books_tagged.json]
```

## 环境准备

### 必需的环境变量

```bash
# LLM 配置（用于润色和打标）
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL_NAME="gpt-4o-mini"

# Embedding 配置（用于向量化）
export EMBEDDING_API_KEY="your-api-key"
export EMBEDDING_BASE_URL="https://api.openai.com/v1"
export EMBEDDING_MODEL_NAME="text-embedding-3-small"
```

建议将配置写入 `project/book_search/.env` 文件。

### 安装依赖

```bash
pip install faiss-cpu
```

## 完整 Pipeline

### 一键执行

```bash
cd project/book_search

python -c "
from src.process.pipeline import PreprocessPipelineConfig, run_preprocess_pipeline
cfg = PreprocessPipelineConfig(
    input_path='data/books.db',
    output_path='data/books_tagged.json',
    enable_text_polish=True,
    enable_polish_embedding=True,
    enable_llm_tagging=True,
    tagging_mode='flat',
    overwrite=False,
    limit=0,
    sleep_seconds=0.1
)
print(run_preprocess_pipeline(cfg))
"
```

### 参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `input_path` | `data/books.db` | 输入数据库 |
| `output_path` | `data/books_tagged.json` | 标签输出文件 |
| `tagging_mode` | `flat` 或 `cascading` | 标签模式 |
| `overwrite` | `False` | 跳过已处理数据 |
| `limit` | `0` | 处理全部 |
| `sleep_seconds` | `0.1` | API 调用间隔，防限流 |

## 分步执行

### Step 1: 文本润色

```bash
cd project/book_search

# 使用统一 CLI
python -m src.tools.cli polish --db-path data/books.db --limit 100

# 或直接调用 Python
python -c "
from src.process.pipeline import PreprocessPipelineConfig, run_preprocess_pipeline
cfg = PreprocessPipelineConfig(
    input_path='data/books.db',
    output_path='data/books_tagged.json',
    enable_text_polish=True,
    enable_polish_embedding=False,
    enable_llm_tagging=False,
    overwrite=False,
    limit=100,
    sleep_seconds=0.1
)
print(run_preprocess_pipeline(cfg))
"
```

润色逻辑：

- 读取 `books` 表的 title 和 intro
- 读取 `chapters` 表的前 5 章正文
- 调用 LLM 优化书名和简介
- 结果写入 `book_polish` 表

### Step 2: 向量化

```bash
python -c "
from src.process.pipeline import PreprocessPipelineConfig, run_preprocess_pipeline
cfg = PreprocessPipelineConfig(
    input_path='data/books.db',
    output_path='data/books_tagged.json',
    enable_text_polish=False,
    enable_polish_embedding=True,
    enable_llm_tagging=False,
    overwrite=False,
    limit=0,
    sleep_seconds=0.1
)
print(run_preprocess_pipeline(cfg))
"
```

向量化逻辑：

- 从 `book_polish` 表读取润色结果
- 拼接为 "书名：{title}\n简介：{intro}"
- 调用 Embedding API 生成向量
- 写入 Faiss 索引 + `book_polish_embedding` 表

### Step 3: LLM 打标

```bash
python -c "
from src.process.pipeline import PreprocessPipelineConfig, run_preprocess_pipeline
cfg = PreprocessPipelineConfig(
    input_path='data/books.db',
    output_path='data/books_tagged.json',
    enable_text_polish=False,
    enable_polish_embedding=False,
    enable_llm_tagging=True,
    tagging_mode='cascading',
    overwrite=False,
    limit=0,
    sleep_seconds=0.1
)
print(run_preprocess_pipeline(cfg))
"
```

## 标签模式

### Flat（扁平标签）

每本书生成独立的标签列表：

```json
{
  "book_id": 123,
  "title": "斗破苍穹",
  "tags": ["玄幻", "修仙", "废柴逆袭", "热血", "升级流"]
}
```

### Cascading（级联标签）

先分大类再细分小类：

```json
{
  "book_id": 123,
  "title": "斗破苍穹",
  "cascaded_tags": {
    "玄幻": ["东方玄幻", "异世大陆"],
    "情节": ["废柴逆袭", "升级打怪"],
    "风格": ["热血", "爽文"]
  }
}
```

## 增量处理

### 场景：新爬取了一批书籍

```bash
# 1. 爬取新数据
cd project/book_search
python -m src.crawler.engine crawl-books --start 1501 --end 2000 --concurrency 12
python -m src.crawler.engine crawl-content --start 1501 --end 2000 --concurrency 8

# 2. 增量预处理（复用同一个 output_path）
cd ..
python -c "
from src.process.pipeline import PreprocessPipelineConfig, run_preprocess_pipeline
cfg = PreprocessPipelineConfig(
    input_path='data/books.db',
    output_path='data/books_tagged.json',
    incremental_tagging=True,
    overwrite=False
)
print(run_preprocess_pipeline(cfg))
"
```

增量机制：

- 润色：跳过 `book_polish` 中已存在的 book_id
- 向量化：跳过 `book_polish_embedding` + Faiss 中已存在的
- 打标：读取已有 `output.json`，合并历史标签，只处理新数据

## 常见问题

### API 限流

```bash
# 增大 sleep_seconds
sleep_seconds=0.5
```

### 覆盖已有结果

```bash
# 设置 overwrite=True
overwrite=True
```

### 只处理部分数据

```bash
# 设置 limit
limit=100  # 只处理前 100 本
```

### 检查处理结果

```python
import sqlite3
import json

conn = sqlite3.connect("data/books.db")
conn.row_factory = sqlite3.Row

# 检查润色结果
polish_count = conn.execute("SELECT COUNT(*) as c FROM book_polish").fetchone()["c"]
print(f"已润色书籍: {polish_count}")

# 检查向量化结果
embed_count = conn.execute("SELECT COUNT(*) as c FROM book_polish_embedding").fetchone()["c"]
print(f"已向量化书籍: {embed_count}")

# 检查标签结果
with open("data/books_tagged.json", "r", encoding="utf-8") as f:
    tags = json.load(f)
print(f"已打标书籍: {len(tags)}")
```
