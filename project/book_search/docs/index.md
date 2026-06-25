# 起点搜书 - 小说智能搜索系统

<div class="grid cards" markdown>

-   :material-rocket-launch-outline:{ .lg .middle } **快速上手**

    ---

    从安装到运行，五分钟启动你的小说搜索系统

    [:octicons-arrow-right-24: 安装指南](getting-started/installation.md)

-   :material-book-open-variant:{ .lg .middle } **系统架构**

    ---

    了解向量检索 + LLM 的 RAG 全链路设计

    [:octicons-arrow-right-24: 架构概览](architecture/overview.md)

-   :material-cog:{ .lg .middle } **核心模块**

    ---

    Agent、LLM、Embedding、Reranker 模块详解

    [:octicons-arrow-right-24: 模块文档](modules/agent.md)

-   :material-api:{ .lg .middle } **API 参考**

    ---

    完整的类与方法 API 文档

    [:octicons-arrow-right-24: API 文档](api/reference.md)

</div>

## 项目简介

起点搜书是一个**基于向量检索和大语言模型的小说推荐与搜索系统**。系统实现了从数据爬取、文本润色、向量化、智能检索到 AI 回答的完整 RAG（Retrieval-Augmented Generation）流程。

### 核心能力

| 能力 | 说明 |
|------|------|
| **数据爬取** | 支持从起点中文网并发抓取书籍元信息、章节目录和正文内容 |
| **文本润色** | 利用 LLM 基于前五章正文自动优化书名和简介 |
| **向量检索** | 生成文本 Embedding 并使用 Faiss 构建高性能向量索引 |
| **智能问答** | 支持多种检索策略（单路/早融合/晚融合）+ Reranker 精排 |
| **LLM 打标** | 自动为书籍生成扁平标签或级联标签 |
| **数据管理** | 支持 ModelScope 上传与增量同步 |

### 技术栈

- **语言**: Python 3.11+
- **向量库**: Faiss (IndexIDMap2 + IndexFlatIP)
- **LLM**: OpenAI 兼容接口 (支持 Qwen、GPT 等)
- **Embedding**: text-embedding-3-small / 自定义模型
- **Reranker**: OpenAI 兼容 Rerank 接口
- **数据库**: SQLite (WAL 模式)

### 项目结构

```
book_search/
├── main.py                     # 兼容入口（委托给 src.main）
├── config.yaml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── main.py                 # 搜索引擎入口
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── search_agent.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py
│   ├── process/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── polish/
│   │   │   ├── __init__.py
│   │   │   ├── core.py
│   │   │   ├── embedding.py
│   │   │   ├── search.py
│   │   │   ├── _db.py
│   │   │   └── _faiss.py
│   │   └── taggers/
│   │       ├── __init__.py
│   │       ├── flat.py
│   │       ├── cascading.py
│   │       └── _utils.py
│   ├── crawler/
│   │   ├── __init__.py
│   │   └── engine.py
│   └── tools/
│       ├── __init__.py
│       ├── cli.py
│       ├── look_db.py
│       ├── modelscope_utils.py
│       ├── upload_modelscope_dataset.py
│       └── download_modelscope_dataset.py
├── scripts/
├── data/
│   ├── books.db
│   └── shards/
├── tests/
└── docs/
```

## 快速体验

```bash
# 1. 进入项目目录
cd project/book_search

# 2. 安装依赖
pip install -r requirements-dev.txt

# 3. 配置环境变量
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.openai.com/v1"
export EMBEDDING_API_KEY="your-api-key"
export EMBEDDING_BASE_URL="https://api.openai.com/v1"

# 4. 启动搜索系统
python main.py
```

## 文档导航

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle } **教程**

    ---

    - [安装指南](getting-started/installation.md)
    - [快速上手](getting-started/quickstart.md)
    - [预处理指南](guides/preprocessing-guide.md)
    - [搜索问答指南](guides/search-guide.md)

-   :material-book:{ .lg .middle } **参考**

    ---

    - [架构概览](architecture/overview.md)
    - [数据流向](architecture/data-flow.md)
    - [环境变量配置](configuration/environment.md)
    - [模型配置](configuration/models.md)

-   :material-tools:{ .lg .middle } **进阶**

    ---

    - [数据爬取模块](modules/data-crawling.md)
    - [ModelScope 上传](guides/modelscope-upload.md)
    - [生产部署](deployment/production.md)

</div>
