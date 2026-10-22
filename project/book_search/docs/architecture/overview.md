# 架构概览

## 系统定位

起点搜书是一个**端到端的小说智能推荐系统**，采用 RAG（Retrieval-Augmented Generation）架构，将向量检索与大语言模型结合，实现精准的书籍搜索和智能问答。

## 整体架构

```mermaid
graph TB
    subgraph "数据层"
        A[起点中文网] -->|爬取| B[(SQLite<br/>books.db)]
        B -->|润色| C[book_polish 表]
        C -->|向量化| D[book_polish_embedding 表]
        D -->|索引| E[Faiss 索引文件]
    end

    subgraph "处理层"
        F[LLM 客户端] -->|文本润色| C
        F -->|打标签| G[标签数据]
        H[Embedding 客户端] -->|向量化| D
        I[Reranker 客户端] -->|精排| J[排序结果]
    end

    subgraph "服务层"
        K[Agent 智能代理]
        K -->|查询重写| F
        K -->|向量检索| E
        K -->|重排序| I
        K -->|生成回答| F
    end

    subgraph "入口层"
        L[main.py<br/>搜索引擎] --> K
    end
```

## 核心设计思想

### 1. 分层解耦

系统采用四层架构，各层职责清晰：

| 层级 | 职责 | 核心模块 |
|------|------|----------|
| **数据层** | 数据存储与索引 | SQLite + Faiss |
| **处理层** | 文本处理与向量化 | LLM/Embedding/Reranker 客户端 |
| **服务层** | 智能代理与检索编排 | agent.agent.Agent + agent.search_agent.SearchAgent |
| **入口层** | 用户交互 | main.py CLI |

### 2. 增量处理

所有处理步骤都支持增量模式：

- **文本润色**: 默认跳过已润色的 book_id
- **向量化**: 默认跳过已有 embedding 的记录
- **LLM 打标**: 自动合并已有标签，优先处理新数据
- **数据爬取**: `crawl-content` 只抓未完成章节

### 3. 可插拔模型

所有模型调用都通过环境变量配置，支持随时切换：

```bash
# 切换 LLM 模型
export LLM_MODEL_NAME="qwen-plus"

# 切换 Embedding 模型
export EMBEDDING_MODEL_NAME="text-embedding-3-small"
```

## 检索策略

系统支持三种检索策略，适用于不同场景：

```mermaid
graph LR
    A[用户查询] --> B{策略选择}

    B -->|single| C[单路检索]
    B -->|early_fusion| D[早期融合]
    B -->|late_fusion| E[晚期融合]

    C --> F[查询重写]
    F --> G[向量检索]
    G --> H[Reranker 精排]

    D --> I[多策略改写]
    I --> J[聚合查询]
    J --> K[单次向量检索]
    K --> H

    E --> L[多策略改写]
    L --> M[多路向量检索]
    M --> N[RRF 融合]
    N --> H

    H --> O[LLM 生成回答]
```

| 策略 | 原理 | 适用场景 |
|------|------|----------|
| **single** | 单个查询（可选重写）直接检索 | 简单明确的查询 |
| **early_fusion** | 将多个改写查询合并为一个综合查询 | 需要语义扩展的查询 |
| **late_fusion** | 多个查询分别检索，用 RRF 融合结果 | 复杂模糊的查询 |

## 数据库设计

### 核心表结构

```mermaid
erDiagram
    books ||--o| book_polish : "book_id"
    books ||--o{ chapters : "book_id"
    book_polish ||--o| book_polish_embedding : "book_id"

    books {
        INTEGER book_id PK
        TEXT title
        TEXT intro
        TEXT author
        TEXT category
    }

    chapters {
        INTEGER chapter_id PK
        INTEGER book_id FK
        TEXT chapter_name
        TEXT content
        BOOLEAN is_content_fetched
    }

    book_polish {
        INTEGER book_id PK
        TEXT polished_title
        TEXT polished_intro
        TEXT model_name
        TEXT updated_at
    }

    book_polish_embedding {
        INTEGER book_id PK
        TEXT text_content
        INTEGER embedding_dim
        TEXT model_name
        TEXT updated_at
    }
```

### Faiss 索引

- **类型**: `IndexIDMap2(IndexFlatIP)` — 支持自定义 ID 的内积索引
- **存储**: 与数据库同目录，后缀 `.polish_embedding.faiss`
- **向量归一化**: 所有向量在写入前做 L2 归一化，使内积等价于余弦相似度

## 模块依赖关系

```mermaid
graph TD
    A[agent.agent.Agent] --> B[llm.client.LLMClient]
    A --> C[llm.client.EmbeddingClient]
    A --> D[llm.client.RerankerClient]

    I[agent.search_agent.SearchAgent] -- "继承" --> A

    E[process.pipeline] --> F[process.polish.core]
    E --> G[process.polish.embedding]
    E --> H[process.taggers]

    F --> B
    G --> C

    I[main.BookSearchEngine] --> A
    I --> G
    I --> J[SQLite]
    I --> K[Faiss]

    L[crawler.engine] --> J
    M[tools.cli] --> F
    M --> G
    M --> H2[process.taggers]

    N[utils] -.-> F
    N -.-> G
    N -.-> H2
    N -.-> L
```

## 扩展点

系统预留了多个扩展点：

1. **模型替换**: 通过环境变量切换任意 OpenAI 兼容模型
2. **检索策略**: 在 `Agent.RETRIEVAL_STRATEGY_ALIASES` 中注册新策略
3. **查询重写**: 在 `Agent` 中添加新的重写模式和 prompt
4. **数据源**: 在 `crawler/` 中添加新的爬取适配器
5. **输出格式**: 修改 `answer_with_context` 的 prompt 模板
