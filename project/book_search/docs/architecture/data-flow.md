# 数据流向

本文档详细描述数据从爬取到最终回答的完整流转过程。

## 端到端数据流

```mermaid
sequenceDiagram
    participant User as 用户
    participant Crawler as 爬虫模块
    participant DB as SQLite
    participant Polish as 润色模块
    participant Embed as 向量化模块
    participant Faiss as Faiss索引
    participant Agent as Agent
    participant LLM as LLM

    rect rgb(230, 245, 255)
    Note over Crawler,DB: 阶段一：数据采集
    Crawler->>DB: 写入 books 表（元信息）
    Crawler->>DB: 写入 chapters 表（目录+正文）
    end

    rect rgb(255, 245, 230)
    Note over Polish,Embed: 阶段二：数据预处理
    Polish->>DB: 读取 books + chapters 前5章
    Polish->>LLM: 调用润色 prompt
    LLM-->>Polish: 返回 polished_title/intro
    Polish->>DB: 写入 book_polish 表

    Embed->>DB: 读取 book_polish
    Embed->>Embed: 构建 embedding 文本
    Embed-->>Embed: 调用 Embedding API
    Embed->>DB: 写入 book_polish_embedding 表
    Embed->>Faiss: 写入向量 + book_id
    end

    rect rgb(240, 255, 240)
    Note over User,LLM: 阶段三：搜索问答
    User->>Agent: 输入查询
    Agent->>Agent: 查询重写（可选）
    Agent->>Faiss: 向量检索 top_k×2
    Faiss-->>Agent: 返回候选 book_id
    Agent->>DB: 查询书籍元数据
    Agent->>Agent: Reranker 精排
    Agent->>LLM: 构建上下文 + 生成回答
    LLM-->>Agent: 返回回答
    Agent-->>User: 展示结果
    end
```

## 阶段一：数据采集

### 数据来源

起点中文网 API，爬取两个维度的数据：

| 数据类型 | 写入表 | 字段 |
|----------|--------|------|
| 书籍元信息 | `books` | book_id, title, intro, author, category |
| 章节目录 | `chapters` | chapter_id, book_id, chapter_name, is_content_fetched |
| 章节正文 | `chapters` | content (UPDATE) |

### 爬取流程

```mermaid
graph LR
    A[crawl-books] --> B[抓取书籍首页]
    B --> C[解析章节目录]
    C --> D[写入 books + chapters]
    D --> E[crawl-content]
    E --> F[逐章抓取正文]
    F --> G[更新 chapters.content]
```

**关键特性**：

- 断点续抓：`is_content_fetched` 标记已抓取章节
- 并发控制：`--concurrency` 参数控制并发数
- 请求限速：`--min-request-interval` + `--request-jitter` 打散请求

## 阶段二：数据预处理

### 2.1 文本润色

```
输入: books.title + books.intro + chapters.content (前5章)
  ↓
LLM 调用 (system: 中文小说文案编辑)
  ↓
输出: book_polish.polished_title + book_polish.polished_intro
```

**润色规则**：

- 基于前五章正文反向优化简介
- 保持叙事人称和文字气质一致
- 简介字数控制在 60-180 字
- 严禁修改世界观、角色姓名等事实

### 2.2 向量化

```
输入: book_polish.polished_title + book_polish.polished_intro
  ↓
拼接为: "书名：{title}\n简介：{intro}"
  ↓
Embedding API 调用
  ↓
输出: book_polish_embedding 表 + Faiss 索引
```

**向量处理**：

- L2 归一化后写入 Faiss
- 使用 `IndexIDMap2` 支持按 book_id 精确更新
- 索引文件与数据库同目录存储

### 2.3 LLM 打标

```
输入: books.title + books.intro
  ↓
LLM 调用 (扁平标签 / 级联标签)
  ↓
输出: JSON 文件 (tags / cascaded_tags)
```

**两种模式**：

- **flat**: 每本书生成独立的标签列表
- **cascading**: 先分大类再细分小类，形成层级标签

## 阶段三：搜索问答

### 查询处理流程

```mermaid
graph TD
    A[用户查询] --> B{启用查询重写?}
    B -->|是| C{重写策略}
    B -->|否| D[原始查询]

    C -->|expansion| E[查询扩展]
    C -->|clarification| F[查询澄清]
    C -->|decomposition| G[查询分解]
    C -->|hyde| H[生成假设文档]

    E --> I[检索查询]
    F --> I
    G --> I
    H --> I
    D --> I

    I --> J[Embedding 向量化]
    J --> K[Faiss 检索]
    K --> L[获取候选书籍元数据]
    L --> M[Reranker 精排]
    M --> N[取 top_k 结果]
    N --> O[构建上下文 prompt]
    O --> P[LLM 生成回答]
```

### 检索策略详解

#### Single（单路检索）

```
query → [rewrite] → embed → faiss.search → rerank → answer
```

最简单的策略，适合查询意图明确的场景。

#### Early Fusion（早期融合）

```
query → [rewrite: expansion, clarification, hyde]
     ↓
合并为: "query\nrewritten1\nrewritten2\nrewritten3"
     ↓
embed → faiss.search → rerank → answer
```

将多个改写版本合并为一个综合查询，利用所有语义信息进行单次检索。

#### Late Fusion（晚期融合）

```
query → [rewrite: expansion, clarification, hyde]
     ↓
对每个查询分别: embed → faiss.search
     ↓
RRF 融合: score = Σ(1/(k+rank))
     ↓
rerank → answer
```

多路独立检索后用 RRF（Reciprocal Rank Fusion）算法融合排序结果。

## 分片与同步流程

```mermaid
graph LR
    A[books.db<br/>主库] -->|export-shards| B[shards/<br/>分片目录]
    B -->|upload| C[ModelScope<br/>数据集]
    C -->|download| D[新服务器<br/>data/]
    D --> E[继续爬取/处理]
```

**增量同步机制**：

1. `export-shards --only-changed`：基于 `source_fingerprint` 判断分片是否变化
2. `upload --incremental`：基于本地 manifest 判断文件是否变化
3. `crawl-content`：基于 `is_content_fetched` 判断章节是否已抓取

## 状态管理

系统通过多个状态字段实现断点续跑：

| 状态 | 位置 | 说明 |
|------|------|------|
| `is_content_fetched` | chapters 表 | 章节正文是否已抓取 |
| `book_polish` 表存在 | book_polish 表 | 书籍是否已润色 |
| `book_polish_embedding` 表存在 | book_polish_embedding 表 | 书籍是否已向量化 |
| Faiss 索引中的 book_id | `.faiss` 文件 | 向量是否已写入索引 |
| `shards/index.json` | shards 目录 | 分片导出状态 |
| `.shards.modelscope-upload-manifest.json` | data 目录 | 上传同步状态 |
