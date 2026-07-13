# 起点搜书

本模块实现基于用户查询的书籍推荐系统，包含查询处理、数据召回、推荐理由生成和结果展示等功能。

当前抓取脚本已经改为直接请求站点真实接口，并将结果写入 SQLite，支持先抓目录、再抓正文、断点续抓。

现在也支持把主库按 book_id 范围导出成多个 shard SQLite 文件，配合上传脚本做增量上传。

## 目录结构

- data/books.db SQLite 抓取数据库
- data/books_tagged.json LLM 打标结果
- src/crawler/engine.py 抓取脚本 (python -m src.crawler.engine)
- src/agent/ 智能代理（Agent + SearchAgent + __init__.py）
- src/llm/client.py LLM/Embedding/Reranker 客户端（含指数退避重试）
- src/process/pipeline.py 预处理编排入口
- src/process/polish/ 文本润色与向量化（含 Faiss 检索 + ID 缓存）
- src/process/taggers/ LLM 批量打标签
- src/tools/cli.py 统一 CLI（polish/embed/search/stats）
- src/tools/ 其他实用工具（上传/下载/数据查看）
- scripts/ 辅助脚本
- tests/ 单元测试

## 核心优化

### 指数退避重试
所有 LLM/Embedding/Reranker API 调用自带指数退避重试（1s → 2s → 4s），对 5xx/429 错误自动恢复，无需手动处理。

### 并行查询改写
`Agent.rewrite_parallel` 使用 `ThreadPoolExecutor` 并发调用多种改写策略（expansion/clarification/hyde），显著降低多策略场景延迟。

### 真实 Reranker 支持
`RerankerClient` 从桩实现升级为真实 API 调用（标准 `/rerank` 端点），未配置 API 时自动降级为 fallback 排序。

### Faiss ID 缓存
`_get_index_ids` 增加模块级缓存，避免每次 O(n) 扫描整个 Faiss id_map，批量写入时性能提升显著。

### 批量元数据查询
`BookSearchEngine.search_books_by_query` 使用 `WHERE book_id IN (...)` 一次批量查询代替逐条查库。

## 快速开始

1. 运行 `python -m src.crawler.engine crawl-books --start 1 --end 100` 抓取书籍
2. 运行 `python main.py` 启动交互式搜索引擎
3. 输入查询即可搜索相关书籍

## LLM 批量打标签

可以使用 `src/process/taggers/` 对每一本小说调用 LLM 自动生成标签。

先配置环境变量（OpenAI 兼容接口）：

```bash
export LLM_API_KEY="你的密钥"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL_NAME="gpt-4o-mini"
```

执行打标签：

```python
from src.process.taggers import LLMTagger

tagger = LLMTagger(model_name="qwen-plus", sleep_seconds=0.2)
tagger.load_books("data/books.json")
stats = tagger.run(output_path="data/books_tagged.json", limit=100)
```

## 数据预处理 Pipeline 运行方式

`src/process/pipeline.py` 提供了统一编排入口，会按顺序执行：

1. 文本润色（基于书名 + 原简介 + 前五章正文润色简介，结果写入 `book_polish`）
2. 生成 embedding（基于 `book_polish` 写入 `book_polish_embedding`）
3. LLM 打标签（输出到 JSON）

### 1) 准备环境变量

至少需要以下配置（可写在 `project/book_search/.env`）：

非敏感参数（路径、模型名、超时等）也可写在 `project/book_search/config.yaml` 中，环境变量会覆盖 config.yaml 中的同名配置。

```bash
export LLM_API_KEY="你的密钥"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL_NAME="gpt-4o-mini"

export EMBEDDING_API_KEY="你的密钥"
export EMBEDDING_BASE_URL="https://api.openai.com/v1"
export EMBEDDING_MODEL_NAME="text-embedding-3-small"
```

### 2) 一次性运行完整 Pipeline

在 `project/book_search` 目录执行：

```bash
python -c "from src.process.pipeline import PreprocessPipelineConfig, run_preprocess_pipeline; cfg=PreprocessPipelineConfig(input_path='data/books.db', output_path='data/books_tagged.json', enable_text_polish=True, enable_polish_embedding=True, enable_llm_tagging=True, tagging_mode='flat', overwrite=False, limit=0, sleep_seconds=0.0); print(run_preprocess_pipeline(cfg))"
```

### 3) 只跑某些步骤（常见场景）

只做“简介润色 + embedding”，暂不打标签：

```bash
python -c "from src.process.pipeline import PreprocessPipelineConfig, run_preprocess_pipeline; cfg=PreprocessPipelineConfig(input_path='data/books.db', output_path='data/books_tagged.json', enable_text_polish=True, enable_polish_embedding=True, enable_llm_tagging=False, overwrite=False, limit=200, sleep_seconds=0.1); print(run_preprocess_pipeline(cfg))"
```

只做 LLM 打标签（复用已有润色和 embedding 结果）：

```bash
python -c "from src.process.pipeline import PreprocessPipelineConfig, run_preprocess_pipeline; cfg=PreprocessPipelineConfig(input_path='data/books.db', output_path='data/books_tagged.json', enable_text_polish=False, enable_polish_embedding=False, enable_llm_tagging=True, tagging_mode='cascading', overwrite=False, limit=0); print(run_preprocess_pipeline(cfg))"
```

### 常用参数说明

- `input_path`：SQLite 数据库路径（通常为 `data/books.db`）
- `output_path`：标签结果输出 JSON 路径
- `tagging_mode`：`flat`（扁平标签）或 `cascading`（级联标签）
- `limit`：本次最多处理书籍数，`0` 表示全部
- `overwrite`：是否覆盖已有结果
- `sleep_seconds`：每次模型调用后的等待秒数，用于降低限流风险
- `incremental_tagging`：是否启用标签增量模式（默认 `True`）

### 增量预处理（新数据续跑）

当 `books.db` 里新增了一批书时，直接复用同一个 `output_path` 再跑一次 pipeline 即可。

机制说明：

- 文本润色和 embedding 本身就是增量写入（默认 `overwrite=False` 时会跳过已处理 book_id）
- LLM 打标会自动读取已有 `output_path`，按 `book_id` 回填历史 `tags/cascaded_tags` 后再执行
- 因此第二次运行会优先处理新进来的书，避免对旧书重复调用模型

示例（推荐保留同一个输出文件）：

```bash
python -c "from src.process.pipeline import PreprocessPipelineConfig, run_preprocess_pipeline; cfg=PreprocessPipelineConfig(input_path='data/books.db', output_path='data/books_tagged.json', incremental_tagging=True, overwrite=False); print(run_preprocess_pipeline(cfg))"
```

提示：如果你偏好脚本方式，也可以继续使用单步骤脚本：

- `python -c "from src.process.taggers import LLMTagger; t=LLMTagger(); t.load_books('data/books.json'); print(t.run(output_path='data/books_tagged.json'))"`

## 抓取命令

在 project/book_search 目录执行：

```bash
python -m src.crawler.engine crawl-books --start 1 --end 10000 --concurrency 12
```

先抓 1 到 10000 的书籍首页和章节目录，写入 data/books.db。

```bash
python -m src.crawler.engine crawl-content --start 1 --end 10000 --concurrency 12 --batch-size 120
```

再基于数据库里的章节目录补全正文内容。

更稳一些的正文抓取建议这样跑：

```bash
python -m src.crawler.engine crawl-content --start 1 --end 10000 --concurrency 8 --batch-size 40 --chapter-progress-every 500 --sqlite-synchronous FULL --min-request-interval 0.03 --request-jitter 0.05 --retry-backoff-base 1.5 --retry-backoff-max 12
```

这组参数更适合长时间后台运行：

- `sqlite-synchronous FULL` 提高断电场景下的落盘可靠性。
- `min-request-interval` 和 `request-jitter` 会把请求节奏打散，减少固定频率特征。
- 请求头会自动轮换 User-Agent，并按书籍页/章节页生成 Referer。
- `crawl-content` 只会抓未完成章节，断掉后再次执行会继续补齐。
- `chapter-progress-every` 会按章节输出进度，长任务更容易观察。

```bash
python -m src.crawler.engine sync-all --start 1 --end 10000 --concurrency 12 --batch-size 120
```

按顺序执行目录抓取和正文抓取。

```bash
python -m src.crawler.engine stats
```

查看当前数据库里的书籍数、章节数和正文完成进度。

```bash
python -m src.crawler.engine crawl-content --start 1 --end 1 --max-pending-per-book 5 --batch-size 5
```

按每本书限制本次处理的未完成章节数，适合分批跑或小样本验证。

```bash
python -m src.crawler.engine export-shards --start 1 --end 10000 --shard-size 200 --output-dir ../data/shards
```

把主库按 book_id 范围导出为 shard SQLite 文件。默认每 200 本书一个分片，并在输出目录下生成一个 index.json。

```bash
python -m src.crawler.engine export-shards --start 1 --end 10000 --shard-size 200 --output-dir ../data/shards --only-changed
```

只重导出发生变化的 shard。依据是输出目录里上一版 `index.json` 记录的 source_fingerprint。

## 说明

- 默认数据库路径是 data/books.db。
- crawl-books 会把书籍元信息写入 books 表，把章节目录写入 chapters 表。
- crawl-content 只会抓还没完成的章节，可以重复执行。
- 当前 SQLite 使用 WAL，配合 `--sqlite-synchronous FULL` 时更适合怕断电的长任务。
- export-shards 不会改动主库，而是从主库读取并导出静态 shard 文件，适合上传到数据集仓库。
- `src.tools.upload_modelscope_dataset` 和 `src.tools.download_modelscope_dataset` 用于与 ModelScope 数据集同步。

## 上传到 ModelScope

以下示例默认是 Ubuntu 环境（bash）。

先安装依赖：

```bash
pip install -r requirements.txt
```

如果你希望命令更短，也可以使用封装好的脚本：

```bash
chmod +x scripts/upload_modelscope.sh
```

默认是 dry-run（只检查不上传）：

```bash
./scripts/upload_modelscope.sh \
	--repo-id wzywuan/Novel-Collection \
	--folder-path data
```

确认后正式上传（加 `--run`）：

```bash
./scripts/upload_modelscope.sh \
	--repo-id wzywuan/Novel-Collection \
	--folder-path data \
	--run
```

对 shards 增量上传：

```bash
./scripts/upload_modelscope.sh \
	--repo-id wzywuan/Novel-Collection \
	--folder-path data/shards \
	--incremental \
	--run
```

设置 token：

```bash
export MODELSCOPE_API_TOKEN="你的 token"
```

先 dry-run 检查将上传哪些文件（不实际上传）：

```bash
python -m src.tools.upload_modelscope_dataset \
	--repo-id wzywuan/Novel-Collection \
	--folder-path data \
	--dry-run
```

确认后执行正式上传：

```bash
python -m src.tools.upload_modelscope_dataset \
	--repo-id wzywuan/Novel-Collection \
	--folder-path data \
	--commit-message "upload dataset folder to repo"
```

如果只想上传默认的 data 目录，`--folder-path` 可以省略。

脚本的 SQLite 优化策略（默认开启）：

- 自动跳过 SQLite 运行时文件：`*.db-wal`、`*.db-shm`、`*.db-journal`
- 当检测到 `.db` 旁边有 WAL/SHM 时，会先生成一致性快照再上传
- 增量模式比较 `.db` 变化时，会把 `-wal/-shm` 的变化也纳入判断，避免漏传

可以通过 `--sqlite-snapshot` 控制行为：

- `auto`（默认）：检测到 WAL/SHM 时才快照
- `always`：所有 `.db` 一律先快照
- `never`：直接上传原始 `.db`

例如：强制所有 `.db` 都走快照上传

```bash
python -m src.tools.upload_modelscope_dataset \
	--repo-id wzywuan/Novel-Collection \
	--folder-path data \
	--sqlite-snapshot always
```

脚本位置：

- `src.tools.upload_modelscope_dataset` 负责登录 ModelScope 并执行 `upload_folder`
- `src.tools.download_modelscope_dataset` 负责从 ModelScope 下载并同步到本地目录
- `requirements.txt` 包含 ModelScope SDK 等最小依赖

## 增量上传建议流程

对于 `data/books.db` 这种大单文件，不适合直接做真正意义上的增量上传。更合适的做法是：

1. 继续抓取到主库 `data/books.db`
2. 定期导出 shards 到 `data/shards`
3. 对 `data/shards` 目录执行增量上传

先导出分片：

```bash
cd project/book_search
python -m src.crawler.engine export-shards --start 1 --end 10000 --shard-size 200 --output-dir ../data/shards
```

后续增量更新建议使用：

```bash
cd project/book_search
python -m src.crawler.engine export-shards --start 1 --end 10000 --shard-size 200 --output-dir ../data/shards --only-changed
```

先看看这次会上传哪些文件：

```bash
cd project/book_search
python -m src.tools.upload_modelscope_dataset \
	--repo-id wzywuan/Novel-Collection \
	--folder-path data/shards \
	--incremental \
	--dry-run
```

确认后执行真正上传：

```bash
cd project/book_search
python -m src.tools.upload_modelscope_dataset \
	--repo-id wzywuan/Novel-Collection \
	--folder-path data/shards \
	--incremental \
	--commit-message "incremental shard upload"
```

说明：

- 增量模式会在待上传目录旁边生成一个本地 manifest，用于判断哪些文件发生了变化。
- 当前增量模式只处理新增和更新文件，不自动删除远端已存在但本地已删除的文件。
- 如果 shard 文件没变，就不会重复上传它。

## 跨服务器断点续跑（建议流程）

如果你希望在新服务器上继续 `crawl-content`、`export-shards --auto-continue`、
以及 `src.tools.upload_modelscope_dataset --incremental` 的断点状态，建议把 `data` 目录全量上传。

先在旧服务器执行 dry-run，确认上传列表里包含隐藏状态文件：

```bash
cd project/book_search
python -m src.tools.upload_modelscope_dataset \
	--repo-id wzywuan/Novel-Collection \
	--folder-path data \
	--include-hidden \
	--dry-run
```

确认后执行全量上传：

```bash
cd project/book_search
python -m src.tools.upload_modelscope_dataset \
	--repo-id wzywuan/Novel-Collection \
	--folder-path data \
	--include-hidden \
	--commit-message "full data upload for resume"
```

然后在新服务器下载到同样路径：

```bash
cd project/book_search
python -m src.tools.download_modelscope_dataset \
	--repo-id wzywuan/Novel-Collection \
	--repo-type dataset \
	--revision master \
	--output-dir data
```

说明：

-- `crawl-content` 的断点在 `data/books.db` 里的 `is_content_fetched` 状态。
-- `export-shards --auto-continue` 依赖 `data/shards/index.json`。
-- `src.tools.upload_modelscope_dataset --incremental` 依赖本地 manifest，
例如 `data/.shards.modelscope-upload-manifest.json`。

- `--include-hidden` 用于确保上述隐藏状态文件也被上传。

可直接复制的“注释版”恢复与续跑命令：

```bash
# 1) 拉取完整数据到 data 目录（保持与旧服务器相同相对路径）
cd project/book_search
python -m src.tools.download_modelscope_dataset \
	--repo-id wzywuan/Novel-Collection \
	--repo-type dataset \
	--revision master \
	--output-dir data

# 2) 一键检查断点状态文件是否齐全
test -f data/books.db && echo "OK data/books.db" || echo "MISS data/books.db"
test -f data/shards/index.json && echo "OK data/shards/index.json" || echo "MISS data/shards/index.json"
test -f data/.shards.modelscope-upload-manifest.json && echo "OK data/.shards.modelscope-upload-manifest.json" || echo "MISS data/.shards.modelscope-upload-manifest.json"

# 3) 查看当前进度（确认能继续）
cd project/book_search
python -m src.crawler.engine stats

# 4) 继续抓正文（断点重跑）
python -m src.crawler.engine crawl-content --start 1 --end 10000 --concurrency 8 --batch-size 40
```

## 从 ModelScope 下载数据

如果你需要在新机器或中断后继续爬取，可以先把 ModelScope 上的数据下载到本地：

```bash
cd project/book_search
python -m src.tools.download_modelscope_dataset \
	--repo-id wzywuan/Novel-Collection \
	--repo-type dataset \
	--revision master \
	--output-dir data/modelscope_download
```

常见用法：

```bash
# 只下载分片和索引
python -m src.tools.download_modelscope_dataset \
	--repo-id wzywuan/Novel-Collection \
	--allow-pattern '*.db' \
	--allow-pattern 'index.json' \
	--output-dir data/shards

# 下载前先清空目标目录（谨慎）
python -m src.tools.download_modelscope_dataset \
	--repo-id wzywuan/Novel-Collection \
	--output-dir data/shards \
	--clean-output
```

说明：

- 脚本会先下载到本地缓存，再同步到 `--output-dir`。
- `--token` 可省略，默认读取 `MODELSCOPE_API_TOKEN` 或 `MODELSCOPE_TOKEN`。
- 下载后可继续执行 `src.crawler.engine` 的 `crawl-content` 或 `export-shards --only-changed`。
