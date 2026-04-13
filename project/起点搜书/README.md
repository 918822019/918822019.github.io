# 起点搜书

本模块实现基于用户查询的书籍推荐系统，包含查询处理、数据召回、推荐理由生成和结果展示等功能。

当前抓取脚本已经改为直接请求站点真实接口，并将结果写入 SQLite，支持先抓目录、再抓正文、断点续抓。

现在也支持把主库按 book_id 范围导出成多个 shard SQLite 文件，配合上传脚本做增量上传。

## 目录结构

- data/books.json 书籍数据
- data/books.db SQLite 抓取数据库
- tools/db_viewer.py 数据库可视化管理界面 (见文件注释)
- script/start_viewer.sh 可视化界面启动脚本 (见文件注释)
- tests/test_db.py 数据库连接测试脚本 (见文件注释)
- query_processor.py 查询预处理
- rag_retriever.py 数据召回
- app.py 推荐API服务
- data_get/main.py 抓取脚本

## 快速开始

1. 准备书籍数据到 data/books.json
2. 运行 app.py 启动服务
3. 前端通过 API 获取推荐结果

## 抓取命令

在 project/起点搜书/data_get 目录执行：

```bash
python main.py crawl-books --start 1 --end 10000 --concurrency 12
```

先抓 1 到 10000 的书籍首页和章节目录，写入 data/books.db。

```bash
python main.py crawl-content --start 1 --end 10000 --concurrency 12 --batch-size 120
```

再基于数据库里的章节目录补全正文内容。

更稳一些的正文抓取建议这样跑：

```bash
python main.py crawl-content --start 1 --end 10000 --concurrency 8 --batch-size 40 --chapter-progress-every 500 --sqlite-synchronous FULL --min-request-interval 0.03 --request-jitter 0.05 --retry-backoff-base 1.5 --retry-backoff-max 12
```

这组参数更适合长时间后台运行：

- `sqlite-synchronous FULL` 提高断电场景下的落盘可靠性。
- `min-request-interval` 和 `request-jitter` 会把请求节奏打散，减少固定频率特征。
- 请求头会自动轮换 User-Agent，并按书籍页/章节页生成 Referer。
- `crawl-content` 只会抓未完成章节，断掉后再次执行会继续补齐。
- `chapter-progress-every` 会按章节输出进度，长任务更容易观察。

```bash
python main.py sync-all --start 1 --end 10000 --concurrency 12 --batch-size 120
```

按顺序执行目录抓取和正文抓取。

```bash
python main.py stats
```

查看当前数据库里的书籍数、章节数和正文完成进度。

```bash
python main.py crawl-content --start 1 --end 1 --max-pending-per-book 5 --batch-size 5
```

按每本书限制本次处理的未完成章节数，适合分批跑或小样本验证。

```bash
python main.py export-shards --start 1 --end 10000 --shard-size 200 --output-dir ../data/shards
```

把主库按 book_id 范围导出为 shard SQLite 文件。默认每 200 本书一个分片，并在输出目录下生成一个 index.json。

```bash
python main.py export-shards --start 1 --end 10000 --shard-size 200 --output-dir ../data/shards --only-changed
```

只重导出发生变化的 shard。依据是输出目录里上一版 `index.json` 记录的 source_fingerprint。

## 说明

- 默认数据库路径是 data/books.db。
- crawl-books 会把书籍元信息写入 books 表，把章节目录写入 chapters 表。
- crawl-content 只会抓还没完成的章节，可以重复执行。
- 当前 SQLite 使用 WAL，配合 `--sqlite-synchronous FULL` 时更适合怕断电的长任务。
- export-shards 不会改动主库，而是从主库读取并导出静态 shard 文件，适合上传到数据集仓库。
- 当前 app.py 仍然读取 data/books.json，还没有切到 SQLite。如果你要，我可以下一步把推荐服务也改成直接读数据库。

## 数据库可视化界面

提供了 Web 界面来查看和管理数据库中的书籍和章节数据。

### 快速开始

```bash
# 1. 测试数据库连接
cd tests
python3 test_db.py

# 2. 启动可视化界面
cd ..
./script/start_viewer.sh
# 或
python3 tools/db_viewer.py

# 3. 访问界面
# 打开浏览器: http://localhost:5000
```

### 功能特性

- 📊 数据统计面板：展示书籍总数、章节数、完成进度等
- 🔍 智能搜索：支持按书名、作者、简介搜索
- 🏷️ 分类筛选：按书籍分类和完成状态过滤
- 📈 进度可视化：直观显示每本书的章节抓取进度
- 🕐 最近更新：展示最近更新的书籍列表

详细用法请查看 `tools/db_viewer.py`、`script/start_viewer.sh` 和 `tests/test_db.py` 的文件注释。

## 上传到 ModelScope

先安装依赖：

```bash
pip install -r requirements.txt
```

设置 token 后，可以把当前项目里的 data 目录直接上传到 ModelScope dataset 仓库：

```bash
export MODELSCOPE_API_TOKEN="你的 token"
python tools/upload_modelscope_dataset.py \
	--repo-id wzywuan/Novel-Collection \
	--folder-path data \
	--commit-message "upload dataset folder to repo"
```

如果只想上传默认的 data 目录，`--folder-path` 可以省略。

脚本默认会尝试对 `data/books.db` 做一致性快照，再上传快照目录，而不是直接上传正在写入的活动数据库文件。

如果你想保留这份临时快照，方便重复上传或人工检查，可以加上：

```bash
python tools/upload_modelscope_dataset.py \
	--repo-id wzywuan/Novel-Collection \
	--keep-snapshot
```

脚本位置：

- `tools/upload_modelscope_dataset.py` 负责登录 ModelScope 并执行 `upload_folder`
- `tools/download_modelscope_dataset.py` 负责从 ModelScope 下载并同步到本地目录
- `requirements.txt` 包含 Flask 和 ModelScope SDK 的最小依赖

## 增量上传建议流程

对于 `data/books.db` 这种大单文件，不适合直接做真正意义上的增量上传。更合适的做法是：

1. 继续抓取到主库 `data/books.db`
2. 定期导出 shards 到 `data/shards`
3. 对 `data/shards` 目录执行增量上传

先导出分片：

```bash
cd project/起点搜书/data_get
python main.py export-shards --start 1 --end 10000 --shard-size 200 --output-dir ../data/shards
```

后续增量更新建议使用：

```bash
cd project/起点搜书/data_get
python main.py export-shards --start 1 --end 10000 --shard-size 200 --output-dir ../data/shards --only-changed
```

先看看这次会上传哪些文件：

```bash
cd project/起点搜书
python tools/upload_modelscope_dataset.py \
	--repo-id wzywuan/Novel-Collection \
	--folder-path data/shards \
	--incremental \
	--dry-run
```

确认后执行真正上传：

```bash
cd project/起点搜书
python tools/upload_modelscope_dataset.py \
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
以及 `tools/upload_modelscope_dataset.py --incremental` 的断点状态，建议把 `data` 目录全量上传。

先在旧服务器执行 dry-run，确认上传列表里包含隐藏状态文件：

```bash
cd project/起点搜书
python tools/upload_modelscope_dataset.py \
	--repo-id wzywuan/Novel-Collection \
	--folder-path data \
	--include-hidden \
	--dry-run
```

确认后执行全量上传：

```bash
cd project/起点搜书
python tools/upload_modelscope_dataset.py \
	--repo-id wzywuan/Novel-Collection \
	--folder-path data \
	--include-hidden \
	--commit-message "full data upload for resume"
```

然后在新服务器下载到同样路径：

```bash
cd project/起点搜书
python tools/download_modelscope_dataset.py \
	--repo-id wzywuan/Novel-Collection \
	--repo-type dataset \
	--revision master \
	--output-dir data
```

说明：

-- `crawl-content` 的断点在 `data/books.db` 里的 `is_content_fetched` 状态。
-- `export-shards --auto-continue` 依赖 `data/shards/index.json`。
-- `tools/upload_modelscope_dataset.py --incremental` 依赖本地 manifest，
例如 `data/.shards.modelscope-upload-manifest.json`。

- `--include-hidden` 用于确保上述隐藏状态文件也被上传。

可直接复制的“注释版”恢复与续跑命令：

```bash
# 1) 拉取完整数据到 data 目录（保持与旧服务器相同相对路径）
cd project/起点搜书
python tools/download_modelscope_dataset.py \
	--repo-id wzywuan/Novel-Collection \
	--repo-type dataset \
	--revision master \
	--output-dir data

# 2) 一键检查断点状态文件是否齐全
test -f data/books.db && echo "OK data/books.db" || echo "MISS data/books.db"
test -f data/shards/index.json && echo "OK data/shards/index.json" || echo "MISS data/shards/index.json"
test -f data/.shards.modelscope-upload-manifest.json && echo "OK data/.shards.modelscope-upload-manifest.json" || echo "MISS data/.shards.modelscope-upload-manifest.json"

# 3) 查看当前进度（确认能继续）
cd data_get
python main.py stats

# 4) 继续抓正文（断点重跑）
python main.py crawl-content --start 1 --end 10000 --concurrency 8 --batch-size 40
```

## 从 ModelScope 下载数据

如果你需要在新机器或中断后继续爬取，可以先把 ModelScope 上的数据下载到本地：

```bash
cd project/起点搜书
python tools/download_modelscope_dataset.py \
	--repo-id wzywuan/Novel-Collection \
	--repo-type dataset \
	--revision master \
	--output-dir data/modelscope_download
```

常见用法：

```bash
# 只下载分片和索引
python tools/download_modelscope_dataset.py \
	--repo-id wzywuan/Novel-Collection \
	--allow-pattern '*.db' \
	--allow-pattern 'index.json' \
	--output-dir data/shards

# 下载前先清空目标目录（谨慎）
python tools/download_modelscope_dataset.py \
	--repo-id wzywuan/Novel-Collection \
	--output-dir data/shards \
	--clean-output
```

说明：

- 脚本会先下载到本地缓存，再同步到 `--output-dir`。
- `--token` 可省略，默认读取 `MODELSCOPE_API_TOKEN` 或 `MODELSCOPE_TOKEN`。
- 下载后可继续执行 `data_get/main.py` 的 `crawl-content` 或 `export-shards --only-changed`。
