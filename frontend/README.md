# frontend

这是从仓库根目录提取出的前端静态站点副本，放在 `frontend/` 目录下。目的是把前端资源整理为可独立查看的子目录，保留原始文件不变。

使用方法：

- 直接在文件管理器中打开 `frontend/index.html`（部分功能可能受同源策略影响）。
- 推荐在本地启动一个静态服务器：

```bash
# 在仓库根目录运行（推荐）
python -m http.server 8000
# 然后打开 http://localhost:8000/frontend/
```

说明：

- `frontend/` 中的页面会从上级目录的 `docs/` 中读取文档索引（`../docs/index.json`），因此建议以仓库根目录为工作目录启动静态服务器。
- 我已保留仓库根目录的 `index.html` 与 `assets/`，本次操作只是复制并调整了一个自包含的 `frontend/` 视图。如需把根目录也改为只保留 `frontend/`，我可以继续移动并更新引用。

## 数据观察器 (Data Inspector)

当前页面已经升级成偏 PC 端的数据中台观察台，目标不是只看样本，而是快速判断抓取数据的规模、质量和异常位置：

- 页面: [frontend/data-inspector.html](frontend/data-inspector.html)
- 脚本: [frontend/assets/data-inspector.js](frontend/assets/data-inspector.js)

使用方法（在仓库根目录启动静态服务器）：

```bash
python -m http.server 8000
# 打开 http://localhost:8000/frontend/data-inspector.html
```

主要功能：

- 支持普通 JSON、NDJSON、SQLite 导出 manifest、shards/index.json 四类观察模式
- 顶部核心指标卡可直接看总量、覆盖率、样本规模和最近导出状态
- 质量信号区会提示 0 覆盖率分片、部分补齐分片、样本缺失等风险
- 字段画像区提供类型推断、字段覆盖率、示例值与去重样本数
- 明细表支持关键词过滤、字段聚焦、排序和导出当前视图 CSV
- 点击任意记录可查看完整 JSON，适合巡检单条样本或单个 shard 元信息

## 从 SQLite 导出完整拉取数据

要观察完整的拉下来的数据（数据库内的 books/chapters），先运行导出脚本将 SQLite 导出到 `frontend/data/`：

```bash
py scripts/data/export_to_frontend.py \
	--db-path data/book_search/books.db \
	--output-dir data/exports \
	--sample-size 200
```

脚本会在 `data/exports/` 下生成：

- `manifest.json`（包含 counts 和文件名）
- `books_full.ndjson` / `chapters_full.ndjson`（完整 NDJSON，可供下载）
- `books_sample.json` / `chapters_sample.json`（各自的样本，用于快速预览）

运行后，在浏览器打开 `http://localhost:8000/frontend/data-inspector.html`。

推荐观察顺序：

- 先看 `data/book_search/shards/index.json`：确认哪些分片正文覆盖率为 0 或仍在补抓
- 再看 `data/exports/manifest.json`：检查导出的 books/chapters 样本是否完整、字段是否稳定
- 最后看 `data/book_search/books.json` 或自定义路径：做通用 JSON / NDJSON 结构观察

如果需要我可以：

- 自动扫描 `data/book_search/shards/` 目录并列出所有分片文件；
- 增加更多导出格式（Parquet/Excel）；
- 对非常大的文件做流式采样以避免一次性加载全部内容。
