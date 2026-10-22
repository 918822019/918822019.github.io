# frontend

博客界面统一由仓库根目录的 `index.html` 和 `assets/` 提供，包含首页、文章阅读器、搜索与主题切换。`frontend/index.html` 保留为兼容入口，会保留查询参数和章节锚点跳转到博客首页。

在仓库根目录运行 `python -m http.server 8000`，访问 `http://localhost:8000/` 即可预览。数据观察器仍为独立页面。

## 博客功能与维护

- 全文搜索：读取 `assets/content/search.json`，支持空格分隔的多个关键词、正文命中片段与高亮。首次搜索时加载索引。
- 我的书架：`/?view=library` 包含收藏、稍后阅读和最近阅读。数据保存在当前浏览器的 `field-notes-library-v1`；支持 JSON 导入、导出。导入会合并收藏，保留较新的阅读进度，忽略已不存在的文章。
- 阅读设置：字号、行距、宽度与专注模式保存在 `field-notes-reading-v1`。续读优先定位到章节及其偏移，章节变更时回退为整体进度。
- 图片查看器支持缩放、上一张／下一张及键盘方向键；代码使用本地 Prism，长代码默认折叠但可完整展开和复制。
- 系列路线：编辑 `scripts/series.json`，按顺序列出文章路径。生成器会拒绝无效路径和重复条目。
- 归档按文章最近一次 Git 提交日期分组；`feed.xml` 为 RSS 2.0，以同一日期生成更新条目。
- Pcode 交互图表在 `assets/experiments.js` 中维护，必须与原报告一致；统一 GPU、历史 E2、手机 CPU 和交付体积分开展示。不要用不同口径的数据生成跨组结论。

修改文章或系列配置后运行 `npm run build-docs-index`，将 `docs/index.json`、`assets/content/search.json`、`assets/content/series.json` 和 `feed.xml` 一并提交。Pages 部署也会重新生成这些文件。`assets/notebook-core.js` 包含可独立测试的搜索、备份校验和合并逻辑。

```bash
npm ci
npm test
npx playwright install chromium
# 在另一个终端，从仓库根目录启动服务器
python -m http.server 8765 --bind 127.0.0.1
npm run test:ui
```

浏览器测试可以用 `TEST_BASE_URL` 指向其他测试地址，或用 `PLAYWRIGHT_CHROMIUM_EXECUTABLE` 指定现有 Chromium。截图与测试备份只写入被忽略的 `.test-artifacts/`。

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
