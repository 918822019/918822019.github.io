# 快速上手

本文档帮助你在 5 分钟内启动小说搜索系统。

## 前提条件

已完成 [安装指南](installation.md) 中的所有步骤。

## 场景一：搜索问答（推荐体验）

这是系统的核心功能——输入问题，AI 搜索相关书籍并生成回答。

### 1. 准备数据

确保 `data/books.db` 存在且包含已润色和向量化的数据。

如果数据库为空，可以先从 ModelScope 下载示例数据：

```bash
cd project/book_search
python -m src.tools.download_modelscope_dataset \
    --repo-id wzywuan/Novel-Collection \
    --repo-type dataset \
    --output-dir data
```

### 2. 启动搜索引擎

```bash
python main.py
```

启动后会自动执行以下步骤：

1. 连接数据库
2. 检查向量化状态
3. 如有必要，提示执行向量化
4. 加载 Faiss 索引
5. 进入交互式问答

### 3. 开始搜索

```
============================================================
📚 小说智能搜索系统
============================================================

Step 1: 连接数据库...
✅ 数据库连接成功: data/books.db

Step 2: 检查数据向量化状态...
   润色后书籍总数: 1500
   已向量化的书: 1500
   待向量化数量: 0

✅ 所有书籍已完成向量化

Step 3: 加载向量索引...
✅ Faiss 索引加载成功: data/books.polish_embedding.faiss
   索引维度: 1536
   向量数量: 1500

============================================================
💬 进入问答模式（输入 'quit' 或 'exit' 退出）
============================================================

🔍 请输入您的问题: 有没有好看的玄幻小说，主角是废柴逆袭的？

⏳ 正在搜索相关书籍...

📖 找到 5 本相关书籍:

1. 《斗破苍穹》
   简介: 三十年河东，三十年河西，莫欺少年穷！年仅15岁的萧家废柴...
   相关性: 0.923

2. 《武动乾坤》
   简介: 大炎王朝天都郡炎城偏僻小院，一个名为林动的少年...
   相关性: 0.891

⏳ 正在生成回答...

💡 AI 推荐:
----------------------------------------------------------------------
根据您的需求，为您推荐以下几本玄幻废柴逆袭小说：
1. 《斗破苍穹》- 经典废柴逆袭代表作...
----------------------------------------------------------------------
```

## 场景二：从零构建数据

如果你想从头开始构建自己的数据集。

### 1. 爬取数据

```bash
cd project/book_search

# 抓取书籍目录（1-1000 号）
python -m src.crawler.engine crawl-books --start 1 --end 1000 --concurrency 12

# 抓取章节正文
python -m src.crawler.engine crawl-content --start 1 --end 1000 --concurrency 8 --batch-size 40

# 查看进度
python -m src.crawler.engine stats
```

### 2. 预处理数据

```bash
cd project/book_search

# 一次性执行完整 Pipeline（润色 + 向量化 + 打标）
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

### 3. 启动搜索

```bash
python main.py
```

## 常用命令速查

| 命令 | 说明 |
|------|------|
| `python main.py` | 启动搜索引擎（交互式问答） |
| `python -m src.tools.cli polish --limit 100 --sleep 0.1` | 执行文本润色（`--model` 指定模型，`--overwrite` 覆盖已有） |
| `python -m src.tools.cli embed --limit 100 --sleep 0.1` | 执行向量化入库 |
| `python -m src.tools.cli search --query "玄幻" --top-k 10` | 基于 embedding 搜索 |
| `python -m src.tools.cli stats` | 查看数据库基础统计（书籍/章节/润色/向量化） |
| `python -m src.tools.stats` | 全线统计（数据库/润色/向量化/打标/分片/Faiss） |
| `python -m src.tools.look_db` | 快速查看数据库表结构 |
| `python -m src.crawler.engine crawl-books --start 1 --end 1000` | 爬取书籍目录 |
| `python -m src.crawler.engine crawl-content --start 1 --end 1000` | 爬取章节正文 |
| `python -m src.crawler.engine stats` | 查看爬取进度 |
| `python -m src.crawler.engine export-shards --start 1 --end 1000 --shard-size 200` | 导出分片 |
| `python -m src.tools.upload_modelscope_dataset --repo-id wzywuan/Novel-Collection --dry-run` | 预览上传 |
| `python -m src.tools.download_modelscope_dataset --repo-id wzywuan/Novel-Collection --output-dir data` | 下载数据 |

## 下一步

- 了解 [系统架构](../architecture/overview.md) 设计
- 阅读 [预处理指南](../guides/preprocessing-guide.md) 了解数据处理细节
- 查看 [搜索问答指南](../guides/search-guide.md) 了解检索策略配置
