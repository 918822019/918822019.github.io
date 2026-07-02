# 918822019.github.io - 个人技术博客与项目集合

个人技术博客仓库，同时汇集 AI 量化、搜索推荐、继续预训练等方向的实验项目。

## 项目结构

```
.
├── data/                       # [集中数据管理] 所有项目的数据统一存放
│   ├── book_search/           # book_search 的数据库、分片、Faiss 索引
│   ├── models/                # 模型权重 (Qwen3.5-0.8B/9B/35B-A3B)
│   ├── cache/                 # 训练缓存 (logits_cache 等)
│   └── exports/               # 前端导出数据
├── frontend/                   # 前端演示页面 (博客阅读器 + 数据巡检台)
├── project/                    # 子项目代码目录 (仅代码，数据引用 data/)
│   ├── book_search/           # 起点搜书 - 书籍推荐系统
│   ├── quant/                 # 模型量化工具集 (SVD/FSQ/RVQ)
│   ├── ContinuePretrain/      # 继续预训练实验
│   └── tvm/                   # Apache TVM 编译器 (submodule)
├── scripts/                    # 构建与数据管理脚本
│   ├── data/                  # 数据管理脚本 (上传/下载/校验/远程直读)
│   ├── build-docs-index.py    # 文档索引生成
│   ├── publish-doc.ps1        # 文档发布 (Windows)
│   └── publish-doc.sh         # 文档发布 (macOS/Linux)
└── docs/                       # 技术文档 / 博客内容
```

## 集中式数据管理

所有项目的数据统一存放在 `data/` 目录下，项目代码只通过相对路径引用。

```powershell
# 从 ModelScope 拉取最新数据
.\scripts\data\download_data.ps1

# 上传数据到 ModelScope
.\scripts\data\upload_data.ps1 -Incremental -CommitMessage "update"

# 同步模型权重
.\scripts\data\sync_models.ps1

# 远程直读 / 选择性下载 / 数据导入
py scripts/data/modelscope_reader.py list
py scripts/data/modelscope_reader.py bootstrap --shards 3 --create-db

# 导出数据库到前端
py scripts/data/export_to_frontend.py

# 数据完整性校验
py scripts/data/data_validate.py
```

详见 [数据架构文档](docs/架构设计/数据架构.md)。

## 子项目

### book_search - 起点搜书
基于向量检索 + LLM 的小说推荐系统。支持爬虫、标签、向量化、RAG 问答全流程。

| 模块 | 说明 |
|------|------|
| `crawler/` | 爬取 69shu.com 小说目录与正文 |
| `process/polish/` | LLM 润色 + Embedding 向量化 |
| `process/agent/` | 智能问答 Agent |
| `tools/` | ModelScope 上传/下载、CLI 工具 |

### quant - 模型量化工具集
MoE 模型量化实验 (SVD + FSQ + KL 蒸馏)。

| 子项目 | 说明 |
|--------|------|
| `svd_quant/` | SVD 量化 + FSQ + 知识蒸馏 |
| `llama.cpp/` | 高性能 LLM 推理引擎 (submodule) |
| `data/models/` | 模型权重 (Qwen3.5-0.8B/9B/35B-A3B) |

### ContinuePretrain - 继续预训练
基于 Qwen3.5-0.8B 的继续预训练实验，包含数据清洗（去噪、去重）管道。

### tvm - Apache TVM
机器学习编译器框架 (submodule)。

## 博客

访问 [https://918822019.github.io](https://918822019.github.io) 查看博客内容。

### 本地预览
```bash
python -m http.server 8000
# 打开 http://localhost:8000
```

### 发布文档
```bash
npm run publish-doc -- "docs: update notes"
```

## 技术栈

| 领域 | 技术 |
|------|------|
| 前端 | 原生 HTML/CSS/JS |
| AI/ML | PyTorch, Transformers, HuggingFace |
| 量化 | SVD, FSQ, RVQ, KL 蒸馏, llama.cpp |
| 搜索 | Faiss, Embedding, RAG |
| 数据 | SQLite, ModelScope |
| 部署 | GitHub Pages, GitHub Actions |

## 链接

- [GitHub 仓库](https://github.com/918822019/918822019.github.io)
- [ModelScope 数据集](https://modelscope.cn/datasets/wzywuan/Novel-Collection)

---

*最后更新: 2026-06-26*
