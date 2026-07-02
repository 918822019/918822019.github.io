# 技术文档总览

```
docs/
├── 架构设计/         # 项目架构与设计文档
│   └── 数据架构.md   # 集中式数据管理架构
├── 模型相关的论文/   # AI 论文阅读笔记
├── 数学/             # 数学相关笔记
├── AI搜索/           # AI 搜索技术对比
├── infra/            # 基础设施笔记
└── RAG/              # RAG 相关笔记

项目文档（独立部署，非 docs/ 内）:
├── project/book_search/docs/   # 起点搜书完整文档
├── project/quant/svd_quant/docs/ # 量化实验文档
└── scripts/data/README.md      # 数据管理脚本文档
```

## 本地查看

```bash
# 在仓库根目录启动静态服务器
python -m http.server 8000
# 打开 http://localhost:8000
```

## 自动生成目录

`docs/index.json` 由 `scripts/build-docs-index.py` 自动生成，供前端导航使用。

```bash
py scripts/build-docs-index.py
```

## 一键发布

```bash
npm run publish-doc -- "docs: update notes"
```

脚本会：`git pull` → 生成 `docs/index.json` → `git commit` → `git push` 到 `doc` 分支。
