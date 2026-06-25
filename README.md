# 918822019.github.io - 个人技术博客与项目集合

这是我的个人技术博客仓库，同时也包含了多个 AI、量化、搜索推荐等方向的子项目。

## 📁 项目结构

```
.
├── index.html                 # 博客前端主页
├── assets/                    # 静态资源
│   ├── styles.css            # 样式文件
│   └── app.js                # 主要交互逻辑
├── frontend/                  # 前端演示页面
├── project/                   # 核心子项目目录
│   ├── quant/                # 模型量化相关项目
│   ├── ContinuePretrain/     # 继续预训练项目
│   ├── book_search/          # 起点搜书 - 书籍推荐系统
│   └── tvm/                  # Apache TVM 编译器框架
├── scripts/                   # 构建部署脚本
└── 时间规划.md               # 个人时间规划笔记
```

## 🚀 子项目概览

### 1. quant - 模型量化工具集
包含多个量化相关的子项目：
- **svd_quant** - 基于 SVD 的 MoE 模型量化工具
- **llama.cpp** - 高性能 LLM 推理引擎 (C/C++)
- **model/** - 多个 Qwen 模型权重
  - Qwen3.5-0.8B
  - Qwen3.5-9B
  - Qwen3.5-35B-A3B

### 2. ContinuePretrain - 继续预训练
- 基于 Qwen3.5-0.8B 的继续预训练实验
- 包含数据处理管道和训练脚本

### 3. book_search - 起点搜书推荐系统
- 基于用户查询的书籍推荐系统
- 支持 SQLite 数据存储、增量爬取、LLM 打标签
- 支持 ModelScope 数据集上传/下载

### 4. tvm - Apache TVM
- 开源机器学习编译框架
- 支持多种硬件后端部署

## 🌐 博客访问

访问 [https://918822019.github.io](https://918822019.github.io) 查看博客内容。

## 📦 快速开始

### 博客本地预览
```bash
# 使用任意静态服务器，例如：
python -m http.server 8000
# 或
npx serve .
```

### 更新文档索引
```bash
bash scripts/publish-doc.sh "docs: update notes"
```

## 🛠️ 技术栈

- **前端**: 原生 HTML/CSS/JS
- **后端**: Python (Flask, FastAPI)
- **AI/ML**: PyTorch, Transformers, Hugging Face
- **量化**: llama.cpp, GGUF, SVD 量化
- **数据**: SQLite, ModelScope
- **部署**: GitHub Pages, GitHub Actions

## 📄 许可证

各子项目遵循各自的许可证，详见对应目录下的 LICENSE 文件。

## 🔗 相关链接

- [GitHub 仓库](https://github.com/918822019/918822019.github.io)
- [ModelScope 数据集](https://modelscope.cn/datasets/wzywuan/Novel-Collection)

---
*持续更新时间: 2026-06-23