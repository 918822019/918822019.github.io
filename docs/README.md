# 技术文档总览

欢迎来到文档区。这个目录下面会存放你所有的技术文档文件。

- `architecture/`：架构设计
- `guides/`：开发指导
- `faqs/`：常见问题

请根据目录结构继续新增文档，然后更新 `docs/index.json`。

## 自动生成目录

本仓库包含自动生成 `docs/index.json` 的机制：

1. `build-docs-index.js` 会递归扫描 `docs/`，生成树状结构 JSON。
2. 在 `push` 到 `main` 时，由 GitHub Actions 工作流 `.github/workflows/auto-build-docs-index.yml` 执行。

手动执行（也可本地预览）：

```bash
node build-docs-index.js
```

如果文件内容变更，工作流会自动提交并推送更新后的 `docs/index.json`。

## 一键更新（推荐）

仓库提供了一个便捷脚本，可以一键生成并推送 `docs/index.json`：

```bash
npm run update-docs
```

脚本会执行：

- 运行 `node build-docs-index.js` 生成索引
- 将 `docs/index.json` 添加到 git 暂存区
- 如果有变更则提交并 `git push`

注意：需要在本地有 `git` 访问权限并配置好用户名/邮箱；另需安装 `node` 与 `npm`。
