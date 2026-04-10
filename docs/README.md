# 技术文档总览

欢迎来到文档区。这个目录下面会存放你所有的技术文档文件。

- `architecture/`：架构设计
- `guides/`：开发指导
- `faqs/`：常见问题

请根据目录结构继续新增文档，然后更新 `docs/index.json`。

## 自动生成目录

本仓库包含自动生成 `docs/index.json` 的机制：

1. `build-docs-index.js` 会递归扫描 `docs/`，生成树状结构 JSON。
2. 在 `push` 到 `doc` 时，由 GitHub Actions 工作流 `.github/workflows/auto-build-docs-index.yml` 执行。

手动执行（也可本地预览）：

```bash
node build-docs-index.js
```

如果文件内容变更，工作流会自动提交并推送更新后的 `docs/index.json`。

## 一键发布（推荐）

仓库提供了 `scripts/publish-doc.sh`，可一键完成“更新索引 + 提交 + 推送到 `doc`”。

### 用法

```bash
bash scripts/publish-doc.sh "docs: update notes"
```

或使用 npm：

```bash
npm run publish-doc -- "docs: update notes"
```

### 脚本会做什么

1. `git pull --rebase origin doc`
2. 生成 `docs/index.json`（优先 `node`，回退 `python3`）
3. `git add -A`
4. 若无变化则退出
5. 自动提交并推送到 `doc`

注意：本地需具备 `git` 推送权限，并安装 `node` 或 `python3`。
