#!/usr/bin/env sh
set -eu

echo "生成 docs/index.json ..."

# 优先使用 node，其次回退到 python3，如果都不可用则报错
if command -v node >/dev/null 2>&1; then
  node build-docs-index.js
elif command -v python3 >/dev/null 2>&1; then
  python3 scripts/build-docs-index.py
else
  echo "错误：无法找到 node 或 python3，请先安装其中一个。"
  exit 1
fi

echo "将 docs/index.json 添加到暂存区 ..."
git add docs/index.json

echo "检查是否有变更..."
if git diff --cached --quiet; then
  echo "No changes in docs/index.json"
  exit 0
fi

echo "提交并推送变更..."
git commit -m "chore: regenerate docs/index.json"
git push

echo "完成。"
