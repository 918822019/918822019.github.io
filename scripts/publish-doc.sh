#!/usr/bin/env sh
set -eu

msg=${1:-"docs: update"}
branch=${2:-"doc"}

if ! command -v git >/dev/null 2>&1; then
  echo "错误：未找到 git。"
  exit 1
fi

echo "[1/5] 拉取远程 ${branch} ..."
git pull --rebase origin "$branch"

echo "[2/5] 生成 docs/index.json ..."
if command -v node >/dev/null 2>&1; then
  node build-docs-index.js
elif command -v python3 >/dev/null 2>&1; then
  python3 scripts/build-docs-index.py
else
  echo "错误：无法找到 node 或 python3。"
  exit 1
fi

echo "[3/5] 暂存改动 ..."
git add -A

echo "[4/5] 检查是否有可提交内容 ..."
if git diff --cached --quiet; then
  echo "没有变化，跳过提交。"
  exit 0
fi

echo "[5/5] 提交并推送到 ${branch} ..."
git commit -m "$msg"
git push origin "$branch"

echo "完成：已推送到 ${branch}"