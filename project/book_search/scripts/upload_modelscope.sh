#!/bin/bash
# ============================================================================
# ModelScope 数据上传脚本（Ubuntu / Bash）
# ============================================================================
# 用法示例：
#   chmod +x script/upload_modelscope.sh
#
#   # 1) 先 dry-run（默认就是 dry-run）
#   ./script/upload_modelscope.sh \
#     --repo-id wzywuan/Novel-Collection \
#     --folder-path data
#
#   # 2) 正式上传（加 --run）
#   ./script/upload_modelscope.sh \
#     --repo-id wzywuan/Novel-Collection \
#     --folder-path data \
#     --run
#
#   # 3) 对 shards 走增量上传
#   ./script/upload_modelscope.sh \
#     --repo-id wzywuan/Novel-Collection \
#     --folder-path data/shards \
#     --incremental \
#     --run
#
# 环境变量：
#   MODELSCOPE_API_TOKEN / MODELSCOPE_TOKEN
# ============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_REPO_ID="wzywuan/Novel-Collection"
DEFAULT_FOLDER_PATH="data"
DEFAULT_REPO_TYPE="dataset"
DEFAULT_SQLITE_SNAPSHOT="auto"

REPO_ID="$DEFAULT_REPO_ID"
FOLDER_PATH="$DEFAULT_FOLDER_PATH"
REPO_TYPE="$DEFAULT_REPO_TYPE"
COMMIT_MESSAGE="upload by script $(date +%F_%H-%M-%S)"
TOKEN="${MODELSCOPE_API_TOKEN:-${MODELSCOPE_TOKEN:-}}"
MANIFEST_PATH=""
SQLITE_SNAPSHOT="$DEFAULT_SQLITE_SNAPSHOT"

INCREMENTAL=0
INCLUDE_HIDDEN=0
DRY_RUN=1

print_help() {
  cat <<'EOF'
ModelScope 上传封装脚本（Ubuntu / Bash）

参数：
  --repo-id <owner/name>          仓库 ID（默认: wzywuan/Novel-Collection）
  --folder-path <path>            上传目录（默认: data）
  --repo-type <dataset|model>     仓库类型（默认: dataset）
  --commit-message <msg>          提交信息（默认: upload by script 时间戳）
  --token <token>                 可选；不传则从环境变量读取
  --manifest-path <path>          增量 manifest 路径（可选）
  --sqlite-snapshot <mode>        auto|always|never（默认: auto）
  --incremental                   启用增量上传
  --include-hidden                上传隐藏文件
  --dry-run                       只检查不上传（默认）
  --run                           正式上传
  -h, --help                      显示帮助

推荐流程：
  1) 先用默认 dry-run 看上传清单
  2) 确认后加 --run 执行正式上传
  3) 大目录建议配合 --incremental
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-id)
      REPO_ID="$2"; shift 2 ;;
    --folder-path)
      FOLDER_PATH="$2"; shift 2 ;;
    --repo-type)
      REPO_TYPE="$2"; shift 2 ;;
    --commit-message)
      COMMIT_MESSAGE="$2"; shift 2 ;;
    --token)
      TOKEN="$2"; shift 2 ;;
    --manifest-path)
      MANIFEST_PATH="$2"; shift 2 ;;
    --sqlite-snapshot)
      SQLITE_SNAPSHOT="$2"; shift 2 ;;
    --incremental)
      INCREMENTAL=1; shift ;;
    --include-hidden)
      INCLUDE_HIDDEN=1; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --run)
      DRY_RUN=0; shift ;;
    -h|--help)
      print_help; exit 0 ;;
    *)
      echo "未知参数: $1"
      print_help
      exit 1 ;;
  esac
done

if [[ "$SQLITE_SNAPSHOT" != "auto" && "$SQLITE_SNAPSHOT" != "always" && "$SQLITE_SNAPSHOT" != "never" ]]; then
  echo "错误: --sqlite-snapshot 仅支持 auto|always|never"
  exit 1
fi

if [[ "$REPO_TYPE" != "dataset" && "$REPO_TYPE" != "model" ]]; then
  echo "错误: --repo-type 仅支持 dataset|model"
  exit 1
fi

if [[ ! -d "$FOLDER_PATH" ]]; then
  echo "错误: 上传目录不存在: $FOLDER_PATH"
  exit 1
fi

if [[ -f ".venv/bin/activate" ]]; then
  # Ubuntu / Linux
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
else
  echo "错误: 未找到 python3/python"
  exit 1
fi

if [[ -z "$TOKEN" ]]; then
  echo "错误: 未检测到 token，请设置 MODELSCOPE_API_TOKEN 或传 --token"
  exit 1
fi

CMD=(
  "$PYTHON_CMD" "-m" "src.tools.upload_modelscope_dataset"
  "--repo-id" "$REPO_ID"
  "--folder-path" "$FOLDER_PATH"
  "--repo-type" "$REPO_TYPE"
  "--commit-message" "$COMMIT_MESSAGE"
  "--sqlite-snapshot" "$SQLITE_SNAPSHOT"
)

if [[ -n "$TOKEN" ]]; then
  CMD+=("--token" "$TOKEN")
fi

if [[ -n "$MANIFEST_PATH" ]]; then
  CMD+=("--manifest-path" "$MANIFEST_PATH")
fi

if [[ "$INCREMENTAL" -eq 1 ]]; then
  CMD+=("--incremental")
fi

if [[ "$INCLUDE_HIDDEN" -eq 1 ]]; then
  CMD+=("--include-hidden")
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  CMD+=("--dry-run")
fi

echo "======================================"
echo "ModelScope 上传参数"
echo "======================================"
echo "repo_id           : $REPO_ID"
echo "folder_path       : $FOLDER_PATH"
echo "repo_type         : $REPO_TYPE"
echo "incremental       : $INCREMENTAL"
echo "include_hidden    : $INCLUDE_HIDDEN"
echo "sqlite_snapshot   : $SQLITE_SNAPSHOT"
echo "dry_run           : $DRY_RUN"
echo "manifest_path     : ${MANIFEST_PATH:-<auto>}"
echo "commit_message    : $COMMIT_MESSAGE"
echo "python            : $PYTHON_CMD"
echo "======================================"

"${CMD[@]}"

echo "完成。"
