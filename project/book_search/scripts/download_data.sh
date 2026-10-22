#!/bin/bash
# ============================================================================
# 下载 ModelScope 数据集脚本
# ============================================================================
# 用法:
#   chmod +x download_data.sh
#   ./download_data.sh
#
# 默认将数据保存到: project/book_search/data/modelscope
# 可通过参数覆盖: ./download_data.sh --dataset <id> --file <path> --local_dir <dir>
# ============================================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_LOCAL_DIR="$ROOT_DIR/data/modelscope"

# 尝试激活虚拟环境（可选）
if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
  # Git Bash / WSL
  source "$ROOT_DIR/.venv/bin/activate"
elif [ -f "$ROOT_DIR/.venv/Scripts/activate" ]; then
  # mingw/cygwin
  source "$ROOT_DIR/.venv/Scripts/activate"
fi

DATASET="wzywuan/Novel-Collection"
FILE_PATH=""  # 默认空表示下载全量数据集
LOCAL_DIR="$DEFAULT_LOCAL_DIR"

# 解析参数（非常简单）
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"; shift 2;;
    --file)
      FILE_PATH="$2"; shift 2;;
    --local_dir)
      LOCAL_DIR="$2"; shift 2;;
    *)
      echo "Unknown arg: $1"; shift;;
  esac
done

mkdir -p "$LOCAL_DIR"

echo "Dataset: $DATASET"
if [ -z "$FILE_PATH" ]; then
  echo "File: (full dataset)"
else
  echo "File: $FILE_PATH"
fi
echo "Local dir: $LOCAL_DIR"

# 检查并选择 python 命令（优先 python3）
if command -v python3 &> /dev/null; then
  PYTHON_CMD=python3
elif command -v python &> /dev/null; then
  PYTHON_CMD=python
else
  echo "错误: 未找到 python，请先安装 Python3"
  exit 1
fi

if ! command -v modelscope &> /dev/null; then
  echo "modelscope CLI 未安装，正在尝试通过 $PYTHON_CMD -m pip 安装..."
  # 如果 pip 模块不可用，尝试通过 ensurepip 引导安装
  if ! $PYTHON_CMD -m pip --version > /dev/null 2>&1; then
    echo "pip 未就绪，尝试通过 ensurepip 安装..."
    $PYTHON_CMD -m ensurepip --upgrade || true
  fi
  $PYTHON_CMD -m pip install --upgrade pip || true
  $PYTHON_CMD -m pip install modelscope
fi

echo "开始下载（这可能需要一些时间）..."
if [ -z "$FILE_PATH" ]; then
  modelscope download --dataset "$DATASET" --local_dir "$LOCAL_DIR"
else
  modelscope download --dataset "$DATASET" "$FILE_PATH" --local_dir "$LOCAL_DIR"
fi

echo "下载完成，保存路径: $LOCAL_DIR"
