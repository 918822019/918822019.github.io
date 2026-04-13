"""
用法说明：
====================
本脚本用于从 ModelScope 下载 dataset/model 仓库内容到本地目录，支持断点续传、文件过滤、token 自动读取。

【常用命令】
1. 下载全部内容到默认目录：
    python tools/download_modelscope_dataset.py --repo-id wzywuan/Novel-Collection

2. 只下载分片和索引到 data/shards：
    python tools/download_modelscope_dataset.py \
      --repo-id wzywuan/Novel-Collection \
      --allow-pattern '*.db' \
      --allow-pattern 'index.json' \
      --output-dir data/shards

3. 下载前清空目标目录（谨慎）：
    python tools/download_modelscope_dataset.py \
      --repo-id wzywuan/Novel-Collection \
      --output-dir data/shards \
      --clean-output

【参数说明】
- --repo-id        ModelScope 仓库 ID，必填
- --output-dir     下载到的本地目录，默认 data/modelscope_download
- --allow-pattern  允许下载的文件模式，可多次传入
- --ignore-pattern 忽略下载的文件模式，可多次传入
- --token          访问令牌，默认自动读取环境变量
- --revision       下载分支/版本，默认 master
- --clean-output   下载前清空 output-dir（谨慎使用）

【依赖环境】
- pip install -r requirements.txt
- 需配置 MODELSCOPE_API_TOKEN 或 MODELSCOPE_TOKEN

【更多说明】
下载后可直接继续执行 crawl-content 或 export-shards --only-changed。
====================
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 ModelScope 下载 dataset/model 仓库内容到本地目录。"
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="ModelScope 仓库 ID，例如 owner_name/dataset_name",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            Path(__file__).resolve().parent.parent / "data" / "modelscope_download"
        ),
        help="下载后同步到的本地目录",
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
        choices=["dataset", "model"],
        help="仓库类型，默认 dataset",
    )
    parser.add_argument(
        "--revision",
        default="master",
        help="要下载的分支/版本，默认 master",
    )
    parser.add_argument(
        "--token",
        default="",
        help=(
            "ModelScope 访问令牌。"
            "如未传入则从 MODELSCOPE_API_TOKEN 或 MODELSCOPE_TOKEN 读取"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=str(Path(__file__).resolve().parent.parent / ".modelscope-cache"),
        help="ModelScope 缓存目录",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="允许下载的文件模式，可多次传入，如 --allow-pattern '*.db'",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=[],
        help="忽略下载的文件模式，可多次传入",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="下载前清空 output-dir（谨慎使用）",
    )
    return parser


def resolve_token(cli_token: str) -> str:
    if cli_token.strip():
        return cli_token.strip()

    for env_name in ("MODELSCOPE_API_TOKEN", "MODELSCOPE_TOKEN"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value

    return ""


def login_if_needed(token: str) -> None:
    if not token:
        return

    from modelscope.hub.api import HubApi  # type: ignore[import-untyped]

    api = HubApi()
    login_func = getattr(api, "login")
    try:
        login_func(token)
    except TypeError:
        login_func(access_token=token)


def snapshot_download_with_fallback(
    repo_id: str,
    repo_type: str,
    revision: str,
    cache_dir: str,
    allow_patterns: list[str],
    ignore_patterns: list[str],
) -> str:
    snapshot_download = importlib.import_module(
        "modelscope.hub.snapshot_download"
    ).snapshot_download

    base_kwargs = {
        "model_id": repo_id,
        "revision": revision,
        "cache_dir": cache_dir,
    }

    # 某些版本参数名是 repo_type，某些版本没有该参数。
    kwargs_candidates: list[dict[str, object]] = [
        {
            **base_kwargs,
            "repo_type": repo_type,
            "allow_file_pattern": allow_patterns or None,
            "ignore_file_pattern": ignore_patterns or None,
        },
        {
            **base_kwargs,
            "repo_type": repo_type,
            "allow_patterns": allow_patterns or None,
            "ignore_patterns": ignore_patterns or None,
        },
        {
            **base_kwargs,
            "repo_type": repo_type,
        },
        {
            **base_kwargs,
            "allow_file_pattern": allow_patterns or None,
            "ignore_file_pattern": ignore_patterns or None,
        },
        {
            **base_kwargs,
        },
    ]

    last_error: Exception | None = None
    for kwargs in kwargs_candidates:
        cleaned = {k: v for k, v in kwargs.items() if v is not None}
        try:
            return str(snapshot_download(**cleaned))
        except TypeError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise RuntimeError(
            "当前 modelscope 版本与下载参数不兼容，请升级 modelscope 后重试。"
        ) from last_error

    raise RuntimeError("调用 snapshot_download 失败。")


def copy_tree(src: Path, dst: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0

    for path in src.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(src)
        target = dst / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        stat = path.stat()
        file_count += 1
        total_bytes += stat.st_size

    return file_count, total_bytes


def main() -> None:
    args = build_parser().parse_args()

    try:
        import modelscope  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "未安装 modelscope。请先执行: pip install -r requirements.txt"
        ) from exc

    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()

    token = resolve_token(args.token)
    login_if_needed(token)

    cache_snapshot_dir = Path(
        snapshot_download_with_fallback(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            cache_dir=str(cache_dir),
            allow_patterns=args.allow_pattern,
            ignore_patterns=args.ignore_pattern,
        )
    ).resolve()

    if not cache_snapshot_dir.exists():
        raise FileNotFoundError(f"下载结果目录不存在: {cache_snapshot_dir}")

    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    file_count, total_bytes = copy_tree(cache_snapshot_dir, output_dir)

    print("下载完成")
    print(f"repo_id={args.repo_id}")
    print(f"revision={args.revision}")
    print(f"cache_snapshot={cache_snapshot_dir}")
    print(f"output_dir={output_dir}")
    print(f"files={file_count}")
    print(f"bytes={total_bytes}")


if __name__ == "__main__":
    main()
