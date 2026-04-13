# =========================================
# ModelScope 数据集分片增量上传脚本 (已移动到 tools/)
#
# 用法示例：
#   # 先 dry-run 看会上传哪些文件
#   cd project/起点搜书
#   python tools/upload_modelscope_dataset.py \
#     --repo-id wzywuan/Novel-Collection \
#     --folder-path data/shards \
#     --incremental \
#     --dry-run
#
#   # 真正上传（只会上传有变化的分片）
#   python tools/upload_modelscope_dataset.py \
#     --repo-id wzywuan/Novel-Collection \
#     --folder-path data/shards \
#     --incremental \
#     --commit-message "shard update"
#
# 依赖：pip install -r requirements.txt
#
# - 支持断点续传，自动生成本地 manifest 追踪已上传内容
# - 只增量上传有变化的文件，不会重复上传未变分片
# - 不会自动删除远端已存在但本地已删的文件
# =========================================
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


DEFAULT_MANIFEST_NAME = ".modelscope-upload-manifest.json"


def build_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器。
    支持指定 repo-id、上传目录、提交信息、token、增量上传等参数。
    """
    parser = argparse.ArgumentParser(
        description="上传本地数据目录到 ModelScope dataset 仓库。"
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="ModelScope dataset 仓库 ID，例如 owner_name/dataset_name",
    )
    parser.add_argument(
        "--folder-path",
        default=str(Path(__file__).resolve().parent.parent / "data"),
        help="要上传的本地目录，默认是项目根下的 data 目录",
    )
    parser.add_argument(
        "--commit-message",
        default="upload dataset folder to repo",
        help="上传时的提交信息",
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
        choices=["dataset", "model"],
        help="仓库类型，当前默认上传到 dataset 仓库",
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
        "--incremental",
        action="store_true",
        help="启用基于本地 manifest 的增量上传，只上传变化文件",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="上传时包含隐藏文件（例如 .shards.modelscope-upload-manifest.json）",
    )
    parser.add_argument(
        "--manifest-path",
        default="",
        help="增量上传 manifest 路径，默认写到待上传目录旁边",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查会上传哪些文件，不实际上传",
    )
    return parser


def resolve_token(cli_token: str) -> str:
    """
    优先用命令行传入的 token，否则读取环境变量 MODELSCOPE_API_TOKEN。
    """
    if cli_token.strip():
        return cli_token.strip()

    for env_name in ("MODELSCOPE_API_TOKEN", "MODELSCOPE_TOKEN"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return ""


def login(api: object, token: str) -> None:
    """
    登录 ModelScope，token 必填。
    """
    if not token:
        raise RuntimeError(
            "缺少 ModelScope token。"
            "请传 --token，或设置 MODELSCOPE_API_TOKEN 环境变量。"
        )

    login_func = getattr(api, "login")
    try:
        login_func(token)
    except TypeError:
        login_func(access_token=token)


def default_manifest_path(folder: Path) -> Path:
    """
    默认 manifest 路径：在上传目录旁边生成 .{目录名}.modelscope-upload-manifest.json
    """
    return folder.parent / f".{folder.name}{DEFAULT_MANIFEST_NAME}"


def compute_file_sha256(file_path: Path) -> str:
    """
    计算文件的 sha256 哈希值，用于判断内容是否变化。
    """
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def collect_folder_state(folder: Path) -> dict[str, dict[str, int | str]]:
    """
    遍历目录下所有非隐藏、非 .tmp 文件，收集 sha256 和大小。
    返回 {相对路径: {sha256, size}}
    """
    state: dict[str, dict[str, int | str]] = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(folder).as_posix()
        path_parts = path.relative_to(folder).parts
        if any(part.startswith(".") for part in path_parts):
            continue
        if path.suffix == ".tmp":
            continue
        stat = path.stat()
        state[relative_path] = {
            "sha256": compute_file_sha256(path),
            "size": stat.st_size,
        }
    return state


def collect_folder_files(
    folder: Path,
    include_hidden: bool,
) -> list[str]:
    """
    收集要上传的相对文件路径。
    - 默认跳过隐藏文件与 .tmp
    - include_hidden=True 时会包含隐藏文件
    """
    files: list[str] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(folder).as_posix()
        path_parts = path.relative_to(folder).parts
        if not include_hidden and any(part.startswith(".") for part in path_parts):
            continue
        if path.suffix == ".tmp":
            continue
        files.append(relative_path)
    return files


def load_manifest(path: Path) -> dict[str, dict[str, int | str]]:
    """
    加载本地 manifest 文件，返回 {相对路径: {sha256, size}}
    """
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    files = data.get("files")
    if not isinstance(files, dict):
        return {}
    return files


def save_manifest(
    path: Path,
    folder: Path,
    files: dict[str, dict[str, int | str]],
) -> None:
    """
    保存 manifest，记录本次上传后所有文件的 sha256 和大小。
    """
    payload = {
        "folder": str(folder),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "files": files,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def diff_folder_state(
    current_files: dict[str, dict[str, int | str]],
    manifest_files: dict[str, dict[str, int | str]],
) -> tuple[list[str], list[str]]:
    """
    比较当前目录和 manifest，返回变化/新增文件列表，以及本地已删文件列表。
    """
    changed: list[str] = []
    removed: list[str] = []

    for relative_path, metadata in current_files.items():
        if manifest_files.get(relative_path) != metadata:
            changed.append(relative_path)

    for relative_path in manifest_files:
        if relative_path not in current_files:
            removed.append(relative_path)

    return changed, removed


def incremental_upload(
    api: object,
    repo_id: str,
    repo_type: str,
    folder: Path,
    token: str,
    commit_message: str,
    manifest_path: Path,
    include_hidden: bool,
    dry_run: bool,
) -> None:
    """
    执行增量上传：只上传有变化的文件，上传后保存 manifest。
    """
    if include_hidden:
        current_files: dict[str, dict[str, int | str]] = {
            relative_path: {
                "sha256": compute_file_sha256(folder / relative_path),
                "size": (folder / relative_path).stat().st_size,
            }
            for relative_path in collect_folder_files(
                folder=folder,
                include_hidden=True,
            )
        }
    else:
        current_files = collect_folder_state(folder)
    manifest_files = load_manifest(manifest_path)
    changed_files, removed_files = diff_folder_state(
        current_files,
        manifest_files,
    )

    summary = {
        "folder": str(folder),
        "manifest_path": str(manifest_path),
        "changed_files": changed_files,
        "removed_files": removed_files,
        "changed_count": len(changed_files),
        "removed_count": len(removed_files),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if dry_run:
        return

    if removed_files:
        print(
            "提示: 检测到本地已删除但远端可能仍存在的文件，"
            "当前脚本不会自动删除远端文件。"
        )

    if not changed_files:
        print("没有检测到需要上传的变更文件。")
        return

    upload_file = getattr(api, "upload_file")
    for relative_path in changed_files:
        local_path = folder / relative_path
        upload_file(
            repo_id=repo_id,
            path_or_fileobj=str(local_path),
            path_in_repo=relative_path,
            commit_message=f"{commit_message}: {relative_path}",
            repo_type=repo_type,
            token=token,
        )

    save_manifest(manifest_path, folder, current_files)
    print(
        f"增量上传完成: repo_id={repo_id} folder_path={folder} "
        f"uploaded_files={len(changed_files)}"
    )


def full_upload(
    api: object,
    repo_id: str,
    repo_type: str,
    folder: Path,
    token: str,
    commit_message: str,
    include_hidden: bool,
    dry_run: bool,
) -> None:
    """
    执行全量上传：遍历目录中文件逐个上传。
    """
    files = collect_folder_files(
        folder=folder,
        include_hidden=include_hidden,
    )

    summary = {
        "folder": str(folder),
        "mode": "full",
        "include_hidden": include_hidden,
        "upload_files": files,
        "upload_count": len(files),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if dry_run:
        return

    if not files:
        print("目录中没有可上传文件。")
        return

    upload_file = getattr(api, "upload_file")
    for relative_path in files:
        local_path = folder / relative_path
        upload_file(
            repo_id=repo_id,
            path_or_fileobj=str(local_path),
            path_in_repo=relative_path,
            commit_message=f"{commit_message}: {relative_path}",
            repo_type=repo_type,
            token=token,
        )

    print(
        f"全量上传完成: repo_id={repo_id} folder_path={folder} "
        f"uploaded_files={len(files)}"
    )


def main() -> None:
    """
    命令行入口。自动登录、参数校验、执行增量上传。
    """
    parser = build_parser()
    args = parser.parse_args()

    folder = Path(args.folder_path).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"本地目录不存在: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"给定路径不是目录: {folder}")

    try:
        from modelscope.hub.api import HubApi  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "未安装 modelscope。请先执行: pip install modelscope"
        ) from exc

    token = resolve_token(args.token)
    api = HubApi()
    login(api, token)

    manifest_path = (
        Path(args.manifest_path).expanduser().resolve()
        if args.manifest_path
        else default_manifest_path(folder)
    )

    if args.incremental:
        incremental_upload(
            api=api,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder=folder,
            token=token,
            commit_message=args.commit_message,
            manifest_path=manifest_path,
            include_hidden=args.include_hidden,
            dry_run=args.dry_run,
        )
    else:
        full_upload(
            api=api,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder=folder,
            token=token,
            commit_message=args.commit_message,
            include_hidden=args.include_hidden,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
