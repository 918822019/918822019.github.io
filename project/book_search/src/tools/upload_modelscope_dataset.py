# =========================================
# ModelScope 数据集分片增量上传脚本 (已移动到 tools/)
#
# 用法示例：
#   # 先 dry-run 看会上传哪些文件
#   cd project/book_search
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
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path


DEFAULT_MANIFEST_NAME = ".modelscope-upload-manifest.json"
SQLITE_RUNTIME_SUFFIXES = ("-wal", "-shm", "-journal")


def build_examples_text() -> str:
    """
    生成详细中文示例，主要面向 Ubuntu/Bash 使用场景。
    """
    return """\
Ubuntu 常用示例：

1) 安装依赖并设置 token
    pip install -r requirements.txt
    export MODELSCOPE_API_TOKEN="你的_token"

2) 先 dry-run 检查本次会上传哪些文件（不实际上传）
    python tools/upload_modelscope_dataset.py \\
      --repo-id 你的用户名/数据集名 \\
      --folder-path data \\
      --dry-run

3) 正式全量上传 data 目录
    python tools/upload_modelscope_dataset.py \\
      --repo-id 你的用户名/数据集名 \\
      --folder-path data \\
      --commit-message "full data upload"

4) 对 data/shards 做增量上传（推荐）
    python tools/upload_modelscope_dataset.py \\
      --repo-id 你的用户名/数据集名 \\
      --folder-path data/shards \\
      --incremental \\
      --commit-message "incremental shard upload"

说明：
- 脚本默认会跳过 SQLite 运行时文件（*.db-wal / *.db-shm / *.db-journal）。
- 当检测到 .db 存在 WAL/SHM 时，默认会自动创建一致性快照再上传，避免上传到不完整数据库。"""


def build_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器。
    支持指定 repo-id、上传目录、提交信息、token、增量上传等参数。
    """
    parser = argparse.ArgumentParser(
        description=(
            "上传本地数据目录到 ModelScope 仓库（默认 dataset）。"
            "已针对 SQLite 数据库上传做安全优化。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=build_examples_text(),
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
    parser.add_argument(
        "--sqlite-snapshot",
        default="auto",
        choices=["auto", "always", "never"],
        help=(
            "SQLite .db 上传模式："
            "auto=检测到 WAL/SHM 时自动快照（默认）；"
            "always=所有 .db 都先快照；"
            "never=直接上传原始文件。"
        ),
    )
    return parser


# 使用共享工具函数
from src.tools.modelscope_utils import resolve_token, login_modelscope as login  # noqa: E402


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


def is_sqlite_runtime_file(path: Path) -> bool:
    """
    判断是否是 SQLite 运行时产物文件（不建议上传）。
    """
    return path.name.endswith(SQLITE_RUNTIME_SUFFIXES)


def should_skip_file(path: Path, folder: Path, include_hidden: bool) -> bool:
    """
    统一的过滤规则。
    """
    path_parts = path.relative_to(folder).parts
    if not include_hidden and any(part.startswith(".") for part in path_parts):
        return True
    if path.suffix == ".tmp":
        return True
    if is_sqlite_runtime_file(path):
        return True
    return False


def compute_effective_file_metadata(file_path: Path) -> dict[str, int | str]:
    """
    计算用于增量比较的元数据。

    对 .db 文件，会把同名 -wal/-shm 的内容纳入哈希计算，避免主库文件不变但 WAL 变化时漏传。
    """
    if file_path.suffix != ".db":
        stat = file_path.stat()
        return {
            "sha256": compute_file_sha256(file_path),
            "size": stat.st_size,
        }

    digest = hashlib.sha256()
    total_size = 0
    db_hash = compute_file_sha256(file_path)
    db_size = file_path.stat().st_size
    digest.update(f"db:{db_hash}".encode("utf-8"))
    total_size += db_size

    for suffix in ("-wal", "-shm"):
        sidecar = file_path.with_name(file_path.name + suffix)
        if sidecar.exists() and sidecar.is_file():
            sidecar_hash = compute_file_sha256(sidecar)
            sidecar_size = sidecar.stat().st_size
            digest.update(f"{suffix}:{sidecar_hash}".encode("utf-8"))
            total_size += sidecar_size

    return {
        "sha256": digest.hexdigest(),
        "size": total_size,
    }


def has_sqlite_sidecars(db_path: Path) -> bool:
    """
    判断 .db 是否存在 WAL/SHM 等伴随文件。
    """
    return any(
        (db_path.with_name(db_path.name + suffix)).exists()
        for suffix in ("-wal", "-shm")
    )


def create_sqlite_snapshot(db_path: Path) -> Path:
    """
    使用 sqlite backup API 创建一致性快照文件。
    """
    temp_file = tempfile.NamedTemporaryFile(
        prefix="sqlite_snapshot_",
        suffix=".db",
        delete=False,
    )
    temp_file.close()
    snapshot_path = Path(temp_file.name)

    source_conn = sqlite3.connect(str(db_path))
    target_conn = sqlite3.connect(str(snapshot_path))
    try:
        source_conn.backup(target_conn)
        target_conn.commit()
    finally:
        target_conn.close()
        source_conn.close()

    return snapshot_path


def prepare_upload_source(
    file_path: Path,
    sqlite_snapshot_mode: str,
) -> tuple[Path, bool]:
    """
    准备实际上传的本地文件。
    返回 (path, is_temp_snapshot)。
    """
    if file_path.suffix != ".db":
        return file_path, False

    if sqlite_snapshot_mode == "never":
        return file_path, False

    if sqlite_snapshot_mode == "always":
        return create_sqlite_snapshot(file_path), True

    if has_sqlite_sidecars(file_path):
        return create_sqlite_snapshot(file_path), True

    return file_path, False


def collect_folder_state(folder: Path) -> dict[str, dict[str, int | str]]:
    """
    遍历目录下所有非隐藏、非 .tmp 文件，收集 sha256 和大小。
    返回 {相对路径: {sha256, size}}
    """
    state: dict[str, dict[str, int | str]] = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if should_skip_file(path=path, folder=folder, include_hidden=False):
            continue
        relative_path = path.relative_to(folder).as_posix()
        state[relative_path] = compute_effective_file_metadata(path)
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
        if should_skip_file(
            path=path,
            folder=folder,
            include_hidden=include_hidden,
        ):
            continue
        relative_path = path.relative_to(folder).as_posix()
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
    sqlite_snapshot_mode: str,
    dry_run: bool,
) -> None:
    """
    执行增量上传：只上传有变化的文件，上传后保存 manifest。
    """
    if include_hidden:
        current_files: dict[str, dict[str, int | str]] = {
            relative_path: compute_effective_file_metadata(
                folder / relative_path
            )
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
        "sqlite_snapshot_mode": sqlite_snapshot_mode,
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
        source_path, is_temp_snapshot = prepare_upload_source(
            local_path,
            sqlite_snapshot_mode,
        )
        try:
            upload_file(
                repo_id=repo_id,
                path_or_fileobj=str(source_path),
                path_in_repo=relative_path,
                commit_message=f"{commit_message}: {relative_path}",
                repo_type=repo_type,
                token=token,
            )
        finally:
            if is_temp_snapshot and source_path.exists():
                source_path.unlink()

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
    sqlite_snapshot_mode: str,
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
        "sqlite_snapshot_mode": sqlite_snapshot_mode,
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
        source_path, is_temp_snapshot = prepare_upload_source(
            local_path,
            sqlite_snapshot_mode,
        )
        try:
            upload_file(
                repo_id=repo_id,
                path_or_fileobj=str(source_path),
                path_in_repo=relative_path,
                commit_message=f"{commit_message}: {relative_path}",
                repo_type=repo_type,
                token=token,
            )
        finally:
            if is_temp_snapshot and source_path.exists():
                source_path.unlink()

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
        from modelscope.hub.api import HubApi
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
            sqlite_snapshot_mode=args.sqlite_snapshot,
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
            sqlite_snapshot_mode=args.sqlite_snapshot,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
