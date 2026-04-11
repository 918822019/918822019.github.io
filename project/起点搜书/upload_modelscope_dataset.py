from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
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
        default=str(Path(__file__).resolve().parent / "data"),
        help="要上传的本地目录，默认是当前项目下的 data 目录",
    )
    parser.add_argument(
        "--sqlite-db-path",
        default=str(Path(__file__).resolve().parent / "data" / "books.db"),
        help=(
            "需要做一致性快照的 SQLite 数据库路径。"
            "默认指向 data/books.db；若不存在则跳过快照"
        ),
    )
    parser.add_argument(
        "--keep-snapshot",
        action="store_true",
        help="上传完成后保留临时快照目录，便于重复上传或人工检查",
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
    return parser


def resolve_token(cli_token: str) -> str:
    if cli_token.strip():
        return cli_token.strip()

    for env_name in ("MODELSCOPE_API_TOKEN", "MODELSCOPE_TOKEN"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return ""


def login(api: object, token: str) -> None:
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


def create_sqlite_snapshot(source_db: Path, target_db: Path) -> None:
    target_db.parent.mkdir(parents=True, exist_ok=True)
    src_conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    dst_conn = sqlite3.connect(target_db)
    try:
        src_conn.backup(dst_conn)
        dst_conn.commit()
    finally:
        dst_conn.close()
        src_conn.close()


def prepare_upload_folder(
    source_folder: Path,
    sqlite_db_path: Path,
) -> tuple[Path, bool]:
    if not sqlite_db_path.exists() or not sqlite_db_path.is_file():
        return source_folder, False

    temp_root = Path(tempfile.mkdtemp(prefix="modelscope_upload_"))
    upload_root = temp_root / source_folder.name

    ignore_names = {
        sqlite_db_path.name,
        f"{sqlite_db_path.name}-wal",
        f"{sqlite_db_path.name}-shm",
    }

    shutil.copytree(
        source_folder,
        upload_root,
        ignore=shutil.ignore_patterns(*sorted(ignore_names)),
    )
    create_sqlite_snapshot(
        sqlite_db_path,
        upload_root / sqlite_db_path.name,
    )
    return upload_root, True


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    folder = Path(args.folder_path).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"本地目录不存在: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"给定路径不是目录: {folder}")

    sqlite_db_path = Path(args.sqlite_db_path).expanduser().resolve()

    try:
        from modelscope.hub.api import HubApi  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "未安装 modelscope。请先执行: pip install modelscope"
        ) from exc

    token = resolve_token(args.token)
    api = HubApi()
    login(api, token)

    upload_folder, used_snapshot = prepare_upload_folder(
        source_folder=folder,
        sqlite_db_path=sqlite_db_path,
    )

    try:
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(upload_folder),
            commit_message=args.commit_message,
            repo_type=args.repo_type,
        )
    finally:
        if used_snapshot and not args.keep_snapshot:
            shutil.rmtree(upload_folder.parent, ignore_errors=True)

    print(
        f"上传完成: repo_id={args.repo_id} "
        f"folder_path={upload_folder} repo_type={args.repo_type} "
        f"used_snapshot={'1' if used_snapshot else '0'}"
    )


if __name__ == "__main__":
    main()
