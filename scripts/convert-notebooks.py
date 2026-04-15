#!/usr/bin/env python3
"""
在 publish 前将 docs/ 目录下所有 .ipynb 转换为同目录同名 .md 文件。
依赖: jupyter nbconvert (pip install nbconvert)

规则:
  - 若 .md 已存在且比 .ipynb 新，跳过（增量转换）
  - 转换成功后不删除 .ipynb
  - 存在失败时以非零退出码退出，让调用方决定是否中止推送
"""
import subprocess
import sys
import os
from pathlib import Path


def parse_cells(nb_path: Path) -> bool:
    """快速判断 .ipynb 是否有真实内容（非空笔记本）。"""
    try:
        import json

        with open(nb_path, encoding="utf-8") as f:
            nb = json.load(f)
        cells = nb.get("cells", [])
        return any(c.get("source") for c in cells)
    except Exception:
        return True  # 无法判断时默认尝试转换


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    docs_root = repo_root / "docs"

    if not docs_root.exists():
        print("错误：docs 目录不存在", file=sys.stderr)
        sys.exit(1)

    notebooks = sorted(docs_root.rglob("*.ipynb"))

    if not notebooks:
        print("未找到 .ipynb 文件，跳过。")
        return

    # Pre-check: nbconvert available?
    check = subprocess.run(
        [sys.executable, "-m", "nbconvert", "--version"],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        print(
            "错误：未找到 nbconvert，请先安装：pip install nbconvert\n"
            "      跳过笔记本转换。",
            file=sys.stderr,
        )
        sys.exit(1)

    ok = skipped = failed = 0

    for nb in notebooks:
        md_path = nb.with_suffix(".md")

        # 增量：已存在的 .md 比 .ipynb 新则跳过
        if md_path.exists() and md_path.stat().st_mtime >= nb.stat().st_mtime:
            print(f"  跳过 (已是最新): {nb.relative_to(repo_root)}")
            skipped += 1
            continue

        # 空笔记本跳过
        if not parse_cells(nb):
            print(f"  跳过 (空笔记本): {nb.relative_to(repo_root)}")
            skipped += 1
            continue

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "nbconvert",
                    "--to",
                    "markdown",
                    "--output-dir",
                    str(nb.parent),
                    str(nb),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                print(f"  转换成功: {nb.relative_to(repo_root)} → {md_path.name}")
                ok += 1
            else:
                print(f"  转换失败: {nb.relative_to(repo_root)}", file=sys.stderr)
                if result.stderr.strip():
                    print(result.stderr.strip(), file=sys.stderr)
                failed += 1
        except FileNotFoundError:
            print(
                "错误：未找到 nbconvert，请先安装：pip install nbconvert",
                file=sys.stderr,
            )
            sys.exit(1)
        except subprocess.TimeoutExpired:
            print(f"  超时: {nb.relative_to(repo_root)}", file=sys.stderr)
            failed += 1

    print(f"\n转换完成：成功 {ok}，跳过 {skipped}，失败 {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
