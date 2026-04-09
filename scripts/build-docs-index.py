#!/usr/bin/env python3
import os
import json
import sys
import locale


def build_tree(current_path, root_dir):
    name = os.path.basename(current_path)
    if os.path.isdir(current_path):
        children = []
        for item in os.listdir(current_path):
            if item in ("index.json", ".DS_Store"):
                continue
            child = build_tree(os.path.join(current_path, item), root_dir)
            if child is not None:
                children.append(child)

        # sort: folders first, then files; try locale-aware compare for zh
        try:
            locale.setlocale(locale.LC_COLLATE, "zh_CN.UTF-8")
            keyfunc = lambda x: (
                0 if x["type"] == "folder" else 1,
                locale.strxfrm(x["name"]),
            )
        except Exception:
            keyfunc = lambda x: (0 if x["type"] == "folder" else 1, x["name"])

        children.sort(key=keyfunc)
        return {"name": name, "type": "folder", "children": children}

    if os.path.isfile(current_path):
        rel = os.path.relpath(current_path, root_dir).replace(os.path.sep, "/")
        return {"name": name, "type": "file", "path": rel}

    return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    docs_root = os.path.join(repo_root, "docs")
    output_file = os.path.join(docs_root, "index.json")

    if not os.path.exists(docs_root):
        print("错误：docs 目录不存在。", file=sys.stderr)
        sys.exit(1)

    tree = build_tree(docs_root, repo_root)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)

    print(f"已生成 {output_file}")


if __name__ == "__main__":
    main()
