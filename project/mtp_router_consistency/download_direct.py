"""Download Mellum2-12B-A2.5B-Instruct safetensors via direct HTTP with resume."""
import json, os, sys, time
from pathlib import Path
import requests

MODEL = "JetBrains/Mellum2-12B-A2.5B-Instruct"
LOCAL_DIR = Path(__file__).resolve().parents[2] / "data" / "models" / MODEL.split("/")[-1]
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# File list from ModelScope
BASE = f"https://modelscope.cn/api/v1/models/{MODEL}/repo?Revision=master&FilePath"

files = [
    "model-00001-of-00005.safetensors",
    "model-00002-of-00005.safetensors",
    "model-00003-of-00005.safetensors",
    "model-00004-of-00005.safetensors",
    "model-00005-of-00005.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
]

for fname in files:
    dest = LOCAL_DIR / fname
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [OK] {fname} ({dest.stat().st_size / 1e9:.2f}GB)")
        continue

    url = f"{BASE}={fname}"
    print(f"\nDownloading {fname}...")

    headers = {}
    mode = "wb"
    if dest.exists():
        existing = dest.stat().st_size
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"  Resuming from {existing / 1e9:.2f}GB")

    resp = requests.get(url, headers=headers, stream=True, timeout=30)
    if resp.status_code >= 400:
        print(f"  ERROR {resp.status_code}")
        continue

    total = int(resp.headers.get("Content-Length", 0)) + (dest.stat().st_size if dest.exists() else 0)
    downloaded = dest.stat().st_size if dest.exists() else 0

    with open(dest, mode) as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                pct = downloaded / total * 100 if total else 0
                print(f"\r  {downloaded / 1e9:.2f}/{total / 1e9:.2f} GB ({pct:.1f}%)", end="")

    print(f"\n  Done: {fname}")

print("\nAll files downloaded!")
