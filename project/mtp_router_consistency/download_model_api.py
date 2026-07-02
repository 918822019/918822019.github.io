"""Resumable model download via ModelScope Python API."""
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
MODEL_NAME = "Mellum2-12B-A2.5B-Instruct"
MODEL_ID = f"JetBrains/{MODEL_NAME}"
LOCAL_DIR = DATA_DIR / MODEL_NAME

try:
    from modelscope.hub.snapshot_download import snapshot_download
except ImportError:
    print("Installing modelscope...")
    os.system(f"{sys.executable} -m pip install modelscope -q")
    from modelscope.hub.snapshot_download import snapshot_download

print(f"Downloading {MODEL_ID} to {LOCAL_DIR}")
print("This may take a while. Model is ~23GB.")
sys.stdout.flush()

snapshot_download(
    model_id=MODEL_ID,
    local_dir=str(LOCAL_DIR),
)

print("Download complete!")
