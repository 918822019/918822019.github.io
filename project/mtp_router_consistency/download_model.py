"""在后台下载 Mellum2 模型权重。

用法:
  py download_model.py --model Mellum2-12B-A2.5B-Instruct

模型变体 (全部含 MTP head 权重):
  - Mellum2-12B-A2.5B-Instruct      (RL-tuned instruction model) ← 推荐
  - Mellum2-12B-A2.5B-Base-Pretrain (pretrain checkpoint)
  - Mellum2-12B-A2.5B-Base          (final base model)

ModelScope 镜像:
  JetBrains/Mellum2-12B-A2.5B-Instruct
  JetBrains/Mellum2-12B-A2.5B-Base-Pretrain
  JetBrains/Mellum2-12B-A2.5B-Base
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELSCOPE_EXE = r"C:\Users\wzy123456\AppData\Local\Programs\Python\Python312\Scripts\modelscope.exe"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "models"


def download_model(model_name: str):
    local_dir = DATA_DIR / model_name
    local_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s to %s", model_name, local_dir)
    logger.info("Note: model is ~23GB, this may take 1-2 hours.")

    cmd = [
        MODELSCOPE_EXE,
        "download",
        "--model", f"JetBrains/{model_name}",
        "--local_dir", str(local_dir),
    ]

    try:
        subprocess.run(cmd, check=True)
        logger.info("Download complete: %s", model_name)
    except subprocess.CalledProcessError as e:
        logger.error("Download failed with exit code %d", e.returncode)
        sys.exit(1)
    except FileNotFoundError:
        logger.error("modelscope CLI not found at %s", MODELSCOPE_EXE)
        logger.info("Try: pip install modelscope")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download Mellum2 model from ModelScope")
    parser.add_argument(
        "--model",
        default="Mellum2-12B-A2.5B-Instruct",
        choices=[
            "Mellum2-12B-A2.5B-Instruct",
            "Mellum2-12B-A2.5B-Base-Pretrain",
            "Mellum2-12B-A2.5B-Base",
        ],
        help="Model variant to download",
    )
    args = parser.parse_args()
    download_model(args.model)


if __name__ == "__main__":
    main()
