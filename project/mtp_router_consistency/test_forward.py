from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_pkg_dir = Path(__file__).parent.resolve()
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

from config import Config

cfg = Config()
model_path = Path(__file__).resolve().parents[2] / cfg.model_path
model_path = str(model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

logger.info("Loading model (CPU, no offload)...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
    attn_implementation="eager",
    low_cpu_mem_usage=True,
)
model.eval()
logger.info("Model loaded on CPU")

prompt = "def fibonacci(n):"
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=16)
input_ids = inputs.input_ids
print(f"Input shape: {input_ids.shape}", flush=True)

with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        output_router_logits=True,
        use_cache=False,
        return_dict=True,
    )
print("Forward pass succeeded", flush=True)
print("Has mtp_logits:", hasattr(outputs, "mtp_logits"), flush=True)
if hasattr(outputs, "mtp_logits") and outputs.mtp_logits is not None:
    print("mtp_logits[0] shape:", outputs.mtp_logits[0].shape, flush=True)

    mtp_tokens = outputs.mtp_logits[0][:, :-1, :].argmax(dim=-1)
    lm_tokens = outputs.logits[:, :-1, :].argmax(dim=-1)
    acc = (mtp_tokens == lm_tokens).float().mean().item()
    print(f"Token accuracy: {acc:.4f}", flush=True)
    for t in range(mtp_tokens.shape[1]):
        print(
            f"  pos {t}: LM={lm_tokens[0,t].item()} MTP={mtp_tokens[0,t].item()} {'✓' if lm_tokens[0,t]==mtp_tokens[0,t] else '✗'}",
            flush=True,
        )
