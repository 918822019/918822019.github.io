"""Model loading utilities for the MTP Router Consistency Analysis.

Handles path resolution, model loading with CPU offload,
and tokenizer setup for BailingMoeV2 models.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


def _resolve_path(model_path: str) -> str:
    p = Path(model_path)
    if not p.is_absolute():
        root = Path(__file__).resolve().parents[2]
        resolved = root / p
        if resolved.exists():
            return str(resolved)
    return model_path


def load_model_and_tokenizer(
    model_path: str,
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Load model and tokenizer with CPU offload.

    Uses device_map='cpu' to force full CPU loading (stable for 6GB VRAM).
    The model has ~12B params, 256 experts per layer, top-8 routing.

    Returns:
        tuple: (model, tokenizer, model_device)
    """
    model_path = _resolve_path(model_path)
    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading model from %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    model_device = model.model.word_embeddings.weight.device
    logger.info("Model device: %s", model_device)
    return model, tokenizer, model_device
