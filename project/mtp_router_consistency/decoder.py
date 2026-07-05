"""MTP data extraction from a single forward pass.

Extracts router logits, LM head logits, and MTP head logits
from the model output. Aligns positions by target token.

Key alignment:
  - LM head:     lm_logits[:, t, :]     predicts token[t+1]
  - MTP head:    mtp_logits[:, t, :]    predicts token[t+2]
  - LM aligned:  lm_logits[:, 1:, :]    predicts token[t+2]  (same as MTP)
  - Decoder router at t+1  matches  MTP router at t  (both handle token[t+1])
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def process_with_mtp(
    model,
    tokenizer,
    prompt_ids: torch.LongTensor,
    device: str = "cpu",
) -> dict:
    """Single forward pass with MTP and router logit extraction.

    Args:
        model: BailingMoeV2ForCausalLM (loaded on CPU)
        tokenizer: Corresponding tokenizer
        prompt_ids: Tokenized input [1, seq_len]
        device: Target device

    Returns:
        dict with: output_ids, decoder_router, mtp_router, lm_logits,
                   mtp_token_logits, lm_logits_for_mtp, layer_routers,
                   ground_truth_lm, ground_truth_mtp
    """
    model.eval()
    prompt_ids = prompt_ids.to(device)

    outputs = model(
        input_ids=prompt_ids,
        output_router_logits=True,
        use_cache=False,
        return_dict=True,
    )

    # all_router is tuple of (router_logits, topk_idx) per MoE layer + MTP
    all_router = outputs.router_logits
    num_decoder_layers = len(all_router) - 1  # exclude MTP

    # Last decoder layer router vs MTP router
    decoder_router = all_router[-2][0]  # [1, seq_len, E]
    mtp_router = all_router[-1][0]      # [1, seq_len, E]

    # All decoder layers' routers stacked for per-layer analysis
    layer_routers = torch.stack(
        [all_router[i][0].squeeze(0) for i in range(num_decoder_layers)]
    )  # [num_decoder_layers, seq_len, E]

    full_len = prompt_ids.shape[1]

    # Align: MTP at position t predicts routing for token[t+1]
    mtp_pred = mtp_router[:, :full_len - 1, :].squeeze(0)
    actual = decoder_router[:, 1:, :].squeeze(0)

    # LM head predicting next token: lm_logits[:, t, :] predicts token[t+1]
    lm_logits = outputs.logits[:, :-1, :].squeeze(0)

    # MTP head predicting token[t+2] (due to internal input_ids roll)
    mtp_logits_raw = getattr(outputs, "mtp_logits", None)
    mtp_token_logits = None
    lm_logits_for_mtp = None
    if mtp_logits_raw is not None and len(mtp_logits_raw) > 0:
        mtp_token_logits = mtp_logits_raw[0][:, :-1, :].squeeze(0)
        lm_logits_for_mtp = outputs.logits[:, 1:, :].squeeze(0)

    # Ground truth tokens for accuracy verification
    ground_truth_lm = prompt_ids[:, 1:].squeeze(0)
    ground_truth_mtp = prompt_ids[:, 2:].squeeze(0) if prompt_ids.shape[1] > 2 else None

    return {
        "output_ids": prompt_ids,
        "num_decoder_layers": num_decoder_layers,
        "decoder_router": actual,
        "mtp_router": mtp_pred,
        "lm_logits": lm_logits,
        "mtp_token_logits": mtp_token_logits,
        "lm_logits_for_mtp": lm_logits_for_mtp,
        "layer_routers": layer_routers,
        "ground_truth_lm": ground_truth_lm,
        "ground_truth_mtp": ground_truth_mtp,
    }
