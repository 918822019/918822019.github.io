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
    model.eval()
    prompt_ids = prompt_ids.to(device)

    outputs = model(
        input_ids=prompt_ids,
        output_router_logits=True,
        use_cache=False,
        return_dict=True,
    )

    all_router = outputs.router_logits

    num_decoder_layers = len(all_router) - 1

    decoder_router = all_router[-2][0]
    mtp_router = all_router[-1][0]

    layer_routers = torch.stack(
        [all_router[i][0].squeeze(0) for i in range(num_decoder_layers)]
    )

    full_len = prompt_ids.shape[1]

    mtp_pred = mtp_router[:, :full_len - 1, :].squeeze(0)
    actual = decoder_router[:, 1:, :].squeeze(0)

    # LM head predicting next token: lm_logits[:, t, :] predicts token[t+1]
    lm_logits = outputs.logits[:, :-1, :].squeeze(0)           # [full_len-1, V]

    mtp_logits_raw = getattr(outputs, "mtp_logits", None)
    mtp_token_logits = None
    lm_logits_for_mtp = None  # LM at positions where it predicts same target as MTP
    if mtp_logits_raw is not None and len(mtp_logits_raw) > 0:
        # MTP head: mtp_logits[0][:, t, :] predicts token[t+2]
        # (input_ids is shifted by -1 internally, so MTP at t sees tok[t+1]'s embedding)
        mtp_token_logits = mtp_logits_raw[0][:, :-1, :].squeeze(0)  # [full_len-1, V]
        # LM at position t+1 also predicts token[t+2], align by target token
        lm_logits_for_mtp = outputs.logits[:, 1:, :].squeeze(0)     # [full_len-1, V]

    return {
        "output_ids": prompt_ids,
        "num_decoder_layers": num_decoder_layers,
        "decoder_router": actual,
        "mtp_router": mtp_pred,
        "lm_logits": lm_logits,
        "mtp_token_logits": mtp_token_logits,
        "lm_logits_for_mtp": lm_logits_for_mtp,
        "layer_routers": layer_routers,
    }
