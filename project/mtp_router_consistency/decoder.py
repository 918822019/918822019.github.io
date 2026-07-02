from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def process_with_mtp(
    model,
    tokenizer,
    prompt_ids: torch.LongTensor,
    max_new_tokens: int,
    device: str = "cpu",
) -> dict:
    model.eval()
    seq_len = prompt_ids.shape[1]
    prompt_ids = prompt_ids.to(device)

    # manual generation loop (no KV cache — full sequence each step to avoid accelerate hook issues)
    generated_tokens = []
    cur_ids = prompt_ids
    for step in range(max_new_tokens):
        outputs = model(input_ids=cur_ids, use_cache=False, return_dict=True)
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        generated_tokens.append(next_token.item())
        cur_ids = torch.cat([cur_ids, next_token], dim=1)

    full_ids = torch.cat([prompt_ids, torch.tensor(generated_tokens, device=device).unsqueeze(0)], dim=1)

    outputs = model(
        input_ids=full_ids,
        output_router_logits=True,
        use_cache=False,
        return_dict=True,
    )

    all_router = outputs.router_logits

    decoder_router_tup = all_router[-2]
    mtp_router_tup = all_router[-1]

    decoder_router = decoder_router_tup[0]
    mtp_router = mtp_router_tup[0]

    full_len = full_ids.shape[1]

    mtp_pred = mtp_router[:, :full_len - 1, :]
    actual = decoder_router[:, 1:, :]

    return {
        "output_ids": full_ids,
        "generated": generated_tokens,
        "actual_logits": list(actual.squeeze(0).unbind(0)),
        "mtp_pred_logits": list(mtp_pred.squeeze(0).unbind(0)),
    }
