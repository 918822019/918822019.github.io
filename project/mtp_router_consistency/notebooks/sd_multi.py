from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = ""
_pkg_dir = Path(__file__).parent.resolve()
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

from model_utils import load_model_and_tokenizer
from config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

cfg = Config()
model, tokenizer, device = load_model_and_tokenizer(
    cfg.model_path, device=cfg.device,
    torch_dtype=getattr(torch, cfg.torch_dtype, torch.bfloat16),
)

prompts = {
    "code_fib": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n",
    "english": "The quick brown fox jumps over the lazy dog, but the dog was",
    "code_sort": "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr)",
}
max_gen = 4

print(f"\n{'='*80}")
print("Spec Decoding: MTP draft across prompt types")
print(f"{'='*80}")
print(f"{'Prompt':<20} {'Toks':<6} {'Draft/GT':<14} {'Accept':<10} {'Spd':<6} {'Draft tokens accepted'}")
print(f"{'------':<20} {'----':<6} {'--------':<14} {'------':<10} {'---':<6} {'---'}")

for pname, prompt in prompts.items():
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
    input_ids = inputs.input_ids.to(device)
    T = input_ids.shape[1]

    # Ground truth
    gen_ids = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_gen):
            out = model(input_ids=gen_ids, use_cache=False, return_dict=True)
            next_tok = out.logits[0, -1].argmax().item()
            gen_ids = torch.cat([gen_ids, torch.tensor([[next_tok]], device=device)], dim=1)

    # MTP draft: LM first token, then MTP autoregressively
    draft = [gen_ids[0, T].item()]
    cur_ids = gen_ids[:, :T+1].clone()
    with torch.no_grad():
        for step in range(max_gen - 1):
            raw = model.model(input_ids=cur_ids, use_cache=False, return_dict=True)
            mtp = raw.mtp_hidden_states[0]
            logits = model.lm_head(mtp[0, -2:-1]).float()
            ntok = logits[0].argmax().item()
            draft.append(ntok)
            cur_ids = torch.cat([cur_ids, torch.tensor([[ntok]], device=device)], dim=1)

    # Verification
    full = torch.cat([input_ids, torch.tensor([draft], device=device)], dim=1)
    with torch.no_grad():
        vout = model(input_ids=full, use_cache=False, return_dict=True)
    vlogits = vout.logits.float()

    accept = 0
    for i in range(max_gen):
        if full[0, T+i].item() == vlogits[0, T-1+i].argmax().item():
            accept += 1
        else:
            break

    dm = sum(1 for i in range(max_gen) if draft[i] == gen_ids[0, T+i].item())
    sp = (accept + 1) / 2

    # Show accepted tokens
    acc_toks = []
    for i in range(max_gen):
        dt = full[0, T+i].item()
        vt = vlogits[0, T-1+i].argmax().item()
        if dt == vt:
            acc_toks.append(tokenizer.decode(dt).strip() or '\\n')
        else:
            break

    acc_str = ' '.join(acc_toks) if acc_toks else '(none)'
    print(f"{pname:<20} {T:<6} {dm}/{max_gen} ({dm/max_gen*100:.0f}%){accept:<5}/{max_gen}{sp:<6.1f}x {acc_str[:50]}")

print(f"\nDone.")
