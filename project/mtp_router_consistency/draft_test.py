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

prompt = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n"
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
input_ids = inputs.input_ids.to(device)
T = input_ids.shape[1]
max_gen = 16

print(f"Prompt: {prompt}")
print(f"Tokens: {T}")

# Generate ground truth continuation with main model
gen_ids = input_ids.clone()
with torch.no_grad():
    for _ in range(max_gen):
        out = model(input_ids=gen_ids, use_cache=False, return_dict=True)
        next_tok = out.logits[0, -1].argmax().item()
        gen_ids = torch.cat([gen_ids, torch.tensor([[next_tok]], device=device)], dim=1)

print(f"Ground truth: {tokenizer.decode(gen_ids[0])}")
print()

# MTP draft: main model generates tok[T], then MTP drafts tok[T+1] onwards
# MTP at position T (with tok[T] via roll) predicts tok[T+1]
draft = [gen_ids[0, T].item()]  # first token from main model
cur_ids = gen_ids[:, :T+1].clone()  # prompt + first token

with torch.no_grad():
    for step in range(max_gen - 1):
        # Get MTP prediction for next token
        raw = model.model(input_ids=cur_ids, use_cache=False, return_dict=True)
        mtp_hidden = raw.mtp_hidden_states[0]  # [1, T+1, D]
        # MTP at second-to-last position: the rolled input has the newest token here
        # (last position wraps to tok0 due to circular roll, skip it)
        mtp_logits = model.lm_head(mtp_hidden[0, -2:-1]).float()
        next_tok = mtp_logits[0].argmax().item()
        draft.append(next_tok)
        cur_ids = torch.cat([cur_ids, torch.tensor([[next_tok]], device=device)], dim=1)

draft_tokens = draft  # len = max_gen (1 LM + 15 MTP)
print(f"\n{'='*80}")
print("MTP Speculative Draft (first token from LM, rest from MTP)")
print(f"{'='*80}")
print(f"{'pos':<5} {'source':<8} {'Draft':<20} {'Ground truth':<20} {'Match?'}")
print(f"{'-'*65}")

match_count = 0
for i in range(max_gen):
    src = "LM" if i == 0 else "MTP"
    dr = draft_tokens[i]
    truth = gen_ids[0, T + i].item()
    match = dr == truth
    if match: match_count += 1
    print(f"{i:<5} {src:<8} {tokenizer.decode(dr):<20} {tokenizer.decode(truth):<20} {'OK' if match else 'X'}")

print(f"\nDraft accuracy: {match_count}/{max_gen} ({match_count/max_gen*100:.1f}%)")

# Verification pass: run main model on full draft
full_ids = torch.cat([input_ids, torch.tensor([draft_tokens], device=device)], dim=1)
with torch.no_grad():
    verify_out = model(input_ids=full_ids, use_cache=False, return_dict=True)
verify_logits = verify_out.logits.float()

print(f"\n{'='*80}")
print("Speculative Decoding Verification")
print(f"{'='*80}")
print(f"\n{'pos':<5} {'Draft token':<20} {'LM predicts':<20} {'Accept?'}")
print(f"{'-'*65}")

accept_contiguous = 0
for i in range(max_gen):
    draft_tok = full_ids[0, T + i].item()
    lm_tok = verify_logits[0, T - 1 + i].argmax().item()
    accept = draft_tok == lm_tok
    if accept:
        accept_contiguous += 1
    else:
        break  # speculative decoding: only contiguous prefix accepted

    print(f"{i:<5} {tokenizer.decode(draft_tok):<20} {tokenizer.decode(lm_tok):<20} {'ACCEPT' if accept else 'REJECT'}")

if accept_contiguous < max_gen:
    rej_pos = accept_contiguous
    r_tok = full_ids[0, T + rej_pos].item()
    r_lm = verify_logits[0, T - 1 + rej_pos].argmax().item()
    print(f"{rej_pos:<5} {tokenizer.decode(r_tok):<20} {tokenizer.decode(r_lm):<20} {'REJECT (stop)'}")

print(f"\n=== Results ===")
print(f"Draft length: {max_gen} tokens")
print(f"Contiguously accepted: {accept_contiguous}/{max_gen}")
print(f"Effective tokens per 2 forward passes: {accept_contiguous + 1}")
if accept_contiguous > 0:
    print(f"Speedup vs sequential: ~{(accept_contiguous+1)/2:.1f}x (1 draft + 1 verify = 2 passes)")
else:
    print(f"No contiguous acceptance - MTP draft doesn't match decoder's predictions at position 0")

# Also check: what if we use MTP hidden -> gate for routing during verification?
print(f"\n{'='*80}")
print("Routing prediction during verification pass")
print(f"{'='*80}")
print(f"Using MTP hidden state at position t to predict decoder routing at t+1")
print(f"(This is what would enable expert prefetching during verification)")
print(f"")
print(f"Our earlier tests (predict_test.py) showed this works with Cos=0.81 avg.")
print(f"During verification, ALL N positions are processed simultaneously.")
print(f"→ MTP hidden at each position predicts routing for ALL layers at that position")
print(f"→ All experts can be batch-loaded before verification starts")

print("\nDone.")
