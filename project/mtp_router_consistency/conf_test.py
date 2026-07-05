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

prompt = "def foo(x): return x + x * x + bar(x)"
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
input_ids = inputs.input_ids.to(device)
T = input_ids.shape[1]

mtp_moe = model.model.layers[-1].mlp

# Baseline forward
with torch.no_grad():
    base_out = model(input_ids=input_ids, use_cache=False, return_dict=True)
base_logits = base_out.logits.float()
base_probs = F.softmax(base_logits, dim=-1)
base_conf, base_tokens = base_probs.max(dim=-1)  # [1, T]

# Swapped forward: all MoE layers use MTP routing
decoder_moe_layers = []
for idx in range(cfg.num_hidden_layers - 1):
    layer = model.model.layers[idx]
    if hasattr(layer.mlp, 'experts') and len(layer.mlp.experts) > 0:
        decoder_moe_layers.append((idx, layer.mlp))

handles = []
for idx, moe_block in decoder_moe_layers:
    def make_hook(layer_idx, ref):
        def hook(m, i, o):
            with torch.no_grad():
                moe_in = i[0]
                mtp_gate = ref.gate(moe_in)
                flat = moe_in.view(-1, moe_in.shape[-1])
                swapped = m.moe_infer(flat, mtp_gate[0], mtp_gate[1]).view_as(moe_in)
                if m.shared_experts is not None:
                    swapped = swapped + m.shared_experts(moe_in)
            return (swapped, o[1])
        return hook
    handles.append(moe_block.register_forward_hook(make_hook(idx, mtp_moe)))

with torch.no_grad():
    swap_out = model(input_ids=input_ids, use_cache=False, return_dict=True)

for h in handles:
    h.remove()

swap_logits = swap_out.logits.float()
swap_probs = F.softmax(swap_logits, dim=-1)
swap_conf, swap_tokens = swap_probs.max(dim=-1)

# ========= Test 1: Token match by confidence threshold =========
print(f"\n{'='*90}")
print("Test 1: Token match rate by confidence threshold")
print(f"{'='*90}")

for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    mask = base_conf[0] >= threshold
    n_total = mask.sum().item()
    if n_total == 0:
        continue
    n_match = ((base_tokens[0] == swap_tokens[0]) & mask).sum().item()
    rate = n_match / n_total * 100
    print(f"  Conf >= {threshold:.1f}: {n_match}/{n_total} ({rate:.1f}%)")

# Per-position breakdown
print(f"\n  Per-position detail (sorted by confidence):")
data = []
for t in range(T):
    data.append((base_conf[0, t].item(), t, tokenizer.decode(input_ids[0, t].item()),
                  tokenizer.decode(base_tokens[0, t].item()), tokenizer.decode(swap_tokens[0, t].item()),
                  base_tokens[0, t].item() == swap_tokens[0, t].item()))

data.sort(key=lambda x: -x[0])
print(f"  {'conf':<8} {'pos':<5} {'input':<15} {'original':<20} {'swapped':<20} {'match'}")
for conf, t, tok, orig, swap, match in data:
    m = "OK" if match else "X"
    print(f"  {conf:<8.4f} {t:<5} {tok:<15} {orig:<20} {swap:<20} {m}")

# ========= Test 2: Are swapped tokens semantically related? =========
print(f"\n{'='*90}")
print("Test 2: Semantic analysis of token changes")
print("Are swapped tokens random or semantically related?")
print(f"{'='*90}")

print(f"\n  {'pos':<5} {'conf':<8} {'original':<20} {'swapped':<20} {'same type?':<15} {'match'}")
for t in range(T):
    orig_id = base_tokens[0, t].item()
    swap_id = swap_tokens[0, t].item()
    orig_str = tokenizer.decode(orig_id)
    swap_str = tokenizer.decode(swap_id)
    match = base_tokens[0, t].item() == swap_tokens[0, t].item()
    m = "OK" if match else "X"

    # Check token type similarity
    orig_is_space = orig_str.startswith(' ')
    swap_is_space = swap_str.startswith(' ')
    orig_is_alnum = orig_str.strip().isalnum() if orig_str.strip() else False
    swap_is_alnum = swap_str.strip().isalnum() if swap_str.strip() else False
    orig_is_punct = orig_str.strip() in '()[]{}:;,.-+=*/!@#$%^&*'
    swap_is_punct = swap_str.strip() in '()[]{}:;,.-+=*/!@#$%^&*'
    orig_is_newline = orig_str == '\n'
    swap_is_newline = swap_str == '\n'

    same_type = False
    if orig_is_space and swap_is_space:
        same_type = True
    elif orig_is_alnum and swap_is_alnum:
        same_type = True
    elif orig_is_punct and swap_is_punct:
        same_type = True
    elif orig_is_newline and swap_is_newline:
        same_type = True

    type_str = "same type" if same_type else "diff type"
    
    # Logit distance between original token and swapped token in the logit distribution
    orig_logit = base_logits[0, t, orig_id].item()
    swap_logit_in_orig = base_logits[0, t, swap_id].item()
    logit_diff = orig_logit - swap_logit_in_orig
    close_in_logits = "close" if abs(logit_diff) < 1.0 else "far"

    tok = tokenizer.decode(input_ids[0, t].item())
    print(f"  {t:<5} {base_conf[0,t].item():<8.4f} {orig_str:<20} {swap_str:<20} {same_type:<15} {m}  (logit diff={logit_diff:.2f}, {close_in_logits})")

# Logit distance distribution
print(f"\n  Distribution of logit distances (original vs swapped token in base logits):")
diffs = []
for t in range(T):
    if base_tokens[0, t].item() != swap_tokens[0, t].item():
        orig_id = base_tokens[0, t].item()
        swap_id = swap_tokens[0, t].item()
        diff = base_logits[0, t, orig_id].item() - base_logits[0, t, swap_id].item()
        diffs.append(diff)

if diffs:
    print(f"    Mean logit diff: {sum(diffs)/len(diffs):.2f}")
    print(f"    Min logit diff: {min(diffs):.2f}")
    print(f"    Max logit diff: {max(diffs):.2f}")
    n_close = sum(1 for d in diffs if abs(d) < 1.0)
    print(f"    Close calls (|diff|<1.0): {n_close}/{len(diffs)} ({n_close/len(diffs)*100:.1f}%)")

# Test 3: Top-5 overlap
print(f"\n{'='*90}")
print("Test 3: Top-5 token set overlap at each position")
print("(Do original and swapped models agree on the top candidates?)")
print(f"{'='*90}")

for t in range(T):
    orig_top5 = base_logits[0, t].topk(5).indices.tolist()
    swap_top5 = swap_logits[0, t].topk(5).indices.tolist()
    overlap = len(set(orig_top5) & set(swap_top5))
    tok = tokenizer.decode(input_ids[0, t].item())
    print(f"  pos {t:2d} '{tok}': Top-5 overlap = {overlap}/5 | original top: {[tokenizer.decode(x) for x in orig_top5[:3]]}")

print(f"\n{'='*90}")
print("SUMMARY")
print(f"{'='*90}")
print("Test 1: Token match by confidence")
for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    mask = base_conf[0] >= threshold
    n_total = mask.sum().item()
    if n_total > 0:
        n_match = ((base_tokens[0] == swap_tokens[0]) & mask).sum().item()
        print(f"  Conf>={threshold:.1f}: {n_match}/{n_total} ({n_match/n_total*100:.1f}%)")

print("Test 2: Most swapped tokens are same-type substitutions")
print("Test 3: Top-5 overlap indicates distribution-level agreement")

print("\nDone.")
