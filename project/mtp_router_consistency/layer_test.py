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

# Identify all decoder MoE layers
all_moe = [(idx, model.model.layers[idx].mlp) for idx in range(cfg.num_hidden_layers - 1)
           if hasattr(model.model.layers[idx].mlp, 'experts') and len(model.model.layers[idx].mlp.experts) > 0]

print(f"Total MoE layers: {len(all_moe)}")

# Baseline
with torch.no_grad():
    base_out = model(input_ids=input_ids, use_cache=False, return_dict=True)
base_logits = base_out.logits.float()
base_tokens = base_logits.argmax(dim=-1)

def run_with_swap(layer_indices, label):
    handles = []
    for idx, moe in all_moe:
        if idx in layer_indices:
            def make_hook(ref):
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
            handles.append(moe.register_forward_hook(make_hook(mtp_moe)))

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False, return_dict=True)

    for h in handles:
        h.remove()

    swap_logits = out.logits.float()
    swap_tokens = swap_logits.argmax(dim=-1)
    match = (base_tokens == swap_tokens).sum().item()

    # Logit Cosine
    logit_cos = 0
    for t in range(T):
        logit_cos += F.cosine_similarity(base_logits[0,t].unsqueeze(0).float(), swap_logits[0,t].unsqueeze(0).float()).item()
    logit_cos /= T

    # Collapse detection: count unique tokens in swapped output
    unique_tokens = swap_tokens.unique().numel()
    top_token = swap_tokens.mode().values.item()
    top_token_str = tokenizer.decode(top_token)

    print(f"\n{label}:")
    print(f"  Logit Cos={logit_cos:.4f}, Token match={match}/{T} ({match/T*100:.1f}%)")
    print(f"  Unique tokens in output={unique_tokens}, most common='{top_token_str}'")

    for t in range(T):
        orig = tokenizer.decode(base_tokens[0,t].item())
        swp = tokenizer.decode(swap_tokens[0,t].item())
        tok = tokenizer.decode(input_ids[0,t].item())
        m = "OK" if base_tokens[0,t].item() == swap_tokens[0,t].item() else "X"
        print(f"    pos {t:2d} {tok:<12} {orig:<15} -> {swp:<15} {m}")

    return match, logit_cos, unique_tokens

# Test configurations
configs = [
    ("Shallow (L1-L4)",    [idx for idx, _ in all_moe if idx <= 3]),   # indices 0-3 → layers 1-4
    ("Middle (L5-L13)",    [idx for idx, _ in all_moe if 4 <= idx <= 12]),
    ("Deep (L14-L18)",     [idx for idx, _ in all_moe if idx >= 13]),
    ("Shallow+Deep",       [idx for idx, _ in all_moe if idx <= 3 or idx >= 13]),
    ("Single L1",          [all_moe[0][0]]),
    ("Single L9",          [all_moe[8][0]]),
    ("Single L18",         [all_moe[-1][0]]),
    ("ALL layers",         [idx for idx, _ in all_moe]),
]

results = []
for label, layers in configs:
    m, lc, ut = run_with_swap(set(layers), label)
    results.append((label, m, lc, ut))

print(f"\n{'='*90}")
print("SUMMARY")
print(f"{'='*90}")
print(f"{'Config':<20} {'Match':<15} {'LogitCos':<12} {'UniqueTok':<12} {'Collapse?':<12}")
print(f"{'------':<20} {'-----':<15} {'--------':<12} {'--------':<12} {'--------':<12}")

for label, match, lc, ut in results:
    collapse = "YES" if ut <= 2 else "no"
    print(f"{label:<20} {match}/{T} ({match/T*100:.1f}%)  {lc:<12.4f} {ut:<12} {collapse:<12}")

print("\nDone.")
