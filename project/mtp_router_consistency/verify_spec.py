from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch

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

prompt = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
input_ids = inputs.input_ids.to(device)
T = input_ids.shape[1]
tokens = [tokenizer.decode(t) for t in input_ids[0]]

print(f"\nPrompt: {prompt}")
print(f"Tokens: {T}")
print()

with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        output_router_logits=True,
        use_cache=False,
        return_dict=True,
    )

all_router = outputs.router_logits
num_layers = len(all_router) - 1
K = cfg.num_experts_per_tok

# Experts per layer per position
# all_router[l][0]: [1, T, E]
layer_experts = {}
for l in range(num_layers):
    router = all_router[l][0].squeeze(0)
    topk = router.topk(K, dim=-1).indices
    layer_experts[l] = topk  # [T, K]

# ========
# Simulate speculative decoding: verify d draft tokens at once
# Sequential baseline: generate 1 token at a time
# ========

print("=" * 70)
print("Simulation: Sequential vs Speculative Decoding")
print("=" * 70)

# Non-speculative baseline: generate N tokens one at a time
# Each generation step processes 1 token → 1 set of experts per layer
# Across N steps, we may load the SAME experts multiple times (redundant)

# Speculative: verify N draft tokens in one pass
# N positions processed simultaneously → expert sets can be shared

for N in [2, 4, 8, 16]:
    if N >= T:
        continue
    
    print(f"\n--- N={N} draft tokens ---")
    
    layers_all = range(num_layers)
    for layer_type, layer_range in [("All layers", layers_all),
                                      ("Shallow (0-4)", range(min(5, num_layers))),
                                      ("Middle (5-14)", range(5, min(15, num_layers))),
                                      ("Deep (15-19)", range(15, num_layers))]:
        
        # Baseline sequential: unique experts loaded over N positions, one at a time
        seq_unique = []
        for l in layer_range:
            all_seen = set()
            for t in range(N):
                all_seen.update(layer_experts[l][t].tolist())
            seq_unique.append(len(all_seen))
        
        # Speculative: unique experts needed for N positions in ONE pass
        # (some positions may share experts)
        spec_unique = []
        for l in layer_range:
            experts_at_all_pos = set()
            for t in range(N):
                experts_at_all_pos.update(layer_experts[l][t].tolist())
            spec_unique.append(len(experts_at_all_pos))
        
        avg_seq = sum(seq_unique) / len(seq_unique)
        avg_spec = sum(spec_unique) / len(spec_unique)
        
        # Savings: in sequential, we load then unload then load again
        # In speculative, we load once and share
        # Total expert loads:
        # Sequential: each step loads K experts per layer → N × len(layers) × K
        # Speculative: loads unique experts per layer once → sum(unique_per_layer)
        seq_total_loads = N * len(layer_range) * K
        spec_total_loads = sum(spec_unique)
        savings = (1 - spec_total_loads / seq_total_loads) * 100
        
        print(f"  {layer_type:<20} sequential: {seq_total_loads:4d} loads, "
              f"speculative: {spec_total_loads:3d} loads, "
              f"savings: {savings:5.1f}%")

# ========
# Key metric: when verifying N draft tokens together,
# how many UNIQUE experts per layer are activated?
# ========
print("\n" + "=" * 70)
print("Unique experts per layer during N-token verification pass")
print("=" * 70)

print(f"\n{'Layer':<8} {'N=2':<10} {'N=4':<10} {'N=8':<10} {'N=16':<10}")
print(f"{'-----':<8} {'----':<10} {'----':<10} {'----':<10} {'-----':<10}")
for l in range(min(20, T)):
    row = [f"L{l:<4}"]
    for N in [2, 4, 8, 16]:
        if N < T:
            experts = set()
            for t in range(N):
                experts.update(layer_experts[l][t].tolist())
            row.append(f"{len(experts):<6}/8{'':>4}")
        else:
            row.append(f"{'N/A':<10}")
    print("  ".join(row))

# Average across all layers
print(f"\n{'Avg':<8}", end="")
for N in [2, 4, 8, 16]:
    if N < T:
        totals = []
        for l in range(20):
            experts = set()
            for t in range(N):
                experts.update(layer_experts[l][t].tolist())
            totals.append(len(experts))
        avg = sum(totals) / len(totals)
        print(f" {avg:.1f}/8{'':>8}", end="")
print()

# ========
# Sequential generation: unique experts per token (worst-case)
# ========
print("\n" + "=" * 70)
print("Sequential generation: unique expert count per layer across N tokens")
print("(Each token loads K new experts; no sharing)")
print("=" * 70)

print(f"\n{'Layer':<8} {'N=1':<10} {'N=2':<10} {'N=4':<10} {'N=8':<10} {'N=16':<10}")
print(f"{'-----':<8} {'----':<10} {'----':<10} {'----':<10} {'----':<10} {'-----':<10}")
for l in range(min(20, T)):
    row = [f"L{l:<4}"]
    for N in [1, 2, 4, 8, 16]:
        if N < T:
            row.append(f"{min(N*K, 256):<6}{'':>4}")
        else:
            row.append(f"{'N/A':<10}")
    print("  ".join(row))

print("\nDone.")
