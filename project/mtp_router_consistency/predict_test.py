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

# Get MTP hidden states + decoder hidden states + router logits
with torch.no_grad():
    # For MTP hidden state: use raw model with output_hidden_states
    raw = model.model(input_ids=input_ids, output_hidden_states=True, use_cache=False, return_dict=True)
    # For decoder router logits: use full model with output_router_logits
    full = model(input_ids=input_ids, output_router_logits=True, output_hidden_states=True, use_cache=False, return_dict=True)

decoder_hidden = raw.hidden_states  # 20 entries: [embeds, after_0..after_18]
mtp_hidden_states = raw.mtp_hidden_states[0]  # [1, T, D], MTP hidden state
all_router = full.router_logits  # 19 entries: 18 decoder MoE + 1 MTP

# MTP hidden[t] should align with decoder_hidden at position t+1 (both encode tok[t+2])
# MTP hidden[t] → through decoder_gate[l] → should predict routing for decoder[l][t+1]

print(f"\n{'='*90}")
print("Test: Feed MTP hidden state through each decoder layer's gate")
print("MTP_hidden[t] → decoder_layer[l].gate → routing_logits[l]")
print("Compare with actual decoder routing_logits at position t+1")
print(f"{'='*90}")

# Identify decoder MoE layers
moe_layers = []
for idx in range(cfg.num_hidden_layers - 1):
    layer = model.model.layers[idx]
    if hasattr(layer.mlp, 'experts') and len(layer.mlp.experts) > 0:
        moe_layers.append((idx, layer))

print(f"\nDecoder MoE layers: {len(moe_layers)} (indices: {[l[0] for l in moe_layers]})")

results = []
for layer_idx, layer in moe_layers:
    decoder_gate = layer.mlp.gate  # BailingMoeV2Gate
    K = cfg.num_experts_per_tok
    E = cfg.num_experts
    D = decoder_hidden[0].shape[-1]

    # For each position t, compare:
    #   gate(mtp_hidden[t]) vs actual_router[layer][t+1]
    cos_sum = 0
    topk_overlap_sum = 0
    pair_count = 0

    print(f"\n  Layer {layer_idx} (decoder gate on MTP hidden):")
    for t in range(T - 1):  # need t+1 < T
        # MTP hidden state at position t → through decoder gate
        mtp_h = mtp_hidden_states[0, t:t+1]  # [1, D]
        # Get decoder layer's actual router logits at position t+1
        # all_router[i][0] has shape [1, T, E]
        router_idx = layer_idx - 1  # all_router[0] = layer 1
        if router_idx >= len(all_router) - 1:
            continue  # skip MTP
        actual_router_logits = all_router[router_idx][0]  # [1, T, E]

        with torch.no_grad():
            # Feed MTP hidden through decoder's gate
            # gate.forward expects [N, D], returns (topk_idx, topk_weight, logits)
            _, _, mtp_router_logits = decoder_gate(mtp_h)

        actual_at_t1 = actual_router_logits[0, t+1:t+2]  # [1, E]

        # Cosine similarity of full 256-dim router logits
        cos = F.cosine_similarity(mtp_router_logits.float(), actual_at_t1.float()).item()

        # Top-8 index overlap
        mtp_topk = mtp_router_logits.topk(K, dim=-1).indices[0].tolist()
        actual_topk = actual_at_t1.topk(K, dim=-1).indices[0].tolist()
        overlap = len(set(mtp_topk) & set(actual_topk))

        cos_sum += cos
        topk_overlap_sum += overlap
        pair_count += 1

        if t < 3:  # only print first few positions
            tok_t1 = tokenizer.decode(input_ids[0, t+1].item())
            print(f"    t={t} (tok[t+1]='{tok_t1}'): Cos={cos:.4f}, Top-8 overlap={overlap}/{K}")

    if pair_count > 0:
        avg_cos = cos_sum / pair_count
        avg_overlap = topk_overlap_sum / pair_count
        results.append((layer_idx, avg_cos, avg_overlap))
        print(f"    Avg Cos={avg_cos:.4f}, Avg overlap={avg_overlap:.2f}/{K}")
    else:
        print(f"    (no valid positions)")

# Summary
print(f"\n{'='*90}")
print("SUMMARY: MTP hidden → decoder gate routing prediction")
print(f"{'='*90}")
print(f"{'Layer':<8} {'Avg Cos':<12} {'Avg Overlap':<15}")
print(f"{'-----':<8} {'-------':<12} {'-----------':<15}")
best_layer = max(results, key=lambda x: x[1])
worst_layer = min(results, key=lambda x: x[1])
for idx, cos, ov in results:
    marker = " ← BEST" if idx == best_layer[0] else (" ← WORST" if idx == worst_layer[0] else "")
    print(f"L{idx:<6} {cos:<12.4f} {ov:<8.2f}/{8}{marker}")

avg_cos_all = sum(r[1] for r in results) / len(results)
avg_ov_all = sum(r[2] for r in results) / len(results)
print(f"{'ALL':<7} {avg_cos_all:<12.4f} {avg_ov_all:<8.2f}/{8}")

# Compare with using MTP's OWN gate on MTP's hidden state
print(f"\n  Reference: MTP's OWN gate on MTP hidden → MTP's true router logits")
print(f"  (upper bound: how well MTP gate predicts its own routing)")
mtp_moe_block = model.model.layers[-1].mlp
mtp_router_logits_all = all_router[-1][0]  # [1, T, E]
cos_mtp_self = 0
topk_self = 0
cnt = 0
for t in range(T - 1):
    mtp_gate_in = mtp_hidden_states[0, t:t+1]
    with torch.no_grad():
        _, _, mtp_own_logits = mtp_moe_block.gate(mtp_gate_in)
    actual_mtp_at_t = mtp_router_logits_all[0, t:t+1].float()
    c = F.cosine_similarity(mtp_own_logits.float(), actual_mtp_at_t).item()
    o = len(set(mtp_own_logits.topk(8).indices[0].tolist()) & set(actual_mtp_at_t.topk(8).indices[0].tolist()))
    cos_mtp_self += c
    topk_self += o
    cnt += 1
print(f"  MTP self-check: avg Cos={cos_mtp_self/cnt:.4f}, avg overlap={topk_self/cnt:.2f}/8")

print(f"\n{'='*90}")
print("CONCLUSION")
print(f"{'='*90}")
if avg_cos_all > 0.5:
    print(f"MTP hidden state → decoder gate routing: Cos={avg_cos_all:.3f}, Top-8 overlap={avg_ov_all:.1f}/8")
    print("MTP hidden state CAN partially predict decoder routing via decoder's own gate.")
    print("This suggests a lightweight predictor from MTP hidden → decoder routing is feasible.")
else:
    print(f"MTP hidden state → decoder gate routing: Cos={avg_cos_all:.3f}, Top-8 overlap={avg_ov_all:.1f}/8")
    print("MTP hidden state alone is NOT sufficient to predict decoder routing.")
    print("Decoder routing depends on per-layer hidden states, not just the final output.")

print("\nDone.")
