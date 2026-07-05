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

prompts = [
    "def foo(x): return x + x * x + bar(x)",     # code
    "The quick brown fox jumps over the lazy dog",  # English
    "def fibonacci(n):\n    if n <= 1:\n        return n",  # long code
    "Machine learning is a subset of artificial intelligence",  # technical
    "a = [1, 2, 3, 4, 5]\nb = [x**2 for x in a]",  # python
]

all_results = {}
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
    input_ids = inputs.input_ids.to(device)
    T = input_ids.shape[1]
    if T < 4:
        print(f"  Skip '{prompt[:30]}': too short ({T} tokens)")
        continue

    with torch.no_grad():
        raw = model.model(input_ids=input_ids, output_hidden_states=True, use_cache=False, return_dict=True)
        full = model(input_ids=input_ids, output_router_logits=True, use_cache=False, return_dict=True)

    mtp_hidden = raw.mtp_hidden_states[0]
    all_router = full.router_logits

    layer_results = {}
    for idx in range(1, cfg.num_hidden_layers - 1):
        layer = model.model.layers[idx]
        if not hasattr(layer.mlp, 'experts') or len(layer.mlp.experts) == 0:
            continue
        gate = layer.mlp.gate
        router_idx = idx - 1
        if router_idx >= len(all_router) - 1:
            continue
        actual_router = all_router[router_idx][0]
        K = cfg.num_experts_per_tok

        cos_sum, ov_sum, cnt = 0, 0, 0
        for t in range(T - 1):
            with torch.no_grad():
                _, _, pred_logits = gate(mtp_hidden[0, t:t+1])
            actual = actual_router[0, t+1:t+2]
            cos = F.cosine_similarity(pred_logits.float(), actual.float()).item()
            ov = len(set(pred_logits.topk(K).indices[0].tolist()) & set(actual.topk(K).indices[0].tolist()))
            cos_sum += cos
            ov_sum += ov
            cnt += 1

        if cnt > 0:
            layer_results[idx] = (cos_sum / cnt, ov_sum / cnt)

    all_results[prompt[:40]] = (T, layer_results)

print(f"\n{'='*110}")
print("Multi-prompt validation: MTP hidden state -> decoder gate routing prediction")
print(f"{'='*110}")

print(f"\n{'Prompt':<42} {'Tokens':<8} {'Avg Cos':<10} {'Avg Overlap':<12} {'Best Layer':<12}")
print(f"{'------':<42} {'------':<8} {'-------':<10} {'-----------':<12} {'----------':<12}")

for label, (T, layers) in all_results.items():
    if not layers:
        continue
    avg_cos = sum(c for c, _ in layers.values()) / len(layers)
    avg_ov = sum(o for _, o in layers.values()) / len(layers)
    best_l = max(layers, key=lambda l: layers[l][0])
    best_cos = layers[best_l][0]
    short = label[:40]
    print(f"{short:<42} {T:<8} {avg_cos:<10.4f} {avg_ov:<6.2f}/8{'':>6} L{best_l} ({best_cos:.3f})")

print(f"\n{'='*110}")
print("Per-layer breakdown across prompts")
print(f"{'='*110}")

layers_to_show = [1, 3, 6, 9, 13, 15, 16, 18]
header = f"{'Layer':<8}" + "".join(f"{p[:12]:<14}" for p, _ in all_results.items())
print(f"\n{header}")

for l in layers_to_show:
    row = f"L{l:<6}"
    for label, (T, layers) in all_results.items():
        if l in layers:
            c, o = layers[l]
            row += f"{c:<.3f}/{o:<5.1f}{'':>4}"
        else:
            row += f"{'N/A':<14}"
    print(row)

# Cross-prompt stability: variance of Cos across prompts per layer
print(f"\n{'='*110}")
print("Cross-prompt stability: does routing prediction work consistently?")
print(f"{'='*110}")

all_layers = set()
for _, (_, layers) in all_results.items():
    all_layers.update(layers.keys())

stable_layers = []
for l in sorted(all_layers):
    cos_vals = []
    for _, (_, layers) in all_results.items():
        if l in layers:
            cos_vals.append(layers[l][0])
    if len(cos_vals) >= 3:
        mean_c = sum(cos_vals) / len(cos_vals)
        std_c = (sum((v - mean_c)**2 for v in cos_vals) / len(cos_vals))**0.5
        stable_layers.append((l, mean_c, std_c))
        flag = " stable" if std_c < 0.05 else (" unstable" if std_c > 0.15 else "")
        print(f"  L{l:<4}: mean Cos={mean_c:.3f}, std={std_c:.4f}{flag}")

print(f"\n{'='*110}")
print("CONCLUSION")
print(f"{'='*110}")
if stable_layers:
    avg_m = sum(s[1] for s in stable_layers) / len(stable_layers)
    avg_s = sum(s[2] for s in stable_layers) / len(stable_layers)
    print(f"  Average Cos across {len(prompts)} prompts: {avg_m:.3f}")
    print(f"  Average std-dev: {avg_s:.4f}")
    print(f"  {'Prediction is consistent across prompts (std < 0.1).' if avg_s < 0.1 else 'Prediction varies by prompt type.'}")
    print(f"  MTP hidden -> routing prediction is {'VALID' if avg_m > 0.7 else 'LIMITED'} for expert prefetching.")

print("\nDone.")
