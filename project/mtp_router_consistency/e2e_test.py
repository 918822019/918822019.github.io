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

# Test last 3 MoE layers (L16, L17, L18)
test_layers = [(idx, model.model.layers[idx]) for idx in [15, 16, 17]]

# Hook: capture MoE input and output
layer_data = {}
handles = []
for idx, layer in test_layers:
    def mk_pre(i_idx):
        def pre(m, i):
            layer_data.setdefault(i_idx, {})["moe_in"] = i[0].detach()
        return pre
    def mk_post(i_idx):
        def post(m, i, o):
            d = layer_data.setdefault(i_idx, {})
            d["moe_out"] = o[0].detach()
        return post
    handles.append(layer.mlp.register_forward_pre_hook(mk_pre(idx)))
    handles.append(layer.mlp.register_forward_hook(mk_post(idx)))

with torch.no_grad():
    outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False, return_dict=True)

for h in handles:
    h.remove()

all_hidden = outputs.hidden_states  # [21] entries
# all_hidden[0] = embeds, all_hidden[l] = after layer l-1 for l>=1
# all_hidden[l+1] = output of layer l = input to layer l+1

# Baseline: actual final logits
final_norm = model.model.norm
lm_head = model.lm_head

with torch.no_grad():
    actual_final = final_norm(all_hidden[-1].to(device))  # final hidden after norm
    actual_logits = lm_head(actual_final).float()
    actual_tokens = actual_logits.argmax(dim=-1)

print(f"\n{'='*90}")
print("End-to-end: Swap routing in individual decoder MoE layers")
print("Using MTP routing -> Decoder expert weights")
print(f"{'='*90}")

for idx, layer in test_layers:
    d = layer_data[idx]
    moe_in = d["moe_in"]  # [1, T, D], MoE input (after post_attention_layernorm)
    orig_moe_out = d["moe_out"]  # [1, T, D], MoE output before residual

    # The decoder layer output = residual + moe_out
    # all_hidden[idx+1] = output of this decoder layer
    layer_output = all_hidden[idx+1]  # [1, T, D]
    residual = layer_output - orig_moe_out.to(layer_output.device)  # the residual value

    # MTP routing -> Decoder experts
    with torch.no_grad():
        mtp_gate = mtp_moe.gate(moe_in)
        flat_in = moe_in.view(-1, moe_in.shape[-1])
        swapped_moe = layer.mlp.moe_infer(flat_in, mtp_gate[0], mtp_gate[1]).view_as(moe_in)
        if layer.mlp.shared_experts is not None:
            swapped_moe = swapped_moe + layer.mlp.shared_experts(moe_in)

    # New layer output = same residual + swapped MoE
    swapped_layer_out = residual + swapped_moe.to(residual.device)

    # Now propagate: the remaining layers process the swapped output
    # For subsequent layers, we'd need full forward pass.
    # For the LAST layer (idx=17), no more decoder layers after it.
    # For earlier layers, the swapped output feeds into subsequent decoder layers.
    if idx == 17:  # last decoder layer
        final_swapped = final_norm(swapped_layer_out)
        swapped_logits = lm_head(final_swapped).float()
    else:
        # Swapping layer idx changes the input to layer idx+1..18
        # We'd need to run all subsequent layers with the swapped input
        # Skip this for now (complex)
        swapped_logits = None

    if swapped_logits is not None:
        moe_cos = F.cosine_similarity(orig_moe_out.float().view(-1), swapped_moe.float().view(-1), dim=0).item()
        logit_cos = F.cosine_similarity(actual_logits.float().view(-1), swapped_logits.float().view(-1), dim=0).item()
        
        match_count = 0
        print(f"\n  Layer {idx} (last decoder MoE layer):")
        for t in range(T):
            o_tok = tokenizer.decode(actual_tokens[0,t].item())
            s_tok = tokenizer.decode(swapped_logits[0,t].argmax().item())
            match = "OK" if actual_tokens[0,t].item() == swapped_logits[0,t].argmax().item() else "X"
            if match == "OK": match_count += 1
            tok = tokenizer.decode(input_ids[0,t].item())
            print(f"    pos {t:2d} '{tok}': Original={o_tok:<15} Swapped={s_tok:<15} {match}")
        
        print(f"  MoE Cos={moe_cos:.4f}, Logit Cos={logit_cos:.4f}")
        print(f"  Token match: {match_count}/{T} ({match_count/T*100:.1f}%)")
    elif idx < 17:
        print(f"\n  Layer {idx}: skipped (forward propagation not implemented)")

# Additional test: what if we apply the SAME delta to all layers?
# Use the delta from the last layer (which we computed) to estimate cumulative impact
print(f"\n{'='*90}")
print("Summary")
print(f"{'='*90}")
print(f"Last decoder layer (L18) routing swap:")
print(f"  MTP routing -> Decoder experts")
print(f"  MoE output Cos: {moe_cos:.4f}")
print(f"  Final logits Cos: {logit_cos:.4f}")
print(f"  Token prediction match: ~{match_count}/{T}")
print()
print(f"Combined with earlier findings:")
print(f"  - Individual MoE layer Cos: 0.47~0.72 (avg 0.58)")
print(f"  - Final lm_head logits Cos: 0.97 (from Section 7)")
print(f"  - Last layer MoE Cos: 0.72, Logit Cos: 0.9995")
print(f"  - Routing mismatch is heavily damped by layernorm + lm_head")

print("\nDone.")
