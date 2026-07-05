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

# Find decoder MoE layers
decoder_moe = [(idx, model.model.layers[idx].mlp) for idx in range(1, cfg.num_hidden_layers - 1)
               if hasattr(model.model.layers[idx].mlp, 'experts')]

print(f"Prompt: {prompt}")
print(f"Tokens: {T}, MoE layers: {len(decoder_moe)}")

# Register hooks that capture MoE inputs for ALL layers
moe_inputs = {}
moe_outputs = {}
handles = []

for idx, moe in decoder_moe:
    def mk_hooks(i_idx):
        def pre(m, inp):
            moe_inputs[i_idx] = inp[0].detach().cpu()
        def post(m, inp, out):
            moe_outputs[i_idx] = out[0].detach().cpu()
        handles.append(moe.register_forward_pre_hook(pre))
        handles.append(moe.register_forward_hook(post))
    mk_hooks(idx)

with torch.no_grad():
    outputs = model(input_ids=input_ids, output_router_logits=True,
                    output_hidden_states=True, use_cache=False, return_dict=True)
for h in handles:
    h.remove()

# Test 1: Progressive collapse - swap layers one by one and track final logit
print(f"\n{'='*80}")
print("Progressive collapse: swap layers one by one, track output")
print(f"{'='*80}")

# For each layer from last to first, swap progressively
# Start from all layers original, then swap one by one from deep to shallow
progressive_results = []

# Sort layers by index (1 to 18)
sorted_layers = sorted([idx for idx, _ in decoder_moe])

for swap_count in range(len(sorted_layers) + 1):
    layers_to_swap = set(sorted_layers[:swap_count])  # swap shallow first
    handles = []

    for idx, moe in decoder_moe:
        if idx in layers_to_swap:
            def mk_swap(ref):
                def hook(m, i, o):
                    with torch.no_grad():
                        inp = i[0]
                        g = ref.gate(inp)
                        flat = inp.view(-1, inp.shape[-1])
                        swapped = m.moe_infer(flat, g[0], g[1]).view_as(inp)
                        if m.shared_experts is not None:
                            swapped = swapped + m.shared_experts(inp)
                    return (swapped, o[1])
                return hook
            handles.append(moe.register_forward_hook(mk_swap(mtp_moe)))

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False, return_dict=True)

    for h in handles:
        h.remove()

    logits = out.logits.float()
    probs = F.softmax(logits, dim=-1)
    tokens = logits.argmax(dim=-1)
    top_conf = probs.max(dim=-1)[0]
    avg_conf = top_conf[0].mean().item()
    unique_tokens = tokens[0].unique().numel()
    most_common = tokens[0].mode().values.item()
    mc_str = tokenizer.decode(most_common)
    local = f"L{sorted_layers[0]}..L{sorted_layers[min(swap_count,len(sorted_layers))-1]}" if swap_count > 0 else "none"
    logit_cos = 0 if swap_count == 0 else F.cosine_similarity(
        logits[0].float().view(-1), outputs.logits[0].float().view(-1), dim=0).item()
    
    match = (tokens == outputs.logits.argmax(dim=-1)).sum().item()
    progressive_results.append((swap_count, unique_tokens, mc_str, avg_conf, logit_cos, match))
    
    print(f"  Swap {swap_count:2d} layers ({layers_to_swap}): uniq={unique_tokens:2d} mode={mc_str:<8} conf={avg_conf:.3f} cos={logit_cos:.4f} match={match}/{T}")

# Test 2: Where does `!` come from? Check lm_head weights
print(f"\n{'='*80}")
print("Where does '!' come from? Check lm_head weight norm per token")
print(f"{'='*80}")

lm_w = model.lm_head.weight  # [V, D]
lm_norm = lm_w.norm(dim=1)

# Find `!` token id
excl_id = tokenizer.encode("!")[0]
print(f"  '!' token ID: {excl_id}")
print(f"  '!' weight norm: {lm_norm[excl_id].item():.2f}")
print(f"  Mean weight norm: {lm_norm.mean().item():.2f}")
print(f"  Max weight norm: {lm_norm.max().item():.2f}")
print(f"  '!' rank by norm: {(lm_norm > lm_norm[excl_id]).sum().item()}/{lm_norm.shape[0]}")

# Top-10 tokens by weight norm
top10 = lm_norm.topk(10)
print(f"  Top-10 tokens by weight norm:")
for i in range(10):
    tok = tokenizer.decode(top10.indices[i].item())
    print(f"    {top10.indices[i].item():6d} norm={top10.values[i].item():.2f} '{tok}'")

# Test 3: After collapse, are all positions producing the SAME hidden state?
print(f"\n{'='*80}")
print("Collapse analysis: Are all positions producing identical hidden states?")
print(f"{'='*80}")

# Register hooks for ALL MoE layers + swap
handles = []
for idx, moe in decoder_moe:
    def mk_swap(ref):
        def hook(m, i, o):
            with torch.no_grad():
                inp = i[0]
                g = ref.gate(inp)
                flat = inp.view(-1, inp.shape[-1])
                swapped = m.moe_infer(flat, g[0], g[1]).view_as(inp)
                if m.shared_experts is not None:
                    swapped = swapped + m.shared_experts(inp)
            return (swapped, o[1])
        return hook
    handles.append(moe.register_forward_hook(mk_swap(mtp_moe)))

# Capture final hidden state
collapsed_hidden = None
def capture_hidden(m, i, o):
    global collapsed_hidden
    collapsed_hidden = o[0].detach().cpu()

# Also capture the last decoder layer's input for comparison
last_layer = model.model.layers[17]  # layer 18
h_last = last_layer.register_forward_hook(capture_hidden)

with torch.no_grad():
    out_collapsed = model(input_ids=input_ids, use_cache=False, return_dict=True)

h_last.remove()
for h in handles:
    h.remove()

collapsed_logits = out_collapsed.logits.float()
collapsed_tokens = collapsed_logits.argmax(dim=-1)

# Check if all positions produce similar hidden states
print(f"  After collapse (all layers swapped):")
pos_diffs = []
for t in range(1, T):
    diff = (collapsed_hidden[0, 0] - collapsed_hidden[0, t]).norm().item()
    pos_diffs.append(diff)
print(f"  Mean inter-position L2 diff: {sum(pos_diffs)/len(pos_diffs):.2f}")
print(f"  Max inter-position L2 diff: {max(pos_diffs):.2f}")
print(f"  Min inter-position L2 diff: {min(pos_diffs):.2f}")

# Compare with normal forward
with torch.no_grad():
    normal_out = model(input_ids=input_ids, use_cache=False, return_dict=True)
normal_hidden_path = normal_out.hidden_states
normal_last_hidden = normal_hidden_path[-2]  # last decoder layer output (before norm)

normal_diffs = []
for t in range(1, T):
    diff = (normal_last_hidden[0, 0] - normal_last_hidden[0, t]).norm().item()
    normal_diffs.append(diff)
print(f"  Normal (no swap) inter-position L2 diff: {sum(normal_diffs)/len(normal_diffs):.2f}")

# Test 4: What if we only swap L1, then run the rest normally?
print(f"\n{'='*80}")
print("Is collapse caused by L1 alone?")
print(f"{'='*80}")

for single_layer in [1, 5, 10, 18]:
    handles = []
    for idx, moe in decoder_moe:
        if idx == single_layer:
            def mk_swap(ref):
                def hook(m, i, o):
                    with torch.no_grad():
                        inp = i[0]
                        g = ref.gate(inp)
                        flat = inp.view(-1, inp.shape[-1])
                        swapped = m.moe_infer(flat, g[0], g[1]).view_as(inp)
                        if m.shared_experts is not None:
                            swapped = swapped + m.shared_experts(inp)
                    return (swapped, o[1])
                return hook
            handles.append(moe.register_forward_hook(mk_swap(mtp_moe)))

    with torch.no_grad():
        out_single = model(input_ids=input_ids, use_cache=False, return_dict=True)

    for h in handles:
        h.remove()

    slogits = out_single.logits.float()
    suniq = slogits.argmax(dim=-1)[0].unique().numel()
    logit_cos_single = F.cosine_similarity(
        slogits[0].float().view(-1), outputs.logits[0].float().view(-1), dim=0).item()
    print(f"  Swap only L{single_layer}: unique_tokens={suniq}, logit_cos={logit_cos_single:.4f}")

print(f"\nDone.")
