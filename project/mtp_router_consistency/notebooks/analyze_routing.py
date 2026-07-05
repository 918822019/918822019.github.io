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
    cfg.model_path, device=cfg.device, torch_dtype=getattr(torch, cfg.torch_dtype, torch.bfloat16),
)

# Longer prompt with potential token repeats
prompt = "def foo(x): return x + x * x + bar(x)"
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_prompt_len)
input_ids = inputs.input_ids.to(device)
tokens = [tokenizer.decode(t) for t in input_ids[0]]

print(f"\nPrompt: {prompt}")
print(f"Tokens ({input_ids.shape[1]}):")
for i, (tid, t) in enumerate(zip(input_ids[0].tolist(), tokens)):
    print(f"  [{i:3d}] id={tid:6d} '{t}'")

with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        output_router_logits=True,
        use_cache=False,
        return_dict=True,
    )

all_router = outputs.router_logits
num_layers = len(all_router) - 1
full_len = input_ids.shape[1]

# Collect top-k experts per decoder layer per position
# all_router[i][0]: [1, full_len, num_experts]
K = cfg.num_experts_per_tok
layer_topk = {}
for l in range(num_layers):
    router = all_router[l][0].squeeze(0)  # [full_len, E]
    topk = router.topk(K, dim=-1).indices  # [full_len, K]
    layer_topk[l] = topk  # [full_len, K]

# MTP router
mtp_router = all_router[-1][0].squeeze(0)  # [full_len, E]
mtp_topk = mtp_router.topk(K, dim=-1).indices  # [full_len, K]

# Last decoder layer
decoder_last = all_router[-2][0].squeeze(0)
decoder_last_topk = decoder_last.topk(K, dim=-1).indices

# =============================================
# Analysis 1: Are same tokens routed to same experts?
# =============================================
print("\n" + "=" * 70)
print("Analysis 1: Token → Expert mapping stability")
print("=" * 70)

# Group positions by token id
token_positions = {}
for pos, tid in enumerate(input_ids[0].tolist()):
    token_positions.setdefault(tid, []).append(pos)

print(f"\nUnique tokens: {len(token_positions)} / {full_len} positions")
print(f"Tokens appearing 2+ times:")
for tid, positions in sorted(token_positions.items()):
    if len(positions) >= 2:
        tok_str = tokenizer.decode(tid)
        print(f"\n  '{tok_str}' (id={tid}) at positions {positions}")

        # Check last decoder layer expert overlap for same token at different positions
        for l in [0, num_layers // 2, num_layers - 1, num_layers - 1]:  # first, middle, last
            sets = [set(layer_topk[l][p].tolist()) for p in positions]
            avg_iou = 0
            count = 0
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    inter = sets[i] & sets[j]
                    union = sets[i] | sets[j]
                    avg_iou += len(inter) / len(union) if union else 1.0
                    count += 1
            avg_iou /= max(count, 1)
            print(f"    Layer {l:2d} avg same-token IoU: {avg_iou:.4f}")

# =============================================
# Analysis 2: Predictability of next-position experts from token
# =============================================
print("\n" + "=" * 70)
print("Analysis 2: Expert continuity between adjacent positions")
print("=" * 70)

for l in [0, num_layers // 2, num_layers - 1]:
    overlap_counts = []
    for t in range(full_len - 1):
        curr_set = set(layer_topk[l][t].tolist())
        next_set = set(layer_topk[l][t + 1].tolist())
        overlap_counts.append(len(curr_set & next_set))
    avg_overlap = sum(overlap_counts) / len(overlap_counts)
    print(f"  Layer {l:2d}: avg overlap between pos t and t+1: {avg_overlap:.2f}/{K}")

# =============================================
# Analysis 3: Can MTP's expert selection predict decoder's?
# MTP at position t routes for predicting tok[t+2].
# The decoder layer at position t+1 also handles tok[t+2].
# Compare: MTP routing at t vs decoder routing at t+1
# =============================================
print("\n" + "=" * 70)
print("Analysis 3: MTP(t) vs Decoder_Last(t+1) — same target token tok[t+2]")
print("=" * 70)

mtp_pred_tok = []
decoder_handles_tok = []
for t in range(full_len - 2):
    mtp_experts = set(mtp_topk[t].tolist())
    dec_experts = set(decoder_last_topk[t + 1].tolist())
    inter = mtp_experts & dec_experts
    mtp_pred_tok.append(len(inter))
    decoder_handles_tok.append(len(inter))

print(f"  Positions compared (t=0..T-2, same target tok[t+2]): {len(mtp_pred_tok)}")
avg_overlap_experts = sum(mtp_pred_tok) / len(mtp_pred_tok) if mtp_pred_tok else 0
print(f"  MTP(t) vs Decoder(t+1) expert overlap: {avg_overlap_experts:.2f}/{K}")
print(f"  Same as before (from Section 2 report): {avg_overlap_experts/K*100:.1f}% overlap")

# But the novel question: token predicts experts?
print(f"\n  --- Per token breakdown ---")
for t in range(full_len - 2):
    target_tok = input_ids[0, t + 2].item()  # tok[t+2] is the common target
    target_str = tokenizer.decode(target_tok)
    mtp_experts = set(mtp_topk[t].tolist())
    dec_experts = set(decoder_last_topk[t + 1].tolist())
    overlap = len(mtp_experts & dec_experts)
    tok_at_t1 = input_ids[0, t + 1].item()
    print(f"    t={t}: MTP sees '{tokenizer.decode(tok_at_t1)}', predicts '{target_str}' → expert overlap {overlap}/{K}")

# =============================================
# Analysis 4: Token-to-expert mapping consistency across layers
# =============================================
print("\n" + "=" * 70)
print("Analysis 4: Do same tokens at adjacent positions share experts?")
print("    (If yes, knowing the token predicts the experts)")
print("=" * 70)

# For each pair of adjacent positions with the SAME token, check expert overlap
for l in [0, num_layers // 2, num_layers - 1]:
    same_tok_overlaps = []
    diff_tok_overlaps = []
    for t in range(full_len - 1):
        tok_t = input_ids[0, t].item()
        tok_t1 = input_ids[0, t + 1].item()
        experts_t = set(layer_topk[l][t].tolist())
        experts_t1 = set(layer_topk[l][t + 1].tolist())
        o = len(experts_t & experts_t1)
        if tok_t == tok_t1:
            same_tok_overlaps.append(o)
        else:
            diff_tok_overlaps.append(o)

    if same_tok_overlaps:
        avg_same = sum(same_tok_overlaps) / len(same_tok_overlaps)
        print(f"  Layer {l:2d}: same-token adjacent overlap: {avg_same:.2f}/{K} ({len(same_tok_overlaps)} pairs)")
    avg_diff = sum(diff_tok_overlaps) / len(diff_tok_overlaps) if diff_tok_overlaps else 0
    print(f"         diff-token adjacent overlap: {avg_diff:.2f}/{K} ({len(diff_tok_overlaps)} pairs)")

# =============================================
# Analysis 5: The key question — given token, can we predict top-8 experts
# =============================================
print("\n" + "=" * 70)
print("Analysis 5: Token → Top-8 expert set stability")
print("    (If a token appears at 2+ positions, are the same experts activated?)")
print("=" * 70)

# Collect all tokens grouped by their selected experts
# For each token id, compute the set of experts activated across all positions
# If token → expert is stable, the sets should be identical or very similar
from collections import defaultdict

token_expert_sets = defaultdict(list)
for pos in range(full_len):
    tid = input_ids[0, pos].item()
    for l in [num_layers - 1]:  # last layer only
        expert_set = frozenset(layer_topk[l][pos].tolist())
        token_expert_sets[tid].append(expert_set)

stable_tokens = 0
unstable_tokens = 0
for tid, sets in token_expert_sets.items():
    unique_sets = set(sets)
    if len(unique_sets) == 1:
        stable_tokens += 1
    else:
        unstable_tokens += 1
        tok_str = tokenizer.decode(tid)
        print(f"  Token '{tok_str}' (id={tid}): {len(unique_sets)} different expert sets across {len(sets)} positions")

total_tokens_with_duplicates = sum(1 for v in token_expert_sets.values() if len(v) >= 2)
print(f"\n  Tokens with 2+ occurrences: {total_tokens_with_duplicates}")
if total_tokens_with_duplicates > 0:
    print(f"  Fully stable (same experts every time): {stable_tokens}")
    print(f"  Unstable (different experts per position): {unstable_tokens}")

# =============================================
# Analysis 6: MTP hidden state vs decoder layer hidden states
# Which decoder layer does MTP's hidden state resemble most?
# =============================================
print("\n" + "=" * 70)
print("Analysis 6: MTP hidden state vs decoder layer hidden states")
print("    MTP[t] predicts tok[t+2], decoder[l][t+1] also encodes tok[t+2]")
print("    Which decoder layer is MTP equivalent to?")
print("=" * 70)

with torch.no_grad():
    # Use raw model to get mtp_hidden_states (CausalLM wrapper doesn't expose it)
    raw_outputs = model.model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

decoder_hidden = raw_outputs.hidden_states  # tuple of 21 tensors [1, T, D]
mtp_hidden_states = raw_outputs.mtp_hidden_states  # list of 1 tensor [1, T, D]
mtp_hidden = mtp_hidden_states[0]  # [1, T, D]

T = full_len
D = decoder_hidden[0].shape[-1]

# Compare: MTP[t] (prepares for tok[t+2]) vs decoder[l][t+1] (also encodes tok[t+2])
# Range: t = 0..T-2 (since we need t+1 < T)
num_decoder_hidden = len(decoder_hidden)

print(f"\n  Decoder layers: {num_decoder_hidden} (including input embeddings)")
print(f"  MTP hidden states: {len(mtp_hidden_states)}")
print(f"  Sequence length: {T}")
print(f"  Hidden dim: {D}")
print()

# For each t', find which decoder layer at position t'+1 is closest to MTP[t']
# (both encode tok[t'+2])
layer_cosines = [[] for _ in range(num_decoder_hidden)]
for t in range(T - 1):  # compare MTP[t] with decoder[l][t+1]
    mtp_vec = mtp_hidden[0, t].float().unsqueeze(0)  # [1, D]
    for l in range(num_decoder_hidden):
        dec_vec = decoder_hidden[l][0, t + 1].float().unsqueeze(0)  # [1, D]
        cos = F.cosine_similarity(mtp_vec, dec_vec).item()
        layer_cosines[l].append(cos)

print(f"  MTP[t] vs Decoder[l][t+1] — Cosine similarity (avg over t=0..T-2):")
print(f"  {'Layer':<8} {'Avg Cos':<10} {'Best Layer':<20}")
print(f"  {'------':<8} {'-------':<10} {'----------':<20}")

for l in range(num_decoder_hidden):
    avg_cos = sum(layer_cosines[l]) / len(layer_cosines[l])
    best_pos = max(range(len(layer_cosines[l])), key=lambda i: layer_cosines[l][i])
    best_val = layer_cosines[l][best_pos]
    label = f"layer_{l}" if l < num_decoder_hidden - 1 else "final_norm"
    marker = " ← BEST" if avg_cos == max(sum(layer_cosines[l]) / len(layer_cosines[l]) for l in range(num_decoder_hidden)) else ""
    print(f"  {label:<8} {avg_cos:.4f}{marker}")

# Find the decoder layer closest to MTP
all_avgs = [sum(layer_cosines[l]) / len(layer_cosines[l]) for l in range(num_decoder_hidden)]
best_layer = max(range(num_decoder_hidden), key=lambda l: all_avgs[l])
print(f"\n  >>> MTP hidden state is closest to Decoder hidden state at entry {best_layer}")
if best_layer < num_decoder_hidden - 1:
    print(f"  >>> Equivalent depth: MTP ~ Decoder Layer {best_layer}")
    equiv_depth_pct = best_layer / (num_decoder_hidden - 2) * 100
    print(f"  >>> MTP encodes tok[t+2] at roughly {equiv_depth_pct:.0f}% of decoder depth")

# Per-position breakdown
print(f"\n  --- Per-position breakdown (best decoder layer per t) ---")
for t in range(T - 1):
    cos_at_t = [layer_cosines[l][t] for l in range(num_decoder_hidden)]
    best_l = max(range(num_decoder_hidden), key=lambda l: cos_at_t[l])
    target_tok = tokenizer.decode(input_ids[0, t + 2].item()) if t + 2 < T else "?"
    print(f"    t={t}: MTP sees '{tokenizer.decode(input_ids[0,t+1].item())}',"
          f" predicts '{target_tok}' → best match Layer {best_l} (cos={cos_at_t[best_l]:.4f})")

print("\nDone.")
