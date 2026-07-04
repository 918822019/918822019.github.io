# MTP Router Consistency Analysis - Findings Summary

## Goal
Investigate whether MTP (Multi-Token Prediction) layer's routing decisions can be used to predict decoder's expert loading in CPU offload scenarios.

## Model
- **Architecture**: BailingMoeV2 (Ling-mini-base-2.0)
- **Parameters**: 12B
- **MoE**: 256 experts per layer, top-8 routing
- **Layers**: 19 MoE layers + 1 MTP layer (20 decoder layers total, 1 dense layer)
- **Hidden size**: 2048
- **Vocab**: 157184

---

## Key Findings

### 1. MTP Layer Architecture
MTP layer has its **own 256 experts**, completely independent from decoder's 256 experts per layer.

Decoder Layer (×19):
  input → Self-Attention → MoE Block (256 experts) → output

MTP Layer (×1):
  [shifted tok[t+1] emb, decoder_last_hidden[t]] → eh_proj → Self-Attention → MoE Block (256 experts) → lm_head → tok[t+2]
```

### 2. Token Prediction Accuracy
| Metric | Sample 1 (def fibonacci(n):) | Sample 2 (def merge_sort(arr):) |
|--------|------------------------------|----------------------------------|
| LM→tok[t+1] accuracy | 75.00% (3/4) | 25.00% (1/4) |
| MTP→tok[t+2] accuracy | 100.00% (3/3) | 33.33% (1/3) |
| MTP-LM gap | +25.00% | +8.33% |

**Insight**: MTP sees tok[t+1]'s embedding (via internal roll), giving it an information advantage.

### 3. LM Head vs MTP Head Logit Distribution
When aligned by target token (both predict tok[t+2]):
| Metric | Sample 1 | Sample 2 |
|--------|-----------|-----------|
| Cosine similarity | 0.9715 | 0.9650 |
| Spearman | 0.8342 | 0.7867 |
| KL divergence | 0.3144 | 0.4777 |
| JS divergence | 0.0833 | 0.1050 |
| Top-8 IoU | 0.5362 | 0.5111 |
| Prob dot product | 0.3631 | 0.4414 |

**Insight**: LM and MTP logits are very similar (Cosine ~0.97), consistent with injectivity theory.

### 4. Router Consistency
MTP router vs Decoder router (last layer):
| Metric | Value |
|--------|-------|
| Cosine similarity | ~0.87 |
| Top-8 IoU | ~0.02 |
| Hit Rate (top-1/3/5) | 0.0000 |

**Insight**: MTP router predicts similar direction but selects completely different experts (IoU ~0.02).

### 5. Token → Expert Mapping Stability
| Token | Occurrences | Layer 0 (shallow) | Layer 9 (middle) | Layer 18 (deep) |
|-------|-------------|-------------------|-------------------|------------------|
| `' x'` | 3 | **1.0000** | 0.5030 | 0.5219 |
| `' +'` | 2 | **1.0000** | 0.7778 | 0.6000 |
| `'(x'` | 2 | 0.6000 | 0.2308 | 0.4545 |

**Insight**:
- **Shallow layers (Layer 0)**: Routing is token-deterministic (IoU=1.0)
- **Middle layers (Layer 9)**: Partially context-dependent (IoU~0.5-0.78)
- **Deep layers (Layer 18)**: Highly context-dependent (IoU~0.45-0.60)

### 6. MTP Hidden State vs Decoder Hidden State
MTP[t] vs Decoder[layer][t+1] Cosine similarity:
| Layer | Avg Cosine |
|-------|------------|
| layer_0 (embeds) | 0.0225 |
| layer_5 | 0.1800 |
| layer_10 (middle) | 0.2437 |
| layer_15 | 0.3490 |
| layer_19 | 0.3751 |
| layer_20 (decoder output) | **0.8278** |

**Insight**: MTP's single layer is almost equivalent to all 20 decoder layers combined. MTP hidden state ≈ decoder's final output (Cosine ~0.83).

### 7. Speculative Decoding Savings
| N draft tokens | Sequential loads | Speculative loads | Savings |
|----------------|------------------|-------------------|---------|
| N=2 | 304 | 296 | 2.6% |
| N=4 | 608 | 419 | 31.1% |
| N=8 | 1216 | 714 | 41.3% |
| N=16 | 2432 | 924 | 62.0% |

**Insight**: Routing diversity is very high. Adjacent positions activate almost completely disjoint expert sets (N=2: 15.6/16 possible experts). Real savings come from cross-position reuse at larger N.

---

## Core Question
Can MTP's routing decision be used to pre-fetch decoder's expert loading in CPU offload?

### Current Evidence
1. **Token alone cannot predict deep layer routing** (context-dependent)
2. **MTP hidden state ≈ decoder's final output** (cos=0.83)
3. **Injectivity theory**: Input sequence → hidden state is injective, so routing is uniquely determined by full context, not just token
4. **MTP router vs Decoder router IoU ~0.02**: Completely different experts

### Open Questions
- Can MTP's routing inform decoder's routing?
- Can MTP hidden state be used to predict decoder's expert selection?
- What is the cost-benefit of MTP's own expert loading vs savings from pre-fetching?

---

## Files
- `decoder.py`: Single forward pass, MTP data extraction
- `compare.py`: All metric functions (8 groups)
- `main.py`: Pipeline + report generation
- `analyze_routing.py`: Token→expert mapping analysis
- `verify_spec.py`: Speculative decoding simulation
