from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


# ============================================================
# Basic comparison metrics (existing)
# ============================================================

def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(p_logits.float(), dim=-1)
    q = F.softmax(q_logits.float(), dim=-1)
    return (p * (p.log() - q.log())).sum(dim=-1).mean()


def js_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(p_logits.float(), dim=-1)
    q = F.softmax(q_logits.float(), dim=-1)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return (0.5 * (kl_pm + kl_qm)).mean()


def cosine_similarity(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = p_logits.float()
    q = q_logits.float()
    return F.cosine_similarity(p, q, dim=-1).mean()


def top_k_iou(p_logits: torch.Tensor, q_logits: torch.Tensor, k: int = 8) -> torch.Tensor:
    p_topk = p_logits.topk(k, dim=-1).indices
    q_topk = q_logits.topk(k, dim=-1).indices
    batch = p_logits.shape[0]
    ious = []
    for i in range(batch):
        p_set = set(p_topk[i].tolist())
        q_set = set(q_topk[i].tolist())
        intersection = p_set & q_set
        union = p_set | q_set
        iou = len(intersection) / len(union) if union else 1.0
        ious.append(iou)
    return torch.tensor(ious).mean()


def spearman_rank_corr(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = p_logits.float()
    q = q_logits.float()
    p_rank = p.argsort(dim=-1, descending=True).argsort(dim=-1).float()
    q_rank = q.argsort(dim=-1, descending=True).argsort(dim=-1).float()
    n = p.shape[-1]
    d = p_rank - q_rank
    rho = 1.0 - (6.0 * (d**2).sum(dim=-1)) / (n * (n**2 - 1.0))
    return rho.mean()


def top_k_hit_rate(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    k: int = 1,
) -> torch.Tensor:
    p_topk = p_logits.topk(k, dim=-1).indices
    q_top1 = q_logits.argmax(dim=-1, keepdim=True)
    hits = (p_topk == q_top1).any(dim=-1).float()
    return hits.mean()


def compute_metrics(
    actual_logits: torch.Tensor,
    mtp_pred_logits: torch.Tensor,
    top_k: int = 8,
) -> dict[str, float]:
    return {
        "kl_div": kl_divergence(actual_logits, mtp_pred_logits).item(),
        "js_div": js_divergence(actual_logits, mtp_pred_logits).item(),
        "cosine_sim": cosine_similarity(actual_logits, mtp_pred_logits).item(),
        f"top_{top_k}_iou": top_k_iou(actual_logits, mtp_pred_logits, k=top_k).item(),
        "spearman_rho": spearman_rank_corr(actual_logits, mtp_pred_logits).item(),
        "top_1_hit_rate": top_k_hit_rate(mtp_pred_logits, actual_logits, k=1).item(),
        "top_3_hit_rate": top_k_hit_rate(mtp_pred_logits, actual_logits, k=3).item(),
        "top_5_hit_rate": top_k_hit_rate(mtp_pred_logits, actual_logits, k=5).item(),
    }


# ============================================================
# 1) Router 置信度 / 熵分析
# ============================================================

def router_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=-1)
    return -(probs * probs.clamp(min=1e-10).log()).sum(dim=-1)


def router_confidence(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=-1)
    return probs.max(dim=-1).values


def compute_entropy_confidence_metrics(
    mtp_logits: torch.Tensor,
    actual_logits: torch.Tensor,
) -> dict[str, float]:
    mtp_ent = router_entropy(mtp_logits)
    act_ent = router_entropy(actual_logits)
    mtp_conf = router_confidence(mtp_logits)
    act_conf = router_confidence(actual_logits)
    return {
        "mtp_entropy_mean": mtp_ent.mean().item(),
        "actual_entropy_mean": act_ent.mean().item(),
        "entropy_diff_mean": (mtp_ent - act_ent).mean().item(),
        "mtp_confidence_mean": mtp_conf.mean().item(),
        "actual_confidence_mean": act_conf.mean().item(),
        "confidence_diff_mean": (mtp_conf - act_conf).mean().item(),
    }


# ============================================================
# 2) Expert 命中分析
# ============================================================

def compute_expert_overlap_metrics(
    mtp_logits: torch.Tensor,
    actual_logits: torch.Tensor,
    k: int = 8,
) -> dict[str, float]:
    mtp_topk = mtp_logits.topk(k, dim=-1).indices
    actual_topk = actual_logits.topk(k, dim=-1).indices
    n = mtp_logits.shape[0]

    overlap_counts = []
    for i in range(n):
        m_set = set(mtp_topk[i].tolist())
        a_set = set(actual_topk[i].tolist())
        overlap_counts.append(len(m_set & a_set))
    oc = torch.tensor(overlap_counts, dtype=torch.float)

    return {
        "avg_overlap_count": oc.mean().item(),
        "max_overlap_count": oc.max().item(),
        "min_overlap_count": oc.min().item(),
        "overlap_ratio": (oc / k).mean().item(),
        "zero_overlap_ratio": (oc == 0).float().mean().item(),
        "full_overlap_ratio": (oc == k).float().mean().item(),
    }


# ============================================================
# 3) 逐层 Router 对比
# ============================================================

def compute_layerwise_metrics(
    layer_routers: torch.Tensor,
    mtp_router: torch.Tensor,
    k: int = 8,
) -> dict[str, Any]:
    num_layers = layer_routers.shape[0]
    seq_len = layer_routers.shape[1]

    cos_sims = []
    ious = []
    for l in range(num_layers):
        layer = layer_routers[l]  # [seq_len, num_experts]
        cos = F.cosine_similarity(layer.float(), mtp_router.float(), dim=-1).mean().item()
        # top_k_iou expects [batch, experts] — pass [seq_len, experts] directly
        iou = top_k_iou(layer, mtp_router, k=k).item()
        cos_sims.append(cos)
        ious.append(iou)
    return {
        "layerwise_cosine": cos_sims,
        "layerwise_topk_iou": ious,
        "layerwise_cosine_mean": sum(cos_sims) / len(cos_sims),
        "layerwise_topk_iou_mean": sum(ious) / len(ious),
        "last_layer_cosine": cos_sims[-1] if cos_sims else 0,
        "last_layer_iou": ious[-1] if ious else 0,
        "layerwise_cosine_improvement": cos_sims[-1] - cos_sims[0] if len(cos_sims) >= 2 else 0,
    }


# ============================================================
# 4) 输出 logits 对比 (LM Head vs MTP Router 置信度关联)
# ============================================================

def compute_output_logits_corr_metrics(
    lm_logits: torch.Tensor,
    mtp_router_logits: torch.Tensor,
) -> dict[str, float]:
    lm_probs = F.softmax(lm_logits.float(), dim=-1)
    lm_confidence = lm_probs.max(dim=-1).values

    mtp_probs = F.softmax(mtp_router_logits.float(), dim=-1)
    mtp_confidence = mtp_probs.max(dim=-1).values

    stack = torch.stack([lm_confidence, mtp_confidence])
    corr = torch.corrcoef(stack)[0, 1].item() if lm_confidence.numel() > 1 else 0.0

    return {
        "lm_mtp_confidence_corr": corr,
        "lm_confidence_mean": lm_confidence.mean().item(),
        "lm_confidence_std": lm_confidence.std().item(),
    }


# ============================================================
# 5) 位置趋势分析
# ============================================================

def compute_position_trends(
    step_metrics: list[dict],
) -> dict[str, Any]:
    if not step_metrics:
        return {}

    trends = {}
    keys = step_metrics[0].keys()
    for key in keys:
        vals = [s[key] for s in step_metrics]
        mid = max(len(vals) // 2, 1)
        trends[key] = {
            "per_position": vals,
            "first_half_avg": sum(vals[:mid]) / mid,
            "second_half_avg": sum(vals[mid:]) / max(len(vals) - mid, 1),
            "trend_direction": "up" if len(vals) >= 2 and vals[-1] > vals[0] else "down",
        }
    return trends


# ============================================================
# 6) Token 贪心解码准确率
# ============================================================

def compute_token_accuracy(
    lm_logits: torch.Tensor,
    mtp_token_logits: torch.Tensor,
) -> dict[str, float]:
    if mtp_token_logits is None:
        return {"token_accuracy": None, "token_exact_match": None}

    # Greedy decode
    lm_tokens = lm_logits.argmax(dim=-1)          # [T]
    mtp_tokens = mtp_token_logits.argmax(dim=-1)  # [T]

    correct = (lm_tokens == mtp_tokens)
    accuracy = correct.float().mean().item()

    return {
        "token_accuracy": accuracy,
        "token_correct_count": correct.sum().item(),
        "token_total_count": lm_tokens.shape[0],
        "token_exact_match": 1.0 if accuracy == 1.0 else 0.0,
        "token_generations_equal": accuracy,
    }


# ============================================================
# 7) LM head vs MTP head 完整 logits 分布对比（对齐 target token）
# ============================================================
# 定理：Language Models are Injective (Sapienza 2025)
# → lm_head(mtp_hidden[t]) 预测 token[t+2]
# → lm_head(decoder_hidden[t+1]) 也预测 token[t+2]
# 如果模型的 hidden state 是单射的，则这两个分布应该高度一致

def compute_lm_mtp_logit_alignment(
    lm_logits: torch.Tensor,     # [T, V], LM head logits predicting tok[t+1]
    mtp_logits: torch.Tensor,    # [T, V], MTP head logits predicting tok[t+2]
) -> dict[str, float]:
    if mtp_logits is None:
        return {"lm_mtp_logit_alignment": None}

    # Align by target token: LM at t+1 vs MTP at t both predict tok[t+2]
    # lm_logits has shape [T, V] where T = full_len-1
    # mtp_logits has shape [T, V] where T = full_len-1
    # LM at position t predicts tok[t+1], so position 1..T predicts tok[2]..tok[T+1]
    # MTP at position t predicts tok[t+2], so position 0..T-1 predicts tok[2]..tok[T+1]
    lm_aligned = lm_logits[1:, :]     # [T-1, V]
    mtp_aligned = mtp_logits[:-1, :]  # [T-1, V]

    cosine = F.cosine_similarity(lm_aligned.float(), mtp_aligned.float(), dim=-1).mean().item()
    kl = kl_divergence(lm_aligned, mtp_aligned).item()
    js = js_divergence(lm_aligned, mtp_aligned).item()
    spearman = spearman_rank_corr(lm_aligned, mtp_aligned).item()
    iou_k = top_k_iou(lm_aligned, mtp_aligned, k=8).item()

    # Per-token softmax dot product (prob distribution overlap)
    lm_prob = F.softmax(lm_aligned.float(), dim=-1)
    mtp_prob = F.softmax(mtp_aligned.float(), dim=-1)
    prob_dot = (lm_prob * mtp_prob).sum(dim=-1).mean().item()

    # L2 distance of logits
    l2_dist = (lm_aligned.float() - mtp_aligned.float()).norm(dim=-1).mean().item()

    return {
        "lm_mtp_logit_cosine": cosine,
        "lm_mtp_logit_kl": kl,
        "lm_mtp_logit_js": js,
        "lm_mtp_logit_spearman": spearman,
        "lm_mtp_logit_top8_iou": iou_k,
        "lm_mtp_logit_prob_dot": prob_dot,
        "lm_mtp_logit_l2": l2_dist,
    }


# ============================================================
# 聚合 (扩展支持嵌套指标)
# ============================================================

def aggregate_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    agg = {}
    skip_keys = {"prompt", "generated_text", "step_metrics", "entropy_confidence",
                  "expert_overlap", "layerwise", "output_corr", "position_trends",
                  "token_accuracy", "logit_alignment"}
    for key in results[0]:
        if key in skip_keys:
            continue
        if isinstance(results[0][key], (int, float)):
            vals = [r[key] for r in results]
            agg[f"avg_{key}"] = sum(vals) / len(vals)
            agg[f"min_{key}"] = min(vals)
            agg[f"max_{key}"] = max(vals)
    return agg
