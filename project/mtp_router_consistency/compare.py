from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


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


def aggregate_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    agg = {}
    for key in results[0]:
        if key in ("prompt", "generated_text", "step_metrics"):
            continue
        vals = [r[key] for r in results]
        agg[f"avg_{key}"] = sum(vals) / len(vals)
        agg[f"min_{key}"] = min(vals)
        agg[f"max_{key}"] = max(vals)
    return agg
