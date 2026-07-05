"""MTP 路由一致性分析 - 核心管线。

合并自: config, model_utils, decoder, compare, main。
"""
from __future__ import annotations

"""MTP 路由一致性分析管线的配置。

所有模型和分析参数在此定义。
"""
from dataclasses import dataclass, field
from pathlib import Path

import torch

@dataclass
class Config:
    # 模型
    model_path: str = "data/models/Ling-mini-base-2.0"
    device: str = "cpu"
    torch_dtype: str = "bfloat16"

    # 生成（当前未使用，仅单次前向）
    max_new_tokens: int = 4
    max_prompt_len: int = 64

    # MoE 架构
    num_experts: int = 256
    num_experts_per_tok: int = 8
    num_hidden_layers: int = 20
    hidden_size: int = 2048

    # 输出
    output_dir: str = str(Path(__file__).resolve().parent.parent / "output")

    # 测试提示词
    prompts: list = field(default_factory=lambda: [
        "def fibonacci(n):",
        "def merge_sort(arr):",
    ])

"""模型加载工具。

处理路径解析、CPU 卸载的模型加载、BailingMoeV2 的分词器设置。
"""
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

def _resolve_path(model_path: str) -> str:
    p = Path(model_path)
    if not p.is_absolute():
        root = Path(__file__).resolve().parents[2]
        resolved = root / p
        if resolved.exists():
            return str(resolved)
    return model_path

def load_model_and_tokenizer(
    model_path: str,
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """加载模型和分词器（CPU 卸载）。

    使用 device_map='cpu' 强制 CPU 加载（6GB VRAM 稳定）。
    模型约 120 亿参数，每层 256 专家，top-8 路由。

    返回:
        tuple: (model, tokenizer, model_device)
    """
    model_path = _resolve_path(model_path)
    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading model from %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    model_device = model.model.word_embeddings.weight.device
    logger.info("Model device: %s", model_device)
    return model, tokenizer, model_device

"""单次前向的 MTP 数据提取。

从模型输出中提取路由 logits、LM head logits 和 MTP head logits。
按目标 token 对齐位置。

关键对齐:
  - LM head:     lm_logits[:, t, :]     预测 token[t+1]
  - MTP head:    mtp_logits[:, t, :]    预测 token[t+2]
  - LM 对齐:    lm_logits[:, 1:, :]    预测 token[t+2]  (与 MTP 对齐)
  - Decoder 路由 t+1  对应  MTP 路由 t  (都处理 token[t+1])
"""
import logging

import torch

logger = logging.getLogger(__name__)

@torch.no_grad()
def process_with_mtp(
    model,
    tokenizer,
    prompt_ids: torch.LongTensor,
    device: str = "cpu",
) -> dict:
    """单次前向，提取 MTP 和路由 logits。

    参数:
        model: BailingMoeV2ForCausalLM (CPU 加载)
        tokenizer: 对应的分词器
        prompt_ids: 分词后的输入 [1, seq_len]
        device: 目标设备

    返回:
        包含 output_ids, decoder_router, mtp_router, lm_logits,
             mtp_token_logits, lm_logits_for_mtp, layer_routers,
             ground_truth_lm, ground_truth_mtp 的字典
    """
    model.eval()
    prompt_ids = prompt_ids.to(device)

    outputs = model(
        input_ids=prompt_ids,
        output_router_logits=True,
        use_cache=False,
        return_dict=True,
    )

    # all_router 是每层 MoE + MTP 的 (router_logits, topk_idx) 元组
    all_router = outputs.router_logits
    num_decoder_layers = len(all_router) - 1  # 排除 MTP

    # 最后一层 Decoder 路由 vs MTP 路由
    decoder_router = all_router[-2][0]  # [1, seq_len, E]
    mtp_router = all_router[-1][0]      # [1, seq_len, E]

    # 所有 Decoder 层的路由堆叠，用于逐层分析
    layer_routers = torch.stack(
        [all_router[i][0].squeeze(0) for i in range(num_decoder_layers)]
    )  # [num_decoder_layers, seq_len, E]

    full_len = prompt_ids.shape[1]

    # 对齐: MTP 在位置 t 预测 token[t+1] 的路由
    mtp_pred = mtp_router[:, :full_len - 1, :].squeeze(0)
    actual = decoder_router[:, 1:, :].squeeze(0)

    # LM head 预测下一个 token: lm_logits[:, t, :] 预测 token[t+1]
    lm_logits = outputs.logits[:, :-1, :].squeeze(0)

    # MTP head 预测 token[t+2] (因为内部 input_ids 滚动)
    mtp_logits_raw = getattr(outputs, "mtp_logits", None)
    mtp_token_logits = None
    lm_logits_for_mtp = None
    if mtp_logits_raw is not None and len(mtp_logits_raw) > 0:
        mtp_token_logits = mtp_logits_raw[0][:, :-1, :].squeeze(0)
        lm_logits_for_mtp = outputs.logits[:, 1:, :].squeeze(0)

    # 用于准确性验证的真实 token
    ground_truth_lm = prompt_ids[:, 1:].squeeze(0)
    ground_truth_mtp = prompt_ids[:, 2:].squeeze(0) if prompt_ids.shape[1] > 2 else None

    return {
        "output_ids": prompt_ids,
        "num_decoder_layers": num_decoder_layers,
        "decoder_router": actual,
        "mtp_router": mtp_pred,
        "lm_logits": lm_logits,
        "mtp_token_logits": mtp_token_logits,
        "lm_logits_for_mtp": lm_logits_for_mtp,
        "layer_routers": layer_routers,
        "ground_truth_lm": ground_truth_lm,
        "ground_truth_mtp": ground_truth_mtp,
    }

"""MTP vs Decoder 路由一致性分析的指标函数。

分组:
1-3: 基础路由比较（KL、JS、余弦、IoU、Spearman、命中率）
4:   熵 / 置信度
5:   专家重叠
6:   逐层比较
7:   LM-MTP logits 相关性
8:   位置趋势
9:   Token 准确率
10:  Logits 对齐（可注入性测试）
11:  真实 token 准确率
"""
from typing import Any

import torch
import torch.nn.functional as F

# ============================================================
# 基础比较指标
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
    p_logits: torch.Tensor, q_logits: torch.Tensor, k: int = 1,
) -> torch.Tensor:
    p_topk = p_logits.topk(k, dim=-1).indices
    q_top1 = q_logits.argmax(dim=-1, keepdim=True)
    hits = (p_topk == q_top1).any(dim=-1).float()
    return hits.mean()

# ============================================================
# 1) 逐位置路由比较
# ============================================================

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
# 2) 路由熵 / 置信度
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
# 3) 专家重叠分析
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
# 4) 逐层 MTP vs Decoder 路由比较
# ============================================================

def compute_layerwise_metrics(
    layer_routers: torch.Tensor,
    mtp_router: torch.Tensor,
    k: int = 8,
) -> dict[str, Any]:
    num_layers = layer_routers.shape[0]

    cos_sims = []
    ious = []
    for l in range(num_layers):
        layer = layer_routers[l]
        cos = F.cosine_similarity(layer.float(), mtp_router.float(), dim=-1).mean().item()
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
# 5) LM head vs MTP 路由置信度相关
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
# 6) 位置趋势
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
# 7) Token 贪心解码准确率（LM vs MTP head）
# ============================================================

def compute_token_accuracy(
    lm_logits: torch.Tensor,
    mtp_token_logits: torch.Tensor,
) -> dict[str, float]:
    """比较 LM head 和 MTP head 的贪心 argmax（同一目标 token）。"""
    if mtp_token_logits is None:
        return {"token_accuracy": None, "token_exact_match": None}

    lm_tokens = lm_logits.argmax(dim=-1)
    mtp_tokens = mtp_token_logits.argmax(dim=-1)

    correct = (lm_tokens == mtp_tokens)
    accuracy = correct.float().mean().item()

    return {
        "token_accuracy": accuracy,
        "token_correct_count": correct.sum().item(),
        "token_total_count": lm_tokens.shape[0],
        "token_exact_match": 1.0 if accuracy == 1.0 else 0.0,
    }

# ============================================================
# 8) LM vs MTP head 完整 logits 分布对齐
# (= 可注入性测试：不同路由路径 → 相同输出)
# ============================================================

def compute_lm_mtp_logit_alignment(
    lm_logits: torch.Tensor,
    mtp_logits: torch.Tensor,
) -> dict[str, float]:
    """按目标 token 对齐后比较完整 logits 分布。

    lm_logits[:, t+1, :] 和 mtp_logits[:, t, :] 都预测 tok[t+2]。
    """
    if mtp_logits is None:
        return {"lm_mtp_logit_alignment": None}

    lm_aligned = lm_logits[1:, :]
    mtp_aligned = mtp_logits[:-1, :]

    cosine = F.cosine_similarity(lm_aligned.float(), mtp_aligned.float(), dim=-1).mean().item()
    kl = kl_divergence(lm_aligned, mtp_aligned).item()
    js = js_divergence(lm_aligned, mtp_aligned).item()
    spearman = spearman_rank_corr(lm_aligned, mtp_aligned).item()
    iou_k = top_k_iou(lm_aligned, mtp_aligned, k=8).item()

    lm_prob = F.softmax(lm_aligned.float(), dim=-1)
    mtp_prob = F.softmax(mtp_aligned.float(), dim=-1)
    prob_dot = (lm_prob * mtp_prob).sum(dim=-1).mean().item()

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
# 9) 真实 token 准确率
# ============================================================

def compute_accuracy_vs_ground_truth(
    lm_logits: torch.Tensor | None,
    mtp_logits: torch.Tensor | None,
    gt_lm: torch.Tensor | None,
    gt_mtp: torch.Tensor | None,
) -> dict[str, float]:
    """检查 LM head / MTP head 的预测是否匹配实际输入 token。"""
    result = {}

    if lm_logits is not None and gt_lm is not None:
        lm_pred = lm_logits.argmax(dim=-1)
        lm_correct = (lm_pred == gt_lm)
        result["lm_gt_accuracy"] = lm_correct.float().mean().item()
        result["lm_gt_correct"] = lm_correct.sum().item()
        result["lm_gt_total"] = gt_lm.shape[0]

    if mtp_logits is not None and gt_mtp is not None:
        mtp_aligned = mtp_logits[:-1, :]
        mtp_pred = mtp_aligned.argmax(dim=-1)
        mtp_correct = (mtp_pred == gt_mtp)
        result["mtp_gt_accuracy"] = mtp_correct.float().mean().item()
        result["mtp_gt_correct"] = mtp_correct.sum().item()
        result["mtp_gt_total"] = gt_mtp.shape[0]
        result["mtp_gt_gap"] = result.get("mtp_gt_accuracy", 0) - result.get("lm_gt_accuracy", 0)

    return result

# ============================================================
# 汇总
# ============================================================

def aggregate_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """汇总各样本的数值指标（平均/最小/最大）。"""
    agg = {}
    skip_keys = {"prompt", "generated_text", "step_metrics", "entropy_confidence",
                  "expert_overlap", "layerwise", "output_corr", "position_trends",
                  "token_accuracy", "logit_alignment", "gt_accuracy"}
    for key in results[0]:
        if key in skip_keys:
            continue
        if isinstance(results[0][key], (int, float)):
            vals = [r[key] for r in results]
            agg[f"avg_{key}"] = sum(vals) / len(vals)
            agg[f"min_{key}"] = min(vals)
            agg[f"max_{key}"] = max(vals)
    return agg

"""MTP 路由一致性分析 - 主流水线。

加载模型，对每个提示运行所有指标组的分析，
生成 Markdown 报告和 JSON 结果。
"""
import json
import logging
import sys
import time
from pathlib import Path

import torch



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

def run_single_prompt(
    model,
    tokenizer,
    prompt: str,
    cfg: Config,
    device: str,
) -> dict:
    """分析单个提示的 MTP vs Decoder 路由。"""
    logger.info("Processing prompt: %s", prompt[:60])

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_prompt_len)
    input_ids = inputs.input_ids

    result = process_with_mtp(model=model, tokenizer=tokenizer, prompt_ids=input_ids, device=device)

    decoder_router = result["decoder_router"]   # [T, E]
    mtp_router = result["mtp_router"]           # [T, E]
    lm_logits = result["lm_logits"]             # [T, V]
    mtp_token_logits = result["mtp_token_logits"]  # [T, V] 或 None
    layer_routers = result["layer_routers"]     # [L, seq_len, E]
    gt_lm = result.get("ground_truth_lm")
    gt_mtp = result.get("ground_truth_mtp")
    T = decoder_router.shape[0]

    # 逐位置路由比较
    step_metrics = []
    for t in range(T):
        m = compute_metrics(
            decoder_router[t].unsqueeze(0), mtp_router[t].unsqueeze(0),
            top_k=cfg.num_experts_per_tok,
        )
        step_metrics.append(m)

    avg_routing_metrics = {}
    if step_metrics:
        for key in step_metrics[0]:
            avg_routing_metrics[key] = sum(s[key] for s in step_metrics) / len(step_metrics)

    # 1) 路由熵 / 置信度
    entropy_conf = compute_entropy_confidence_metrics(mtp_router, decoder_router)

    # 2) 专家重叠
    expert_overlap = compute_expert_overlap_metrics(
        mtp_router, decoder_router, k=cfg.num_experts_per_tok,
    )

    # 3) 逐层路由比较
    layerwise = compute_layerwise_metrics(
        layer_routers[:, :T, :], mtp_router, k=cfg.num_experts_per_tok,
    )

    # 4) LM head vs MTP 路由置信度相关
    output_corr = compute_output_logits_corr_metrics(lm_logits, mtp_router)

    # 5) 位置趋势
    position_trends = compute_position_trends(step_metrics)

    # 6) Token 贪心解码准确率（LM vs MTP head）
    lm_logits_mtp = result.get("lm_logits_for_mtp")
    token_acc = compute_token_accuracy(lm_logits_mtp, mtp_token_logits)

    # 7) LM vs MTP head 完整 logits 分布对齐
    logit_alignment = compute_lm_mtp_logit_alignment(lm_logits, mtp_token_logits)

    # 8) 真实 token 准确率
    gt_accuracy = compute_accuracy_vs_ground_truth(lm_logits, mtp_token_logits, gt_lm, gt_mtp)

    generated_text = tokenizer.decode(result["output_ids"][0], skip_special_tokens=True)

    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "num_generated": 0,
        "num_comparisons": T,
        "avg_routing_metrics": avg_routing_metrics,
        "step_metrics": step_metrics,
        "entropy_confidence": entropy_conf,
        "expert_overlap": expert_overlap,
        "layerwise": layerwise,
        "output_corr": output_corr,
        "position_trends": position_trends,
        "token_accuracy": token_acc,
        "logit_alignment": logit_alignment,
        "gt_accuracy": gt_accuracy,
    }

def _fmt(v, dec=4):
    if isinstance(v, float):
        return f"{v:.{dec}f}"
    return str(v)

def _table_row(key, val, dec=4):
    return f"| {key} | {_fmt(val, dec)} |"

def generate_report(results: list[dict], agg: dict, elapsed: float, cfg: Config) -> str:
    """从分析结果生成完整的 Markdown 报告。"""
    lines = []
    lines.append("# MTP Multi-Dimension Analysis Report")
    lines.append("")
    lines.append(f"**Model**: {cfg.model_path}")
    lines.append(f"**Device**: {cfg.device}")
    lines.append(f"**Analysis**: Single forward pass (no generation)")
    lines.append(f"**Time**: {elapsed:.1f}s")
    lines.append(f"**Samples**: {len(results)}")
    lines.append("")

    # 0) 路由一致性指标
    lines.append("## 0) Router Consistency")
    lines.append("")
    lines.append("| Metric | Mean | Min | Max |")
    lines.append("|------|------|--------|--------|")
    for key in sorted(agg.keys()):
        if key.startswith("avg_"):
            base = key[4:]
            min_k = f"min_{base}"
            max_k = f"max_{base}"
            lines.append(f"| {base} | {agg[key]:.4f} | {agg.get(min_k, 0):.4f} | {agg.get(max_k, 0):.4f} |")
    lines.append("")

    # 1) 熵 & 置信度
    lines.append("## 1) Router Entropy / Confidence")
    lines.append("")
    keys_describe = {
        "mtp_entropy_mean": "MTP Router Entropy (higher = less certain)",
        "actual_entropy_mean": "Decoder Router Entropy",
        "entropy_diff_mean": "Entropy Diff (MTP - Decoder)",
        "mtp_confidence_mean": "MTP Router Confidence (top-1 prob)",
        "actual_confidence_mean": "Decoder Router Confidence",
        "confidence_diff_mean": "Confidence Diff (MTP - Decoder)",
    }
    for i, r in enumerate(results):
        lines.append(f"### Sample {i+1}")
        ec = r.get("entropy_confidence", {})
        for k, desc in keys_describe.items():
            if k in ec:
                lines.append(_table_row(desc, ec[k]))
        lines.append("")

    # 2) 专家重叠
    lines.append("## 2) Expert Overlap Analysis")
    lines.append("")
    eo_keys = {
        "avg_overlap_count": "Avg overlapping experts (top-k)",
        "max_overlap_count": "Max overlapping experts",
        "min_overlap_count": "Min overlapping experts",
        "overlap_ratio": "Overlap ratio (overlap/k)",
        "zero_overlap_ratio": "Zero overlap ratio",
        "full_overlap_ratio": "Full overlap ratio (k/k)",
    }
    for i, r in enumerate(results):
        lines.append(f"### Sample {i+1}")
        eo = r.get("expert_overlap", {})
        for k, desc in eo_keys.items():
            if k in eo:
                lines.append(_table_row(desc, eo[k]))
        lines.append("")

    # 3) 逐层比较
    lines.append("## 3) Layer-wise Router Comparison (MTP vs Decoder Layers)")
    lines.append("")
    lw_keys = {
        "layerwise_cosine_mean": "Layer-wise Mean Cosine",
        "layerwise_topk_iou_mean": "Layer-wise Mean Top-K IoU",
        "last_layer_cosine": "Last Layer Cosine",
        "last_layer_iou": "Last Layer Top-K IoU",
        "layerwise_cosine_improvement": "Last Cosine - First Cosine",
    }
    for i, r in enumerate(results):
        lines.append(f"### Sample {i+1}")
        lw = r.get("layerwise", {})
        for k, desc in lw_keys.items():
            if k in lw:
                lines.append(_table_row(desc, lw[k]))
        lines.append("")
        cos_list = lw.get("layerwise_cosine", [])
        if cos_list:
            lines.append("Layer-wise Cosine (shallow→deep):")
            lines.append("")
            lines.append(f"`{', '.join(f'{v:.3f}' for v in cos_list[:3])} ... {', '.join(f'{v:.3f}' for v in cos_list[-3:])}`")
            lines.append("")

    # 4) Logits 相关
    lines.append("## 4) Logit Correlation (LM Head vs MTP Router)")
    lines.append("")
    oc_keys = {
        "lm_mtp_confidence_corr": "LM confidence vs MTP confidence correlation",
        "lm_confidence_mean": "LM Head mean confidence",
        "lm_confidence_std": "LM Head confidence std",
    }
    for i, r in enumerate(results):
        lines.append(f"### Sample {i+1}")
        oc = r.get("output_corr", {})
        for k, desc in oc_keys.items():
            if k in oc:
                lines.append(_table_row(desc, oc[k]))
        lines.append("")

    # 5) 位置趋势
    lines.append("## 5) Position Trends")
    lines.append("")
    for i, r in enumerate(results):
        lines.append(f"### Sample {i+1}")
        pt = r.get("position_trends", {})
        if pt:
            lines.append("| Metric | First Half | Second Half | Trend |")
            lines.append("|------|----------|----------|------|")
            for key, trend in pt.items():
                lines.append(
                    f"| {key} | {trend['first_half_avg']:.4f} | "
                    f"{trend['second_half_avg']:.4f} | {trend['trend_direction']} |")
        else:
            lines.append("*No trend data*")
        lines.append("")

    # 6) Token 准确率
    lines.append("## 6) Token Greedy Decoding Accuracy")
    lines.append("")
    lines.append("Comparing LM head and MTP head greedy decoding (same target token).")
    lines.append("")
    lines.append("| Sample | Accuracy | Correct/Total |")
    lines.append("|------|--------|-------------|")
    for i, r in enumerate(results):
        ta = r.get("token_accuracy", {})
        acc = ta.get("token_accuracy")
        if acc is not None:
            lines.append(f"| Sample {i+1} | {acc:.2%} | {ta.get('token_correct_count', 0)}/{ta.get('token_total_count', 0)} |")
        else:
            lines.append(f"| Sample {i+1} | N/A | - |")
    lines.append("")

    # 7) Logits 对齐（可注入性）
    lines.append("## 7) LM head vs MTP head Full Logit Alignment")
    lines.append("")
    lines.append("Aligned by target token: `lm_logits[:, t+1, :]` vs `mtp_logits[:, t, :]`")
    lines.append("")
    lines.append("| Metric | Sample 1 | Sample 2 |")
    lines.append("|------|-------|-------|")
    la_keys = [
        ("lm_mtp_logit_cosine", "Cosine Similarity"),
        ("lm_mtp_logit_kl", "KL Divergence"),
        ("lm_mtp_logit_js", "JS Divergence"),
        ("lm_mtp_logit_spearman", "Spearman Correlation"),
        ("lm_mtp_logit_top8_iou", "Top-8 IoU"),
        ("lm_mtp_logit_prob_dot", "Probability Dot Product"),
        ("lm_mtp_logit_l2", "Logits L2 Distance"),
    ]
    for key, desc in la_keys:
        v1 = results[0].get("logit_alignment", {}).get(key, "N/A") if len(results) > 0 else "N/A"
        v2 = results[1].get("logit_alignment", {}).get(key, "N/A") if len(results) > 1 else "N/A"
        v1s = f"{v1:.4f}" if isinstance(v1, float) else str(v1)
        v2s = f"{v2:.4f}" if isinstance(v2, float) else str(v2)
        lines.append(f"| {desc} | {v1s} | {v2s} |")
    lines.append("")

    # 8) 真实 token 准确率
    lines.append("## 8) Ground Truth Token Accuracy")
    lines.append("")
    lines.append("| Sample | LM→tok[t+1] | MTP→tok[t+2] | MTP-LM Gap |")
    lines.append("|------|-------------------|-------------------|-------------|")
    for i, r in enumerate(results):
        ga = r.get("gt_accuracy", {})
        lma = f"{ga.get('lm_gt_accuracy', 0):.2%}" if ga.get('lm_gt_accuracy') is not None else "N/A"
        mtpa = f"{ga.get('mtp_gt_accuracy', 0):.2%}" if ga.get('mtp_gt_accuracy') is not None else "N/A"
        gap = f"{ga.get('mtp_gt_gap', 0):+.2%}" if ga.get('mtp_gt_gap') is not None else "N/A"
        lines.append(f"| Sample {i+1} | {lma} | {mtpa} | {gap} |")
    lines.append("")

    # 样本详情
    lines.append("## Sample Details")
    lines.append("")
    for i, r in enumerate(results):
        lines.append(f"### Sample {i+1}")
        lines.append("")
        lines.append(f"**Prompt**: `{r['prompt'][:80]}`")
        lines.append(f"**Positions**: {r['num_comparisons']}")
        lines.append("")
        if r.get("avg_routing_metrics"):
            lines.append("| Metric | Value |")
            lines.append("|----------|-----|")
            for k, v in r["avg_routing_metrics"].items():
                lines.append(_table_row(k, v))
        lines.append("")

    return "\n".join(lines)

def main():
    cfg = Config()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = getattr(torch, cfg.torch_dtype, torch.bfloat16)
    model, tokenizer, model_device = load_model_and_tokenizer(
        cfg.model_path, device=cfg.device, torch_dtype=torch_dtype,
    )

    t0 = time.time()
    results = []
    for prompt in cfg.prompts:
        try:
            r = run_single_prompt(model, tokenizer, prompt, cfg, str(model_device))
            results.append(r)
        except Exception as e:
            logger.exception("Failed on prompt: %s", prompt)
    elapsed = time.time() - t0

    if not results:
        logger.error("No results collected.")
        sys.exit(1)

    all_avg = [r["avg_routing_metrics"] for r in results if r.get("avg_routing_metrics")]
    agg = aggregate_results(all_avg) if all_avg else {}

    # 写报告
    report = generate_report(results, agg, elapsed, cfg)
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report saved to %s", report_path)

    # 写 JSON（清理 tensor 数据）
    clean_results = []
    for r in results:
        cr = dict(r)
        skip_keys = ("step_metrics", "entropy_confidence", "expert_overlap",
                     "layerwise", "output_corr", "position_trends",
                     "logit_alignment", "gt_accuracy")
        for key in skip_keys:
            if key in cr and isinstance(cr[key], dict):
                cr[key] = {k: v for k, v in cr[key].items()
                           if isinstance(v, (int, float, str, bool, list))}
        clean_results.append(cr)

    results_json = {
        "config": {"model_path": cfg.model_path, "device": cfg.device},
        "aggregated": agg,
        "results": clean_results,
        "elapsed_seconds": elapsed,
    }
    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", json_path)

    print(f"\nDone. Report: {report_path}")
    print(f"Results: {json_path}")

if __name__ == "__main__":
    main()
