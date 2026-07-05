"""MTP Router Consistency Analysis - Main Pipeline.

Loads model, runs per-prompt analysis through all metric groups,
generates a Markdown report and JSON results.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import torch

_pkg_dir = Path(__file__).parent.resolve()
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

from compare import (
    aggregate_results,
    compute_accuracy_vs_ground_truth,
    compute_metrics,
    compute_entropy_confidence_metrics,
    compute_expert_overlap_metrics,
    compute_layerwise_metrics,
    compute_lm_mtp_logit_alignment,
    compute_output_logits_corr_metrics,
    compute_position_trends,
    compute_token_accuracy,
)
from config import Config
from decoder import process_with_mtp
from model_utils import load_model_and_tokenizer

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
    """Analyze MTP vs Decoder routing for a single prompt."""
    logger.info("Processing prompt: %s", prompt[:60])

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_prompt_len)
    input_ids = inputs.input_ids

    result = process_with_mtp(model=model, tokenizer=tokenizer, prompt_ids=input_ids, device=device)

    decoder_router = result["decoder_router"]   # [T, E]
    mtp_router = result["mtp_router"]           # [T, E]
    lm_logits = result["lm_logits"]             # [T, V]
    mtp_token_logits = result["mtp_token_logits"]  # [T, V] or None
    layer_routers = result["layer_routers"]     # [L, seq_len, E]
    gt_lm = result.get("ground_truth_lm")
    gt_mtp = result.get("ground_truth_mtp")
    T = decoder_router.shape[0]

    # Per-position router comparison
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

    # 1) Router entropy / confidence
    entropy_conf = compute_entropy_confidence_metrics(mtp_router, decoder_router)

    # 2) Expert overlap
    expert_overlap = compute_expert_overlap_metrics(
        mtp_router, decoder_router, k=cfg.num_experts_per_tok,
    )

    # 3) Layer-wise router comparison
    layerwise = compute_layerwise_metrics(
        layer_routers[:, :T, :], mtp_router, k=cfg.num_experts_per_tok,
    )

    # 4) LM head vs MTP router confidence correlation
    output_corr = compute_output_logits_corr_metrics(lm_logits, mtp_router)

    # 5) Position trends
    position_trends = compute_position_trends(step_metrics)

    # 6) Token greedy decoding accuracy (LM vs MTP head)
    lm_logits_mtp = result.get("lm_logits_for_mtp")
    token_acc = compute_token_accuracy(lm_logits_mtp, mtp_token_logits)

    # 7) LM vs MTP head full logit distribution alignment
    logit_alignment = compute_lm_mtp_logit_alignment(lm_logits, mtp_token_logits)

    # 8) Ground truth accuracy
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
    """Generate a comprehensive Markdown report from analysis results."""
    lines = []
    lines.append("# MTP Multi-Dimension Analysis Report")
    lines.append("")
    lines.append(f"**Model**: {cfg.model_path}")
    lines.append(f"**Device**: {cfg.device}")
    lines.append(f"**Analysis**: Single forward pass (no generation)")
    lines.append(f"**Time**: {elapsed:.1f}s")
    lines.append(f"**Samples**: {len(results)}")
    lines.append("")

    # 0) Router consistency metrics
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

    # 1) Entropy & confidence
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

    # 2) Expert overlap
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

    # 3) Layer-wise comparison
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

    # 4) Logit correlation
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

    # 5) Position trends
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

    # 6) Token accuracy
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

    # 7) Logit alignment (injectivity)
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

    # 8) Ground truth accuracy
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

    # Sample details
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

    # Write report
    report = generate_report(results, agg, elapsed, cfg)
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report saved to %s", report_path)

    # Write JSON (clean tensor data)
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
