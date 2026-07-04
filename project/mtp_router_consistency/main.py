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
    logger.info("Processing prompt: %s", prompt[:60])

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.max_prompt_len,
    )
    input_ids = inputs.input_ids

    result = process_with_mtp(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=input_ids,
        device=device,
    )

    decoder_router = result["decoder_router"]   # [T, E]
    mtp_router = result["mtp_router"]           # [T, E]
    lm_logits = result["lm_logits"]             # [T, V]
    mtp_token_logits = result["mtp_token_logits"]  # [T, V] or None
    layer_routers = result["layer_routers"]     # [L, full_len, E]
    T = decoder_router.shape[0]

    # ---- 原有逐位置 Router 对比 ----
    step_metrics = []
    for t in range(T):
        m = compute_metrics(
            decoder_router[t].unsqueeze(0),
            mtp_router[t].unsqueeze(0),
            top_k=cfg.num_experts_per_tok,
        )
        step_metrics.append(m)

    avg_routing_metrics = {}
    if step_metrics:
        for key in step_metrics[0]:
            avg_routing_metrics[key] = sum(s[key] for s in step_metrics) / len(step_metrics)

    # ---- 1) Router 置信度 / 熵 ----
    entropy_conf = compute_entropy_confidence_metrics(mtp_router, decoder_router)

    # ---- 2) Expert 命中分析 ----
    expert_overlap = compute_expert_overlap_metrics(
        mtp_router, decoder_router, k=cfg.num_experts_per_tok,
    )

    # ---- 3) 逐层 Router 对比 ----
    layerwise = compute_layerwise_metrics(
        layer_routers[:, :T, :], mtp_router, k=cfg.num_experts_per_tok,
    )

    # ---- 4) 输出 logits 相关 (LM 置信度与 MTP 置信度关联) ----
    output_corr = compute_output_logits_corr_metrics(lm_logits, mtp_router)

    # ---- 5) 位置趋势 ----
    position_trends = compute_position_trends(step_metrics)

    # ---- 6) Token 贪心解码准确率 ----
    # lm_logits_for_mtp 与 mtp_token_logits 预测同一目标 token (tok[t+2])
    lm_logits_mtp = result.get("lm_logits_for_mtp")
    token_acc = compute_token_accuracy(lm_logits_mtp, mtp_token_logits)

    # ---- 7) LM head vs MTP head 完整 logits 分布对比（对齐 target token）----
    logit_alignment = compute_lm_mtp_logit_alignment(lm_logits, mtp_token_logits)

    # ---- 8) 对真实 token (Ground Truth) 的预测精度 ----
    gt_lm = result.get("ground_truth_lm")
    gt_mtp = result.get("ground_truth_mtp")
    gt_accuracy = compute_accuracy_vs_ground_truth(
        lm_logits, mtp_token_logits, gt_lm, gt_mtp,
    )

    generated_text = tokenizer.decode(
        result["output_ids"][0], skip_special_tokens=True
    )

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
    lines = []
    lines.append("# MTP 多维度分析报告")
    lines.append("")
    lines.append(f"**模型**: {cfg.model_path}")
    lines.append(f"**设备**: {cfg.device}")
    lines.append(f"**分析模式**: 单次前向传播 (不生成新 token)")
    lines.append(f"**用时**: {elapsed:.1f}s")
    lines.append(f"**测试样本数**: {len(results)}")
    lines.append("")

    # ============== 0) 原有路由一致性指标 ==============
    lines.append("## 0) Router 一致性指标")
    lines.append("")
    lines.append("| 指标 | 均值 | 最小值 | 最大值 |")
    lines.append("|------|------|--------|--------|")
    for key in sorted(agg.keys()):
        if key.startswith("avg_"):
            base = key[4:]
            min_k = f"min_{base}"
            max_k = f"max_{base}"
            lines.append(
                f"| {base} | {agg[key]:.4f} | {agg.get(min_k, 0):.4f} | {agg.get(max_k, 0):.4f} |"
            )
    lines.append("")

    # ============== 1) 熵 & 置信度 ==============
    lines.append("## 1) Router 置信度 / 熵分析")
    lines.append("")
    lines.append("| 指标 | 含义 | 值 |")
    lines.append("|------|------|-----|")
    keys_describe = {
        "mtp_entropy_mean": "MTP Router 熵（越高越不确定）",
        "actual_entropy_mean": "Decoder Router 熵",
        "entropy_diff_mean": "熵差 (MTP - Decoder)",
        "mtp_confidence_mean": "MTP Router 置信度 (top-1 prob)",
        "actual_confidence_mean": "Decoder Router 置信度",
        "confidence_diff_mean": "置信度差 (MTP - Decoder)",
    }
    for r in results:
        lines.append(f"### 样本 {results.index(r)+1}")
        ec = r.get("entropy_confidence", {})
        for k, desc in keys_describe.items():
            if k in ec:
                lines.append(_table_row(desc, ec[k]))
        lines.append("")

    # ============== 2) Expert 命中 ==============
    lines.append("## 2) Expert 命中分析")
    lines.append("")
    lines.append("| 指标 | 含义 | 值 |")
    lines.append("|------|------|-----|")
    eo_keys = {
        "avg_overlap_count": "平均重叠 expert 数 (top-k 中)",
        "max_overlap_count": "最大重叠 expert 数",
        "min_overlap_count": "最小重叠 expert 数",
        "overlap_ratio": "重叠比例 (overlap/k)",
        "zero_overlap_ratio": "零重叠比例 (完全不命中)",
        "full_overlap_ratio": "完全重叠比例 (k/k)",
    }
    for r in results:
        lines.append(f"### 样本 {results.index(r)+1}")
        eo = r.get("expert_overlap", {})
        for k, desc in eo_keys.items():
            if k in eo:
                lines.append(_table_row(desc, eo[k]))
        lines.append("")

    # ============== 3) 逐层对比 ==============
    lines.append("## 3) 逐层 Router 对比 (MTP vs 各 Decoder Layer)")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|------|-----|")
    lw_keys = {
        "layerwise_cosine_mean": "逐层 Cosine 均值",
        "layerwise_topk_iou_mean": "逐层 Top-K IoU 均值",
        "last_layer_cosine": "最后一层 Cosine (即原指标)",
        "last_layer_iou": "最后一层 Top-K IoU",
        "layerwise_cosine_improvement": "末层 Cosine - 首层 Cosine",
    }
    for r in results:
        lines.append(f"### 样本 {results.index(r)+1}")
        lw = r.get("layerwise", {})
        for k, desc in lw_keys.items():
            if k in lw:
                lines.append(_table_row(desc, lw[k]))
        lines.append("")
        # 简短的趋势说明
        cos_list = lw.get("layerwise_cosine", [])
        if cos_list:
            lines.append("逐层 Cosine 序列 (浅→深):")
            lines.append("")
            n = len(cos_list)
            # 显示头部、中间、尾部
            snippet = cos_list[:3] + (["..."] if n > 6 else []) + cos_list[-3:]
            lines.append(f"`{', '.join(f'{v:.3f}' for v in cos_list[:3])} ... {', '.join(f'{v:.3f}' for v in cos_list[-3:])}`")
            lines.append("")

    # ============== 4) 输出 logits 关联 ==============
    lines.append("## 4) 输出 logits 关联分析 (LM Head vs MTP Router)")
    lines.append("")
    lines.append("| 指标 | 含义 | 值 |")
    lines.append("|------|------|-----|")
    oc_keys = {
        "lm_mtp_confidence_corr": "LM 置信度 vs MTP 置信度 相关系数",
        "lm_confidence_mean": "LM Head 平均置信度",
        "lm_confidence_std": "LM Head 置信度标准差",
    }
    for r in results:
        lines.append(f"### 样本 {results.index(r)+1}")
        oc = r.get("output_corr", {})
        for k, desc in oc_keys.items():
            if k in oc:
                lines.append(_table_row(desc, oc[k]))
        lines.append("")

    # ============== 5) 位置趋势 ==============
    lines.append("## 5) 位置趋势分析")
    lines.append("")
    for r in results:
        lines.append(f"### 样本 {results.index(r)+1}")
        pt = r.get("position_trends", {})
        if pt:
            lines.append("| 指标 | 前半均值 | 后半均值 | 趋势 |")
            lines.append("|------|----------|----------|------|")
            for key, trend in pt.items():
                lines.append(
                    f"| {key} | {trend['first_half_avg']:.4f} | "
                    f"{trend['second_half_avg']:.4f} | {trend['trend_direction']} |"
                )
        else:
            lines.append("*无趋势数据*")
        lines.append("")

    # ============== 6) Token 贪心解码准确率 ==============
    lines.append("## 6) Token 贪心解码准确率")
    lines.append("")
    lines.append("比较 Decoder (LM Head) 和 MTP Head 在贪心解码下预测的 token 是否一致。")
    lines.append("")
    lines.append("| 样本 | 准确率 | 正确数/总数 |")
    lines.append("|------|--------|-------------|")
    for i, r in enumerate(results):
        ta = r.get("token_accuracy", {})
        acc = ta.get("token_accuracy")
        if acc is not None:
            correct = ta.get("token_correct_count", 0)
            total = ta.get("token_total_count", 0)
            lines.append(f"| 样本 {i+1} | {acc:.2%} | {correct}/{total} |")
        else:
            lines.append(f"| 样本 {i+1} | N/A (无 MTP token logits) | - |")
    lines.append("")

    # ============== 7) LM head vs MTP head 完整 logits 分布 ==============
    lines.append("## 7) LM head vs MTP head 完整 Logits 分布对比")
    lines.append("")
    lines.append("对齐目标： `lm_logits[:, t+1, :]` vs `mtp_logits[:, t, :]` 都预测 `token[t+2]`")
    lines.append("")
    lines.append("| 指标 | 含义 | 样本1 | 样本2 |")
    lines.append("|------|------|-------|-------|")
    la_keys = [
        ("lm_mtp_logit_cosine", "Cosine 相似度"),
        ("lm_mtp_logit_kl", "KL 散度"),
        ("lm_mtp_logit_js", "JS 散度"),
        ("lm_mtp_logit_spearman", "Spearman 相关"),
        ("lm_mtp_logit_top8_iou", "Top-8 IoU"),
        ("lm_mtp_logit_prob_dot", "概率分布点积"),
        ("lm_mtp_logit_l2", "Logits L2 距离"),
    ]
    la_vals = [r.get("logit_alignment", {}) for r in results]
    for key, desc in la_keys:
        v1 = la_vals[0].get(key, "N/A") if len(la_vals) > 0 else "N/A"
        v2 = la_vals[1].get(key, "N/A") if len(la_vals) > 1 else "N/A"
        v1s = f"{v1:.4f}" if isinstance(v1, float) else str(v1)
        v2s = f"{v2:.4f}" if isinstance(v2, float) else str(v2)
        lines.append(f"| {key} | {desc} | {v1s} | {v2s} |")
    lines.append("")
    lines.append("**解读**: Cosine ~1.0 且 KL/JS ≈ 0 说明两个 logits 分布几乎相同，与 injectivity 理论一致。")
    lines.append("")

    # ============== 8) 对真实 token 的预测精度 ==============
    lines.append("## 8) 对真实 Token (Ground Truth) 的预测精度")
    lines.append("")
    lines.append("比较 LM Head / MTP Head 的贪心解码结果与序列中的真实 token。")
    lines.append("")
    lines.append("| 样本 | LM→tok[t+1] 精度 | MTP→tok[t+2] 精度 | MTP-LM 差距 |")
    lines.append("|------|-------------------|-------------------|-------------|")
    for i, r in enumerate(results):
        ga = r.get("gt_accuracy", {})
        lma = ga.get("lm_gt_accuracy")
        mtpa = ga.get("mtp_gt_accuracy")
        gap = ga.get("mtp_gt_gap")
        lma_s = f"{lma:.2%}" if lma is not None else "N/A"
        mtpa_s = f"{mtpa:.2%}" if mtpa is not None else "N/A"
        gap_s = f"{gap:+.2%}" if gap is not None else "N/A"
        lines.append(f"| 样本 {i+1} | {lma_s} ({ga.get('lm_gt_correct','-')}/{ga.get('lm_gt_total','-')}) | {mtpa_s} ({ga.get('mtp_gt_correct','-')}/{ga.get('mtp_gt_total','-')}) | {gap_s} |")
    lines.append("")
    lines.append("**解读**: LM 精度反映了模型对 prompt 内已知 token 的拟合程度; MTP 精度反映其在看到 tok[t+1] 嵌入后预测 tok[t+2] 的能力。")
    lines.append("")

    # ============== 样本详情 ==============
    lines.append("## 各样本详情")
    lines.append("")
    for i, r in enumerate(results):
        lines.append(f"### 样本 {i+1}")
        lines.append("")
        lines.append(f"**Prompt**: `{r['prompt'][:80]}`")
        lines.append(f"**可比较位置数**: {r['num_comparisons']}")
        lines.append("")
        if r.get("avg_routing_metrics"):
            lines.append("| 路由指标 | 值 |")
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
    device = cfg.device

    model, tokenizer, model_device = load_model_and_tokenizer(
        cfg.model_path, device=device, torch_dtype=torch_dtype,
    )

    t0 = time.time()
    results = []
    for prompt in cfg.prompts:
        try:
            r = run_single_prompt(
                model, tokenizer, prompt, cfg, str(model_device),
            )
            results.append(r)
        except Exception as e:
            logger.exception("Failed on prompt: %s", prompt)
    elapsed = time.time() - t0

    if not results:
        logger.error("No results collected.")
        sys.exit(1)

    # 聚合路由指标
    all_avg = [r["avg_routing_metrics"] for r in results if r.get("avg_routing_metrics")]
    agg = aggregate_results(all_avg) if all_avg else {}

    report = generate_report(results, agg, elapsed, cfg)
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report saved to %s", report_path)

    # 清理 tensor 数据使 JSON 可序列化
    clean_results = []
    for r in results:
        cr = dict(r)
        for key in ("step_metrics", "entropy_confidence", "expert_overlap",
                     "layerwise", "output_corr", "position_trends",
                     "logit_alignment", "gt_accuracy"):
            if key in cr and isinstance(cr[key], dict):
                cr[key] = {k: v for k, v in cr[key].items()
                           if isinstance(v, (int, float, str, bool, list))}
        clean_results.append(cr)

    results_json = {
        "config": {
            "model_path": cfg.model_path,
            "device": cfg.device,
            "max_new_tokens": cfg.max_new_tokens,
        },
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
