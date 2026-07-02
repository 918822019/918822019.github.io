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

from compare import aggregate_results, compute_metrics
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
        max_new_tokens=cfg.max_new_tokens,
        device=device,
    )

    actual = result["actual_logits"]
    mtp_pred = result["mtp_pred_logits"]

    aligned = []
    for t in range(len(mtp_pred)):
        metrics = compute_metrics(
            actual[t].unsqueeze(0), mtp_pred[t].unsqueeze(0),
            top_k=cfg.num_experts_per_tok,
        )
        aligned.append(metrics)

    avg_metrics = {}
    if aligned:
        for key in aligned[0]:
            avg_metrics[key] = sum(m[key] for m in aligned) / len(aligned)
    else:
        logger.warning("No aligned comparison pairs for prompt.")

    generated_text = tokenizer.decode(
        result["output_ids"][0], skip_special_tokens=True
    )

    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "num_generated": len(result["generated"]),
        "num_comparisons": len(aligned),
        "avg_metrics": avg_metrics,
        "step_metrics": aligned,
    }


def generate_report(results: list[dict], agg: dict, elapsed: float, cfg: Config) -> str:
    lines = []
    lines.append("# MTP vs Router 一致性测试报告")
    lines.append("")
    lines.append(f"**模型**: {cfg.model_path}")
    lines.append(f"**设备**: {cfg.device}")
    lines.append(f"**max_new_tokens**: {cfg.max_new_tokens}")
    lines.append(f"**用时**: {elapsed:.1f}s")
    lines.append(f"**测试样本数**: {len(results)}")
    lines.append("")
    lines.append("## 聚合指标")
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
    lines.append("## 各样本详情")
    lines.append("")
    for i, r in enumerate(results):
        lines.append(f"### 样本 {i+1}")
        lines.append("")
        lines.append(f"**Prompt**: `{r['prompt'][:80]}`")
        lines.append(f"**生成长度**: {r['num_generated']}")
        lines.append(f"**比较对数**: {r['num_comparisons']}")
        lines.append("")
        if r["avg_metrics"]:
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            for k, v in r["avg_metrics"].items():
                lines.append(f"| {k} | {v:.4f} |")
        lines.append("")
        lines.append(f"**生成文本预览**:")
        lines.append(f"```")
        lines.append(r["generated_text"][:200])
        lines.append("```")
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

    all_avg = [r["avg_metrics"] for r in results if r["avg_metrics"]]
    agg = aggregate_results(all_avg) if all_avg else {}

    report = generate_report(results, agg, elapsed, cfg)
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report saved to %s", report_path)

    results_json = {
        "config": {
            "model_path": cfg.model_path,
            "device": cfg.device,
            "max_new_tokens": cfg.max_new_tokens,
        },
        "aggregated": agg,
        "results": results,
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
