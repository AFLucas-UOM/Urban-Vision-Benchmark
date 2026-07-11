from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from persistence import write_csv, write_json


def _macro(rows: list[dict[str, Any]], keys=("precision", "recall", "f1", "accuracy", "mean_matched_iou", "mean_inference_ms")) -> dict[str, float]:
    return {key: round(mean(float(row.get(key) or 0) for row in rows), 4) for key in keys} if rows else {key: 0.0 for key in keys}


def generate(run_dir: Path, run_config: dict[str, Any], combination_rows: list[dict[str, Any]],
             per_image: list[dict[str, Any]], overlaps: list[dict[str, Any]]) -> dict[str, Any]:
    write_csv(run_dir / "targeted_per_prompt_metrics.csv", combination_rows)
    write_csv(run_dir / "per_prompt_metrics.csv", combination_rows)
    write_csv(run_dir / "per_image_metrics.csv", per_image)
    write_csv(run_dir / "fp_nontarget_overlap.csv", overlaps)
    synonym = [row for row in combination_rows if row["prompt_group"] == "synonym-comparison"]
    broad = [row for row in combination_rows if row["prompt_group"] in {"broad", "optional-broad"}]
    write_csv(run_dir / "synonym_comparison.csv", synonym); write_csv(run_dir / "broad_prompt_metrics.csv", broad)
    by_target: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in combination_rows:
        for target in row["target_classes"]: by_target[(row["model"], target)].append(row)
        by_group[(row["model"], row["prompt_group"])].append(row)
        if row["prompt_group"] in {"class-targeted", "synonym-comparison"}:
            by_model[row["model"]].append(row)
    target_rows = []
    for (model, target), rows in sorted(by_target.items()):
        best = max(rows, key=lambda row: (float(row.get("f1", 0)), row["prompt_id"]))
        target_rows.append({"model": model, "target_class": target, "best_prompt_id": best["prompt_id"],
                            "best_prompt": best["prompt"], "recall": best["recall"], "f1": best["f1"]})
    group_rows = [{"model": model, "prompt_group": group, "n_prompts": len(rows), **_macro(rows)}
                  for (model, group), rows in sorted(by_group.items())]
    model_rows = [{"model": model, "headline_prompt_groups": "class-targeted,synonym-comparison",
                   "n_headline_prompts": len(rows), "ap_used_for_ranking": False, **_macro(rows)}
                  for model, rows in sorted(by_model.items())]
    write_csv(run_dir / "per_target_class_metrics.csv", target_rows)
    write_csv(run_dir / "prompt_group_summary.csv", group_rows)
    write_csv(run_dir / "per_model_summary.csv", model_rows)
    write_csv(run_dir / "full_comparison.csv", combination_rows)
    confusion_rows = []
    for row in overlaps:
        if row.get("fp_overlap_class"):
            confusion_rows.append({"model": row["model"], "prompt_id": row["prompt_id"],
                                   "non_target_class": row["fp_overlap_class"], "count": 1})
    write_csv(run_dir / "confusion_fp_nontarget.csv", confusion_rows)
    prompt_confusion_rows = [
        {"model": row["model"], "prompt_id": row["prompt_id"], "gt_class": class_name, "matched_count": count}
        for row in combination_rows
        for class_name, count in (row.get("prompt_class_confusion") or {}).items()
    ]
    write_csv(run_dir / "prompt_vs_class_confusion.csv", prompt_confusion_rows)
    plots = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if model_rows:
            fig, ax = plt.subplots(figsize=(8, 5)); ax.bar([r["model"] for r in model_rows], [r["f1"] for r in model_rows])
            ax.set(ylabel="Macro F1", title="Class-targeted PromptDetect comparison", ylim=(0, 1)); ax.tick_params(axis="x", rotation=25)
            fig.tight_layout(); fig.savefig(run_dir / "model_f1.png", dpi=200); plt.close(fig); plots.append("model_f1.png")
            fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter([r["recall"] for r in model_rows], [r["precision"] for r in model_rows])
            for row in model_rows: ax.annotate(row["model"], (row["recall"], row["precision"]))
            ax.set(xlabel="Recall", ylabel="Precision", title="Precision vs recall", xlim=(0, 1), ylim=(0, 1))
            fig.tight_layout(); fig.savefig(run_dir / "precision_recall_scatter.png", dpi=200); plt.close(fig); plots.append("precision_recall_scatter.png")
    except ImportError:
        pass
    summary = {**run_config, "evaluation_protocol": "targeted-v1", "per_prompt_metrics": combination_rows,
               "per_model_summary": model_rows, "per_target_class_metrics": target_rows,
               "prompt_group_summary": group_rows, "plots": plots,
               "notes": ["Unmatched detections are false positives, including on negative images.",
                         "AP for constant-confidence models is not a ranking-quality measure; use P/R/F1 and matched IoU."]}
    write_json(run_dir / "evaluation_summary.json", summary)
    lines = ["# PromptDetect dissertation protocol report", "", f"Protocol: `{run_config['protocol_version']}`",
             "", "## Headline targeted macro averages", "",
             "Includes class-targeted and synonym-comparison prompts; excludes broad and optional prompts. AP is not used for universal ranking.", ""]
    for row in model_rows: lines.append(f"- {row['model']}: P={row['precision']:.4f}, R={row['recall']:.4f}, F1={row['f1']:.4f}")
    (run_dir / "dissertation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
