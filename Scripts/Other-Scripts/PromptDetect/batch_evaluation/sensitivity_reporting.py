"""Persistence, reporting and plotting for prompt-sensitivity runs.

Consumes the full-precision aggregates from ``prompt_sensitivity`` and writes
the run-level CSV/JSON/Markdown outputs plus dissertation-ready figures.
Rounding to four decimals happens here, at the output boundary only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from persistence import slug, write_csv, write_json
from prompt_sensitivity import (
    DEFAULT_CONSISTENCY_IOU,
    VARIANT_TYPES,
    aggregate_family_consistency,
    aggregate_model_consistency,
    aggregate_pair_consistency,
    family_statistics,
    model_macro_rows,
    pairwise_image_rows,
    predictions_by_prompt_for_model,
    sensitivity_rows,
    validate_family_gt_counts,
    variant_difference_rows,
)

ROUND_DIGITS = 4

# Okabe-Ito colourblind-safe palette; hues are assigned in fixed order and
# never cycled (variant types and models each keep a stable colour).
VARIANT_COLORS = {"canonical": "#0072B2", "lexical-paraphrase": "#E69F00",
                  "descriptive-paraphrase": "#009E73", "instruction": "#CC79A7"}
MODEL_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]
NEUTRAL_BAR = "#4878A8"

METHODOLOGICAL_NOTES = [
    "All prompts in a sensitivity family share identical target ground truth; wording is the only experimental variable.",
    "Target-negative images are retained; every detection on them counts as a false positive.",
    "Synonym-comparison and broad/superclass prompts belong to the separate dissertation protocol and are never mixed into these paraphrase-sensitivity statistics.",
    "Prediction consistency compares the raw prediction sets of two prompt variants (greedy IoU matching); it does not use ground truth and measures output stability, not correctness.",
    "A prompt pair producing zero boxes on an image is fully consistent for that image (pairwise F1 = Jaccard = 1.0, both_empty=true).",
    "Relative F1 degradation is (max_f1 - min_f1) / max_f1; when max_f1 == 0 it is reported as 0.0 with all_variants_failed=true.",
    "Backends with constant confidence make AP collapse to one operating point; AP is reported but never used as the sole cross-model ranking measure.",
    "Low metric variance can also mean every prompt performed poorly; interpret variability together with mean and worst-case performance.",
]


def _rounded(rows: list[dict[str, Any]], drop: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    return [{key: (round(value, ROUND_DIGITS) if isinstance(value, float) else value)
             for key, value in row.items() if key not in drop} for row in rows]


# --- plotting ---------------------------------------------------------------------


def _save(fig, run_dir: Path, stem: str, plots: list[str]) -> None:
    for suffix in (".png", ".svg"):
        fig.savefig(run_dir / f"{stem}{suffix}", dpi=200)
        plots.append(f"{stem}{suffix}")


def _style(ax) -> None:
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save_sensitivity_plots(run_dir: Path, run_config: dict[str, Any],
                           per_prompt: list[dict[str, Any]], family_stats: list[dict[str, Any]],
                           model_rows: list[dict[str, Any]],
                           model_consistency: list[dict[str, Any]]) -> list[str]:
    plots: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return plots
    if not family_stats:
        return plots
    dataset = run_config.get("dataset", "?")
    iou = run_config.get("iou_threshold", "?")
    models = sorted({row["model"] for row in family_stats})
    families = sorted({row["sensitivity_family"] for row in family_stats})
    model_color = {model: MODEL_COLORS[index % len(MODEL_COLORS)] for index, model in enumerate(models)}
    by_family_model = {(row["sensitivity_family"], row["model"]): row for row in family_stats}

    # 1. F1 by prompt variant, one figure per model (families x 4 variant bars).
    for model in models:
        rows = [row for row in per_prompt if row["model"] == model]
        by_family_variant = {(row["sensitivity_family"], row.get("variant_type")): float(row.get("f1") or 0.0)
                             for row in rows}
        fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(families)), 4.5))
        width = 0.8 / len(VARIANT_TYPES)
        for offset, variant in enumerate(VARIANT_TYPES):
            xs = [index + (offset - (len(VARIANT_TYPES) - 1) / 2) * width for index in range(len(families))]
            ax.bar(xs, [by_family_variant.get((family, variant), 0.0) for family in families],
                   width * 0.92, label=variant, color=VARIANT_COLORS[variant])
        ax.set_xticks(range(len(families)), families, rotation=25, ha="right", fontsize=8)
        ax.set(ylabel=f"Targeted F1 (IoU ≥ {iou})", ylim=(0, 1.02),
               title=f"F1 per prompt variant — {model} on {dataset} (targeted GT)")
        ax.legend(title="Prompt variant", fontsize=8, title_fontsize=8)
        _style(ax); fig.tight_layout()
        _save(fig, run_dir, f"sensitivity_f1_by_variant__{slug(model)}", plots); plt.close(fig)

    # 2. Mean family F1 with within-family population-std error bars, grouped by model.
    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(families)), 4.8))
    width = 0.8 / max(1, len(models))
    for offset, model in enumerate(models):
        xs = [index + (offset - (len(models) - 1) / 2) * width for index in range(len(families))]
        means = [by_family_model.get((family, model), {}).get("f1_mean", 0.0) for family in families]
        stds = [by_family_model.get((family, model), {}).get("f1_std", 0.0) for family in families]
        ax.bar(xs, means, width * 0.92, yerr=stds, capsize=2, label=model,
               color=model_color[model], error_kw={"linewidth": 0.9})
    ax.set_xticks(range(len(families)), families, rotation=25, ha="right", fontsize=8)
    ax.set(ylabel="Mean targeted F1 across 4 prompt variants", ylim=(0, 1.05),
           title=f"Mean F1 per sensitivity family on {dataset} "
                 "(error bars: within-family population std of F1)")
    ax.legend(fontsize=8)
    _style(ax); fig.tight_layout()
    _save(fig, run_dir, "sensitivity_family_f1_mean_std", plots); plt.close(fig)

    # 3. Best-to-worst F1 degradation by model (mean absolute across families).
    fig, ax = plt.subplots(figsize=(7, 4.2))
    degradation = [sum(by_family_model[(family, model)]["f1_degradation_abs"]
                       for family in families if (family, model) in by_family_model)
                   / max(1, sum((family, model) in by_family_model for family in families))
                   for model in models]
    ax.bar(models, degradation, color=NEUTRAL_BAR, width=0.6)
    ax.set(ylabel="Mean (best F1 − worst F1) across families",
           title=f"Best-to-worst F1 degradation per model on {dataset}")
    ax.tick_params(axis="x", rotation=25)
    _style(ax); fig.tight_layout()
    _save(fig, run_dir, "sensitivity_f1_degradation_by_model", plots); plt.close(fig)

    # 4. Worst-family relative degradation by model, annotated with the family.
    if model_rows:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ordered = sorted(model_rows, key=lambda row: row["model"])
        values = [row["worst_family_relative_degradation"] for row in ordered]
        ax.bar([row["model"] for row in ordered], values, color=NEUTRAL_BAR, width=0.6)
        for index, row in enumerate(ordered):
            ax.annotate(row["worst_family"], (index, values[index]), ha="center", va="bottom",
                        fontsize=7, rotation=15)
        ax.set(ylabel="Relative F1 degradation in worst family\n(max F1 − min F1) / max F1",
               ylim=(0, 1.1), title=f"Worst-family relative F1 degradation per model on {dataset}")
        ax.tick_params(axis="x", rotation=25)
        _style(ax); fig.tight_layout()
        _save(fig, run_dir, "sensitivity_worst_family_degradation_by_model", plots); plt.close(fig)

        # 5. Canonical vs instruction-style macro F1 per model.
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        xs = range(len(ordered))
        ax.bar([x - 0.18 for x in xs], [row["macro_canonical_f1"] or 0.0 for row in ordered], 0.34,
               label="canonical prompt", color=VARIANT_COLORS["canonical"])
        ax.bar([x + 0.18 for x in xs], [row["macro_instruction_f1"] or 0.0 for row in ordered], 0.34,
               label="instruction prompt", color=VARIANT_COLORS["instruction"])
        ax.set_xticks(list(xs), [row["model"] for row in ordered], rotation=25, ha="right")
        ax.set(ylabel="Macro F1 (mean of family F1)", ylim=(0, 1.02),
               title=f"Canonical vs instruction-style prompt F1 on {dataset}")
        ax.legend(fontsize=8)
        _style(ax); fig.tight_layout()
        _save(fig, run_dir, "sensitivity_canonical_vs_instruction_f1", plots); plt.close(fig)

    # 6. Pairwise prediction-consistency F1 by model (agreement, not accuracy).
    if model_consistency:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ordered = sorted(model_consistency, key=lambda row: row["model"])
        ax.bar([row["model"] for row in ordered], [row["macro_pairwise_box_f1"] for row in ordered],
               color=NEUTRAL_BAR, width=0.6)
        ax.set(ylabel="Macro pairwise box F1 between prompt variants", ylim=(0, 1.02),
               title=f"Prediction consistency per model on {dataset}\n"
                     "(agreement between prompt variants; not accuracy against GT)")
        ax.tick_params(axis="x", rotation=25)
        _style(ax); fig.tight_layout()
        _save(fig, run_dir, "consistency_pairwise_f1_by_model", plots); plt.close(fig)

    # 7. Model x family heatmap of the F1 range (single-hue sequential ramp).
    data = [[by_family_model.get((family, model), {}).get("f1_range", 0.0) for family in families]
            for model in models]
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(families)), max(3.5, 0.7 * len(models))))
    mesh = ax.imshow(data, cmap="Blues", vmin=0, vmax=max(0.2, max(max(row) for row in data)))
    ax.set_xticks(range(len(families)), families, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)), models, fontsize=8)
    ax.set_title(f"F1 range (max − min across 4 prompt variants) on {dataset}")
    top = max(max(row) for row in data)
    for i in range(len(models)):
        for j in range(len(families)):
            ax.text(j, i, f"{data[i][j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if top and data[i][j] > top * 0.6 else "#333333")
    fig.colorbar(mesh, ax=ax, shrink=0.85, label="F1 range")
    fig.tight_layout()
    _save(fig, run_dir, "sensitivity_f1_range_heatmap", plots); plt.close(fig)
    return plots


# --- Markdown report ----------------------------------------------------------------


def _report_lines(run_config: dict[str, Any], family_stats: list[dict[str, Any]],
                  model_rows: list[dict[str, Any]], model_consistency: list[dict[str, Any]],
                  family_gt: list[dict[str, Any]]) -> list[str]:
    dataset = run_config.get("dataset", "?")
    lines = [
        "# Prompt-sensitivity report",
        "",
        f"Protocol: `{run_config.get('protocol_version')}` (hash `{str(run_config.get('protocol_hash'))[:12]}...`), "
        f"dataset **{dataset}**, split `{run_config.get('split')}`, "
        f"conf ≥ {run_config.get('conf_threshold')}, IoU ≥ {run_config.get('iou_threshold')}, "
        f"consistency IoU ≥ {run_config.get('consistency_iou_threshold', DEFAULT_CONSISTENCY_IOU)}.",
        "",
        "## What this experiment measures",
        "",
        "Prompt sensitivity is the degree to which a prompt-based detector changes its output when the",
        "*same request* is worded differently. Each sensitivity family contains four semantically",
        "equivalent prompts (canonical, lexical paraphrase, descriptive paraphrase, instruction) with",
        "identical `target_classes`, so the target ground truth is exactly the same for all four prompts;",
        "wording is the only experimental variable. Synonym comparisons (e.g. “no entry” vs “one way”)",
        "and broad superclass prompts (e.g. “traffic sign”) change the *meaning* or the *scope* of the",
        "request, not only its wording — they remain in the main dissertation protocol and are deliberately",
        "excluded from these statistics.",
        "",
        "Ground truth reuses the canonical test-split annotations unchanged: boxes of the family's target",
        "classes are positive GT; all other annotated objects are non-target. Images without any target box",
        "are retained — every detection on them is a false positive — so precision reflects behaviour on",
        "realistic target-free scenes.",
        "",
    ]
    if family_gt:
        lines += ["## Family ground truth (identical for all four variants)", "",
                  "| family | target classes | positive images | negative images | target GT boxes |",
                  "|---|---|---|---|---|"]
        for row in family_gt:
            lines.append(f"| {row['sensitivity_family']} | {', '.join(row['target_classes'])} | "
                         f"{row['positive_images']} | {row['negative_images']} | {row['target_gt_boxes']} |")
        lines.append("")
    if model_rows:
        lines += ["## Model-level sensitivity summary", "",
                  "| model | families | macro mean F1 | mean within-family F1 std | mean F1 range | "
                  "mean rel. degradation | worst family (rel. degr.) | macro canonical F1 | macro instruction F1 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for row in sorted(model_rows, key=lambda item: item["model"]):
            lines.append(
                f"| {row['model']} | {row['n_families']} | {row['macro_mean_f1']:.4f} | "
                f"{row['mean_within_family_f1_std']:.4f} | {row['mean_f1_range']:.4f} | "
                f"{row['mean_relative_f1_degradation']:.4f} | {row['worst_family']} "
                f"({row['worst_family_relative_degradation']:.4f}) | "
                f"{'-' if row['macro_canonical_f1'] is None else format(row['macro_canonical_f1'], '.4f')} | "
                f"{'-' if row['macro_instruction_f1'] is None else format(row['macro_instruction_f1'], '.4f')} |")
        most = max(model_rows, key=lambda row: row["mean_relative_f1_degradation"])
        least = min(model_rows, key=lambda row: row["mean_relative_f1_degradation"])
        lines += ["",
                  f"Most prompt-sensitive model (largest mean relative F1 degradation): **{most['model']}** "
                  f"({most['mean_relative_f1_degradation']:.4f}); least sensitive: **{least['model']}** "
                  f"({least['mean_relative_f1_degradation']:.4f}). A model with a high mean but a large spread is "
                  "*accurate but prompt-sensitive*; a slightly weaker model with a small spread may be more "
                  "operationally stable. Judge models on mean **and** variation, never on the mean alone.",
                  ""]
    if family_stats:
        worst_families = sorted(family_stats, key=lambda row: -row["f1_degradation_rel"])[:5]
        lines += ["## Largest best-to-worst degradations (dataset x model x family)", "",
                  "| model | family | best prompt (variant) | worst prompt (variant) | abs F1 degr. | rel. F1 degr. |",
                  "|---|---|---|---|---|---|"]
        for row in worst_families:
            lines.append(f"| {row['model']} | {row['sensitivity_family']} | {row['best_prompt']} "
                         f"({row['best_variant_type']}) | {row['worst_prompt']} ({row['worst_variant_type']}) | "
                         f"{row['f1_degradation_abs']:.4f} | {row['f1_degradation_rel']:.4f} |")
        failed = [row for row in family_stats if row["all_variants_failed"]]
        if failed:
            lines += ["", "Families where **every** variant scored F1 = 0 (relative degradation reported as 0.0):",
                      ""] + [f"- {row['model']} / {row['sensitivity_family']}" for row in failed]
        lines.append("")
    if model_consistency:
        lines += ["## Prediction consistency (agreement, not accuracy)", "",
                  "Pairwise box F1 compares the raw predictions of two prompt variants on the same image via",
                  "greedy IoU matching — no ground truth is involved. Two prompts can reach similar targeted F1",
                  "while detecting *different* objects, so this table must be read together with the GT-based",
                  "metrics above. High consistency can equally mean consistently correct or consistently wrong.",
                  "",
                  "| model | families | macro pairwise box F1 | macro pairwise Jaccard | least consistent family |",
                  "|---|---|---|---|---|"]
        for row in sorted(model_consistency, key=lambda item: item["model"]):
            iou_cell = "-" if row["macro_matched_pair_iou"] is None else f"{row['macro_matched_pair_iou']:.4f}"
            lines.append(f"| {row['model']} | {row['n_families']} | {row['macro_pairwise_box_f1']:.4f} | "
                         f"{row['macro_pairwise_box_jaccard']:.4f} | {row['least_consistent_family']} "
                         f"({row['min_family_pairwise_box_f1']:.4f}) |")
        lines.append("")
    lines += ["## Reading guidance", ""]
    lines += [f"- {note}" for note in METHODOLOGICAL_NOTES]
    lines += ["- Instruction-style prompts are compared with canonical prompts in "
              "`prompt_sensitivity_variant_differences.csv` (negative `canonical_f1_difference` = variant worse).",
              "- The prompt variants are a controlled sample of natural wordings, not an exhaustive language study; "
              "results apply to the tested models, prompts, datasets and protocol version.",
              ""]
    return lines


# --- entry point ------------------------------------------------------------------


def generate_sensitivity(run_dir: Path, run_config: dict[str, Any],
                         per_prompt_rows: list[dict[str, Any]],
                         prediction_rows: list[dict[str, Any]],
                         image_ids: list[str]) -> dict[str, Any]:
    """Write every prompt-sensitivity output (CSVs, JSON summary, report, plots)."""
    prompts = run_config.get("prompt_definitions") or []
    sens_rows = sensitivity_rows(per_prompt_rows)
    family_gt = validate_family_gt_counts(sens_rows) if sens_rows else []
    family_stats = family_statistics(sens_rows)
    variant_rows = variant_difference_rows(sens_rows)
    model_rows = model_macro_rows(family_stats, sens_rows)
    consistency_iou = float(run_config.get("consistency_iou_threshold") or DEFAULT_CONSISTENCY_IOU)
    image_rows: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in sens_rows}):
        by_prompt = predictions_by_prompt_for_model(prediction_rows, model)
        image_rows.extend(pairwise_image_rows(prompts, by_prompt, image_ids, model,
                                              run_config.get("dataset", "?"), consistency_iou))
    pair_rows = aggregate_pair_consistency(image_rows)
    family_consistency = aggregate_family_consistency(pair_rows)
    model_consistency = aggregate_model_consistency(family_consistency)

    write_csv(run_dir / "prompt_sensitivity_per_prompt.csv", _rounded(sens_rows))
    write_csv(run_dir / "prompt_sensitivity_per_family.csv", _rounded(family_stats))
    write_csv(run_dir / "prompt_sensitivity_variant_differences.csv", _rounded(variant_rows))
    write_csv(run_dir / "prompt_sensitivity_per_model.csv", _rounded(model_rows))
    write_csv(run_dir / "prompt_pair_consistency_per_image.csv", _rounded(image_rows, drop=("matched_iou_sum",)))
    write_csv(run_dir / "prompt_pair_consistency.csv", _rounded(pair_rows))
    write_csv(run_dir / "prompt_consistency_per_family.csv", _rounded(family_consistency))
    write_csv(run_dir / "prompt_consistency_per_model.csv", _rounded(model_consistency))

    plots = save_sensitivity_plots(run_dir, run_config, sens_rows, family_stats, model_rows, model_consistency)

    summary = {
        "evaluation_protocol": run_config.get("evaluation_protocol"),
        "protocol_version": run_config.get("protocol_version"),
        "protocol_hash": run_config.get("protocol_hash"),
        "dataset_manifest_hash": run_config.get("dataset_manifest_hash"),
        "dataset": run_config.get("dataset"), "split": run_config.get("split"),
        "models": run_config.get("models"),
        "thresholds": {"conf_threshold": run_config.get("conf_threshold"),
                       "iou_threshold": run_config.get("iou_threshold"),
                       "consistency_iou_threshold": consistency_iou,
                       "max_detections": run_config.get("max_detections")},
        "sensitivity_families": [prompt for prompt in prompts if prompt.get("sensitivity_family")],
        "family_ground_truth": family_gt,
        "per_prompt_metrics": _rounded(sens_rows),
        "per_family_sensitivity": _rounded(family_stats),
        "variant_differences": _rounded(variant_rows),
        "per_model_sensitivity": _rounded(model_rows),
        "prediction_consistency": {"per_pair": _rounded(pair_rows),
                                   "per_family": _rounded(family_consistency),
                                   "per_model": _rounded(model_consistency)},
        "plots": plots,
        "notes": METHODOLOGICAL_NOTES,
    }
    write_json(run_dir / "prompt_sensitivity_summary.json", summary)
    lines = _report_lines(run_config, family_stats, model_rows, model_consistency, family_gt)
    (run_dir / "prompt_sensitivity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
