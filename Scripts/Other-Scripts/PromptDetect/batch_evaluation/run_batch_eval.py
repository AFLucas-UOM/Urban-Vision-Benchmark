#!/usr/bin/env python3
"""Batch evaluation of PromptDetect models against MDWD / MTSD ground truth.

Runs 1-15 text prompts through the selected prompt-based models (SAM 3,
SAM 3.1, Cosmos Reason2 variants, LocateAnything) over a dataset split, then
scores the predictions against the ground-truth boxes with IoU matching:
precision/recall/F1/accuracy, AP@50 and mAP@50:95, mean matched IoU, FP/FN
and duplicate counts, per-prompt and per-model breakdowns, plus a
prompt-vs-class confusion matrix and plots.

Safety:
* models must be selected explicitly (``--models``); nothing runs by default;
* heavy checkpoints (Cosmos Reason2 8B/32B) additionally require
  ``--allow-heavy``;
* ``--dry-run`` loads the dataset and prints the plan without touching any
  model; ``--max-images`` caps the run for smoke tests;
* every run writes into its own timestamped folder under
  Results/PromptDetect/BatchEvaluation/<DATASET>/ - nothing is overwritten.

Example:
    python run_batch_eval.py --dataset MTSD --split all \\
        --prompts "traffic sign" "stop sign" --models sam3 locateanything_3b \\
        --max-images 25 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import config  # noqa: E402
import metrics as metrics_mod  # noqa: E402
from dataset_loader import ground_truth_index_rows, load_ground_truth  # noqa: E402


def resolve_models(slugs: list[str], allow_heavy: bool) -> list[str]:
    registry = config.available_models()
    labels = []
    for slug in slugs:
        key = slug.lower()
        if key not in registry:
            options = ", ".join(sorted(registry))
            raise SystemExit(f"Unknown model '{slug}'. Available: {options}")
        labels.append(registry[key])
    heavy = [label for label in labels if label in config.HEAVY_MODEL_LABELS]
    if heavy and not allow_heavy:
        raise SystemExit(
            f"Heavy model(s) requested without --allow-heavy: {heavy}. "
            f"These need large VRAM/runtime; re-run with --allow-heavy to confirm."
        )
    if heavy:
        print(f"WARNING: heavy model(s) enabled: {heavy}")
    return labels


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plots(run_dir: Path, prompt_rows: list[dict], confusion: dict, class_names: list[str]) -> list[str]:
    saved = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"Plotting skipped (matplotlib unavailable: {exc})")
        return saved

    rows = [r for r in prompt_rows if r["prompt"] != "<all prompts combined>"]
    if rows:
        labels = [f"{r['model']}\n{r['prompt']}" for r in rows]
        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(rows)), 5))
        width = 0.27
        for offset, metric in ((-1, "precision"), (0, "recall"), (1, "f1")):
            ax.bar(x + offset * width, [r[metric] for r in rows], width, label=metric)
        ax.set_xticks(x, labels, fontsize=8, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title("Precision / Recall / F1 per model and prompt")
        ax.legend()
        fig.tight_layout()
        target = run_dir / "metrics_bars.png"
        fig.savefig(target, dpi=200)
        plt.close(fig)
        saved.append(target.name)

    if confusion:
        prompts = sorted(confusion)
        classes = [c for c in class_names if any(confusion[p].get(c) for p in prompts)]
        if classes:
            data = np.array([[confusion[p].get(c, 0) for c in classes] for p in prompts])
            fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(classes)), max(4, 0.5 * len(prompts))))
            mesh = ax.imshow(data, cmap="Blues")
            ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right", fontsize=8)
            ax.set_yticks(range(len(prompts)), prompts, fontsize=8)
            ax.set_title("Matched detections: prompt vs ground-truth class")
            for i in range(len(prompts)):
                for j in range(len(classes)):
                    if data[i, j]:
                        ax.text(j, i, str(data[i, j]), ha="center", va="center", fontsize=7,
                                color="white" if data[i, j] > data.max() * 0.6 else "#333333")
            fig.colorbar(mesh, ax=ax, shrink=0.8)
            fig.tight_layout()
            target = run_dir / "confusion_matrix.png"
            fig.savefig(target, dpi=200)
            plt.close(fig)
            saved.append(target.name)
    return saved


def save_visual_samples(run_dir: Path, gt: dict, predictions: list[dict],
                        model_labels: list[str], prompts: list[str], n_samples: int) -> int:
    from PIL import Image, ImageDraw

    from prompt_runner import prediction_boxes

    samples_dir = run_dir / "samples"
    rendered = 0
    for model in model_labels:
        grouped = {}
        for prompt in prompts:
            for image_id, boxes in prediction_boxes(predictions, model, prompt).items():
                grouped.setdefault(image_id, []).extend(boxes)
        for record in gt["records"][:n_samples]:
            try:
                with Image.open(record["image_path"]) as img:
                    canvas = img.convert("RGB")
            except Exception:
                continue
            draw = ImageDraw.Draw(canvas)
            line = max(2, canvas.width // 500)
            for box in record["boxes"]:
                draw.rectangle([box["x0"], box["y0"], box["x1"], box["y1"]],
                               outline="#2a78d6", width=line)          # GT blue
            for box in grouped.get(record["image_id"], []):
                draw.rectangle([box["x0"], box["y0"], box["x1"], box["y1"]],
                               outline="#e34948", width=line)          # predictions red
            if canvas.width > 1280:
                canvas = canvas.resize((1280, int(canvas.height * 1280 / canvas.width)))
            out = samples_dir / config.model_slug(model) / f"{Path(record['image_id']).name}"
            out.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(out.with_suffix(".jpg"), quality=88)
            rendered += 1
    return rendered


def run_evaluation(
    dataset: str, split: str, prompts: list[str], model_labels: list[str],
    max_images: int | None, conf_threshold: float, iou_threshold: float,
    save_visuals: int, progress=None,
) -> tuple[Path, dict]:
    """Full (non-dry) evaluation. Returns (run_dir, summary dict)."""
    from prompt_runner import prediction_boxes, run_models

    gt = load_ground_truth(dataset, split, max_images)
    gt_by_image = {r["image_id"]: r["boxes"] for r in gt["records"]}
    run_dir = config.new_run_dir(dataset)

    run_config = {
        "dataset": gt["dataset"], "split": gt["split"], "source": gt["source"],
        "prompts": prompts, "models": model_labels,
        "max_images": max_images, "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold, "map_iou_range": config.MAP_IOU_RANGE,
        "n_images": len(gt["records"]),
        "n_gt_boxes": sum(len(b) for b in gt_by_image.values()),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    write_csv(run_dir / "ground_truth_index.csv", ground_truth_index_rows(gt))

    predictions, statuses = run_models(
        gt, prompts, model_labels, conf_threshold, config.MAX_DETECTIONS, progress
    )
    write_csv(run_dir / "predictions.csv", predictions)

    prompt_rows, per_image_rows = [], []
    confusion_total: dict[str, dict[str, int]] = {}
    for model in model_labels:
        union_by_image: dict[str, list[dict]] = {}
        for prompt in prompts:
            preds_by_image = prediction_boxes(predictions, model, prompt)
            for image_id, boxes in preds_by_image.items():
                union_by_image.setdefault(image_id, []).extend(boxes)
            evaluation = metrics_mod.evaluate_prompt(
                preds_by_image, gt_by_image, iou_threshold, config.MAP_IOU_RANGE
            )
            prompt_rows.append({"model": model, "prompt": prompt, **evaluation["summary"]})
            per_image_rows.extend(
                {"model": model, "prompt": prompt, **row} for row in evaluation["per_image"]
            )
            for key, row in metrics_mod.confusion_matrix(evaluation["matches"], gt["class_names"]).items():
                bucket = confusion_total.setdefault(key, {})
                for class_name, count in row.items():
                    bucket[class_name] = bucket.get(class_name, 0) + count
        if len(prompts) > 1:
            evaluation = metrics_mod.evaluate_prompt(
                union_by_image, gt_by_image, iou_threshold, config.MAP_IOU_RANGE
            )
            prompt_rows.append({"model": model, "prompt": "<all prompts combined>",
                                **evaluation["summary"]})

    write_csv(run_dir / "per_prompt_metrics.csv", prompt_rows)
    write_csv(run_dir / "per_image_metrics.csv", per_image_rows)
    per_class_rows = [
        {"prompt": prompt, "gt_class": class_name, "matched_detections": count}
        for prompt, row in sorted(confusion_total.items())
        for class_name, count in sorted(row.items())
    ]
    write_csv(run_dir / "per_class_matches.csv", per_class_rows)
    plots = save_plots(run_dir, prompt_rows, confusion_total, gt["class_names"])

    rendered = 0
    if save_visuals > 0:
        rendered = save_visual_samples(run_dir, gt, predictions, model_labels, prompts, save_visuals)

    summary = {
        **run_config,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "model_statuses": statuses,
        "per_prompt_metrics": prompt_rows,
        "confusion_prompt_vs_class": confusion_total,
        "plots": plots,
        "visual_samples": rendered,
        "notes": [
            "Cosmos/LocateAnything emit no per-box confidence (backend assigns 1.0), "
            "so their AP values collapse to a single precision/recall point.",
            "Metrics are class-agnostic per prompt: every prompt is scored against "
            "ALL ground-truth boxes; the confusion matrix shows which classes each "
            "prompt actually matched.",
        ],
    }
    (run_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return run_dir, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="PromptDetect batch evaluation vs ground truth.")
    parser.add_argument("--dataset", required=True, choices=["MDWD", "MTSD", "mdwd", "mtsd"])
    parser.add_argument("--split", default="test",
                        help="train / valid / test, or 'all' (MTSD QA fallback).")
    parser.add_argument("--prompts", nargs="+", required=True, metavar="PROMPT",
                        help=f"{config.MIN_PROMPTS}-{config.MAX_PROMPTS} text prompts.")
    parser.add_argument("--models", nargs="+", required=True, metavar="MODEL",
                        help=f"Model slugs: {', '.join(sorted(config.available_models()))}")
    parser.add_argument("--allow-heavy", action="store_true",
                        help=f"Required to run heavy models: {sorted(config.HEAVY_MODEL_LABELS)}")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--conf-threshold", type=float, default=config.CONF_THRESHOLD)
    parser.add_argument("--iou-threshold", type=float, default=config.IOU_MATCH_THRESHOLD)
    parser.add_argument("--save-visuals", type=int, default=0,
                        help="Render GT-vs-prediction overlays for the first N images per model.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load the dataset and print the plan; run no models, write nothing.")
    args = parser.parse_args()

    if not (config.MIN_PROMPTS <= len(args.prompts) <= config.MAX_PROMPTS):
        raise SystemExit(f"Provide between {config.MIN_PROMPTS} and {config.MAX_PROMPTS} prompts "
                         f"(got {len(args.prompts)}).")
    model_labels = resolve_models(args.models, args.allow_heavy)

    gt = load_ground_truth(args.dataset, args.split, args.max_images)
    n_boxes = sum(len(r["boxes"]) for r in gt["records"])
    print(f"Dataset : {gt['dataset']} split={gt['split']} ({gt['source']})")
    print(f"Images  : {len(gt['records'])}  |  GT boxes: {n_boxes}  |  classes: {len(gt['class_names'])}")
    print(f"Prompts : {args.prompts}")
    print(f"Models  : {model_labels}")
    print(f"Thresholds: conf={args.conf_threshold}  IoU={args.iou_threshold}")

    if args.dry_run:
        estimated = len(gt["records"]) * len(args.prompts) * len(model_labels)
        print(f"\nDRY RUN - no models loaded, nothing written.")
        print(f"A full run would execute ~{estimated:,} predict() calls "
              f"({len(model_labels)} model(s) x {len(args.prompts)} prompt(s) x "
              f"{len(gt['records'])} image(s)).")
        return 0

    run_dir, summary = run_evaluation(
        args.dataset, args.split, args.prompts, model_labels,
        args.max_images, args.conf_threshold, args.iou_threshold, args.save_visuals,
    )
    print(f"\nResults written to: {run_dir}")
    for row in summary["per_prompt_metrics"]:
        print(f"  {row['model']:<22} {row['prompt']:<28} "
              f"P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f} "
              f"AP50={row['ap50']:.3f} mAP50-95={row['map50_95']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
