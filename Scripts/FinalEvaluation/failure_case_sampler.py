#!/usr/bin/env python3
"""Sample dissertation-ready qualitative failure/success cases from existing outputs.

Matches predictions against ground truth (greedy IoU) and exports annotated
images per category:

    false_positive   prediction with no matching GT box
    false_negative   GT box no prediction matched
    small_object_fn  false negative whose GT box is <= 32x32 px (COCO 'small')
    duplicate        extra prediction on an already-matched GT box
    low_conf_tp      correct detection with confidence < 0.5
    clean_success    images with every GT matched and zero false positives

Prediction sources (no model is ever rerun unless YOU pass --model):
  * --dataset PromptDetect --latest | --run <folder>
        uses predictions.csv + ground_truth_index.csv from an existing
        Results/PromptDetect/BatchEvaluation run;
  * --dataset MDWD|MTSD --pred-labels <dir>
        YOLO-format prediction .txt files (class cx cy w h [conf]) named per
        image stem - e.g. produced by `yolo predict save_txt=True save_conf=True`;
  * --dataset MDWD|MTSD --model <selector>
        EXPLICIT opt-in: resolves a trained checkpoint via the
        inference-benchmark inventory ('best' = highest val mAP@50-95 in the
        MDWD summary CSVs) and runs it over the seeded sample to obtain
        predictions (class-aware matching). Light by design (--max-images).

If no prediction source is available the script reports that clearly and
exits without writing anything. Original images and result folders are never
modified; outputs go to a fresh timestamped folder under
Documents/Final-Figures/ErrorAnalysis/<DATASET>/.

Examples:
    python failure_case_sampler.py --dataset PromptDetect --latest --max-samples 25
    python failure_case_sampler.py --dataset MDWD --model best --max-samples 25
    python failure_case_sampler.py --dataset MTSD --model yolo26m --max-samples 25
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path = SCRIPT_DIR) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "Datasets").exists() and (parent / "Scripts").exists():
            return parent
    raise RuntimeError("Could not locate project root.")


ROOT = find_project_root()
OUT_ROOT = ROOT / "Documents" / "Final-Figures" / "ErrorAnalysis"
PROMPT_RUNS = ROOT / "Results" / "PromptDetect" / "BatchEvaluation"
MDWD_DATASET = ROOT / "Datasets" / "MDWD" / "MDWD-YOLO26"
MTSD_PREPARED = ROOT / "Datasets" / "MTSD" / "Prepared" / "MTSD-YOLO"
MDWD_SUMMARIES = ROOT / "Results" / "MDWD-Results"
BENCH_DIR = ROOT / "Scripts" / "Other-Scripts" / "Inference-Benchmark"

COLORS = {"gt": "#2a78d6", "tp": "#1baf7a", "false_positive": "#e34948",
          "duplicate": "#eb6834", "false_negative": "#eda100", "low_conf_tp": "#4a3aa7"}
SMALL_AREA = 32 * 32
LOW_CONF = 0.5


def iou(a: dict, b: dict) -> float:
    ix0, iy0 = max(a["x0"], b["x0"]), max(a["y0"], b["y0"])
    ix1, iy1 = min(a["x1"], b["x1"]), min(a["y1"], b["y1"])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = ((a["x1"] - a["x0"]) * (a["y1"] - a["y0"])
             + (b["x1"] - b["x0"]) * (b["y1"] - b["y0"]) - inter)
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Ground truth + prediction loading
# ---------------------------------------------------------------------------

def load_yolo_ground_truth(dataset_dir: Path, split: str, max_images: int, seed: int) -> list[dict]:
    import yaml
    from PIL import Image

    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset not found: {dataset_dir}"
                         + (" - run Prepare-MTSD-Detection-Dataset.ipynb first."
                            if "MTSD" in str(dataset_dir) else ""))
    names = list((yaml.safe_load((dataset_dir / "data.yaml").read_text(encoding="utf-8")) or {}).get("names", []))
    images_dir, labels_dir = dataset_dir / split / "images", dataset_dir / split / "labels"
    pool = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    rng = random.Random(seed)
    if max_images and max_images < len(pool):
        pool = sorted(rng.sample(pool, max_images))
    records = []
    for image_path in pool:
        with Image.open(image_path) as img:
            width, height = img.size
        boxes = []
        label_path = labels_dir / (image_path.stem + ".txt")
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                boxes.append({"class_id": class_id,
                              "class_name": names[class_id] if 0 <= class_id < len(names) else str(class_id),
                              "x0": (cx - bw / 2) * width, "y0": (cy - bh / 2) * height,
                              "x1": (cx + bw / 2) * width, "y1": (cy + bh / 2) * height})
        records.append({"image_id": image_path.name, "image_path": image_path, "boxes": boxes})
    return records


def load_pred_labels(pred_dir: Path, records: list[dict]) -> dict[str, list[dict]]:
    from PIL import Image

    preds: dict[str, list[dict]] = {}
    for record in records:
        label_path = pred_dir / (Path(record["image_id"]).stem + ".txt")
        if not label_path.exists():
            continue
        with Image.open(record["image_path"]) as img:
            width, height = img.size
        boxes = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            score = float(parts[5]) if len(parts) > 5 else 1.0
            boxes.append({"class_id": class_id, "score": score,
                          "x0": (cx - bw / 2) * width, "y0": (cy - bh / 2) * height,
                          "x1": (cx + bw / 2) * width, "y1": (cy + bh / 2) * height})
        preds[record["image_id"]] = boxes
    return preds


def resolve_checkpoint(selector: str, dataset: str) -> tuple[str, Path]:
    """Resolve --model via the inference-benchmark inventory; 'best' via summary CSVs."""
    if str(BENCH_DIR) not in sys.path:
        sys.path.insert(0, str(BENCH_DIR))
    from inference_speed_benchmark import build_inventory, select_models

    inventory = build_inventory()
    if selector.lower() == "best":
        if dataset != "MDWD":
            raise SystemExit("--model best is only defined for MDWD (uses the MDWD summary CSVs); "
                             "name a variant explicitly for MTSD.")
        best_variant, best_map = None, -1.0
        for summary_path in MDWD_SUMMARIES.glob("*/Model-Size-Comparison/*summary*.csv"):
            with summary_path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        value = float(row.get("mAP@50-95") or 0)
                    except ValueError:
                        continue
                    if value > best_map:
                        best_map, best_variant = value, row.get("model_variant")
        if not best_variant:
            raise SystemExit("Could not determine 'best' from the MDWD summary CSVs.")
        print(f"'best' resolved to {best_variant} (val mAP@50-95 {best_map:.4f})")
        selector = best_variant
    chosen, unmatched = select_models(inventory, [selector], "detection", dataset)
    chosen = [c for c in chosen if c.status == "ready"]
    if unmatched or not chosen:
        raise SystemExit(f"No trained checkpoint matches '{selector}' for {dataset} "
                         f"(see inference_speed_benchmark.py --list-models).")
    entry = chosen[0]
    if len(chosen) > 1:
        print(f"'{selector}' matched {len(chosen)} checkpoints; using {entry.id} ({entry.suite}).")
    if entry.family == "rfdetr":
        raise SystemExit("Live generation currently supports YOLO checkpoints only; "
                         "for RF-DETR export predictions first and use --pred-labels.")
    return entry.id, Path(entry.checkpoint)


def generate_predictions(checkpoint: Path, records: list[dict], conf: float) -> dict[str, list[dict]]:
    print(f"Generating predictions with {checkpoint.name} on {len(records)} images "
          f"(explicitly requested via --model)...")
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    preds: dict[str, list[dict]] = {}
    for record in records:
        result = model.predict(str(record["image_path"]), conf=conf, verbose=False)[0]
        boxes = []
        for xyxy, score, class_id in zip(result.boxes.xyxy.tolist(),
                                         result.boxes.conf.tolist(),
                                         result.boxes.cls.tolist()):
            boxes.append({"class_id": int(class_id), "score": float(score),
                          "x0": xyxy[0], "y0": xyxy[1], "x1": xyxy[2], "y1": xyxy[3]})
        preds[record["image_id"]] = boxes
    return preds


def load_promptdetect_run(run_dir: Path, model_filter: str | None) -> tuple[list[dict], dict, str]:
    gt_path, pred_path = run_dir / "ground_truth_index.csv", run_dir / "predictions.csv"
    if not gt_path.exists() or not pred_path.exists():
        raise SystemExit(f"{run_dir} is missing ground_truth_index.csv / predictions.csv.")
    records_by_id: dict[str, dict] = {}
    with gt_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            record = records_by_id.setdefault(
                row["image_id"], {"image_id": row["image_id"],
                                  "image_path": Path(row["image_path"]), "boxes": []})
            if row.get("x0"):
                record["boxes"].append({"class_name": row.get("class_name", ""),
                                        "class_id": -1,
                                        **{k: float(row[k]) for k in ("x0", "y0", "x1", "y1")}})
    preds: dict[str, list[dict]] = defaultdict(list)
    models_seen = set()
    with pred_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            models_seen.add(row["model"])
            if model_filter and model_filter.lower() not in row["model"].lower():
                continue
            if row.get("x0") in ("", None):
                continue
            preds[row["image_id"]].append({
                "class_id": -1, "score": float(row["score"] or 1.0),
                "label": f"{row['model']}|{row['prompt']}",
                **{k: float(row[k]) for k in ("x0", "y0", "x1", "y1")}})
    label = model_filter or "+".join(sorted(models_seen)) or "promptdetect"
    return list(records_by_id.values()), dict(preds), label


# ---------------------------------------------------------------------------
# Matching + rendering
# ---------------------------------------------------------------------------

def classify_image(gt_boxes: list[dict], pred_boxes: list[dict],
                   iou_threshold: float, class_aware: bool) -> dict:
    preds = sorted(pred_boxes, key=lambda p: -p["score"])
    matched_gt: set[int] = set()
    cases = {"tp": [], "false_positive": [], "duplicate": [], "low_conf_tp": []}
    for pred in preds:
        best_iou, best_index = 0.0, None
        for index, gt in enumerate(gt_boxes):
            if class_aware and pred.get("class_id", -1) != gt.get("class_id", -1):
                continue
            value = iou(pred, gt)
            if value > best_iou:
                best_iou, best_index = value, index
        if best_index is not None and best_iou >= iou_threshold:
            if best_index in matched_gt:
                cases["duplicate"].append((pred, best_iou))
            else:
                matched_gt.add(best_index)
                bucket = "low_conf_tp" if pred["score"] < LOW_CONF else "tp"
                cases[bucket].append((pred, best_iou))
        else:
            cases["false_positive"].append((pred, best_iou))
    cases["false_negative"] = [gt for i, gt in enumerate(gt_boxes) if i not in matched_gt]
    return cases


def render(record: dict, cases: dict, out_path: Path) -> bool:
    from PIL import Image, ImageDraw

    try:
        with Image.open(record["image_path"]) as img:
            canvas = img.convert("RGB")
    except Exception:
        return False
    draw = ImageDraw.Draw(canvas)
    line = max(2, canvas.width // 450)

    def box(b, colour, tag):
        draw.rectangle([b["x0"], b["y0"], b["x1"], b["y1"]], outline=colour, width=line)
        draw.text((b["x0"] + 3, max(0, b["y0"] - 14)), tag, fill=colour)

    for gt in record["boxes"]:
        box(gt, COLORS["gt"], gt.get("class_name", "GT"))
    for pred, value in cases["tp"]:
        box(pred, COLORS["tp"], f"TP {pred['score']:.2f}")
    for pred, value in cases["low_conf_tp"]:
        box(pred, COLORS["low_conf_tp"], f"TP(low) {pred['score']:.2f}")
    for pred, value in cases["false_positive"]:
        box(pred, COLORS["false_positive"], f"FP {pred['score']:.2f}")
    for pred, value in cases["duplicate"]:
        box(pred, COLORS["duplicate"], f"DUP {pred['score']:.2f}")
    for gt in cases["false_negative"]:
        box(gt, COLORS["false_negative"], "MISSED")
    if canvas.width > 1600:
        canvas = canvas.resize((1600, int(canvas.height * 1600 / canvas.width)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample qualitative error cases from existing outputs.")
    parser.add_argument("--dataset", required=True, choices=["MDWD", "MTSD", "PromptDetect"])
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--model", default=None,
                        help="MDWD/MTSD: checkpoint selector ('best', 'yolo26m', ...) - EXPLICIT "
                             "opt-in to run inference on the sample. PromptDetect: model-name filter.")
    parser.add_argument("--pred-labels", type=Path, default=None,
                        help="Folder of YOLO-format prediction .txt files (class cx cy w h [conf]).")
    parser.add_argument("--run", type=Path, default=None, help="PromptDetect batch-eval run folder.")
    parser.add_argument("--latest", action="store_true", help="Use the newest PromptDetect run.")
    parser.add_argument("--max-samples", type=int, default=25, help="Per category (default 25).")
    parser.add_argument("--max-images", type=int, default=200,
                        help="Image pool cap for MDWD/MTSD sampling/generation (default 200).")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--conf-threshold", type=float, default=0.30,
                        help="Confidence floor when generating predictions with --model.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---- load GT + predictions ------------------------------------------------
    class_aware = False
    if args.dataset == "PromptDetect":
        if args.run:
            run_dir = args.run if args.run.is_absolute() else ROOT / args.run
        else:
            runs = sorted(PROMPT_RUNS.glob("*/*/evaluation_summary.json")) if PROMPT_RUNS.is_dir() else []
            if not runs:
                print("No PromptDetect batch-evaluation runs found under "
                      f"{PROMPT_RUNS}.\nRun run_batch_eval.py first - nothing to sample; exiting.")
                return 0
            if not args.latest:
                print("Multiple/unspecified runs - pass --latest or --run <folder>. Newest is:\n  "
                      f"{runs[-1].parent}")
                return 0
            run_dir = runs[-1].parent
        records, preds, model_label = load_promptdetect_run(run_dir, args.model)
        source_note = str(run_dir.relative_to(ROOT))
    else:
        dataset_dir = MDWD_DATASET if args.dataset == "MDWD" else MTSD_PREPARED
        records = load_yolo_ground_truth(dataset_dir, args.split, args.max_images, args.seed)
        if args.pred_labels:
            preds = load_pred_labels(args.pred_labels, records)
            model_label = args.pred_labels.name
            source_note = str(args.pred_labels)
            class_aware = True
        elif args.model:
            model_label, checkpoint = resolve_checkpoint(args.model, args.dataset)
            preds = generate_predictions(checkpoint, records, args.conf_threshold)
            source_note = str(checkpoint.relative_to(ROOT))
            class_aware = True
        else:
            print(f"No prediction source for {args.dataset}: pass --pred-labels <dir> "
                  "(stored YOLO predictions) or --model <selector> (explicit opt-in to "
                  "run a trained checkpoint over the sample). Nothing written; exiting.")
            return 0

    # ---- classify ----------------------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = OUT_ROOT / args.dataset / f"{stamp}-{model_label.replace(' ', '_').replace('|', '_')}"
    buckets: dict[str, list[tuple]] = defaultdict(list)
    per_image_cases = {}
    for record in records:
        cases = classify_image(record["boxes"], preds.get(record["image_id"], []),
                               args.iou_threshold, class_aware)
        per_image_cases[record["image_id"]] = (record, cases)
        for pred, value in cases["false_positive"]:
            buckets["false_positive"].append((record, pred, value))
        for pred, value in cases["duplicate"]:
            buckets["duplicate"].append((record, pred, value))
        for pred, value in cases["low_conf_tp"]:
            buckets["low_conf_tp"].append((record, pred, value))
        for gt in cases["false_negative"]:
            area = (gt["x1"] - gt["x0"]) * (gt["y1"] - gt["y0"])
            bucket = "small_object_fn" if area <= SMALL_AREA else "false_negative"
            buckets[bucket].append((record, gt, 0.0))
        if record["boxes"] and not cases["false_positive"] and not cases["false_negative"] \
                and not cases["duplicate"]:
            buckets["clean_success"].append((record, None, 1.0))

    rng = random.Random(args.seed)
    index_rows, rendered_images = [], set()
    for category, items in sorted(buckets.items()):
        sample = items if len(items) <= args.max_samples else rng.sample(items, args.max_samples)
        for record, item, value in sample:
            out_path = out_dir / category / f"{Path(record['image_id']).stem}.jpg"
            if record["image_id"] not in rendered_images:
                _, cases = per_image_cases[record["image_id"]]
                if not render(record, cases, out_path):
                    continue
                rendered_images.add(record["image_id"])
            index_rows.append({
                "category": category, "image_id": record["image_id"],
                "model": model_label, "dataset": args.dataset,
                "score": round(item.get("score", 0.0), 4) if isinstance(item, dict) and "score" in item else "",
                "iou": round(value, 4) if value else "",
                "class": item.get("class_name", item.get("label", "")) if isinstance(item, dict) else "",
                "annotated_image": str(out_path.relative_to(ROOT)),
                "source": source_note,
            })

    if not index_rows:
        print("No cases found (no predictions overlapped the sample?) - nothing written.")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "error_case_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    (OUT_ROOT / "LATEST.txt").parent.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "LATEST.txt").write_text(str(out_dir) + "\n", encoding="utf-8")
    (out_dir / "sampler_config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n",
                                                 encoding="utf-8")

    print(f"\nCategories: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(buckets.items())))
    print(f"Rendered {len(rendered_images)} annotated images; index: "
          f"{out_dir / 'error_case_index.csv'}")
    print("Legend: GT blue, TP green, low-conf TP violet, FP red, duplicate orange, missed amber.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
