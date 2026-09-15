#!/usr/bin/env python3
"""Robustness slice analysis over stored evaluation evidence (read-only).

Analyses how performance changes across meaningful data slices instead of a
single aggregate score. It never runs model inference: it consumes stored
predictions and ground truth only, and reports an analysis as PENDING (with
the command that would produce its inputs) when they do not exist.

Workstreams and their evidence sources:
  * MDWD supervised detection - stored Ultralytics predictions.json files
    from the leakage sensitivity re-evaluation
    (Documents/Final-Reports/MDWD-Leakage-Analysis/scratch/runs/*__original/),
    which cover the full official valid/test splits with the original
    protocol; ground truth from Datasets/MDWD/MDWD-YOLO26.
  * MTSD supervised detection - stored unified COCO predictions from completed
    run folders under Results/MTSD-Runs.  This is an inventory analysis: runs
    remain distinguishable by their individual protocol/run identifier.
  * MTSD attribute classification - per-head/per-class metrics stored by the
    completed GRP-1..GRP-3 snapshot round (historical status). Image-property
    slices need per-crop predictions the pipeline does not store - reported
    as missing, not recomputed.
  * PromptDetect MDWD - stored batch-evaluation runs (predictions.csv +
    ground_truth_index.csv); current runs are pilots and are labelled so.
  * PromptDetect MTSD - PENDING until a batch evaluation exists.

Outputs (timestamped; --overwrite writes to .../latest/):
    Documents/Final-Reports/Robustness-Slices/<stamp>/
        robustness_slice_summary.md      robustness_slice_results.csv
        robustness_slice_results.json    robustness_slice_config.json
        insufficient_or_missing_inputs.md
    Documents/Final-Figures/Robustness-Slices/<stamp>/   (PNG + PDF + SVG)

Usage:
    python Scripts/FinalEvaluation/robustness_slice_analysis.py --self-test
    python Scripts/FinalEvaluation/robustness_slice_analysis.py --dry-run
    python Scripts/FinalEvaluation/robustness_slice_analysis.py --dataset MDWD --task detection
    python Scripts/FinalEvaluation/robustness_slice_analysis.py            # all available
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evidence_lib as ev  # noqa: E402

GENERATOR = "Scripts/FinalEvaluation/robustness_slice_analysis.py"
WORKSTREAMS = ("MDWD", "MTSD", "PromptDetect-MDWD", "PromptDetect-MTSD")

CANONICAL_SAM3_RUNS = {
    "MDWD": ev.PROMPT_RESULTS / "MDWD" /
            "20260821-170445-optimized-v2-bounded-sensitivity",
    "MTSD": ev.PROMPT_RESULTS / "MTSD" /
            "20260822-102455-optimized-v2-bounded-sensitivity",
}
RETAINED_DETECTION_SLICE_CONFIGS = {
    "MDWD": ev.REPORTS_ROOT / "Robustness-Slices" / "20260908-103240" /
            "robustness_slice_config.json",
    "MTSD": ev.REPORTS_ROOT / "Robustness-Slices" / "20260914-123609" /
            "robustness_slice_config.json",
}


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------

def slice_row(*, workstream: str, status: str, model: str, run: str, split: str,
              dimension: str, value: str, metric: str, metric_value: float,
              n_images: int, n_objects: int, min_support: int,
              aggregate: float | None, source: str, snapshot: str = "",
              notes: str = "", ci: dict | None = None) -> dict:
    support = n_objects if dimension in OBJECT_DIMENSIONS else n_images
    row = {
        "workstream": workstream, "status": status, "model": model, "run": run,
        "split": split, "slice_dimension": dimension, "slice_value": value,
        "metric": metric, "value": round(float(metric_value), 4),
        "n_images": n_images, "n_objects": n_objects,
        "support_ok": "ok" if support >= min_support else "insufficient_support",
        "aggregate_value": round(float(aggregate), 4) if aggregate is not None else "",
        "delta_vs_aggregate": (round(float(metric_value) - float(aggregate), 4)
                               if aggregate is not None else ""),
        "source_file": source, "snapshot": snapshot, "notes": notes,
    }
    if ci:
        row["ci_low"], row["ci_high"] = ci["low"], ci["high"]
    return row


OBJECT_DIMENSIONS = {"object_size", "object_position", "class", "class_frequency"}
CSV_FIELDS = ["workstream", "status", "model", "run", "split", "slice_dimension",
              "slice_value", "metric", "value", "ci_low", "ci_high", "n_images",
              "n_objects", "support_ok", "aggregate_value", "delta_vs_aggregate",
              "source_file", "snapshot", "notes"]


# ---------------------------------------------------------------------------
# MDWD supervised detection slices (stored predictions)
# ---------------------------------------------------------------------------

def analyse_mdwd_detection(args, pd_metrics, missing: list[str]) -> tuple[list[dict], dict]:
    runs = ev.discover_leakage_prediction_runs()
    if args.run not in ("", "latest"):
        override = Path(args.run)
        runs = [r for r in runs if override in (r["run_dir"], r["run_dir"].parent)] or runs
    if not runs:
        missing.append(
            "MDWD detection: no stored per-image predictions found under "
            f"`{ev.rel(ev.LEAKAGE_RUNS)}`. Produce them first with:\n"
            "`python Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py "
            "--run-inference --models best-per-family`")
        return [], {}

    rows: list[dict] = []
    thresholds_used: dict = {}
    gt_cache: dict[str, dict] = {}
    stats_cache: dict[str, dict] = {}

    for run in runs:
        split = run["split"]
        if split not in gt_cache:
            gt_cache[split] = ev.load_mdwd_split_gt(split)
            stats_cache[split] = {stem: ev.image_stats(record["path"])
                                  for stem, record in gt_cache[split]["records"].items()}
        gt = gt_cache[split]
        stats = stats_cache[split]
        class_names = gt["class_names"]
        preds = ev.load_leakage_predictions(run["predictions"], class_names,
                                            min_score=args.working_conf)
        source = ev.rel(run["predictions"])
        run_id = run["run_dir"].name
        model = f"{run['model']}@{run['suite']}"

        # Per-image matching --------------------------------------------------
        per_image: dict[str, dict] = {}
        class_counts: dict[str, Counter] = defaultdict(Counter)
        gt_records_all: list[dict] = []
        for stem, record in gt["records"].items():
            result = ev.match_detections(pd_metrics, preds.get(stem, []),
                                         record["boxes"], iou_threshold=0.5)
            per_image[stem] = result
            for gt_record in result["gt_records"]:
                gt_record["image"] = stem
                gt_records_all.append(gt_record)
                class_counts[gt_record["class_name"]]["gt"] += 1
                class_counts[gt_record["class_name"]]["tp"] += int(gt_record["matched"])
            for pred_record in result["pred_records"]:
                if not pred_record["is_tp"]:
                    class_counts[pred_record["class_name"]]["fp"] += 1

        totals = {k: sum(r[k] for r in per_image.values()) for k in ("tp", "fp", "fn")}
        aggregate = ev.prf_from_counts(**totals)
        matched_ious = [g["iou"] for g in gt_records_all if g["matched"]]
        aggregate_miou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
        n_images_total = len(gt["records"])
        n_objects_total = len(gt_records_all)

        def base(**kw) -> dict:
            return slice_row(workstream="MDWD-detection", status=ev.STATUS_FINAL,
                             model=model, run=run_id, split=split,
                             min_support=args.min_support, source=source,
                             snapshot="Roboflow v20 (MDWD-YOLO26)", **kw)

        for metric_name, metric_value in [*aggregate.items(),
                                          ("mean_matched_iou", aggregate_miou)]:
            rows.append(base(dimension="aggregate", value="all", metric=metric_name,
                             metric_value=metric_value, n_images=n_images_total,
                             n_objects=n_objects_total, aggregate=None))

        # Image-level slices ---------------------------------------------------
        split_thresholds = {}
        for feature in ("brightness", "contrast", "sharpness"):
            values = [stats[stem][feature] for stem in gt["records"]]
            split_thresholds[feature] = ev.tercile_thresholds(values)
        thresholds_used[f"MDWD/{split}"] = {
            f: {"tercile_33": round(t[0], 2), "tercile_67": round(t[1], 2)}
            for f, t in split_thresholds.items()}

        def image_slices(assign) -> dict[str, list[str]]:
            groups: dict[str, list[str]] = defaultdict(list)
            for stem in gt["records"]:
                groups[assign(stem)].append(stem)
            return groups

        dimension_assigners = {
            "brightness": lambda stem: ev.tercile_label(
                stats[stem]["brightness"], split_thresholds["brightness"]),
            "contrast": lambda stem: ev.tercile_label(
                stats[stem]["contrast"], split_thresholds["contrast"]),
            "sharpness": lambda stem: ev.tercile_label(
                stats[stem]["sharpness"], split_thresholds["sharpness"],
                ("blurred", "intermediate", "sharp")),
            "object_density": lambda stem: ev.density_label(
                len(gt["records"][stem]["boxes"])),
        }
        for dimension, assign in dimension_assigners.items():
            for value, stems in sorted(image_slices(assign).items()):
                counts = {k: sum(per_image[s][k] for s in stems) for k in ("tp", "fp", "fn")}
                slice_metrics = ev.prf_from_counts(**counts)
                n_objects = sum(len(gt["records"][s]["boxes"]) for s in stems)
                for metric_name, metric_value in slice_metrics.items():
                    ci = None
                    if args.bootstrap_samples and metric_name == "f1":
                        units = [(per_image[s]["tp"], per_image[s]["fp"], per_image[s]["fn"])
                                 for s in stems]
                        ci = ev.bootstrap_ci(
                            units, lambda u: ev.prf_from_counts(
                                sum(x[0] for x in u), sum(x[1] for x in u),
                                sum(x[2] for x in u))["f1"],
                            n_bootstrap=args.bootstrap_samples, seed=args.seed)
                    rows.append(base(dimension=dimension, value=value,
                                     metric=metric_name, metric_value=metric_value,
                                     n_images=len(stems), n_objects=n_objects,
                                     aggregate=aggregate[metric_name], ci=ci))

        # Object-level slices (recall / matched IoU over GT boxes) --------------
        def object_slices(assign) -> dict[str, list[dict]]:
            groups: dict[str, list[dict]] = defaultdict(list)
            for gt_record in gt_records_all:
                groups[assign(gt_record)].append(gt_record)
            return groups

        support_by_class = Counter(g["class_name"] for g in gt_records_all)
        median_support = sorted(support_by_class.values())[len(support_by_class) // 2]
        object_assigners = {
            "object_size": lambda g: ev.coco_size_label(
                max(0.0, (g["x1"] - g["x0"]) * (g["y1"] - g["y0"]))),
            "object_position": lambda g: ev.position_label(
                g, gt["records"][g["image"]]["width"], gt["records"][g["image"]]["height"]),
            "class": lambda g: g["class_name"],
            "class_frequency": lambda g: ("common" if support_by_class[g["class_name"]]
                                          >= median_support else "rare"),
        }
        for dimension, assign in object_assigners.items():
            for value, group in sorted(object_slices(assign).items()):
                matched = [g for g in group if g["matched"]]
                recall = len(matched) / len(group) if group else 0.0
                miou = (sum(g["iou"] for g in matched) / len(matched)) if matched else 0.0
                images = {g["image"] for g in group}
                extra = {}
                if dimension == "class":
                    counts = class_counts[value]
                    extra = ev.prf_from_counts(counts["tp"], counts["fp"],
                                               counts["gt"] - counts["tp"])
                for metric_name, metric_value in [("recall", recall),
                                                  ("mean_matched_iou", miou),
                                                  *((k, v) for k, v in extra.items()
                                                    if k != "recall")]:
                    agg = (aggregate.get(metric_name) if metric_name in aggregate
                           else aggregate_miou if metric_name == "mean_matched_iou" else None)
                    rows.append(base(dimension=dimension, value=value,
                                     metric=metric_name, metric_value=metric_value,
                                     n_images=len(images), n_objects=len(group),
                                     aggregate=agg))
    missing.append(
        "MDWD detection: capture-group and geographic slices are not applicable - "
        "the Roboflow export carries no capture-group or GPS metadata.")
    thresholds_used["MDWD/notes"] = {
        "object_size": "COCO-style thresholds (area < 32^2 small, < 96^2 medium) in "
                       "resized 640x640 evaluation pixels",
        "working_confidence": args.working_conf,
        "iou_threshold": 0.5,
        "matching": "class-aware greedy (PromptDetect batch_evaluation/metrics.py)"}
    return rows, thresholds_used


# ---------------------------------------------------------------------------
# MTSD supervised detection slices (stored unified COCO predictions)
# ---------------------------------------------------------------------------

def _mtsd_prepared_variant(run_dir: Path) -> Path:
    """Return the prepared COCO root implied by a recorded MTSD run name."""
    name = run_dir.name.lower()
    if "noaug" in name:
        variant = "MTSD-Unaugmented"
    elif "strong" in name:
        variant = "MTSD-Augmented-Strong"
    else:
        variant = "MTSD-Augmented"
    return ev.ROOT / "Datasets" / "MTSD" / "Prepared" / variant / "MTSD-COCO"


def _mtsd_annotation_path(run_dir: Path, split: str) -> tuple[Path | None, Path | None]:
    """Find the exact unified-eval annotation when retained, else its source COCO file."""
    eval_names = ("unified_evaluation_validation", "unified_evaluation_valid") if split == "valid" else (
        "unified_evaluation_test", "unified_evaluation")
    for directory in eval_names:
        path = run_dir / directory / "evaluation_annotations.coco.json"
        if path.exists():
            return path, _mtsd_prepared_variant(run_dir) / split
    path = _mtsd_prepared_variant(run_dir) / split / "_annotations.coco.json"
    return (path, path.parent) if path.exists() else (None, None)


def _load_mtsd_gt(run_dir: Path, split: str) -> tuple[dict | None, str]:
    """Load MTSD COCO GT in the matcher schema, preserving unified exclusions."""
    annotation_path, image_dir = _mtsd_annotation_path(run_dir, split)
    if annotation_path is None or image_dir is None:
        return None, ""
    payload = ev.read_json(annotation_path)
    record = ev.read_json(run_dir / "run_record.json")
    excluded = set(record.get("evaluation_excluded_categories", ["Tourist Sign"]))
    categories = {int(row["id"]): row["name"] for row in payload.get("categories", [])
                  if row.get("name") not in excluded}
    images = {int(row["id"]): row for row in payload.get("images", [])}
    records = {
        image_id: {
            "path": image_dir / row["file_name"],
            "width": float(row["width"]), "height": float(row["height"]),
            "boxes": [],
        }
        for image_id, row in images.items()
    }
    for row in payload.get("annotations", []):
        category_id = int(row["category_id"])
        image_id = int(row["image_id"])
        if category_id not in categories or image_id not in records:
            continue
        x, y, w, h = (float(value) for value in row["bbox"])
        records[image_id]["boxes"].append({
            "class_name": categories[category_id], "x0": x, "y0": y,
            "x1": x + w, "y1": y + h,
        })
    return {"records": records, "class_names": categories}, ev.rel(annotation_path)


def _load_mtsd_predictions(path: Path, class_names: dict[int, str], min_score: float) -> dict[int, list[dict]]:
    predictions: dict[int, list[dict]] = defaultdict(list)
    for row in ev.read_json(path):
        category_id = int(row.get("category_id", -1))
        score = float(row.get("score", 0.0))
        if category_id not in class_names or score < min_score:
            continue
        x, y, w, h = (float(value) for value in row["bbox"])
        predictions[int(row["image_id"])].append({
            "class_name": class_names[category_id], "x0": x, "y0": y,
            "x1": x + w, "y1": y + h, "score": score,
        })
    return predictions


def analyse_mtsd_detection(args, pd_metrics, missing: list[str]) -> tuple[list[dict], dict]:
    """Calculate slices for every completed stored unified MTSD prediction export.

    The inventory intentionally retains distinct runs rather than silently selecting
    a winner.  Results therefore support within-run robustness statements; users
    must restrict comparisons to matching split/taxonomy/protocol conditions.
    """
    prediction_files = sorted((ev.ROOT / "Results" / "MTSD-Runs").rglob("unified_*_predictions.json"))
    rows: list[dict] = []
    thresholds_used: dict = {}
    gt_cache: dict[tuple[str, str], tuple[dict, str]] = {}
    stats_cache: dict[tuple[str, str], dict] = {}
    skipped = 0

    for prediction_path in prediction_files:
        run_dir = prediction_path.parent
        record_path = run_dir / "run_record.json"
        if not record_path.exists():
            skipped += 1
            continue
        record = ev.read_json(record_path)
        if record.get("status") != "completed":
            skipped += 1
            continue
        split = "valid" if "validation" in prediction_path.name else "test"
        gt, annotation_source = _load_mtsd_gt(run_dir, split)
        if not gt or not gt["records"]:
            skipped += 1
            continue
        cache_key = (annotation_source, split)
        if cache_key not in gt_cache:
            gt_cache[cache_key] = (gt, annotation_source)
            stats_cache[cache_key] = {
                image_id: ev.image_stats(data["path"])
                for image_id, data in gt["records"].items() if data["path"].exists()
            }
        gt, annotation_source = gt_cache[cache_key]
        stats = stats_cache[cache_key]
        preds = _load_mtsd_predictions(prediction_path, gt["class_names"], args.working_conf)
        per_image: dict[int, dict] = {}
        class_counts: dict[str, Counter] = defaultdict(Counter)
        gt_records_all: list[dict] = []
        for image_id, image in gt["records"].items():
            result = ev.match_detections(pd_metrics, preds.get(image_id, []), image["boxes"], 0.5)
            per_image[image_id] = result
            for gt_record in result["gt_records"]:
                gt_record["image"] = image_id
                gt_records_all.append(gt_record)
                class_counts[gt_record["class_name"]]["gt"] += 1
                class_counts[gt_record["class_name"]]["tp"] += int(gt_record["matched"])
            for pred_record in result["pred_records"]:
                if not pred_record["is_tp"]:
                    class_counts[pred_record["class_name"]]["fp"] += 1

        totals = {key: sum(result[key] for result in per_image.values()) for key in ("tp", "fp", "fn")}
        aggregate = ev.prf_from_counts(**totals)
        matched = [row for row in gt_records_all if row["matched"]]
        aggregate_miou = sum(row["iou"] for row in matched) / len(matched) if matched else 0.0
        model = f"{record.get('family', 'MTSD')} {record.get('scale', '')}".strip()
        run_id = ev.rel(run_dir)

        def base(**kw) -> dict:
            return slice_row(workstream="MTSD-detection", status=ev.STATUS_HISTORICAL,
                             model=model, run=run_id, split=split, min_support=args.min_support,
                             source=ev.rel(prediction_path),
                             snapshot="completed unified-prediction run inventory; no final shortlist applied",
                             **kw)

        for metric_name, metric_value in [*aggregate.items(), ("mean_matched_iou", aggregate_miou)]:
            rows.append(base(dimension="aggregate", value="all", metric=metric_name,
                             metric_value=metric_value, n_images=len(per_image),
                             n_objects=len(gt_records_all), aggregate=None))

        split_thresholds = {
            feature: ev.tercile_thresholds([value[feature] for value in stats.values()])
            for feature in ("brightness", "contrast", "sharpness")
        }
        thresholds_used[f"MTSD/{run_dir.name}/{split}"] = {
            **{feature: {"tercile_33": round(values[0], 2), "tercile_67": round(values[1], 2)}
               for feature, values in split_thresholds.items()},
            "annotation_source": annotation_source,
        }
        image_assigners = {
            "brightness": lambda image_id: ev.tercile_label(stats[image_id]["brightness"], split_thresholds["brightness"]),
            "contrast": lambda image_id: ev.tercile_label(stats[image_id]["contrast"], split_thresholds["contrast"]),
            "sharpness": lambda image_id: ev.tercile_label(stats[image_id]["sharpness"], split_thresholds["sharpness"], ("blurred", "intermediate", "sharp")),
            "object_density": lambda image_id: ev.density_label(len(gt["records"][image_id]["boxes"])),
        }
        for dimension, assign in image_assigners.items():
            groups: dict[str, list[int]] = defaultdict(list)
            for image_id in per_image:
                if image_id in stats or dimension == "object_density":
                    groups[assign(image_id)].append(image_id)
            for value, ids in sorted(groups.items()):
                counts = {key: sum(per_image[image_id][key] for image_id in ids) for key in ("tp", "fp", "fn")}
                for metric_name, metric_value in ev.prf_from_counts(**counts).items():
                    rows.append(base(dimension=dimension, value=value, metric=metric_name,
                                     metric_value=metric_value, n_images=len(ids),
                                     n_objects=sum(len(gt["records"][image_id]["boxes"]) for image_id in ids),
                                     aggregate=aggregate[metric_name]))

        support_by_class = Counter(row["class_name"] for row in gt_records_all)
        median_support = sorted(support_by_class.values())[len(support_by_class) // 2]
        object_assigners = {
            "object_size": lambda row: ev.coco_size_label(max(0.0, (row["x1"] - row["x0"]) * (row["y1"] - row["y0"]))),
            "object_position": lambda row: ev.position_label(row, gt["records"][row["image"]]["width"], gt["records"][row["image"]]["height"]),
            "class": lambda row: row["class_name"],
            "class_frequency": lambda row: "common" if support_by_class[row["class_name"]] >= median_support else "rare",
        }
        for dimension, assign in object_assigners.items():
            groups: dict[str, list[dict]] = defaultdict(list)
            for gt_record in gt_records_all:
                groups[assign(gt_record)].append(gt_record)
            for value, group in sorted(groups.items()):
                matched_group = [row for row in group if row["matched"]]
                values = [("recall", len(matched_group) / len(group) if group else 0.0),
                          ("mean_matched_iou", sum(row["iou"] for row in matched_group) / len(matched_group) if matched_group else 0.0)]
                if dimension == "class":
                    counts = class_counts[value]
                    values.extend((key, metric) for key, metric in ev.prf_from_counts(
                        counts["tp"], counts["fp"], counts["gt"] - counts["tp"]).items() if key != "recall")
                for metric_name, metric_value in values:
                    overall = aggregate.get(metric_name, aggregate_miou if metric_name == "mean_matched_iou" else None)
                    rows.append(base(dimension=dimension, value=value, metric=metric_name,
                                     metric_value=metric_value, n_images=len({row["image"] for row in group}),
                                     n_objects=len(group), aggregate=overall))

    if not rows:
        missing.append("MTSD supervised detection: no completed, compatible unified prediction exports were found.")
    elif skipped:
        missing.append(f"MTSD supervised detection: skipped {skipped} prediction exports without a completed run record or compatible COCO ground truth.")
    return rows, thresholds_used


# ---------------------------------------------------------------------------
# MTSD attribute classification slices (stored per-head metrics)
# ---------------------------------------------------------------------------

def analyse_attributes(args, missing: list[str]) -> list[dict]:
    variants = ev.discover_attribute_variants(include_smoke=False)
    if not variants:
        missing.append(
            "MTSD attribute classification: no completed variants found under "
            f"`{ev.rel(ev.ATTR_OUTPUTS / 'metrics')}` - run "
            "`Scripts/MTSD-Scripts/AttributeClassification/run_all.py` first.")
        return []
    rows: list[dict] = []
    for variant in variants:
        payload = variant["payload"]
        source = ev.rel(variant["metrics_path"])
        snapshot = "GRP-1..GRP-3 manifest snapshot (historical)"
        run_id = payload.get("run_id", variant["variant"])
        for head, head_payload in payload.get("attributes", {}).items():
            n = int(head_payload.get("n", 0))

            def head_row(**kw) -> dict:
                return slice_row(workstream="MTSD-attributes", status=variant["status"],
                                 model=variant["variant"], run=run_id, split="test",
                                 min_support=args.min_support, source=source,
                                 snapshot=snapshot, **kw)

            for metric_name in ("accuracy", "macro_f1"):
                rows.append(head_row(dimension="attribute_head", value=head,
                                     metric=metric_name,
                                     metric_value=head_payload.get(metric_name, 0.0),
                                     n_images=n, n_objects=n, aggregate=None))
            macro = head_payload.get("macro_f1")
            for class_name, f1 in head_payload.get("per_class_f1", {}).items():
                support = int(head_payload.get("support", {}).get(class_name, 0))
                rows.append(head_row(dimension=f"{head}_class", value=class_name,
                                     metric="f1", metric_value=f1,
                                     n_images=support, n_objects=support,
                                     aggregate=macro,
                                     notes="per-class F1 vs head macro-F1"))
    missing.append(
        "MTSD attribute classification: object-size / brightness / blur / "
        "capture-group slices need per-crop predictions, which the training "
        "pipeline does not store (only aggregate metrics + confusion matrices). "
        "To enable them, extend `mtsd_attr` evaluation to dump per-crop "
        "predictions and rerun the (final) attribute round - do not rerun just "
        "for this analysis while the annotation scope is still open.")
    return rows


# ---------------------------------------------------------------------------
# PromptDetect slices (stored batch-eval runs; pilots labelled as such)
# ---------------------------------------------------------------------------

def analyse_promptdetect(args, pd_metrics, dataset: str, missing: list[str]) -> tuple[list[dict], dict]:
    runs = ev.discover_promptdetect_runs(dataset)
    if args.run not in ("", "latest"):
        override = Path(args.run)
        runs = [r for r in runs if r["run_dir"] == override] or runs
    elif runs:
        runs = [runs[-1]]  # latest run per dataset by timestamp ordering
    if not runs:
        missing.append(
            f"PromptDetect {dataset}: no batch-evaluation runs under "
            f"`{ev.rel(ev.PROMPT_RESULTS / dataset)}` - run "
            "`Scripts/Other-Scripts/PromptDetect/batch_evaluation/run_batch_eval.py "
            f"--dataset {dataset} --split test --prompts ... --models ...` first.")
        return [], {}

    rows: list[dict] = []
    thresholds_used: dict = {}
    for run in runs:
        data = ev.load_promptdetect_run(run["run_dir"])
        config = data["config"]
        iou_threshold = float(config.get("iou_threshold", 0.5))
        source = ev.rel(run["run_dir"] / "predictions.csv")
        status = run["status"]
        if status == ev.STATUS_PILOT:
            missing.append(
                f"PromptDetect {dataset} run `{run['run_id']}` is a PILOT "
                f"({run['n_images']} images) - slices are reported for pipeline "
                "validation only and are not dissertation evidence.")

        # Image stats + sizes (rebased onto this checkout) ---------------------
        stats: dict[str, dict] = {}
        dims: dict[str, tuple[float, float]] = {}
        for image_id, record in data["gt_by_image"].items():
            local = ev.rebase_stored_path(record["stored_path"])
            if local and local.exists():
                stats[image_id] = ev.image_stats(local)
                from PIL import Image

                with Image.open(local) as image:
                    dims[image_id] = image.size
            else:
                dims[image_id] = (640.0, 640.0)
        feature_thresholds = {
            feature: ev.tercile_thresholds([s[feature] for s in stats.values()])
            for feature in ("brightness", "contrast", "sharpness")} if stats else {}
        thresholds_used[f"PromptDetect-{dataset}/{run['run_id']}"] = {
            f: {"tercile_33": round(t[0], 2), "tercile_67": round(t[1], 2)}
            for f, t in feature_thresholds.items()}

        pairs = sorted({(model, prompt) for (model, prompt, _) in data["predictions"]})
        for model, prompt in pairs:
            per_image: dict[str, dict] = {}
            gt_records_all: list[dict] = []
            no_confidence = False
            for image_id, record in data["gt_by_image"].items():
                preds = data["predictions"].get((model, prompt, image_id), [])
                if preds and not any(p["has_confidence"] for p in preds):
                    no_confidence = True
                # Class-agnostic matching: prompts are not GT class names.
                result = ev.match_detections(pd_metrics, preds, record["boxes"],
                                             iou_threshold=iou_threshold,
                                             class_aware=False)
                per_image[image_id] = result
                for gt_record in result["gt_records"]:
                    gt_record["image"] = image_id
                    gt_records_all.append(gt_record)
            totals = {k: sum(r[k] for r in per_image.values()) for k in ("tp", "fp", "fn")}
            aggregate = ev.prf_from_counts(**totals)
            matched = [g for g in gt_records_all if g["matched"]]
            aggregate_miou = (sum(g["iou"] for g in matched) / len(matched)) if matched else 0.0
            n_images_total = len(per_image)
            notes = ("no per-box confidence (constant scores); compare on "
                     "precision/recall/F1/matched IoU, not confidence-ranked AP"
                     if no_confidence else "")

            def base(**kw) -> dict:
                return slice_row(workstream=f"PromptDetect-{dataset}", status=status,
                                 model=model, run=f"{run['run_id']} | {prompt}",
                                 split=config.get("split", ""),
                                 min_support=args.min_support, source=source,
                                 snapshot=f"{config.get('n_images')} images / "
                                          f"{config.get('n_gt_boxes')} GT boxes",
                                 notes=kw.pop("notes", notes), **kw)

            for metric_name, metric_value in [*aggregate.items(),
                                              ("mean_matched_iou", aggregate_miou),
                                              ("fp_per_image", totals["fp"] / max(1, n_images_total)),
                                              ("fn_per_image", totals["fn"] / max(1, n_images_total))]:
                rows.append(base(dimension="aggregate", value="all",
                                 metric=metric_name, metric_value=metric_value,
                                 n_images=n_images_total,
                                 n_objects=len(gt_records_all), aggregate=None))

            # Image-level slices ------------------------------------------------
            if stats:
                assigners = {
                    "brightness": lambda i: ev.tercile_label(
                        stats[i]["brightness"], feature_thresholds["brightness"]),
                    "contrast": lambda i: ev.tercile_label(
                        stats[i]["contrast"], feature_thresholds["contrast"]),
                    "sharpness": lambda i: ev.tercile_label(
                        stats[i]["sharpness"], feature_thresholds["sharpness"],
                        ("blurred", "intermediate", "sharp")),
                    "object_density": lambda i: ev.density_label(
                        len(data["gt_by_image"][i]["boxes"])),
                }
                for dimension, assign in assigners.items():
                    groups: dict[str, list[str]] = defaultdict(list)
                    for image_id in per_image:
                        if image_id in stats or dimension == "object_density":
                            groups[assign(image_id)].append(image_id)
                    for value, ids in sorted(groups.items()):
                        counts = {k: sum(per_image[i][k] for i in ids)
                                  for k in ("tp", "fp", "fn")}
                        for metric_name, metric_value in ev.prf_from_counts(**counts).items():
                            rows.append(base(dimension=dimension, value=value,
                                             metric=metric_name, metric_value=metric_value,
                                             n_images=len(ids),
                                             n_objects=sum(len(data["gt_by_image"][i]["boxes"])
                                                           for i in ids),
                                             aggregate=aggregate[metric_name]))

            # Object-level slices ------------------------------------------------
            object_assigners = {
                "object_size": lambda g: ev.coco_size_label(
                    max(0.0, (g["x1"] - g["x0"]) * (g["y1"] - g["y0"]))),
                "class": lambda g: g["class_name"],
            }
            for dimension, assign in object_assigners.items():
                groups2: dict[str, list[dict]] = defaultdict(list)
                for gt_record in gt_records_all:
                    groups2[assign(gt_record)].append(gt_record)
                for value, group in sorted(groups2.items()):
                    matched = [g for g in group if g["matched"]]
                    recall = len(matched) / len(group) if group else 0.0
                    miou = (sum(g["iou"] for g in matched) / len(matched)) if matched else 0.0
                    for metric_name, metric_value in [("recall", recall),
                                                      ("mean_matched_iou", miou)]:
                        agg = (aggregate["recall"] if metric_name == "recall"
                               else aggregate_miou)
                        rows.append(base(dimension=dimension, value=value,
                                         metric=metric_name, metric_value=metric_value,
                                         n_images=len({g['image'] for g in group}),
                                         n_objects=len(group), aggregate=agg))
    return rows, thresholds_used


# ---------------------------------------------------------------------------
# New extension: canonical SAM 3 prompt robustness with retained thresholds
# ---------------------------------------------------------------------------

def _retained_detection_test_thresholds(dataset: str) -> tuple[dict, str]:
    """Load, but never recompute, the retained detector test-split terciles."""
    config_path = RETAINED_DETECTION_SLICE_CONFIGS[dataset]
    if not config_path.exists():
        raise FileNotFoundError(
            f"retained {dataset} detector slice config not found: {ev.rel(config_path)}")
    resolved = ev.read_json(config_path).get("resolved_thresholds", {})
    if dataset == "MDWD":
        selected = resolved.get("MDWD/test")
        source_key = "MDWD/test"
        if not selected:
            raise RuntimeError(f"MDWD/test thresholds absent from {ev.rel(config_path)}")
    else:
        candidates = []
        for key, value in resolved.items():
            if not (key.startswith("MTSD/") and key.endswith("/test")):
                continue
            feature_values = tuple(
                (round(float(value.get(feature, {}).get("tercile_33", 0.0)), 6),
                 round(float(value.get(feature, {}).get("tercile_67", 0.0)), 6))
                for feature in ("brightness", "contrast", "sharpness"))
            if all(low != 0.0 or high != 0.0 for low, high in feature_values):
                candidates.append((feature_values, key, value))
        unique = {item[0] for item in candidates}
        if len(unique) != 1:
            raise RuntimeError(
                "MTSD retained test thresholds are absent or inconsistent; "
                "refusing to select or recompute them")
        preferred = [item for item in candidates
                     if "strongaug-wandb-followup-yolo26m-img1280-s42/test"
                     in item[1]]
        _, source_key, selected = preferred[0] if preferred else candidates[0]
    thresholds = {
        feature: (float(selected[feature]["tercile_33"]),
                  float(selected[feature]["tercile_67"]))
        for feature in ("brightness", "contrast", "sharpness")}
    return thresholds, f"{ev.rel(config_path)}::{source_key}"


def _target_classes(row: dict) -> set[str]:
    raw = row.get("target_classes", "")
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid target_classes value {raw!r}") from exc
    if not isinstance(parsed, (list, tuple)) or not parsed:
        raise ValueError(f"target_classes must be a non-empty list, got {raw!r}")
    return {str(value) for value in parsed}


def analyse_sam3_canonical_extension(args, pd_metrics, dataset: str,
                                     missing: list[str]) -> tuple[list[dict], dict]:
    """Slice canonical SAM 3 using detector-aligned image and object rules."""
    run_dir = CANONICAL_SAM3_RUNS[dataset]
    required = ["prompt_sensitivity_per_prompt.csv", "predictions.csv",
                "ground_truth_index.csv", "run_config.json"]
    absent = [name for name in required if not (run_dir / name).exists()]
    if absent:
        missing.append(
            f"SAM 3 canonical {dataset}: missing required stored files "
            f"{', '.join(absent)} under `{ev.rel(run_dir)}`; no slices produced.")
        return [], {}

    summary_rows = ev.read_csv_rows(run_dir / "prompt_sensitivity_per_prompt.csv")
    families = [row for row in summary_rows
                if row.get("model") == "SAM 3"
                and row.get("variant_type") == "canonical"]
    families.sort(key=lambda row: row.get("sensitivity_family", ""))
    expected_families = 4 if dataset == "MDWD" else 5
    if len(families) != expected_families:
        missing.append(
            f"SAM 3 canonical {dataset}: expected {expected_families} canonical "
            f"families but found {len(families)}; no slices produced.")
        return [], {}

    data = ev.load_promptdetect_run(run_dir)
    config = data["config"]
    configured_confidence = float(config.get("conf_threshold", 0.3))
    configured_iou = float(config.get("iou_threshold", 0.5))
    if abs(configured_confidence - 0.30) > 1e-9 or abs(configured_iou - 0.50) > 1e-9:
        raise RuntimeError(
            f"{dataset} stored protocol is confidence={configured_confidence}, "
            f"IoU={configured_iou}; expected 0.30/0.50")

    thresholds, threshold_source = _retained_detection_test_thresholds(dataset)
    stats: dict[str, dict] = {}
    missing_images = []
    for image_id, record in data["gt_by_image"].items():
        local = ev.rebase_stored_path(record["stored_path"])
        if not local or not local.exists():
            missing_images.append(image_id)
            continue
        stats[image_id] = ev.image_stats(local)
    if missing_images:
        missing.append(
            f"SAM 3 canonical {dataset}: {len(missing_images)} test images could "
            "not be resolved, so exact tercile slices were not produced.")
        return [], {}

    family_results: dict[str, dict] = {}
    for family in families:
        family_name = family["sensitivity_family"]
        prompt = family["prompt"]
        target_classes = _target_classes(family)
        per_image = {}
        for image_id, record in data["gt_by_image"].items():
            gt_boxes = [box for box in record["boxes"]
                        if box["class_name"] in target_classes]
            predictions = [prediction for prediction in
                           data["predictions"].get(("SAM 3", prompt, image_id), [])
                           if (not prediction["has_confidence"]
                               or prediction["score"] >= configured_confidence)]
            per_image[image_id] = ev.match_detections(
                pd_metrics, predictions, gt_boxes, iou_threshold=configured_iou,
                class_aware=False)
        totals = {key: sum(value[key] for value in per_image.values())
                  for key in ("tp", "fp", "fn")}
        aggregate_metrics = ev.prf_from_counts(**totals)
        aggregate_f1 = aggregate_metrics["f1"]
        reported_f1 = float(family["f1"])
        if abs(aggregate_f1 - reported_f1) > 5e-4:
            raise RuntimeError(
                f"{dataset}/{family_name}: rematched canonical F1 {aggregate_f1:.6f} "
                f"does not reproduce prompt_sensitivity_per_prompt.csv {reported_f1:.6f}")
        family_results[family_name] = {
            "prompt": prompt, "target_classes": sorted(target_classes),
            "per_image": per_image, "aggregate_f1": aggregate_f1,
            "aggregate_recall": aggregate_metrics["recall"],
        }

    rows: list[dict] = []
    run_id = f"{dataset}/{run_dir.name}"
    source = (f"{ev.rel(run_dir / 'prompt_sensitivity_per_prompt.csv')}; "
              f"{ev.rel(run_dir / 'predictions.csv')}; "
              f"{ev.rel(run_dir / 'ground_truth_index.csv')}")
    snapshot = ("NEW ANALYSIS; SAM 3 canonical prompts; detector-aligned "
                "brightness/contrast/sharpness, density and COCO-size rules; "
                f"retained detector test terciles from {threshold_source}")
    n_images_total = len(data["gt_by_image"])
    total_target_objects = sum(
        sum(1 for record in data["gt_by_image"].values()
            for box in record["boxes"]
            if box["class_name"] in result["target_classes"])
        for result in family_results.values())
    macro_aggregate = sum(result["aggregate_f1"]
                          for result in family_results.values()) / len(family_results)
    macro_aggregate_recall = sum(result["aggregate_recall"]
                                 for result in family_results.values()) / len(family_results)

    rows.append(slice_row(
        workstream=f"PromptDetect-SAM3-canonical-{dataset}", status=ev.STATUS_FINAL,
        model="SAM 3 canonical macro", run=run_id, split="test",
        dimension="aggregate", value="all", metric="macro_f1",
        metric_value=macro_aggregate, n_images=n_images_total,
        n_objects=total_target_objects, min_support=args.min_support,
        aggregate=None, source=source, snapshot=snapshot,
        notes=f"unweighted macro-average across {len(family_results)} target families"))
    rows.append(slice_row(
        workstream=f"PromptDetect-SAM3-canonical-{dataset}", status=ev.STATUS_FINAL,
        model="SAM 3 canonical macro", run=run_id, split="test",
        dimension="aggregate", value="all", metric="macro_recall",
        metric_value=macro_aggregate_recall, n_images=n_images_total,
        n_objects=total_target_objects, min_support=args.min_support,
        aggregate=None, source=source, snapshot=snapshot,
        notes=f"unweighted macro-average across {len(family_results)} target-family recalls"))
    for family_name, result in family_results.items():
        family_objects = sum(
            len([box for box in record["boxes"]
                 if box["class_name"] in result["target_classes"]])
            for record in data["gt_by_image"].values())
        rows.append(slice_row(
            workstream=f"PromptDetect-SAM3-canonical-{dataset}",
            status=ev.STATUS_FINAL, model=f"SAM 3 — {family_name}", run=run_id,
            split="test", dimension="aggregate", value="all", metric="f1",
            metric_value=result["aggregate_f1"], n_images=n_images_total,
            n_objects=family_objects, min_support=args.min_support,
            aggregate=None, source=source, snapshot=snapshot,
            notes=f"canonical prompt: {result['prompt']}"))
        rows.append(slice_row(
            workstream=f"PromptDetect-SAM3-canonical-{dataset}",
            status=ev.STATUS_FINAL, model=f"SAM 3 — {family_name}", run=run_id,
            split="test", dimension="aggregate", value="all", metric="recall",
            metric_value=result["aggregate_recall"], n_images=n_images_total,
            n_objects=family_objects, min_support=args.min_support,
            aggregate=None, source=source, snapshot=snapshot,
            notes=f"canonical prompt: {result['prompt']}"))

    label_names = {
        "brightness": ("low", "medium", "high"),
        "contrast": ("low", "medium", "high"),
        "sharpness": ("blurred", "intermediate", "sharp"),
        "object_density": ("single-object", "low-clutter", "high-clutter"),
    }
    for dimension, labels in label_names.items():
        grouped: dict[str, list[str]] = defaultdict(list)
        for image_id in data["gt_by_image"]:
            if dimension == "object_density":
                density_boxes = [
                    box for box in data["gt_by_image"][image_id]["boxes"]
                    if not (dataset == "MTSD"
                            and box["class_name"] == "Tourist Sign")]
                label = ev.density_label(len(density_boxes))
            else:
                label = ev.tercile_label(stats[image_id][dimension],
                                         thresholds[dimension], labels)
            grouped[label].append(image_id)
        for slice_value in labels:
            image_ids = grouped[slice_value]
            family_slice_f1 = {}
            slice_target_objects = 0
            for family_name, result in family_results.items():
                counts = {key: sum(result["per_image"][image_id][key]
                                   for image_id in image_ids)
                          for key in ("tp", "fp", "fn")}
                f1 = ev.prf_from_counts(**counts)["f1"]
                family_slice_f1[family_name] = f1
                family_objects = sum(
                    len([box for box in data["gt_by_image"][image_id]["boxes"]
                         if box["class_name"] in result["target_classes"]])
                    for image_id in image_ids)
                slice_target_objects += family_objects
                rows.append(slice_row(
                    workstream=f"PromptDetect-SAM3-canonical-{dataset}",
                    status=ev.STATUS_FINAL, model=f"SAM 3 — {family_name}",
                    run=run_id, split="test", dimension=dimension,
                    value=slice_value, metric="f1", metric_value=f1,
                    n_images=len(image_ids), n_objects=family_objects,
                    min_support=args.min_support,
                    aggregate=result["aggregate_f1"], source=source,
                    snapshot=snapshot, notes=f"canonical prompt: {result['prompt']}"))
            macro_f1 = sum(family_slice_f1.values()) / len(family_slice_f1)
            rows.append(slice_row(
                workstream=f"PromptDetect-SAM3-canonical-{dataset}",
                status=ev.STATUS_FINAL, model="SAM 3 canonical macro", run=run_id,
                split="test", dimension=dimension, value=slice_value,
                metric="macro_f1", metric_value=macro_f1,
                n_images=len(image_ids), n_objects=slice_target_objects,
                min_support=args.min_support, aggregate=macro_aggregate,
                source=source, snapshot=snapshot,
                notes=f"unweighted macro-average across {len(family_results)} family F1 values"))

    # Object-size slices are target-object-level. False positives cannot be
    # assigned uniquely to a ground-truth size bin, so report recall, not F1.
    size_labels = ("small", "medium", "large")
    size_by_family: dict[str, dict[str, list[dict]]] = {}
    for family_name, result in family_results.items():
        groups: dict[str, list[dict]] = defaultdict(list)
        for image_id, matched in result["per_image"].items():
            for gt_record in matched["gt_records"]:
                area = max(0.0, ((gt_record["x1"] - gt_record["x0"])
                                 * (gt_record["y1"] - gt_record["y0"])))
                groups[ev.coco_size_label(area)].append(
                    {**gt_record, "image": image_id})
        size_by_family[family_name] = groups
        for size in size_labels:
            group = groups[size]
            recall = (sum(int(record["matched"]) for record in group) / len(group)
                      if group else 0.0)
            rows.append(slice_row(
                workstream=f"PromptDetect-SAM3-canonical-{dataset}",
                status=ev.STATUS_FINAL, model=f"SAM 3 — {family_name}", run=run_id,
                split="test", dimension="object_size", value=size, metric="recall",
                metric_value=recall,
                n_images=len({record["image"] for record in group}),
                n_objects=len(group), min_support=args.min_support,
                aggregate=result["aggregate_recall"], source=source,
                snapshot=snapshot,
                notes=(f"canonical prompt: {result['prompt']}; COCO area thresholds; "
                       "object-level recall because false positives are not "
                       "uniquely attributable to a GT size bin")))

    for size in size_labels:
        eligible = []
        for family_name, groups in size_by_family.items():
            group = groups[size]
            if len(group) >= args.min_support:
                recall = sum(int(record["matched"]) for record in group) / len(group)
                eligible.append((family_name, group, recall))
        if not eligible:
            continue
        rows.append(slice_row(
            workstream=f"PromptDetect-SAM3-canonical-{dataset}",
            status=ev.STATUS_FINAL, model="SAM 3 canonical macro", run=run_id,
            split="test", dimension="object_size", value=size,
            metric="macro_recall",
            metric_value=sum(item[2] for item in eligible) / len(eligible),
            n_images=len({record["image"] for _, group, _ in eligible
                          for record in group}),
            n_objects=sum(len(group) for _, group, _ in eligible),
            min_support=args.min_support, aggregate=macro_aggregate_recall,
            source=source, snapshot=snapshot,
            notes=(f"unweighted macro-average across {len(eligible)} family recalls "
                   f"whose {size} support is >= {args.min_support}; COCO area thresholds")))

    threshold_record = {
        feature: {"tercile_33": values[0], "tercile_67": values[1]}
        for feature, values in thresholds.items()}
    threshold_record.update({
        "source": threshold_source, "reuse_policy": "loaded, not recomputed",
        "confidence_threshold": configured_confidence,
        "iou_threshold": configured_iou,
        "object_density_rule": "single-object=1; low-clutter=2-4; high-clutter>=5",
        "object_density_exclusions": (["Tourist Sign"] if dataset == "MTSD" else []),
        "object_size_rule": "COCO box-area thresholds: small<32^2; medium<96^2; otherwise large",
        "object_size_metric": "macro-recall; false positives are not assigned to GT size bins",
    })
    return rows, {f"PromptDetect-SAM3-canonical-{dataset}/test": threshold_record}


def audit_attribute_crop_slice_inputs(missing: list[str]) -> None:
    """Enforce the no-approximation gate for VJEPA crop-condition slicing."""
    manifest_path = ev.ATTR_OUTPUTS / "manifests" / "manifest.json"
    variant = "vjepa21_vitl_lora"
    if not manifest_path.exists():
        missing.append(
            "VJEPA 2.1-L LoRA crop-condition slices: crop manifest is missing; "
            "no attribute slices produced.")
        return
    manifest = ev.read_json(manifest_path)
    test_records = [record for record in manifest.get("records", [])
                    if record.get("split") == "test"]
    subproject_root = ev.ROOT / "Scripts" / "MTSD-Scripts" / "AttributeClassification"
    existing_crops = sum(
        (subproject_root / Path(record.get("crop_path", "").replace("\\", "/"))).exists()
        for record in test_records)
    prediction_candidates = []
    for path in ev.ATTR_OUTPUTS.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
                ".csv", ".json", ".jsonl", ".parquet"}:
            continue
        lower = path.name.lower()
        if variant in path.as_posix().lower() and any(
                token in lower for token in ("prediction", "per_crop", "logit")):
            prediction_candidates.append(path)
    if len(test_records) != 1890 or existing_crops != 1890:
        missing.append(
            "VJEPA 2.1-L LoRA crop-condition slices: expected 1,890 retained test "
            f"crop records/images but found {len(test_records)} records and "
            f"{existing_crops} images; no attribute slices produced.")
        return
    if not prediction_candidates:
        missing.append(
            "VJEPA 2.1-L LoRA crop-condition slices: all 1,890 crop records and "
            "crop images still exist, but no per-crop prediction or logit records "
            "were retained. Only aggregate metrics/confusion matrices and the "
            "checkpoint remain. Per instruction, this branch was stopped: no "
            "terciles were computed and no detector thresholds were reused.")
        return
    missing.append(
        "VJEPA 2.1-L LoRA crop-condition slices: candidate per-crop files were "
        "found but are not consumed automatically without schema validation: "
        + ", ".join(f"`{ev.rel(path)}`" for path in prediction_candidates))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(rows: list[dict], figures_dir: Path, min_support: int) -> list[str]:
    ev.apply_plot_style()
    import matplotlib.pyplot as plt

    saved: list[str] = []

    def bars(subset: list[dict], title: str, name: str, value_order: list[str],
             ylabel: str) -> None:
        models = sorted({r["model"] for r in subset})
        values = [v for v in value_order if any(r["slice_value"] == v for r in subset)]
        if not models or not values:
            return
        figure, axis = plt.subplots(figsize=(max(5.0, 1.6 * len(values)), 3.4))
        width = 0.8 / len(models)
        for index, model in enumerate(models):
            heights, positions = [], []
            for v_index, value in enumerate(values):
                match = [r for r in subset if r["model"] == model and r["slice_value"] == value]
                if match:
                    heights.append(match[0]["value"])
                    positions.append(v_index + index * width)
            axis.bar(positions, heights, width=width * 0.92, label=model)
        axis.set_xticks([i + 0.4 - width / 2 for i in range(len(values))])
        axis.set_xticklabels(values)
        axis.set_ylim(0, 1.02)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend(fontsize=8)
        saved.extend(ev.save_figure(figure, figures_dir, name))

    def det(dimension, metric, split):
        return [r for r in rows if r["workstream"] == "MDWD-detection"
                and r["slice_dimension"] == dimension and r["metric"] == metric
                and r["split"] == split and r["support_ok"] == "ok"]

    bars(det("object_size", "recall", "test"),
         "MDWD detection - recall by object size (test)",
         "mdwd_recall_by_object_size_test", ["small", "medium", "large"], "Recall")
    bars(det("brightness", "f1", "test"),
         "MDWD detection - F1 by image brightness (test)",
         "mdwd_f1_by_brightness_test", ["low", "medium", "high"], "F1")
    bars(det("sharpness", "f1", "test"),
         "MDWD detection - F1 by image sharpness (test)",
         "mdwd_f1_by_sharpness_test", ["blurred", "intermediate", "sharp"], "F1")
    class_rows = det("class", "f1", "test")
    bars(class_rows, "MDWD detection - per-class F1 (test)",
         "mdwd_per_class_f1_test", sorted({r["slice_value"] for r in class_rows}), "F1")

    # Attribute figures ------------------------------------------------------
    attr = [r for r in rows if r["workstream"] == "MTSD-attributes"]
    head_rows = [r for r in attr if r["slice_dimension"] == "attribute_head"
                 and r["metric"] == "macro_f1"]
    bars(head_rows, "MTSD attributes - macro-F1 by head (test, GRP-1..3 snapshot)",
         "mtsd_attr_macro_f1_by_head",
         ["view_angle", "mounting", "condition", "sign_shape"], "Macro-F1")
    condition_rows = [r for r in attr if r["slice_dimension"] == "condition_class"]
    bars(condition_rows,
         "MTSD attributes - condition head per-class F1 (weakest head)",
         "mtsd_attr_condition_per_class_f1",
         ["Good", "Weathered", "Heavily Damaged"], "F1")
    view_rows = [r for r in attr if r["slice_dimension"] == "view_angle_class"]
    bars(view_rows, "MTSD attributes - view-angle per-class F1",
         "mtsd_attr_view_angle_per_class_f1", ["Front", "Back", "Side"], "F1")

    # PromptDetect model x prompt heatmap ------------------------------------
    pd_rows = [r for r in rows if r["workstream"].startswith("PromptDetect")
               and r["slice_dimension"] == "aggregate" and r["metric"] == "f1"]
    if pd_rows:
        models = sorted({r["model"] for r in pd_rows})
        prompts = sorted({r["run"].split(" | ", 1)[1] for r in pd_rows if " | " in r["run"]})
        if models and prompts:
            import numpy as np

            grid = np.full((len(models), len(prompts)), np.nan)
            for r in pd_rows:
                prompt = r["run"].split(" | ", 1)[1] if " | " in r["run"] else ""
                if prompt in prompts:
                    grid[models.index(r["model"]), prompts.index(prompt)] = r["value"]
            figure, axis = plt.subplots(
                figsize=(max(4.5, 1.1 * len(prompts)), max(2.6, 0.7 * len(models))))
            image = axis.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            axis.set_xticks(range(len(prompts)))
            axis.set_xticklabels(prompts, rotation=30, ha="right", fontsize=8)
            axis.set_yticks(range(len(models)))
            axis.set_yticklabels(models, fontsize=8)
            for i in range(len(models)):
                for j in range(len(prompts)):
                    if not np.isnan(grid[i, j]):
                        axis.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                                  fontsize=8,
                                  color="white" if grid[i, j] < 0.6 else "black")
            axis.set_title("PromptDetect - F1 by model and prompt (pilot runs)")
            figure.colorbar(image, ax=axis, shrink=0.85, label="F1")
            saved.extend(ev.save_figure(figure, figures_dir, "promptdetect_model_prompt_f1"))

    # Slice degradation overview (largest drops vs aggregate) ------------------
    drop_rows = [r for r in rows if r["workstream"] == "MDWD-detection"
                 and r["support_ok"] == "ok" and r["delta_vs_aggregate"] != ""
                 and r["metric"] in ("f1", "recall") and r["split"] == "test"]
    worst: dict[tuple, dict] = {}
    for r in drop_rows:
        key = (r["slice_dimension"], r["slice_value"], r["metric"])
        if key not in worst or r["delta_vs_aggregate"] < worst[key]["delta_vs_aggregate"]:
            worst[key] = r
    ranked = sorted(worst.values(), key=lambda r: r["delta_vs_aggregate"])[:10]
    if ranked:
        figure, axis = plt.subplots(figsize=(6.4, 0.42 * len(ranked) + 1.2))
        labels = [f"{r['slice_dimension']}={r['slice_value']} ({r['metric']})"
                  for r in ranked]
        deltas = [r["delta_vs_aggregate"] for r in ranked]
        axis.barh(range(len(ranked)), deltas,
                  color=["#b2182b" if d < 0 else "#2166ac" for d in deltas])
        axis.set_yticks(range(len(ranked)))
        axis.set_yticklabels(labels, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel("Slice metric - aggregate metric")
        axis.set_title("MDWD detection - largest slice deviations (test, worst model per slice)")
        saved.extend(ev.save_figure(figure, figures_dir, "mdwd_slice_degradation_test"))

    # New extension figures: canonical SAM 3 with detector-aligned slicing.
    for dataset in ("MDWD", "MTSD"):
        workstream = f"PromptDetect-SAM3-canonical-{dataset}"
        figure_specs = (
            ("brightness", ["low", "medium", "high"], "macro_f1", "Macro-F1"),
            ("contrast", ["low", "medium", "high"], "macro_f1", "Macro-F1"),
            ("sharpness", ["blurred", "intermediate", "sharp"], "macro_f1", "Macro-F1"),
            ("object_density", ["single-object", "low-clutter", "high-clutter"],
             "macro_f1", "Macro-F1"),
            ("object_size", ["small", "medium", "large"],
             "macro_recall", "Macro-recall"),
        )
        for dimension, value_order, metric, metric_label in figure_specs:
            subset = [r for r in rows if r["workstream"] == workstream
                      and r["model"] == "SAM 3 canonical macro"
                      and r["slice_dimension"] == dimension
                      and r["metric"] == metric
                      and r["support_ok"] == "ok"]
            bars(subset,
                 f"{dataset} SAM 3 canonical - {metric_label} by {dimension}",
                 f"{dataset.lower()}_sam3_canonical_{metric}_by_{dimension}",
                 value_order, f"{metric_label} across target families")
    return saved


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def write_reports(report_dir: Path, figures: list[str], rows: list[dict],
                  thresholds: dict, missing: list[str], args) -> None:
    generated_at = ev.provenance(None, GENERATOR)["generated_at"]
    ev.write_csv(report_dir / "robustness_slice_results.csv", rows, CSV_FIELDS)
    ev.write_json(report_dir / "robustness_slice_results.json",
                  {"generated_at": generated_at, "commit_sha": ev.git_commit_sha(),
                   "generator": GENERATOR, "rows": rows})
    ev.write_json(report_dir / "robustness_slice_config.json",
                  {"generated_at": generated_at, "generator": GENERATOR,
                   "args": {k: str(v) if isinstance(v, Path) else v
                            for k, v in vars(args).items()},
                   "resolved_thresholds": thresholds,
                   "min_support_rule": "slices below --min-support are kept in the "
                                       "CSV/JSON but marked insufficient_support and "
                                       "excluded from figures/summary tables"})

    (report_dir / "insufficient_or_missing_inputs.md").write_text(
        "# Robustness slices - missing or insufficient inputs\n\n"
        f"Generated: {generated_at}\n\n"
        + "\n".join(f"- {note}" for note in missing)
        + "\n\n"
        + f"Slices marked `insufficient_support` (support < {args.min_support}): "
        + str(sum(1 for r in rows if r["support_ok"] != "ok"))
        + f" of {len(rows)} rows (retained in the CSV/JSON, never highlighted).\n",
        encoding="utf-8")

    is_extension = args.task == "extensions"
    lines = [
        ("# Robustness slice analysis — new canonical-prompt extension"
         if is_extension else "# Robustness slice analysis"),
        f"\nGenerated: {generated_at}  |  Commit: `{ev.git_commit_sha()[:12]}`",
        "\nPerformance across data slices, computed **only from stored predictions "
        "and ground truth** (no model was run). Full machine-readable results: "
        "`robustness_slice_results.csv/.json`; thresholds and matching rules: "
        "`robustness_slice_config.json`; gaps: `insufficient_or_missing_inputs.md`.",
    ]
    if is_extension:
        lines.append(
            "\n> **New analysis, separate from the retained robustness report.** "
            "This output contains SAM 3 canonical-prompt image-condition and "
            "object-size slices. It does not alter or supersede any earlier "
            "retained report. "
            "The requested VJEPA crop-condition branch is documented below as "
            "unavailable because per-crop predictions were not retained.")
    for workstream in sorted({r["workstream"] for r in rows}):
        ws_rows = [r for r in rows if r["workstream"] == workstream]
        status = ws_rows[0]["status"]
        lines.append(f"\n## {workstream} - status: **{status}**\n")
        if status in (ev.STATUS_PILOT, ev.STATUS_HISTORICAL):
            if status == ev.STATUS_PILOT:
                lines.append("> Pilot-scale run: pipeline validation only, not dissertation evidence.\n")
            elif workstream == "MTSD-attributes":
                lines.append("> Historical GRP-1..GRP-3 snapshot; a final-scope round is still pending.\n")
            else:
                lines.append("> Completed stored-run inventory; compare only runs with matching split, taxonomy and protocol.\n")
        aggregates = [r for r in ws_rows if r["slice_dimension"] == "aggregate"]
        if aggregates:
            lines.append("| Model | Run | Split | Metric | Value | Images | Objects |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for r in aggregates:
                lines.append(f"| {r['model']} | {r['run']} | {r['split']} | {r['metric']} "
                             f"| {r['value']} | {r['n_images']} | {r['n_objects']} |")
        if is_extension:
            macro_slices = [r for r in ws_rows
                            if r["model"] == "SAM 3 canonical macro"
                            and r["slice_dimension"] in {
                                "brightness", "contrast", "sharpness",
                                "object_density", "object_size"}
                            and r["metric"] in {"macro_f1", "macro_recall"}]
            if macro_slices:
                lines.append(
                    "\nCanonical SAM 3 macro metrics using detector-aligned "
                    "slice definitions:\n")
                lines.append("| Dimension | Slice | Metric | Value | Images | Target boxes | Support |")
                lines.append("| --- | --- | --- | --- | --- | --- | --- |")
                for r in macro_slices:
                    lines.append(
                        f"| {r['slice_dimension']} | {r['slice_value']} | "
                        f"{r['metric']} | {r['value']} | {r['n_images']} | "
                        f"{r['n_objects']} | "
                        f"{r['support_ok']} |")
                lines.append(
                    "\nObject density is an image-level split and therefore reports "
                    "macro-F1. Object size is a target-object split and reports "
                    "macro-recall because false positives cannot be uniquely assigned "
                    "to a ground-truth size bin. Size-family cells below the minimum "
                    "support are excluded from that bin's macro-average.")
        deviations = [r for r in ws_rows if r["support_ok"] == "ok"
                      and r["delta_vs_aggregate"] != ""
                      and abs(r["delta_vs_aggregate"]) >= 0.03]
        deviations.sort(key=lambda r: r["delta_vs_aggregate"])
        if deviations:
            lines.append("\nSlices deviating >= 0.03 from their aggregate "
                         "(sufficient support only):\n")
            lines.append("| Model | Split | Slice | Metric | Value | Aggregate | Delta | Support |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for r in deviations[:20]:
                lines.append(
                    f"| {r['model']} | {r['split']} | {r['slice_dimension']}="
                    f"{r['slice_value']} | {r['metric']} | {r['value']} | "
                    f"{r['aggregate_value']} | {r['delta_vs_aggregate']:+} | "
                    f"{r['n_objects'] if r['slice_dimension'] in OBJECT_DIMENSIONS else r['n_images']} |")
        insufficient = sum(1 for r in ws_rows if r["support_ok"] != "ok")
        if insufficient:
            lines.append(f"\n> {insufficient} slice rows have support below "
                         f"{args.min_support} and are marked `insufficient_support`.")
    if figures:
        lines.append("\n## Figures\n")
        lines.extend(f"- `{f}`" for f in figures if f.endswith(".png"))
    if is_extension and missing:
        lines.append("\n## Requested branch not produced\n")
        lines.extend(f"- {note}" for note in missing)
    lines.append("\n*Read-only analysis - no datasets, checkpoints or previous "
                 "results were modified.*\n")
    (report_dir / "robustness_slice_summary.md").write_text("\n".join(lines),
                                                            encoding="utf-8")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_self_tests() -> int:
    pd_metrics = ev.load_pd_metrics()
    cases = []

    cases.append(("COCO size categories at boundaries",
                  [ev.coco_size_label(a) for a in (31 * 31, 32 * 32, 95 * 95, 96 * 96)]
                  == ["small", "medium", "medium", "large"]))
    thresholds = ev.tercile_thresholds([1, 2, 3, 4, 5, 6, 7, 8, 9])
    cases.append(("tercile thresholds on 1..9",
                  [ev.tercile_label(v, thresholds) for v in (1, 5, 9)]
                  == ["low", "medium", "high"]))
    cases.append(("density buckets",
                  [ev.density_label(n) for n in (1, 2, 4, 5)]
                  == ["single-object", "low-clutter", "low-clutter", "high-clutter"]))
    central = {"x0": 300, "y0": 300, "x1": 340, "y1": 340}
    edge = {"x0": 0, "y0": 0, "x1": 40, "y1": 40}
    cases.append(("position central vs near-edge",
                  (ev.position_label(central, 640, 640),
                   ev.position_label(edge, 640, 640)) == ("central", "near-edge")))

    gts = [{"class_name": "A", "x0": 0, "y0": 0, "x1": 10, "y1": 10},
           {"class_name": "B", "x0": 100, "y0": 100, "x1": 130, "y1": 130}]
    preds = [{"class_name": "A", "x0": 1, "y0": 1, "x1": 10, "y1": 10, "score": 0.9},
             {"class_name": "B", "x0": 300, "y0": 300, "x1": 320, "y1": 320, "score": 0.8}]
    result = ev.match_detections(pd_metrics, preds, gts, 0.5)
    cases.append(("class-aware matching: 1 TP, 1 FP, 1 FN",
                  (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)))
    cases.append(("matched GT carries IoU > 0.8",
                  any(g["matched"] and g["iou"] > 0.8 for g in result["gt_records"])))
    cross = ev.match_detections(
        pd_metrics, [{"class_name": "B", "x0": 1, "y0": 1, "x1": 10, "y1": 10, "score": 0.9}],
        [{"class_name": "A", "x0": 0, "y0": 0, "x1": 10, "y1": 10}], 0.5)
    cases.append(("class-aware matching rejects cross-class overlap",
                  (cross["tp"], cross["fp"], cross["fn"]) == (0, 1, 1)))

    row = slice_row(workstream="t", status="final", model="m", run="r", split="test",
                    dimension="brightness", value="low", metric="f1", metric_value=0.5,
                    n_images=3, n_objects=30, min_support=15, aggregate=0.6,
                    source="s")
    cases.append(("image slice below min-support marked insufficient",
                  row["support_ok"] == "insufficient_support"
                  and row["delta_vs_aggregate"] == -0.1))
    row2 = slice_row(workstream="t", status="final", model="m", run="r", split="test",
                     dimension="object_size", value="small", metric="recall",
                     metric_value=0.5, n_images=3, n_objects=30, min_support=15,
                     aggregate=None, source="s")
    cases.append(("object slice uses object support",
                  row2["support_ok"] == "ok" and row2["delta_vs_aggregate"] == ""))

    failures = 0
    for label, ok in cases:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        failures += 0 if ok else 1
    print(f"\nSelf-test: {len(cases) - failures}/{len(cases)} passed.")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Robustness slice analysis over stored evidence (read-only).")
    parser.add_argument("--dataset", default="all",
                        choices=[*WORKSTREAMS, "all"],
                        help="Workstream to analyse (default: every one with evidence).")
    parser.add_argument(
        "--task", default="auto",
        choices=["detection", "attributes", "extensions", "auto"],
        help=("Analysis family. 'extensions' creates a separate new-analysis "
              "report for canonical SAM 3 and audits the VJEPA per-crop gate."))
    parser.add_argument("--run", default="latest",
                        help="'latest' or a path to a specific stored run directory.")
    parser.add_argument("--min-support", type=int, default=15)
    parser.add_argument("--bootstrap-samples", type=int, default=0,
                        help="Optional per-slice F1 bootstrap CI sample count (0 = off).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--working-conf", type=float, default=0.25,
                        help="Score threshold applied to stored MDWD detection predictions.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="Write to .../latest/ instead of a new timestamped folder.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return run_self_tests()

    pd_metrics = ev.load_pd_metrics()
    rows: list[dict] = []
    thresholds: dict = {}
    missing: list[str] = []

    wants = lambda ws: args.dataset in ("all", ws)  # noqa: E731
    if args.task == "extensions":
        selected = ("MDWD", "MTSD") if args.dataset == "all" else (
            args.dataset.replace("PromptDetect-", ""),)
        selected = tuple(dataset for dataset in selected
                         if dataset in ("MDWD", "MTSD"))
        for dataset in selected:
            print(f"Analysing NEW canonical SAM 3 {dataset} slices "
                  "(reused detector thresholds)...")
            new_rows, new_thresholds = analyse_sam3_canonical_extension(
                args, pd_metrics, dataset, missing)
            rows += new_rows
            thresholds.update(new_thresholds)
        if args.dataset in ("all", "MTSD", "PromptDetect-MTSD"):
            print("Auditing VJEPA 2.1-L LoRA per-crop inputs...")
            audit_attribute_crop_slice_inputs(missing)
    if wants("MDWD") and args.task in ("detection", "auto"):
        print("Analysing MDWD detection slices (stored predictions)...")
        new_rows, new_thresholds = analyse_mdwd_detection(args, pd_metrics, missing)
        rows += new_rows
        thresholds.update(new_thresholds)
    if wants("MTSD") and args.task in ("detection", "auto"):
        print("Analysing MTSD detection slices (stored unified predictions)...")
        new_rows, new_thresholds = analyse_mtsd_detection(args, pd_metrics, missing)
        rows += new_rows
        thresholds.update(new_thresholds)
    if wants("MTSD") and args.task in ("attributes", "auto"):
        print("Analysing MTSD attribute slices (stored per-head metrics)...")
        rows += analyse_attributes(args, missing)
    for dataset in ("MDWD", "MTSD"):
        if wants(f"PromptDetect-{dataset}") and args.task in ("detection", "auto"):
            print(f"Analysing PromptDetect {dataset} slices (stored batch runs)...")
            new_rows, new_thresholds = analyse_promptdetect(args, pd_metrics, dataset, missing)
            rows += new_rows
            thresholds.update(new_thresholds)

    print(f"\nSlice rows: {len(rows)} "
          f"({sum(1 for r in rows if r['support_ok'] != 'ok')} below min-support); "
          f"missing/pending notes: {len(missing)}")
    if args.dry_run:
        print("Dry run: no files written.")
        return 0

    output_family = ("Robustness-Slices-New-Analysis"
                     if args.task == "extensions" else "Robustness-Slices")
    report_dir, figures_dir = ev.new_output_dirs(output_family, args.overwrite,
                                                 args.output_dir)
    figures = make_figures(rows, figures_dir, args.min_support) if rows else []
    write_reports(report_dir, figures, rows, thresholds, missing, args)
    print(f"Reports: {report_dir}\nFigures: {figures_dir} ({len(figures)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
