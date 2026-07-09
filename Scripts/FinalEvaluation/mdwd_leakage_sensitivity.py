#!/usr/bin/env python3
"""MDWD split-leakage sensitivity analysis (read-only).

The MDWD Roboflow v20 export is ~10x offline-augmented; the EDA reports 30
source-image identities whose augmented derivatives land in more than one
train/valid/test split. This tool independently re-derives that leakage,
builds leakage-excluded ("clean") valid/test subsets, optionally re-evaluates
trained checkpoints on original vs clean subsets, and reports whether the
leakage materially inflated the reported MDWD benchmark metrics.

Read-only guarantees:
  * Datasets/, Results/MDWD-Runs/, Results/MDWD-Results/ are never written to.
    Subset evaluation uses a scratch data.yaml pointing at .txt lists of
    absolute image paths, and Ultralytics' dataset-cache writer is patched to
    a no-op so no ``labels.cache`` file is created or replaced anywhere.
  * All outputs go to a dedicated folder (default
    Documents/Final-Reports/MDWD-Leakage-Analysis/).

Reuses the repository's existing machinery so results cannot disagree with
the working tools: ``mdwd_eda.mapper`` (source-stem normalisation, dataset
map, integrity issues) and PromptDetect's ``batch_evaluation/metrics.py``
(greedy IoU matching, AP) via a thin class-aware wrapper.

Outputs (default; override with --output-dir):
    Documents/Final-Reports/MDWD-Leakage-Analysis/leaked_source_identities.csv
    Documents/Final-Reports/MDWD-Leakage-Analysis/affected_evaluation_images.csv
    Documents/Final-Reports/MDWD-Leakage-Analysis/subset_counts.csv
    Documents/Final-Reports/MDWD-Leakage-Analysis/metric_comparison.csv
    Documents/Final-Reports/MDWD-Leakage-Analysis/metric_comparison.json
    Documents/Final-Reports/MDWD-Leakage-Analysis/mdwd_leakage_sensitivity_report.md
    Documents/Final-Reports/MDWD-Leakage-Analysis/scratch/   (lists, yaml, val runs)

Usage:
    python Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py --self-test
    python Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py             # audit only
    python Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py --run-inference --models yolo26l
    python Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py --run-inference --models best-per-family
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path = SCRIPT_DIR) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "Datasets").exists() and (parent / "Scripts").exists():
            return parent
    raise RuntimeError("Could not locate project root.")


ROOT = find_project_root()
MDWD_ANALYSIS = ROOT / "Scripts" / "MDWD-Scripts" / "MDWD-Analysis"
BATCH_EVAL_DIR = ROOT / "Scripts" / "Other-Scripts" / "PromptDetect" / "batch_evaluation"
RUNS_ROOT = ROOT / "Results" / "MDWD-Runs"
RESULTS_ROOT = ROOT / "Results" / "MDWD-Results"
EDA_ISSUES_CSV = ROOT / "Documents" / "MDWD-EDA" / "GeneratedCSVs" / "integrity_issues.csv"
DEFAULT_OUTPUT_DIR = ROOT / "Documents" / "Final-Reports" / "MDWD-Leakage-Analysis"

SPLIT_ORDER = ("train", "valid", "test")
EVAL_SPLITS = ("valid", "test")
UL_SPLIT = {"valid": "val", "test": "test"}  # folder name -> ultralytics split arg
SIZE_ORDER = {"n": 0, "s": 1, "m": 2, "l": 3, "x": 4}
CKPT_RE = re.compile(r"^E\d+_(yolo\d+)([nsmlx])_")
MAP_IOU_RANGE = [round(0.50 + 0.05 * i, 2) for i in range(10)]

# |delta| thresholds in absolute percentage points of mAP@50 / mAP@50-95.
# Rationale: the clean subsets remove ~4-5% of the 369 evaluation images, so
# composition noise alone can move mAP by a few tenths of a point; shifts
# below typical seed-to-seed training variance (~0.5 pp) cannot be attributed
# to leakage.
DELTA_BANDS = ((0.5, "negligible"), (1.0, "small"), (2.0, "moderate"))


def _load_mdwd_eda():
    if str(MDWD_ANALYSIS) not in sys.path:
        sys.path.insert(0, str(MDWD_ANALYSIS))
    from mdwd_eda import mapper  # noqa: PLC0415

    return mapper


def _load_pd_metrics():
    """Load batch_evaluation/metrics.py standalone (stdlib-only module)."""
    spec = importlib.util.spec_from_file_location("pd_metrics", BATCH_EVAL_DIR / "metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Leakage core (pure functions over (file_name, split) pairs; self-testable)
# ---------------------------------------------------------------------------

def leakage_category(splits) -> str:
    return "+".join(s for s in SPLIT_ORDER if s in set(splits))


def derive_leakage(entries: list[tuple[str, str]], stem_fn) -> dict[str, dict]:
    """Group (file_name, split) pairs by source stem; return leaked stems only.

    A source identity is *leaked* when its derivatives occur in more than one
    split. Returns {stem: {splits, category, counts, files}}.
    """
    files_by_stem: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for file_name, split in entries:
        files_by_stem[stem_fn(file_name)][split].append(file_name)
    leaked = {}
    for stem, by_split in files_by_stem.items():
        if len(by_split) > 1:
            splits = [s for s in SPLIT_ORDER if s in by_split]
            leaked[stem] = {
                "splits": splits,
                "category": leakage_category(splits),
                "counts": {s: len(by_split.get(s, [])) for s in SPLIT_ORDER},
                "files": {s: sorted(by_split[s]) for s in splits},
            }
    return leaked


def clean_eval_subsets(entries: list[tuple[str, str]], stem_fn) -> dict[str, dict]:
    """Build leakage-excluded valid/test subsets.

    Definition: an evaluation image is *excluded* when its source stem also
    occurs in any other split (train or the other evaluation split). A
    valid<->test leaked stem is therefore excluded from both clean subsets
    (conservative: no evaluated image shares a source with any other split).
    """
    leaked = derive_leakage(entries, stem_fn)
    subsets = {}
    for split in EVAL_SPLITS:
        kept, excluded = [], []
        for file_name, file_split in entries:
            if file_split != split:
                continue
            (excluded if stem_fn(file_name) in leaked else kept).append(file_name)
        subsets[split] = {"kept": sorted(kept), "excluded": sorted(excluded)}
    return subsets


# ---------------------------------------------------------------------------
# Part A - audit
# ---------------------------------------------------------------------------

def run_audit(variant: str) -> dict:
    mapper = _load_mdwd_eda()
    print(f"Scanning {variant} (full dataset map; a minute or two)...")
    dataset_map = mapper.build_map(variant)
    images = dataset_map.images
    entries = list(zip(images["file_name"], images["split"]))
    leaked = derive_leakage(entries, mapper.source_stem)
    subsets = clean_eval_subsets(entries, mapper.source_stem)

    # Independent identity check: the excess of per-split unique stems over
    # globally unique stems equals sum(len(splits)-1) over leaked stems.
    per_split_unique = {
        s: images.loc[images["split"] == s, "source_stem"].nunique() for s in SPLIT_ORDER
    }
    total_unique = images["source_stem"].nunique()
    excess = sum(per_split_unique.values()) - total_unique
    expected_excess = sum(len(info["splits"]) - 1 for info in leaked.values())
    identity = {
        "per_split_unique_stems": per_split_unique,
        "total_unique_stems": int(total_unique),
        "excess": int(excess),
        "expected_excess_from_leaked": int(expected_excess),
        "consistent": bool(excess == expected_excess),
    }

    cross_check = cross_check_eda_csv(leaked)
    label_quality = label_quality_audit(dataset_map, subsets)

    n_boxes = {
        (split, "original"): int(images.loc[images["split"] == split, "n_boxes"].sum())
        for split in EVAL_SPLITS
    }
    kept_boxes = {}
    for split in EVAL_SPLITS:
        kept = set(subsets[split]["kept"])
        mask = (images["split"] == split) & images["file_name"].isin(kept)
        kept_boxes[(split, "clean")] = int(images.loc[mask, "n_boxes"].sum())
    subset_rows = []
    for split in EVAL_SPLITS:
        split_images = images[images["split"] == split]
        original_n = int(len(split_images))
        kept_n = len(subsets[split]["kept"])
        for subset, count, boxes in (
            ("original", original_n, n_boxes[(split, "original")]),
            ("clean", kept_n, kept_boxes[(split, "clean")]),
        ):
            names = subsets[split]["kept"] if subset == "clean" else list(split_images["file_name"])
            stems = {mapper.source_stem(n) for n in names}
            subset_rows.append({
                "split": split,
                "subset": subset,
                "n_images": count,
                "n_source_stems": len(stems),
                "n_boxes": boxes,
                "n_removed_images": original_n - count if subset == "clean" else 0,
                "pct_removed": round(100 * (original_n - count) / original_n, 2) if subset == "clean" else 0.0,
            })

    return {
        "dataset_map": dataset_map,
        "mapper": mapper,
        "variant": variant,
        "n_images_per_split": {s: int((images["split"] == s).sum()) for s in SPLIT_ORDER},
        "leaked": leaked,
        "subsets": subsets,
        "identity": identity,
        "cross_check": cross_check,
        "label_quality": label_quality,
        "subset_rows": subset_rows,
        "category_counts": dict(Counter(info["category"] for info in leaked.values())),
    }


def cross_check_eda_csv(leaked: dict[str, dict]) -> dict:
    """Compare independently derived leakage against the EDA integrity CSV."""
    if not EDA_ISSUES_CSV.exists():
        return {"csv_found": False, "match": False, "detail": f"not found: {EDA_ISSUES_CSV}"}
    eda = {}
    with EDA_ISSUES_CSV.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("type") == "source_in_multiple_splits":
                eda[row["file"]] = {s.strip() for s in row.get("detail", "").split(",") if s.strip()}
    ours = {stem: set(info["splits"]) for stem, info in leaked.items()}
    only_ours = sorted(set(ours) - set(eda))
    only_eda = sorted(set(eda) - set(ours))
    split_mismatches = sorted(
        stem for stem in set(ours) & set(eda) if ours[stem] != eda[stem]
    )
    return {
        "csv_found": True,
        "eda_count": len(eda),
        "recomputed_count": len(ours),
        "only_recomputed": only_ours,
        "only_eda": only_eda,
        "split_set_mismatches": split_mismatches,
        "match": not (only_ours or only_eda or split_mismatches),
    }


def label_quality_audit(dataset_map, subsets: dict) -> dict:
    """Audit out-of-range boxes and empty annotations, separate from leakage.

    Each out-of-range label line (per mdwd_eda's strict 0..1 bounds) is also
    re-tested against Ultralytics' looser training-loader rule (1% tolerance:
    image rejected when any coordinate > 1.01 or any value < -0.01).
    """
    variant_dir = ROOT / "Datasets" / "MDWD" / dataset_map.variant
    out_of_range, empty = [], []
    for issue in dataset_map.issues:
        if issue["type"] == "out_of_range_box":
            split, file = issue["split"], issue["file"]
            line_no = int(str(issue.get("detail", "")).replace("line", "").strip() or 0)
            record = {"split": split, "file": file, "line": line_no,
                      "ultralytics_rejects_image": None, "values": ""}
            label_path = variant_dir / split / "labels" / file
            if label_path.exists() and line_no:
                lines = label_path.read_text(encoding="utf-8").splitlines()
                if line_no <= len(lines):
                    parts = lines[line_no - 1].split()
                    try:
                        values = [float(v) for v in parts]
                        coords = values[1:5]
                        record["values"] = " ".join(parts)
                        record["ultralytics_rejects_image"] = bool(
                            max(coords) > 1.01 or min(values) < -0.01
                        )
                    except ValueError:
                        record["ultralytics_rejects_image"] = True
            out_of_range.append(record)
        elif issue["type"] == "empty_annotation":
            empty.append({"split": issue["split"], "file": issue["file"]})

    eval_hits = [r for r in out_of_range + empty if r["split"] in EVAL_SPLITS]
    clean_names = {name for split in EVAL_SPLITS for name in subsets[split]["kept"]}
    label_stem_overlap = [
        r for r in eval_hits
        if any(name.startswith(Path(r["file"]).stem) for name in clean_names)
    ]
    return {
        "out_of_range": out_of_range,
        "empty_annotations": empty,
        "counts_by_split": {
            "out_of_range_box": dict(Counter(r["split"] for r in out_of_range)),
            "empty_annotation": dict(Counter(r["split"] for r in empty)),
        },
        "in_evaluation_splits": len(eval_hits),
        "in_clean_subsets": len(label_stem_overlap),
        "ultralytics_rejected_images": sum(
            1 for r in out_of_range if r["ultralytics_rejects_image"]
        ),
    }


# ---------------------------------------------------------------------------
# Part B - checkpoint discovery and Ultralytics evaluation
# ---------------------------------------------------------------------------

def discover_checkpoints() -> list[dict]:
    checkpoints = []
    for suite_dir in sorted(RUNS_ROOT.iterdir()):
        if not suite_dir.is_dir():
            continue
        if suite_dir.name.startswith("[OLD]") or suite_dir.name.upper().startswith("RF-DETR"):
            continue
        for best in sorted(suite_dir.glob("E*/weights/best.pt")):
            match = CKPT_RE.match(best.parents[1].name)
            if not match:
                continue
            checkpoints.append({
                "family": match.group(1),
                "size": match.group(2),
                "model": match.group(1) + match.group(2),
                "suite": suite_dir.name,
                "run_name": best.parents[1].name,
                "path": best,
            })
    return checkpoints


def resolve_models(spec: str, checkpoints: list[dict]) -> list[dict]:
    if spec.strip() == "best-per-family":
        by_group: dict[tuple[str, str], dict] = {}
        for ckpt in checkpoints:
            key = (ckpt["family"], ckpt["suite"])
            if key not in by_group or SIZE_ORDER[ckpt["size"]] > SIZE_ORDER[by_group[key]["size"]]:
                by_group[key] = ckpt
        return sorted(by_group.values(), key=lambda c: (c["family"], c["suite"]))
    selected = []
    for token in [t.strip() for t in spec.split(",") if t.strip()]:
        name, _, suite = token.partition("@")
        matches = [c for c in checkpoints if c["model"] == name and (not suite or c["suite"] == suite)]
        if not matches:
            available = sorted({f"{c['model']}@{c['suite']}" for c in checkpoints})
            raise SystemExit(f"No checkpoint matches {token!r}. Available: {', '.join(available)}")
        matches.sort(key=lambda c: (not c["suite"].endswith("EUVIP"), c["suite"]))
        selected.append(matches[0])
    return selected


def patch_ultralytics_cache_writes() -> None:
    """Prevent Ultralytics from creating/replacing labels.cache in Datasets/."""
    import ultralytics.data.dataset as uld  # noqa: PLC0415

    def _no_cache_write(prefix, path, x, version):
        # The stock function also stamps the in-memory dict with the cache
        # version (get_labels() pops it afterwards); keep the mutation,
        # skip only the write to disk.
        x["version"] = version
        return None

    uld.save_dataset_cache_file = _no_cache_write
    print("  [patch] ultralytics dataset-cache writes disabled (read-only run)")


def write_scratch(out_dir: Path, variant_dir: Path, subsets: dict, class_names: list[str]) -> Path:
    lists_dir = out_dir / "scratch" / "lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    list_paths = {}
    for split in EVAL_SPLITS:
        list_path = lists_dir / f"clean_{split}.txt"
        image_paths = [str(variant_dir / split / "images" / name) for name in subsets[split]["kept"]]
        list_path.write_text("\n".join(image_paths) + "\n", encoding="utf-8")
        list_paths[split] = list_path
    yaml_path = out_dir / "scratch" / "clean.yaml"
    names = "[" + ", ".join(f"'{n}'" for n in class_names) + "]"
    yaml_path.write_text(
        "# Auto-generated by mdwd_leakage_sensitivity.py - leakage-excluded subsets.\n"
        "# Points at absolute image paths inside the untouched dataset; labels\n"
        "# resolve via Ultralytics' images->labels path substitution.\n"
        f"train: {list_paths['valid'].resolve()}\n"
        f"val: {list_paths['valid'].resolve()}\n"
        f"test: {list_paths['test'].resolve()}\n"
        f"nc: {len(class_names)}\n"
        f"names: {names}\n",
        encoding="utf-8",
    )
    return yaml_path


def run_val(checkpoint: dict, data_yaml: Path, split: str, subset: str,
            args, runs_root: Path) -> tuple[dict, Path]:
    """One Ultralytics validation run; returns (metrics, predictions.json path)."""
    from ultralytics import YOLO  # noqa: PLC0415

    run_name = f"{checkpoint['model']}__{checkpoint['suite']}__{split}__{subset}"
    run_dir = runs_root / run_name
    cached_metrics = run_dir / "ul_metrics.json"
    predictions = run_dir / "predictions.json"
    if args.skip_cached and cached_metrics.exists() and predictions.exists():
        print(f"  [cached] {run_name}")
        return json.loads(cached_metrics.read_text(encoding="utf-8")), predictions

    print(f"  [val] {run_name} ...")
    model = YOLO(str(checkpoint["path"]))
    results = model.val(
        data=str(data_yaml), split=UL_SPLIT[split], imgsz=args.imgsz, batch=args.batch,
        conf=args.conf, iou=args.iou, device=(args.device or None), workers=args.workers,
        plots=False, save_json=True, verbose=False,
        project=str(runs_root), name=run_name, exist_ok=True,
    )
    rd = results.results_dict
    precision = float(rd["metrics/precision(B)"])
    recall = float(rd["metrics/recall(B)"])
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "map50": float(rd["metrics/mAP50(B)"]),
        "map50_95": float(rd["metrics/mAP50-95(B)"]),
    }
    save_dir = Path(results.save_dir)
    (save_dir / "ul_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics, save_dir / "predictions.json"


# ---------------------------------------------------------------------------
# Part C - supplementary class-aware metrics (reuses batch_evaluation/metrics)
# ---------------------------------------------------------------------------

def gt_from_map(dataset_map, split: str) -> dict[str, list[dict]]:
    """Ground truth as {file_stem: [box dicts]} in absolute pixels."""
    images = dataset_map.images
    boxes = dataset_map.boxes
    gt: dict[str, list[dict]] = {}
    dims: dict[str, tuple] = {}
    for row in images[images["split"] == split].itertuples():
        gt[Path(row.file_name).stem] = []
        dims[row.file_name] = (row.width, row.height)
    for row in boxes[boxes["split"] == split].itertuples():
        width, height = dims[row.file_name]
        gt[Path(row.file_name).stem].append({
            "class_name": row.class_name,
            "x0": (row.cx - row.box_w / 2) * width,
            "y0": (row.cy - row.box_h / 2) * height,
            "x1": (row.cx + row.box_w / 2) * width,
            "y1": (row.cy + row.box_h / 2) * height,
        })
    return gt


def load_predictions_json(path: Path, class_names: list[str]) -> dict[str, list[dict]]:
    """Ultralytics val predictions.json -> {file_stem: [pred dicts]}.

    For non-COCO datasets Ultralytics writes image_id = filename stem and
    category_id = class_index + 1 (1-based), bbox = [x, y, w, h] pixels.
    """
    preds: dict[str, list[dict]] = defaultdict(list)
    for entry in json.loads(path.read_text(encoding="utf-8")):
        x, y, w, h = entry["bbox"]
        class_index = int(entry["category_id"]) - 1
        preds[str(entry["image_id"])].append({
            "class_name": class_names[class_index] if 0 <= class_index < len(class_names) else "?",
            "x0": x, "y0": y, "x1": x + w, "y1": y + h,
            "score": float(entry.get("score", 0.0)),
        })
    return dict(preds)


def class_aware_evaluate(pd_metrics, preds: dict, gt: dict, class_names: list[str]) -> dict:
    """Per-class greedy matching via evaluate_prompt; micro counts, macro AP."""
    totals = Counter()
    iou_weighted_sum, iou_weight = 0.0, 0
    ap50_values, map_values = [], []
    for class_name in class_names:
        gt_c = {img: [b for b in bs if b["class_name"] == class_name] for img, bs in gt.items()}
        preds_c = {img: [p for p in ps if p["class_name"] == class_name] for img, ps in preds.items()}
        summary = pd_metrics.evaluate_prompt(preds_c, gt_c, 0.5, MAP_IOU_RANGE)["summary"]
        for key in ("tp", "fp", "fn", "duplicates"):
            totals[key] += summary[key]
        iou_weighted_sum += summary["mean_matched_iou"] * summary["tp"]
        iou_weight += summary["tp"]
        if summary["n_gt"] > 0:
            ap50_values.append(summary["ap50"])
            map_values.append(summary["map50_95"])
    counts = pd_metrics.prf(totals["tp"], totals["fp"], totals["fn"])
    return {
        "tp": int(totals["tp"]), "fp": int(totals["fp"]), "fn": int(totals["fn"]),
        "duplicates": int(totals["duplicates"]),
        "precision": counts["precision"], "recall": counts["recall"], "f1": counts["f1"],
        "mean_matched_iou": round(iou_weighted_sum / iou_weight, 4) if iou_weight else 0.0,
        "ap50_macro": round(sum(ap50_values) / len(ap50_values), 4) if ap50_values else 0.0,
        "map50_95_macro": round(sum(map_values) / len(map_values), 4) if map_values else 0.0,
    }


def supplementary_metrics(pd_metrics, preds: dict, gt: dict, class_names: list[str],
                          working_conf: float) -> dict:
    """AP/IoU from all predictions; TP/FP/FN/P/R/F1 at the working confidence."""
    full = class_aware_evaluate(pd_metrics, preds, gt, class_names)
    working = {
        img: [p for p in ps if p["score"] >= working_conf] for img, ps in preds.items()
    }
    operating = class_aware_evaluate(pd_metrics, working, gt, class_names)
    return {
        "tp": operating["tp"], "fp": operating["fp"], "fn": operating["fn"],
        "duplicates": operating["duplicates"],
        "precision": operating["precision"], "recall": operating["recall"], "f1": operating["f1"],
        "mean_matched_iou": full["mean_matched_iou"],
        "ap50_macro": full["ap50_macro"],
        "map50_95_macro": full["map50_95_macro"],
    }


def prediction_consistency(clean_preds: dict, original_preds_filtered: dict) -> dict:
    """Determinism check: clean-run predictions vs filtered original-run ones."""
    stems = set(clean_preds) | set(original_preds_filtered)
    mismatched = sorted(
        s for s in stems if len(clean_preds.get(s, [])) != len(original_preds_filtered.get(s, []))
    )
    return {"n_images_compared": len(stems), "n_count_mismatches": len(mismatched),
            "mismatched_stems": mismatched[:20]}


# ---------------------------------------------------------------------------
# Reported-benchmark cross-reference
# ---------------------------------------------------------------------------

def reported_metrics(checkpoint: dict) -> dict | None:
    results_dir = RESULTS_ROOT / checkpoint["suite"].replace("-EUVIP", "") / "Model-Size-Comparison"
    candidates = sorted(results_dir.glob("*_multi_model_template_summary.csv"))
    if not candidates:
        return None
    with candidates[0].open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("model_variant") == checkpoint["model"]:
                def _f(key):
                    try:
                        return float(row[key])
                    except (KeyError, TypeError, ValueError):
                        return None
                return {
                    "valid": {"precision": _f("precision"), "recall": _f("recall"),
                              "map50": _f("mAP@50"), "map50_95": _f("mAP@50-95")},
                    "test": {"precision": _f("test_precision"), "recall": _f("test_recall"),
                             "map50": _f("test_mAP@50"), "map50_95": _f("test_mAP@50-95")},
                    "source": str(candidates[0].relative_to(ROOT)),
                }
    return None


# ---------------------------------------------------------------------------
# Delta classification
# ---------------------------------------------------------------------------

def classify_delta(delta_pp: float) -> str:
    magnitude = abs(delta_pp)
    for bound, label in DELTA_BANDS:
        if magnitude <= bound:
            return label
    return "substantial"


SEVERITY_RANK = {"negligible": 0, "small": 1, "moderate": 2, "substantial": 3}


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_audit_csvs(out_dir: Path, audit: dict) -> None:
    mapper = audit["mapper"]
    leak_rows = []
    cross = audit["cross_check"]
    only_recomputed = set(cross.get("only_recomputed", []))
    for stem in sorted(audit["leaked"]):
        info = audit["leaked"][stem]
        leak_rows.append({
            "source_stem": stem,
            "splits": ", ".join(info["splits"]),
            "category": info["category"],
            "n_train_derivatives": info["counts"]["train"],
            "n_valid_derivatives": info["counts"]["valid"],
            "n_test_derivatives": info["counts"]["test"],
            "in_eda_csv": bool(cross.get("csv_found")) and stem not in only_recomputed,
        })
    write_csv(out_dir / "leaked_source_identities.csv",
              ["source_stem", "splits", "category", "n_train_derivatives",
               "n_valid_derivatives", "n_test_derivatives", "in_eda_csv"], leak_rows)

    images = audit["dataset_map"].images
    n_boxes_by_file = dict(zip(images["file_name"], images["n_boxes"]))
    affected_rows = []
    for split in EVAL_SPLITS:
        for file_name in audit["subsets"][split]["excluded"]:
            stem = mapper.source_stem(file_name)
            affected_rows.append({
                "split": split,
                "file_name": file_name,
                "source_stem": stem,
                "category": audit["leaked"][stem]["category"],
                "excluded_from_clean_valid": split == "valid",
                "excluded_from_clean_test": split == "test",
                "n_boxes": int(n_boxes_by_file.get(file_name, 0)),
            })
    write_csv(out_dir / "affected_evaluation_images.csv",
              ["split", "file_name", "source_stem", "category",
               "excluded_from_clean_valid", "excluded_from_clean_test", "n_boxes"],
              affected_rows)

    write_csv(out_dir / "subset_counts.csv",
              ["split", "subset", "n_images", "n_source_stems", "n_boxes",
               "n_removed_images", "pct_removed"], audit["subset_rows"])


METRIC_FIELDS = [
    "model", "suite", "checkpoint", "split", "subset", "n_images", "n_gt_boxes",
    "ul_precision", "ul_recall", "ul_f1", "ul_map50", "ul_map50_95",
    "sup_tp", "sup_fp", "sup_fn", "sup_duplicates", "sup_precision", "sup_recall",
    "sup_f1", "sup_mean_matched_iou", "sup_ap50_macro", "sup_map50_95_macro",
]
RATIO_FIELDS = [
    "ul_precision", "ul_recall", "ul_f1", "ul_map50", "ul_map50_95",
    "sup_precision", "sup_recall", "sup_f1", "sup_mean_matched_iou",
    "sup_ap50_macro", "sup_map50_95_macro",
]


def comparison_rows(model_results: list[dict]) -> list[dict]:
    rows = []
    for result in model_results:
        for split in EVAL_SPLITS:
            by_subset = {}
            for subset in ("original", "clean"):
                run = result["runs"][(split, subset)]
                row = {
                    "model": result["checkpoint"]["model"],
                    "suite": result["checkpoint"]["suite"],
                    "checkpoint": str(result["checkpoint"]["path"].relative_to(ROOT)),
                    "split": split, "subset": subset,
                    "n_images": run["n_images"], "n_gt_boxes": run["n_gt_boxes"],
                    **{f"ul_{k}": round(v, 4) for k, v in run["ul"].items()},
                    **{f"sup_{k}": v for k, v in run["sup"].items()},
                }
                by_subset[subset] = row
                rows.append(row)
            delta = {
                "model": result["checkpoint"]["model"],
                "suite": result["checkpoint"]["suite"],
                "checkpoint": by_subset["original"]["checkpoint"],
                "split": split, "subset": "delta_clean_minus_original",
                "n_images": by_subset["clean"]["n_images"] - by_subset["original"]["n_images"],
                "n_gt_boxes": by_subset["clean"]["n_gt_boxes"] - by_subset["original"]["n_gt_boxes"],
            }
            for field in RATIO_FIELDS:
                delta[field] = round(by_subset["clean"][field] - by_subset["original"][field], 4)
            for field in ("sup_tp", "sup_fp", "sup_fn", "sup_duplicates"):
                delta[field] = by_subset["clean"][field] - by_subset["original"][field]
            rows.append(delta)
    return rows


def build_report(audit: dict, args, generated_at: str, model_results: list[dict] | None,
                 environment: dict | None) -> str:
    leaked = audit["leaked"]
    subsets = audit["subsets"]
    affected = {s: len(subsets[s]["excluded"]) for s in EVAL_SPLITS}
    if model_results:
        overall = max(
            (c for r in model_results for c in r["classification"].values()),
            key=lambda label: SEVERITY_RANK[label],
        ).upper()
    else:
        overall = "AUDIT-ONLY (metrics pending)"

    lines = [
        "# MDWD split-leakage sensitivity analysis",
        f"\nGenerated: {generated_at}  |  Overall: **{overall}**",
        "\nPurpose: measure whether the cross-split source-image leakage in the",
        "MDWD Roboflow v20 export materially inflated the reported validation/test",
        "detection metrics. Machine-readable outputs sit next to this report",
        "(`leaked_source_identities.csv`, `affected_evaluation_images.csv`,",
        "`subset_counts.csv`, `metric_comparison.csv/.json`).",
        "\n## 1. Leakage definition and source-identity normalisation\n",
        "Roboflow names each augmented copy `<source_stem>_<ext>.rf.<hash>.jpg`;",
        "the source identity of a file is the part of its stem before `.rf.`",
        "(`mdwd_eda.mapper.source_stem`, reused verbatim). A source identity is",
        "**leaked** when its augmented derivatives occur in more than one of",
        "train/valid/test. Limitation: the original file extension is folded into",
        "the stem (`_jpg` vs `_jpeg`), so one photograph exported under two",
        "different extensions would count as two identities; this matches the",
        "EDA's definition and errs on the side of distinct sources.",
        "\n## 2. Leakage audit (independently recomputed)\n",
        f"- Leaked source identities: **{len(leaked)}**",
        f"- Direction breakdown: " + ", ".join(
            f"{cat}: {n}" for cat, n in sorted(audit["category_counts"].items())),
        f"- Affected validation images: **{affected['valid']}** of "
        f"{audit['n_images_per_split']['valid']}",
        f"- Affected test images: **{affected['test']}** of "
        f"{audit['n_images_per_split']['test']}",
    ]
    identity = audit["identity"]
    lines.append(
        f"- Identity check: per-split unique stems "
        f"{' + '.join(str(identity['per_split_unique_stems'][s]) for s in SPLIT_ORDER)} = "
        f"{sum(identity['per_split_unique_stems'].values())} vs "
        f"{identity['total_unique_stems']} deduplicated -> excess {identity['excess']}, "
        f"expected from leaked stems {identity['expected_excess_from_leaked']} "
        f"({'consistent' if identity['consistent'] else 'INCONSISTENT'})"
    )
    cross = audit["cross_check"]
    if cross.get("csv_found"):
        status = "MATCH" if cross["match"] else "MISMATCH"
        lines.append(
            f"- Cross-check vs EDA `integrity_issues.csv`: **{status}** "
            f"(EDA {cross['eda_count']} vs recomputed {cross['recomputed_count']}; "
            f"only-recomputed {len(cross['only_recomputed'])}, only-EDA {len(cross['only_eda'])}, "
            f"split-set mismatches {len(cross['split_set_mismatches'])})"
        )
    else:
        lines.append(f"- Cross-check vs EDA CSV: skipped ({cross['detail']})")

    lines += [
        "\n## 3. Clean evaluation subsets\n",
        "An evaluation image is excluded when its source stem also occurs in any",
        "other split (train or the other evaluation split); valid<->test leaks are",
        "excluded from both clean subsets, so no evaluated image shares a source",
        "with any other split.\n",
        "| Split | Subset | Images | Source stems | GT boxes | Removed | % removed |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in audit["subset_rows"]:
        lines.append(
            f"| {row['split']} | {row['subset']} | {row['n_images']} | "
            f"{row['n_source_stems']} | {row['n_boxes']} | {row['n_removed_images']} | "
            f"{row['pct_removed']} |"
        )

    lq = audit["label_quality"]
    oor_total = len(lq["out_of_range"])
    lines += [
        "\n## 4. Other integrity issues (separate from leakage)\n",
        f"- Out-of-range boxes: {oor_total} "
        f"(by split: {lq['counts_by_split']['out_of_range_box'] or 'none'}); "
        f"empty annotations: {len(lq['empty_annotations'])} "
        f"(by split: {lq['counts_by_split']['empty_annotation'] or 'none'}).",
        f"- In evaluation splits (valid/test): {lq['in_evaluation_splits']}; "
        f"overlapping the clean subsets: {lq['in_clean_subsets']}. "
        + ("They therefore do not influence the evaluation metrics or the leakage "
           "sensitivity subsets." if lq["in_evaluation_splits"] == 0 else
           "NOTE: evaluation splits are affected - see metric caveats."),
        f"- Loader behaviour: the `mdwd_eda` audit flags any coordinate outside "
        f"[0, 1] strictly; Ultralytics' training loader tolerates 1% overshoot and "
        f"rejects the whole image beyond that (coordinate > 1.01 or value < -0.01). "
        f"Of the {oor_total} flagged label lines, "
        f"{lq['ultralytics_rejected_images']} would cause Ultralytics to drop the "
        f"image from training entirely; the rest train with the slightly "
        f"out-of-bounds box as-is. Empty label files act as background (negative) "
        f"images during training.",
    ]

    lines.append("\n## 5. Metric sensitivity (original vs leakage-excluded)\n")
    if not model_results:
        lines += [
            "PENDING - run with `--run-inference` to populate this section, e.g.:",
            "```",
            "python Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py "
            "--run-inference --models best-per-family",
            "```",
        ]
    else:
        lines += [
            "Two metric sources per run: `UL` = Ultralytics `model.val()` (same",
            f"protocol as the original benchmarks: imgsz={args.imgsz}, conf={args.conf},",
            f"iou={args.iou}, batch={args.batch}); `SUP` = class-aware greedy matching",
            "reusing PromptDetect's `batch_evaluation/metrics.py` (counts at working",
            f"confidence {args.working_conf}; AP over all predictions). SUP absolute",
            "values are not comparable to UL (different matcher); interpret only the",
            "original-vs-clean delta within each source. SUP clean-subset metrics are",
            "computed by filtering the original run's predictions, so their delta is a",
            "pure subset-composition effect.\n",
        ]
        for result in model_results:
            ckpt = result["checkpoint"]
            lines.append(f"### {ckpt['model']} ({ckpt['suite']})\n")
            lines.append(f"Checkpoint: `{ckpt['path'].relative_to(ROOT)}`\n")
            lines.append("| Split | Subset | Images | UL P | UL R | UL F1 | UL mAP50 | "
                         "UL mAP50-95 | SUP F1 | SUP mIoU | SUP FP | SUP FN |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for split in EVAL_SPLITS:
                for subset in ("original", "clean"):
                    run = result["runs"][(split, subset)]
                    ul, sup = run["ul"], run["sup"]
                    lines.append(
                        f"| {split} | {subset} | {run['n_images']} | {ul['precision']:.4f} | "
                        f"{ul['recall']:.4f} | {ul['f1']:.4f} | {ul['map50']:.4f} | "
                        f"{ul['map50_95']:.4f} | {sup['f1']:.4f} | "
                        f"{sup['mean_matched_iou']:.4f} | {sup['fp']} | {sup['fn']} |"
                    )
            lines.append("")
            for split in EVAL_SPLITS:
                d = result["deltas"][split]
                lines.append(
                    f"- {split}: delta mAP50 {d['ul_map50']:+.4f} "
                    f"({d['ul_map50'] * 100:+.2f} pp), delta mAP50-95 "
                    f"{d['ul_map50_95']:+.4f} ({d['ul_map50_95'] * 100:+.2f} pp) "
                    f"-> **{result['classification'][split]}**"
                )
            rep = result.get("reported")
            if rep:
                lines.append(
                    f"- Reported benchmark ({rep['source']}): val mAP50 "
                    f"{rep['valid']['map50']:.4f} / mAP50-95 {rep['valid']['map50_95']:.4f}; "
                    f"test mAP50 {rep['test']['map50']:.4f} / mAP50-95 "
                    f"{rep['test']['map50_95']:.4f}. Reported val metrics come from the "
                    f"best training epoch, so small differences vs this standalone "
                    f"re-validation are expected."
                )
            consistency = result.get("consistency", {})
            for split in EVAL_SPLITS:
                check = consistency.get(split)
                if check and check["n_count_mismatches"]:
                    lines.append(
                        f"- Determinism check ({split}): {check['n_count_mismatches']} of "
                        f"{check['n_images_compared']} clean-subset images had different "
                        f"prediction counts between the clean run and the filtered "
                        f"original run (batch-composition jitter; SUP deltas use the "
                        f"filtered original predictions and are unaffected)."
                    )
            lines.append("")
        lines += [
            "### Delta classification thresholds\n",
            "Absolute percentage-point change in mAP50 / mAP50-95 (clean - original);",
            "the headline label per model/split uses the larger |delta| of the two:",
            "negligible <= 0.5 pp < small <= 1.0 pp < moderate <= 2.0 pp < substantial.",
            "Rationale: the clean subsets remove ~4-5% of evaluation images, so",
            "composition noise alone can move mAP by a few tenths of a point;",
            "only shifts clearly above typical seed-to-seed run variance can be",
            "attributed to leakage inflation.",
        ]

    lines += [
        "\n## 6. Method notes and limitations\n",
        "- No stored per-image predictions existed in the repository (the original",
        "  notebooks ran `model.val()` without `save_json`), so metrics were",
        "  recomputed by checkpoint inference gated behind `--run-inference`.",
        "- Subset evaluation uses a scratch data.yaml pointing at .txt lists of",
        "  absolute image paths inside the untouched dataset; Ultralytics'",
        "  dataset-cache writer is patched to a no-op so no `labels.cache` is",
        "  created or modified anywhere under `Datasets/`.",
        "- RF-DETR is out of scope here: it was trained/evaluated with a different",
        "  framework (COCO evaluator inside the rfdetr package), its benchmark is",
        "  incomplete (single N-scale run), and no Ultralytics-compatible",
        "  re-validation path exists for it.",
        "- The SUP matcher is greedy and per-class (reused from PromptDetect batch",
        "  evaluation); its absolute AP differs from Ultralytics' by construction.",
    ]
    if environment:
        lines.append(
            f"- Environment: ultralytics {environment['ultralytics']}, "
            f"torch {environment['torch']}, device {environment['device']}."
        )
    lines.append(
        "\n*Read-only analysis - no dataset, checkpoint, notebook, or historical "
        "result files were modified. All outputs live under "
        "`Documents/Final-Reports/MDWD-Leakage-Analysis/`. Re-run via "
        "`python Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py`.*\n"
    )
    return "\n".join(lines)


def build_json_payload(audit: dict, args, generated_at: str,
                       model_results: list[dict] | None, environment: dict | None) -> dict:
    payload = {
        "generated_at": generated_at,
        "variant": audit["variant"],
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "leakage": {
            "n_leaked_source_identities": len(audit["leaked"]),
            "category_counts": audit["category_counts"],
            "affected_valid_images": len(audit["subsets"]["valid"]["excluded"]),
            "affected_test_images": len(audit["subsets"]["test"]["excluded"]),
            "identity_check": audit["identity"],
            "eda_cross_check": audit["cross_check"],
            "leaked_source_identities": {
                stem: {k: v for k, v in info.items() if k != "files"}
                for stem, info in sorted(audit["leaked"].items())
            },
        },
        "subsets": audit["subset_rows"],
        "label_quality": {
            "counts_by_split": audit["label_quality"]["counts_by_split"],
            "in_evaluation_splits": audit["label_quality"]["in_evaluation_splits"],
            "in_clean_subsets": audit["label_quality"]["in_clean_subsets"],
            "ultralytics_rejected_images": audit["label_quality"]["ultralytics_rejected_images"],
            "out_of_range": audit["label_quality"]["out_of_range"],
            "empty_annotations": audit["label_quality"]["empty_annotations"],
        },
        "delta_thresholds_pp": {label: bound for bound, label in DELTA_BANDS},
        "environment": environment,
        "models": [],
    }
    for result in model_results or []:
        payload["models"].append({
            "model": result["checkpoint"]["model"],
            "suite": result["checkpoint"]["suite"],
            "checkpoint": str(result["checkpoint"]["path"].relative_to(ROOT)),
            "runs": {
                f"{split}/{subset}": {
                    "n_images": run["n_images"], "n_gt_boxes": run["n_gt_boxes"],
                    "ultralytics": run["ul"], "supplementary": run["sup"],
                }
                for (split, subset), run in result["runs"].items()
            },
            "deltas_clean_minus_original": result["deltas"],
            "classification": result["classification"],
            "reported_benchmark": result.get("reported"),
            "prediction_consistency": result.get("consistency"),
        })
    return payload


# ---------------------------------------------------------------------------
# Part E - self-test
# ---------------------------------------------------------------------------

def run_self_tests() -> int:
    mapper = _load_mdwd_eda()
    stem = mapper.source_stem
    h = "0123456789abcdef0123456789abcdef"

    def entry(name, split):
        return (name, split)

    cases = []

    same_split = [entry(f"a_jpg.rf.{i}{h[1:]}.jpg", "train") for i in range(3)]
    cases.append(("same source, multiple augmentations in one split -> not leaked",
                  len(derive_leakage(same_split, stem)) == 0))

    tv = [entry(f"b_jpg.rf.1{h[1:]}.jpg", "train"), entry(f"b_jpg.rf.2{h[1:]}.jpg", "valid")]
    leaked_tv = derive_leakage(tv, stem)
    sub_tv = clean_eval_subsets(tv, stem)
    cases.append(("train+valid derivatives -> leaked, category train+valid",
                  set(leaked_tv) == {"b_jpg"} and leaked_tv["b_jpg"]["category"] == "train+valid"))
    cases.append(("train+valid leak -> excluded from clean valid only",
                  sub_tv["valid"]["excluded"] == [f"b_jpg.rf.2{h[1:]}.jpg"]
                  and sub_tv["test"]["excluded"] == []))

    tt = [entry(f"c_jpg.rf.1{h[1:]}.jpg", "train"), entry(f"c_jpg.rf.2{h[1:]}.jpg", "test")]
    leaked_tt = derive_leakage(tt, stem)
    sub_tt = clean_eval_subsets(tt, stem)
    cases.append(("train+test derivatives -> leaked, category train+test",
                  leaked_tt.get("c_jpg", {}).get("category") == "train+test"))
    cases.append(("train+test leak -> excluded from clean test only",
                  sub_tt["test"]["excluded"] == [f"c_jpg.rf.2{h[1:]}.jpg"]
                  and sub_tt["valid"]["excluded"] == []))

    vt = [entry(f"d_jpg.rf.1{h[1:]}.jpg", "valid"), entry(f"d_jpg.rf.2{h[1:]}.jpg", "test")]
    leaked_vt = derive_leakage(vt, stem)
    sub_vt = clean_eval_subsets(vt, stem)
    cases.append(("valid+test derivatives -> leaked, category valid+test",
                  leaked_vt.get("d_jpg", {}).get("category") == "valid+test"))
    cases.append(("valid+test leak -> excluded from BOTH clean subsets",
                  sub_vt["valid"]["excluded"] == [f"d_jpg.rf.1{h[1:]}.jpg"]
                  and sub_vt["test"]["excluded"] == [f"d_jpg.rf.2{h[1:]}.jpg"]))

    all3 = [entry(f"e_jpg.rf.1{h[1:]}.jpg", "train"), entry(f"e_jpg.rf.2{h[1:]}.jpg", "valid"),
            entry(f"e_jpg.rf.3{h[1:]}.jpg", "test")]
    leaked_all3 = derive_leakage(all3, stem)
    sub_all3 = clean_eval_subsets(all3, stem)
    cases.append(("train+valid+test derivatives -> category train+valid+test",
                  leaked_all3.get("e_jpg", {}).get("category") == "train+valid+test"))
    cases.append(("train+valid+test leak -> excluded from both clean subsets",
                  len(sub_all3["valid"]["excluded"]) == 1 and len(sub_all3["test"]["excluded"]) == 1))

    similar = [entry(f"IMG_100_jpg.rf.1{h[1:]}.jpg", "train"),
               entry(f"IMG_1000_jpg.rf.2{h[1:]}.jpg", "valid")]
    cases.append(("similar-looking stems (IMG_100 vs IMG_1000) -> distinct, not leaked",
                  len(derive_leakage(similar, stem)) == 0))

    ext_fold = [entry(f"10_jpeg.rf.1{h[1:]}.jpg", "train"), entry(f"10_jpg.rf.2{h[1:]}.jpg", "valid")]
    cases.append(("extension folded into stem: 10_jpeg vs 10_jpg -> distinct identities",
                  len(derive_leakage(ext_fold, stem)) == 0))

    cases.append(("filename without .rf. -> stem passthrough",
                  stem("plain_photo.jpg") == "plain_photo"))

    mixed = (same_split
             + [entry(f"f_jpg.rf.1{h[1:]}.jpg", "train"), entry(f"f_jpg.rf.2{h[1:]}.jpg", "valid")]
             + [entry(f"g_jpg.rf.1{h[1:]}.jpg", "valid"), entry(f"g_jpg.rf.2{h[1:]}.jpg", "test")]
             + [entry(f"h_jpg.rf.1{h[1:]}.jpg", "valid")]
             + [entry(f"i_jpg.rf.1{h[1:]}.jpg", "test")])
    sub_mixed = clean_eval_subsets(mixed, stem)
    cases.append(("synthetic frame: clean valid keeps 1 of 3, clean test keeps 1 of 2",
                  len(sub_mixed["valid"]["kept"]) == 1 and len(sub_mixed["valid"]["excluded"]) == 2
                  and len(sub_mixed["test"]["kept"]) == 1 and len(sub_mixed["test"]["excluded"]) == 1))

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
        description="MDWD split-leakage sensitivity analysis (read-only).")
    parser.add_argument("--variant", default="MDWD-YOLO26",
                        help="Dataset variant to audit/evaluate (splits are identical across variants).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-test", action="store_true",
                        help="Run deterministic in-memory tests of the leakage logic and exit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run the audit and print results; write nothing.")
    parser.add_argument("--run-inference", action="store_true",
                        help="Re-evaluate checkpoints on original and clean subsets (loads models).")
    parser.add_argument("--models", default="yolo26l",
                        help="Comma list of <model>[@<suite>] or 'best-per-family'.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="", help="Ultralytics device ('' = auto).")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--working-conf", type=float, default=0.25,
                        help="Operating confidence for supplementary TP/FP/FN/P/R/F1.")
    parser.add_argument("--skip-cached", action="store_true",
                        help="Reuse existing scratch/runs metric+prediction caches.")
    args = parser.parse_args()

    if args.self_test:
        return run_self_tests()

    generated_at = datetime.now().isoformat(timespec="seconds")
    audit = run_audit(args.variant)
    affected = {s: len(audit["subsets"][s]["excluded"]) for s in EVAL_SPLITS}
    cross = audit["cross_check"]
    print(f"\nLeaked source identities: {len(audit['leaked'])} "
          f"({', '.join(f'{c}: {n}' for c, n in sorted(audit['category_counts'].items()))})")
    print(f"Affected evaluation images: valid {affected['valid']}, test {affected['test']}")
    print(f"Identity check: {'consistent' if audit['identity']['consistent'] else 'INCONSISTENT'}")
    if cross.get("csv_found"):
        print(f"EDA cross-check: {'MATCH' if cross['match'] else 'MISMATCH'}")
    lq = audit["label_quality"]
    print(f"Other issues: {len(lq['out_of_range'])} out-of-range boxes, "
          f"{len(lq['empty_annotations'])} empty annotations "
          f"({lq['in_evaluation_splits']} in evaluation splits)")

    checkpoints = discover_checkpoints()
    selected = resolve_models(args.models, checkpoints) if args.run_inference else []
    if args.run_inference:
        print("\nSelected checkpoints:")
        for ckpt in selected:
            print(f"  {ckpt['model']}@{ckpt['suite']}  {ckpt['path'].relative_to(ROOT)}")

    if args.dry_run:
        print(f"\nDry run: no files written. Would write to {args.output_dir}")
        if args.run_inference:
            print(f"Would run {len(selected) * 4} Ultralytics val runs "
                  f"(original/clean x valid/test per checkpoint).")
        return 0

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_audit_csvs(out_dir, audit)

    model_results = None
    environment = None
    if args.run_inference:
        os.environ.setdefault("WANDB_MODE", "disabled")
        import torch  # noqa: PLC0415
        import ultralytics  # noqa: PLC0415

        patch_ultralytics_cache_writes()
        pd_metrics = _load_pd_metrics()
        environment = {
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "device": (args.device or
                       (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")),
        }
        variant_dir = ROOT / "Datasets" / "MDWD" / args.variant
        clean_yaml = write_scratch(out_dir, variant_dir, audit["subsets"], audit["dataset_map"].class_names)
        original_yaml = variant_dir / "data.yaml"
        runs_root = out_dir / "scratch" / "runs"
        class_names = audit["dataset_map"].class_names

        gt_full = {split: gt_from_map(audit["dataset_map"], split) for split in EVAL_SPLITS}
        clean_stems = {
            split: {Path(n).stem for n in audit["subsets"][split]["kept"]} for split in EVAL_SPLITS
        }
        subset_counts = {(r["split"], r["subset"]): r for r in audit["subset_rows"]}

        model_results = []
        for ckpt in selected:
            print(f"\nEvaluating {ckpt['model']}@{ckpt['suite']}")
            runs = {}
            consistency = {}
            for split in EVAL_SPLITS:
                ul_orig, pred_json_orig = run_val(ckpt, original_yaml, split, "original", args, runs_root)
                ul_clean, pred_json_clean = run_val(ckpt, clean_yaml, split, "clean", args, runs_root)
                preds_orig = load_predictions_json(pred_json_orig, class_names)
                preds_clean = load_predictions_json(pred_json_clean, class_names)
                gt_orig = gt_full[split]
                gt_clean = {s: b for s, b in gt_orig.items() if s in clean_stems[split]}
                preds_orig_filtered = {
                    s: p for s, p in preds_orig.items() if s in clean_stems[split]
                }
                consistency[split] = prediction_consistency(preds_clean, preds_orig_filtered)
                sup_orig = supplementary_metrics(pd_metrics, preds_orig, gt_orig,
                                                 class_names, args.working_conf)
                sup_clean = supplementary_metrics(pd_metrics, preds_orig_filtered, gt_clean,
                                                  class_names, args.working_conf)
                runs[(split, "original")] = {
                    "ul": ul_orig, "sup": sup_orig,
                    "n_images": subset_counts[(split, "original")]["n_images"],
                    "n_gt_boxes": subset_counts[(split, "original")]["n_boxes"],
                }
                runs[(split, "clean")] = {
                    "ul": ul_clean, "sup": sup_clean,
                    "n_images": subset_counts[(split, "clean")]["n_images"],
                    "n_gt_boxes": subset_counts[(split, "clean")]["n_boxes"],
                }
            deltas = {}
            classification = {}
            for split in EVAL_SPLITS:
                delta = {
                    key: round(runs[(split, "clean")]["ul"][key] - runs[(split, "original")]["ul"][key], 4)
                    for key in ("precision", "recall", "f1", "map50", "map50_95")
                }
                deltas[split] = {f"ul_{k}": v for k, v in delta.items()}
                headline_pp = max(abs(delta["map50"]), abs(delta["map50_95"])) * 100
                classification[split] = classify_delta(headline_pp)
            model_results.append({
                "checkpoint": ckpt, "runs": runs, "deltas": deltas,
                "classification": classification, "reported": reported_metrics(ckpt),
                "consistency": consistency,
            })
            for split in EVAL_SPLITS:
                d = deltas[split]
                print(f"  {split}: dmAP50 {d['ul_map50']:+.4f}, dmAP50-95 "
                      f"{d['ul_map50_95']:+.4f} -> {classification[split]}")

        write_csv(out_dir / "metric_comparison.csv", METRIC_FIELDS, comparison_rows(model_results))

    payload = build_json_payload(audit, args, generated_at, model_results, environment)
    (out_dir / "metric_comparison.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    report = build_report(audit, args, generated_at, model_results, environment)
    (out_dir / "mdwd_leakage_sensitivity_report.md").write_text(report, encoding="utf-8")
    print(f"\nOutputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
