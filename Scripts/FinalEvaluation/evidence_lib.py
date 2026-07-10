#!/usr/bin/env python3
"""Shared, read-only helpers for the FinalEvaluation dissertation tooling.

Used by robustness_slice_analysis.py, annotation_effort_report.py,
bootstrap_uncertainty.py and build_dissertation_dashboard.py. Everything here
reads existing repository artefacts; nothing writes outside the output
directory handed to it by the calling script.

Design rules shared by all consumers:
  * paths resolve from the repository root (sentinel walk, pathlib);
  * generated rows carry repo-relative provenance paths, never absolute ones;
  * evidence status vocabulary is fixed: final / historical / pilot / smoke /
    pending / excluded - nothing is promoted to "final" just because a file
    exists.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path = SCRIPT_DIR) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "Datasets").exists() and (parent / "Scripts").exists():
            return parent
    raise RuntimeError("Could not locate project root.")


ROOT = find_project_root()
REPORTS_ROOT = ROOT / "Documents" / "Final-Reports"
FIGURES_ROOT = ROOT / "Documents" / "Final-Figures"
TABLES_ROOT = ROOT / "Documents" / "Final-Tables"
BATCH_EVAL_DIR = ROOT / "Scripts" / "Other-Scripts" / "PromptDetect" / "batch_evaluation"
PROMPT_RESULTS = ROOT / "Results" / "PromptDetect" / "BatchEvaluation"
LEAKAGE_RUNS = REPORTS_ROOT / "MDWD-Leakage-Analysis" / "scratch" / "runs"
ATTR_OUTPUTS = ROOT / "Scripts" / "MTSD-Scripts" / "AttributeClassification" / "outputs"
MDWD_VARIANT_DIR = ROOT / "Datasets" / "MDWD" / "MDWD-YOLO26"

# Fixed evidence-status vocabulary (do not invent new values).
STATUS_FINAL = "final"
STATUS_HISTORICAL = "historical snapshot"
STATUS_PILOT = "pilot"
STATUS_SMOKE = "smoke test"
STATUS_PENDING = "pending"
STATUS_EXCLUDED = "excluded"

# A PromptDetect batch run below this image count is a pilot, never headline
# evidence. Conservative by design; raise only with an explicit protocol.
PROMPTDETECT_PILOT_MAX_IMAGES = 100

COCO_SMALL_MAX_AREA = 32 ** 2
COCO_MEDIUM_MAX_AREA = 96 ** 2


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------

def rel(path: Path | str) -> str:
    """Repo-relative POSIX path for provenance fields (never absolute)."""
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(k for row in rows for k in row)) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def new_output_dirs(category: str, overwrite: bool,
                    output_dir: Path | None = None) -> tuple[Path, Path]:
    """(report_dir, figures_dir) - timestamped by default, `latest/` with --overwrite."""
    stamp = "latest" if overwrite else timestamp()
    report_dir = (output_dir or (REPORTS_ROOT / category)) / stamp
    figures_dir = FIGURES_ROOT / category / stamp
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return report_dir, figures_dir


def git_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def provenance(source_file, generating_script: str, **extra) -> dict:
    record = {"source_file": rel(source_file) if source_file else "",
              "generating_script": generating_script,
              "commit_sha": git_commit_sha(),
              "generated_at": datetime.now().isoformat(timespec="seconds")}
    record.update(extra)
    return record


def rebase_stored_path(stored: str) -> Path | None:
    """Rebase an absolute path recorded on another session onto this checkout.

    Existing result CSVs store machine-absolute image paths; only the part
    from the first path component that exists directly under the repository
    root (``Datasets``, ``Results``, ...) is trusted.
    """
    parts = Path(str(stored).replace("\\", "/")).parts
    for index, part in enumerate(parts):
        if (ROOT / part).exists() and part in {"Datasets", "Results", "Documents", "Scripts"}:
            candidate = ROOT.joinpath(*parts[index:])
            return candidate
    return None


# ---------------------------------------------------------------------------
# Image statistics (brightness / contrast / blur)
# ---------------------------------------------------------------------------

def image_stats(path: Path) -> dict:
    """Brightness (mean grey), contrast (std grey), blur (variance of Laplacian).

    Computed on the stored image pixels (for MDWD these are the 640x640
    resized evaluation images). Higher blur value = sharper image.
    """
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        grey = np.asarray(image.convert("L"), dtype=np.float32)
    lap = (-4.0 * grey[1:-1, 1:-1] + grey[:-2, 1:-1] + grey[2:, 1:-1]
           + grey[1:-1, :-2] + grey[1:-1, 2:])
    return {"brightness": float(grey.mean()), "contrast": float(grey.std()),
            "sharpness": float(lap.var())}


def tercile_thresholds(values: list[float]) -> tuple[float, float]:
    """Dataset-derived 33.3/66.7 percentile thresholds (documented in outputs)."""
    ordered = sorted(values)
    if not ordered:
        return 0.0, 0.0

    def pct(fraction: float) -> float:
        index = fraction * (len(ordered) - 1)
        low = int(index)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)

    return pct(1 / 3), pct(2 / 3)


def tercile_label(value: float, thresholds: tuple[float, float],
                  labels: tuple[str, str, str] = ("low", "medium", "high")) -> str:
    if value <= thresholds[0]:
        return labels[0]
    if value <= thresholds[1]:
        return labels[1]
    return labels[2]


def coco_size_label(area_px: float) -> str:
    if area_px < COCO_SMALL_MAX_AREA:
        return "small"
    if area_px < COCO_MEDIUM_MAX_AREA:
        return "medium"
    return "large"


def density_label(n_objects: int) -> str:
    if n_objects <= 1:
        return "single-object"
    if n_objects <= 4:
        return "low-clutter"
    return "high-clutter"


def position_label(box: dict, width: float, height: float,
                   central_fraction: float = 0.5) -> str:
    """central = box centre inside the central `central_fraction` rectangle."""
    cx = (box["x0"] + box["x1"]) / 2
    cy = (box["y0"] + box["y1"]) / 2
    margin_x = width * (1 - central_fraction) / 2
    margin_y = height * (1 - central_fraction) / 2
    inside = (margin_x <= cx <= width - margin_x) and (margin_y <= cy <= height - margin_y)
    return "central" if inside else "near-edge"


# ---------------------------------------------------------------------------
# Detection matching (reuses PromptDetect batch_evaluation/metrics.py)
# ---------------------------------------------------------------------------

def load_pd_metrics():
    """Load batch_evaluation/metrics.py standalone (stdlib-only module)."""
    spec = importlib.util.spec_from_file_location("pd_metrics", BATCH_EVAL_DIR / "metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def match_detections(pd_metrics, preds: list[dict], gts: list[dict],
                     iou_threshold: float = 0.5, class_aware: bool = True) -> dict:
    """Per-image class-aware greedy matching.

    Returns {"tp","fp","fn","gt_records","pred_records"}; gt_records carry
    matched/iou/label per ground-truth box, pred_records carry is_tp/score.
    """
    class_names = sorted({b.get("class_name", "") for b in gts}
                         | ({p.get("class_name", "") for p in preds} if class_aware else set()))
    groups = class_names if class_aware else [None]
    totals = {"tp": 0, "fp": 0, "fn": 0}
    gt_records, pred_records = [], []
    for group in groups:
        gt_c = [b for b in gts if group is None or b.get("class_name", "") == group]
        pr_c = [p for p in preds if group is None or p.get("class_name", "") == group]
        result = pd_metrics.match_image(pr_c, gt_c, iou_threshold)
        totals["tp"] += result["tp"]
        totals["fp"] += result["fp"]
        totals["fn"] += result["fn"]
        matched_ids = {id(gt): (pred, iou) for pred, gt, iou in result["matches"]}
        for gt in gt_c:
            pred, iou = matched_ids.get(id(gt), (None, 0.0))
            gt_records.append({**gt, "matched": pred is not None, "iou": iou,
                               "match_score": float(pred.get("score", 0.0)) if pred else None})
        for pred, flag in zip(result["preds_sorted"], result["pred_flags"]):
            pred_records.append({**pred, "is_tp": flag})
    return {**totals, "gt_records": gt_records, "pred_records": pred_records}


def prf_from_counts(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ---------------------------------------------------------------------------
# Ground truth / prediction loaders for stored evidence
# ---------------------------------------------------------------------------

def load_mdwd_split_gt(split: str, variant_dir: Path = MDWD_VARIANT_DIR) -> dict:
    """MDWD valid/test ground truth: {image_stem: {"path","width","height","boxes"}}.

    Boxes are pixel xyxy dicts with class_name (labels are YOLO-normalised in
    the 640x640 export; measurements are therefore resized evaluation pixels).
    """
    import yaml
    from PIL import Image

    class_names = list((yaml.safe_load((variant_dir / "data.yaml")
                                       .read_text(encoding="utf-8")) or {}).get("names", []))
    images_dir = variant_dir / split / "images"
    labels_dir = variant_dir / split / "labels"
    records: dict[str, dict] = {}
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        boxes = []
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(parts[0])
                    cx, cy, bw, bh = (float(v) for v in parts[1:5])
                except ValueError:
                    continue
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    continue  # same strict rule as mdwd_eda; all hits are in train
                boxes.append({
                    "class_name": class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id),
                    "x0": (cx - bw / 2) * width, "y0": (cy - bh / 2) * height,
                    "x1": (cx + bw / 2) * width, "y1": (cy + bh / 2) * height,
                })
        records[image_path.stem] = {"path": image_path, "width": width,
                                    "height": height, "boxes": boxes}
    return {"class_names": class_names, "records": records,
            "source": rel(variant_dir / split)}


def discover_leakage_prediction_runs() -> list[dict]:
    """Stored MDWD detection predictions (from the leakage sensitivity runs).

    Only `original`-subset runs are offered: they cover the full official
    valid/test splits with the original evaluation protocol.
    """
    runs = []
    if not LEAKAGE_RUNS.is_dir():
        return runs
    for run_dir in sorted(LEAKAGE_RUNS.iterdir()):
        predictions = run_dir / "predictions.json"
        if not predictions.exists():
            continue
        parts = run_dir.name.split("__")
        if len(parts) != 4 or parts[3] != "original":
            continue
        runs.append({"model": parts[0], "suite": parts[1], "split": parts[2],
                     "subset": parts[3], "predictions": predictions,
                     "run_dir": run_dir})
    return runs


def load_leakage_predictions(predictions_path: Path, class_names: list[str],
                             min_score: float = 0.0) -> dict[str, list[dict]]:
    """Ultralytics predictions.json -> {image_stem: [pred dicts]} (category_id is 1-based)."""
    preds: dict[str, list[dict]] = {}
    for entry in read_json(predictions_path):
        score = float(entry.get("score", 0.0))
        if score < min_score:
            continue
        x, y, w, h = entry["bbox"]
        class_index = int(entry["category_id"]) - 1
        preds.setdefault(str(entry["image_id"]), []).append({
            "class_name": class_names[class_index] if 0 <= class_index < len(class_names) else "?",
            "x0": x, "y0": y, "x1": x + w, "y1": y + h, "score": score,
        })
    return preds


def discover_promptdetect_runs(dataset: str | None = None) -> list[dict]:
    """PromptDetect batch-eval runs with pilot/final classification."""
    runs = []
    if not PROMPT_RESULTS.is_dir():
        return runs
    for config_path in sorted(PROMPT_RESULTS.glob("*/*/run_config.json")):
        try:
            config = read_json(config_path)
        except Exception:
            continue
        if dataset and str(config.get("dataset", "")).upper() != dataset.upper():
            continue
        n_images = int(config.get("n_images") or 0)
        runs.append({
            "run_dir": config_path.parent,
            "run_id": f"{config_path.parent.parent.name}/{config_path.parent.name}",
            "config": config,
            "n_images": n_images,
            "status": STATUS_PILOT if n_images < PROMPTDETECT_PILOT_MAX_IMAGES else STATUS_FINAL,
        })
    return runs


def load_promptdetect_run(run_dir: Path) -> dict:
    """Load config, GT and predictions of one batch-eval run, keyed for matching."""
    config = read_json(run_dir / "run_config.json")
    gt_by_image: dict[str, dict] = {}
    for row in read_csv_rows(run_dir / "ground_truth_index.csv"):
        record = gt_by_image.setdefault(
            row["image_id"], {"image_id": row["image_id"],
                              "stored_path": row.get("image_path", ""), "boxes": []})
        if row.get("x0") not in ("", None):
            record["boxes"].append({"class_name": row.get("class_name", ""),
                                    "x0": float(row["x0"]), "y0": float(row["y0"]),
                                    "x1": float(row["x1"]), "y1": float(row["y1"])})
    preds: dict[tuple, list[dict]] = {}
    for row in read_csv_rows(run_dir / "predictions.csv"):
        key = (row["model"], row["prompt"], row["image_id"])
        preds.setdefault(key, [])
        if row.get("x0") in ("", None):
            continue  # "<no detections>" placeholder row - model ran, found nothing
        preds[key].append({
            "class_name": row.get("predicted_label", ""),
            "x0": float(row["x0"]), "y0": float(row["y0"]),
            "x1": float(row["x1"]), "y1": float(row["y1"]),
            "score": float(row.get("score", 0.0) or 0.0),
            "has_confidence": str(row.get("has_confidence", "")).lower() == "true",
        })
    per_image_rows = read_csv_rows(run_dir / "per_image_metrics.csv")
    return {"config": config, "gt_by_image": gt_by_image, "predictions": preds,
            "per_image_rows": per_image_rows, "run_dir": run_dir}


def discover_attribute_variants(include_smoke: bool = False) -> list[dict]:
    """Completed attribute-classification variants with their test metrics."""
    variants = []
    metrics_root = ATTR_OUTPUTS / "metrics"
    if not metrics_root.is_dir():
        return variants
    for variant_dir in sorted(metrics_root.iterdir()):
        metrics_path = variant_dir / "test_metrics.json"
        if not metrics_path.exists():
            continue
        smoke = variant_dir.name.endswith("-smoke")
        if smoke and not include_smoke:
            continue
        try:
            payload = read_json(metrics_path)
        except Exception:
            continue
        variants.append({"variant": variant_dir.name, "metrics_path": metrics_path,
                         "payload": payload,
                         "status": STATUS_SMOKE if smoke else STATUS_HISTORICAL})
    return variants


def confusion_to_samples(confusion: list[list[int]]) -> list[tuple[int, int]]:
    """Reconstruct per-sample (true, pred) outcomes from a stored confusion matrix.

    Valid for unpaired per-head bootstrap: each crop contributes exactly one
    cell count. It cannot align samples across heads or variants.
    """
    samples = []
    for true_index, row in enumerate(confusion):
        for pred_index, count in enumerate(row):
            samples.extend([(true_index, pred_index)] * int(count))
    return samples


# ---------------------------------------------------------------------------
# Bootstrap core (percentile intervals, seeded)
# ---------------------------------------------------------------------------

def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = fraction * (len(sorted_values) - 1)
    low = int(index)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (index - low)


def bootstrap_ci(units: list, metric_fn, n_bootstrap: int = 2000, seed: int = 42,
                 confidence_level: float = 0.95) -> dict:
    """Seeded percentile bootstrap over resampled units.

    metric_fn(list_of_units) -> float. Returns point estimate on the full set
    plus the percentile interval over `n_bootstrap` resamples.
    """
    rng = random.Random(seed)
    point = metric_fn(units)
    if not units:
        return {"point": point, "low": point, "high": point, "n_units": 0,
                "n_bootstrap": 0, "method": "percentile"}
    stats = sorted(metric_fn([units[rng.randrange(len(units))] for _ in range(len(units))])
                   for _ in range(n_bootstrap))
    alpha = (1 - confidence_level) / 2
    return {"point": round(point, 4),
            "low": round(percentile(stats, alpha), 4),
            "high": round(percentile(stats, 1 - alpha), 4),
            "n_units": len(units), "n_bootstrap": n_bootstrap,
            "confidence_level": confidence_level, "method": "percentile"}


def paired_bootstrap(units: list, metric_a, metric_b, n_bootstrap: int = 2000,
                     seed: int = 42, confidence_level: float = 0.95) -> dict:
    """Paired percentile bootstrap of metric_a(units) - metric_b(units).

    The same resampled unit indices feed both metrics, preserving pairing.
    Also reports the proportion of resamples where A strictly exceeds B.
    """
    rng = random.Random(seed)
    observed = metric_a(units) - metric_b(units)
    if not units:
        return {"observed_difference": observed, "low": observed, "high": observed,
                "p_a_better": 0.0, "n_units": 0, "n_bootstrap": 0, "method": "paired percentile"}
    differences = []
    a_wins = 0
    for _ in range(n_bootstrap):
        sample = [units[rng.randrange(len(units))] for _ in range(len(units))]
        difference = metric_a(sample) - metric_b(sample)
        differences.append(difference)
        a_wins += difference > 0
    differences.sort()
    alpha = (1 - confidence_level) / 2
    return {"observed_difference": round(observed, 4),
            "low": round(percentile(differences, alpha), 4),
            "high": round(percentile(differences, 1 - alpha), 4),
            "p_a_better": round(a_wins / n_bootstrap, 4),
            "n_units": len(units), "n_bootstrap": n_bootstrap,
            "confidence_level": confidence_level, "method": "paired percentile"}


# ---------------------------------------------------------------------------
# Figure helper (PNG + PDF + SVG, publication-restrained styling)
# ---------------------------------------------------------------------------

def save_figure(figure, figures_dir: Path, name: str) -> list[str]:
    """Save one matplotlib figure as PNG (300 dpi), PDF and SVG."""
    saved = []
    for extension in ("png", "pdf", "svg"):
        target = figures_dir / f"{name}.{extension}"
        figure.savefig(target, dpi=300 if extension == "png" else None,
                       bbox_inches="tight")
        saved.append(rel(target))
    import matplotlib.pyplot as plt

    plt.close(figure)
    return saved


def apply_plot_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 110, "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "legend.frameon": False, "savefig.facecolor": "white",
    })
