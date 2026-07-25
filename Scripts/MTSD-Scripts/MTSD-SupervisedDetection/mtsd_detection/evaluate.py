from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable


IOU_THRESHOLDS = [round(0.50 + 0.05 * index, 2) for index in range(10)]
MAX_DETS = [1, 10, 100]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def xyxy_to_coco(box: list[float] | tuple[float, float, float, float]) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in box]
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        raise ValueError(f"Non-finite xyxy box: {box}")
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"Degenerate xyxy box: {box}")
    return [x0, y0, x1 - x0, y1 - y0]


def clamp_coco_bbox(bbox: Iterable[float], width: int, height: int) -> tuple[list[float] | None, bool]:
    values = [float(value) for value in bbox]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return None, False
    x, y, box_width, box_height = values
    x0 = max(0.0, min(x, float(width)))
    y0 = max(0.0, min(y, float(height)))
    x1 = max(0.0, min(x + box_width, float(width)))
    y1 = max(0.0, min(y + box_height, float(height)))
    if x1 <= x0 or y1 <= y0:
        return None, True
    cleaned = [x0, y0, x1 - x0, y1 - y0]
    clamped = any(abs(before - after) > 1e-6 for before, after in zip(values, cleaned))
    return cleaned, clamped


def normalize_detection(image_id: int, category_id: int, xyxy, score: float) -> dict[str, Any]:
    return {
        "image_id": int(image_id),
        "category_id": int(category_id),
        "bbox": xyxy_to_coco(xyxy),
        "score": float(score),
    }


def normalize_clamped_detection(
    image_id: int,
    category_id: int,
    xyxy,
    score: float,
    width: int,
    height: int,
) -> tuple[dict[str, Any] | None, bool]:
    values = [float(value) for value in xyxy]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return None, False
    x0, y0, x1, y1 = values
    bbox, clamped = clamp_coco_bbox([x0, y0, x1 - x0, y1 - y0], width, height)
    if bbox is None:
        return None, clamped
    return {
        "image_id": int(image_id),
        "category_id": int(category_id),
        "bbox": bbox,
        "score": float(score),
    }, clamped


def load_rgb_image(path: Path):
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def validate_coco_results(annotation_payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    image_ids = {int(row["id"]) for row in annotation_payload.get("images", [])}
    category_ids = {int(row["id"]) for row in annotation_payload.get("categories", [])}
    for index, row in enumerate(rows):
        if int(row.get("image_id", -1)) not in image_ids:
            raise ValueError(f"Prediction {index} has unknown image_id {row.get('image_id')}")
        if int(row.get("category_id", -1)) not in category_ids:
            raise ValueError(f"Prediction {index} has unknown category_id {row.get('category_id')}")
        bbox = row.get("bbox", [])
        if len(bbox) != 4 or not all(math.isfinite(float(value)) for value in bbox):
            raise ValueError(f"Prediction {index} has invalid bbox {bbox}")
        if float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            raise ValueError(f"Prediction {index} has non-positive bbox {bbox}")
        score = float(row.get("score", -1))
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Prediction {index} has invalid score {score}")


def _filtered_evaluation_payload(
    annotation_payload: dict[str, Any],
    rows: list[dict[str, Any]],
    excluded_category_names: Iterable[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], int]:
    requested = {str(name).casefold() for name in excluded_category_names}
    categories = sorted(annotation_payload.get("categories", []), key=lambda row: int(row["id"]))
    known_names = {str(row["name"]).casefold() for row in categories}
    missing = sorted(requested - known_names)
    if missing:
        raise ValueError(f"Unknown excluded COCO categories: {missing}")
    excluded_ids = {int(row["id"]) for row in categories if str(row["name"]).casefold() in requested}
    kept_categories = [row for row in categories if int(row["id"]) not in excluded_ids]
    kept_ids = {int(row["id"]) for row in kept_categories}
    filtered = {
        **annotation_payload,
        "categories": kept_categories,
        "annotations": [
            row for row in annotation_payload.get("annotations", [])
            if int(row["category_id"]) in kept_ids
        ],
    }
    filtered_rows = [row for row in rows if int(row["category_id"]) in kept_ids]
    excluded_names = [str(row["name"]) for row in categories if int(row["id"]) in excluded_ids]
    return filtered, filtered_rows, excluded_names, len(rows) - len(filtered_rows)


def _mean_valid(values) -> float:
    import numpy as np

    array = np.asarray(values, dtype=float)
    valid = array[array >= 0]
    return float(np.mean(valid)) if valid.size else 0.0


def _per_class_metrics(evaluator, categories: list[dict[str, Any]], annotation_payload: dict[str, Any]) -> list[dict[str, Any]]:
    import numpy as np

    area_index = 0
    max_det_index = list(evaluator.params.maxDets).index(100)
    iou50_index = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.50))[0])
    iou75_index = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.75))[0])
    support = {int(row["id"]): 0 for row in categories}
    for annotation in annotation_payload.get("annotations", []):
        category_id = int(annotation["category_id"])
        if category_id in support:
            support[category_id] += 1
    rows = []
    for class_index, category in enumerate(categories):
        precision = evaluator.eval["precision"][:, :, class_index, area_index, max_det_index]
        recall = evaluator.eval["recall"][:, class_index, area_index, max_det_index]
        rows.append({
            "category_id": int(category["id"]),
            "category_name": str(category["name"]),
            "ground_truth_count": support[int(category["id"])],
            "ap50_95": _mean_valid(precision),
            "ap50": _mean_valid(precision[iou50_index]),
            "ap75": _mean_valid(precision[iou75_index]),
            "ar100": _mean_valid(recall),
            "ar50": float(recall[iou50_index]) if recall[iou50_index] >= 0 else 0.0,
        })
    return rows


def _pr_curve_rows(evaluator, categories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import numpy as np

    area_index = 0
    max_det_index = list(evaluator.params.maxDets).index(100)
    iou50_index = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.50))[0])
    recall_thresholds = evaluator.params.recThrs
    precision = evaluator.eval["precision"][iou50_index, :, :, area_index, max_det_index]
    scores = evaluator.eval["scores"][iou50_index, :, :, area_index, max_det_index]
    rows = []
    for class_index, category in enumerate(categories):
        for recall_index, recall in enumerate(recall_thresholds):
            class_precision = float(precision[recall_index, class_index])
            rows.append({
                "category_id": int(category["id"]),
                "category_name": str(category["name"]),
                "iou": 0.50,
                "recall": float(recall),
                "precision": class_precision if class_precision >= 0 else "",
                "score": float(scores[recall_index, class_index]) if class_precision >= 0 else "",
            })
    return rows


def _common_confidence_f1(evaluator) -> tuple[dict[str, float], list[dict[str, float]]]:
    import numpy as np

    iou_index = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.50))[0])
    area_range = list(evaluator.params.areaRng[0])
    max_det = 100
    detections: list[tuple[float, int, int]] = []
    ground_truth_count = 0
    for item in evaluator.evalImgs:
        if item is None or list(item["aRng"]) != area_range or int(item["maxDet"]) != max_det:
            continue
        gt_ignore = np.asarray(item["gtIgnore"], dtype=bool)
        ground_truth_count += int(np.count_nonzero(~gt_ignore))
        scores = np.asarray(item["dtScores"], dtype=float)
        matches = np.asarray(item["dtMatches"])[iou_index]
        ignored = np.asarray(item["dtIgnore"])[iou_index].astype(bool)
        for score, match, ignore in zip(scores, matches, ignored):
            if not ignore:
                detections.append((float(score), int(match > 0), int(match <= 0)))
    if not detections or ground_truth_count == 0:
        empty = [{"confidence": 1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                  "true_positives": 0, "false_positives": 0, "false_negatives": ground_truth_count}]
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "f1_score_threshold": 1.0}, empty
    detections.sort(key=lambda row: row[0], reverse=True)
    scores = np.asarray([row[0] for row in detections], dtype=float)
    cumulative_tp = np.cumsum([row[1] for row in detections])
    cumulative_fp = np.cumsum([row[2] for row in detections])
    endpoints = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True])
    curve = []
    for index in endpoints:
        tp = int(cumulative_tp[index])
        fp = int(cumulative_fp[index])
        false_negatives = max(ground_truth_count - tp, 0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / ground_truth_count
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        curve.append({
            "confidence": float(scores[index]),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": false_negatives,
        })
    best = max(curve, key=lambda row: (row["f1"], row["confidence"]))
    summary = {
        "precision": best["precision"],
        "recall": best["recall"],
        "f1": best["f1"],
        "f1_score_threshold": best["confidence"],
    }
    return summary, curve


def _plot_evaluation_artifacts(artifacts_dir: Path, pr_rows: list[dict[str, Any]], f1_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figure, axis = plt.subplots(figsize=(9, 7))
    category_names = list(dict.fromkeys(str(row["category_name"]) for row in pr_rows))
    for name in category_names:
        selected = [row for row in pr_rows if row["category_name"] == name and row["precision"] != ""]
        axis.plot([row["recall"] for row in selected], [row["precision"] for row in selected], label=name, linewidth=1.2)
    axis.set(xlabel="Recall", ylabel="Precision", title="Per-class precision-recall at IoU 0.50", xlim=(0, 1), ylim=(0, 1.02))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, loc="lower left")
    figure.tight_layout()
    figure.savefig(artifacts_dir / "pr_curves_iou50.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 6))
    confidence = [row["confidence"] for row in f1_rows]
    axis.plot(confidence, [row["precision"] for row in f1_rows], label="Precision")
    axis.plot(confidence, [row["recall"] for row in f1_rows], label="Recall")
    axis.plot(confidence, [row["f1"] for row in f1_rows], label="F1", linewidth=2)
    axis.set(xlabel="Common confidence threshold", ylabel="Score", title="Common confidence sweep at IoU 0.50", xlim=(1, 0), ylim=(0, 1.02))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(artifacts_dir / "confidence_f1_curve.png", dpi=180)
    plt.close(figure)


def _export_evaluation_artifacts(
    artifacts_dir: Path,
    annotation_payload: dict[str, Any],
    metrics: dict[str, Any],
    per_class: list[dict[str, Any]],
    pr_rows: list[dict[str, Any]],
    f1_rows: list[dict[str, Any]],
    coco_summary: str,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _write_json(artifacts_dir / "evaluation_annotations.coco.json", annotation_payload)
    _write_json(artifacts_dir / "metrics_summary.json", metrics)
    _write_csv(artifacts_dir / "per_class_metrics.csv", per_class,
               ["category_id", "category_name", "ground_truth_count", "ap50_95", "ap50", "ap75", "ar100", "ar50"])
    _write_csv(artifacts_dir / "pr_curves_iou50.csv", pr_rows,
               ["category_id", "category_name", "iou", "recall", "precision", "score"])
    _write_csv(artifacts_dir / "confidence_f1_curve.csv", f1_rows,
               ["confidence", "precision", "recall", "f1", "true_positives", "false_positives", "false_negatives"])
    (artifacts_dir / "coco_summary.txt").write_text(coco_summary, encoding="utf-8")
    _plot_evaluation_artifacts(artifacts_dir, pr_rows, f1_rows)


def coco_eval(
    annotation_file: Path,
    predictions_file: Path,
    *,
    excluded_category_names: Iterable[str] = (),
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("Unified evaluation requires pycocotools") from exc
    annotation_payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    input_rows = json.loads(predictions_file.read_text(encoding="utf-8"))
    annotation_payload, rows, excluded_names, excluded_predictions = _filtered_evaluation_payload(
        annotation_payload, input_rows, excluded_category_names
    )
    validate_coco_results(annotation_payload, rows)
    categories = sorted(annotation_payload.get("categories", []), key=lambda row: int(row["id"]))
    metadata = {
        "evaluator": "pycocotools.COCOeval",
        "iou_thresholds": IOU_THRESHOLDS,
        "max_dets": MAX_DETS,
        "prediction_count": len(rows),
        "input_prediction_count": len(input_rows),
        "excluded_prediction_count": excluded_predictions,
        "evaluated_image_count": len(annotation_payload.get("images", [])),
        "evaluated_category_count": len(categories),
        "excluded_category_names": excluded_names,
        "common_f1_iou": 0.50,
        "common_f1_averaging": "micro",
    }
    if not rows:
        per_class = [{
            "category_id": int(category["id"]), "category_name": str(category["name"]),
            "ground_truth_count": sum(1 for row in annotation_payload.get("annotations", [])
                                      if int(row["category_id"]) == int(category["id"])),
            "ap50_95": 0.0, "ap50": 0.0, "ap75": 0.0, "ar100": 0.0, "ar50": 0.0,
        } for category in categories]
        f1_rows = [{"confidence": 1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "true_positives": 0, "false_positives": 0,
                    "false_negatives": len(annotation_payload.get("annotations", []))}]
        metrics = {"map50_95": 0.0, "map50": 0.0, "map75": 0.0, "ar100": 0.0,
                   "map_small": 0.0, "map_medium": 0.0, "map_large": 0.0,
                   "ar_small": 0.0, "ar_medium": 0.0, "ar_large": 0.0,
                   "precision": 0.0, "recall": 0.0, "f1": 0.0,
                   "f1_score_threshold": 1.0, "per_class": per_class, **metadata}
        if artifacts_dir is not None:
            _export_evaluation_artifacts(artifacts_dir, annotation_payload, metrics, per_class, [], f1_rows, "No predictions.\n")
        return metrics
    truth = COCO()
    truth.dataset = annotation_payload
    truth.createIndex()
    predictions = truth.loadRes(rows)
    evaluator = COCOeval(truth, predictions, "bbox")
    evaluator.params.catIds = [int(row["id"]) for row in categories]
    evaluator.params.maxDets = MAX_DETS
    summary_capture = io.StringIO()
    with contextlib.redirect_stdout(summary_capture):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    per_class = _per_class_metrics(evaluator, categories, annotation_payload)
    pr_rows = _pr_curve_rows(evaluator, categories)
    f1_summary, f1_rows = _common_confidence_f1(evaluator)
    metrics = {
        "map50_95": float(evaluator.stats[0]),
        "map50": float(evaluator.stats[1]),
        "map75": float(evaluator.stats[2]),
        "map_small": float(evaluator.stats[3]),
        "map_medium": float(evaluator.stats[4]),
        "map_large": float(evaluator.stats[5]),
        "ar1": float(evaluator.stats[6]),
        "ar10": float(evaluator.stats[7]),
        "ar100": float(evaluator.stats[8]),
        "ar_small": float(evaluator.stats[9]),
        "ar_medium": float(evaluator.stats[10]),
        "ar_large": float(evaluator.stats[11]),
        **f1_summary,
        "per_class": per_class,
        **metadata,
    }
    if artifacts_dir is not None:
        _export_evaluation_artifacts(
            artifacts_dir, annotation_payload, metrics, per_class, pr_rows, f1_rows, summary_capture.getvalue()
        )
    return metrics


def safe_unified_evaluation(evaluator, require: bool = False) -> dict[str, Any]:
    try:
        return {"unified_eval_status": "completed", "unified_eval_error": None,
                "unified_test_metrics": evaluator()}
    except Exception as exc:
        if require:
            raise RuntimeError(f"Required unified COCO evaluation failed: {exc}") from exc
        return {"unified_eval_status": "failed", "unified_eval_error": str(exc),
                "unified_test_metrics": {}}


def _excluded_category_ids(truth: dict[str, Any], excluded_category_names: Iterable[str]) -> set[int]:
    requested = {str(name).casefold() for name in excluded_category_names}
    return {int(row["id"]) for row in truth.get("categories", []) if str(row["name"]).casefold() in requested}


def _add_prediction_metadata(metrics: dict[str, Any], artifacts_dir: Path | None, metadata: dict[str, int]) -> dict[str, Any]:
    metrics.update(metadata)
    if artifacts_dir is not None:
        _write_json(artifacts_dir / "metrics_summary.json", metrics)
    return metrics


def export_yolo_predictions(
    checkpoint: Path,
    coco_dir: Path,
    split: str,
    output: Path,
    image_size: int,
    device: str,
    *,
    excluded_category_names: Iterable[str] = (),
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    from ultralytics import YOLO

    annotation = coco_dir / split / "_annotations.coco.json"
    truth = json.loads(annotation.read_text(encoding="utf-8"))
    images_by_name = {row["file_name"]: row for row in truth["images"]}
    excluded_ids = _excluded_category_ids(truth, excluded_category_names)
    rows = []
    dropped = 0
    clamped = 0
    excluded = 0
    model = YOLO(str(checkpoint))
    resolved_device = None if device == "auto" else device
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        torch = None
    inference_started = time.perf_counter()
    evaluated_images = 0
    for result in model.predict(source=str(coco_dir / split), imgsz=image_size, device=resolved_device,
                                conf=0.001, iou=0.7, max_det=100, stream=True, verbose=False):
        image = images_by_name.get(Path(result.path).name)
        if image is None:
            continue
        evaluated_images += 1
        for xyxy, score, class_id in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
            category_id = int(class_id)
            if category_id in excluded_ids:
                excluded += 1
                continue
            row, was_clamped = normalize_clamped_detection(
                int(image["id"]), category_id, xyxy, float(score), int(image["width"]), int(image["height"])
            )
            clamped += int(was_clamped)
            if row is None:
                dropped += 1
            else:
                rows.append(row)
    inference_seconds = time.perf_counter() - inference_started
    peak_vram_mib = None
    if torch is not None:
        try:
            if torch.cuda.is_available():
                peak_vram_mib = torch.cuda.max_memory_allocated() / 1024 ** 2
        except RuntimeError:
            pass
    _write_json(output, rows)
    metrics = coco_eval(annotation, output, excluded_category_names=excluded_category_names, artifacts_dir=artifacts_dir)
    return _add_prediction_metadata(metrics, artifacts_dir, {
        "dropped_degenerate_predictions": dropped,
        "clamped_predictions": clamped,
        "excluded_predictions_during_export": excluded,
        "inference_seconds": inference_seconds,
        "inference_latency_ms_per_image": (
            1000.0 * inference_seconds / evaluated_images if evaluated_images else None
        ),
        "inference_throughput_images_per_second": (
            evaluated_images / inference_seconds if inference_seconds > 0 else None
        ),
        "inference_peak_vram_mib": peak_vram_mib,
    })


def export_rfdetr_predictions(
    spec,
    checkpoint: Path,
    coco_dir: Path,
    split: str,
    output: Path,
    image_size: int,
    device: str,
    *,
    excluded_category_names: Iterable[str] = (),
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    from .model_registry import _rfdetr_class
    import torch

    resolved_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model = _rfdetr_class(spec)(pretrain_weights=str(checkpoint), resolution=image_size,
                                gradient_checkpointing=False, device=resolved_device)
    annotation = coco_dir / split / "_annotations.coco.json"
    truth = json.loads(annotation.read_text(encoding="utf-8"))
    excluded_ids = _excluded_category_ids(truth, excluded_category_names)
    rows = []
    dropped = 0
    clamped = 0
    excluded = 0
    try:
        torch.cuda.reset_peak_memory_stats()
    except RuntimeError:
        pass
    inference_started = time.perf_counter()
    evaluated_images = 0
    for image in truth["images"]:
        rgb_image = load_rgb_image(coco_dir / split / image["file_name"])
        detections = model.predict(rgb_image, threshold=0.001)
        evaluated_images += 1
        xyxy = getattr(detections, "xyxy", [])
        confidence = getattr(detections, "confidence", [])
        class_id = getattr(detections, "class_id", [])
        for box, score, category in zip(xyxy, confidence, class_id):
            category_id = int(category)
            if category_id in excluded_ids:
                excluded += 1
                continue
            row, was_clamped = normalize_clamped_detection(
                int(image["id"]), category_id, box, float(score), int(image["width"]), int(image["height"])
            )
            clamped += int(was_clamped)
            if row is None:
                dropped += 1
            else:
                rows.append(row)
    inference_seconds = time.perf_counter() - inference_started
    try:
        peak_vram_mib = torch.cuda.max_memory_allocated() / 1024 ** 2
    except RuntimeError:
        peak_vram_mib = None
    _write_json(output, rows)
    metrics = coco_eval(annotation, output, excluded_category_names=excluded_category_names, artifacts_dir=artifacts_dir)
    return _add_prediction_metadata(metrics, artifacts_dir, {
        "dropped_degenerate_predictions": dropped,
        "clamped_predictions": clamped,
        "excluded_predictions_during_export": excluded,
        "input_color_mode": "RGB",
        "inference_seconds": inference_seconds,
        "inference_latency_ms_per_image": (
            1000.0 * inference_seconds / evaluated_images if evaluated_images else None
        ),
        "inference_throughput_images_per_second": (
            evaluated_images / inference_seconds if inference_seconds > 0 else None
        ),
        "inference_peak_vram_mib": peak_vram_mib,
    })
