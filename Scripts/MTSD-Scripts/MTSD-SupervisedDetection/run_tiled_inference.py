#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mtsd_detection.evaluate import coco_eval, load_rgb_image, normalize_clamped_detection  # noqa: E402
from mtsd_detection.manifests import sha256_file  # noqa: E402
from mtsd_detection.model_registry import registry  # noqa: E402


DEFAULT_STATE = ROOT / "Results" / "MTSD-Runs" / "Supervised-Matrix" / "20260719-003416" / "state.json"
DEFAULT_DATASET = ROOT / "Datasets" / "MTSD" / "Prepared" / "MTSD-Augmented" / "MTSD-COCO"
DEFAULT_OUTPUT_ROOT = ROOT / "Results" / "MTSD-Results" / "Tiled-Evaluation-NoTourist"
EXCLUDED_CATEGORIES = ("Tourist Sign",)


def tile_origins(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    origins = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if origins[-1] != last:
        origins.append(last)
    return origins


def tile_windows(width: int, height: int, tile_size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    return [
        (left, top, min(left + tile_size, width), min(top + tile_size, height))
        for top in tile_origins(height, tile_size, overlap)
        for left in tile_origins(width, tile_size, overlap)
    ]


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def merge_class_aware_nms(candidates: list[dict[str, Any]], iou_threshold: float, max_detections: int) -> list[dict[str, Any]]:
    if not candidates:
        return []
    import torch
    from torchvision.ops import batched_nms

    boxes = torch.tensor([row["xyxy"] for row in candidates], dtype=torch.float32)
    scores = torch.tensor([row["score"] for row in candidates], dtype=torch.float32)
    classes = torch.tensor([row["category_id"] for row in candidates], dtype=torch.int64)
    keep = batched_nms(boxes, scores, classes, float(iou_threshold))[:max_detections].tolist()
    return [candidates[index] for index in keep]


def _predict_yolo(model, crops, image_size: int, device: str, batch_size: int):
    resolved_device = None if device == "auto" else device
    results = []
    for batch in _chunks(crops, batch_size):
        results.extend(model.predict(source=batch, imgsz=image_size, device=resolved_device, conf=0.001,
                                     iou=0.7, max_det=300, batch=len(batch), stream=False, verbose=False))
    return [(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
             result.boxes.cls.cpu().tolist()) for result in results]


def _predict_rfdetr(model, crops, batch_size: int):
    results = []
    for batch in _chunks(crops, batch_size):
        detections = model.predict(batch, threshold=0.001)
        if len(batch) == 1:
            detections = [detections]
        results.extend(detections)
    return [(getattr(result, "xyxy", []), getattr(result, "confidence", []),
             getattr(result, "class_id", [])) for result in results]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _image_size(record: dict[str, Any], trainer: str) -> int:
    if trainer == "ultralytics":
        return int(record.get("train_args", {}).get("imgsz", 640))
    return int(record.get("model_args", {}).get("resolution", 576))


def _run_model(
    model_key: str,
    state: dict[str, Any],
    dataset_root: Path,
    output_dir: Path,
    tile_size: int,
    overlap: float,
    nms_iou: float,
    max_detections: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    import torch

    spec = registry()[model_key]
    run_dir = Path(state["models"][model_key]["run_dir"])
    source_record = run_dir / "run_record.json"
    record = json.loads(source_record.read_text(encoding="utf-8"))
    checkpoint = Path(record["checkpoint_best"])
    model_resolution = _image_size(record, spec.trainer)
    if spec.trainer == "ultralytics":
        from ultralytics import YOLO

        model = YOLO(str(checkpoint))
    else:
        from mtsd_detection.model_registry import _rfdetr_class

        resolved_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        model = _rfdetr_class(spec)(pretrain_weights=str(checkpoint), resolution=model_resolution,
                                    gradient_checkpointing=False, device=resolved_device)
    annotation = dataset_root / "test" / "_annotations.coco.json"
    truth = json.loads(annotation.read_text(encoding="utf-8"))
    excluded_ids = {
        int(row["id"]) for row in truth["categories"]
        if str(row["name"]).casefold() in {name.casefold() for name in EXCLUDED_CATEGORIES}
    }
    predictions: list[dict[str, Any]] = []
    tile_count = 0
    raw_prediction_count = 0
    nms_removed = 0
    dropped = 0
    clamped = 0
    excluded = 0
    started = time.perf_counter()
    for image_index, image_record in enumerate(truth["images"], start=1):
        rgb = load_rgb_image(dataset_root / "test" / image_record["file_name"])
        windows = tile_windows(rgb.width, rgb.height, tile_size, overlap)
        crops = [rgb.crop(window) for window in windows]
        tile_count += len(crops)
        if spec.trainer == "ultralytics":
            tile_results = _predict_yolo(model, crops, model_resolution, device, batch_size)
        else:
            tile_results = _predict_rfdetr(model, crops, batch_size)
        candidates: list[dict[str, Any]] = []
        for window, (boxes, scores, classes) in zip(windows, tile_results):
            left, top, _, _ = window
            for box, score, category in zip(boxes, scores, classes):
                raw_prediction_count += 1
                category_id = int(category)
                if category_id in excluded_ids:
                    excluded += 1
                    continue
                x0, y0, x1, y1 = [float(value) for value in box]
                row, was_clamped = normalize_clamped_detection(
                    int(image_record["id"]), category_id,
                    [x0 + left, y0 + top, x1 + left, y1 + top], float(score),
                    int(image_record["width"]), int(image_record["height"]),
                )
                clamped += int(was_clamped)
                if row is None:
                    dropped += 1
                    continue
                clean_x, clean_y, clean_width, clean_height = row["bbox"]
                candidates.append({
                    "xyxy": [clean_x, clean_y, clean_x + clean_width, clean_y + clean_height],
                    "score": float(score),
                    "category_id": category_id,
                    "prediction": row,
                })
        merged = merge_class_aware_nms(candidates, nms_iou, max_detections)
        nms_removed += len(candidates) - len(merged)
        for candidate in merged:
            predictions.append(candidate["prediction"])
        if image_index % 25 == 0 or image_index == len(truth["images"]):
            print(f"{model_key}: {image_index}/{len(truth['images'])} images, {tile_count} tiles", flush=True)
    model_dir = output_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=False)
    predictions_file = model_dir / "predictions.json"
    _write_json(predictions_file, predictions)
    metrics = coco_eval(annotation, predictions_file, excluded_category_names=EXCLUDED_CATEGORIES,
                        artifacts_dir=model_dir)
    metrics.update({
        "tile_count": tile_count,
        "raw_tile_prediction_count": raw_prediction_count,
        "nms_removed_predictions": nms_removed,
        "dropped_degenerate_predictions": dropped,
        "clamped_predictions": clamped,
        "excluded_predictions_during_export": excluded,
    })
    _write_json(model_dir / "metrics_summary.json", metrics)
    evaluation_seconds = time.perf_counter() - started
    result = {
        "model": model_key,
        "family": spec.family,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_run_record": str(source_record.resolve()),
        "model_resolution": model_resolution,
        "tile_size": tile_size,
        "overlap": overlap,
        "merge": "torchvision.ops.batched_nms",
        "nms_iou": nms_iou,
        "max_detections_per_image": max_detections,
        "excluded_categories": list(EXCLUDED_CATEGORIES),
        "evaluation_seconds": evaluation_seconds,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    _write_json(model_dir / "evaluation_record.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Overlapping full-resolution tiled inference for trained MTSD models")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--baseline-evaluation", type=Path,
                        help="Optional no-tourist evaluation_manifest.json for direct non-tiled deltas")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--nms-iou", type=float, default=0.55)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.tile_size <= 0 or not 0 <= args.overlap < 1 or not 0 <= args.nms_iou <= 1:
        raise ValueError("Invalid tile size, overlap, or NMS IoU")
    state = json.loads(args.state.read_text(encoding="utf-8"))
    unknown = [key for key in args.models if key not in state["models"] or key not in registry()]
    if unknown:
        raise ValueError(f"Models are absent from the completed matrix: {unknown}")
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for index, model_key in enumerate(args.models, start=1):
        print(f"[{index}/{len(args.models)}] {model_key}: tiled inference started", flush=True)
        result = _run_model(model_key, state, args.dataset_root, output_dir, args.tile_size, args.overlap,
                            args.nms_iou, args.max_detections, args.batch_size, args.device)
        results.append(result)
        metrics = result["metrics"]
        print(f"[{index}/{len(args.models)}] {model_key}: mAP50-95={metrics['map50_95']:.4f}, "
              f"small-AP={metrics['map_small']:.4f}, F1={metrics['f1']:.4f}", flush=True)
    scalar_keys = ("map50_95", "map50", "map75", "ar100", "map_small", "ar_small",
                   "precision", "recall", "f1", "f1_score_threshold", "prediction_count")
    rows = [{"model": result["model"], "model_resolution": result["model_resolution"],
             "tile_size": result["tile_size"], "overlap": result["overlap"],
             **{key: result["metrics"].get(key) for key in scalar_keys},
             "evaluation_seconds": result["evaluation_seconds"]} for result in results]
    if args.baseline_evaluation is not None:
        baseline_manifest = json.loads(args.baseline_evaluation.read_text(encoding="utf-8"))
        baseline = {row["model"]: row for row in baseline_manifest["model_metrics"]}
        missing = [row["model"] for row in rows if row["model"] not in baseline]
        if missing:
            raise ValueError(f"Baseline evaluation is missing tiled models: {missing}")
        for row in rows:
            base = baseline[row["model"]]
            for metric in ("map50_95", "map50", "map75", "ar100", "map_small", "ar_small", "f1"):
                row[f"baseline_{metric}"] = base[metric]
                row[f"delta_{metric}"] = row[metric] - base[metric]
    _write_csv(output_dir / "model_metrics.csv", rows)
    _write_json(output_dir / "evaluation_manifest.json", {
        "status": "completed", "models": args.models, "excluded_categories": list(EXCLUDED_CATEGORIES),
        "tile_size": args.tile_size, "overlap": args.overlap, "nms_iou": args.nms_iou,
        "baseline_evaluation": str(args.baseline_evaluation.resolve()) if args.baseline_evaluation else None,
        "completed_at": datetime.now(timezone.utc).isoformat(), "model_metrics": rows,
    })
    print(json.dumps({"status": "completed", "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
