#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mtsd_detection.evaluate import (  # noqa: E402
    coco_eval,
    export_rfdetr_predictions,
    export_yolo_predictions,
)
from mtsd_detection.manifests import sha256_file  # noqa: E402
from mtsd_detection.model_registry import registry  # noqa: E402


DEFAULT_STATE = ROOT / "Results" / "MTSD-Runs" / "Supervised-Matrix" / "20260719-003416" / "state.json"
DEFAULT_DATASET = ROOT / "Datasets" / "MTSD" / "Prepared" / "MTSD-Augmented" / "MTSD-COCO"
DEFAULT_OUTPUT_ROOT = ROOT / "Results" / "MTSD-Results" / "Unified-Evaluation-NoTourist"
EXCLUDED_CATEGORIES = ("Tourist Sign",)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _image_size(record: dict[str, Any], trainer: str) -> int:
    if trainer == "ultralytics":
        return int(record.get("train_args", {}).get("imgsz", 640))
    return int(record.get("model_args", {}).get("resolution", 576))


def main() -> int:
    parser = argparse.ArgumentParser(description="Rerun unified MTSD COCO evaluation without retraining")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--reuse-predictions", action="store_true")
    args = parser.parse_args()

    state = json.loads(args.state.read_text(encoding="utf-8"))
    requested = args.models or list(state["requested_matrix"])
    unknown = [model for model in requested if model not in state["models"] or model not in registry()]
    if unknown:
        raise ValueError(f"Models are absent from the completed matrix: {unknown}")
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=False)
    annotation = args.dataset_root / "test" / "_annotations.coco.json"
    model_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []

    for index, model_key in enumerate(requested, start=1):
        spec = registry()[model_key]
        run_dir = Path(state["models"][model_key]["run_dir"])
        record_path = run_dir / "run_record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        checkpoint = Path(record["checkpoint_best"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing trained checkpoint for {model_key}: {checkpoint}")
        model_dir = output_dir / model_key
        model_dir.mkdir()
        predictions = model_dir / "predictions.json"
        started = time.perf_counter()
        print(f"[{index}/{len(requested)}] {model_key}: evaluation started", flush=True)
        if args.reuse_predictions and predictions.is_file():
            metrics = coco_eval(annotation, predictions, excluded_category_names=EXCLUDED_CATEGORIES,
                                artifacts_dir=model_dir)
        elif spec.trainer == "ultralytics":
            metrics = export_yolo_predictions(
                checkpoint, args.dataset_root, "test", predictions, _image_size(record, spec.trainer), args.device,
                excluded_category_names=EXCLUDED_CATEGORIES, artifacts_dir=model_dir,
            )
        else:
            metrics = export_rfdetr_predictions(
                spec, checkpoint, args.dataset_root, "test", predictions, _image_size(record, spec.trainer), args.device,
                excluded_category_names=EXCLUDED_CATEGORIES, artifacts_dir=model_dir,
            )
        elapsed = time.perf_counter() - started
        evaluation_record = {
            "model": model_key,
            "family": spec.family,
            "scale": spec.scale,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "source_run_record": str(record_path.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "annotation_sha256": sha256_file(annotation),
            "excluded_categories": list(EXCLUDED_CATEGORIES),
            "image_size": _image_size(record, spec.trainer),
            "evaluation_seconds": elapsed,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        }
        _write_json(model_dir / "evaluation_record.json", evaluation_record)
        scalar_keys = (
            "map50_95", "map50", "map75", "ar100", "map_small", "map_medium", "map_large",
            "ar_small", "ar_medium", "ar_large", "precision", "recall", "f1", "f1_score_threshold",
            "prediction_count", "dropped_degenerate_predictions", "clamped_predictions",
        )
        model_rows.append({
            "model": model_key,
            "family": spec.family,
            "scale": spec.scale,
            "image_size": _image_size(record, spec.trainer),
            **{key: metrics.get(key) for key in scalar_keys},
            "evaluation_seconds": elapsed,
            "predictions": str(predictions.resolve()),
        })
        class_rows.extend({"model": model_key, **row} for row in metrics["per_class"])
        print(f"[{index}/{len(requested)}] {model_key}: mAP50-95={metrics['map50_95']:.4f}, "
              f"mAP50={metrics['map50']:.4f}, F1={metrics['f1']:.4f}", flush=True)

    _write_csv(output_dir / "model_metrics.csv", model_rows)
    _write_csv(output_dir / "per_class_metrics.csv", class_rows)
    manifest = {
        "status": "completed",
        "models": requested,
        "model_count": len(requested),
        "excluded_categories": list(EXCLUDED_CATEGORIES),
        "source_state": str(args.state.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model_metrics": model_rows,
    }
    _write_json(output_dir / "evaluation_manifest.json", manifest)
    print(json.dumps({"status": "completed", "output_dir": str(output_dir), "model_count": len(requested)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
