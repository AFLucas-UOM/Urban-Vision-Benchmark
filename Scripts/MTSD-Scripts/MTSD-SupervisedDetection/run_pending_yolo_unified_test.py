#!/usr/bin/env python3
"""Evaluate completed FINAL YOLO runs that still lack unified test metrics."""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mtsd_detection.evaluate import export_yolo_predictions  # noqa: E402
from mtsd_detection.manifests import sha256_file  # noqa: E402
from mtsd_detection.model_registry import registry  # noqa: E402


EXCLUDED = ("Tourist Sign",)
FAMILIES = ("YOLO11-MTSD", "YOLO12-MTSD", "YOLO26-MTSD")
FINAL_NAME = re.compile(r"^FINAL_(yolo(?:11|12|26)[nsm])_mtsd_(strong|noaug)_img(\d+)_")
SCALARS = (
    "map50_95", "map50", "map75", "ar100", "map_small", "map_medium", "map_large",
    "ar_small", "ar_medium", "ar_large", "precision", "recall", "f1", "f1_score_threshold",
    "prediction_count", "dropped_degenerate_predictions", "clamped_predictions",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pending_runs() -> list[tuple[Path, dict]]:
    found: list[tuple[Path, dict]] = []
    for family in FAMILIES:
        for run_dir in sorted((REPO / "Results" / "MTSD-Runs" / family).glob("FINAL_*")):
            record_path = run_dir / "run_record.json"
            if not record_path.is_file():
                continue
            record = read_json(record_path)
            if record.get("status") != "completed" or record.get("unified_test_metrics"):
                continue
            match = FINAL_NAME.match(run_dir.name)
            if not match:
                continue
            if match.group(1) not in registry():
                continue
            checkpoint = Path(record.get("checkpoint_best", ""))
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint for {run_dir}: {checkpoint}")
            found.append((run_dir, record))
    return found


def label(run_dir: Path) -> str:
    match = FINAL_NAME.match(run_dir.name)
    assert match
    model, augmentation, size = match.groups()
    return f"{model.upper()}-{('NoAug' if augmentation == 'noaug' else 'Strong')}-{size}"


def dataset_root(record: dict) -> Path:
    data_path = str(record.get("train_args", {}).get("data", ""))
    if "MTSD-Augmented-Strong" in data_path:
        return REPO / "Datasets" / "MTSD" / "Prepared" / "MTSD-Augmented-Strong" / "MTSD-COCO"
    if "MTSD-Unaugmented" in data_path:
        return REPO / "Datasets" / "MTSD" / "Prepared" / "MTSD-Unaugmented" / "MTSD-COCO"
    raise ValueError(f"Cannot determine prepared test corpus from train_args.data: {data_path}")


def main() -> int:
    runs = pending_runs()
    if not runs:
        print(json.dumps({"status": "nothing_to_evaluate", "count": 0}, indent=2))
        return 0
    output_dir = REPO / "Results" / "MTSD-Results" / "Unified-Evaluation-NoTourist" / (
        datetime.now().strftime("%Y%m%d-%H%M%S-pending-yolo-test")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    class_rows: list[dict] = []
    manifest_runs = []
    for index, (run_dir, record) in enumerate(runs, start=1):
        match = FINAL_NAME.match(run_dir.name)
        assert match
        model_key, augmentation, size_text = match.groups()
        spec = registry()[model_key]
        size = int(size_text)
        dataset = dataset_root(record)
        model_dir = output_dir / label(run_dir)
        model_dir.mkdir()
        predictions = model_dir / "predictions.json"
        checkpoint = Path(record["checkpoint_best"])
        started = time.perf_counter()
        print(f"[{index}/{len(runs)}] {label(run_dir)}: starting", flush=True)
        metrics = export_yolo_predictions(
            checkpoint,
            dataset,
            "test",
            predictions,
            size,
            "cuda",
            excluded_category_names=EXCLUDED,
            artifacts_dir=model_dir,
        )
        elapsed = time.perf_counter() - started
        evaluation_record = {
            "model": model_key,
            "label": label(run_dir),
            "family": spec.family,
            "scale": spec.scale,
            "augmentation": augmentation,
            "image_size": size,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "source_run_record": str((run_dir / "run_record.json").resolve()),
            "dataset_root": str(dataset.resolve()),
            "annotation_sha256": sha256_file(dataset / "test" / "_annotations.coco.json"),
            "excluded_categories": list(EXCLUDED),
            "evaluation_split": "test",
            "evaluation_seconds": elapsed,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        }
        (model_dir / "evaluation_record.json").write_text(
            json.dumps(evaluation_record, indent=2, default=str) + "\n", encoding="utf-8"
        )
        row = {
            "model": model_key,
            "label": label(run_dir),
            "family": spec.family,
            "scale": spec.scale,
            "augmentation": augmentation,
            "image_size": size,
            "run_dir": str(run_dir.resolve()),
            **{key: metrics.get(key) for key in SCALARS},
            "evaluation_seconds": elapsed,
            "predictions": str(predictions.resolve()),
        }
        rows.append(row)
        class_rows.extend({"model": model_key, "label": label(run_dir), **item} for item in metrics["per_class"])
        manifest_runs.append({"run_dir": str(run_dir.resolve()), "label": label(run_dir), "metrics": row})
        print(
            f"[{index}/{len(runs)}] {label(run_dir)}: "
            f"mAP50-95={metrics['map50_95']:.4f}, mAP50={metrics['map50']:.4f}, F1={metrics['f1']:.4f}",
            flush=True,
        )
    with (output_dir / "model_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)
    manifest = {
        "status": "completed",
        "model_count": len(rows),
        "excluded_categories": list(EXCLUDED),
        "evaluation_split": "test",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir.resolve()),
        "runs": manifest_runs,
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "completed", "count": len(rows), "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
