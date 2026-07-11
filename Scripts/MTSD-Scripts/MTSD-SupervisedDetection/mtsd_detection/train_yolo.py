from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .model_registry import ModelSpec

NO_ONLINE_AUGMENTATION_ARGS = {"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "degrees": 0.0,
    "translate": 0.0, "scale": 0.0, "shear": 0.0, "perspective": 0.0, "flipud": 0.0,
    "fliplr": 0.0, "bgr": 0.0, "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0,
    "copy_paste": 0.0, "erasing": 0.0, "auto_augment": None}


def train(spec: ModelSpec, checkpoint: Path, dataset_yaml: Path, run_dir: Path,
          training: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    from ultralytics import YOLO
    if run_dir.exists(): raise FileExistsError(f"Immutable run directory already exists: {run_dir}")
    model = YOLO(str(checkpoint))
    epochs = 2 if smoke else int(training["epochs"])
    image_size = 320 if smoke else int(training["image_size"])
    started = time.perf_counter()
    device = None if training.get("device", "auto") == "auto" else training.get("device")
    args = dict(data=str(dataset_yaml), epochs=epochs, imgsz=image_size,
                batch=int(training["effective_batch"]), optimizer=training["optimizer"],
                lr0=float(training["lr0"]), lrf=float(training["lrf"]),
                weight_decay=float(training["weight_decay"]), warmup_epochs=float(training["warmup_epochs"]),
                patience=int(training["patience"]), seed=int(training["seed"]),
                deterministic=bool(training["deterministic"]), device=device,
                project=str(run_dir.parent), name=run_dir.name, exist_ok=False, **NO_ONLINE_AUGMENTATION_ARGS)
    result = model.train(**args)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file(): raise FileNotFoundError(f"Training completed without best checkpoint: {best}")
    best_model = YOLO(str(best))
    native = {}
    for split in ("val", "test"):
        metrics = best_model.val(data=str(dataset_yaml), split=split, imgsz=image_size,
                                 batch=int(training["effective_batch"]), device=device)
        native[split] = dict(getattr(metrics, "results_dict", {}) or {})
    return {"checkpoint_best": str(best), "training_seconds": time.perf_counter() - started,
            "native_metrics": native, "train_args": args, "result_type": type(result).__name__}
