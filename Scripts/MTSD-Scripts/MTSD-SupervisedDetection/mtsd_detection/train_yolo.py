from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .model_registry import ModelSpec
from .wandb_utils import attach_yolo_logger, summarize_yolo_results

NO_ONLINE_AUGMENTATION_ARGS = {"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "degrees": 0.0,
    "translate": 0.0, "scale": 0.0, "shear": 0.0, "perspective": 0.0, "flipud": 0.0,
    "fliplr": 0.0, "bgr": 0.0, "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0,
    "copy_paste": 0.0, "erasing": 0.0, "auto_augment": None}

YOLO_ONLINE_AUGMENTATION_PRESETS = {
    "disabled": NO_ONLINE_AUGMENTATION_ARGS,
    "traffic_metric_push": {
        "hsv_h": 0.015, "hsv_s": 0.45, "hsv_v": 0.35,
        "degrees": 2.0, "translate": 0.08, "scale": 0.35,
        "shear": 1.0, "perspective": 0.0005,
        "flipud": 0.0, "fliplr": 0.25, "bgr": 0.0,
        "mosaic": 0.60, "mixup": 0.05, "cutmix": 0.0,
        "copy_paste": 0.0, "erasing": 0.10, "auto_augment": None,
        "close_mosaic": 10,
    },
}


def yolo_online_augmentation_args(training: dict[str, Any]) -> dict[str, Any]:
    preset = str(training.get("yolo_online_augmentation", "disabled"))
    if preset not in YOLO_ONLINE_AUGMENTATION_PRESETS:
        choices = ", ".join(sorted(YOLO_ONLINE_AUGMENTATION_PRESETS))
        raise ValueError(f"Unknown YOLO online augmentation preset '{preset}'. Choose one of: {choices}")
    return dict(YOLO_ONLINE_AUGMENTATION_PRESETS[preset])


def yolo_batch_plan(training: dict[str, Any], image_size: int) -> dict[str, int]:
    configured = training.get("ultralytics_physical_batch_by_image_size", {})
    choices = sorted((int(size), int(batch)) for size, batch in configured.items())
    default_batch = int(training["effective_batch"])
    physical_batch = next((batch for size, batch in choices if image_size <= size),
                          choices[-1][1] if choices else default_batch)
    optimizer_batch = int(training.get("ultralytics_optimizer_batch", 64))
    if physical_batch <= 0 or optimizer_batch < physical_batch or optimizer_batch % physical_batch:
        raise ValueError(
            f"Invalid Ultralytics batch plan: physical={physical_batch}, optimizer={optimizer_batch}"
        )
    return {
        "physical_batch": physical_batch,
        "optimizer_effective_batch": optimizer_batch,
        "gradient_accumulation_steps": optimizer_batch // physical_batch,
    }


def train(spec: ModelSpec, checkpoint: Path, dataset_yaml: Path, run_dir: Path,
          training: dict[str, Any], smoke: bool = False, wandb_run: Any = None,
          wandb_log_interval_steps: int = 100) -> dict[str, Any]:
    from ultralytics import YOLO
    if run_dir.exists(): raise FileExistsError(f"Immutable run directory already exists: {run_dir}")
    model = YOLO(str(checkpoint))
    attach_yolo_logger(model, wandb_run, wandb_log_interval_steps)
    epochs = 2 if smoke else int(training["epochs"])
    image_size = 320 if smoke else int(training["image_size"])
    batch_plan = yolo_batch_plan(training, image_size)
    started = time.perf_counter()
    device = None if training.get("device", "auto") == "auto" else training.get("device")
    args = dict(data=str(dataset_yaml), epochs=epochs, imgsz=image_size,
                batch=batch_plan["physical_batch"], nbs=batch_plan["optimizer_effective_batch"],
                optimizer=training["optimizer"],
                lr0=float(training["lr0"]), lrf=float(training["lrf"]),
                weight_decay=float(training["weight_decay"]), warmup_epochs=float(training["warmup_epochs"]),
                patience=int(training["patience"]), seed=int(training["seed"]),
                deterministic=bool(training["deterministic"]), device=device, plots=True,
                project=str(run_dir.parent), name=run_dir.name, exist_ok=False,
                **yolo_online_augmentation_args(training))
    result = model.train(**args)
    training_seconds = time.perf_counter() - started
    best = run_dir / "weights" / "best.pt"
    if not best.is_file(): raise FileNotFoundError(f"Training completed without best checkpoint: {best}")
    best_model = YOLO(str(best))
    native = {}
    evaluation_started = time.perf_counter()
    for split in ("val", "test"):
        metrics = best_model.val(data=str(dataset_yaml), split=split, imgsz=image_size,
                                 batch=batch_plan["physical_batch"], device=device)
        native[split] = dict(getattr(metrics, "results_dict", {}) or {})
    try:
        training_summary = summarize_yolo_results(run_dir)
    except Exception as exc:
        print(f"YOLO results.csv summary skipped: {exc}")
        training_summary = {}
    return {"checkpoint_best": str(best), "training_seconds": training_seconds,
            "native_evaluation_seconds": time.perf_counter() - evaluation_started,
            "native_metrics": native, "training_summary": training_summary,
            "train_args": args, "batch_plan": batch_plan,
            "result_type": type(result).__name__}
