from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .model_registry import ModelSpec, _rfdetr_class
from .wandb_utils import attach_rfdetr_logger

RFDETR_RESOLUTIONS = {"n": 384, "s": 512, "m": 576, "large": 640}


def rfdetr_resolution(spec: ModelSpec, training: dict[str, Any]) -> int:
    overrides = training.get("per_model_overrides", {})
    model_override = overrides.get(spec.key, {}) if isinstance(overrides, dict) else {}
    configured = model_override.get("resolution", model_override.get("image_size"))
    if configured is not None:
        return int(configured)
    return RFDETR_RESOLUTIONS[spec.scale]


def summarize_training(run_dir: Path, native_metrics: dict[str, Any]) -> dict[str, Any]:
    log_path = run_dir / "log.txt"
    stats = {}
    if log_path.is_file():
        rows = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows:
            try:
                stats = json.loads(rows[-1])
            except json.JSONDecodeError as exc:
                print(f"RF-DETR log.txt summary skipped: {exc}")
    coco_options = [metrics for metrics in (stats.get("test_coco_eval_bbox"),
                                             stats.get("ema_test_coco_eval_bbox"))
                    if isinstance(metrics, list) and metrics]
    coco = max(coco_options, key=lambda metrics: metrics[0]) if coco_options else []
    precision = native_metrics.get("precision")
    recall = native_metrics.get("recall")
    summary = {
        "precision": precision,
        "recall": recall,
        "F1": (2 * precision * recall / (precision + recall)
               if isinstance(precision, (int, float)) and isinstance(recall, (int, float))
               and precision + recall > 0 else None),
        "mAP50-95": coco[0] if len(coco) > 0 else None,
        "mAP50": coco[1] if len(coco) > 1 else native_metrics.get("map"),
        "mAP75": coco[2] if len(coco) > 2 else None,
        "AR100": coco[8] if len(coco) > 8 else None,
        "completed_epochs": int(stats["epoch"]) + 1 if "epoch" in stats else None,
        "train_loss": stats.get("train_loss"),
        "train_cls_loss": stats.get("train_loss_ce"),
        "train_box_loss": stats.get("train_loss_bbox"),
        "train_giou_loss": stats.get("train_loss_giou"),
        "val_loss": stats.get("test_loss"),
        "lr_pg0": stats.get("train_lr"),
    }
    return summary


def normalize_coco_categories(dataset_dir: Path) -> list[str]:
    changed = []
    for split in ("train", "valid", "test"):
        path = dataset_dir / split / "_annotations.coco.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        categories = sorted(payload["categories"], key=lambda row: row["id"])
        mapping = {row["id"]: index for index, row in enumerate(categories)}
        dirty = False
        for row in categories:
            old = row["id"]; row["id"] = mapping[old]
            if not row.get("supercategory"): row["supercategory"] = "mtsd"; dirty = True
            dirty |= old != row["id"]
        for annotation in payload["annotations"]:
            old = annotation["category_id"]; annotation["category_id"] = mapping[old]; dirty |= old != annotation["category_id"]
        if dirty:
            backup = path.with_suffix(path.suffix + ".bak")
            if not backup.exists(): shutil.copy2(path, backup)
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"); changed.append(str(path))
    return changed


def train(spec: ModelSpec, checkpoint: Path, dataset_dir: Path, run_dir: Path,
          training: dict[str, Any], smoke: bool = False, wandb_run: Any = None,
          wandb_log_interval_steps: int = 100) -> dict[str, Any]:
    if run_dir.exists(): raise FileExistsError(f"Immutable run directory already exists: {run_dir}")
    normalize_coco_categories(dataset_dir)
    run_dir.mkdir(parents=True)
    device = training.get("device", "auto")
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    resolution = rfdetr_resolution(spec, training)
    model = _rfdetr_class(spec)(pretrain_weights=str(checkpoint),
                                resolution=resolution,
                                gradient_checkpointing=True, device=device)
    attach_rfdetr_logger(model, wandb_run, wandb_log_interval_steps)
    started = time.perf_counter()
    args = dict(dataset_dir=str(dataset_dir), epochs=2 if smoke else int(training["epochs"]),
                batch_size=int(training["effective_batch"]), grad_accum_steps=1,
                lr=float(training["lr0"]), weight_decay=float(training["weight_decay"]),
                warmup_epochs=float(training["warmup_epochs"]), checkpoint_interval=10,
                multi_scale=True, expanded_scales=True, do_random_resize_via_padding=False,
                square_resize_div_64=True,
                early_stopping=True, early_stopping_patience=int(training["patience"]),
                output_dir=str(run_dir), tensorboard=True, run_test=True, wandb=False)
    result = model.train(**args)
    candidates = [run_dir / name for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth", "checkpoint.pth")]
    best = next((path for path in candidates if path.is_file()), None)
    if best is None: raise FileNotFoundError(f"RF-DETR completed without a recognised best checkpoint in {run_dir}")
    native_path = run_dir / "results.json"
    native_metrics = json.loads(native_path.read_text(encoding="utf-8")) if native_path.is_file() else {}
    return {"checkpoint_best": str(best), "training_seconds": time.perf_counter() - started,
            "native_metrics": {"test": native_metrics}, "native_metrics_path": str(native_path),
            "training_summary": summarize_training(run_dir, native_metrics), "train_args": args,
            "model_args": {"resolution": resolution, "gradient_checkpointing": True, "device": device}}
