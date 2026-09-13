from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .model_registry import ModelSpec, _rfdetr_class
from .wandb_utils import attach_rfdetr_logger

RFDETR_RESOLUTIONS = {"n": 384, "s": 512, "m": 576, "large": 640}


def _resume_early_stopping_state(
    resume: str | Path,
    min_delta: float = 0.001,
) -> dict[str, Any]:
    """Reconstruct uninterrupted early-stopping state across resume segments."""
    resume_path = Path(resume)
    if not resume_path.is_file():
        return {"restored": False, "best_map": 0.0, "counter": 0, "completed_evaluations": 0}

    import torch

    run_dirs: list[Path] = []
    seen: set[Path] = set()
    checkpoint_path = resume_path.resolve()
    while checkpoint_path.is_file() and checkpoint_path not in seen:
        seen.add(checkpoint_path)
        run_dirs.append(checkpoint_path.parent)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_args = checkpoint.get("args")
        previous = getattr(checkpoint_args, "resume", "") if checkpoint_args is not None else ""
        del checkpoint
        if not previous:
            break
        checkpoint_path = Path(previous).resolve()

    best_map = 0.0
    counter = 0
    completed = 0
    for run_dir in reversed(run_dirs):
        log_path = run_dir / "log.txt"
        if not log_path.is_file():
            continue
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                candidates = []
                for key in ("test_coco_eval_bbox", "ema_test_coco_eval_bbox"):
                    metrics = row.get(key)
                    if isinstance(metrics, list) and metrics:
                        candidates.append(float(metrics[0]))
                if not candidates:
                    continue
                current_map = max(candidates)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            completed += 1
            if current_map > best_map + min_delta:
                best_map = current_map
                counter = 0
            else:
                counter += 1
    return {
        "restored": completed > 0,
        "best_map": best_map,
        "counter": counter,
        "completed_evaluations": completed,
        "history_run_dirs": [str(path) for path in reversed(run_dirs)],
        "resume_checkpoint": str(resume_path),
    }


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
          wandb_log_interval_steps: int = 100,
          defer_test_evaluation: bool = False) -> dict[str, Any]:
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
    overrides = training.get("per_model_overrides", {})
    model_override = overrides.get(spec.key, {}) if isinstance(overrides, dict) else {}
    resume = str(model_override.get("resume", ""))
    early_stopping_min_delta = float(model_override.get("early_stopping_min_delta", 0.001))
    resume_early_stopping = _resume_early_stopping_state(resume, early_stopping_min_delta)
    model = _rfdetr_class(spec)(pretrain_weights=str(checkpoint),
                                resolution=resolution,
                                gradient_checkpointing=True, device=device)
    attach_rfdetr_logger(model, wandb_run, wandb_log_interval_steps)
    if resume_early_stopping["restored"]:
        from rfdetr.util.early_stopping import EarlyStoppingCallback

        early_stopping = EarlyStoppingCallback(
            model=model.model,
            patience=int(training["patience"]),
            min_delta=early_stopping_min_delta,
            use_ema=False,
        )
        early_stopping.best_map = float(resume_early_stopping["best_map"])
        early_stopping.counter = int(resume_early_stopping["counter"])
        model.callbacks["on_fit_epoch_end"].append(early_stopping.update)
        print(
            "RF-DETR resume restored early stopping: "
            f"best_map={early_stopping.best_map:.6f}, "
            f"counter={early_stopping.counter}/{early_stopping.patience}, "
            f"completed_evaluations={resume_early_stopping['completed_evaluations']}"
        )
    started = time.perf_counter()
    args = dict(dataset_dir=str(dataset_dir), epochs=2 if smoke else int(training["epochs"]),
                batch_size=int(model_override.get("batch_size", training["effective_batch"])),
                grad_accum_steps=int(model_override.get("grad_accum_steps", 1)),
                resume=resume,
                lr=float(training["lr0"]), weight_decay=float(training["weight_decay"]),
                warmup_epochs=float(training["warmup_epochs"]), checkpoint_interval=10,
                multi_scale=bool(model_override.get("multi_scale", True)),
                expanded_scales=bool(model_override.get("expanded_scales", True)),
                do_random_resize_via_padding=False,
                square_resize_div_64=True,
                early_stopping=not resume_early_stopping["restored"],
                early_stopping_patience=int(training["patience"]),
                early_stopping_min_delta=early_stopping_min_delta,
                output_dir=str(run_dir), tensorboard=True,
                run_test=not defer_test_evaluation, wandb=False)
    result = model.train(**args)
    candidates = [run_dir / name for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth", "checkpoint.pth")]
    best = next((path for path in candidates if path.is_file()), None)
    if best is None: raise FileNotFoundError(f"RF-DETR completed without a recognised best checkpoint in {run_dir}")
    native_path = run_dir / "results.json"
    native_metrics = json.loads(native_path.read_text(encoding="utf-8")) if native_path.is_file() else {}
    return {"checkpoint_best": str(best), "training_seconds": time.perf_counter() - started,
            "native_metrics": {"test": native_metrics}, "native_metrics_path": str(native_path),
            "training_summary": summarize_training(run_dir, native_metrics), "train_args": args,
            "model_args": {"resolution": resolution, "gradient_checkpointing": True, "device": device},
            "resume_early_stopping_state": resume_early_stopping,
            "test_evaluation_deferred": defer_test_evaluation}
