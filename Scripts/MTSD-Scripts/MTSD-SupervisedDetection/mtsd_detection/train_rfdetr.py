from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .model_registry import ModelSpec, _rfdetr_class


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
          training: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
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
    model = _rfdetr_class(spec)(pretrain_weights=str(checkpoint),
                                resolution=320 if smoke else int(training["image_size"]),
                                gradient_checkpointing=True, device=device)
    started = time.perf_counter()
    args = dict(dataset_dir=str(dataset_dir), epochs=2 if smoke else int(training["epochs"]),
                batch_size=int(training["effective_batch"]), grad_accum_steps=1,
                lr=float(training["lr0"]), weight_decay=float(training["weight_decay"]),
                warmup_epochs=float(training["warmup_epochs"]), checkpoint_interval=10,
                multi_scale=True, expanded_scales=True, do_random_resize_via_padding=False,
                square_resize_div_64=True,
                early_stopping=True, early_stopping_patience=int(training["patience"]),
                output_dir=str(run_dir), wandb=False)
    result = model.train(**args)
    candidates = [run_dir / name for name in ("checkpoint_best_ema.pth", "checkpoint_best_total.pth", "checkpoint_best_regular.pth", "checkpoint.pth")]
    best = next((path for path in candidates if path.is_file()), None)
    if best is None: raise FileNotFoundError(f"RF-DETR completed without a recognised best checkpoint in {run_dir}")
    native_path = run_dir / "results.json"
    native_metrics = json.loads(native_path.read_text(encoding="utf-8")) if native_path.is_file() else {}
    return {"checkpoint_best": str(best), "training_seconds": time.perf_counter() - started,
            "native_metrics": {"test": native_metrics}, "native_metrics_path": str(native_path),
            "train_args": args}
