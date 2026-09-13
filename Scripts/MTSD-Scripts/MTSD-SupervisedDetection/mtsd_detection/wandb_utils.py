from __future__ import annotations

import csv
import math
import os
from datetime import timedelta
from pathlib import Path
from typing import Any


def _number(value: Any) -> float | None:
    try:
        if hasattr(value, "detach"):
            value = value.detach().mean()
        if hasattr(value, "item"):
            value = value.item()
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _flatten_numeric(prefix: str, value: Any, output: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _flatten_numeric(f"{prefix}/{key}" if prefix else str(key), nested, output)
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _flatten_numeric(f"{prefix}/{index}", nested, output)
    else:
        numeric = _number(value)
        if numeric is not None:
            output[prefix] = numeric


def _safe_log(run: Any, payload: dict[str, Any], message: str) -> None:
    if run is None or not payload:
        return
    try:
        run.log(payload)
    except Exception as exc:
        print(f"W&B {message} skipped: {exc}")


def _guarded(callback, message: str):
    def wrapped(value):
        try:
            callback(value)
        except Exception as exc:
            print(f"W&B {message} skipped: {exc}")
    return wrapped


def _configure_metrics(run: Any) -> None:
    try:
        run.define_metric("epoch")
        run.define_metric("trainer/global_step")
        for pattern in (
            "train/*", "val/*", "metrics/*", "lr/*", "rfdetr/*",
            "native/*", "test/*", "final/*", "runtime/*",
        ):
            run.define_metric(pattern, step_metric="epoch")
        run.define_metric("lr", step_metric="epoch")
    except Exception as exc:
        print(f"W&B metric-axis configuration skipped: {exc}")


def start_run(config: dict[str, Any], name: str, group: str, mode: str, tags: list[str]):
    if mode == "disabled":
        return None
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(config["repo_root"]) / ".env", override=False)
    except ImportError:
        pass
    try:
        import wandb
    except ImportError:
        print("W&B is not installed; continuing without experiment tracking.")
        return None

    # RF-DETR emits high-volume progress/evaluation prints.  On Windows,
    # W&B's console-capture pipe can raise OSError(22) when the parent
    # terminal/session is detached.  Keep metric, artifact, and metadata
    # logging enabled while disabling only terminal capture.  Pass this as
    # an explicit setting: W&B 0.28.1 did not honour WANDB_CONSOLE=off for
    # these runs and silently selected console="wrap" instead.
    wandb_settings = wandb.Settings(console="off")

    entity = os.getenv(config["wandb"]["entity_env"])
    runs_root = Path(config["outputs"]["runs_root"])
    runs_root.mkdir(parents=True, exist_ok=True)
    init_args = dict(
        project=config["wandb"]["project"], entity=entity, name=name,
        group=group, tags=tags, config=config, job_type="train",
        resume="never", dir=str(runs_root), reinit=True,
        settings=wandb_settings,
    )
    try:
        run = wandb.init(**init_args, mode=mode)
    except Exception as exc:
        if mode != "online":
            print(f"W&B initialization failed; continuing without tracking: {exc}")
            return None
        print(f"W&B online initialization failed ({exc}); retrying in offline mode.")
        try:
            run = wandb.init(**init_args, mode="offline")
        except Exception as offline_exc:
            print(f"W&B offline initialization also failed: {offline_exc}")
            return None
    _configure_metrics(run)
    return run


def _read_last_csv_row(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    output = {}
    for key, value in rows[-1].items():
        numeric = _number(value)
        if numeric is not None:
            output[str(key).strip()] = numeric
    return output


def summarize_yolo_results(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "results.csv"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    clean_rows = []
    for row in rows:
        clean = {}
        for key, value in row.items():
            numeric = _number(value)
            if numeric is not None:
                clean[str(key).strip()] = numeric
        clean_rows.append(clean)
    if not clean_rows:
        return {}
    map_keys = ("metrics/mAP50-95(B)", "metrics/mAP50-95")
    map_key = next((key for key in map_keys if any(key in row for row in clean_rows)), None)
    best = max(clean_rows, key=lambda row: row.get(map_key, -math.inf)) if map_key else clean_rows[-1]
    final = clean_rows[-1]

    def pick(row: dict[str, float], *keys: str) -> float | None:
        return next((row[key] for key in keys if key in row), None)

    summary = {
        "precision": pick(best, "metrics/precision(B)", "metrics/precision"),
        "recall": pick(best, "metrics/recall(B)", "metrics/recall"),
        "mAP50": pick(best, "metrics/mAP50(B)", "metrics/mAP50"),
        "mAP50-95": pick(best, *map_keys),
        "best_epoch": int(best.get("epoch", clean_rows.index(best))) + 1,
        "completed_epochs": len(clean_rows),
        "train_box_loss": pick(final, "train/box_loss"),
        "train_cls_loss": pick(final, "train/cls_loss"),
        "train_dfl_loss": pick(final, "train/dfl_loss"),
        "val_box_loss": pick(best, "val/box_loss"),
        "val_cls_loss": pick(best, "val/cls_loss"),
        "val_dfl_loss": pick(best, "val/dfl_loss"),
        "lr_pg0": pick(final, "lr/pg0"),
        "lr_pg1": pick(final, "lr/pg1"),
        "lr_pg2": pick(final, "lr/pg2"),
        "train_elapsed_seconds": pick(final, "time"),
    }
    precision, recall = summary["precision"], summary["recall"]
    summary["F1"] = (2 * precision * recall / (precision + recall)
                     if precision is not None and recall is not None and precision + recall > 0 else None)
    return summary


def attach_yolo_logger(model: Any, run: Any, log_interval_steps: int = 100) -> bool:
    if run is None or not hasattr(model, "add_callback"):
        return False
    interval = max(1, int(log_interval_steps))
    state = {"global_step": 0, "elapsed": 0.0}

    def add_losses(payload: dict[str, Any], trainer: Any) -> None:
        values = getattr(trainer, "loss_items", None)
        if values is None:
            values = getattr(trainer, "tloss", None)
        if values is None:
            return
        try:
            values = list(values)
        except TypeError:
            values = [values]
        total = 0.0
        count = 0
        for name, value in zip(("box_loss", "cls_loss", "dfl_loss"), values):
            numeric = _number(value)
            if numeric is not None:
                payload[f"train/{name}"] = numeric
                total += numeric
                count += 1
        if count:
            payload["train/loss"] = total

    def add_lr(payload: dict[str, Any], trainer: Any) -> None:
        optimizer = getattr(trainer, "optimizer", None)
        groups = getattr(optimizer, "param_groups", []) if optimizer is not None else []
        for index, group in enumerate(groups):
            numeric = _number(group.get("lr"))
            if numeric is not None:
                payload[f"lr/pg{index}"] = numeric
        if "lr/pg0" in payload:
            payload["lr"] = payload["lr/pg0"]

    def base(trainer: Any) -> dict[str, Any]:
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        return {"trainer/global_step": state["global_step"], "epoch": epoch}

    def log_batch(trainer: Any) -> None:
        state["global_step"] += 1
        if state["global_step"] % interval:
            return
        payload = base(trainer)
        add_losses(payload, trainer)
        add_lr(payload, trainer)
        _safe_log(run, payload, "YOLO step logging")

    def log_epoch(trainer: Any) -> None:
        payload = base(trainer)
        metrics = getattr(trainer, "metrics", {}) or {}
        if isinstance(metrics, dict):
            _flatten_numeric("", metrics, payload)
        row = _read_last_csv_row(Path(getattr(trainer, "save_dir", "")) / "results.csv")
        payload.update(row)
        aliases = {
            "metrics/precision": ("metrics/precision(B)", "metrics/precision"),
            "metrics/recall": ("metrics/recall(B)", "metrics/recall"),
            "metrics/mAP50": ("metrics/mAP50(B)", "metrics/mAP50"),
            "metrics/mAP50-95": ("metrics/mAP50-95(B)", "metrics/mAP50-95"),
        }
        for target, candidates in aliases.items():
            value = next((payload[key] for key in candidates if key in payload), None)
            if value is not None:
                payload[target] = value
        precision, recall = payload.get("metrics/precision"), payload.get("metrics/recall")
        if precision is not None and recall is not None and precision + recall > 0:
            payload["metrics/F1"] = 2 * precision * recall / (precision + recall)
        elapsed = payload.get("time")
        if elapsed is not None:
            payload["runtime/train_elapsed_seconds"] = elapsed
            payload["runtime/epoch_seconds"] = max(0.0, elapsed - state["elapsed"])
            state["elapsed"] = elapsed
        add_losses(payload, trainer)
        add_lr(payload, trainer)
        _safe_log(run, payload, "YOLO epoch logging")

    model.add_callback("on_train_batch_end", _guarded(log_batch, "YOLO step logging"))
    model.add_callback("on_fit_epoch_end", _guarded(log_epoch, "YOLO epoch logging"))
    return True


def _duration_seconds(value: Any) -> float | None:
    if not isinstance(value, str):
        return _number(value)
    try:
        parts = value.split(":")
        if len(parts) != 3:
            return None
        return timedelta(hours=float(parts[0]), minutes=float(parts[1]), seconds=float(parts[2])).total_seconds()
    except (TypeError, ValueError):
        return None


def attach_rfdetr_logger(model: Any, run: Any, log_interval_steps: int = 100) -> bool:
    callbacks = getattr(model, "callbacks", None)
    if run is None or callbacks is None:
        return False
    interval = max(1, int(log_interval_steps))

    def log_batch(values: dict[str, Any]) -> None:
        step = int(values.get("step", 0)) + 1
        if step % interval == 0:
            _safe_log(run, {"trainer/global_step": step, "epoch": int(values.get("epoch", 0)) + 1},
                      "RF-DETR step logging")

    def log_epoch(values: dict[str, Any]) -> None:
        epoch = int(values.get("epoch", 0)) + 1
        payload: dict[str, Any] = {"epoch": epoch}
        for key, value in values.items():
            numeric = _number(value)
            if numeric is not None:
                payload[f"rfdetr/{key}"] = numeric
        aliases = {
            "train/loss": "train_loss",
            "train/cls_loss": "train_loss_ce",
            "train/box_loss": "train_loss_bbox",
            "train/giou_loss": "train_loss_giou",
            "train/class_error": "train_class_error",
            "val/loss": "test_loss",
            "lr": "train_lr",
            "lr/pg0": "train_lr",
        }
        for target, source in aliases.items():
            numeric = _number(values.get(source))
            if numeric is not None:
                payload[target] = numeric

        base_coco = values.get("test_coco_eval_bbox")
        ema_coco = values.get("ema_test_coco_eval_bbox")
        for prefix, metrics in (("rfdetr/base", base_coco), ("rfdetr/ema", ema_coco)):
            if isinstance(metrics, (list, tuple)):
                for name, index in (("mAP50-95", 0), ("mAP50", 1), ("mAP75", 2), ("AR100", 8)):
                    if len(metrics) > index and _number(metrics[index]) is not None:
                        payload[f"{prefix}/{name}"] = _number(metrics[index])

        coco_options = [metrics for metrics in (base_coco, ema_coco)
                        if isinstance(metrics, (list, tuple)) and metrics]
        chosen_coco = max(coco_options, key=lambda metrics: _number(metrics[0]) or -math.inf) if coco_options else None
        if isinstance(chosen_coco, (list, tuple)):
            for name, index in (("mAP50-95", 0), ("mAP50", 1), ("mAP75", 2), ("AR100", 8)):
                if len(chosen_coco) > index and _number(chosen_coco[index]) is not None:
                    payload[f"metrics/{name}"] = _number(chosen_coco[index])

        chose_ema = chosen_coco is ema_coco
        results = (values.get("ema_test_results_json") if chose_ema else values.get("test_results_json")) or {}
        if isinstance(results, dict):
            precision = _number(results.get("precision"))
            recall = _number(results.get("recall"))
            if precision is not None:
                payload["metrics/precision"] = precision
            if recall is not None:
                payload["metrics/recall"] = recall
            if precision is not None and recall is not None and precision + recall > 0:
                payload["metrics/F1"] = 2 * precision * recall / (precision + recall)

        for source, target in (("train_epoch_time", "runtime/train_epoch_seconds"),
                               ("epoch_time", "runtime/epoch_seconds")):
            seconds = _duration_seconds(values.get(source))
            if seconds is not None:
                payload[target] = seconds
        _safe_log(run, payload, "RF-DETR epoch logging")

    callbacks["on_train_batch_start"].append(_guarded(log_batch, "RF-DETR step logging"))
    callbacks["on_fit_epoch_end"].append(_guarded(log_epoch, "RF-DETR epoch logging"))
    return True


def log_dataset_artifact(run: Any, dataset_root: Path, record: dict[str, Any]) -> None:
    if run is None:
        return
    try:
        import wandb
        name = str(record.get("dataset_version", "mtsd-dataset")).replace("/", "-")
        metadata = {
            "dataset_version": record.get("dataset_version"),
            "dataset_variant": record.get("dataset_variant"),
            "annotation_source_mode": record.get("annotation_source_mode"),
            "included_groups": record.get("included_groups"),
            "prep_manifest_sha256": record.get("prep_manifest_sha256"),
            "split_manifest_sha256": record.get("split_manifest_sha256"),
            "offline_augmentation_variant": record.get("offline_augmentation_variant"),
        }
        artifact = wandb.Artifact(name=name, type="dataset", metadata=metadata)
        candidates = [
            dataset_root / "prep_manifest.json",
            dataset_root / "split_manifest.csv",
            dataset_root / "augmentation_manifest.csv",
            dataset_root / "MTSD-YOLO" / "data.yaml",
        ]
        candidates.extend(dataset_root / "MTSD-COCO" / split / "_annotations.coco.json"
                          for split in ("train", "valid", "test"))
        for path in candidates:
            if path.is_file():
                artifact.add_file(str(path), name=str(path.relative_to(dataset_root)).replace("\\", "/"))
        run.log_artifact(artifact)
    except Exception as exc:
        print(f"W&B dataset artifact logging skipped: {exc}")


def _final_metrics(record: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    summary = record.get("training_summary", {}) or {}
    _flatten_numeric("final", summary, payload)
    native = record.get("native_metrics", {}) or {}
    _flatten_numeric("native", native, payload)
    unified = record.get("unified_test_metrics", {}) or {}
    _flatten_numeric("test", unified, payload)
    aliases = {
        "test/mAP50-95": unified.get("map50_95"),
        "test/mAP50": unified.get("map50"),
        "test/mAP75": unified.get("map75"),
        "test/AR100": unified.get("ar100"),
        "test/precision": unified.get("precision"),
        "test/recall": unified.get("recall"),
        "test/F1": unified.get("f1"),
        "runtime/train_seconds": record.get("training_seconds"),
        "runtime/native_evaluation_seconds": record.get("native_evaluation_seconds"),
        "runtime/unified_evaluation_seconds": record.get("unified_evaluation_seconds"),
    }
    for key, value in aliases.items():
        numeric = _number(value)
        if numeric is not None:
            payload[key] = numeric
    epoch = summary.get("completed_epochs") or record.get("train_args", {}).get("epochs")
    if _number(epoch) is not None:
        payload["epoch"] = int(float(epoch))
    return payload


def _log_run_artifacts(run: Any, record: dict[str, Any]) -> None:
    run_dir = Path(record.get("run_dir", ""))
    if not run_dir.is_dir():
        return
    try:
        import wandb
        files = wandb.Artifact(f"{run_dir.name}-run-files", type="run-output",
                               metadata={"model": record.get("model"), "status": record.get("status")})
        added_files = False
        for pattern in ("*.csv", "*.json", "*.yaml", "*.txt", "*.log", "*.png", "*.jpg"):
            for path in sorted(run_dir.glob(pattern)):
                files.add_file(str(path), name=path.name)
                added_files = True
        if added_files:
            run.log_artifact(files)

        checkpoints = []
        for path in (
            run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt",
            run_dir / "checkpoint_best_total.pth", run_dir / "checkpoint_best_ema.pth",
            run_dir / "checkpoint_best_regular.pth", run_dir / "checkpoint.pth",
        ):
            if path.is_file():
                checkpoints.append(path)
        if checkpoints:
            model = wandb.Artifact(f"{run_dir.name}-weights", type="model", metadata={
                "model": record.get("model"), "checkpoint_source_sha256": record.get("checkpoint_sha256"),
                "test_map50_95": record.get("unified_test_metrics", {}).get("map50_95"),
            })
            for path in checkpoints:
                model.add_file(str(path), name=path.name)
            run.log_artifact(model)
    except Exception as exc:
        print(f"W&B run artifact logging skipped: {exc}")


def finish_run(run: Any, record: dict[str, Any], exit_code: int = 0) -> None:
    if run is None:
        return
    try:
        payload = _final_metrics(record)
        _safe_log(run, payload, "final metric logging")
        summary = {key: value for key, value in payload.items() if key != "epoch"}
        summary.update({
            "status": record.get("status"),
            "checkpoint_path": record.get("checkpoint_best") or record.get("checkpoint_path"),
            "checkpoint_source_sha256": record.get("checkpoint_sha256"),
            "unified_eval_status": record.get("unified_eval_status"),
            "error": record.get("error"),
        })
        run.summary.update({key: value for key, value in summary.items() if value is not None})
        _log_run_artifacts(run, record)
    except Exception as exc:
        print(f"W&B finalization metadata skipped: {exc}")
    finally:
        try:
            run.finish(exit_code=exit_code)
        except Exception as exc:
            print(f"W&B run finish failed: {exc}")
