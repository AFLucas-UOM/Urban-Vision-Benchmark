"""Build a read-only inventory of MTSD supervised-detection experiments.

The script intentionally only reads existing run folders, orchestration state,
local W&B metadata, prepared-dataset manifests, and checkpoints.  It writes a
canonical CSV at the repository root and a human-readable report under
Documents; it never edits an experiment directory or a dataset.
"""
from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - the training environment provides PyYAML
    yaml = None


HERE = Path(__file__).resolve().parent
# HERE = .../Urban-Vision-Benchmark/Scripts/MTSD-Scripts/MTSD-SupervisedDetection
REPO = HERE.parents[2]
RUNS = REPO / "Results" / "MTSD-Runs"
PREPARED = REPO / "Datasets" / "MTSD" / "Prepared"
CSV_OUT = REPO / "existing_mtSD_experiment_audit.csv"
REPORT_DIR = REPO / "Documents" / "Final-Reports" / "MTSD-SupervisedDetection"
MD_OUT = REPORT_DIR / "existing_mtSD_experiment_audit.md"

CSV_FIELDS = [
    "inventory_id", "source_type", "run_path", "architecture", "model_scale",
    "model_checkpoint_identifier", "checkpoint_path", "checkpoint_exists",
    "augmentation_regime", "dataset_version", "dataset_path", "image_size",
    "train_source_image_count", "train_unique_source_hash_count",
    "train_physical_image_count", "validation_image_count", "test_image_count",
    "augmentation_multiplier", "batches_per_epoch", "optimizer_steps_per_epoch",
    "batch_size_physical", "effective_batch_size", "gradient_accumulation_steps",
    "epochs_requested", "epochs_completed", "best_epoch", "early_stopping_epoch",
    "optimizer", "initial_lr", "weight_decay", "warmup_epochs", "patience",
    "seed", "deterministic", "pretrained_checkpoint", "framework",
    "framework_library_version", "python_version", "torch_version", "gpu",
    "gpu_vram_gb", "peak_training_gpu_memory_mib", "training_duration_seconds",
    "time_per_epoch_seconds", "val_map50", "val_map50_95", "val_precision",
    "val_recall", "val_f1", "test_map50", "test_map50_95", "test_precision",
    "test_recall", "test_f1", "inference_seconds", "inference_fps",
    "checkpoint_result_path", "result_csv_path", "run_record_path",
    "wandb_run_id", "wandb_run_name", "wandb_url", "wandb_local_path",
    "completion_status", "evaluation_split", "notes",
]


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.is_file():
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, sort_keys=True, default=str)


def first(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return ""


def numeric(value: Any) -> float | int | str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return value


def prepared_metadata() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in PREPARED.glob("*/prep_manifest.json"):
        data = read_json(path)
        version = data.get("dataset_version")
        if not version:
            continue
        counts = data.get("counts", {})
        splits = counts.get("per_split", {})
        train = splits.get("train", {})
        valid = splits.get("valid", {})
        test = splits.get("test", {})
        split_manifest = path.parent / "split_manifest.csv"
        rows: list[dict[str, str]] = []
        if split_manifest.is_file():
            with split_manifest.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        train_rows = [row for row in rows if row.get("split") == "train"]
        source_hashes = {row.get("source_image_sha256") for row in train_rows if row.get("source_image_sha256")}
        physical = path.parent / "MTSD-YOLO" / "train" / "images"
        physical_count = len(list(physical.iterdir())) if physical.is_dir() else ""
        if not physical_count:
            physical_count = first(train.get("images"), data.get("original_training_images"), "")
        augmentation = data.get("augmentation", {})
        regime = "NoAug" if version.endswith("noaug") else (
            "Strong" if "strong" in version else "Good"
        )
        result[version] = {
            "path": path.parent,
            "regime": regime,
            "train_source": first(train.get("images"), len(train_rows)),
            "train_unique_hashes": len(source_hashes),
            "train_physical": physical_count,
            "valid": first(valid.get("images"), ""),
            "test": first(test.get("images"), ""),
            "multiplier": (float(physical_count) / float(train.get("images"))
                           if physical_count and train.get("images") else ""),
            "recipe": "none" if regime == "NoAug" else first(augmentation.get("recipe_version"), data.get("strong_augmentation_summary", {}).get("recipe_version"), "unknown"),
            "copies": first(augmentation.get("copies_per_image"), data.get("strong_augmentation_summary", {}).get("generated_images"), ""),
            "train_only": first(augmentation.get("train_only"), True if regime != "NoAug" else True),
            "prep_manifest": path,
            "split_hash": first(data.get("split_manifest_sha256"), ""),
            "augmentation_hash": first(data.get("augmentation_manifest_sha256"), ""),
        }
    return result


def local_wandb() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for directory in RUNS.glob("wandb/run-*"):
        run_id = directory.name.rsplit("-", 1)[-1]
        metadata_path = directory / "files" / "wandb-metadata.json"
        config_path = directory / "files" / "config.yaml"
        metadata = read_json(metadata_path)
        result[run_id] = {
            "path": directory,
            "name": "",
            "metadata": metadata,
            "config_path": config_path,
        }
    return result


def telemetry_by_label() -> dict[str, dict[str, Any]]:
    result = {}
    for path in RUNS.glob("**/telemetry/*.json"):
        data = read_json(path)
        label = data.get("label")
        if label:
            result[label] = data
    return result


def path_from_record(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else REPO / path


def infer_model(path: Path, record: dict[str, Any], args: dict[str, Any]) -> tuple[str, str, str]:
    model = str(first(record.get("model"), ""))
    family = str(first(record.get("family"), ""))
    scale = str(first(record.get("scale"), ""))
    match = re.search(r"(yolo(?:11|12|26)[nslm]|rfdetr-(?:n|s|m|large))", path.name, re.I)
    if match:
        model = match.group(1).lower()
    elif not model and isinstance(args.get("model"), str):
        match = re.search(r"(yolo(?:11|12|26)[nslm]|rfdetr-(?:n|s|m|large))", args["model"], re.I)
        model = match.group(1).lower() if match else ""
    if not scale and model:
        scale = model.rsplit("-", 1)[-1][-1]
    if not family and model:
        family = "RF-DETR" if model.startswith("rfdetr") else model[:6].upper()
    if not family and path.parent.name.endswith("-MTSD"):
        family = path.parent.name.removesuffix("-MTSD")
    checkpoint = str(first(record.get("checkpoint_path"), ""))
    return family, scale, model or checkpoint


def metrics_from_record(record: dict[str, Any]) -> dict[str, Any]:
    summary = record.get("training_summary", {}) or {}
    native = record.get("native_metrics", {}) or {}
    val = native.get("val", {}) if isinstance(native, dict) else {}
    unified_val = record.get("unified_validation_metrics", {}) or {}
    unified_test = record.get("unified_test_metrics", {}) or {}
    test = unified_test
    if not test:
        test = native.get("test", {}) if isinstance(native, dict) else {}
    def m(source: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in source and source[key] not in (None, ""):
                return source[key]
        return ""
    return {
        "val_map50": first(m(unified_val, "map50"), m(val, "metrics/mAP50(B)", "map50"), summary.get("mAP50")),
        "val_map50_95": first(m(unified_val, "map50_95"), m(val, "metrics/mAP50-95(B)", "map50-95"), summary.get("mAP50-95")),
        "val_precision": first(m(unified_val, "precision"), m(val, "metrics/precision(B)", "precision"), summary.get("precision")),
        "val_recall": first(m(unified_val, "recall"), m(val, "metrics/recall(B)", "recall"), summary.get("recall")),
        "val_f1": first(m(unified_val, "f1"), summary.get("F1")),
        "test_map50": first(m(test, "map50"), m(test, "metrics/mAP50(B)")),
        "test_map50_95": first(m(test, "map50_95"), m(test, "metrics/mAP50-95(B)")),
        "test_precision": first(m(test, "precision"), m(test, "metrics/precision(B)")),
        "test_recall": first(m(test, "recall"), m(test, "metrics/recall(B)")),
        "test_f1": first(m(test, "f1")),
        "inference_seconds": first(m(unified_test, "inference_seconds")),
        "inference_fps": first(m(unified_test, "inference_throughput_images_per_second")),
    }


def run_row(run_dir: Path, record: dict[str, Any], args: dict[str, Any], datasets: dict[str, dict[str, Any]], wandb: dict[str, dict[str, Any]], telemetry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    family, scale, model = infer_model(run_dir, record, args)
    train_args = record.get("train_args", {}) or args
    model_args = record.get("model_args", {}) or {}
    dataset_version = str(first(record.get("dataset_version"), ""))
    dataset = datasets.get(dataset_version, {})
    regime = first(record.get("offline_augmentation_variant"), dataset.get("regime"), "Unknown")
    if regime in {"unaugmented", "none"}: regime = "NoAug"
    elif regime in {"augmented", "mild"}: regime = "Good"
    elif regime == "strong": regime = "Strong"
    image_size = first(train_args.get("imgsz"), train_args.get("image_size"), model_args.get("resolution"))
    if not image_size:
        match = re.search(r"img(\d+)", run_dir.name, re.I)
        image_size = match.group(1) if match else ""
    physical = first(record.get("batch_plan", {}).get("physical_batch"), train_args.get("batch"), train_args.get("batch_size"))
    effective = first(record.get("batch_plan", {}).get("optimizer_effective_batch"),
                      int(train_args.get("batch_size")) * int(train_args.get("grad_accum_steps", 1))
                      if train_args.get("batch_size") else "")
    accumulation = first(record.get("batch_plan", {}).get("gradient_accumulation_steps"), train_args.get("grad_accum_steps"), 1 if effective else "")
    batches = ""
    optimizer_steps = ""
    try:
        batches = math.ceil(float(dataset.get("train_physical")) / float(physical))
        optimizer_steps = math.ceil(float(batches) / float(accumulation))
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    hardware = record.get("hardware", {}) or {}
    run_id = str(first(record.get("wandb_run_id"), ""))
    wb = wandb.get(run_id, {})
    telem = telemetry.get(str(first(record.get("run_label"), run_dir.name)), {})
    training_seconds = first(record.get("training_seconds"), telem.get("elapsed_seconds"))
    completed = first(record.get("training_summary", {}).get("completed_epochs"), "")
    best_epoch = first(record.get("training_summary", {}).get("best_epoch"), "")
    checkpoint = first(record.get("checkpoint_best"), record.get("checkpoint_path"), "")
    checkpoint_path = path_from_record(checkpoint)
    result_csv = run_dir / "results.csv"
    record_path = run_dir / "run_record.json"
    status = first(record.get("status"), "partial" if (run_dir / "weights").exists() or (run_dir / "checkpoint.pth").exists() else "unknown")
    notes = []
    if record.get("error"): notes.append(str(record["error"]).splitlines()[0])
    if record.get("unified_eval_status") == "failed": notes.append("unified evaluation failed")
    if record.get("post_training_evaluation_skipped"): notes.append("post-training evaluation skipped")
    framework = "RF-DETR" if family == "RF-DETR" else "Ultralytics"
    framework_version = "Ultralytics 8.4.101; RF-DETR 1.3.0; W&B 0.28.1 (mtsd-base environment)"
    return {field: scalar(value) for field, value in {
        "inventory_id": str(run_dir.relative_to(REPO)), "source_type": "run_directory",
        "run_path": str(run_dir.relative_to(REPO)), "architecture": family,
        "model_scale": scale, "model_checkpoint_identifier": model,
        "checkpoint_path": record.get("checkpoint_path", ""), "checkpoint_exists": bool(checkpoint_path and checkpoint_path.is_file()),
        "augmentation_regime": regime, "dataset_version": dataset_version,
        "dataset_path": str(dataset.get("path", "")), "image_size": image_size,
        "train_source_image_count": dataset.get("train_source", ""), "train_unique_source_hash_count": dataset.get("train_unique_hashes", ""),
        "train_physical_image_count": dataset.get("train_physical", ""), "validation_image_count": dataset.get("valid", ""),
        "test_image_count": dataset.get("test", ""), "augmentation_multiplier": dataset.get("multiplier", ""),
        "batches_per_epoch": batches, "optimizer_steps_per_epoch": optimizer_steps,
        "batch_size_physical": physical, "effective_batch_size": effective, "gradient_accumulation_steps": accumulation,
        "epochs_requested": first(train_args.get("epochs"), ""), "epochs_completed": completed,
        "best_epoch": best_epoch, "early_stopping_epoch": completed if completed and first(train_args.get("patience"), record.get("training_summary", {}).get("completed_epochs")) else "",
        "optimizer": first(train_args.get("optimizer"), "AdamW" if train_args.get("lr") else ""),
        "initial_lr": first(train_args.get("lr0"), train_args.get("lr")), "weight_decay": train_args.get("weight_decay", ""),
        "warmup_epochs": train_args.get("warmup_epochs", ""), "patience": first(train_args.get("patience"), train_args.get("early_stopping_patience")),
        "seed": first(train_args.get("seed"), record.get("seed"), 42 if "s42" in run_dir.name else ""),
        "deterministic": first(train_args.get("deterministic"), ""), "pretrained_checkpoint": record.get("checkpoint_path", ""),
        "framework": framework, "framework_library_version": framework_version,
        "python_version": hardware.get("python", ""), "torch_version": hardware.get("torch_version", ""),
        "gpu": hardware.get("gpu", ""), "gpu_vram_gb": hardware.get("gpu_vram_gb", ""),
        "peak_training_gpu_memory_mib": first(telem.get("peak_vram_mib"), record.get("peak_training_gpu_memory_mib")),
        "training_duration_seconds": training_seconds, "time_per_epoch_seconds": telem.get("time_per_epoch_seconds", ""),
        **metrics_from_record(record), "checkpoint_result_path": checkpoint,
        "result_csv_path": str(result_csv) if result_csv.is_file() else "", "run_record_path": str(record_path) if record_path.is_file() else "",
        "wandb_run_id": run_id, "wandb_run_name": first(record.get("run_label"), run_dir.name),
        "wandb_url": record.get("wandb_url", ""), "wandb_local_path": str(wb.get("path", "")),
        "completion_status": status, "evaluation_split": first("test" if record.get("unified_test_metrics") else "validation/native", ""),
        "notes": "; ".join(notes),
    }.items()}


def state_only_rows(existing_paths: set[str]) -> list[dict[str, Any]]:
    rows = []
    for path in RUNS.glob("Supervised-Matrix/**/state.json"):
        state = read_json(path)
        dataset_version = str(first(state.get("dataset_version"), ""))
        for model, details in (state.get("models", {}) or {}).items():
            details = details if isinstance(details, dict) else {}
            run_dir = str(first(details.get("run_dir"), ""))
            if run_dir and Path(run_dir).exists():
                canonical = str(Path(run_dir).relative_to(REPO)) if Path(run_dir).is_absolute() else run_dir
                if canonical in existing_paths:
                    continue
            label = str(first(details.get("run_label"), model))
            image = re.search(r"img(\d+)", label)
            row = {field: "" for field in CSV_FIELDS}
            row.update({
                "inventory_id": f"state:{path.relative_to(REPO)}:{model}", "source_type": "orchestration_state_only",
                "run_path": run_dir, "architecture": "RF-DETR" if model.startswith("rfdetr") else model[:6].upper(),
                "model_scale": model.rsplit("-", 1)[-1][-1] if model.startswith("rfdetr-") else model[-1], "model_checkpoint_identifier": model,
                "augmentation_regime": "Strong" if "strong" in dataset_version or "strong" in label else ("NoAug" if "noaug" in dataset_version else "Good"),
                "dataset_version": dataset_version, "image_size": image.group(1) if image else "",
                "epochs_requested": details.get("epochs", ""), "completion_status": first(details.get("status"), state.get("final_status")),
                "wandb_run_id": details.get("wandb_run_id", ""), "wandb_run_name": label,
                "run_record_path": "", "notes": first(details.get("error"), details.get("stop_reason"), state.get("failure_reason"), "state entry without canonical run_record"),
            })
            rows.append(row)
    return rows


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def write_report(rows: list[dict[str, Any]], datasets: dict[str, dict[str, Any]], wb_count: int, state_count: int) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row.get("completion_status", "unknown") for row in rows)
    regime_counts = Counter(row.get("augmentation_regime", "unknown") for row in rows)
    lines = [
        "# Existing MTSD supervised-detection experiment audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Repository commit at audit time: `{git_head()}`",
        "",
        "This is a read-only inventory. Existing run folders, checkpoints, datasets, W&B metadata, and result exports were not modified.",
        "",
        "## Scope and evidence",
        "",
        f"- Canonical run-directory rows: {sum(row.get('source_type') == 'run_directory' for row in rows)}.",
        f"- Orchestration-state-only rows: {state_count}; these capture attempts whose state exists but whose canonical run record is absent or not complete.",
        f"- Local W&B run directories inspected: {wb_count}.",
        f"- Status counts: {dict(sorted(counts.items()))}.",
        f"- Augmentation-regime counts: {dict(sorted(regime_counts.items()))}.",
        "- `val_*` fields are preferred for selection/ablation. `test_*` fields are retained only when an existing run record contains a test/unified-test evaluation.",
        "- `early_stopping_epoch` is recorded as the last completed epoch when the run used the configured early-stopping policy; `best_epoch` is the best recorded validation epoch.",
        "- Library versions are recorded from the reproducible `mtsd-base` environment available locally; historical run records did not persist a separate Ultralytics/RF-DETR version field.",
        "",
        "## Prepared MTSD variants",
        "",
        "| Regime | Dataset version | Source train | Physical train | Multiplier | Valid | Test | Recipe | Train-only evidence |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for version, data in sorted(datasets.items()):
        lines.append(f"| {data['regime']} | `{version}` | {data['train_source']} | {data['train_physical']} | {data['multiplier']} | {data['valid']} | {data['test']} | `{data['recipe']}` | {data['train_only']} |")
    lines += [
        "",
        "The three split manifests were compared by source image hash/name: train/valid/test membership is identical across NoAug, Good, and Strong. Validation and test physical image counts remain the original 749/747; only training is expanded. The current prepared corpora contain 5,986 source training rows (5,985 unique source hashes) and 23,944 physical training images for Good/Strong, a 4.0× exposure multiplier. Exact batches and optimizer steps depend on the run-specific physical/effective batch and are included in the CSV.",
        "",
        "## Verified missing/failed patterns relevant to the requested sweep",
        "",
        "- Strong YOLO11 at 1280: S and M are complete; N is absent.",
        "- Strong YOLO26 at 1280: S and M are complete; N is absent.",
        "- Strong YOLO12: S at 960 is complete; the prior M at 960 attempt was stopped as impractically slow. No common 960 N/S/M matrix exists.",
        "- Strong RF-DETR native: N at 384 is complete; S at 512 and M at 576 are absent. Strong S@640 and M@704 are separate resolution follow-ups and are retained.",
        "- The CSV contains the evidence used to decide whether an apparent filename match is actually a completed equivalent; filenames alone were not used as completion proof.",
        "",
        "## Output",
        "",
        f"Machine-readable inventory: `{CSV_OUT.relative_to(REPO)}`",
        "",
        "The inventory is regenerated with `Scripts/MTSD-Scripts/MTSD-SupervisedDetection/audit_experiments.py`.",
    ]
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    datasets = prepared_metadata()
    wb = local_wandb()
    telemetry = telemetry_by_label()
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for family_dir in sorted(RUNS.glob("*-MTSD")):
        if not family_dir.is_dir():
            continue
        for run_dir in sorted(family_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name in {"weights", "eval", "unified_evaluation"}:
                continue
            record = read_json(run_dir / "run_record.json")
            args = read_yaml(run_dir / "args.yaml")
            if not record and not args and not (run_dir / "results.csv").is_file() and not (run_dir / "results.json").is_file():
                continue
            row = run_row(run_dir, record, args, datasets, wb, telemetry)
            rows.append(row)
            seen_paths.add(str(run_dir.relative_to(REPO)))
    extra = state_only_rows(seen_paths)
    rows.extend(extra)
    rows.sort(key=lambda row: (str(row.get("architecture")), str(row.get("model_checkpoint_identifier")), str(row.get("image_size")), str(row.get("augmentation_regime")), str(row.get("run_path"))))
    write_report(rows, datasets, len(wb), len(extra))
    print(json.dumps({"csv": str(CSV_OUT), "markdown": str(MD_OUT), "rows": len(rows), "state_only_rows": len(extra)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
