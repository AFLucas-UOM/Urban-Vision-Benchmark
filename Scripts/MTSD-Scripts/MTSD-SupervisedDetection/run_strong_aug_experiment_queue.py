#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mtsd_detection.config import load_config
from mtsd_detection.manifests import sha256_file
from mtsd_detection.model_registry import resolve_checkpoint_info, validate_model_keys
from mtsd_detection.train_yolo import yolo_batch_plan, yolo_online_augmentation_args

DEFAULT_QUEUE_ROOT = Path(
    "Results/MTSD-Runs/Strong-Augmentation-Controlled/"
    "20260724-strong-offline-v2-controlled-s42"
)
DEFAULT_QA_REPORT = Path(
    "Results/MTSD-Results/Dataset-QA/"
    "Strong-Augmentation-Final-QA-20260724-R2/qa_report.json"
)
RUNNER = Path("Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py")
DEFAULT_RUNNER_PYTHON = Path(r"C:\Users\fridge\anaconda3\envs\mtsd-base\python.exe")

MODEL_FAMILIES = {
    "yolo11s": "YOLO11",
    "yolo26s": "YOLO26",
    "yolo12s": "YOLO12",
    "rfdetr-n": "RF-DETR",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runner_python() -> str:
    configured = Path(os.environ.get("MTSD_RUNNER_PYTHON", str(DEFAULT_RUNNER_PYTHON)))
    return str(configured if configured.is_file() else Path(sys.executable))


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _execution_environment(wandb_mode: str) -> dict[str, str]:
    environment = dict(os.environ)
    if wandb_mode != "online":
        guard = str((HERE / "offline_guard").resolve())
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            guard if not existing_pythonpath else guard + os.pathsep + existing_pythonpath
        )
        environment["UV_OFFLINE_GUARD"] = "1"
    else:
        environment.pop("UV_OFFLINE_GUARD", None)
    environment.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "DO_NOT_TRACK": "1",
        "ULTRALYTICS_HUB": "false",
        "PIP_NO_INDEX": "1",
        "PYTHONUNBUFFERED": "1",
    })
    if wandb_mode == "disabled":
        environment["WANDB_MODE"] = "disabled"
        environment["WANDB_DISABLED"] = "true"
    else:
        environment["WANDB_MODE"] = wandb_mode
        environment.pop("WANDB_DISABLED", None)
    return environment


def _run_directory(job: dict[str, Any]) -> Path:
    return Path("Results/MTSD-Runs") / f"{MODEL_FAMILIES[job['model']]}-MTSD" / job["label"]


def _state_path(job: dict[str, Any]) -> Path:
    return Path("Results/MTSD-Runs/Supervised-Matrix") / job["label"] / "state.json"


def _base_command(job: dict[str, Any], wandb_mode: str, wandb_group: str) -> list[str]:
    command = [
        _runner_python(),
        str(RUNNER),
        "--final",
        "--dataset-variant", "augmented",
        "--offline-augmentation", "strong",
        "--models", job["model"],
        "--image-size", str(job["image_size"]),
        "--epochs", str(job["epochs"]),
        "--run-label", job["label"],
        "--output-name", job["label"],
        "--wandb-mode", wandb_mode,
        "--wandb-group", wandb_group,
        "--device", "cuda",
        "--yolo-online-augmentation", "disabled",
    ]
    if job["kind"] == "feasibility":
        command.append("--feasibility-test")
    else:
        command.append("--require-unified-eval")
    return command


def _jobs(profile: str = "controlled") -> list[dict[str, Any]]:
    base = [
        {
            "key": "yolo11s_960",
            "model": "yolo11s",
            "image_size": 960,
            "epochs": 100,
            "label": "strongaug-yolo11s-img960-s42",
            "kind": "full",
            "condition": "always",
        },
        {
            "key": "yolo11s_1280",
            "model": "yolo11s",
            "image_size": 1280,
            "epochs": 100,
            "label": "strongaug-yolo11s-img1280-s42",
            "kind": "full",
            "condition": "always",
        },
        {
            "key": "yolo26s_960",
            "model": "yolo26s",
            "image_size": 960,
            "epochs": 100,
            "label": "strongaug-yolo26s-img960-s42",
            "kind": "full",
            "condition": "always",
        },
        {
            "key": "yolo26s_1280",
            "model": "yolo26s",
            "image_size": 1280,
            "epochs": 100,
            "label": "strongaug-yolo26s-img1280-s42",
            "kind": "full",
            "condition": "always",
        },
        {
            "key": "rfdetr_n_384",
            "model": "rfdetr-n",
            "image_size": 384,
            "epochs": 100,
            "label": "strongaug-rfdetr-n-img384-s42",
            "kind": "full",
            "condition": "always",
        },
        {
            "key": "yolo12s_960_feasibility",
            "model": "yolo12s",
            "image_size": 960,
            "epochs": 2,
            "label": "feasibility-strongaug-yolo12s-img960-e2-s42",
            "kind": "feasibility",
            "condition": "always",
            "timeout_seconds": 3600,
        },
        {
            "key": "yolo12s_960_full",
            "model": "yolo12s",
            "image_size": 960,
            "epochs": 100,
            "label": "strongaug-yolo12s-img960-s42",
            "kind": "full",
            "condition": "feasibility_passed",
        },
        {
            "key": "yolo12s_640_fallback",
            "model": "yolo12s",
            "image_size": 640,
            "epochs": 100,
            "label": "strongaug-yolo12s-img640-s42",
            "kind": "full",
            "condition": "feasibility_timed_out",
        },
    ]
    if profile == "requested":
        requested = [
            {
                **job,
                "label": job["label"].replace("strongaug-", "strongaug-wandb-cuda-"),
            }
            for job in base[:5]
        ]
        return [
            *requested,
            {
                "key": "yolo12s_960_full",
                "model": "yolo12s",
                "image_size": 960,
                "epochs": 100,
                "label": "strongaug-wandb-cuda-yolo12s-img960-s42",
                "kind": "full",
                "condition": "always",
            },
        ]
    return base


def _gpu_sample() -> dict[str, float] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
        values = [float(value.strip()) for value in result.stdout.splitlines()[0].split(",")]
        return {
            "utilization_percent": values[0],
            "memory_used_mib": values[1],
            "memory_total_mib": values[2],
            "temperature_c": values[3],
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _compute_processes() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip() or "No running" in line:
            continue
        values = [value.strip() for value in line.split(",", 2)]
        if len(values) == 3:
            process_name = values[1]
            try:
                numeric_memory = float(values[2])
            except ValueError:
                numeric_memory = None
            is_python = Path(process_name).name.casefold() in {
                "python", "python.exe", "pythonw", "pythonw.exe",
            }
            # WDDM reports ordinary desktop graphics clients in this query with
            # "[N/A]" memory. They are not CUDA compute holders. Keep Python
            # rows even when WDDM hides their memory, since trainers are Python.
            if numeric_memory is None and not is_python:
                continue
            rows.append({
                "pid": values[0],
                "process_name": process_name,
                "memory_mib": values[2],
            })
    return rows


def _wait_for_gpu_release(timeout_seconds: int = 180) -> None:
    deadline = time.monotonic() + timeout_seconds
    while _compute_processes() and time.monotonic() < deadline:
        time.sleep(5)
    active = _compute_processes()
    if active:
        raise RuntimeError(f"GPU remains occupied after cleanup: {active}")


def _terminate_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    else:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


def _clear_cuda(environment: dict[str, str]) -> None:
    subprocess.run(
        [
            _runner_python(),
            "-c",
            "import gc; gc.collect(); import torch; "
            "torch.cuda.empty_cache() if torch.cuda.is_available() else None",
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    gc.collect()
    _wait_for_gpu_release()


def _probe_local_models(environment: dict[str, str], checkpoints: dict[str, dict[str, Any]]) -> dict[str, Any]:
    probe = (
        "import gc,json,os,torch;"
        "from ultralytics import YOLO;"
        f"paths={json.dumps({key: value['path'] for key, value in checkpoints.items()})};"
        "[YOLO(paths[key]) for key in ('yolo11s','yolo26s','yolo12s')];"
        "import rfdetr;"
        "model=rfdetr.RFDETRNano(pretrain_weights=paths['rfdetr-n'],resolution=384,"
        "gradient_checkpointing=True,device='cpu');"
        "del model;gc.collect();"
        "print(json.dumps({'offline_guard_active':os.getenv('UV_OFFLINE_GUARD_ACTIVE'),"
        "'cuda_available':torch.cuda.is_available(),"
        "'cuda_device_count':torch.cuda.device_count(),"
        "'torch_version':torch.__version__,"
        "'models_loaded':sorted(paths)}))"
    )
    result = subprocess.run(
        [_runner_python(), "-c", probe],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "Offline local-model probe failed:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def _find_baseline(model: str, image_size: int) -> Path | None:
    matches: list[Path] = []
    for path in Path("Results/MTSD-Runs").glob("*-MTSD/*/run_record.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        record_size = (
            record.get("train_args", {}).get("imgsz")
            or record.get("model_args", {}).get("resolution")
        )
        if (
            record.get("status") == "completed"
            and record.get("model") == model
            and int(record_size or -1) == image_size
            and record.get("dataset_version") == "mtsd-qa-v1-aug"
        ):
            matches.append(path)
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def _rf_baseline() -> tuple[Path, dict[str, Any]]:
    baseline = _find_baseline("rfdetr-n", 384)
    if baseline is None:
        raise FileNotFoundError("Completed mild RF-DETR-N 384 baseline record was not found")
    record = json.loads(baseline.read_text(encoding="utf-8"))
    if (
        record.get("model_args", {}).get("resolution") != 384
        or record.get("train_args", {}).get("batch_size") != 32
        or record.get("train_args", {}).get("grad_accum_steps") != 1
    ):
        raise RuntimeError(f"Unexpected RF-DETR-N baseline protocol: {baseline}")
    return baseline, record


def _protocol_comparisons(config: dict[str, Any]) -> list[dict[str, Any]]:
    expected = {
        "epochs": int(config["training"]["epochs"]),
        "optimizer": config["training"]["optimizer"],
        "lr0": float(config["training"]["lr0"]),
        "lrf": float(config["training"]["lrf"]),
        "weight_decay": float(config["training"]["weight_decay"]),
        "warmup_epochs": float(config["training"]["warmup_epochs"]),
        "patience": int(config["training"]["patience"]),
        "seed": int(config["training"]["seed"]),
        "deterministic": bool(config["training"]["deterministic"]),
        "amp": bool(config["training"].get("amp", True)),
    }
    comparisons = []
    for model, image_size in (
        ("yolo11s", 960),
        ("yolo11s", 1280),
        ("yolo26s", 960),
        ("yolo26s", 1280),
        ("yolo12s", 640),
    ):
        path = _find_baseline(model, image_size)
        if path is None:
            comparisons.append({
                "model": model,
                "image_size": image_size,
                "status": "baseline_missing",
                "path": None,
            })
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        train_args = dict(record.get("train_args", {}))
        args_path = path.parent / "args.yaml"
        if train_args.get("amp") is None and args_path.is_file():
            saved_args = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
            train_args["amp"] = saved_args.get("amp")
        differences = {
            key: {"baseline": train_args.get(key), "strong": value}
            for key, value in expected.items()
            if train_args.get(key) != value
        }
        if differences:
            raise RuntimeError(
                f"Comparable baseline protocol drift for {model} img{image_size}: {differences}"
            )
        comparisons.append({
            "model": model,
            "image_size": image_size,
            "status": "exact_hyperparameter_match",
            "path": str(path.resolve()),
            "checked": expected,
        })
    return comparisons


def preflight(
    config: dict[str, Any],
    qa_report_path: Path,
    queue_root: Path,
    *,
    allow_existing_queue: bool,
    job_profile: str,
    wandb_mode: str,
    wandb_group: str,
) -> dict[str, Any]:
    qa = json.loads(qa_report_path.read_text(encoding="utf-8"))
    if qa.get("final_status") != "pass":
        raise RuntimeError(f"Strong augmentation QA is not a final pass: {qa_report_path}")
    dataset_root = Path("Datasets/MTSD/Prepared/MTSD-Augmented-Strong")
    prep_path = dataset_root / "prep_manifest.json"
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    if qa.get("prep_manifest_sha256") != sha256_file(prep_path):
        raise RuntimeError("QA report does not match the current strong dataset manifest")
    if prep.get("dataset_version") != "mtsd-qa-v1-strong-offline-v2":
        raise RuntimeError(f"Unexpected strong dataset version: {prep.get('dataset_version')}")
    if prep.get("augmentation", {}).get("online_training_augmentation") != "disabled":
        raise RuntimeError("Prepared strong dataset does not record disabled online augmentation")
    if config["training"].get("yolo_online_augmentation") != "disabled":
        raise RuntimeError("Ultralytics online augmentation must remain disabled")
    online_args = yolo_online_augmentation_args(config["training"])
    enabled_online = {
        key: value for key, value in online_args.items()
        if value not in {0, 0.0, None, False}
    }
    if enabled_online:
        raise RuntimeError(f"Ultralytics online augmentation is active: {enabled_online}")
    coco_normalization_noop = True
    for split in ("train", "valid", "test"):
        annotation_path = (
            dataset_root / "MTSD-COCO" / split / "_annotations.coco.json"
        )
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = sorted(payload["categories"], key=lambda row: row["id"])
        if any(
            int(category["id"]) != index
            or category.get("supercategory") != "mtsd"
            for index, category in enumerate(categories)
        ):
            coco_normalization_noop = False
            break
    if not coco_normalization_noop:
        raise RuntimeError(
            "RF-DETR category normalization would mutate the prepared dataset"
        )

    checkpoints = {
        spec.key: resolve_checkpoint_info(spec, REPO_ROOT, require_local=True)
        for spec in validate_model_keys(["yolo11s", "yolo26s", "yolo12s", "rfdetr-n"])
    }
    rf_path, rf_record = _rf_baseline()
    protocol_comparisons = _protocol_comparisons(config)
    environment = _execution_environment(wandb_mode)
    model_probe = _probe_local_models(environment, checkpoints)
    if wandb_mode != "online" and model_probe.get("offline_guard_active") != "1":
        raise RuntimeError("Offline network guard did not activate in the model probe")
    if not model_probe.get("cuda_available") or int(model_probe.get("cuda_device_count") or 0) < 1:
        raise RuntimeError(f"Selected runner Python cannot see CUDA: {model_probe}")

    active_compute = _compute_processes()
    if active_compute:
        raise RuntimeError(f"GPU is already occupied by compute processes: {active_compute}")
    if queue_root.exists() and not allow_existing_queue:
        raise FileExistsError(f"Immutable queue root already exists: {queue_root}")

    jobs = _jobs(job_profile)
    collisions = []
    for job in jobs:
        if _run_directory(job).exists() or _state_path(job).exists():
            collisions.append({
                "label": job["label"],
                "run_directory": str(_run_directory(job)),
                "state_path": str(_state_path(job)),
            })
    if collisions and not allow_existing_queue:
        raise FileExistsError(f"Immutable run/state collision: {collisions}")

    batch_plans = {
        str(size): yolo_batch_plan(config["training"], size)
        for size in (640, 960, 1280)
    }
    if any(plan["optimizer_effective_batch"] != 64 for plan in batch_plans.values()):
        raise RuntimeError(f"Inconsistent effective YOLO batch plans: {batch_plans}")
    return {
        "created_at": _now(),
        "status": "pass",
        "qa_report": str(qa_report_path.resolve()),
        "qa_report_sha256": sha256_file(qa_report_path),
        "dataset_root": str(dataset_root.resolve()),
        "dataset_version": prep["dataset_version"],
        "prep_manifest_sha256": sha256_file(prep_path),
        "split_manifest_sha256": prep["split_manifest_sha256"],
        "checkpoints": checkpoints,
        "model_probe": model_probe,
        "offline_environment": {
            key: environment.get(key)
            for key in (
                "UV_OFFLINE_GUARD", "WANDB_MODE", "WANDB_DISABLED",
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                "PIP_NO_INDEX",
            )
        },
        "yolo_online_augmentation_args": online_args,
        "rfdetr_coco_normalization_noop": coco_normalization_noop,
        "batch_plans": batch_plans,
        "baseline_protocol_comparisons": protocol_comparisons,
        "rfdetr_baseline": {
            "path": str(rf_path.resolve()),
            "resolution": rf_record["model_args"]["resolution"],
            "batch_size": rf_record["train_args"]["batch_size"],
            "gradient_accumulation": rf_record["train_args"]["grad_accum_steps"],
            "multi_scale": rf_record["train_args"]["multi_scale"],
            "expanded_scales": rf_record["train_args"]["expanded_scales"],
            "internal_augmentation": (
                "RandomHorizontalFlip; RandomSelect between SquareResize and "
                "RandomResize+RandomSizeCrop+SquareResize. No installed public "
                "switch disables these transforms."
            ),
            "causal_limitation": "descriptive_only_uncontrolled_rfdetr_internal_augmentation",
        },
        "jobs": [
            {
                **job,
                "command": _base_command(job, wandb_mode, wandb_group),
                "run_directory": str(_run_directory(job)),
                "state_path": str(_state_path(job)),
            }
            for job in jobs
        ],
    }


def _mark_inner_timeout(job: dict[str, Any], elapsed: float) -> None:
    state_path = _state_path(job)
    if not state_path.is_file():
        return
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["current_model"] = None
    state["final_status"] = "stopped_by_feasibility_threshold"
    state["finished_at"] = _now()
    model_state = state.get("models", {}).get(job["model"], {})
    model_state.update({
        "status": "stopped_by_feasibility_threshold",
        "finished_at": state["finished_at"],
        "elapsed_seconds": elapsed,
    })
    _atomic_write(state_path, state)


def _run_job(
    job: dict[str, Any],
    queue_root: Path,
    queue_state: dict[str, Any],
    state_path: Path,
    environment: dict[str, str],
    wandb_mode: str,
    wandb_group: str,
) -> dict[str, Any]:
    active = _compute_processes()
    if active:
        raise RuntimeError(f"Refusing to launch {job['label']}; GPU is occupied: {active}")
    command = _base_command(job, wandb_mode, wandb_group)
    log_path = queue_root / "logs" / f"{job['label']}.log"
    telemetry_path = queue_root / "telemetry" / f"{job['label']}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = _now()
    started = time.monotonic()
    job_state = queue_state["jobs"][job["key"]]
    job_state.update({
        "status": "running",
        "started_at": started_at,
        "command": command,
        "log_path": str(log_path),
        "run_directory": str(_run_directory(job)),
    })
    queue_state["current_job"] = job["key"]
    queue_state["updated_at"] = started_at
    _atomic_write(state_path, queue_state)

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    samples: list[dict[str, float]] = []
    timed_out = False
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creationflags,
        )
        job_state["pid"] = process.pid
        _atomic_write(state_path, queue_state)
        last_state_write = time.monotonic()
        timeout = job.get("timeout_seconds")
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if timeout is not None and elapsed >= float(timeout):
                timed_out = True
                _terminate_tree(process)
                break
            sample = _gpu_sample()
            if sample is not None:
                sample["elapsed_seconds"] = elapsed
                samples.append(sample)
            if time.monotonic() - last_state_write >= 60:
                job_state["heartbeat_at"] = _now()
                job_state["elapsed_seconds"] = elapsed
                _atomic_write(state_path, queue_state)
                last_state_write = time.monotonic()
            sleep_seconds = 5.0
            if timeout is not None:
                sleep_seconds = min(
                    sleep_seconds,
                    max(0.05, float(timeout) - elapsed),
                )
            time.sleep(sleep_seconds)
        try:
            return_code = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            _terminate_tree(process)
            return_code = process.wait(timeout=60)

    elapsed = time.monotonic() - started
    telemetry = {
        "label": job["label"],
        "started_at": started_at,
        "finished_at": _now(),
        "elapsed_seconds": elapsed,
        "time_per_epoch_seconds": elapsed / job["epochs"],
        "sample_interval_seconds": 5,
        "sample_count": len(samples),
        "average_gpu_utilization_percent": (
            sum(sample["utilization_percent"] for sample in samples) / len(samples)
            if samples else None
        ),
        "peak_gpu_utilization_percent": (
            max((sample["utilization_percent"] for sample in samples), default=None)
        ),
        "peak_vram_mib": max((sample["memory_used_mib"] for sample in samples), default=None),
        "peak_temperature_c": max((sample["temperature_c"] for sample in samples), default=None),
        "physical_batch": (
            yolo_batch_plan(load_config(HERE / "config/default.yaml")["training"], job["image_size"])["physical_batch"]
            if job["model"].startswith("yolo") else 32
        ),
        "gradient_accumulation": (
            yolo_batch_plan(load_config(HERE / "config/default.yaml")["training"], job["image_size"])["gradient_accumulation_steps"]
            if job["model"].startswith("yolo") else 1
        ),
        "effective_batch": (
            yolo_batch_plan(load_config(HERE / "config/default.yaml")["training"], job["image_size"])["optimizer_effective_batch"]
            if job["model"].startswith("yolo") else 32
        ),
        "amp_active": job["model"].startswith("yolo"),
        "return_code": return_code,
        "timeout_seconds": job.get("timeout_seconds"),
        "timeout_triggered": timed_out,
    }
    _atomic_write(telemetry_path, telemetry)
    if timed_out:
        _mark_inner_timeout(job, elapsed)
        status = "stopped_by_feasibility_threshold"
    else:
        record_path = _run_directory(job) / "run_record.json"
        completed = False
        if return_code == 0 and record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            completed = record.get("status") == "completed"
        status = "completed" if completed else "failed"
    job_state.update({
        "status": status,
        "finished_at": telemetry["finished_at"],
        "elapsed_seconds": elapsed,
        "return_code": return_code,
        "telemetry_path": str(telemetry_path),
        "timeout_triggered": timed_out,
    })
    queue_state["current_job"] = None
    queue_state["updated_at"] = _now()
    _atomic_write(state_path, queue_state)
    _clear_cuda(environment)
    return telemetry


def _metric(record: dict[str, Any], key: str) -> float | None:
    value = record.get("unified_test_metrics", {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _difference(new: float | None, old: float | None) -> float | None:
    return new - old if new is not None and old is not None else None


def update_comparison(queue_root: Path, jobs: list[dict[str, Any]]) -> None:
    rows = []
    for job in jobs:
        if job["kind"] != "full":
            continue
        new_path = _run_directory(job) / "run_record.json"
        if not new_path.is_file():
            continue
        new = json.loads(new_path.read_text(encoding="utf-8"))
        if new.get("status") != "completed":
            continue
        baseline_path = _find_baseline(job["model"], job["image_size"])
        baseline = (
            json.loads(baseline_path.read_text(encoding="utf-8"))
            if baseline_path else {}
        )
        base_map = _metric(baseline, "map50_95")
        new_map = _metric(new, "map50_95")
        base_f1 = _metric(baseline, "f1")
        new_f1 = _metric(new, "f1")
        base_precision = _metric(baseline, "precision")
        new_precision = _metric(new, "precision")
        base_recall = _metric(baseline, "recall")
        new_recall = _metric(new, "recall")
        base_time = baseline.get("training_seconds")
        new_time = new.get("training_seconds")
        rows.append({
            "model": job["model"],
            "image_size": job["image_size"],
            "previous_augmentation_recipe": "photometric-v1+motion-blur-v1" if baseline else "missing",
            "new_augmentation_recipe": "strong-offline-v2",
            "baseline_run_record": str(baseline_path) if baseline_path else None,
            "strong_run_record": str(new_path),
            "baseline_map50_95": base_map,
            "strong_augmentation_map50_95": new_map,
            "absolute_map_difference": _difference(new_map, base_map),
            "baseline_f1": base_f1,
            "strong_augmentation_f1": new_f1,
            "absolute_f1_difference": _difference(new_f1, base_f1),
            "precision_difference": _difference(new_precision, base_precision),
            "recall_difference": _difference(new_recall, base_recall),
            "training_time_difference_seconds": _difference(
                float(new_time) if isinstance(new_time, (int, float)) else None,
                float(base_time) if isinstance(base_time, (int, float)) else None,
            ),
        })
    (queue_root / "comparison.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "model", "image_size", "previous_augmentation_recipe", "new_augmentation_recipe",
        "baseline_run_record", "strong_run_record", "baseline_map50_95",
        "strong_augmentation_map50_95", "absolute_map_difference", "baseline_f1",
        "strong_augmentation_f1", "absolute_f1_difference", "precision_difference",
        "recall_difference", "training_time_difference_seconds",
    ]
    with (queue_root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_queue(
    queue_root: Path,
    preflight_result: dict[str, Any],
    resume: bool,
    *,
    job_profile: str,
    wandb_mode: str,
    wandb_group: str,
) -> dict[str, Any]:
    jobs = _jobs(job_profile)
    state_path = queue_root / "queue_state.json"
    if resume:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        queue_root.mkdir(parents=True, exist_ok=False)
        state = {
            "schema_version": "mtsd-strong-controlled-queue-v1",
            "created_at": _now(),
            "updated_at": _now(),
            "status": "running",
            "current_job": None,
            "qa_report": preflight_result["qa_report"],
            "preflight": str(queue_root / "preflight.json"),
            "feasibility_outcome": "pending",
            "jobs": {
                job["key"]: {
                    **job,
                    "status": (
                        "pending_condition"
                        if job["condition"] != "always"
                        else "pending"
                    ),
                    "command": _base_command(job, wandb_mode, wandb_group),
                    "run_directory": str(_run_directory(job)),
                    "state_path": str(_state_path(job)),
                }
                for job in jobs
            },
        }
        _atomic_write(queue_root / "preflight.json", preflight_result)
        _atomic_write(queue_root / "commands.json", {"jobs": preflight_result["jobs"]})
        _atomic_write(state_path, state)
    environment = _execution_environment(wandb_mode)

    for job in jobs:
        job_state = state["jobs"][job["key"]]
        if job_state["status"] in {"completed", "skipped", "stopped_by_feasibility_threshold"}:
            continue
        condition = job["condition"]
        if condition == "feasibility_passed" and state["feasibility_outcome"] != "passed":
            if state["feasibility_outcome"] == "timed_out":
                job_state["status"] = "skipped"
                job_state["skip_reason"] = "960 feasibility threshold was not met"
                _atomic_write(state_path, state)
            continue
        if condition == "feasibility_timed_out" and state["feasibility_outcome"] != "timed_out":
            if state["feasibility_outcome"] == "passed":
                job_state["status"] = "skipped"
                job_state["skip_reason"] = "960 feasibility threshold passed; fallback not permitted"
                _atomic_write(state_path, state)
            continue
        telemetry = _run_job(job, queue_root, state, state_path, environment, wandb_mode, wandb_group)
        status = state["jobs"][job["key"]]["status"]
        if job["kind"] == "feasibility":
            if status == "completed" and telemetry["elapsed_seconds"] <= 3600:
                state["feasibility_outcome"] = "passed"
                state["jobs"]["yolo12s_960_full"]["status"] = "pending"
                state["jobs"]["yolo12s_640_fallback"].update({
                    "status": "skipped",
                    "skip_reason": "960 feasibility threshold passed; fallback not permitted",
                })
            elif status == "stopped_by_feasibility_threshold":
                state["feasibility_outcome"] = "timed_out"
                state["jobs"]["yolo12s_640_fallback"]["status"] = "pending"
                state["jobs"]["yolo12s_960_full"].update({
                    "status": "skipped",
                    "skip_reason": "960 feasibility threshold was not met",
                })
            else:
                state["status"] = "failed"
                state["failure_reason"] = "YOLO12-S feasibility run failed before the timeout decision"
                _atomic_write(state_path, state)
                return state
            _atomic_write(state_path, state)
        elif status != "completed":
            state["status"] = "failed"
            state["failure_reason"] = f"Required run failed: {job['label']}"
            _atomic_write(state_path, state)
            return state
        update_comparison(queue_root, jobs)

    terminal = {"completed", "skipped", "stopped_by_feasibility_threshold"}
    state["status"] = (
        "completed"
        if all(job["status"] in terminal for job in state["jobs"].values())
        else "running"
    )
    state["updated_at"] = _now()
    if state["status"] == "completed":
        state["finished_at"] = _now()
    _atomic_write(state_path, state)
    update_comparison(queue_root, jobs)
    return state


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Offline, sequential strong-augmentation MTSD experiment queue",
    )
    result.add_argument("--config", type=Path, default=HERE / "config/default.yaml")
    result.add_argument("--qa-report", type=Path, default=DEFAULT_QA_REPORT)
    result.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    result.add_argument("--preflight-only", action="store_true")
    result.add_argument("--launch", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument(
        "--job-profile",
        choices=("controlled", "requested"),
        default="controlled",
        help="Use 'requested' for the six W&B runs requested on 2026-07-24.",
    )
    result.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    result.add_argument("--wandb-group", default="strong-offline-v2-controlled-s42")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    os.chdir(REPO_ROOT)
    if args.resume and not args.queue_root.is_absolute():
        args.queue_root = REPO_ROOT / args.queue_root
    if args.qa_report.is_absolute():
        qa_report = args.qa_report
    else:
        qa_report = REPO_ROOT / args.qa_report
    queue_root = args.queue_root if args.queue_root.is_absolute() else REPO_ROOT / args.queue_root
    config = load_config(args.config if args.config.is_absolute() else REPO_ROOT / args.config)
    result = preflight(
        config,
        qa_report,
        queue_root,
        allow_existing_queue=args.resume,
        job_profile=args.job_profile,
        wandb_mode=args.wandb_mode,
        wandb_group=args.wandb_group,
    )
    if args.preflight_only or not args.launch:
        print(json.dumps(result, indent=2))
        return 0
    state = run_queue(
        queue_root,
        result,
        args.resume,
        job_profile=args.job_profile,
        wandb_mode=args.wandb_mode,
        wandb_group=args.wandb_group,
    )
    print(json.dumps({
        "queue_root": str(queue_root),
        "state": str(queue_root / "queue_state.json"),
        "status": state["status"],
        "current_job": state.get("current_job"),
        "feasibility_outcome": state.get("feasibility_outcome"),
    }, indent=2))
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
