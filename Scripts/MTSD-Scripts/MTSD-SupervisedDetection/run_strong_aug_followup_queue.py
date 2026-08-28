#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from mtsd_detection.dataset_validation import validate_variant
from mtsd_detection.manifests import sha256_file
from mtsd_detection.model_registry import resolve_checkpoint_info, validate_model_keys
from mtsd_detection.train_yolo import yolo_batch_plan, yolo_online_augmentation_args

RUNNER = Path("Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py")
DEFAULT_QA_REPORT = Path(
    "Results/MTSD-Results/Dataset-QA/"
    "Strong-Augmentation-Final-QA-20260724-R2/qa_report.json"
)
DEFAULT_QUEUE_ROOT = Path(
    "Results/MTSD-Runs/Strong-Augmentation-Followup/"
    "20260725-strong-offline-v2-m-rfdetr-wandb-s42"
)
WANDB_GROUP = "strong-offline-v2-followup-m-rfdetr-s42"
MODEL_FAMILIES = {
    "yolo26m": "YOLO26",
    "yolo11m": "YOLO11",
    "yolo12s": "YOLO12",
    "yolo12m": "YOLO12",
    "rfdetr-m": "RF-DETR",
    "rfdetr-s": "RF-DETR",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runner_python() -> str:
    configured = Path(os.environ.get("MTSD_RUNNER_PYTHON", sys.executable))
    return str(configured if configured.is_file() else Path(sys.executable))


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update({
        "WANDB_MODE": "online",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "DO_NOT_TRACK": "1",
        "ULTRALYTICS_HUB": "false",
        "PIP_NO_INDEX": "1",
        "PYTHONUNBUFFERED": "1",
    })
    environment.pop("WANDB_DISABLED", None)
    environment.pop("UV_OFFLINE_GUARD", None)
    return environment


def _jobs() -> list[dict[str, Any]]:
    return [
        {"key": "yolo26m_1280", "model": "yolo26m", "image_size": 1280,
         "epochs": 100, "label": "strongaug-wandb-followup-yolo26m-img1280-s42",
         "kind": "full", "condition": "always"},
        {"key": "yolo11m_1280", "model": "yolo11m", "image_size": 1280,
         "epochs": 100, "label": "strongaug-wandb-followup-yolo11m-img1280-s42",
         "kind": "full", "condition": "always"},
        {"key": "yolo12s_1280", "model": "yolo12s", "image_size": 1280,
         "epochs": 100, "label": "strongaug-wandb-followup-yolo12s-img1280-s42",
         "kind": "full", "condition": "always"},
        {"key": "rfdetr_m_704", "model": "rfdetr-m", "image_size": 704,
         "epochs": 100, "label": "strongaug-wandb-followup-rfdetr-m-img704-s42",
         "kind": "full", "condition": "always",
         "config_override": {"training": {"per_model_overrides": {"rfdetr-m": {"resolution": 704}}}}},
        {"key": "rfdetr_s_640", "model": "rfdetr-s", "image_size": 640,
         "epochs": 100, "label": "strongaug-wandb-followup-rfdetr-s-img640-s42",
         "kind": "full", "condition": "always",
         "config_override": {"training": {"per_model_overrides": {"rfdetr-s": {"resolution": 640}}}}},
        {"key": "yolo12m_1280_feasibility", "model": "yolo12m", "image_size": 1280,
         "epochs": 2, "label": "feasibility-strongaug-wandb-followup-yolo12m-img1280-e2-s42",
         "kind": "feasibility", "condition": "always", "timeout_seconds": 3600},
        {"key": "yolo12m_1280_full", "model": "yolo12m", "image_size": 1280,
         "epochs": 100, "label": "strongaug-wandb-followup-yolo12m-img1280-s42",
         "kind": "full", "condition": "feasibility_passed"},
    ]


def _run_directory(job: dict[str, Any]) -> Path:
    return Path("Results/MTSD-Runs") / f"{MODEL_FAMILIES[job['model']]}-MTSD" / job["label"]


def _state_path(job: dict[str, Any]) -> Path:
    return Path("Results/MTSD-Runs/Supervised-Matrix") / job["label"] / "state.json"


def _job_config(job: dict[str, Any], queue_root: Path) -> Path | None:
    override = job.get("config_override")
    if not override:
        return None
    config = load_config(HERE / "config/default.yaml")
    for section, values in override.items():
        config.setdefault(section, {})
        for key, value in values.items():
            if isinstance(value, dict) and isinstance(config[section].get(key), dict):
                merged = dict(config[section][key])
                merged.update(value)
                config[section][key] = merged
            else:
                config[section][key] = value
    path = queue_root / "configs" / f"{job['label']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _base_command(job: dict[str, Any], queue_root: Path) -> list[str]:
    command = [
        _runner_python(), str(RUNNER),
        "--final",
        "--dataset-variant", "augmented",
        "--offline-augmentation", "strong",
        "--models", job["model"],
        "--image-size", str(job["image_size"]),
        "--epochs", str(job["epochs"]),
        "--run-label", job["label"],
        "--output-name", job["label"],
        "--wandb-mode", "online",
        "--wandb-group", WANDB_GROUP,
        "--device", "cuda",
        "--yolo-online-augmentation", "disabled",
    ]
    config_path = _job_config(job, queue_root)
    if config_path is not None:
        command[2:2] = ["--config", str(config_path)]
    if job["kind"] == "feasibility":
        command.append("--feasibility-test")
    else:
        command.append("--require-unified-eval")
    return command


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
        if len(values) != 3:
            continue
        process_name = values[1]
        try:
            numeric_memory = float(values[2])
        except ValueError:
            numeric_memory = None
        is_python = Path(process_name).name.casefold() in {"python", "python.exe"}
        if numeric_memory is None and not is_python:
            continue
        rows.append({"pid": values[0], "process_name": process_name, "memory_mib": values[2]})
    return rows


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


def _terminate_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    subprocess.run(
        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _clear_cuda() -> None:
    subprocess.run(
        [_runner_python(), "-c",
         "import gc; gc.collect(); import torch; "
         "torch.cuda.empty_cache() if torch.cuda.is_available() else None"],
        cwd=REPO_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _probe_models(checkpoints: dict[str, dict[str, Any]]) -> dict[str, Any]:
    probe = (
        "import gc,json,torch;"
        "from ultralytics import YOLO;"
        f"paths={json.dumps({key: value['path'] for key, value in checkpoints.items()})};"
        "[YOLO(paths[key]) for key in ('yolo26m','yolo11m','yolo12s','yolo12m')];"
        "import rfdetr;"
        "a=rfdetr.RFDETRMedium(pretrain_weights=paths['rfdetr-m'],resolution=704,"
        "gradient_checkpointing=True,device='cpu');"
        "b=rfdetr.RFDETRSmall(pretrain_weights=paths['rfdetr-s'],resolution=640,"
        "gradient_checkpointing=True,device='cpu');"
        "del a,b;gc.collect();"
        "print(json.dumps({'cuda_available':torch.cuda.is_available(),"
        "'cuda_device_count':torch.cuda.device_count(),"
        "'torch_version':torch.__version__}))"
    )
    result = subprocess.run(
        [_runner_python(), "-c", probe],
        cwd=REPO_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Local model probe failed:\n{result.stdout}\n{result.stderr}")
    return json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])


def _find_run_record(model: str, image_size: int, dataset_version: str) -> str | None:
    matches = []
    for path in Path("Results/MTSD-Runs").glob("*-MTSD/*/run_record.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        size = record.get("train_args", {}).get("imgsz") or record.get("model_args", {}).get("resolution")
        if (
            record.get("status") == "completed"
            and record.get("model") == model
            and int(size or -1) == image_size
            and record.get("dataset_version") == dataset_version
        ):
            matches.append(path)
    return str(max(matches, key=lambda path: path.stat().st_mtime).resolve()) if matches else None


def _baseline_map() -> dict[str, str | None]:
    strong = "mtsd-qa-v1-strong-offline-v2"
    return {
        "yolo26m_1280": _find_run_record("yolo26s", 1280, strong),
        "yolo11m_1280": _find_run_record("yolo11s", 1280, strong),
        "yolo12s_1280": _find_run_record("yolo12s", 960, strong),
        "rfdetr_m_704": _find_run_record("rfdetr-n", 384, strong),
        "rfdetr_s_640": _find_run_record("rfdetr-n", 384, strong),
        "yolo12m_1280_full": _find_run_record("yolo12s", 960, strong),
    }


def preflight(queue_root: Path, qa_report: Path, allow_existing_queue: bool) -> dict[str, Any]:
    config = load_config(HERE / "config/default.yaml")
    qa = json.loads(qa_report.read_text(encoding="utf-8"))
    if qa.get("final_status") != "pass":
        raise RuntimeError(f"Strong augmentation QA is not a final pass: {qa_report}")
    dataset_root = Path("Datasets/MTSD/Prepared/MTSD-Augmented-Strong")
    prep_path = dataset_root / "prep_manifest.json"
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    if prep.get("dataset_version") != "mtsd-qa-v1-strong-offline-v2":
        raise RuntimeError(f"Unexpected dataset version: {prep.get('dataset_version')}")
    if prep.get("split_manifest_sha256") != qa.get("split_manifest_sha256"):
        raise RuntimeError("QA report split hash does not match current strong dataset split")
    current_prep_hash = sha256_file(prep_path)
    prep_hash_warning = None
    if current_prep_hash != qa.get("prep_manifest_sha256"):
        prep_hash_warning = {
            "qa_report_prep_manifest_sha256": qa.get("prep_manifest_sha256"),
            "current_prep_manifest_sha256": current_prep_hash,
            "decision": (
                "accepted_after_strict_validation_because_dataset_version_and_"
                "split_manifest_sha256_match"
            ),
        }
    validation = validate_variant(dataset_root, strict=True, policy=config["validation"])
    if not validation["ok"]:
        raise RuntimeError(f"Strong dataset strict validation failed: {validation}")
    online_args = yolo_online_augmentation_args(config["training"])
    enabled_online = {key: value for key, value in online_args.items() if value not in {0, 0.0, None, False}}
    if enabled_online:
        raise RuntimeError(f"Ultralytics online augmentation is active: {enabled_online}")
    checkpoints = {
        spec.key: resolve_checkpoint_info(spec, REPO_ROOT, require_local=True)
        for spec in validate_model_keys(["yolo26m", "yolo11m", "yolo12s", "yolo12m", "rfdetr-m", "rfdetr-s"])
    }
    probe = _probe_models(checkpoints)
    if not probe.get("cuda_available") or int(probe.get("cuda_device_count") or 0) < 1:
        raise RuntimeError(f"Selected runner Python cannot see CUDA: {probe}")
    active = _compute_processes()
    if active:
        raise RuntimeError(f"GPU is already occupied by compute processes: {active}")
    if queue_root.exists() and not allow_existing_queue:
        occupied = [
            child.name for child in queue_root.iterdir()
            if child.name not in {"configs"}
        ]
        if occupied:
            raise FileExistsError(
                f"Immutable queue root already exists with run state/artifacts: "
                f"{queue_root} ({occupied})"
            )
    collisions = []
    jobs = _jobs()
    for job in jobs:
        if _run_directory(job).exists() or _state_path(job).exists():
            collisions.append({"label": job["label"], "run_directory": str(_run_directory(job)),
                               "state_path": str(_state_path(job))})
    if collisions and not allow_existing_queue:
        raise FileExistsError(f"Immutable run/state collision: {collisions}")
    batch_plans = {str(size): yolo_batch_plan(config["training"], size) for size in (1280,)}
    return {
        "created_at": _now(),
        "status": "pass",
        "dataset_root": str(dataset_root.resolve()),
        "dataset_version": prep["dataset_version"],
        "prep_manifest_sha256": current_prep_hash,
        "prep_manifest_hash_warning": prep_hash_warning,
        "qa_report": str(qa_report.resolve()),
        "checkpoints": checkpoints,
        "model_probe": probe,
        "environment": {key: _environment().get(key) for key in (
            "WANDB_MODE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE", "PIP_NO_INDEX")},
        "yolo_online_augmentation_args": online_args,
        "batch_plans": batch_plans,
        "baselines": _baseline_map(),
        "rfdetr_resolution_note": (
            "RF-DETR-S is intentionally raised from default 512 to 640; "
            "RF-DETR-M is intentionally raised from default 576 to 704."
        ),
        "jobs": [{**job, "command": _base_command(job, queue_root),
                  "run_directory": str(_run_directory(job)),
                  "state_path": str(_state_path(job))} for job in jobs],
    }


def _run_job(job: dict[str, Any], queue_root: Path, state: dict[str, Any], state_path: Path) -> dict[str, Any]:
    active = _compute_processes()
    if active:
        raise RuntimeError(f"Refusing to launch {job['label']}; GPU is occupied: {active}")
    command = _base_command(job, queue_root)
    log_path = queue_root / "logs" / f"{job['label']}.log"
    telemetry_path = queue_root / "telemetry" / f"{job['label']}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = _now()
    started = time.monotonic()
    job_state = state["jobs"][job["key"]]
    job_state.update({"status": "running", "started_at": started_at, "command": command,
                      "log_path": str(log_path), "run_directory": str(_run_directory(job))})
    state["current_job"] = job["key"]
    state["updated_at"] = started_at
    _atomic_write(state_path, state)
    samples: list[dict[str, float]] = []
    timed_out = False
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=_environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        job_state["pid"] = process.pid
        _atomic_write(state_path, state)
        last_state_write = time.monotonic()
        timeout = job.get("timeout_seconds")
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if timeout is not None and elapsed >= float(timeout):
                timed_out = True
                _terminate_tree(process)
                break
            sample = _gpu_sample()
            if sample:
                sample["elapsed_seconds"] = elapsed
                samples.append(sample)
            if time.monotonic() - last_state_write >= 60:
                job_state["heartbeat_at"] = _now()
                job_state["elapsed_seconds"] = elapsed
                _atomic_write(state_path, state)
                last_state_write = time.monotonic()
            time.sleep(min(5.0, max(0.05, float(timeout) - elapsed)) if timeout else 5.0)
        try:
            return_code = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            _terminate_tree(process)
            return_code = process.wait(timeout=60)
    elapsed = time.monotonic() - started
    record_path = _run_directory(job) / "run_record.json"
    completed = False
    if return_code == 0 and record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        completed = record.get("status") == "completed"
    status = "completed" if completed else ("stopped_by_feasibility_threshold" if timed_out else "failed")
    telemetry = {
        "label": job["label"],
        "started_at": started_at,
        "finished_at": _now(),
        "elapsed_seconds": elapsed,
        "time_per_epoch_seconds": elapsed / job["epochs"],
        "sample_count": len(samples),
        "average_gpu_utilization_percent": (
            sum(sample["utilization_percent"] for sample in samples) / len(samples) if samples else None),
        "peak_gpu_utilization_percent": max((sample["utilization_percent"] for sample in samples), default=None),
        "peak_vram_mib": max((sample["memory_used_mib"] for sample in samples), default=None),
        "return_code": return_code,
        "timeout_seconds": timeout,
        "timeout_triggered": timed_out,
    }
    _atomic_write(telemetry_path, telemetry)
    job_state.update({"status": status, "finished_at": telemetry["finished_at"],
                      "elapsed_seconds": elapsed, "return_code": return_code,
                      "telemetry_path": str(telemetry_path), "timeout_triggered": timed_out})
    state["current_job"] = None
    state["updated_at"] = _now()
    _atomic_write(state_path, state)
    _clear_cuda()
    return telemetry


def run_queue(queue_root: Path, preflight_result: dict[str, Any], resume: bool) -> dict[str, Any]:
    jobs = _jobs()
    state_path = queue_root / "queue_state.json"
    if resume:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        if state_path.exists():
            raise FileExistsError(f"Immutable queue state already exists: {state_path}")
        queue_root.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": "mtsd-strong-followup-queue-v1",
            "created_at": _now(),
            "updated_at": _now(),
            "status": "running",
            "current_job": None,
            "qa_report": preflight_result["qa_report"],
            "preflight": str(queue_root / "preflight.json"),
            "feasibility_outcome": "pending",
            "jobs": {job["key"]: {**job, "status": (
                "pending_condition" if job["condition"] != "always" else "pending"),
                "command": _base_command(job, queue_root),
                "run_directory": str(_run_directory(job)),
                "state_path": str(_state_path(job))} for job in jobs},
        }
        _atomic_write(queue_root / "preflight.json", preflight_result)
        _atomic_write(queue_root / "commands.json", {"jobs": preflight_result["jobs"]})
        _atomic_write(state_path, state)
    for job in jobs:
        job_state = state["jobs"][job["key"]]
        if job_state["status"] in {"completed", "skipped", "stopped_by_feasibility_threshold"}:
            continue
        if job["condition"] == "feasibility_passed" and state["feasibility_outcome"] != "passed":
            continue
        telemetry = _run_job(job, queue_root, state, state_path)
        status = state["jobs"][job["key"]]["status"]
        if job["kind"] == "feasibility":
            log_text = Path(state["jobs"][job["key"]]["log_path"]).read_text(
                encoding="utf-8", errors="ignore") if "log_path" in state["jobs"][job["key"]] else ""
            oom = "out of memory" in log_text.lower() or "cuda error" in log_text.lower()
            if status == "completed" and telemetry["elapsed_seconds"] <= float(job["timeout_seconds"]):
                state["feasibility_outcome"] = "passed"
                state["jobs"]["yolo12m_1280_full"]["status"] = "pending"
            else:
                state["feasibility_outcome"] = "failed_or_infeasible"
                state["feasibility_reason"] = (
                    "oom_or_cuda_error" if oom else
                    "timeout" if status == "stopped_by_feasibility_threshold" else
                    "nonzero_return_code"
                )
                state["jobs"]["yolo12m_1280_full"].update({
                    "status": "skipped",
                    "skip_reason": f"YOLO12-M 1280 feasibility was {state['feasibility_reason']}",
                })
            _atomic_write(state_path, state)
            continue
        if status != "completed":
            state["status"] = "failed"
            state["failure_reason"] = f"Required run failed: {job['label']}"
            _atomic_write(state_path, state)
            return state
    terminal = {"completed", "skipped", "stopped_by_feasibility_threshold"}
    state["status"] = (
        "completed" if all(job["status"] in terminal for job in state["jobs"].values()) else "running")
    state["updated_at"] = _now()
    if state["status"] == "completed":
        state["finished_at"] = _now()
    _atomic_write(state_path, state)
    return state


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Strong-offline MTSD follow-up M/RF-DETR queue")
    result.add_argument("--qa-report", type=Path, default=DEFAULT_QA_REPORT)
    result.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    result.add_argument("--preflight-only", action="store_true")
    result.add_argument("--launch", action="store_true")
    result.add_argument("--resume", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    os.chdir(REPO_ROOT)
    queue_root = args.queue_root if args.queue_root.is_absolute() else REPO_ROOT / args.queue_root
    qa_report = args.qa_report if args.qa_report.is_absolute() else REPO_ROOT / args.qa_report
    result = preflight(queue_root, qa_report, allow_existing_queue=args.resume)
    if args.preflight_only or not args.launch:
        print(json.dumps(result, indent=2, default=str))
        return 0
    state = run_queue(queue_root, result, args.resume)
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
