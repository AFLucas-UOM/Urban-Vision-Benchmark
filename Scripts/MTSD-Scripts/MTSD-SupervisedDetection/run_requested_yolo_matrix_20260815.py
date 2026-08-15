#!/usr/bin/env python3
"""Run the requested YOLO MTSD matrix without retraining completed runs.

The queue is immutable by output name, sequential by design, and resumable.
Strong-augmentation jobs use final-mode QA gates; NoAug jobs intentionally use
development mode because the canonical runner reserves --final for augmented
dissertation runs. Final test scoring is deferred so every checkpoint can be
evaluated together with the same unified held-out-test protocol afterwards.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNNER = HERE / "run_mtsd_supervised.py"
PYTHON = Path(r"C:\Users\fridge\anaconda3\envs\mtsd-base\python.exe")
QUEUE_RECORD = REPO / "Results" / "MTSD-Runs" / "requested_yolo_matrix_20260815.json"
GROUP = "requested-yolo-matrix-20260815-s42"


JOBS = [
    {"model": "yolo11n", "augmentation": "strong", "dataset": "augmented", "image_size": 640,
     "name": "FINAL_yolo11n_mtsd_strong_img640_pb32_eb64_e100_adamw_s42", "final": True},
    {"model": "yolo11n", "augmentation": "strong", "dataset": "augmented", "image_size": 960,
     "name": "FINAL_yolo11n_mtsd_strong_img960_pb16_eb64_e100_adamw_s42", "final": True},
    {"model": "yolo11s", "augmentation": "none", "dataset": "unaugmented", "image_size": 640,
     "name": "FINAL_yolo11s_mtsd_noaug_img640_pb32_eb64_e100_adamw_s42", "final": False},
    {"model": "yolo11s", "augmentation": "strong", "dataset": "augmented", "image_size": 640,
     "name": "FINAL_yolo11s_mtsd_strong_img640_pb32_eb64_e100_adamw_s42", "final": True},
    {"model": "yolo11m", "augmentation": "strong", "dataset": "augmented", "image_size": 640,
     "name": "FINAL_yolo11m_mtsd_strong_img640_pb32_eb64_e100_adamw_s42", "final": True},
    {"model": "yolo11m", "augmentation": "strong", "dataset": "augmented", "image_size": 960,
     "name": "FINAL_yolo11m_mtsd_strong_img960_pb16_eb64_e100_adamw_s42", "final": True},
    {"model": "yolo12n", "augmentation": "strong", "dataset": "augmented", "image_size": 960,
     "name": "FINAL_yolo12n_mtsd_strong_img960_pb16_eb64_e100_adamw_s42", "final": True},
    {"model": "yolo12n", "augmentation": "strong", "dataset": "augmented", "image_size": 1280,
     "name": "FINAL_yolo12n_mtsd_strong_img1280_pb8_eb64_e100_adamw_s42", "final": True},
    {"model": "yolo12s", "augmentation": "strong", "dataset": "augmented", "image_size": 1280,
     "name": "FINAL_yolo12s_mtsd_strong_img1280_pb8_eb64_e100_adamw_s42", "final": True},
    {"model": "yolo26n", "augmentation": "strong", "dataset": "augmented", "image_size": 640,
     "name": "FINAL_yolo26n_mtsd_strong_img640_pb32_eb64_e100_adamw_s42", "final": True},
    {"model": "yolo26n", "augmentation": "strong", "dataset": "augmented", "image_size": 960,
     "name": "FINAL_yolo26n_mtsd_strong_img960_pb16_eb64_e100_adamw_s42", "final": True},
    {"model": "yolo26s", "augmentation": "none", "dataset": "unaugmented", "image_size": 640,
     "name": "FINAL_yolo26s_mtsd_noaug_img640_pb32_eb64_e100_adamw_s42", "final": False},
    {"model": "yolo26s", "augmentation": "strong", "dataset": "augmented", "image_size": 640,
     "name": "FINAL_yolo26s_mtsd_strong_img640_pb32_eb64_e100_adamw_s42", "final": True},
    {"model": "yolo26m", "augmentation": "strong", "dataset": "augmented", "image_size": 640,
     "name": "FINAL_yolo26m_mtsd_strong_img640_pb32_eb64_e100_adamw_s42", "final": True},
    {"model": "yolo26m", "augmentation": "strong", "dataset": "augmented", "image_size": 960,
     "name": "FINAL_yolo26m_mtsd_strong_img960_pb16_eb64_e100_adamw_s42", "final": True},
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def family_for(model: str) -> str:
    return "YOLO" + model[4:-1].upper()


def target_for(job: dict) -> Path:
    return REPO / "Results" / "MTSD-Runs" / f"{family_for(job['model'])}-MTSD" / job["name"]


def command_for(job: dict) -> list[str]:
    args = [
        str(PYTHON), str(RUNNER),
        "--dataset-variant", job["dataset"],
        "--offline-augmentation", job["augmentation"],
        "--models", job["model"],
        "--epochs", "100",
        "--output-name", job["name"],
        "--wandb-mode", "online",
        "--wandb-group", GROUP,
        "--device", "cuda",
        "--defer-final-test-evaluation",
        "--reuse-prepared-validation",
        "--require-unified-eval",
        "--image-size", str(job["image_size"]),
        "--yolo-online-augmentation", "disabled",
    ]
    if job["final"]:
        args.insert(2, "--final")
    return args


def read_record(target: Path) -> dict | None:
    path = target / "run_record.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "invalid_record"}


def write_queue(payload: dict) -> None:
    QUEUE_RECORD.parent.mkdir(parents=True, exist_ok=True)
    temporary = QUEUE_RECORD.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, QUEUE_RECORD)


def preflight() -> int:
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    rows = []
    for index, job in enumerate(JOBS, start=1):
        target = target_for(job)
        record = read_record(target)
        rows.append({
            "index": index,
            "model": job["model"],
            "augmentation": job["augmentation"],
            "image_size": job["image_size"],
            "name": job["name"],
            "target_exists": target.exists(),
            "record_status": record.get("status") if record else None,
            "action": "skip_completed" if record and record.get("status") == "completed"
            else "blocked_existing_target" if target.exists() else "train",
        })
    print(json.dumps({"job_count": len(rows), "jobs": rows}, indent=2))
    return 0


def launch() -> int:
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    state = {"started_at": now(), "status": "running", "jobs": []}
    write_queue(state)
    for index, job in enumerate(JOBS, start=1):
        target = target_for(job)
        record = read_record(target)
        entry = {"index": index, **job, "run_dir": str(target), "started_at": now()}
        if record and record.get("status") == "completed":
            entry.update(status="skipped_completed", finished_at=now())
            state["jobs"].append(entry)
            write_queue(state)
            print(f"[{index}/{len(JOBS)}] skip completed {job['name']}", flush=True)
            continue
        if target.exists():
            entry.update(
                status="blocked_existing_target",
                finished_at=now(),
                note="Immutable target exists without a completed run_record.json; no overwrite attempted.",
            )
            state["jobs"].append(entry)
            write_queue(state)
            print(f"[{index}/{len(JOBS)}] blocked existing target {target}", flush=True)
            continue
        entry["status"] = "running"
        state["jobs"].append(entry)
        write_queue(state)
        print(f"[{index}/{len(JOBS)}] start {job['name']}", flush=True)
        result = subprocess.run(command_for(job), cwd=REPO, check=False)
        entry.update(
            status="completed" if result.returncode == 0 else "failed",
            returncode=result.returncode,
            finished_at=now(),
        )
        write_queue(state)
        if result.returncode != 0:
            print(f"[{index}/{len(JOBS)}] failed with return code {result.returncode}; continuing", flush=True)
    state.update(status="completed", finished_at=now())
    write_queue(state)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.preflight_only and args.launch:
        parser.error("choose only one of --preflight-only or --launch")
    return preflight() if args.preflight_only or not args.launch else launch()


if __name__ == "__main__":
    raise SystemExit(main())
