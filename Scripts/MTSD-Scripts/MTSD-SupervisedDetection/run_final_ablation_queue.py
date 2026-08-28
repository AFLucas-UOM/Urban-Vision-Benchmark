#!/usr/bin/env python3
"""Run the remaining frozen MTSD ablations sequentially.

This queue is intentionally conservative: each job has an immutable output
name, completed runs are skipped, no final-test evaluation is requested, and
only one GPU training process is launched at a time.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNNER = HERE / "run_mtsd_supervised.py"
PYTHON = Path(os.environ.get("MTSD_RUNNER_PYTHON", sys.executable))
QUEUE_RECORD = REPO / "Results" / "MTSD-Runs" / "final_ablation_queue_20260812.json"


JOBS = [
    {"model": "yolo26n", "variant": "strong", "dataset": "augmented", "image_size": 1280,
     "name": "FINAL_yolo26n_mtsd_strong_img1280_pb8_eb64_e100_adamw_s42", "final": True},
    {"model": "yolo11n", "variant": "strong", "dataset": "augmented", "image_size": 1280,
     "name": "FINAL_yolo11n_mtsd_strong_img1280_pb8_eb64_e100_adamw_s42", "final": True},
    {"model": "yolo12n", "variant": "strong", "dataset": "augmented", "image_size": 640,
     "name": "FINAL_yolo12n_mtsd_strong_img640_pb32_eb64_e100_adamw_s42", "final": True},
    {"model": "yolo12s", "variant": "strong", "dataset": "augmented", "image_size": 640,
     "name": "FINAL_yolo12s_mtsd_strong_img640_pb32_eb64_e100_adamw_s42", "final": True},
    {"model": "yolo12m", "variant": "strong", "dataset": "augmented", "image_size": 640,
     "name": "FINAL_yolo12m_mtsd_strong_img640_pb32_eb64_e100_adamw_s42", "final": True},
    {"model": "rfdetr-s", "variant": "strong", "dataset": "augmented", "image_size": 512,
     "name": "FINAL_rfdetrs_mtsd_strong_img512_pb32_eb32_e100_adamw_s42", "final": True},
    {"model": "rfdetr-m", "variant": "strong", "dataset": "augmented", "image_size": 576,
     "name": "FINAL_rfdetrm_mtsd_strong_img576_pb32_eb32_e100_adamw_s42", "final": True},
    {"model": "rfdetr-m", "variant": "none", "dataset": "unaugmented", "image_size": 576,
     "name": "FINAL_rfdetrm_mtsd_noaug_img576_pb32_eb32_e100_adamw_s42", "final": False},
    {"model": "yolo26s", "variant": "none", "dataset": "unaugmented", "image_size": 960,
     "name": "FINAL_yolo26s_mtsd_noaug_img960_pb16_eb64_e100_adamw_s42", "final": False},
    {"model": "yolo26s", "variant": "none", "dataset": "unaugmented", "image_size": 1280,
     "name": "FINAL_yolo26s_mtsd_noaug_img1280_pb8_eb64_e100_adamw_s42", "final": False},
    {"model": "yolo26s", "variant": "mild", "dataset": "augmented", "image_size": 960,
     "name": "FINAL_yolo26s_mtsd_good_img960_pb16_eb64_e100_adamw_s42", "final": True},
    {"model": "yolo12s", "variant": "none", "dataset": "unaugmented", "image_size": 640,
     "name": "FINAL_yolo12s_mtsd_noaug_img640_pb32_eb64_e100_adamw_s42", "final": False},
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_dir(job: dict) -> Path:
    family = "RF-DETR" if job["model"].startswith("rfdetr") else job["model"].upper().replace("N", "N").replace("S", "S").replace("M", "M")
    if job["model"].startswith("yolo"):
        family = job["model"].split("yolo", 1)[1][:-1].upper() if job["model"][-1] in "nsm" else job["model"].upper()
        family = family.replace("YOLO", "YOLO")
    return REPO / "Results" / "MTSD-Runs" / f"{family}-MTSD" / job["name"]


def family_for(model: str) -> str:
    if model.startswith("rfdetr"):
        return "RF-DETR"
    return "YOLO" + model[4:-1].upper()


def command(job: dict) -> list[str]:
    args = [
        str(PYTHON), str(RUNNER),
        "--dataset-variant", job["dataset"],
        "--offline-augmentation", job["variant"],
        "--models", job["model"],
        "--epochs", "100",
        "--output-name", job["name"],
        "--wandb-mode", "online",
        "--wandb-group", "final-mtsd-ablation-s42",
        "--device", "cuda",
        "--defer-final-test-evaluation",
        "--reuse-prepared-validation",
        "--require-unified-eval",
    ]
    if job["model"].startswith("yolo"):
        args.extend(["--image-size", str(job["image_size"]), "--yolo-online-augmentation", "disabled"])
    if job["final"]:
        args.insert(2, "--final")
    return args


def read_record(path: Path) -> dict | None:
    record = path / "run_record.json"
    if not record.is_file():
        return None
    try:
        return json.loads(record.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "invalid_record"}


def write_queue(payload: dict) -> None:
    QUEUE_RECORD.parent.mkdir(parents=True, exist_ok=True)
    temporary = QUEUE_RECORD.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, QUEUE_RECORD)


def main() -> int:
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    state = {"started_at": now(), "status": "running", "jobs": []}
    write_queue(state)
    for index, job in enumerate(JOBS, start=1):
        target = REPO / "Results" / "MTSD-Runs" / f"{family_for(job['model'])}-MTSD" / job["name"]
        record = read_record(target)
        entry = {"index": index, **job, "run_dir": str(target), "started_at": now()}
        if record and record.get("status") == "completed":
            entry.update(status="skipped_completed")
            state["jobs"].append(entry); write_queue(state)
            print(f"[{index}/{len(JOBS)}] skip completed {job['name']}", flush=True)
            continue
        if target.exists():
            print(f"[{index}/{len(JOBS)}] waiting for existing active target {target}", flush=True)
            deadline = time.time() + 8 * 60 * 60
            while time.time() < deadline:
                record = read_record(target)
                if record is not None:
                    break
                time.sleep(30)
            record = read_record(target)
            if record and record.get("status") == "completed":
                entry.update(status="skipped_completed", finished_at=now())
            else:
                entry.update(status="blocked_existing_directory", finished_at=now(),
                             note="Immutable target exists without a completed run_record.json")
            state["jobs"].append(entry); write_queue(state)
            print(f"[{index}/{len(JOBS)}] {entry['status']} {target}", flush=True)
            continue
        entry["status"] = "running"
        state["jobs"].append(entry); write_queue(state)
        print(f"[{index}/{len(JOBS)}] start {job['name']}", flush=True)
        result = subprocess.run(command(job), cwd=REPO, check=False)
        entry.update(status="completed" if result.returncode == 0 else "failed",
                     returncode=result.returncode, finished_at=now())
        write_queue(state)
        if result.returncode != 0:
            print(f"[{index}/{len(JOBS)}] failed with return code {result.returncode}; continuing", flush=True)
    state.update(status="completed", finished_at=now())
    write_queue(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
