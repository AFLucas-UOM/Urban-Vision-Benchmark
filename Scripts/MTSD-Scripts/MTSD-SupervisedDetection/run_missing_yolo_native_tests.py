#!/usr/bin/env python3
"""Run native held-out-test evaluation for completed MTSD YOLO checkpoints.

This is post-training evaluation only.  Each completed run record is updated
immediately after its test pass, making the job safe to resume after an
interruption.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def pending_runs() -> list[tuple[Path, dict]]:
    found: list[tuple[Path, dict]] = []
    for record_path in sorted((REPO / "Results" / "MTSD-Runs").rglob("run_record.json")):
        record = read_json(record_path)
        if (
            record.get("status") != "completed"
            or record.get("trainer") != "ultralytics"
            or record_path.parent.name.startswith("PROBE_")
            or record.get("native_metrics", {}).get("test")
        ):
            continue
        checkpoint = Path(record.get("checkpoint_best", ""))
        data = record.get("train_args", {}).get("data")
        if not checkpoint.is_file() or not data:
            raise FileNotFoundError(f"Incomplete evaluation inputs for {record_path.parent}")
        found.append((record_path, record))
    return found


def main() -> int:
    from ultralytics import YOLO

    runs = pending_runs()
    print(f"Discovered {len(runs)} pending native YOLO test evaluations.", flush=True)
    for index, (record_path, record) in enumerate(runs, start=1):
        run_dir = record_path.parent
        args = record["train_args"]
        checkpoint = Path(record["checkpoint_best"])
        output_dir = run_dir / "native_test_evaluation"
        print(f"[{index}/{len(runs)}] {run_dir.name}: starting", flush=True)
        started = time.perf_counter()
        model = YOLO(str(checkpoint))
        metrics = model.val(
            data=str(args["data"]),
            split="test",
            imgsz=int(args["imgsz"]),
            batch=int(args["batch"]),
            device="cuda",
            project=str(output_dir.parent),
            name=output_dir.name,
            exist_ok=True,
            plots=True,
        )
        elapsed = time.perf_counter() - started
        record.setdefault("native_metrics", {})["test"] = dict(
            getattr(metrics, "results_dict", {}) or {}
        )
        record["native_test_evaluation"] = {
            "split": "test",
            "checkpoint": str(checkpoint),
            "output_dir": str(output_dir),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "seconds": elapsed,
        }
        record["native_test_evaluation_seconds"] = elapsed
        record["test_evaluation_deferred"] = False
        write_json(record_path, record)
        summary = record["native_metrics"]["test"]
        print(
            f"[{index}/{len(runs)}] {run_dir.name}: "
            f"mAP50-95={summary.get('metrics/mAP50-95(B)', float('nan')):.4f}, "
            f"mAP50={summary.get('metrics/mAP50(B)', float('nan')):.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
