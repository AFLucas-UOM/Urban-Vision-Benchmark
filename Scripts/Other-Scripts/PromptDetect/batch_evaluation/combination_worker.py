#!/usr/bin/env python3
"""Isolated model/prompt/chunk inference worker.

The coordinator never imports or loads a model. Each worker loads one model,
runs one prompt over a bounded image slice, persists raw predictions, and
exits. Process exit is the reliable RAM/VRAM cleanup boundary on Windows.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dataset_loader import load_ground_truth
from persistence import write_csv, write_json
from prompt_runner import run_models

COSMOS_MAX_SIDE_ENV = "PROMPTDETECT_COSMOS_MAX_SIDE"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PromptDetect isolated combination worker")
    p.add_argument("--dataset", required=True, choices=("MDWD", "MTSD"))
    p.add_argument("--split", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--prompt-id", required=True)
    p.add_argument("--conf-threshold", required=True, type=float)
    p.add_argument("--max-detections", required=True, type=int)
    p.add_argument("--nms-iou-threshold", required=True, type=float)
    p.add_argument("--image-limit", type=int)
    p.add_argument("--image-start", required=True, type=int)
    p.add_argument("--image-end", required=True, type=int)
    p.add_argument("--output-dir", required=True, type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    try:
        gt = load_ground_truth(args.dataset, args.split)
        if args.image_limit is not None:
            gt["records"] = gt["records"][:args.image_limit]
        total = len(gt["records"])
        if not (0 <= args.image_start < args.image_end <= total):
            raise ValueError(
                f"Invalid image slice [{args.image_start}:{args.image_end}] for {total} records"
            )
        gt["records"] = gt["records"][args.image_start:args.image_end]

        last_bucket = -1

        def progress(fraction: float, message: str) -> None:
            nonlocal last_bucket
            bucket = min(20, int(fraction * 20))
            if bucket != last_bucket:
                last_bucket = bucket
                print(f"[{fraction:5.1%}] {message}", flush=True)

        predictions, statuses = run_models(
            gt,
            [args.prompt],
            [args.model],
            args.conf_threshold,
            args.max_detections,
            nms_iou_threshold=args.nms_iou_threshold,
            progress=progress,
        )
        load_status = statuses[0] if statuses else {
            "model": args.model, "ok": False, "error": "worker returned no model status"
        }
        if not load_status.get("ok"):
            raise RuntimeError(load_status.get("error") or f"Could not load {args.model}")
        rows = [{**row, "prompt_id": args.prompt_id} for row in predictions]
        write_csv(args.output_dir / "predictions.csv", rows)
        write_json(args.output_dir / "worker_status.json", {
            "status": "completed",
            "dataset": args.dataset,
            "split": args.split,
            "model": args.model,
            "prompt_id": args.prompt_id,
            "prompt": args.prompt,
            "image_start": args.image_start,
            "image_end": args.image_end,
            "n_images": len(gt["records"]),
            "n_prediction_rows": len(rows),
            "max_detections": args.max_detections,
            "nms_iou_threshold": args.nms_iou_threshold,
            "cosmos_max_side": int(__import__("os").getenv(COSMOS_MAX_SIDE_ENV, "1536")),
            "model_status": load_status,
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        return 0
    except Exception as exc:
        write_json(args.output_dir / "worker_status.json", {
            "status": "failed",
            "dataset": args.dataset,
            "model": args.model,
            "prompt_id": args.prompt_id,
            "image_start": args.image_start,
            "image_end": args.image_end,
            "error": f"{type(exc).__name__}: {exc}",
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
