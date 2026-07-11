"""Run PromptDetect models over a ground-truth dataset, one prompt at a time.

Reuses the existing ``backend.DetectionBackend`` unchanged: models are loaded
sequentially (loading a new model frees the previous one, so GPU memory is
never double-booked), and every (model, prompt, image) triple produces zero
or more prediction rows in a flat, CSV-ready format.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from PIL import Image

import config  # noqa: F401  (sys.path side effect for `backend`)

ProgressCb = Callable[[float, str], None]
CombinationCb = Callable[[str, str, list[dict], dict], None]


def run_models(
    gt: dict,
    prompts: list[str],
    model_labels: list[str],
    conf_threshold: float,
    max_detections: int,
    progress: ProgressCb | None = None,
    on_combination_complete: CombinationCb | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return (prediction_rows, model_statuses)."""
    from backend import DetectionBackend

    notify = progress or (lambda fraction, message: print(f"[{fraction:5.1%}] {message}"))
    backend = DetectionBackend()
    records = gt["records"]
    predictions: list[dict] = []
    statuses: list[dict] = []
    total_steps = max(1, len(model_labels) * len(records) * max(1, len(prompts)))
    step = 0

    for model_label in model_labels:
        notify(step / total_steps, f"Loading {model_label}...")
        load_started = time.perf_counter()
        status = backend.load(model_label)
        load_ms = (time.perf_counter() - load_started) * 1000
        statuses.append({"model": model_label, "model_load_ms": round(load_ms, 1), **status})
        if not status.get("ok"):
            notify(step / total_steps, f"SKIPPED {model_label}: {status.get('error')}")
            step += len(records) * max(1, len(prompts))
            continue

        # A prompt combination becomes durable as soon as its image loop ends;
        # the model stays loaded across prompts, so there is no reload penalty.
        for prompt in prompts:
            combination_rows: list[dict] = []
            for record in records:
                step += 1
                notify(step / total_steps, f"{model_label} | {prompt} | {record['image_id']}")
                try:
                    image = np.array(Image.open(record["image_path"]).convert("RGB"))
                except Exception as exc:
                    statuses.append({"model": model_label, "ok": False,
                                     "error": f"unreadable image {record['image_id']}: {exc}"})
                    continue
                started = time.perf_counter()
                result = backend.predict(
                    image=image, text_prompt=prompt,
                    conf_threshold=conf_threshold, max_detections=max_detections,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                masks = result.get("masks")
                if masks is None or len(masks) != len(result["boxes"]):
                    masks = [None] * len(result["boxes"])
                for box, score, label, mask in zip(
                    result["boxes"], result["scores"], result["labels"],
                    masks,
                ):
                    row = {
                        "model": model_label,
                        "prompt": prompt,
                        "image_id": record["image_id"],
                        "image_path": str(record["image_path"]),
                        "x0": round(float(box[0]), 2), "y0": round(float(box[1]), 2),
                        "x1": round(float(box[2]), 2), "y1": round(float(box[3]), 2),
                        "score": round(float(score), 4),
                        "predicted_label": label,
                        "has_mask": mask is not None,
                        "has_confidence": result.get("has_confidence", True),
                        "inference_ms": round(elapsed_ms, 1),
                    }
                    predictions.append(row); combination_rows.append(row)
                if not result["boxes"]:
                    # keep a zero-detection marker so per-image timing survives
                    row = {
                        "model": model_label, "prompt": prompt,
                        "image_id": record["image_id"], "image_path": str(record["image_path"]),
                        "x0": "", "y0": "", "x1": "", "y1": "", "score": "",
                        "predicted_label": "<no detections>", "has_mask": False,
                        "has_confidence": result.get("has_confidence", True),
                        "inference_ms": round(elapsed_ms, 1),
                    }
                    predictions.append(row); combination_rows.append(row)
            if on_combination_complete:
                on_combination_complete(model_label, prompt, combination_rows,
                                        {"model_load_ms": round(load_ms, 1)})

    # release the last model
    try:
        if backend._loaded is not None:  # noqa: SLF001 - backend has no public close
            backend._loaded.engine.close()
    except Exception:
        pass
    return predictions, statuses


def prediction_boxes(prediction_rows: list[dict], model: str, prompt: str) -> dict[str, list[dict]]:
    """Group real (non-marker) prediction rows by image for one model+prompt."""
    grouped: dict[str, list[dict]] = {}
    for row in prediction_rows:
        if row["model"] != model or row["prompt"] != prompt or row["x0"] == "":
            continue
        grouped.setdefault(row["image_id"], []).append({
            "x0": float(row["x0"]), "y0": float(row["y0"]),
            "x1": float(row["x1"]), "y1": float(row["y1"]),
            "score": float(row["score"]) if row["score"] != "" else 0.0,
            "prompt": prompt, "predicted_label": row.get("predicted_label", prompt),
            "has_confidence": row.get("has_confidence", True),
        })
    return grouped
