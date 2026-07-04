"""
utils.py — Shared utilities for the PromptDetect evaluation tool.

Covers: visualisation, mask-to-box conversion, detection filtering,
DataFrame creation, multi-format export, logging helpers, and image I/O.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_dir: str = "logs") -> logging.Logger:
    """Create a logger that writes to both a dated log file and the console."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("promptdetect")
    if logger.handlers:          # avoid duplicate handlers on reload
        return logger
    logger.setLevel(logging.DEBUG)

    log_file = Path(log_dir) / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def log_run(
    logger: logging.Logger,
    prompt: str,
    image_name: str,
    num_detections: int,
    runtime: Dict[str, float],
) -> None:
    logger.info(
        "run | prompt=%r | image=%r | detections=%d | "
        "pre=%.3fs | inf=%.3fs | post=%.3fs | total=%.3fs",
        prompt, image_name, num_detections,
        runtime.get("preprocess", 0),
        runtime.get("inference", 0),
        runtime.get("postprocess", 0),
        runtime.get("total", 0),
    )


# ---------------------------------------------------------------------------
# Colour utilities
# ---------------------------------------------------------------------------

def generate_colors(n: int) -> List[Tuple[int, int, int]]:
    """Return *n* visually distinct BGR colours."""
    colors: List[Tuple[int, int, int]] = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))          # cv2 hue ∈ [0, 180)
        hsv = np.uint8([[[hue, 220, 220]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_detections(
    image: np.ndarray,
    boxes: List[List[float]],
    labels: List[str],
    scores: List[float],
    masks: Optional[List[Optional[np.ndarray]]] = None,
    show_masks: bool = False,
    show_scores: bool = True,
    box_color: Tuple[int, int, int] = (0, 220, 80),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """
    Render bounding boxes (and optionally semi-transparent masks) on a copy of *image*.

    Always draws boxes; masks are drawn only when show_masks=True and masks is provided.
    The image is expected to be an RGB uint8 numpy array.
    """
    result = image.copy()
    n = len(boxes)
    colors = generate_colors(n)

    # -- optional mask overlay --
    if show_masks and masks:
        overlay = result.copy()
        for mask, color in zip(masks, colors):
            if mask is not None and mask.any():
                overlay[mask.astype(bool)] = color
        result = cv2.addWeighted(result, 0.55, overlay, 0.45, 0)

    # -- bounding boxes + labels --
    for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = map(int, box)
        color = colors[idx % len(colors)] if colors else box_color

        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        text = f"{label}: {score:.2f}" if (show_scores and score > 0) else str(label)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        # filled label background
        cv2.rectangle(result, (x1, max(y1 - th - 10, 0)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(result, text, (x1 + 3, max(y1 - 4, th)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)

        # small detection-index badge
        cv2.putText(result, f"#{idx + 1}", (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1, cv2.LINE_AA)

    return result


# ---------------------------------------------------------------------------
# Detection filtering
# ---------------------------------------------------------------------------

def filter_detections(
    boxes: List[List[float]],
    scores: List[float],
    labels: List[str],
    masks: Optional[List] = None,
    conf_threshold: float = 0.25,
    min_area: float = 100.0,
    max_area: float = float("inf"),
    max_detections: int = 100,
) -> Tuple[List, List, List, List]:
    """
    Keep only detections that pass confidence, area, and count constraints.
    Returns (boxes, scores, labels, masks) — all lists have matching lengths.
    """
    out_b, out_s, out_l, out_m = [], [], [], []
    masks = masks or []

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if score < conf_threshold:
            continue
        x1, y1, x2, y2 = box
        area = abs((x2 - x1) * (y2 - y1))
        if area < min_area or area > max_area:
            continue
        out_b.append(box)
        out_s.append(score)
        out_l.append(label)
        out_m.append(masks[i] if i < len(masks) else None)
        if len(out_b) >= max_detections:
            break

    return out_b, out_s, out_l, out_m


# ---------------------------------------------------------------------------
# Pandas DataFrame
# ---------------------------------------------------------------------------

def create_detection_dataframe(
    boxes: List[List[float]],
    scores: List[float],
    labels: List[str],
    prompt: str,
    include_confidence: bool = True,
) -> pd.DataFrame:
    """
    Build a tidy DataFrame from raw detection outputs.

    When *include_confidence* is False (generative VLMs with no real per-box
    score), the Confidence column is omitted entirely rather than filled with a
    misleading constant 1.0.
    """
    records = []
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        record = {
            "Detection ID": i + 1,
            "Prompt": prompt,
            "Label": label,
        }
        if include_confidence:
            record["Confidence"] = round(float(score), 4)
        record.update({
            "Xmin": int(x1),
            "Ymin": int(y1),
            "Xmax": int(x2),
            "Ymax": int(y2),
            "Width": int(w),
            "Height": int(h),
            "Area": int(w * h),
        })
        records.append(record)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def export_csv(df: pd.DataFrame, path: str) -> str:
    _ensure_dir(path)
    df.to_csv(path, index=False)
    return path


def export_json(data: Any, path: str) -> str:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def export_coco_json(
    detections_per_image: List[Dict],
    path: str,
    category_name: str = "detection",
) -> str:
    """
    Serialise detections in COCO object-detection format.
    Each item in detections_per_image must have keys:
      filename, width, height, boxes (list of [x1,y1,x2,y2]), scores.
    """
    coco: Dict[str, Any] = {
        "info": {
            "description": "SAM 3 Evaluation Results",
            "version": "1.0",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": category_name, "supercategory": "object"}],
    }

    ann_id = 1
    for img_id, item in enumerate(detections_per_image, start=1):
        coco["images"].append({
            "id": img_id,
            "file_name": item.get("filename", f"image_{img_id}"),
            "width": item.get("width", 0),
            "height": item.get("height", 0),
        })
        boxes = item.get("boxes", [])
        scores = item.get("scores", [0.0] * len(boxes))
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [int(x1), int(y1), int(w), int(h)],
                "area": int(w * h),
                "score": float(score),
                "iscrowd": 0,
            })
            ann_id += 1

    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
    return path


def export_yolo(
    boxes: List[List[float]],
    image_width: int,
    image_height: int,
    path: str,
    class_id: int = 0,
) -> str:
    """Write YOLO-format labels (normalised cx cy w h per line)."""
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for box in boxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2 / image_width
            cy = (y1 + y2) / 2 / image_height
            bw = (x2 - x1) / image_width
            bh = (y2 - y1) / image_height
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    return path


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image_from_url(url: str) -> np.ndarray:
    """Fetch an image from *url* and return an RGB uint8 numpy array."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PromptDetect/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img)
    except Exception as exc:
        raise ValueError(f"Could not load image from URL: {exc}") from exc
