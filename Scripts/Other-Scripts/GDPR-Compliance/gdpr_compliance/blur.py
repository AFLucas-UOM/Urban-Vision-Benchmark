"""Redaction primitives: strong size-scaled Gaussian blur or pixelation."""

from __future__ import annotations

import cv2
import numpy as np

from .detections import Detection


def _blur_roi(roi: np.ndarray) -> np.ndarray:
    """Gaussian blur whose strength scales with the region size.

    The kernel spans roughly half of the region's smaller side (with a floor
    for tiny regions), which renders both distant plates and close-up faces
    unrecognisable while staying proportionate.
    """
    k = max(15, (min(roi.shape[:2]) // 2) | 1)  # odd kernel, ~1/2 of the region
    return cv2.GaussianBlur(roi, (k, k), 0)


def _pixelate_roi(roi: np.ndarray, cells: int = 8) -> np.ndarray:
    """Mosaic pixelation: downscale to ``cells`` blocks and scale back up."""
    height, width = roi.shape[:2]
    small = cv2.resize(roi, (max(1, min(cells, width)), max(1, min(cells, height))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def redact(image_bgr: np.ndarray, detections: list[Detection],
           method: str = "gaussian") -> np.ndarray:
    """Return a copy of the image with every (padded) region redacted."""
    if method not in {"gaussian", "pixelate"}:
        raise ValueError(f"Unknown redaction method: {method}")

    result = image_bgr.copy()
    height, width = result.shape[:2]
    for det in detections:
        x0, y0 = max(0, det.x), max(0, det.y)
        x1, y1 = min(width, det.x + det.w), min(height, det.y + det.h)
        if x1 <= x0 or y1 <= y0:
            continue
        roi = result[y0:y1, x0:x1]
        result[y0:y1, x0:x1] = _blur_roi(roi) if method == "gaussian" else _pixelate_roi(roi)
    return result


def annotate(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw labelled boxes (for the before-side of comparison sheets)."""
    canvas = image_bgr.copy()
    thickness = max(2, min(canvas.shape[:2]) // 400)
    for det in detections:
        cv2.rectangle(canvas, (det.x, det.y), (det.x + det.w, det.y + det.h),
                      (60, 60, 230), thickness)
        cv2.putText(canvas, f"{det.label} {det.score:.2f}",
                    (det.x, max(0, det.y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, thickness / 3.2, (60, 60, 230),
                    max(1, thickness - 1), cv2.LINE_AA)
    return canvas


def side_by_side(before: np.ndarray, after: np.ndarray, target_width: int = 1800,
                 gap: int = 6) -> np.ndarray:
    """Compose a before/after comparison sheet, resized for quick review."""
    height, width = before.shape[:2]
    panel_width = (target_width - gap) // 2
    scale = panel_width / width
    size = (panel_width, max(1, int(height * scale)))
    left = cv2.resize(before, size, interpolation=cv2.INTER_AREA)
    right = cv2.resize(after, size, interpolation=cv2.INTER_AREA)
    divider = np.full((size[1], gap, 3), 255, dtype=np.uint8)
    return np.hstack([left, divider, right])
