"""Detector interface and shared image-scaling helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from ..detections import Detection


class BaseDetector(ABC):
    """Interface every sensitive-region detector implements.

    Detectors are constructed **once** per pipeline run (models are loaded in
    ``__init__``) and then called for every image, so per-image work must not
    reload weights. ``detect`` receives a BGR array in display orientation and
    returns :class:`Detection` boxes in that image's pixel coordinates, each
    carrying the pad fraction the pipeline should apply before redaction.
    """

    name: str = "base"
    label: str = "region"
    pad_fraction: float = 0.15

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        """Return detections in pixel coordinates of ``image_bgr``."""


def scaled_bgr(image_bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Downscale for detection speed; return (image, scale factor to original)."""
    height, width = image_bgr.shape[:2]
    scale = max(height, width) / max_side if max(height, width) > max_side else 1.0
    if scale > 1.0:
        image_bgr = cv2.resize(image_bgr, (int(width / scale), int(height / scale)),
                               interpolation=cv2.INTER_AREA)
    return image_bgr, scale


def scaled_gray(image_bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Downscaled grayscale variant of :func:`scaled_bgr`."""
    small, scale = scaled_bgr(image_bgr, max_side)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), scale
