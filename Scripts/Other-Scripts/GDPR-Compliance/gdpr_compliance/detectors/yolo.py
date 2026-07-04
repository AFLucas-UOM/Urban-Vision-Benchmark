"""Optional Ultralytics-YOLO detector for user-supplied weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..detections import Detection
from .base import BaseDetector


class YoloDetector(BaseDetector):
    """Plug in any Ultralytics YOLO weights without touching the pipeline.

    Example: ``redact.py preview --yolo-weights plate.pt --yolo-label
    licence_plate``. Requires the optional ``ultralytics`` package.
    """

    name = "yolo"

    def __init__(self, weights: Path, label: str = "yolo_region",
                 confidence: float = 0.30):
        from ultralytics import YOLO  # imported lazily; optional dependency

        self.label = label
        self.confidence = confidence
        self._model = YOLO(str(weights))

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        results = self._model.predict(image_bgr, conf=self.confidence, verbose=False)
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                x0, y0, x1, y1 = (float(v) for v in box.xyxy[0])
                detections.append(Detection(int(x0), int(y0), int(x1 - x0),
                                            int(y1 - y0), self.label,
                                            float(box.conf[0]), self.name,
                                            self.pad_fraction))
        return detections
