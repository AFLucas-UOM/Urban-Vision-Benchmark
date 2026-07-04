"""QR code detection via OpenCV's built-in multi-code detector."""

from __future__ import annotations

import cv2
import numpy as np

from ..detections import Detection
from .base import BaseDetector, scaled_gray


class QRCodeDetector(BaseDetector):
    """QR codes via ``cv2.QRCodeDetector.detectMulti``.

    Precise on well-resolved, decodable codes and essentially free, which is
    why it stays in the default detector set even under the SAM 3.1 backend.
    """

    name = "qrcodes"
    label = "qr_code"
    pad_fraction = 0.10

    def __init__(self, max_side: int = 2400):
        self.max_side = max_side
        self._detector = cv2.QRCodeDetector()

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        gray, scale = scaled_gray(image_bgr, self.max_side)
        try:
            found, points = self._detector.detectMulti(gray)
        except cv2.error:  # pragma: no cover - malformed inputs
            return []
        detections: list[Detection] = []
        if found and points is not None:
            for quad in points:
                xs, ys = quad[:, 0] * scale, quad[:, 1] * scale
                x0, y0 = int(xs.min()), int(ys.min())
                detections.append(Detection(x0, y0, int(xs.max()) - x0,
                                            int(ys.max()) - y0,
                                            self.label, 0.5, self.name,
                                            self.pad_fraction))
        return detections
