"""Face detection via OpenCV's YuNet ONNX model (Haar cascade fallback)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..detections import Detection
from .base import BaseDetector, scaled_bgr

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


class FaceDetector(BaseDetector):
    """Human faces via YuNet (Wu et al., libfacedetection).

    Small (~230 KB), CPU-fast and far more robust than Haar cascades for
    in-the-wild street photography. Falls back to the (much weaker) Haar
    cascade only when the ONNX file is missing, and says so loudly.
    """

    name = "faces"
    label = "face"
    pad_fraction = 0.20

    def __init__(self, score_threshold: float = 0.6, max_side: int = 1600):
        self.max_side = max_side
        self.score_threshold = score_threshold
        self._yunet = None
        self._haar = None
        if YUNET_MODEL.exists():
            self._yunet = cv2.FaceDetectorYN.create(
                str(YUNET_MODEL), "", (320, 320),
                score_threshold=score_threshold, nms_threshold=0.35,
            )
        else:  # pragma: no cover - depends on local model availability
            print(f"[faces] YuNet model missing ({YUNET_MODEL}).\n"
                  f"        Download it from {YUNET_URL}\n"
                  f"        Falling back to the (much weaker) Haar cascade.")
            self._haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        small, scale = scaled_bgr(image_bgr, self.max_side)

        detections: list[Detection] = []
        if self._yunet is not None:
            self._yunet.setInputSize((small.shape[1], small.shape[0]))
            _, faces = self._yunet.detect(small)
            for face in (faces if faces is not None else []):
                x, y, w, h = (float(v) * scale for v in face[:4])
                detections.append(Detection(int(x), int(y), int(w), int(h),
                                            self.label, float(face[14]),
                                            self.name, self.pad_fraction))
        else:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            for (x, y, w, h) in self._haar.detectMultiScale(gray, 1.1, 5, minSize=(24, 24)):
                detections.append(Detection(int(x * scale), int(y * scale),
                                            int(w * scale), int(h * scale),
                                            self.label, 0.5, f"{self.name}-haar",
                                            self.pad_fraction))
        return detections
