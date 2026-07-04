"""Vehicle licence plate detection.

Primary: a local single-class YOLO-style ONNX model run through OpenCV DNN
(no Ultralytics / ONNX Runtime dependency). Legacy: OpenCV's Haar cascade,
kept as an explicitly named, clearly weaker alternative.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..detections import Detection
from .base import BaseDetector, scaled_gray

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
NUMBER_PLATE_MODEL = MODELS_DIR / "number_plate_model_2025.onnx"


class PlateDetector(BaseDetector):
    """Vehicle licence plates via the local 2025 ONNX detector.

    The model is a single-class YOLO-style detector exported to ONNX. It is run
    through OpenCV DNN to avoid adding a runtime dependency on Ultralytics or
    ONNX Runtime. If the ONNX file is not present, the detector falls back to
    the older Haar cascade so the CLI still works in a degraded mode.
    """

    name = "plates"
    label = "licence_plate"
    pad_fraction = 0.12

    def __init__(
        self,
        model_path: Path = NUMBER_PLATE_MODEL,
        input_size: int = 640,
        confidence: float = 0.60,
        nms_threshold: float = 0.45,
        min_aspect: float = 1.8,
        max_aspect: float = 7.0,
        max_area_fraction: float = 0.08,
    ):
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.max_area_fraction = max_area_fraction
        self._net = None
        self._fallback = None
        if self.model_path.exists():
            self._net = cv2.dnn.readNetFromONNX(str(self.model_path))
        else:  # pragma: no cover - depends on local model availability
            print(f"[plates] ONNX model missing ({self.model_path}).\n"
                  "         Falling back to OpenCV's older Haar plate cascade.")
            self._fallback = HaarPlateDetector()

    def _letterbox(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Resize with unchanged aspect ratio and pad to the model input size."""
        height, width = image_bgr.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        new_w, new_h = int(round(width * scale)), int(round(height * scale))
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas, scale, pad_x, pad_y

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        if self._fallback is not None:
            return self._fallback.detect(image_bgr)

        height, width = image_bgr.shape[:2]
        padded, scale, pad_x, pad_y = self._letterbox(image_bgr)
        blob = cv2.dnn.blobFromImage(
            padded, scalefactor=1 / 255.0, size=(self.input_size, self.input_size),
            swapRB=True, crop=False,
        )
        self._net.setInput(blob)
        output = self._net.forward()
        predictions = np.squeeze(output).T  # (8400, 5): cx, cy, w, h, conf

        boxes: list[list[int]] = []
        scores: list[float] = []
        for cx, cy, bw, bh, score in predictions:
            score = float(score)
            if score < self.confidence:
                continue
            x0 = (float(cx) - float(bw) / 2 - pad_x) / scale
            y0 = (float(cy) - float(bh) / 2 - pad_y) / scale
            x1 = (float(cx) + float(bw) / 2 - pad_x) / scale
            y1 = (float(cy) + float(bh) / 2 - pad_y) / scale
            x0 = max(0, min(width - 1, int(round(x0))))
            y0 = max(0, min(height - 1, int(round(y0))))
            x1 = max(0, min(width, int(round(x1))))
            y1 = max(0, min(height, int(round(y1))))
            box_w, box_h = x1 - x0, y1 - y0
            if box_w <= 0 or box_h <= 0:
                continue
            aspect = box_w / box_h
            area_fraction = (box_w * box_h) / (width * height)
            if not (self.min_aspect <= aspect <= self.max_aspect):
                continue
            if area_fraction > self.max_area_fraction:
                continue
            boxes.append([x0, y0, box_w, box_h])
            scores.append(score)

        keep = cv2.dnn.NMSBoxes(boxes, scores, self.confidence, self.nms_threshold)
        if len(keep) == 0:
            return []
        return [
            Detection(*boxes[int(i)], self.label, scores[int(i)], self.name,
                      self.pad_fraction)
            for i in np.array(keep).flatten()
        ]


class HaarPlateDetector(BaseDetector):
    """Legacy vehicle licence plate detector using OpenCV's Haar cascade.

    The bundled ``haarcascade_russian_plate_number`` targets EU-style
    rectangular plates but produces frequent false positives on sign text,
    roof lines, and other high-contrast rectangular regions. Kept only as a
    dependency-free point of comparison; prefer ``plates`` or ``sam31``.
    """

    name = "plates_haar"
    label = "licence_plate"
    pad_fraction = 0.15

    def __init__(self, max_side: int = 2000):
        self.max_side = max_side
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        gray, scale = scaled_gray(image_bgr, self.max_side)
        gray = cv2.equalizeHist(gray)
        hits = self._cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=6,
                                              minSize=(36, 12))
        return [
            Detection(int(x * scale), int(y * scale), int(w * scale), int(h * scale),
                      self.label, 0.5, self.name, self.pad_fraction)
            for (x, y, w, h) in hits
        ]
