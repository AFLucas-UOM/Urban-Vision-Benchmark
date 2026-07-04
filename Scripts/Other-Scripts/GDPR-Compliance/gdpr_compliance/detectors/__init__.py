"""Detector registry and backend resolution.

Two backends are supported:

- ``sam31``   - SAM 3.1 open-vocabulary detection (faces + plates + QR via
  text prompts, one shared model) plus OpenCV's exact QR detector.
- ``classic`` - dedicated per-category detectors (YuNet faces, ONNX plate
  model, OpenCV QR).

``--backend auto`` picks ``sam31`` when the ``sam3`` package is importable
in the current environment and falls back to ``classic`` with an explicit
notice otherwise - never silently.
"""

from __future__ import annotations

from .base import BaseDetector
from .faces import FaceDetector
from .plates import HaarPlateDetector, PlateDetector
from .qrcodes import QRCodeDetector
from .sam31 import Sam31Detector, sam31_available
from .yolo import YoloDetector

DETECTOR_REGISTRY: dict[str, type[BaseDetector]] = {
    FaceDetector.name: FaceDetector,
    PlateDetector.name: PlateDetector,
    HaarPlateDetector.name: HaarPlateDetector,
    QRCodeDetector.name: QRCodeDetector,
    Sam31Detector.name: Sam31Detector,
}

BACKEND_DEFAULTS: dict[str, list[str]] = {
    "sam31": ["sam31", "qrcodes"],
    "classic": ["faces", "plates", "qrcodes"],
}


def resolve_backend(requested: str) -> str:
    """Resolve ``auto``/``sam31``/``classic`` to a concrete backend."""
    if requested == "classic":
        return "classic"
    if requested == "sam31":
        if not sam31_available():
            raise SystemExit(
                "--backend sam31 requested but the 'sam3' package is not "
                "importable here. Activate the mtsd-base conda env first "
                "(conda activate mtsd-base) or use --backend classic."
            )
        return "sam31"
    if requested == "auto":
        if sam31_available():
            return "sam31"
        print("[backend] SAM 3.1 not importable in this environment - using the "
              "classical detectors (YuNet / plate ONNX / OpenCV QR).\n"
              "[backend] For SAM 3.1: conda activate mtsd-base, then rerun.")
        return "classic"
    raise SystemExit(f"Unknown backend '{requested}' (auto | sam31 | classic).")


def build_detectors(names: list[str], sam_options: dict | None = None) -> list[BaseDetector]:
    """Instantiate the requested detectors, failing early on unknown names."""
    unknown = [n for n in names if n not in DETECTOR_REGISTRY]
    if unknown:
        raise SystemExit(
            f"Unknown detector(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(DETECTOR_REGISTRY))}"
        )
    detectors: list[BaseDetector] = []
    for name in names:
        if name == Sam31Detector.name:
            detectors.append(Sam31Detector(**(sam_options or {})))
        else:
            detectors.append(DETECTOR_REGISTRY[name]())
    return detectors
