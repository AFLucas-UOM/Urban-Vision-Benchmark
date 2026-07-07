"""SAM 3.1 open-vocabulary detector (Meta's official ``sam3`` package).

SAM 3 / 3.1 accept free-text concept prompts and return boxes + scores in a
single pass, so faces, licence plates, and QR codes are detected by prompting
the *same* loaded model three times per image. The image is encoded once
(``set_image``) and each category adds only a cheap prompt pass.

Requirements and limitations (explicit by design):

- Needs the ``sam3`` package + PyTorch, i.e. the **mtsd-base** conda env
  (the checkpoint downloads from the gated ``facebook/sam3.1`` HF repo on
  first use; access must already be granted).
- Prompted concept detection is open-vocabulary, not a dedicated
  face/plate/QR model: recall depends on the prompt. The default prompts
  below were chosen for street-scene imagery; tune per category with
  ``--sam-prompt label=prompt`` if needed.
- QR codes are the weakest category for a concept detector, so the pipeline
  keeps OpenCV's exact QR detector in the default set alongside SAM 3.1
  rather than silently relying on the prompt.
"""

from __future__ import annotations

import importlib.util
from contextlib import nullcontext

import numpy as np

from ..detections import Detection
from .base import BaseDetector

#: label -> (text prompt, pad fraction, min confidence)
DEFAULT_CATEGORIES: dict[str, tuple[str, float, float]] = {
    "face": ("human face", 0.20, 0.35),
    "licence_plate": ("vehicle licence plate", 0.12, 0.30),
    "qr_code": ("QR code sticker", 0.10, 0.40),
}


def sam31_available() -> bool:
    """True when the ``sam3`` package and torch are importable."""
    return (importlib.util.find_spec("sam3") is not None
            and importlib.util.find_spec("torch") is not None)


class Sam31Detector(BaseDetector):
    """All three sensitive categories through one SAM 3.1 model.

    The model is built once in ``__init__`` (checkpoint via
    ``download_ckpt_from_hf`` unless an explicit path is given) and reused
    for every image; ``detect`` encodes the image once and runs one text
    prompt per category.
    """

    name = "sam31"
    label = "sam31"

    def __init__(
        self,
        version: str = "sam3.1",
        device: str | None = None,
        checkpoint: str | None = None,
        categories: dict[str, tuple[str, float, float]] | None = None,
        confidence_floor: float = 0.05,
    ):
        if not sam31_available():
            raise RuntimeError(
                "SAM 3.1 backend requested but the 'sam3' package (or torch) is "
                "not importable in this environment.\n"
                "Run inside the mtsd-base conda env, e.g.:\n"
                "  conda activate mtsd-base\n"
                "  python Scripts/Other-Scripts/GDPR-Compliance/redact.py preview\n"
                "or select the classical detectors with --backend classic."
            )

        import torch
        from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
        from sam3.model.sam3_image_processor import Sam3Processor

        self._torch = torch
        self.version = version
        self.categories = dict(categories or DEFAULT_CATEGORIES)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[sam31] Loading {version} on {self.device} "
              f"(one-time; reused for every image)...")
        checkpoint_path = checkpoint or download_ckpt_from_hf(version=version)
        model = build_sam3_image_model(
            device=self.device,
            eval_mode=True,
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            # Sam3Processor expects segmentation outputs ('pred_masks') even
            # when only boxes are consumed, so this must stay enabled.
            enable_segmentation=True,
        )
        # Low internal floor; per-category thresholds are applied in detect().
        self._processor = Sam3Processor(model, device=self.device,
                                        confidence_threshold=confidence_floor)
        self.checkpoint_path = str(checkpoint_path)
        print(f"[sam31] Model ready ({checkpoint_path})")

    def _autocast(self):
        if self.device.startswith("cuda"):
            return self._torch.autocast(device_type="cuda", dtype=self._torch.bfloat16)
        return nullcontext()

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        from PIL import Image

        height, width = image_bgr.shape[:2]
        pil = Image.fromarray(image_bgr[:, :, ::-1])  # BGR -> RGB

        detections: list[Detection] = []
        with self._torch.inference_mode(), self._autocast():
            state = self._processor.set_image(pil)
            for label, (prompt, pad_fraction, min_score) in self.categories.items():
                out = self._processor.set_text_prompt(state=state, prompt=prompt)
                boxes = _to_list(out.get("boxes"))
                scores = [float(s) for s in _to_list(out.get("scores"))]
                for box, score in zip(boxes, scores):
                    if score < min_score:
                        continue
                    x0, y0, x1, y1 = _to_xyxy_pixels(box, width, height)
                    w, h = int(round(x1 - x0)), int(round(y1 - y0))
                    if w <= 0 or h <= 0:
                        continue
                    detections.append(Detection(int(round(x0)), int(round(y0)),
                                                w, h, label, score, self.name,
                                                pad_fraction))
        return detections


def _to_list(value) -> list:
    """Tensor / ndarray / iterable -> plain list."""
    if value is None:
        return []
    if hasattr(value, "cpu"):
        value = value.cpu().float().numpy() if hasattr(value, "float") else value.cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value) if hasattr(value, "__iter__") else []


def _to_xyxy_pixels(box, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Normalise a SAM box (pixels or 0..1, xyxy or cxcywh) to pixel xyxy."""
    if hasattr(box, "cpu"):
        box = box.cpu().float().numpy()
    if isinstance(box, np.ndarray):
        box = box.tolist()
    x1, y1, x2, y2 = (float(v) for v in box[:4])

    if max(x1, y1, x2, y2) <= 1.0:            # normalised -> pixels
        x1, x2 = x1 * img_w, x2 * img_w
        y1, y2 = y1 * img_h, y2 * img_h
    if x2 < x1 or y2 < y1:                    # cxcywh -> xyxy
        cx, cy, bw, bh = x1, y1, x2, y2
        x1, y1 = cx - bw / 2, cy - bh / 2
        x2, y2 = cx + bw / 2, cy + bh / 2

    return (max(0.0, x1), max(0.0, y1), min(float(img_w), x2), min(float(img_h), y2))
