"""Image discovery and EXIF-correct loading/saving."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
EXCLUDED_DIR_NAMES = {"Ignore", "Export", "labelstudio_output", "LabelStudioData",
                      "Annotations", "_comparisons", "_replaced_originals"}
ORIENTATION_TAG = 274


def discover_images(input_root: Path) -> list[Path]:
    """Every image under ``input_root``, skipping non-capture directories."""
    files = []
    for path in sorted(input_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_parts = path.relative_to(input_root).parts[:-1]
        if any(part in EXCLUDED_DIR_NAMES for part in relative_parts):
            continue
        files.append(path)
    return files


def load_display_oriented(path: Path) -> tuple[np.ndarray, bytes | None]:
    """Read an image in display orientation; return (BGR array, EXIF bytes).

    The EXIF orientation is baked into the pixels and the returned EXIF block
    has its orientation tag reset, so saved results are never double-rotated
    while all other metadata (GPS, timestamps, camera) is preserved.
    """
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        exif = img.getexif()
        if ORIENTATION_TAG in exif:
            exif[ORIENTATION_TAG] = 1
        exif_bytes = exif.tobytes() if len(exif) else None
        rgb = np.asarray(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), exif_bytes


def save_like_original(image_bgr: np.ndarray, source: Path, target: Path,
                       exif_bytes: bytes | None, jpeg_quality: int) -> None:
    """Write a processed image next to its mirrored path, keeping metadata."""
    target.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    save_kwargs: dict = {}
    if source.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs.update(format="JPEG", quality=jpeg_quality, subsampling=1)
    if exif_bytes:
        save_kwargs["exif"] = exif_bytes
    pil.save(target, **save_kwargs)
