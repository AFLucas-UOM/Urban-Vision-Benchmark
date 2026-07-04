"""Preview and apply orchestration.

``preview`` never touches originals: redacted copies, comparison sheets, and
reports go to a separate output tree that mirrors the input structure.
``apply_results`` is the only code path that writes into the input tree, and
it backs up every original it replaces (unless explicitly disabled).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2

from .blur import annotate, redact, side_by_side
from .detectors.base import BaseDetector
from .imaging import (IMAGE_EXTENSIONS, discover_images, load_display_oriented,
                      save_like_original)
from .reporting import ImageResult, write_reports

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

BACKUP_DIR_NAME = "_replaced_originals"
COMPARISONS_DIR_NAME = "_comparisons"


def run_preview(
    input_root: Path,
    output_root: Path,
    detectors: list[BaseDetector],
    *,
    backend: str,
    method: str = "gaussian",
    make_comparisons: bool = True,
    copy_clean: bool = False,
    jpeg_quality: int = 95,
    limit: int | None = None,
) -> dict:
    """Process every image under ``input_root`` into ``output_root``.

    Only images with at least one detection are written (unless
    ``copy_clean`` asks for a complete mirror). Returns the report payload.
    """
    if output_root.resolve() == input_root.resolve():
        raise SystemExit("Output directory must equal the input directory? No - "
                         "this tool never writes into the originals in preview.")
    output_root.mkdir(parents=True, exist_ok=True)

    targets = discover_images(input_root)
    if limit:
        targets = targets[:limit]

    iterator = tqdm(targets, desc="Redacting", unit="img") if tqdm else targets
    results: list[ImageResult] = []

    for path in iterator:
        relative = path.relative_to(input_root)
        result = ImageResult(source=str(path), relative_path=relative.as_posix())
        try:
            image, exif_bytes = load_display_oriented(path)
            height, width = image.shape[:2]

            raw, padded = [], []
            for detector in detectors:
                for det in detector.detect(image):
                    raw.append(det)
                    padded.append(det.padded(width, height))
            result.detections_raw = raw
            result.detections_padded = padded

            if raw:
                result.redaction_method = method
                redacted = redact(image, padded, method=method)
                preview_path = output_root / relative
                save_like_original(redacted, path, preview_path,
                                   exif_bytes, jpeg_quality)
                result.preview = str(preview_path)
                if make_comparisons:
                    sheet = side_by_side(annotate(image, padded), redacted)
                    sheet_path = (output_root / COMPARISONS_DIR_NAME / relative
                                  ).with_suffix(".jpg")
                    sheet_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(sheet_path), sheet,
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
                    result.comparison = str(sheet_path)
            elif copy_clean:
                target = output_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                result.preview = str(target)
        except Exception as error:  # noqa: BLE001 - record and continue
            result.error = f"{type(error).__name__}: {error}"
        results.append(result)

    return write_reports(
        output_root, results,
        input_root=input_root, backend=backend,
        detector_names=[d.name for d in detectors], method=method,
    )


def apply_results(output_root: Path, input_root: Path,
                  backup: bool = True) -> tuple[int, int]:
    """Replace originals with their verified redacted counterparts.

    Every image in ``output_root`` (except comparison sheets and backups) is
    copied over the matching file in ``input_root``. When ``backup`` is on,
    each original is first copied to ``output_root/_replaced_originals/``.
    Returns ``(replaced, missing)`` counts.
    """
    processed = [
        path for path in sorted(output_root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and COMPARISONS_DIR_NAME not in path.relative_to(output_root).parts
        and BACKUP_DIR_NAME not in path.relative_to(output_root).parts
    ]

    replaced = missing = 0
    backup_root = output_root / BACKUP_DIR_NAME
    iterator = tqdm(processed, desc="Replacing", unit="img") if tqdm else processed
    for path in iterator:
        relative = path.relative_to(output_root)
        original = input_root / relative
        if not original.exists():
            missing += 1
            continue
        if backup:
            backup_path = backup_root / relative
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            if not backup_path.exists():
                shutil.copy2(original, backup_path)
        shutil.copy2(path, original)
        replaced += 1
    return replaced, missing
