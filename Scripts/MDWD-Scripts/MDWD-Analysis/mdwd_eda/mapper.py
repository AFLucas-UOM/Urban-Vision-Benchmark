"""Dataset mapper: index MDWD images, labels, splits and bounding boxes.

Supports the two annotation formats actually present under ``Datasets/MDWD``:

* **YOLO** (MDWD-YOLO11/12/26): ``data.yaml`` + ``<split>/images`` and
  ``<split>/labels`` with one ``class cx cy w h`` row per box (normalised).
* **COCO** (MDWD-RFDETR): ``<split>/_annotations.coco.json`` with images and
  boxes in absolute pixels (``bbox = [x, y, w, h]``).

The mapper is pure: it returns two pandas DataFrames (one row per image, one
row per bounding box) plus a list of integrity issues, and writes nothing.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from . import config


@dataclass
class DatasetMap:
    """Everything the EDA needs to know about one dataset variant."""

    variant: str
    format: str
    class_names: list[str]
    images: pd.DataFrame  # one row per image file
    boxes: pd.DataFrame  # one row per annotation box
    issues: list[dict] = field(default_factory=list)

    @property
    def issue_counts(self) -> dict[str, int]:
        return dict(Counter(issue["type"] for issue in self.issues))


def source_stem(filename: str) -> str:
    """Collapse a Roboflow export name to its source-image stem.

    Roboflow names augmented copies ``<orig>_<ext>.rf.<hash>.<ext>``; every
    augmented duplicate of one capture shares the part before ``.rf.``.
    """
    stem = Path(filename).stem
    return stem.split(".rf.")[0] if ".rf." in stem else stem


def _read_image_size(path: Path) -> tuple[int | None, int | None, str | None]:
    """Return (width, height, error). Header-only read, no full decode."""
    try:
        with Image.open(path) as img:
            return img.size[0], img.size[1], None
    except Exception as exc:  # noqa: BLE001 - report every unreadable file
        return None, None, f"{type(exc).__name__}: {exc}"


def load_class_names(variant: Path, fmt: str) -> list[str]:
    """Read the class vocabulary from data.yaml (YOLO) or the COCO categories."""
    if fmt == "yolo":
        data = yaml.safe_load((variant / "data.yaml").read_text(encoding="utf-8"))
        names = data.get("names")
        if isinstance(names, dict):  # {0: "...", 1: "..."} form
            names = [names[key] for key in sorted(names)]
        return list(names or [])
    for split in config.SPLITS:
        coco_path = variant / split / "_annotations.coco.json"
        if coco_path.exists():
            payload = json.loads(coco_path.read_text(encoding="utf-8"))
            categories = sorted(payload.get("categories", []), key=lambda c: c["id"])
            # Roboflow COCO exports prepend a super-category placeholder at id 0.
            return [c["name"] for c in categories]
    return []


def _map_yolo_split(
    variant: Path,
    split: str,
    class_names: list[str],
    image_rows: list[dict],
    box_rows: list[dict],
    issues: list[dict],
    max_images: int | None,
) -> None:
    images_dir = variant / split / "images"
    labels_dir = variant / split / "labels"
    if not images_dir.is_dir():
        issues.append({"type": "missing_split", "split": split, "path": str(images_dir)})
        return

    image_files = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in config.IMAGE_EXTENSIONS
    )
    if max_images is not None:
        image_files = image_files[:max_images]
    seen_label_stems = set()

    for image_path in image_files:
        width, height, read_error = _read_image_size(image_path)
        label_path = labels_dir / f"{image_path.stem}.txt"
        n_boxes = 0
        if read_error:
            issues.append(
                {"type": "unreadable_image", "split": split, "file": image_path.name, "detail": read_error}
            )
        if not label_path.exists():
            issues.append({"type": "missing_label", "split": split, "file": image_path.name})
        else:
            seen_label_stems.add(image_path.stem)
            n_boxes = _parse_yolo_label(
                label_path, split, image_path.name, width, height, class_names, box_rows, issues
            )
            if n_boxes == 0:
                issues.append({"type": "empty_annotation", "split": split, "file": image_path.name})
        image_rows.append(
            {
                "variant": variant.name,
                "split": split,
                "file_name": image_path.name,
                "source_stem": source_stem(image_path.name),
                "width": width,
                "height": height,
                "n_boxes": n_boxes,
                "readable": read_error is None,
            }
        )

    if labels_dir.is_dir() and max_images is None:
        image_stems = {p.stem for p in image_files}
        for label_path in labels_dir.glob("*.txt"):
            if label_path.stem not in image_stems:
                issues.append({"type": "orphan_label", "split": split, "file": label_path.name})


def _parse_yolo_label(
    label_path: Path,
    split: str,
    image_name: str,
    width: int | None,
    height: int | None,
    class_names: list[str],
    box_rows: list[dict],
    issues: list[dict],
) -> int:
    n_valid = 0
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
        except (ValueError, IndexError):
            issues.append(
                {"type": "malformed_label_row", "split": split, "file": label_path.name, "detail": f"line {line_no}"}
            )
            continue
        if not 0 <= class_id < len(class_names):
            issues.append(
                {"type": "invalid_class_id", "split": split, "file": label_path.name, "detail": str(class_id)}
            )
            continue
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            issues.append(
                {"type": "out_of_range_box", "split": split, "file": label_path.name, "detail": f"line {line_no}"}
            )
            continue
        pixel_area = bw * bh * width * height if width and height else None
        box_rows.append(
            {
                "split": split,
                "file_name": image_name,
                "class_id": class_id,
                "class_name": class_names[class_id],
                "cx": cx,
                "cy": cy,
                "box_w": bw,
                "box_h": bh,
                "area_frac": bw * bh,
                "pixel_area": pixel_area,
                "aspect_ratio": (bw / bh) if bh else None,
            }
        )
        n_valid += 1
    return n_valid


def _map_coco_split(
    variant: Path,
    split: str,
    image_rows: list[dict],
    box_rows: list[dict],
    issues: list[dict],
    max_images: int | None,
) -> None:
    coco_path = variant / split / "_annotations.coco.json"
    if not coco_path.exists():
        issues.append({"type": "missing_split", "split": split, "path": str(coco_path)})
        return
    payload = json.loads(coco_path.read_text(encoding="utf-8"))
    categories = {c["id"]: c["name"] for c in payload.get("categories", [])}
    images = payload.get("images", [])
    if max_images is not None:
        images = images[:max_images]
    images_by_id: dict[int, dict] = {}
    boxes_per_image: Counter[int] = Counter()

    for annotation in payload.get("annotations", []):
        boxes_per_image[annotation["image_id"]] += 1

    for image in images:
        images_by_id[image["id"]] = image
        on_disk = (variant / split / image["file_name"]).exists()
        if not on_disk:
            issues.append({"type": "missing_image", "split": split, "file": image["file_name"]})
        if boxes_per_image[image["id"]] == 0:
            issues.append({"type": "empty_annotation", "split": split, "file": image["file_name"]})
        image_rows.append(
            {
                "variant": variant.name,
                "split": split,
                "file_name": image["file_name"],
                "source_stem": source_stem(image["file_name"]),
                "width": image.get("width"),
                "height": image.get("height"),
                "n_boxes": boxes_per_image[image["id"]],
                "readable": on_disk,
            }
        )

    for annotation in payload.get("annotations", []):
        image = images_by_id.get(annotation["image_id"])
        if image is None:
            if max_images is None:
                issues.append(
                    {"type": "orphan_annotation", "split": split, "detail": f"image_id {annotation['image_id']}"}
                )
            continue
        width, height = image.get("width"), image.get("height")
        if annotation["category_id"] not in categories:
            issues.append(
                {"type": "invalid_class_id", "split": split, "file": image["file_name"],
                 "detail": str(annotation["category_id"])}
            )
            continue
        x, y, bw, bh = annotation["bbox"]
        if bw <= 0 or bh <= 0 or x < 0 or y < 0 or (width and x + bw > width + 1) or (height and y + bh > height + 1):
            issues.append({"type": "out_of_range_box", "split": split, "file": image["file_name"]})
            continue
        box_rows.append(
            {
                "split": split,
                "file_name": image["file_name"],
                "class_id": annotation["category_id"],
                "class_name": categories[annotation["category_id"]],
                "cx": (x + bw / 2) / width if width else None,
                "cy": (y + bh / 2) / height if height else None,
                "box_w": bw / width if width else None,
                "box_h": bh / height if height else None,
                "area_frac": (bw * bh) / (width * height) if width and height else None,
                "pixel_area": bw * bh,
                "aspect_ratio": bw / bh if bh else None,
            }
        )


def build_map(variant_name: str = config.DEFAULT_VARIANT, max_images: int | None = None) -> DatasetMap:
    """Index one MDWD variant into image/box DataFrames plus integrity issues.

    Args:
        variant_name: Folder name under Datasets/MDWD (e.g. ``MDWD-YOLO26``).
        max_images: Optional per-split cap for quick sample runs. Orphan-label
            and cross-split checks are skipped when a cap is active.
    """
    variant = config.variant_dir(variant_name)
    fmt = config.detect_format(variant)
    class_names = load_class_names(variant, fmt)

    image_rows: list[dict] = []
    box_rows: list[dict] = []
    issues: list[dict] = []

    for split in config.SPLITS:
        if fmt == "yolo":
            _map_yolo_split(variant, split, class_names, image_rows, box_rows, issues, max_images)
        else:
            _map_coco_split(variant, split, image_rows, box_rows, issues, max_images)

    images = pd.DataFrame(image_rows)
    boxes = pd.DataFrame(box_rows)

    if not images.empty and max_images is None:
        duplicated = images[images.duplicated("file_name", keep=False)]
        for file_name in sorted(duplicated["file_name"].unique()):
            splits = ", ".join(sorted(duplicated.loc[duplicated["file_name"] == file_name, "split"]))
            issues.append({"type": "duplicate_filename", "file": file_name, "detail": splits})
        leaked = images.groupby("source_stem")["split"].nunique()
        for stem in leaked[leaked > 1].index:
            splits = ", ".join(sorted(images.loc[images["source_stem"] == stem, "split"].unique()))
            issues.append({"type": "source_in_multiple_splits", "file": stem, "detail": splits})

    return DatasetMap(
        variant=variant.name,
        format=fmt,
        class_names=class_names,
        images=images,
        boxes=boxes,
        issues=issues,
    )
