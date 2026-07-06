"""Ground-truth loading for PromptDetect batch evaluation.

Every loader returns the same internal structure - a list of image records:

    {"image_path": Path, "image_id": str, "width": int|None, "height": int|None,
     "boxes": [{"class_name": str, "x0": .., "y0": .., "x1": .., "y1": ..}, ...]}

Supported sources (auto-selected by dataset name):

* **MDWD**  - YOLO layout at Datasets/MDWD/MDWD-YOLO26 (train/valid/test).
* **MTSD**  - the prepared detection dataset at
  Datasets/MTSD/Prepared/MTSD-YOLO when it exists (train/valid/test);
  otherwise falls back to the QA COCO annotations under
  Datasets/MTSD/Annotations/GRP-*/Final-QA/ (split must be "all"; groups are
  discovered dynamically, current and future alike).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

import config


def normalise_split(split: str) -> str:
    split = config.SPLIT_ALIASES.get(split.lower(), split.lower())
    if split not in config.VALID_SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {config.VALID_SPLITS}")
    return split


# --- YOLO layout -----------------------------------------------------------------

def _load_yolo_split(dataset_dir: Path, split: str, max_images: int | None) -> tuple[list[dict], list[str]]:
    data_yaml = yaml.safe_load((dataset_dir / "data.yaml").read_text(encoding="utf-8")) or {}
    class_names = list(data_yaml.get("names", []))
    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Split folder not found: {images_dir}")

    records = []
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if max_images is not None:
        image_paths = image_paths[:max_images]
    for image_path in image_paths:
        boxes = []
        label_path = labels_dir / (image_path.stem + ".txt")
        width = height = None
        if label_path.exists():
            lines = [l.split() for l in label_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            if lines:
                from PIL import Image

                with Image.open(image_path) as img:
                    width, height = img.size
                for parts in lines:
                    try:
                        class_id = int(parts[0])
                        cx, cy, bw, bh = (float(v) for v in parts[1:5])
                    except (ValueError, IndexError):
                        continue
                    boxes.append({
                        "class_name": class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id),
                        "x0": (cx - bw / 2) * width, "y0": (cy - bh / 2) * height,
                        "x1": (cx + bw / 2) * width, "y1": (cy + bh / 2) * height,
                    })
        records.append({
            "image_path": image_path, "image_id": image_path.name,
            "width": width, "height": height, "boxes": boxes,
        })
    return records, class_names


# --- MTSD QA fallback (COCO) --------------------------------------------------------

def _load_mtsd_qa(max_images: int | None) -> tuple[list[dict], list[str]]:
    qa_files = sorted(config.MTSD_ANNOTATIONS_ROOT.glob("GRP-*/Final-QA/QA-*.json"))
    qa_files = [p for p in qa_files if ".bak" not in p.name.lower()]
    if not qa_files:
        raise FileNotFoundError(f"No QA annotation files under {config.MTSD_ANNOTATIONS_ROOT}")
    records: list[dict] = []
    class_names: list[str] = []
    for qa_path in qa_files:
        data = json.loads(qa_path.read_text(encoding="utf-8-sig"))
        categories = {c["id"]: c["name"] for c in data.get("categories", [])}
        for name in categories.values():
            if name not in class_names:
                class_names.append(name)
        boxes_by_image: dict = {}
        for annotation in data.get("annotations", []):
            x, y, w, h = annotation.get("bbox", [0, 0, 0, 0])
            boxes_by_image.setdefault(annotation["image_id"], []).append({
                "class_name": categories.get(annotation.get("category_id"), "?"),
                "x0": float(x), "y0": float(y), "x1": float(x) + float(w), "y1": float(y) + float(h),
            })
        for image in data.get("images", []):
            source = str(image.get("source_image", "") or "").replace("\\", "/")
            image_path = config.PROJECT_ROOT / source
            if not image_path.exists():
                continue  # missing files are excluded, matching the prep notebook
            records.append({
                "image_path": image_path,
                "image_id": f"{qa_path.parents[1].name}/{image.get('file_name')}",
                "width": image.get("width"), "height": image.get("height"),
                "boxes": boxes_by_image.get(image["id"], []),
            })
            if max_images is not None and len(records) >= max_images:
                return records, class_names
    return records, class_names


# --- public API ---------------------------------------------------------------------

def load_ground_truth(dataset: str, split: str, max_images: int | None = None) -> dict:
    """Load a dataset split into the common internal format.

    Returns {"dataset", "split", "source", "class_names", "records"}.
    """
    dataset = dataset.upper()
    split = normalise_split(split)
    if dataset == "MDWD":
        if split == "all":
            raise ValueError("MDWD is split-based; choose train/valid/test.")
        records, class_names = _load_yolo_split(config.MDWD_YOLO_DIR, split, max_images)
        source = str(config.MDWD_YOLO_DIR)
    elif dataset == "MTSD":
        if config.MTSD_PREPARED_YOLO_DIR.is_dir() and split != "all":
            records, class_names = _load_yolo_split(config.MTSD_PREPARED_YOLO_DIR, split, max_images)
            source = str(config.MTSD_PREPARED_YOLO_DIR)
        elif split == "all":
            records, class_names = _load_mtsd_qa(max_images)
            source = str(config.MTSD_ANNOTATIONS_ROOT)
        else:
            raise FileNotFoundError(
                f"Prepared MTSD dataset not found at {config.MTSD_PREPARED_YOLO_DIR}. "
                f"Run Prepare-MTSD-Detection-Dataset.ipynb first, or use --split all "
                f"to evaluate directly against the QA annotations."
            )
    else:
        raise ValueError(f"Unknown dataset {dataset!r}; expected MDWD or MTSD.")
    return {
        "dataset": dataset, "split": split, "source": source,
        "class_names": class_names, "records": records,
    }


def ground_truth_index_rows(gt: dict) -> list[dict]:
    """Flatten ground truth to CSV rows (one per box; box-less images included)."""
    rows = []
    for record in gt["records"]:
        if not record["boxes"]:
            rows.append({"image_id": record["image_id"], "image_path": str(record["image_path"]),
                         "class_name": "", "x0": "", "y0": "", "x1": "", "y1": ""})
        for box in record["boxes"]:
            rows.append({"image_id": record["image_id"], "image_path": str(record["image_path"]),
                         **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in box.items()}})
    return rows
