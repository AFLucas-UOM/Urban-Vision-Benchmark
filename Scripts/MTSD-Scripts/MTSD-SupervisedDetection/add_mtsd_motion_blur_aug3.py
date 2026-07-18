from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from mtsd_detection.augmentation import mild_motion_blur
from mtsd_detection.dataset_validation import validate_variant
from mtsd_detection.manifests import sha256_file, write_csv, write_manifest

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _save_aug3(source: Path, destination: Path, kernel_size: int) -> None:
    with Image.open(source) as raw:
        image = mild_motion_blur(ImageOps.exif_transpose(raw), kernel_size, 0.85)
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_args: dict[str, Any] = {}
        if destination.suffix.lower() in {".jpg", ".jpeg"}:
            save_args = {"quality": 92, "subsampling": 0}
        image.save(destination, **save_args)


def add_motion_blur_aug3(dataset_root: Path, kernel_size: int = 5,
                         seed: int = 42) -> dict[str, int]:
    manifest_path = dataset_root / "prep_manifest.json"
    aug_manifest_path = dataset_root / "augmentation_manifest.csv"
    split_manifest_path = dataset_root / "split_manifest.csv"
    required = (manifest_path, aug_manifest_path, split_manifest_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "MTSD preparation is not complete; missing: " + ", ".join(missing))

    yolo = dataset_root / "MTSD-YOLO" / "train"
    coco = dataset_root / "MTSD-COCO" / "train"
    coco_path = coco / "_annotations.coco.json"
    payload = json.loads(coco_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_rows = {row["out_name"]: row for row in _load_csv(split_manifest_path)
                  if row["split"] == "train"}
    aug_rows = _load_csv(aug_manifest_path)

    existing_aug3 = {row["source_out_name"] for row in aug_rows
                     if int(row["copy_index"]) == 3}
    originals = [row for row in payload["images"]
                 if not row.get("augmented_from") and row["file_name"] in split_rows]
    original_annotations: dict[int, list[dict[str, Any]]] = {}
    for annotation in payload["annotations"]:
        original_annotations.setdefault(int(annotation["image_id"]), []).append(annotation)

    next_image_id = max((int(row["id"]) for row in payload["images"]), default=0) + 1
    next_annotation_id = max((int(row["id"]) for row in payload["annotations"]), default=0) + 1
    added = 0
    for original in sorted(originals, key=lambda row: row["file_name"]):
        source_name = original["file_name"]
        if source_name in existing_aug3:
            continue
        source = yolo / "images" / source_name
        stem, suffix = Path(source_name).stem, Path(source_name).suffix
        if suffix.lower() not in IMAGE_SUFFIXES or not source.is_file():
            raise FileNotFoundError(f"Missing original training image: {source}")
        generated = f"{stem}_aug3{suffix}"
        yolo_destination = yolo / "images" / generated
        _save_aug3(source, yolo_destination, kernel_size)
        shutil.copy2(yolo / "labels" / f"{stem}.txt",
                     yolo / "labels" / f"{stem}_aug3.txt")
        shutil.copy2(yolo_destination, coco / generated)

        payload["images"].append({
            "id": next_image_id, "file_name": generated,
            "width": original["width"], "height": original["height"],
            "mtsd_group": original.get("mtsd_group"),
            "augmented_from": source_name,
        })
        for annotation in original_annotations.get(int(original["id"]), []):
            copied = dict(annotation)
            copied.update({"id": next_annotation_id, "image_id": next_image_id})
            payload["annotations"].append(copied)
            next_annotation_id += 1
        next_image_id += 1

        seed_material = f"{seed}:{source_name}:3:motion-blur-v1"
        aug_rows.append({
            "source_out_name": source_name,
            "source_image_hash": split_rows[source_name]["source_image_sha256"],
            "split": "train", "copy_index": 3,
            "ops_applied": json.dumps({"motion_blur": {
                "direction": "horizontal", "kernel_size": kernel_size,
                "blur_weight": 0.85}}, sort_keys=True),
            "seed_material": seed_material, "generated_filename": generated,
            "generated_image_hash": sha256_file(yolo_destination),
            "label_source": str(yolo / "labels" / f"{stem}.txt"),
        })
        added += 1

    if added == 0:
        return {"added_images": 0, "total_aug3_images": len(existing_aug3)}

    _atomic_json(coco_path, payload)
    fieldnames = list(aug_rows[0])
    write_csv(aug_manifest_path, aug_rows, fieldnames)

    counts = manifest.setdefault("counts", {})
    counts["augmented_images"] = len(aug_rows)
    counts["final_images"] = sum(len(json.loads(
        (dataset_root / "MTSD-COCO" / split / "_annotations.coco.json")
        .read_text(encoding="utf-8"))["images"])
        for split in ("train", "valid", "test"))
    counts["final_boxes"] = sum(len(json.loads(
        (dataset_root / "MTSD-COCO" / split / "_annotations.coco.json")
        .read_text(encoding="utf-8"))["annotations"])
        for split in ("train", "valid", "test"))
    augmentation = manifest.setdefault("augmentation", {})
    augmentation.update({
        "enabled": True, "copies_per_image": 3,
        "recipe_version": "photometric-v1+motion-blur-v1",
    })
    augmentation.setdefault("ops", {})["motion_blur"] = {
        "copy_index": 3, "direction": "horizontal",
        "kernel_size": kernel_size, "blur_weight": 0.85,
    }
    manifest["augmentation_manifest_sha256"] = sha256_file(aug_manifest_path)
    write_manifest(manifest_path, manifest)

    validation = validate_variant(dataset_root, strict=True)
    if not validation["ok"]:
        raise RuntimeError("aug3 was written but dataset validation failed:\n" +
                           json.dumps(validation, indent=2))
    return {"added_images": added,
            "total_aug3_images": sum(int(row["copy_index"]) == 3 for row in aug_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add a mild motion-blurred aug3 copy to every MTSD training image.")
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("Datasets/MTSD/Prepared/MTSD-Augmented"))
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = add_motion_blur_aug3(args.dataset_root.resolve(), args.kernel_size, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
