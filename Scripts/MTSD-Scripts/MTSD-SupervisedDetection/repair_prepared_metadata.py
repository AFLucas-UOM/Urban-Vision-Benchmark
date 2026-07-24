#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mtsd_detection.config import load_config
from mtsd_detection.manifests import sha256_file, write_csv, write_manifest

SPLITS = ("train", "valid", "test")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def repair_variant(variant_root: Path) -> list[dict[str, Any]]:
    split_path = variant_root / "split_manifest.csv"
    manifest_path = variant_root / "prep_manifest.json"
    if not split_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared variant is incomplete: {variant_root}")
    with split_path.open(encoding="utf-8", newline="") as stream:
        split_rows = list(csv.DictReader(stream))
    rows_by_name = {row["out_name"]: row for row in split_rows}
    repairs: list[dict[str, Any]] = []
    for split in SPLITS:
        coco_path = variant_root / "MTSD-COCO" / split / "_annotations.coco.json"
        payload = json.loads(coco_path.read_text(encoding="utf-8"))
        annotation_counts: dict[int, int] = {}
        for annotation in payload.get("annotations", []):
            image_id = int(annotation["image_id"])
            annotation_counts[image_id] = annotation_counts.get(image_id, 0) + 1
        changed = False
        for image_row in payload.get("images", []):
            file_name = image_row["file_name"]
            split_row = rows_by_name.get(file_name)
            if split_row is None or int(split_row.get("n_boxes", -1)) != 0:
                continue
            if annotation_counts.get(int(image_row["id"]), 0):
                raise RuntimeError(f"Refusing dimension repair for annotated image: {file_name}")
            label_path = (
                variant_root / "MTSD-YOLO" / split / "labels" /
                f"{Path(file_name).stem}.txt"
            )
            if not label_path.is_file() or label_path.read_text(encoding="utf-8").strip():
                raise RuntimeError(f"Refusing dimension repair for non-empty label: {label_path}")
            image_path = variant_root / "MTSD-COCO" / split / file_name
            with Image.open(image_path) as image:
                actual_width, actual_height = image.size
            recorded = (int(image_row["width"]), int(image_row["height"]))
            actual = (actual_width, actual_height)
            if recorded == actual:
                continue
            image_row["width"], image_row["height"] = actual
            split_row["width"], split_row["height"] = map(str, actual)
            repairs.append({
                "split": split,
                "file_name": file_name,
                "recorded_size": list(recorded),
                "actual_size": list(actual),
                "reason": "empty_image_metadata_dimension_normalization",
            })
            changed = True
        if changed:
            _write_json_atomic(coco_path, payload)
    if repairs:
        write_csv(split_path, split_rows, list(split_rows[0]))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["split_manifest_sha256"] = sha256_file(split_path)
        manifest["cross_variant_split_manifest_sha256"] = sha256_file(split_path)
        manifest.setdefault("prepared_metadata_repairs", []).extend(repairs)
        write_manifest(manifest_path, manifest)
    return repairs


def refresh_strong_summary(variant_root: Path) -> dict[str, int]:
    jsonl_path = variant_root / "augmentation_manifest.jsonl"
    summary_path = variant_root / "visual_qa" / "summary.json"
    manifest_path = variant_root / "prep_manifest.json"
    if not jsonl_path.is_file() or not summary_path.is_file():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    class_names = list(summary["class_distribution_before"])
    added: Counter[int] = Counter()
    with jsonl_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            added.update(
                int(source["class_id"])
                for source in row.get("copy_paste_source_annotations", [])
            )
    added_by_name = {
        name: added[index] for index, name in enumerate(class_names)
    }
    summary["copy_paste_added_by_class"] = added_by_name
    summary["class_distribution_after_copy_paste_additions"] = {
        name: int(summary["class_distribution_before"][name]) + added_by_name[name]
        for name in class_names
    }
    _write_json_atomic(summary_path, summary)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["strong_augmentation_summary"] = summary
    write_manifest(manifest_path, manifest)
    return added_by_name


def normalize_strong_dataset_version(variant_root: Path) -> str | None:
    manifest_path = variant_root / "prep_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recipe = manifest.get("augmentation", {}).get("recipe_version")
    if recipe != "strong-offline-v2":
        return None
    old_version = str(manifest["dataset_version"])
    new_version = old_version.replace("-strong-strong-offline-v2", "-strong-offline-v2")
    if new_version == old_version:
        return new_version
    data_path = variant_root / "MTSD-YOLO" / "data.yaml"
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data["mtsd"]["version"] = new_version
    data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    for split in SPLITS:
        coco_path = variant_root / "MTSD-COCO" / split / "_annotations.coco.json"
        payload = json.loads(coco_path.read_text(encoding="utf-8"))
        payload.setdefault("info", {})["description"] = f"MTSD {new_version} {split}"
        _write_json_atomic(coco_path, payload)
    manifest["dataset_version"] = new_version
    write_manifest(manifest_path, manifest)
    return new_version


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair and refresh verified prepared MTSD metadata",
    )
    parser.add_argument("--config", type=Path, default=HERE / "config" / "default.yaml")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["MTSD-Augmented-Strong", "MTSD-Unaugmented"],
    )
    args = parser.parse_args()
    config = load_config(args.config)
    prepared_root = Path(config["dataset"]["prepared_root"])
    results = {}
    for name in args.variants:
        variant_root = prepared_root / name
        results[name] = {
            "dimension_repairs": repair_variant(variant_root),
            "copy_paste_added_by_class": refresh_strong_summary(variant_root),
            "dataset_version": normalize_strong_dataset_version(variant_root),
        }
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
