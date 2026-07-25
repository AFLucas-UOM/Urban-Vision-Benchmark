from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageOps, ImageStat

from .utils import fingerprint, sha256_file

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _add(findings: list[dict[str, str]], severity: str, code: str, detail: str) -> None:
    findings.append({"severity": severity, "code": code, "detail": detail})


def _read_yolo_boxes(path: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_text, centre_x, centre_y, box_width, box_height = line.split()
        cx, cy = float(centre_x) * width, float(centre_y) * height
        bw, bh = float(box_width) * width, float(box_height) * height
        boxes.append((int(class_text), cx - bw / 2, cy - bh / 2, bw, bh))
    return boxes


def _visually_equivalent_images(left: Path, right: Path, rms_tolerance: float = 3.0) -> bool:
    try:
        with Image.open(left) as left_raw, Image.open(right) as right_raw:
            left_image = ImageOps.exif_transpose(left_raw).convert("RGB")
            right_image = ImageOps.exif_transpose(right_raw).convert("RGB")
            if left_image.size != right_image.size:
                return False
            comparison_size = (min(256, left_image.width), min(256, left_image.height))
            left_image.thumbnail(comparison_size, Image.Resampling.LANCZOS)
            right_image.thumbnail(comparison_size, Image.Resampling.LANCZOS)
            if left_image.size != right_image.size:
                return False
            rms = ImageStat.Stat(ImageChops.difference(left_image, right_image)).rms
            return max(rms, default=float("inf")) <= rms_tolerance
    except Exception:
        return False


def _validate_augmentation_manifests(
    variant_root: Path,
    manifest: dict[str, Any],
    split_rows: list[dict[str, str]],
    findings: list[dict[str, str]],
) -> set[str]:
    split_by_name = {row["out_name"]: row["split"] for row in split_rows}
    split_by_hash = {row["source_image_sha256"]: row["split"] for row in split_rows}
    generated_names: set[str] = set()
    jsonl_path = variant_root / "augmentation_manifest.jsonl"
    csv_path = variant_root / "augmentation_manifest.csv"
    if manifest.get("augmentation_manifest_jsonl_sha256"):
        if not jsonl_path.is_file():
            _add(findings, "fatal", "missing_augmentation_jsonl", str(jsonl_path))
            return generated_names
        if sha256_file(jsonl_path) != manifest["augmentation_manifest_jsonl_sha256"]:
            _add(findings, "fatal", "augmentation_jsonl_hash_mismatch", str(jsonl_path))
        for line_number, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
            except Exception as exc:
                _add(findings, "fatal", "invalid_augmentation_jsonl",
                     f"{jsonl_path}:{line_number}: {exc}")
                continue
            generated = str(row.get("generated_filename", ""))
            if not generated or generated in generated_names:
                _add(findings, "fatal", "duplicate_generated_filename", generated or f"line {line_number}")
                continue
            generated_names.add(generated)
            if row.get("split") != "train":
                _add(findings, "fatal", "augmentation_manifest_non_train_split", generated)
            sources = row.get("source_images", [])
            hashes = row.get("source_image_hashes", [])
            if not sources or len(sources) != len(hashes):
                _add(findings, "fatal", "augmentation_source_audit_incomplete", generated)
            for source in sources:
                if split_by_name.get(source) != "train":
                    _add(findings, "fatal", "augmentation_source_split_leakage",
                         f"{generated} <- {source} ({split_by_name.get(source)})")
            for source_hash in hashes:
                if split_by_hash.get(source_hash) != "train":
                    _add(findings, "fatal", "augmentation_source_hash_leakage",
                         f"{generated} <- {source_hash} ({split_by_hash.get(source_hash)})")
            yolo_image = variant_root / "MTSD-YOLO" / "train" / "images" / generated
            coco_image = variant_root / "MTSD-COCO" / "train" / generated
            for image_path in (yolo_image, coco_image):
                if not image_path.is_file():
                    _add(findings, "fatal", "missing_generated_image", str(image_path))
            expected_hash = row.get("generated_image_hash")
            if not expected_hash:
                _add(findings, "fatal", "missing_generated_hash", generated)
            elif yolo_image.is_file() and sha256_file(yolo_image) != expected_hash:
                _add(findings, "fatal", "generated_hash_mismatch", str(yolo_image))
            elif coco_image.is_file() and sha256_file(coco_image) != expected_hash:
                _add(findings, "fatal", "generated_hash_mismatch", str(coco_image))
            if not row.get("recipe_version") or row.get("random_seed") is None:
                _add(findings, "fatal", "augmentation_recipe_audit_incomplete", generated)
    elif csv_path.is_file() and manifest.get("augmentation", {}).get("enabled"):
        with csv_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                generated = row.get("generated_filename", "")
                if generated:
                    generated_names.add(generated)
                if row.get("split") != "train":
                    _add(findings, "fatal", "augmentation_manifest_non_train_split", generated)
                source_hash = row.get("source_image_hash") or row.get("source_image_sha256")
                if source_hash and split_by_hash.get(source_hash) != "train":
                    _add(findings, "fatal", "augmentation_source_hash_leakage",
                         f"{generated} <- {source_hash}")
                image_path = variant_root / "MTSD-YOLO" / "train" / "images" / generated
                expected_hash = row.get("generated_image_hash")
                if expected_hash and image_path.is_file() and sha256_file(image_path) != expected_hash:
                    _add(findings, "fatal", "generated_hash_mismatch", str(image_path))
    return generated_names


def validate_variant(variant_root: Path, strict: bool = False,
                     policy: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = policy or {"require_all_classes_in_test": False,
                        "minimum_boxes_per_class": {"train": 1, "valid": 0, "test": 0},
                        "low_support_warning_threshold": 3}
    findings: list[dict[str, str]] = []
    manifest_path = variant_root / "prep_manifest.json"
    split_path = variant_root / "split_manifest.csv"
    if not manifest_path.is_file(): _add(findings, "fatal", "missing_manifest", str(manifest_path))
    if not split_path.is_file(): _add(findings, "fatal", "missing_split_manifest", str(split_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    names = manifest.get("class_names", [])
    if manifest:
        recorded_fingerprint = manifest.get("manifest_fingerprint")
        unsigned = dict(manifest); unsigned.pop("manifest_fingerprint", None)
        if recorded_fingerprint != fingerprint(unsigned):
            _add(findings, "fatal", "manifest_fingerprint_mismatch", str(manifest_path))
        if split_path.exists() and manifest.get("split_manifest_sha256") != sha256_file(split_path):
            _add(findings, "fatal", "split_manifest_hash_mismatch", str(split_path))
        if manifest.get("group_scope") == "explicit":
            approved = set(manifest.get("approved_groups", []))
            included = {row["group"] for row in manifest.get("groups", []) if row.get("status") == "included"}
            if approved != included:
                _add(findings, "fatal", "approved_group_scope_mismatch", f"approved={sorted(approved)} included={sorted(included)}")
        for group in manifest.get("groups", []):
            source = Path(str(group.get("source_path") or ""))
            expected = group.get("source_sha256")
            if group.get("status") == "included" and expected and (not source.is_file() or sha256_file(source) != expected):
                _add(findings, "fatal", "stale_annotation_hash", str(source))
        source_findings = manifest.get("validation_findings", {})
        if source_findings.get("box_clipped_to_image"):
            _add(findings, "warning", "clipped_source_boxes",
                 str(source_findings["box_clipped_to_image"]))
        if source_findings.get("image_without_valid_boxes"):
            _add(findings, "warning", "retained_background_images",
                 str(source_findings["image_without_valid_boxes"]))
    seen_hashes: dict[str, str] = {}
    within_split_hashes: set[tuple[str, str]] = set()
    seen_names: set[str] = set()
    split_rows: list[dict[str, str]] = []
    if split_path.exists():
        with split_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                split_rows.append(row)
                if row["out_name"] in seen_names:
                    _add(findings, "fatal", "duplicate_out_name", row["out_name"])
                seen_names.add(row["out_name"])
                old = seen_hashes.setdefault(row["source_image_sha256"], row["split"])
                if old != row["split"]:
                    _add(findings, "fatal", "hash_leakage", f"{row['source_image_sha256']} in {old} and {row['split']}")
                key = (row["source_image_sha256"], row["split"])
                if key in within_split_hashes:
                    _add(findings, "warning", "within_split_duplicate_hash", f"{row['source_image_sha256']} in {row['split']}")
                within_split_hashes.add(key)
    generated_names = _validate_augmentation_manifests(
        variant_root, manifest, split_rows, findings,
    ) if manifest else set()
    yolo = variant_root / "MTSD-YOLO"
    coco = variant_root / "MTSD-COCO"
    class_counts: dict[str, list[int]] = {split: [0] * len(names) for split in ("train", "valid", "test")}
    for split in ("train", "valid", "test"):
        images = {p.stem: p for p in (yolo / split / "images").glob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
        labels = {p.stem: p for p in (yolo / split / "labels").glob("*.txt")}
        if split in {"valid", "test"} and not images:
            _add(findings, "fatal", "empty_required_split", split)
        for stem in sorted(images.keys() - labels.keys()): _add(findings, "fatal", "missing_label", f"{split}/{stem}")
        for stem in sorted(labels.keys() - images.keys()): _add(findings, "fatal", "orphan_label", f"{split}/{stem}")
        for stem, path in images.items():
            if ("_aug" in stem or path.name in generated_names) and (
                split != "train" or manifest.get("offline_augmentation_selection") == "none"
            ):
                _add(findings, "fatal", "augmentation_outside_train", str(path))
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    if image.width <= 0 or image.height <= 0: raise ValueError("zero dimension")
            except Exception as exc: _add(findings, "fatal", "corrupt_image", f"{path}: {exc}")
        for path in labels.values():
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    parts = line.split(); assert len(parts) == 5
                    class_id = int(parts[0]); coords = [float(v) for v in parts[1:]]
                    assert 0 <= class_id < len(names) and all(0 <= v <= 1 for v in coords) and coords[2] > 0 and coords[3] > 0
                    class_counts[split][class_id] += 1
                except Exception: _add(findings, "fatal", "invalid_yolo_box", f"{path}:{line_no}: {line}")
        coco_path = coco / split / "_annotations.coco.json"
        if not coco_path.is_file():
            _add(findings, "fatal", "missing_coco", str(coco_path)); continue
        payload = json.loads(coco_path.read_text(encoding="utf-8"))
        if [c["name"] for c in sorted(payload.get("categories", []), key=lambda c: c["id"])] != names:
            _add(findings, "fatal", "class_vocabulary_mismatch", str(coco_path))
        coco_names = {Path(row["file_name"]).stem for row in payload.get("images", [])}
        if coco_names != set(images): _add(findings, "fatal", "yolo_coco_membership_mismatch", split)
        coco_images = {int(row["id"]): row for row in payload.get("images", [])}
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for image_id, row in coco_images.items():
            image_path = coco / split / row["file_name"]
            if not image_path.is_file():
                _add(findings, "fatal", "coco_missing_image", str(image_path))
                continue
            try:
                with Image.open(image_path) as image:
                    actual_size = image.size
                recorded_size = (int(row["width"]), int(row["height"]))
                if actual_size != recorded_size:
                    _add(findings, "fatal", "coco_image_dimension_mismatch",
                         f"{image_path}: actual={actual_size} metadata={recorded_size}")
            except Exception as exc:
                _add(findings, "fatal", "corrupt_coco_image", f"{image_path}: {exc}")
        for box in payload.get("annotations", []):
            image_id = int(box.get("image_id", -1))
            annotations_by_image[image_id].append(box)
            image_row = coco_images.get(image_id)
            if image_row is None:
                _add(findings, "fatal", "orphan_coco_annotation",
                     f"{coco_path}: annotation {box.get('id')}")
                continue
            x, y, w, h = box.get("bbox", [0, 0, 0, 0])
            if (min(x, y) < 0 or w <= 0 or h <= 0 or
                    x + w > float(image_row["width"]) + 1e-6 or
                    y + h > float(image_row["height"]) + 1e-6 or
                    not 0 <= int(box.get("category_id", -1)) < len(names)):
                _add(findings, "fatal", "invalid_coco_box", f"{coco_path}: annotation {box.get('id')}")
        for image_id, row in coco_images.items():
            stem = Path(row["file_name"]).stem
            label_path = labels.get(stem)
            if label_path is None:
                continue
            try:
                yolo_boxes = _read_yolo_boxes(label_path, int(row["width"]), int(row["height"]))
            except Exception as exc:
                _add(findings, "fatal", "invalid_yolo_box", f"{label_path}: {exc}")
                continue
            coco_boxes = annotations_by_image.get(image_id, [])
            if len(yolo_boxes) != len(coco_boxes):
                _add(findings, "fatal", "yolo_coco_annotation_count_mismatch",
                     f"{split}/{row['file_name']}: yolo={len(yolo_boxes)} coco={len(coco_boxes)}")
                continue
            for index, (yolo_box, coco_box) in enumerate(zip(yolo_boxes, coco_boxes)):
                expected = (
                    int(coco_box["category_id"]),
                    float(coco_box["bbox"][0]), float(coco_box["bbox"][1]),
                    float(coco_box["bbox"][2]), float(coco_box["bbox"][3]),
                )
                if yolo_box[0] != expected[0] or any(
                    abs(left - right) > 0.11
                    for left, right in zip(yolo_box[1:], expected[1:])
                ):
                    _add(findings, "fatal", "yolo_coco_box_mismatch",
                         f"{split}/{row['file_name']} box={index} yolo={yolo_box} coco={expected}")
    for generated in generated_names:
        for split in ("valid", "test"):
            if ((yolo / split / "images" / generated).exists() or
                    (coco / split / generated).exists()):
                _add(findings, "fatal", "generated_derivative_split_leakage",
                     f"{generated} appears in {split}")
    minimums = policy.get("minimum_boxes_per_class", {})
    low_threshold = int(policy.get("low_support_warning_threshold", 3))
    for split, counts in class_counts.items():
        minimum = int(minimums.get(split, 0))
        for class_id, count in enumerate(counts):
            if count < minimum:
                _add(findings, "fatal", "minimum_class_support", f"{split}/{names[class_id]}={count} < {minimum}")
            elif count == 0 and split in {"valid", "test"}:
                severity = "fatal" if split == "test" and policy.get("require_all_classes_in_test", False) else "warning"
                _add(findings, severity, "class_absent_from_split", f"{split}/{names[class_id]}")
            elif count < low_threshold:
                _add(findings, "warning", "low_class_support", f"{split}/{names[class_id]}={count}")
    fatals = [row for row in findings if row["severity"] == "fatal"]
    warnings = [row for row in findings if row["severity"] == "warning"]
    return {"ok": not fatals, "fatal_count": len(fatals),
            "warning_count": len(warnings), "findings": findings}


def validate_prepared(prepared_root: Path, strict: bool = False,
                      policy: dict[str, Any] | None = None,
                      variant_names: list[str] | None = None) -> dict[str, Any]:
    if variant_names is None:
        preferred = (
            "MTSD-Augmented", "MTSD-Augmented-Mild",
            "MTSD-Augmented-Strong", "MTSD-Unaugmented",
        )
        variant_names = [name for name in preferred if (prepared_root / name).exists()]
    results = {name: validate_variant(prepared_root / name, strict, policy)
               for name in variant_names if (prepared_root / name).exists()}
    if not results:
        return {"ok": False, "variants": {}, "findings": [{"severity": "fatal", "code": "no_variants", "detail": str(prepared_root)}]}
    plain = prepared_root / "MTSD-Unaugmented"
    cross = []
    if plain.exists():
        augmented_variants = [
            prepared_root / name for name in variant_names
            if name != plain.name and (prepared_root / name).exists()
        ]
        for aug in augmented_variants:
            for split in ("train", "valid", "test"):
                for relative in (Path("MTSD-YOLO") / split / "images", Path("MTSD-YOLO") / split / "labels"):
                    left, right = aug / relative, plain / relative
                    right_hashes = {p.relative_to(right).as_posix(): sha256_file(p) for p in right.glob("*") if p.is_file()}
                    # Augmented train contains extra offline copies. Compare only the
                    # original names present in the unaugmented shared split.
                    left_hashes = {
                        name: sha256_file(left / name)
                        for name in right_hashes
                        if (left / name).is_file()
                    }
                    if left_hashes != right_hashes:
                        differing = [
                            name for name, expected in right_hashes.items()
                            if left_hashes.get(name) != expected
                        ]
                        if relative.name == "images":
                            differing = [
                                name for name in differing
                                if not (left / name).is_file() or
                                not _visually_equivalent_images(left / name, right / name)
                            ]
                        if differing:
                            _add(cross, "fatal", "variant_split_hash_mismatch",
                                 f"{aug.name}/{split}/{relative.name}: {len(differing)} files, "
                                 f"examples={differing[:3]}")
    return {"ok": all(row["ok"] for row in results.values()) and not cross,
            "variants": results, "findings": cross}
