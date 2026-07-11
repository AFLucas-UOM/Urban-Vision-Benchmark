from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from .utils import fingerprint, sha256_file

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _add(findings: list[dict[str, str]], severity: str, code: str, detail: str) -> None:
    findings.append({"severity": severity, "code": code, "detail": detail})


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
    if split_path.exists():
        with split_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
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
            if "_aug" in stem and (split != "train" or "Unaugmented" in variant_root.name):
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
        for box in payload.get("annotations", []):
            x, y, w, h = box.get("bbox", [0, 0, 0, 0])
            if min(x, y) < 0 or w <= 0 or h <= 0 or not 0 <= int(box.get("category_id", -1)) < len(names):
                _add(findings, "fatal", "invalid_coco_box", f"{coco_path}: annotation {box.get('id')}")
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
                      policy: dict[str, Any] | None = None) -> dict[str, Any]:
    results = {name: validate_variant(prepared_root / name, strict, policy)
               for name in ("MTSD-Augmented", "MTSD-Unaugmented") if (prepared_root / name).exists()}
    if not results:
        return {"ok": False, "variants": {}, "findings": [{"severity": "fatal", "code": "no_variants", "detail": str(prepared_root)}]}
    aug, plain = prepared_root / "MTSD-Augmented", prepared_root / "MTSD-Unaugmented"
    cross = []
    if aug.exists() and plain.exists():
        for split in ("valid", "test"):
            for relative in (Path("MTSD-YOLO") / split / "images", Path("MTSD-YOLO") / split / "labels"):
                left, right = aug / relative, plain / relative
                left_hashes = {p.relative_to(left).as_posix(): sha256_file(p) for p in left.glob("*") if p.is_file()}
                right_hashes = {p.relative_to(right).as_posix(): sha256_file(p) for p in right.glob("*") if p.is_file()}
                if left_hashes != right_hashes: _add(cross, "fatal", "variant_split_hash_mismatch", f"{split}/{relative.name}")
    return {"ok": all(row["ok"] for row in results.values()) and not cross,
            "variants": results, "findings": cross}
