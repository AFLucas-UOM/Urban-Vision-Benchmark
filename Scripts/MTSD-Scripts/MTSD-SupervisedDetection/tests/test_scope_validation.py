import csv
import json
from pathlib import Path

from PIL import Image
import pytest

from mtsd_detection.annotation_sources import CLASS_NAMES
from mtsd_detection.dataset_validation import validate_variant
from mtsd_detection.manifests import sha256_file, write_manifest
from mtsd_detection.qa_gate import enforce_qa_gate, load_qa_gate
from run_mtsd_supervised import main as supervised_main


def test_qa_gate_final_and_development_policy(tmp_path):
    path = tmp_path / "gate.yaml"
    path.write_text(
        "gate_version: v1\naudit_path: audit\naudit_timestamp: now\n"
        "audited_groups: [GRP-1]\napproved_scope: [GRP-1]\n"
        "finding_counts: {duplicates: 1}\nresolution_status: unresolved\n",
        encoding="utf-8",
    )
    gate = load_qa_gate(path, tmp_path)
    with pytest.raises(PermissionError, match="Final operation refused"):
        enforce_qa_gate(gate, final=True, acknowledged_override=False)
    with pytest.raises(PermissionError, match="acknowledge-open-qa-gate"):
        enforce_qa_gate(gate, final=False, acknowledged_override=False)
    enforce_qa_gate(gate, final=False, acknowledged_override=True)


def test_final_cli_refuses_auto_scope_and_gate_or_raw_overrides():
    with pytest.raises(PermissionError, match="group_scope=auto"):
        supervised_main(["--dry-run", "--final", "--group-scope", "auto"])
    with pytest.raises(ValueError, match="acknowledge-open-qa-gate"):
        supervised_main(["--dry-run", "--final", "--acknowledge-open-qa-gate"])
    with pytest.raises(ValueError, match="raw-XML"):
        supervised_main(["--dry-run", "--final", "--allow-raw-xml-fallback"])


def _variant(root: Path, test_images: int = 1, test_class: int = 0):
    split_rows = []
    for split, count in (("train", 1), ("valid", 1), ("test", test_images)):
        coco_images, annotations = [], []
        for index in range(count):
            stem = f"{split}_{index}"
            image_path = root / "MTSD-YOLO" / split / "images" / f"{stem}.jpg"
            color = {"train": (10, 0, 0), "valid": (0, 10, 0), "test": (0, 0, 10)}[split]
            image_path.parent.mkdir(parents=True, exist_ok=True); Image.new("RGB", (10, 10), color).save(image_path)
            label = root / "MTSD-YOLO" / split / "labels" / f"{stem}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            class_id = test_class if split == "test" else 0
            label.write_text(f"{class_id} 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            coco_image = root / "MTSD-COCO" / split / f"{stem}.jpg"
            coco_image.parent.mkdir(parents=True, exist_ok=True); Image.new("RGB", (10, 10), color).save(coco_image)
            coco_images.append({"id": index + 1, "file_name": f"{stem}.jpg", "width": 10, "height": 10})
            annotations.append({"id": index + 1, "image_id": index + 1, "category_id": class_id,
                                "bbox": [3, 3, 4, 4], "area": 16, "iscrowd": 0})
            split_rows.append({"source_image_sha256": sha256_file(image_path), "split": split,
                               "out_name": f"{stem}.jpg"})
        coco = {"images": coco_images, "annotations": annotations,
                "categories": [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES)]}
        path = root / "MTSD-COCO" / split / "_annotations.coco.json"
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(coco), encoding="utf-8")
    split_path = root / "split_manifest.csv"
    with split_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_image_sha256", "split", "out_name"])
        writer.writeheader(); writer.writerows(split_rows)
    manifest = {"class_names": CLASS_NAMES, "group_scope": "explicit", "approved_groups": [], "groups": [],
                "split_manifest_sha256": sha256_file(split_path)}
    write_manifest(root / "prep_manifest.json", manifest)


def test_missing_rare_class_warns_but_strict_succeeds(tmp_path):
    _variant(tmp_path)
    result = validate_variant(tmp_path, strict=True, policy={"require_all_classes_in_test": False,
        "minimum_boxes_per_class": {"train": 0, "valid": 0, "test": 0}, "low_support_warning_threshold": 0})
    assert result["ok"] and result["warning_count"] > 0
    assert any(row["code"] == "class_absent_from_split" for row in result["findings"])


def test_empty_test_is_fatal_and_minimum_support_can_be_fatal(tmp_path):
    empty = tmp_path / "empty"; _variant(empty, test_images=0)
    result = validate_variant(empty, policy={"require_all_classes_in_test": False,
        "minimum_boxes_per_class": {"train": 0, "valid": 0, "test": 0}, "low_support_warning_threshold": 0})
    assert not result["ok"] and any(row["code"] == "empty_required_split" for row in result["findings"])
    minimum = tmp_path / "minimum"; _variant(minimum)
    result = validate_variant(minimum, policy={"require_all_classes_in_test": False,
        "minimum_boxes_per_class": {"train": 0, "valid": 0, "test": 1}, "low_support_warning_threshold": 0})
    assert not result["ok"] and any(row["code"] == "minimum_class_support" for row in result["findings"])
