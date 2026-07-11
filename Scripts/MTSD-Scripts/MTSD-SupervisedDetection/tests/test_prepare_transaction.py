import json
from pathlib import Path

from PIL import Image

import mtsd_detection.prepare_dataset as prep
from mtsd_detection.annotation_sources import CLASS_NAMES
from mtsd_detection.utils import sha256_file


def test_one_authoritative_split_drives_both_variants_and_formats(tmp_path, monkeypatch):
    (tmp_path / "Scripts").mkdir(); annotations = tmp_path / "Datasets/MTSD/Annotations"
    images_dir = tmp_path / "Datasets/MTSD/GRP-1/Images"; images_dir.mkdir(parents=True)
    images, annotations_rows = [], []
    for index in range(10):
        path = images_dir / f"image_{index}.jpg"
        Image.new("RGB", (20, 20), (index * 20, index, 255 - index * 20)).save(path)
        images.append({"id": index + 1, "file_name": path.name,
                       "source_image": path.relative_to(tmp_path).as_posix(), "width": 20, "height": 20})
        annotations_rows.append({"id": index + 1, "image_id": index + 1,
                                 "category_id": (index % len(CLASS_NAMES)) + 1, "bbox": [2, 2, 10, 10]})
    qa = annotations / "GRP-1/Final-QA/QA-GRP1.json"; qa.parent.mkdir(parents=True)
    qa.write_text(json.dumps({"images": images, "annotations": annotations_rows,
                              "categories": [{"id": i + 1, "name": name} for i, name in enumerate(CLASS_NAMES)]}),
                  encoding="utf-8")
    gate = tmp_path / "gate.yaml"
    gate.write_text("gate_version: v1\naudit_path: audit\naudit_timestamp: now\naudited_groups: [GRP-1]\n"
                    "approved_scope: [GRP-1]\nfinding_counts: {}\nresolution_status: resolved\napproved_by: tester\n",
                    encoding="utf-8")
    config = {
        "repo_root": str(tmp_path),
        "dataset": {"annotations_root": str(annotations), "prepared_root": str(tmp_path / "prepared"),
                    "version_base": "mtsd-qa-v1"},
        "annotations": {"group_scope": "explicit", "approved_groups": ["GRP-1"],
                        "unexpected_group_policy": "fail", "qa_gate_file": str(gate)},
        "split": {"ratios": {"train": .8, "valid": .1, "test": .1}, "seed": 42},
        "augmentation": {"recipe_version": "photometric-v1", "copies_per_image": 1, "seed": 42,
            "ops": {"brightness": {"min": 1, "max": 1}, "contrast": {"min": 1, "max": 1},
                    "color": {"min": 1, "max": 1}, "gaussian_blur": {"p": 0, "sigma_max": 0},
                    "gaussian_noise": {"p": 0, "sigma_max": 0},
                    "jpeg_compression": {"p": 0, "quality_min": 90, "quality_max": 90},
                    "gamma": {"min": 1, "max": 1}}},
        "validation": {"require_all_classes_in_test": False,
                       "minimum_boxes_per_class": {"train": 0, "valid": 0, "test": 0},
                       "low_support_warning_threshold": 0},
    }
    original = prep.assign_splits; calls = []
    def counted(*args, **kwargs):
        calls.append(1); return original(*args, **kwargs)
    monkeypatch.setattr(prep, "assign_splits", counted)
    result = prep.prepare(config, variants="both", final=True)
    assert result["validation"]["ok"] and len(calls) == 1
    augmented = tmp_path / "prepared/MTSD-Augmented"
    plain = tmp_path / "prepared/MTSD-Unaugmented"
    assert (augmented / "split_manifest.csv").read_bytes() == (plain / "split_manifest.csv").read_bytes()
    for split in ("valid", "test"):
        for path in (plain / "MTSD-YOLO" / split / "images").glob("*"):
            assert sha256_file(path) == sha256_file(augmented / "MTSD-YOLO" / split / "images" / path.name)
        yolo = {path.stem for path in (plain / "MTSD-YOLO" / split / "images").glob("*")}
        coco_payload = json.loads((plain / "MTSD-COCO" / split / "_annotations.coco.json").read_text())
        assert yolo == {Path(row["file_name"]).stem for row in coco_payload["images"]}
    assert not list((augmented / "MTSD-YOLO/valid/images").glob("*_aug*"))
    assert not list((augmented / "MTSD-YOLO/test/images").glob("*_aug*"))
