import csv
import json
import random
import shutil
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest
import yaml

from mtsd_detection.annotation_sources import CLASS_NAMES
from mtsd_detection.dataset_validation import _validate_augmentation_manifests
from mtsd_detection.strong_augmentation import (
    apply_photometric,
    generate_copy_paste_sample,
    generate_mosaic_sample,
    generate_strong_augmentation,
    transform_boxes,
    validate_horizontal_flip_config,
)
from mtsd_detection.utils import sha256_file

HERE = Path(__file__).resolve().parents[1]


def _strong_config():
    payload = yaml.safe_load((HERE / "config/default.yaml").read_text(encoding="utf-8"))
    return deepcopy(payload["strong_augmentation"])


def _record(tmp_path, index, class_id=0, split="train", box=(30, 30, 70, 70)):
    source = tmp_path / "sources" / f"image_{index}.jpg"
    source.parent.mkdir(parents=True, exist_ok=True)
    values = np.zeros((100, 100, 3), dtype=np.uint8)
    values[:, :, 0] = 30 + index * 20
    values[20:80, 20:80, 1] = 150
    Image.fromarray(values).save(source, quality=95, subsampling=0)
    x0, y0, x1, y1 = box
    return {
        "source_path": source,
        "source_image_sha256": sha256_file(source),
        "out_name": source.name,
        "file_name": source.name,
        "group": "GRP-1",
        "width": 100,
        "height": 100,
        "split": split,
        "clean_boxes": [{
            "class_id": class_id,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "source_annotation_id": index + 1,
        }],
    }


def _base_variant(root, records):
    coco_images = []
    coco_annotations = []
    annotation_id = 1
    for image_id, record in enumerate(records, 1):
        yolo_image = root / "MTSD-YOLO/train/images" / record["out_name"]
        yolo_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["source_path"], yolo_image)
        label = root / "MTSD-YOLO/train/labels" / f"{Path(record['out_name']).stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for box in record["clean_boxes"]:
            cx = (box["x0"] + box["x1"]) / 200
            cy = (box["y0"] + box["y1"]) / 200
            bw = (box["x1"] - box["x0"]) / 100
            bh = (box["y1"] - box["y0"]) / 100
            lines.append(f"{box['class_id']} {cx} {cy} {bw} {bh}")
            coco_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": box["class_id"],
                "bbox": [box["x0"], box["y0"], box["x1"] - box["x0"], box["y1"] - box["y0"]],
                "area": (box["x1"] - box["x0"]) * (box["y1"] - box["y0"]),
                "iscrowd": 0,
            })
            annotation_id += 1
        label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        coco_image = root / "MTSD-COCO/train" / record["out_name"]
        coco_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["source_path"], coco_image)
        coco_images.append({
            "id": image_id,
            "file_name": record["out_name"],
            "width": 100,
            "height": 100,
        })
    payload = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": index, "name": name} for index, name in enumerate(CLASS_NAMES)],
    }
    (root / "MTSD-COCO/train/_annotations.coco.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )


def test_photometric_augmentation_is_deterministic():
    config = _strong_config()["photometric"]
    image = np.full((64, 96, 3), 127, dtype=np.uint8)
    first, first_ops = apply_photometric(
        image, random.Random(1234), np.random.default_rng(5678), config,
    )
    second, second_ops = apply_photometric(
        image, random.Random(1234), np.random.default_rng(5678), config,
    )
    assert np.array_equal(first, second)
    assert first_ops == second_ops


def test_rotation_scaling_translation_and_clipping_are_bbox_aware():
    affine = cv2.getRotationMatrix2D((50, 50), 5.0, 1.2)
    affine[:, 2] += (12, -9)
    matrix = np.vstack([affine, [0, 0, 1]])
    boxes = [{"class_id": 0, "x0": 10, "y0": 10, "x1": 90, "y1": 90}]
    box_filter = {
        "visible_area_threshold": 0.20,
        "min_width_px": 2,
        "min_height_px": 2,
        "min_area_px": 4,
    }
    kept, transformed, clipped, discarded = transform_boxes(
        boxes, matrix, (100, 100), box_filter, "source.jpg",
    )
    assert len(kept) == 1 and transformed and clipped and not discarded
    assert 0 <= kept[0]["x0"] < kept[0]["x1"] <= 100
    assert 0 <= kept[0]["y0"] < kept[0]["y1"] <= 100


@pytest.mark.parametrize(
    ("name", "matrix", "expected"),
    [
        (
            "translation",
            np.asarray([[1, 0, 7], [0, 1, 11], [0, 0, 1]], dtype=np.float64),
            [17, 31, 37, 51],
        ),
        (
            "scaling",
            np.asarray([[1.5, 0, 0], [0, 0.5, 0], [0, 0, 1]], dtype=np.float64),
            [15, 10, 45, 20],
        ),
        (
            "rotation",
            np.asarray([[0, -1, 100], [1, 0, 0], [0, 0, 1]], dtype=np.float64),
            [60, 10, 80, 30],
        ),
        (
            "shear",
            np.asarray([[1, 0.2, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
            [14, 20, 38, 40],
        ),
        (
            "crop_and_resize",
            np.asarray([[1.25, 0, -12.5], [0, 5 / 3, -100 / 3], [0, 0, 1]], dtype=np.float64),
            [0, 0, 25, 100 / 3],
        ),
    ],
)
def test_independent_affine_box_coordinates(name, matrix, expected):
    del name
    boxes = [{"class_id": 0, "x0": 10, "y0": 20, "x1": 30, "y1": 40}]
    box_filter = {
        "visible_area_threshold": 0.0,
        "min_width_px": 0.1,
        "min_height_px": 0.1,
        "min_area_px": 0.01,
    }
    kept, transformed, _, discarded = transform_boxes(
        boxes, matrix, (100, 100), box_filter, "source.jpg",
    )
    assert not discarded
    assert len(kept) == len(transformed) == 1
    assert transformed[0]["output_bbox_xyxy"] == pytest.approx(expected, abs=1e-6)


def test_independent_perspective_box_coordinates():
    matrix = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.002, 0.001, 1.0],
    ], dtype=np.float64)
    source_box = [10.0, 20.0, 30.0, 40.0]
    source_points = np.asarray([
        [10.0, 20.0, 1.0],
        [30.0, 20.0, 1.0],
        [30.0, 40.0, 1.0],
        [10.0, 40.0, 1.0],
    ])
    homogeneous = (matrix @ source_points.T).T
    expected_points = homogeneous[:, :2] / homogeneous[:, 2:3]
    expected = [
        expected_points[:, 0].min(),
        expected_points[:, 1].min(),
        expected_points[:, 0].max(),
        expected_points[:, 1].max(),
    ]
    box_filter = {
        "visible_area_threshold": 0.0,
        "min_width_px": 0.1,
        "min_height_px": 0.1,
        "min_area_px": 0.01,
    }
    _, transformed, _, discarded = transform_boxes(
        [{
            "class_id": 0,
            "x0": source_box[0],
            "y0": source_box[1],
            "x1": source_box[2],
            "y1": source_box[3],
        }],
        matrix,
        (100, 100),
        box_filter,
        "source.jpg",
    )
    assert not discarded
    assert transformed[0]["output_bbox_xyxy"] == pytest.approx(expected, abs=1e-6)


def test_crop_visibility_threshold_and_tiny_box_rejection():
    translated = np.asarray([[1, 0, 50], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    box = [{"class_id": 0, "x0": 20, "y0": 20, "x1": 80, "y1": 80}]
    strict_filter = {
        "visible_area_threshold": 0.60,
        "min_width_px": 2,
        "min_height_px": 2,
        "min_area_px": 4,
    }
    kept, _, _, discarded = transform_boxes(
        box, translated, (100, 100), strict_filter, "source.jpg",
    )
    assert not kept
    assert discarded[0]["reason"] == "visible_area_below_threshold"
    tiny_filter = {**strict_filter, "visible_area_threshold": 0.0, "min_area_px": 5000}
    kept, _, _, discarded = transform_boxes(
        box, np.eye(3), (100, 100), tiny_filter, "source.jpg",
    )
    assert not kept
    assert discarded[0]["reason"] == "area_below_minimum"


def test_motion_blur_is_deterministic_and_horizontal():
    config = _strong_config()["photometric"]
    for name in ("brightness", "contrast", "saturation_or_color", "gamma"):
        config[name]["probability"] = 0.0
    for name in ("gaussian_blur", "gaussian_noise", "jpeg_compression"):
        config[name]["probability"] = 0.0
    for name in (
        "partial_shadow_probability",
        "gradient_probability",
        "local_change_probability",
        "haze_probability",
    ):
        config["illumination"][name] = 0.0
    config["motion_blur"] = {"probability": 1.0, "kernel_sizes": [7]}
    image = np.zeros((31, 31, 3), dtype=np.uint8)
    image[:, 15, :] = 255
    first, first_ops = apply_photometric(
        image, random.Random(5), np.random.default_rng(6), config,
    )
    second, second_ops = apply_photometric(
        image, random.Random(5), np.random.default_rng(6), config,
    )
    assert np.array_equal(first, second)
    assert first_ops == second_ops == [{
        "name": "motion_blur",
        "direction": "horizontal",
        "kernel_size": 7,
    }]
    assert np.count_nonzero(first[15, :, 0]) == 7
    assert np.count_nonzero(first[:, 15, 0]) == 31


def test_four_image_mosaic_transforms_all_source_boxes(tmp_path):
    records = [_record(tmp_path, index, class_id=index) for index in range(4)]
    config = _strong_config()
    config["mosaic"]["canvas_size"] = [200, 200]
    config["mosaic"]["centre_range"] = [0.5, 0.5]
    config["mosaic"]["photometric_probability"] = 0.0
    image, boxes, audit = generate_mosaic_sample(
        records[0], records, random.Random(7), np.random.default_rng(8), config,
    )
    assert image.shape == (200, 200, 3)
    assert len(audit["mosaic_source_positions"]) == 4
    assert {box["class_id"] for box in boxes} == {0, 1, 2, 3}
    assert all(0 <= box["x0"] < box["x1"] <= 200 for box in boxes)
    assert all(0 <= box["y0"] < box["y1"] <= 200 for box in boxes)


def test_horizontal_flip_requires_validated_directional_remap():
    geometry = _strong_config()["geometric"]
    geometry["horizontal_flip"]["enabled"] = True
    geometry["horizontal_flip"]["probability"] = 1.0
    with pytest.raises(ValueError, match="remap_validated"):
        validate_horizontal_flip_config(geometry, len(CLASS_NAMES))
    geometry["horizontal_flip"]["remap_validated"] = True
    geometry["horizontal_flip"]["direction_sensitive_class_ids"] = [1, 2]
    geometry["horizontal_flip"]["class_remap"] = {"1": 2, "2": 1}
    assert validate_horizontal_flip_config(geometry, len(CLASS_NAMES)) == {1: 2, 2: 1}


def test_rare_class_copy_paste_adds_annotation(tmp_path):
    destination = _record(tmp_path, 0, class_id=0, box=(5, 5, 25, 25))
    source = _record(tmp_path, 1, class_id=1, box=(35, 35, 65, 65))
    config = _strong_config()
    config["copy_paste"]["maximum_paste_count"] = 1
    config["copy_paste"]["photometric_probability"] = 0.0
    image, boxes, audit = generate_copy_paste_sample(
        destination,
        [destination, source],
        {1},
        [100, 1] + [0] * (len(CLASS_NAMES) - 2),
        random.Random(18),
        np.random.default_rng(19),
        config,
    )
    assert image.shape == (100, 100, 3)
    assert len(boxes) == 2
    assert boxes[-1]["class_id"] == 1
    assert audit["copy_paste_sources"][0]["source_annotation_id"] == 2


def test_rare_class_copy_paste_rejects_tiny_pasted_box(tmp_path):
    destination = _record(tmp_path, 0, class_id=0, box=(5, 5, 25, 25))
    source = _record(tmp_path, 1, class_id=1, box=(35, 35, 36, 36))
    config = _strong_config()
    config["copy_paste"]["maximum_paste_count"] = 1
    config["copy_paste"]["photometric_probability"] = 0.0
    with pytest.raises(ValueError, match="below_minimum"):
        generate_copy_paste_sample(
            destination,
            [destination, source],
            {1},
            [100, 1] + [0] * (len(CLASS_NAMES) - 2),
            random.Random(18),
            np.random.default_rng(19),
            config,
        )


def test_strong_generation_is_reproducible_and_yolo_coco_equivalent(tmp_path):
    records = [_record(tmp_path, index, class_id=index % 3) for index in range(4)]
    config = _strong_config()
    config["mosaic"]["canvas_size"] = [200, 200]
    config["mosaic"]["centre_range"] = [0.5, 0.5]
    config["mosaic"]["photometric_probability"] = 0.0
    config["copy_paste"]["photometric_probability"] = 0.0
    roots = [tmp_path / "run_a", tmp_path / "run_b"]
    for root in roots:
        _base_variant(root, records)
        generate_strong_augmentation(
            root,
            records,
            config,
            CLASS_NAMES,
            category_plan=["photometric", "geometric", "mosaic", "copy_paste"],
            source_pool=records,
        )
    assert (roots[0] / "augmentation_manifest.jsonl").read_bytes() == \
           (roots[1] / "augmentation_manifest.jsonl").read_bytes()
    first_images = sorted((roots[0] / "MTSD-YOLO/train/images").glob("*augstrong*"))
    second_images = sorted((roots[1] / "MTSD-YOLO/train/images").glob("*augstrong*"))
    assert [sha256_file(path) for path in first_images] == [sha256_file(path) for path in second_images]
    coco = json.loads((roots[0] / "MTSD-COCO/train/_annotations.coco.json").read_text())
    image_by_name = {row["file_name"]: row for row in coco["images"]}
    annotations = {}
    for row in coco["annotations"]:
        annotations.setdefault(row["image_id"], []).append(row)
    for image_path in first_images:
        image_row = image_by_name[image_path.name]
        label = roots[0] / "MTSD-YOLO/train/labels" / f"{image_path.stem}.txt"
        yolo_rows = [line.split() for line in label.read_text().splitlines()]
        coco_rows = annotations[image_row["id"]]
        assert len(yolo_rows) == len(coco_rows)
        assert [int(row[0]) for row in yolo_rows] == [row["category_id"] for row in coco_rows]


def test_split_leakage_in_augmentation_manifest_is_fatal(tmp_path):
    train = _record(tmp_path, 0, split="train")
    valid = _record(tmp_path, 1, split="valid")
    variant = tmp_path / "variant"
    generated = "image_0_augstrong1.jpg"
    yolo_image = variant / "MTSD-YOLO/train/images" / generated
    coco_image = variant / "MTSD-COCO/train" / generated
    yolo_image.parent.mkdir(parents=True)
    coco_image.parent.mkdir(parents=True)
    shutil.copy2(train["source_path"], yolo_image)
    shutil.copy2(train["source_path"], coco_image)
    row = {
        "recipe_version": "strong-offline-v2",
        "random_seed": 42,
        "generated_filename": generated,
        "source_images": [valid["out_name"]],
        "source_image_hashes": [valid["source_image_sha256"]],
        "split": "train",
        "generated_image_hash": sha256_file(yolo_image),
    }
    jsonl = variant / "augmentation_manifest.jsonl"
    jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
    split_rows = [
        {"out_name": train["out_name"], "source_image_sha256": train["source_image_sha256"], "split": "train"},
        {"out_name": valid["out_name"], "source_image_sha256": valid["source_image_sha256"], "split": "valid"},
    ]
    findings = []
    _validate_augmentation_manifests(
        variant,
        {"augmentation_manifest_jsonl_sha256": sha256_file(jsonl)},
        split_rows,
        findings,
    )
    assert any(row["code"] == "augmentation_source_split_leakage" for row in findings)
    assert any(row["code"] == "augmentation_source_hash_leakage" for row in findings)


def test_strong_mosaic_sources_are_train_only(tmp_path):
    train_records = [_record(tmp_path, index, class_id=index % 3) for index in range(4)]
    valid_record = _record(tmp_path, 9, class_id=4, split="valid")
    root = tmp_path / "train_only"
    _base_variant(root, train_records)
    config = _strong_config()
    config["mosaic"]["canvas_size"] = [200, 200]
    config["mosaic"]["centre_range"] = [0.5, 0.5]
    config["mosaic"]["photometric_probability"] = 0.0
    generate_strong_augmentation(
        root,
        [*train_records, valid_record],
        config,
        CLASS_NAMES,
        category_plan=["mosaic"],
    )
    audit = json.loads((root / "augmentation_manifest.jsonl").read_text().strip())
    assert valid_record["out_name"] not in audit["source_images"]
    assert set(audit["source_images"]).issubset({
        record["out_name"] for record in train_records
    })


def test_existing_mild_recipe_configuration_is_preserved():
    payload = yaml.safe_load((HERE / "config/default.yaml").read_text(encoding="utf-8"))
    mild = payload["augmentation"]
    assert mild["recipe_version"] == "photometric-v1+motion-blur-v1"
    assert mild["copies_per_image"] == 3
    assert mild["ops"]["brightness"] == {"min": 0.8, "max": 1.2}
    assert mild["ops"]["motion_blur"] == {
        "copy_index": 3,
        "direction": "horizontal",
        "kernel_size": 9,
        "blur_weight": 0.85,
    }
