"""Real-consumer compatibility over a tiny synthetic prepared dataset.

Builds an actual prepared dataset with the production ``prepare()`` function
(both variants, deterministic photometric augmentation, one EXIF-oriented
image) and then instantiates the real downstream loaders:

- Ultralytics (YOLO11/12/26 all share this loading path): data.yaml
  resolution + label parsing;
- torchvision/pycocotools CocoDetection (the RF-DETR ingestion path);
- the MTSD attribute-classification crop loader (AttributeCropDataset) and
  its split policy (crops of one photo can never straddle splits).

Also covers pipeline guarantees not exercised elsewhere: rebuild/archival
protection, augmentation determinism across rebuilds, descendant-split
integrity, and cross-split duplicate-hash detection.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image

import mtsd_detection.prepare_dataset as prep
from mtsd_detection.annotation_sources import CLASS_NAMES
from mtsd_detection.dataset_validation import validate_variant
from mtsd_detection.utils import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[4]
ATTR_SUBPROJECT = REPO_ROOT / "Scripts" / "MTSD-Scripts" / "AttributeClassification"


def _build_config(tmp_path, n_images=24, copies=1):
    (tmp_path / "Scripts").mkdir(exist_ok=True)
    annotations = tmp_path / "Datasets/MTSD/Annotations"
    images_dir = tmp_path / "Datasets/MTSD/GRP-1/Images"
    images_dir.mkdir(parents=True, exist_ok=True)
    images, rows = [], []
    for index in range(n_images):
        path = images_dir / f"image_{index}.jpg"
        image = Image.new("RGB", (64, 48), (index * 9 % 255, 40, 200))
        if index == 0:  # one EXIF-oriented image exercises both export paths
            exif = Image.Exif(); exif[274] = 6
            image.save(path, quality=95, exif=exif)
            width, height = 48, 64
        else:
            image.save(path, quality=95)
            width, height = 64, 48
        images.append({"id": index + 1, "file_name": path.name,
                       "source_image": path.relative_to(tmp_path).as_posix(),
                       "width": width, "height": height})
        rows.append({"id": index + 1, "image_id": index + 1,
                     "category_id": (index % len(CLASS_NAMES)) + 1,
                     "bbox": [4, 4, 20, 16]})
    qa = annotations / "GRP-1/Final-QA/QA-GRP1.json"
    qa.parent.mkdir(parents=True, exist_ok=True)
    qa.write_text(json.dumps({
        "images": images, "annotations": rows,
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CLASS_NAMES)]}),
        encoding="utf-8")
    gate = tmp_path / "gate.yaml"
    gate.write_text(
        "gate_version: v1\naudit_path: audit\naudit_timestamp: now\n"
        "audited_groups: [GRP-1]\napproved_scope: [GRP-1]\nfinding_counts: {}\n"
        "resolution_status: resolved\napproved_by: tester\n", encoding="utf-8")
    return {
        "repo_root": str(tmp_path),
        "dataset": {"annotations_root": str(annotations),
                    "prepared_root": str(tmp_path / "prepared"),
                    "version_base": "mtsd-qa-v1"},
        "annotations": {"group_scope": "explicit", "approved_groups": ["GRP-1"],
                        "unexpected_group_policy": "fail",
                        "qa_gate_file": str(gate)},
        "split": {"ratios": {"train": .8, "valid": .1, "test": .1}, "seed": 42},
        "augmentation": {"recipe_version": "photometric-v1",
                         "copies_per_image": copies, "seed": 42,
                         "ops": {"brightness": {"min": 0.9, "max": 1.1},
                                 "contrast": {"min": 0.9, "max": 1.1},
                                 "color": {"min": 0.95, "max": 1.05},
                                 "gaussian_blur": {"p": 0.5, "sigma_max": 0.6},
                                 "gaussian_noise": {"p": 0.3, "sigma_max": 0.01},
                                 "jpeg_compression": {"p": 0.3, "quality_min": 80,
                                                      "quality_max": 92},
                                 "gamma": {"min": 0.95, "max": 1.05}}},
        "validation": {"require_all_classes_in_test": False,
                       "minimum_boxes_per_class": {"train": 0, "valid": 0, "test": 0},
                       "low_support_warning_threshold": 0},
    }


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("consumers")
    config = _build_config(tmp_path)
    result = prep.prepare(config, variants="both", final=True)
    assert result["validation"]["ok"]
    return tmp_path, config


# ---------------------------------------------------------------------------
# Rebuild protection, augmentation determinism, split-descendant integrity
# ---------------------------------------------------------------------------

def test_rebuild_refused_without_flag_and_archives_with_it(tmp_path):
    config = _build_config(tmp_path, n_images=12)
    prep.prepare(config, variants="unaugmented", final=True)
    with pytest.raises(FileExistsError):
        prep.prepare(config, variants="unaugmented", final=True)
    prep.prepare(config, variants="unaugmented", final=True, rebuild=True)
    archived = list((tmp_path / "prepared" / "_archive").glob("MTSD-Unaugmented-*"))
    assert len(archived) == 1  # the old version was archived, not destroyed


def test_augmentation_deterministic_across_rebuilds(tmp_path):
    config = _build_config(tmp_path, n_images=12)

    def aug_hashes():
        manifest = (tmp_path / "prepared/MTSD-Augmented/augmentation_manifest.csv")
        lines = manifest.read_text(encoding="utf-8").splitlines()[1:]
        return sorted(line.split(",")[6] for line in lines if line)  # generated hash col

    prep.prepare(config, variants="augmented", final=True)
    first = aug_hashes()
    assert first, "augmentation produced no rows"
    prep.prepare(config, variants="augmented", final=True, rebuild=True)
    assert aug_hashes() == first  # bit-identical under the configured seed


def test_augmented_descendants_share_source_split(prepared):
    tmp_path, _ = prepared
    root = tmp_path / "prepared/MTSD-Augmented/MTSD-YOLO"
    train_stems = {p.stem for p in (root / "train/images").glob("*")}
    for split in ("valid", "test"):
        assert not [p for p in (root / split / "images").glob("*_aug*")]
    for aug in (root / "train/images").glob("*_aug*"):
        source_stem = aug.stem.rsplit("_aug", 1)[0]
        assert source_stem in train_stems  # descendant lives with its source


def test_cross_split_duplicate_hash_is_fatal(prepared):
    tmp_path, config = prepared
    variant = tmp_path / "prepared/MTSD-Unaugmented"
    tampered = tmp_path / "tampered"
    shutil.copytree(variant, tampered)
    split_csv = tampered / "split_manifest.csv"
    lines = split_csv.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    hash_col, split_col = header.index("source_image_sha256"), header.index("split")
    rows = [line.split(",") for line in lines[1:] if line]
    train_hash = next(r[hash_col] for r in rows if r[split_col] == "train")
    for row in rows:
        if row[split_col] == "test":
            row[hash_col] = train_hash  # simulate one photo leaking across splits
            break
    split_csv.write_text("\n".join([",".join(header)] +
                                   [",".join(r) for r in rows]) + "\n",
                         encoding="utf-8")
    result = validate_variant(tampered, strict=True, policy=config["validation"])
    codes = {f["code"] for f in result["findings"]}
    assert "hash_leakage" in codes and not result["ok"]


# ---------------------------------------------------------------------------
# Real consumer: Ultralytics (YOLO11 / YOLO12 / YOLO26 share this loader)
# ---------------------------------------------------------------------------

def test_ultralytics_reads_prepared_yolo_variant(prepared):
    ultralytics = pytest.importorskip("ultralytics")
    import numpy as np
    import yaml
    from ultralytics.data.utils import verify_image_label

    tmp_path, _ = prepared
    yolo_root = tmp_path / "prepared/MTSD-Augmented/MTSD-YOLO"
    payload = yaml.safe_load((yolo_root / "data.yaml").read_text(encoding="utf-8"))
    assert payload["nc"] == len(CLASS_NAMES) and payload["names"] == CLASS_NAMES
    for split_key, rel in (("train", payload["train"]), ("val", payload["val"]),
                           ("test", payload["test"])):
        image_dir = (yolo_root / "train" / Path(rel)).resolve() if not Path(rel).is_absolute() \
            else Path(rel)
        image_dir = (yolo_root / Path(rel.replace("../", ""))).resolve()
        assert image_dir.is_dir(), f"{split_key}: {image_dir}"
        images = sorted(image_dir.glob("*.jpg"))
        assert images
        for image_path in images[:4]:
            label_path = image_dir.parent / "labels" / f"{image_path.stem}.txt"
            # Ultralytics' own per-sample verifier: flags any of the problems
            # the pipeline is supposed to have prevented. Returns
            # (im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg).
            (_, label, shape, _, _, n_missing, n_found, _, n_corrupt,
             message) = verify_image_label(
                (str(image_path), str(label_path), "", False,
                 len(CLASS_NAMES), 0, 0, False))
            assert n_corrupt == 0, f"{image_path}: {message}"
            assert n_missing == 0 and n_found == 1
            assert label is not None and np.asarray(label).shape[1] == 5
            assert shape[0] > 0 and shape[1] > 0


# ---------------------------------------------------------------------------
# Real consumer: torchvision CocoDetection (RF-DETR ingestion path)
# ---------------------------------------------------------------------------

def test_torchvision_coco_detection_reads_prepared_coco_variant(prepared):
    torchvision = pytest.importorskip("torchvision")
    pytest.importorskip("pycocotools")
    from torchvision.datasets import CocoDetection

    tmp_path, _ = prepared
    for split in ("train", "valid", "test"):
        root = tmp_path / "prepared/MTSD-Augmented/MTSD-COCO" / split
        dataset = CocoDetection(str(root), str(root / "_annotations.coco.json"))
        assert len(dataset) > 0
        image, targets = dataset[0]
        assert image.size[0] > 0 and image.size[1] > 0
        for target in targets:
            x, y, w, h = target["bbox"]
            assert w > 0 and h > 0 and x >= 0 and y >= 0
            assert 0 <= target["category_id"] < len(CLASS_NAMES)
        names = [c["name"] for c in sorted(dataset.coco.dataset["categories"],
                                           key=lambda c: c["id"])]
        assert names == CLASS_NAMES
        # The EXIF-oriented image must load with QA (displayed) dimensions in
        # this loader, which does NOT apply EXIF orientation.
        record = next((row for row in dataset.coco.dataset["images"]
                       if row["file_name"] == "grp1_image_0.jpg"), None)
        if record is not None:
            with Image.open(root / record["file_name"]) as img:
                assert (img.width, img.height) == (record["width"], record["height"])


# ---------------------------------------------------------------------------
# Real consumer: attribute-classification crop loader and split policy
# ---------------------------------------------------------------------------

def test_attribute_crop_loader_and_split_policy(tmp_path):
    if str(ATTR_SUBPROJECT) not in sys.path:
        sys.path.insert(0, str(ATTR_SUBPROJECT))
    torch = pytest.importorskip("torch")
    from mtsd_attr.data_manifest import split_of
    from mtsd_attr.dataset import (MISSING_LABEL, AttributeCropDataset,
                                   build_transforms)

    fractions = {"train": 0.8, "val": 0.1, "test": 0.1}
    # Same-photo determinism: the split is a pure function of the image key,
    # so every crop of one photo inherits one split - forever.
    for index in range(200):
        key = f"GRP-9/img_{index}.jpg"
        first = split_of(key, "mtsd-attr-split-v1", fractions)
        assert split_of(key, "mtsd-attr-split-v1", fractions) == first
        assert first in fractions

    crops = tmp_path / "outputs/crops/GRP-1"
    crops.mkdir(parents=True)
    Image.new("RGB", (30, 44), (200, 10, 10)).save(crops / "ann_000001.jpg")
    records = [{"crop_id": "GRP-1/a.jpg#1", "group": "GRP-1",
                "image_key": "GRP-1/a.jpg", "source_image": "x",
                "annotation_id": 1, "category": "Stop Sign",
                "bbox": [0, 0, 30, 44],
                "crop_path": "outputs/crops/GRP-1/ann_000001.jpg",
                "labels": {"view_angle": "Front", "mounting": None},
                "split": "test"}]
    attributes = {"view_angle": {"classes": ["Front", "Back", "Side"]},
                  "mounting": {"classes": ["Pole-Mounted", "Wall-Mounted"]}}
    dataset = AttributeCropDataset(records, attributes, tmp_path,
                                   build_transforms(32, False, {}))
    tensor, target, crop_id = dataset[0]
    assert tensor.shape == (3, 32, 32)
    assert target.tolist() == [0, MISSING_LABEL]  # masked label policy
    assert crop_id == "GRP-1/a.jpg#1"
