import json
import random
from pathlib import Path

import pytest

from mtsd_detection.annotation_sources import CLASS_NAMES, discover_sources
from mtsd_detection.manifests import write_manifest
from mtsd_detection.model_registry import DEFAULT_ORDER, registry, resolve_checkpoint, resolve_checkpoint_info
from mtsd_detection.splitting import assign_splits
from mtsd_detection.config import load_config
from mtsd_detection.utils import find_repo_root


def test_default_model_order_and_family_checkpoint(tmp_path):
    assert len(DEFAULT_ORDER) == 13
    assert DEFAULT_ORDER == ["yolo12n", "yolo12s", "yolo12m", "yolo26n", "yolo26s", "yolo26m",
                             "yolo11n", "yolo11s", "yolo11m", "rfdetr-n", "rfdetr-s", "rfdetr-m", "yolo26l"]
    path = tmp_path / "Models" / "YOLO12" / "yolo12n.pt"
    path.parent.mkdir(parents=True); path.write_bytes(b"x")
    assert resolve_checkpoint(registry()["yolo12n"], tmp_path) == path
    info = resolve_checkpoint_info(registry()["yolo12n"], tmp_path)
    assert info["source"] == "local_repository" and len(info["sha256"]) == 64
    with pytest.raises(FileNotFoundError, match="Automatic downloads"):
        resolve_checkpoint_info(registry()["yolo11n"], tmp_path)


def test_split_matches_notebook_algorithm():
    rows = [{"group": group, "out_name": f"{group}-{i}"} for group in ("GRP-1", "GRP-2") for i in range(13)]
    actual = assign_splits(rows, {"train": .8, "valid": .1, "test": .1}, 42)
    rng = random.Random(42); expected = {}
    for group in ("GRP-1", "GRP-2"):
        subset = sorted([r for r in rows if r["group"] == group], key=lambda r: r["out_name"]); rng.shuffle(subset)
        for i, row in enumerate(subset): expected[row["out_name"]] = "train" if i < 10 else "valid" if i < 11 else "test"
    assert actual == expected


def _qa(path: Path, image: Path):
    payload = {"images": [{"id": 1, "file_name": image.name, "source_image": image.as_posix(), "width": 10, "height": 10}],
               "annotations": [], "categories": [{"id": i + 1, "name": name} for i, name in enumerate(CLASS_NAMES)]}
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload), encoding="utf-8")


def test_qa_discovery_excludes_backup_and_wins_over_xml(tmp_path):
    image = tmp_path / "image.jpg"; image.write_bytes(b"x")
    qa = tmp_path / "Datasets/MTSD/Annotations/GRP-1/Final-QA/QA-GRP1.json"; _qa(qa, image.relative_to(tmp_path))
    _qa(qa.with_name("QA-GRP1.json.bak.json"), image.relative_to(tmp_path))
    xml = qa.parents[1] / "Fiverr-Annotations/a.xml"; xml.parent.mkdir(); xml.write_text("<bad/>")
    rows, groups, mode = discover_sources(tmp_path / "Datasets/MTSD/Annotations", tmp_path, True, True)
    assert len(rows) == 1 and groups[0]["source_type"] == "final_qa" and mode == "qa_only"


def test_raw_xml_requires_both_flags(tmp_path):
    root = tmp_path / "Datasets/MTSD/Annotations"
    xml = root / "GRP-4/Fiverr-Annotations/a.xml"; xml.parent.mkdir(parents=True)
    xml.write_text("<annotation><filename>a.jpg</filename><size><width>10</width><height>10</height></size></annotation>")
    with pytest.raises(PermissionError): discover_sources(root, tmp_path, True, False)


def test_explicit_scope_and_unexpected_group_policy(tmp_path):
    image = tmp_path / "image.jpg"; image.write_bytes(b"x")
    root = tmp_path / "Datasets/MTSD/Annotations"
    _qa(root / "GRP-1/Final-QA/QA-GRP1.json", image.relative_to(tmp_path))
    _qa(root / "GRP-2/Final-QA/QA-GRP2.json", image.relative_to(tmp_path))
    rows, groups, _ = discover_sources(root, tmp_path, group_scope="explicit",
                                       approved_groups=["GRP-1"], unexpected_group_policy="warn")
    assert {row["group"] for row in rows} == {"GRP-1"}
    assert any(row["group"] == "GRP-2" and row["status"] == "unexpected" for row in groups)
    with pytest.raises(ValueError, match="outside approved scope"):
        discover_sources(root, tmp_path, group_scope="explicit",
                         approved_groups=["GRP-1"], unexpected_group_policy="fail")


def test_explicit_scope_missing_or_invalid_approved_group_fails(tmp_path):
    root = tmp_path / "Datasets/MTSD/Annotations"
    (tmp_path / "Datasets/MTSD/GRP-9").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Approved group GRP-9"):
        discover_sources(root, tmp_path, group_scope="explicit", approved_groups=["GRP-9"])
    invalid = root / "GRP-1/Final-QA/QA-GRP1.json"
    invalid.parent.mkdir(parents=True); invalid.write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid QA JSON"):
        discover_sources(root, tmp_path, group_scope="explicit", approved_groups=["GRP-1"])


def test_manifest_fingerprint_stable_except_payload(tmp_path):
    first = write_manifest(tmp_path / "a.json", {"x": 1})
    second = write_manifest(tmp_path / "b.json", {"x": 1})
    assert first["manifest_fingerprint"] == second["manifest_fingerprint"]


def test_real_registry_resolves_all_required_family_subfolder_checkpoints():
    root = find_repo_root(Path(__file__).resolve())
    expected = {
        **{f"yolo{family}{scale}": root / "Models" / f"YOLO{family}" / f"yolo{family}{scale}.pt"
           for family in ("11", "12", "26") for scale in ("n", "s", "m", "l")},
        "rfdetr-n": root / "Models/RF-DETR/rf-detr-nano.pth",
        "rfdetr-s": root / "Models/RF-DETR/rf-detr-small.pth",
        "rfdetr-m": root / "Models/RF-DETR/rf-detr-medium.pth",
    }
    for key, path in expected.items():
        assert path.is_file(), path
        assert resolve_checkpoint(registry()[key], root).resolve() == path.resolve()


def test_default_scope_wandb_and_clean_ablation_configuration():
    root = find_repo_root(Path(__file__).resolve())
    config = load_config(root / "Scripts/MTSD-Scripts/MTSD-SupervisedDetection/config/default.yaml", root)
    assert config["annotations"]["group_scope"] == "explicit"
    assert config["annotations"]["approved_groups"] == ["GRP-1", "GRP-2", "GRP-3", "GRP-5", "GRP-6"]
    assert config["wandb"]["project"] == "MSc-MTSD-SupervisedDetection"
    assert config["matrix"]["aug_ablation"] == ["yolo11m", "yolo12m", "yolo26m"]
    assert config["matrix"]["aug_ablation_optional"] == ["yolo26l"]
