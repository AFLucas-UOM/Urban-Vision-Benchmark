from pathlib import Path
import hashlib
import json

import pytest

from protocol import load_protocol, select_prompts, validate_vocabulary
from protocol_reporting import generate
from dataset_loader import _load_yolo_split
from persistence import combination_dir, compatible_complete, write_json
from run_dissertation_protocol import enforce_final_mtsd, main as protocol_main
from targeted_metrics import deduplicate_union, evaluate_targeted

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_fixed_semantics():
    protocol = load_protocol(ROOT / "prompt_protocols/dissertation_protocol.yaml")
    mtsd = protocol["datasets"]["MTSD"]
    traffic = next(row for row in mtsd["prompts"] if row["id"] == "mtsd-p12")
    optional = next(row for row in mtsd["prompts"] if row["id"] == "mtsd-opt1")
    assert "Blind-Spot Mirror (Convex Mirror)" not in traffic["target_classes"]
    assert len(optional["target_classes"]) == 12
    assert optional not in select_prompts(mtsd)
    assert [row["id"] for row in mtsd["prompts"] if row["group"] == "synonym-comparison"] == ["mtsd-p3", "mtsd-p4", "mtsd-p6", "mtsd-p7"]


def test_unknown_target_rejected_before_models():
    spec = {"class_vocabulary": ["A"], "prompts": [{"id": "p", "target_classes": ["TYPO"]}]}
    with pytest.raises(ValueError, match="TYPO"): validate_vocabulary("X", spec, ["A"])


def test_targeted_negatives_and_nontarget_overlap_count_as_fp():
    records = [{"image_id": "positive", "boxes": [{"class_name": "A", "x0": 0, "y0": 0, "x1": 10, "y1": 10}]},
               {"image_id": "negative", "boxes": [{"class_name": "B", "x0": 0, "y0": 0, "x1": 10, "y1": 10}]}]
    predictions = {"positive": [{"x0": 0, "y0": 0, "x1": 10, "y1": 10, "score": .9}],
                   "negative": [{"x0": 0, "y0": 0, "x1": 10, "y1": 10, "score": .8}]}
    result = evaluate_targeted(predictions, records, ["A"], .5, [.5])
    assert result["summary"]["tp"] == 1 and result["summary"]["fp"] == 1
    assert result["summary"]["negative_images"] == 1
    assert result["fp_nontarget_overlap"][0]["fp_overlap_class"] == "B"


def test_deduplicated_union_is_deterministic():
    boxes = [{"x0": 0, "y0": 0, "x1": 10, "y1": 10, "score": 1},
             {"x0": 1, "y0": 1, "x1": 9, "y1": 9, "score": 1}]
    assert deduplicate_union(boxes) == [boxes[0]]


def test_duplicate_prediction_and_empty_predictions():
    records = [{"image_id": "x", "boxes": [{"class_name": "A", "x0": 0, "y0": 0, "x1": 10, "y1": 10}]}]
    duplicate = {"x": [{"x0": 0, "y0": 0, "x1": 10, "y1": 10, "score": .9},
                       {"x0": 0, "y0": 0, "x1": 10, "y1": 10, "score": .8}]}
    result = evaluate_targeted(duplicate, records, ["A"], .5, [.5])
    assert result["summary"]["tp"] == 1 and result["summary"]["fp"] == 1 and result["summary"]["duplicates"] == 1
    empty = evaluate_targeted({}, records, ["A"], .5, [.5])
    assert empty["summary"]["tp"] == 0 and empty["summary"]["fn"] == 1


def test_broad_targets_and_zero_target_slice():
    records = [{"image_id": "x", "boxes": [
        {"class_name": "A", "x0": 0, "y0": 0, "x1": 10, "y1": 10},
        {"class_name": "B", "x0": 20, "y0": 20, "x1": 30, "y1": 30}]}]
    preds = {"x": [{"x0": 0, "y0": 0, "x1": 10, "y1": 10, "score": .9},
                   {"x0": 20, "y0": 20, "x1": 30, "y1": 30, "score": .8}]}
    broad = evaluate_targeted(preds, records, ["A", "B"], .5, [.5])
    assert broad["summary"]["tp"] == 2 and broad["summary"]["fn"] == 0
    zero = evaluate_targeted(preds, records, ["C"], .5, [.5])
    assert zero["summary"]["target_gt_boxes"] == 0 and zero["summary"]["fp"] == 2 and zero["summary"]["fn"] == 0


def test_traffic_sign_excludes_mirror_and_confusion_is_diagnostic_only():
    records = [{"image_id": "x", "boxes": [{"class_name": "Blind-Spot Mirror (Convex Mirror)",
                                               "x0": 0, "y0": 0, "x1": 10, "y1": 10}]}]
    preds = {"x": [{"x0": 0, "y0": 0, "x1": 10, "y1": 10, "score": .9}]}
    result = evaluate_targeted(preds, records, ["Stop Sign"], .5, [.5])
    assert result["summary"]["tp"] == 0 and result["summary"]["fp"] == 1 and result["summary"]["fn"] == 0
    assert result["fp_nontarget_overlap"][0]["fp_overlap_class"] == "Blind-Spot Mirror (Convex Mirror)"
    assert result["prompt_class_confusion"] == {"Blind-Spot Mirror (Convex Mirror)": 1}


def test_headline_includes_synonyms_but_excludes_broad_and_ap(tmp_path):
    base = {"dataset": "X", "split": "test", "model": "M", "target_classes": ["A"],
            "precision": 1, "recall": 1, "accuracy": 1, "mean_matched_iou": 1,
            "mean_inference_ms": 10, "prompt_class_confusion": {}}
    rows = [
        {**base, "prompt_id": "target", "prompt": "target", "prompt_group": "class-targeted", "f1": 1, "ap50": 0},
        {**base, "prompt_id": "syn", "prompt": "synonym", "prompt_group": "synonym-comparison", "f1": .5, "ap50": 0},
        {**base, "prompt_id": "broad", "prompt": "broad", "prompt_group": "broad", "f1": 0, "ap50": 1},
    ]
    summary = generate(tmp_path, {"protocol_version": "v", "dataset": "X"}, rows, [], [])
    headline = summary["per_model_summary"][0]
    assert headline["f1"] == .75 and headline["n_headline_prompts"] == 2
    assert headline["ap_used_for_ranking"] is False
    assert "ap50" not in headline and "map50_95" not in headline
    persisted = json.loads((tmp_path / "evaluation_summary.json").read_text())
    assert persisted["evaluation_protocol"] == "targeted-v1" and len(persisted["per_prompt_metrics"]) == 3


def test_partial_resume_skips_only_compatible_completed_combination(tmp_path):
    expected = {"protocol_hash": "p", "dataset_manifest_hash": "d", "model": "M",
                "prompt_id": "one", "conf_threshold": .3, "iou_threshold": .5, "split": "test"}
    directory = combination_dir(tmp_path, "M", "one")
    write_json(directory / "status.json", {**expected, "status": "completed"})
    assert compatible_complete(directory, expected)
    assert not compatible_complete(combination_dir(tmp_path, "M", "two"), {**expected, "prompt_id": "two"})
    with pytest.raises(ValueError, match="Incompatible"):
        compatible_complete(directory, {**expected, "protocol_hash": "changed"})


def test_final_mtsd_refuses_qa_fallback():
    spec = {"source": {"path": "canonical", "required_group_scope": "explicit",
                       "approved_groups": ["GRP-1"], "required_annotation_source_mode": "qa_only"}}
    with pytest.raises(RuntimeError, match="canonical prepared source"):
        enforce_final_mtsd({"source": "qa", "source_type": "qa_fallback"}, spec)
    with pytest.raises(ValueError, match="incompatible"):
        protocol_main(["--dataset", "MTSD", "--dry-run", "--final", "--allow-qa-fallback"])


def test_final_mtsd_accepts_only_verified_locked_manifest(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "prepared/MTSD-Unaugmented/MTSD-YOLO"; source.mkdir(parents=True)
    split = source.parent / "split_manifest.csv"; split.write_text("out_name,split\na.jpg,test\n", encoding="utf-8")
    approved = ["GRP-1"]
    manifest = {"group_scope": "explicit", "approved_groups": approved,
                "annotation_source_mode": "qa_only", "split_manifest_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
                "qa_gate": {"resolved": True, "override_used": False, "resolution_status": "resolved"}}
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    manifest["manifest_fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    manifest_path = source.parent / "prep_manifest.json"; manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    spec = {"source": {"path": "prepared/MTSD-Unaugmented/MTSD-YOLO",
                       "required_group_scope": "explicit", "approved_groups": approved,
                       "required_annotation_source_mode": "qa_only"}}
    gt = {"source": str(source), "source_type": "prepared", "manifest_path": str(manifest_path), "manifest": manifest}
    decisions = enforce_final_mtsd(gt, spec)
    assert decisions["manifest_verified"] and decisions["split_manifest_verified"]
    bad = dict(manifest); bad["group_scope"] = "auto"; gt["manifest"] = bad
    with pytest.raises(RuntimeError): enforce_final_mtsd(gt, spec)


def test_promptdetect_prepared_test_membership_is_folder_membership(tmp_path):
    from PIL import Image
    import yaml
    root = tmp_path / "MTSD-YOLO"
    (root / "test/images").mkdir(parents=True); (root / "test/labels").mkdir(parents=True)
    for name in ("b.jpg", "a.jpg"):
        Image.new("RGB", (10, 10)).save(root / "test/images" / name)
        (root / "test/labels" / f"{Path(name).stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "data.yaml").write_text(yaml.safe_dump({"names": ["A"]}), encoding="utf-8")
    records, _ = _load_yolo_split(root, "test", None)
    assert [row["image_id"] for row in records] == ["a.jpg", "b.jpg"]
