from pathlib import Path
import csv
import hashlib
import json

import pytest
from PIL import Image

import config
from backend import CosmosReason2Engine, Sam3NativeEngine
from protocol import load_protocol, select_prompts, validate_vocabulary
from protocol_reporting import generate
from dataset_loader import _load_yolo_split
from persistence import combination_dir, compatible_complete, write_json
from run_dissertation_protocol import (
    derive_detection_limit,
    enforce_final_mtsd,
    main as protocol_main,
    worker_chunk_ranges,
    worker_chunk_size,
)
from targeted_metrics import deduplicate_union, evaluate_targeted
from utils import filter_detections
from wandb_utils import _wandb_id, start_run, tracking_target
from visualizations import save_visualizations

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_fixed_semantics():
    # mtsd-opt1 (optional-broad) was promoted to mtsd-p13 (broad) in f244d47.
    protocol = load_protocol(ROOT / "prompt_protocols/dissertation_protocol.yaml")
    mtsd = protocol["datasets"]["MTSD"]
    traffic = next(row for row in mtsd["prompts"] if row["id"] == "mtsd-p12")
    all_objects = next(row for row in mtsd["prompts"] if row["id"] == "mtsd-p13")
    assert "Blind-Spot Mirror (Convex Mirror)" not in traffic["target_classes"]
    assert len(all_objects["target_classes"]) == 12
    assert all_objects["group"] == "broad" and all_objects in select_prompts(mtsd)
    assert [row["id"] for row in mtsd["prompts"] if row["group"] == "synonym-comparison"] == ["mtsd-p3", "mtsd-p4", "mtsd-p6", "mtsd-p7"]


def test_fixed_protocols_lock_current_eleven_group_mtsd_scope():
    expected = [f"GRP-{index}" for index in range(1, 12)]
    for filename in ("dissertation_protocol.yaml", "prompt_sensitivity_protocol.yaml"):
        protocol = load_protocol(ROOT / "prompt_protocols" / filename)
        assert protocol["datasets"]["MTSD"]["source"]["approved_groups"] == expected


def test_promptdetect_wandb_target_is_fixed_for_final_comparisons():
    assert tracking_target() == "mark-bugeja-university-of-malta/MSc-MDWD-MDWD-PromptDetect"


def test_wandb_init_is_pinned_to_requested_target(tmp_path, monkeypatch):
    import wandb

    captured = {}

    def fake_init(**kwargs):
        captured.update(kwargs)
        return type("Run", (), {"id": "offline-id", "url": None})()

    monkeypatch.setattr(wandb, "init", fake_init)
    start_run(tmp_path / "run", {"dataset": "MDWD", "evaluation_protocol": "targeted-v2",
                                  "split": "test"}, "offline")
    assert captured["entity"] == "mark-bugeja-university-of-malta"
    assert captured["project"] == "MSc-MDWD-MDWD-PromptDetect"
    assert captured["mode"] == "offline"


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


def test_filter_detections_applies_score_order_nms_then_cap():
    boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]]
    scores = [.8, .9, .7]
    kept, kept_scores, _, _ = filter_detections(
        boxes, scores, ["x"] * 3, max_detections=2, nms_iou_threshold=.5,
    )
    assert kept == [boxes[1], boxes[2]]
    assert kept_scores == [.9, .7]


def test_targeted_visualization_uses_prompt_taxonomy_and_limit(tmp_path):
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)
    gt = {"dataset": "X", "split": "test", "records": [{
        "image_id": "sample.jpg", "image_path": image_path,
        "boxes": [
            {"class_name": "target", "x0": 10, "y0": 10, "x1": 50, "y1": 50},
            {"class_name": "other", "x0": 100, "y0": 10, "x1": 150, "y1": 50},
        ],
    }]}
    predictions = [{
        "model": "M", "prompt": "find target", "image_id": "sample.jpg",
        "x0": 100, "y0": 10, "x1": 150, "y1": 50, "score": 1,
        "predicted_label": "other", "has_confidence": "False",
    }]
    rows = save_visualizations(
        tmp_path / "run", gt, predictions, ["M"],
        [{"id": "p1", "prompt": "find target", "target_classes": ["target"]}],
        .5, max_images=10,
    )
    assert len(rows) == 1
    assert rows[0]["prompt_id"] == "p1"
    assert rows[0]["tp"] == 0 and rows[0]["fp"] == 1 and rows[0]["fn"] == 1
    assert (tmp_path / "run" / "visualizations" / "index.html").is_file()

    save_visualizations(
        tmp_path / "run", gt, [], ["M"],
        [{"id": "p2", "prompt": "another target wording", "target_classes": ["target"]}],
        .5, max_images=10,
    )
    with (tmp_path / "run" / "visualizations" / "visualization_index.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        indexed = list(csv.DictReader(stream))
    assert len(indexed) == 2
    assert {row["prompt_id"] for row in indexed} == {"p1", "p2"}


def test_detection_limit_is_derived_from_complete_test_gt():
    gt = {"split": "test", "records": [
        {"image_id": "one", "boxes": [{}]},
        {"image_id": "busy", "boxes": [{}, {}, {}, {}]},
    ]}
    result = derive_detection_limit(gt, {"max_detections_policy": "test-set-max-gt-per-image"})
    assert result == {
        "value": 4,
        "policy": "test-set-max-gt-per-image",
        "source_split": "test",
        "source_image_id": "busy",
        "source_image_gt_boxes": 4,
    }
    with pytest.raises(ValueError, match="test split"):
        derive_detection_limit({**gt, "split": "valid"},
                               {"max_detections_policy": "test-set-max-gt-per-image"})


def test_worker_chunks_are_aggressive_for_vlms_but_not_sam():
    defaults = {"vlm_worker_chunk_size": 64}
    assert worker_chunk_ranges("SAM 3", 130, defaults) == [(0, 130)]
    assert worker_chunk_ranges("LocateAnything 3B", 130, defaults) == [
        (0, 64), (64, 128), (128, 130),
    ]
    assert worker_chunk_size("Cosmos Reason2 2B", 130, defaults) == 8
    assert worker_chunk_ranges("Cosmos Reason2 2B", 18, defaults) == [
        (0, 8), (8, 16), (16, 18),
    ]


def test_cosmos_resolution_policy_is_model_specific_and_applied_only_to_cosmos():
    import inspect

    assert config.cosmos_max_side("Cosmos Reason2 2B") == 2560
    assert config.cosmos_max_side("Cosmos Reason2 8B") == 1536
    assert "PROMPTDETECT_COSMOS_MAX_SIDE" in inspect.getsource(CosmosReason2Engine.predict_raw)
    assert "PROMPTDETECT_COSMOS_MAX_SIDE" not in inspect.getsource(Sam3NativeEngine.predict_raw)


def test_replacement_wandb_run_uses_a_fresh_versioned_id(tmp_path):
    run_id = _wandb_id(tmp_path / "20260818-134723-optimized-v2", {
        "dataset": "MTSD",
        "evaluation_protocol": "targeted-v2",
        "wandb_run_revision": "bounded-v2",
    })
    assert run_id.endswith("_bounded_v2")


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
    assert persisted["evaluation_protocol"] == "targeted-v2" and len(persisted["per_prompt_metrics"]) == 3


def test_partial_resume_skips_only_compatible_completed_combination(tmp_path):
    expected = {"protocol_hash": "p", "dataset_manifest_hash": "d", "model": "M",
                "prompt_id": "one", "conf_threshold": .3, "iou_threshold": .5, "split": "test"}
    directory = combination_dir(tmp_path, "M", "one")
    write_json(directory / "status.json", {**expected, "status": "completed"})
    assert compatible_complete(directory, expected)
    assert not compatible_complete(combination_dir(tmp_path, "M", "two"), {**expected, "prompt_id": "two"})
    with pytest.raises(ValueError, match="Incompatible"):
        compatible_complete(directory, {**expected, "protocol_hash": "changed"})
    with pytest.raises(ValueError, match="max_detections"):
        compatible_complete(directory, {**expected, "max_detections": 18})


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
