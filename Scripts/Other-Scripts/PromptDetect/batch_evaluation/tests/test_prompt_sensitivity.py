"""Tests for the controlled prompt-sensitivity extension (protocol validation,
GT-based aggregation, prediction consistency, reporting and resume behaviour).
All records are small synthetic fixtures; no checkpoints or GPU inference."""

import csv
import json
from pathlib import Path
from statistics import pstdev

import pytest
import yaml

from persistence import combination_dir, compatible_complete, write_csv, write_json
from prompt_sensitivity import (
    VARIANT_TYPES,
    family_statistics,
    model_macro_rows,
    pair_consistency,
    relative_degradation,
    validate_family_gt_counts,
    variant_difference_rows,
)
from protocol import load_protocol
from protocol_reporting import generate
from targeted_metrics import evaluate_targeted

ROOT = Path(__file__).resolve().parents[1]


# --- fixtures ----------------------------------------------------------------------

def _family(fid="famA", ids=("p1", "p2", "p3", "p4"), texts=None, targets=("A",), variants=None):
    variants = list(variants or VARIANT_TYPES)
    texts = list(texts or [f"{fid} wording {index}" for index in range(len(ids))])
    return [{"id": pid, "prompt": text, "group": "paraphrase-sensitivity",
             "target_classes": list(targets), "sensitivity_family": fid, "variant_type": variant}
            for pid, text, variant in zip(ids, texts, variants)]


def _write_protocol(tmp_path, prompts, vocab=("A", "B"), name="proto.yaml"):
    payload = {"protocol_version": "prompt-sensitivity-test",
               "defaults": {"conf_threshold": 0.3, "iou_threshold": 0.5, "max_detections": 100},
               "models": {"primary": ["sam3"]},
               "datasets": {"MDWD": {"class_vocabulary": list(vocab), "prompts": prompts}}}
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _metric_rows(f1s, fam="famA", model="M", dataset="D", prefix="p"):
    rows = []
    for index, value in enumerate(f1s):
        rows.append({"dataset": dataset, "model": model, "sensitivity_family": fam,
                     "prompt_id": f"{prefix}{index + 1}", "prompt": f"{fam} wording {index}",
                     "variant_type": VARIANT_TYPES[index], "target_classes": ["A"],
                     "precision": value, "recall": value, "f1": value, "mean_matched_iou": 0.5,
                     "target_gt_boxes": 3, "positive_images": 2, "negative_images": 1,
                     "mean_inference_ms": 10.0})
    return rows


BOX = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}


# --- protocol validation (tests 1-6, 19) ---------------------------------------------

def test_rejects_duplicate_prompt_ids(tmp_path):
    prompts = _family(ids=("p1", "p2", "p3", "p1"))
    with pytest.raises(ValueError, match="unique"):
        load_protocol(_write_protocol(tmp_path, prompts))


def test_rejects_missing_sensitivity_metadata(tmp_path):
    prompts = _family()
    del prompts[1]["variant_type"]
    with pytest.raises(ValueError, match="missing sensitivity metadata"):
        load_protocol(_write_protocol(tmp_path, prompts))


def test_rejects_family_with_different_target_classes(tmp_path):
    prompts = _family()
    prompts[2]["target_classes"] = ["B"]
    with pytest.raises(ValueError, match="mixes different target_classes"):
        load_protocol(_write_protocol(tmp_path, prompts))


def test_rejects_duplicate_prompt_text_in_family(tmp_path):
    prompts = _family(texts=("same wording", "same wording", "other", "another"))
    with pytest.raises(ValueError, match="duplicate prompt text"):
        load_protocol(_write_protocol(tmp_path, prompts))


def test_rejects_zero_or_multiple_canonical_variants(tmp_path):
    none = _family(variants=("lexical-paraphrase", "lexical-paraphrase", "descriptive-paraphrase", "instruction"))
    with pytest.raises(ValueError, match="exactly one canonical"):
        load_protocol(_write_protocol(tmp_path, none, name="none.yaml"))
    double = _family(variants=("canonical", "canonical", "descriptive-paraphrase", "instruction"))
    with pytest.raises(ValueError, match="exactly one canonical"):
        load_protocol(_write_protocol(tmp_path, double, name="double.yaml"))


def test_rejects_unknown_target_class_and_wrong_family_size(tmp_path):
    unknown = _family(targets=("TYPO",))
    with pytest.raises(ValueError, match="TYPO"):
        load_protocol(_write_protocol(tmp_path, unknown, name="unknown.yaml"))
    short = _family()[:3]
    with pytest.raises(ValueError, match="exactly 4"):
        load_protocol(_write_protocol(tmp_path, short, name="short.yaml"))


def test_existing_protocols_still_load():
    classic = load_protocol(ROOT / "prompt_protocols/dissertation_protocol.yaml")
    assert classic["protocol_version"] == "dissertation-v1"
    sensitivity = load_protocol(ROOT / "prompt_protocols/prompt_sensitivity_protocol.yaml")
    prompts = [row for spec in sensitivity["datasets"].values() for row in spec["prompts"]]
    families = {}
    for row in prompts:
        families.setdefault(row["sensitivity_family"], []).append(row)
    assert len(prompts) == 36 and len(families) == 9
    assert all(len(members) == 4 for members in families.values())
    assert len([f for f in families if f.startswith("mtsd-")]) == 5
    assert len([f for f in families if f.startswith("mdwd-")]) == 4
    # "one way sign" stays a synonym comparison, never a controlled paraphrase
    assert all(row["prompt"] != "one way sign" for row in prompts)
    assert all("Other Waste" not in row["target_classes"] for row in prompts)


def test_protocol_hash_changes_with_prompt_or_family_definition(tmp_path):
    base = load_protocol(_write_protocol(tmp_path, _family(), name="a.yaml"))
    reworded = _family()
    reworded[1]["prompt"] = "completely different wording"
    changed_text = load_protocol(_write_protocol(tmp_path, reworded, name="b.yaml"))
    renamed = _family(fid="famB")
    changed_family = load_protocol(_write_protocol(tmp_path, renamed, name="c.yaml"))
    assert base["protocol_hash"] != changed_text["protocol_hash"]
    assert base["protocol_hash"] != changed_family["protocol_hash"]


# --- ground-truth construction (tests 7-10) ------------------------------------------

def test_positive_gt_contains_only_declared_target_classes():
    records = [{"image_id": "x", "boxes": [{"class_name": "A", **BOX},
                                           {"class_name": "B", "x0": 20, "y0": 20, "x1": 30, "y1": 30}]}]
    result = evaluate_targeted({}, records, ["A"], .5, [.5])
    assert result["summary"]["target_gt_boxes"] == 1 and result["summary"]["fn"] == 1


def test_target_negative_images_remain_in_evaluation():
    records = [{"image_id": "pos", "boxes": [{"class_name": "A", **BOX}]},
               {"image_id": "neg", "boxes": []}]
    result = evaluate_targeted({}, records, ["A"], .5, [.5])
    assert result["summary"]["n_images"] == 2
    assert result["summary"]["positive_images"] == 1 and result["summary"]["negative_images"] == 1


def test_detection_on_target_negative_image_is_fp():
    records = [{"image_id": "neg", "boxes": []}]
    result = evaluate_targeted({"neg": [{**BOX, "score": .9}]}, records, ["A"], .5, [.5])
    assert result["summary"]["fp"] == 1 and result["summary"]["tp"] == 0


def test_fp_overlapping_non_target_class_is_recorded():
    records = [{"image_id": "x", "boxes": [{"class_name": "B", **BOX}]}]
    result = evaluate_targeted({"x": [{**BOX, "score": .9}]}, records, ["A"], .5, [.5])
    assert result["summary"]["fp"] == 1
    assert result["fp_nontarget_overlap"][0]["fp_overlap_class"] == "B"
    assert result["summary"]["fp_overlapping_nontarget"] == 1


def test_family_gt_count_validation():
    rows = _metric_rows([1, 1, 1, 1])
    rows[0]["target_gt_boxes"] = 3
    summary = validate_family_gt_counts(rows)
    assert summary == [{"sensitivity_family": "famA", "target_classes": ["A"], "n_prompts": 4,
                        "positive_images": 2, "negative_images": 1, "target_gt_boxes": 3}]
    rows[3]["positive_images"] = 99
    with pytest.raises(ValueError, match="identical target GT"):
        validate_family_gt_counts(rows)


# --- GT-based sensitivity statistics (tests 11-13) -----------------------------------

def test_family_statistics_spread_and_degradation():
    stats = family_statistics(_metric_rows([0.8, 0.6, 0.4, 0.2]))[0]
    assert stats["f1_mean"] == pytest.approx(0.5)
    assert stats["f1_std"] == pytest.approx(pstdev([0.8, 0.6, 0.4, 0.2]))
    assert stats["f1_min"] == 0.2 and stats["f1_max"] == 0.8
    assert stats["f1_range"] == pytest.approx(0.6)
    assert stats["best_prompt_id"] == "p1" and stats["best_variant_type"] == "canonical"
    assert stats["worst_prompt_id"] == "p4" and stats["worst_variant_type"] == "instruction"
    assert stats["f1_degradation_abs"] == pytest.approx(0.6)
    assert stats["f1_degradation_rel"] == pytest.approx(0.75)
    assert stats["all_variants_failed"] is False
    assert stats["canonical_f1"] == 0.8 and stats["instruction_f1"] == 0.2


def test_relative_degradation_handles_zero_max():
    assert relative_degradation(0.0, 0.0) == (0.0, True)
    stats = family_statistics(_metric_rows([0, 0, 0, 0]))[0]
    assert stats["f1_degradation_rel"] == 0.0 and stats["all_variants_failed"] is True


def test_canonical_variant_differences_sign():
    rows = variant_difference_rows(_metric_rows([0.8, 0.6, 0.9, 0.2]))
    by_id = {row["prompt_id"]: row for row in rows}
    assert set(by_id) == {"p2", "p3", "p4"}  # canonical p1 excluded
    assert by_id["p2"]["canonical_f1_difference"] == pytest.approx(-0.2)  # worse than canonical
    assert by_id["p3"]["canonical_f1_difference"] == pytest.approx(0.1)   # better than canonical
    assert all(row["canonical_f1"] == 0.8 for row in rows)


def test_model_macro_summary():
    rows = _metric_rows([0.8, 0.6, 0.4, 0.2], fam="famA") + \
           _metric_rows([1.0, 1.0, 1.0, 0.5], fam="famB", prefix="q")
    macro = model_macro_rows(family_statistics(rows), rows)[0]
    assert macro["n_families"] == 2
    assert macro["macro_mean_f1"] == pytest.approx((0.5 + 0.875) / 2)
    assert macro["mean_f1_range"] == pytest.approx((0.6 + 0.5) / 2)
    assert macro["max_f1_range"] == pytest.approx(0.6)
    assert macro["worst_family"] == "famA"  # rel 0.75 > famB rel 0.5
    assert macro["macro_worst_prompt_f1"] == pytest.approx((0.2 + 0.5) / 2)
    assert macro["macro_canonical_f1"] == pytest.approx((0.8 + 1.0) / 2)
    assert macro["macro_instruction_f1"] == pytest.approx((0.2 + 0.5) / 2)
    assert macro["mean_inference_ms"] == pytest.approx(10.0)


# --- prediction consistency (tests 14-16) --------------------------------------------

def test_pair_consistency_identical_boxes():
    boxes = [{**BOX, "score": .9}, {"x0": 20, "y0": 20, "x1": 30, "y1": 30, "score": .8}]
    result = pair_consistency(boxes, [dict(box) for box in boxes], .5)
    assert result["matched"] == 2 and result["pairwise_box_f1"] == 1.0
    assert result["pairwise_box_jaccard"] == 1.0 and result["both_empty"] is False
    assert result["mean_matched_pair_iou"] == pytest.approx(1.0)


def test_pair_consistency_partial_overlap():
    boxes_a = [{**BOX, "score": .9}, {"x0": 50, "y0": 50, "x1": 60, "y1": 60, "score": .8}]
    boxes_b = [{**BOX, "score": .7}]
    result = pair_consistency(boxes_a, boxes_b, .5)
    assert result["matched"] == 1
    assert result["pairwise_box_f1"] == pytest.approx(2 / 3)
    assert result["pairwise_box_jaccard"] == pytest.approx(1 / 2)
    assert result["unmatched_a"] == 1 and result["unmatched_b"] == 0
    disjoint = pair_consistency(boxes_a, [{"x0": 90, "y0": 90, "x1": 99, "y1": 99, "score": .5}], .5)
    assert disjoint["matched"] == 0 and disjoint["pairwise_box_f1"] == 0.0


def test_pair_consistency_both_empty():
    result = pair_consistency([], [], .5)
    assert result["both_empty"] is True
    assert result["pairwise_box_f1"] == 1.0 and result["pairwise_box_jaccard"] == 1.0
    one_sided = pair_consistency([{**BOX, "score": .9}], [], .5)
    assert one_sided["both_empty"] is False and one_sided["pairwise_box_f1"] == 0.0


# --- reports-only regeneration (test 17) ---------------------------------------------

def _synthetic_sensitivity_run(tmp_path):
    prompts = _family()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = {"evaluation_protocol": "prompt-sensitivity-v1", "protocol_version": "prompt-sensitivity-v1",
           "protocol_hash": "hash", "dataset_manifest_hash": "manifest", "dataset": "MDWD", "split": "test",
           "models": ["M"], "conf_threshold": 0.3, "iou_threshold": 0.5, "consistency_iou_threshold": 0.5,
           "max_detections": 100, "prompt_definitions": prompts,
           "prompt_ids": [row["id"] for row in prompts], "n_images": 2}
    write_json(run_dir / "run_config.json", cfg)
    write_csv(run_dir / "ground_truth_index.csv", [
        {"image_id": "img1", "image_path": "x", "class_name": "A", **BOX},
        {"image_id": "img2", "image_path": "y", "class_name": "", "x0": "", "y0": "", "x1": "", "y1": ""}])
    for index, prompt in enumerate(prompts):
        directory = combination_dir(run_dir, "M", prompt["id"])
        write_json(directory / "metrics.json", {
            "dataset": "MDWD", "split": "test", "model": "M", "prompt_id": prompt["id"],
            "prompt": prompt["prompt"], "prompt_group": prompt["group"],
            "target_classes": prompt["target_classes"], "sensitivity_family": prompt["sensitivity_family"],
            "variant_type": prompt["variant_type"], "prompt_class_confusion": {},
            "tp": 1, "fp": index, "fn": 0, "duplicates": 0,
            "precision": 1 / (1 + index), "recall": 1.0, "f1": 2 / (2 + index),
            "accuracy": 1 / (1 + index), "mean_matched_iou": 0.9, "ap50": 0.0, "map50_95": 0.0,
            "n_gt": 1, "n_images": 2, "target_gt_boxes": 1, "positive_images": 1, "negative_images": 1,
            "fp_overlapping_nontarget": 0, "mean_inference_ms": 5.0,
            "has_confidence": True, "ap_meaningful": True})
        write_csv(directory / "predictions.csv", [
            {"model": "M", "prompt": prompt["prompt"], "prompt_id": prompt["id"], "image_id": "img1",
             "image_path": "x", **BOX, "score": 0.9, "predicted_label": prompt["prompt"],
             "has_mask": False, "has_confidence": True, "inference_ms": 5}])
        write_csv(directory / "per_image.csv", [
            {"model": "M", "prompt_id": prompt["id"], "prompt": prompt["prompt"], "image_id": "img1",
             "n_gt": 1, "n_pred": 1, "tp": 1, "fp": 0, "fn": 0, "duplicates": 0,
             "precision": 1, "recall": 1, "f1": 1, "accuracy": 1}])
        write_csv(directory / "fp_nontarget_overlap.csv", [])
    return run_dir


def test_reports_only_regenerates_sensitivity_outputs(tmp_path):
    from run_dissertation_protocol import reports_only
    run_dir = _synthetic_sensitivity_run(tmp_path)
    assert reports_only(run_dir) == 0
    for name in ("prompt_sensitivity_per_prompt.csv", "prompt_sensitivity_per_family.csv",
                 "prompt_sensitivity_variant_differences.csv", "prompt_sensitivity_per_model.csv",
                 "prompt_pair_consistency_per_image.csv", "prompt_pair_consistency.csv",
                 "prompt_consistency_per_family.csv", "prompt_consistency_per_model.csv",
                 "prompt_sensitivity_summary.json", "prompt_sensitivity_report.md"):
        assert (run_dir / name).is_file(), f"missing {name}"
    summary = json.loads((run_dir / "prompt_sensitivity_summary.json").read_text(encoding="utf-8"))
    assert summary["evaluation_protocol"] == "prompt-sensitivity-v1"
    family = summary["per_family_sensitivity"][0]
    assert family["sensitivity_family"] == "famA" and family["n_prompts"] == 4
    # img1: all four prompts predicted the identical box; img2: nobody predicted.
    with (run_dir / "prompt_pair_consistency_per_image.csv").open(encoding="utf-8", newline="") as stream:
        image_rows = list(csv.DictReader(stream))
    assert len(image_rows) == 12  # 6 prompt pairs x 2 images
    assert all(float(row["pairwise_box_f1"]) == 1.0 for row in image_rows)
    empties = [row for row in image_rows if row["image_id"] == "img2"]
    assert empties and all(row["both_empty"] == "True" for row in empties)
    # per-image pairwise counts survive into the per-model macro
    per_model = summary["prediction_consistency"]["per_model"]
    assert per_model[0]["macro_pairwise_box_f1"] == 1.0


# --- classic outputs and resume compatibility (tests 18, 20) --------------------------

def test_targeted_v1_outputs_not_removed_or_renamed(tmp_path):
    base = {"dataset": "X", "split": "test", "model": "M", "target_classes": ["A"],
            "precision": 1, "recall": 1, "accuracy": 1, "mean_matched_iou": 1,
            "mean_inference_ms": 10, "prompt_class_confusion": {}, "f1": 1, "ap50": 0}
    rows = [{**base, "prompt_id": "t", "prompt": "t", "prompt_group": "class-targeted"}]
    generate(tmp_path, {"protocol_version": "v", "dataset": "X"}, rows, [], [])
    for name in ("targeted_per_prompt_metrics.csv", "per_prompt_metrics.csv", "per_image_metrics.csv",
                 "fp_nontarget_overlap.csv", "synonym_comparison.csv", "broad_prompt_metrics.csv",
                 "per_target_class_metrics.csv", "prompt_group_summary.csv", "per_model_summary.csv",
                 "full_comparison.csv", "confusion_fp_nontarget.csv", "prompt_vs_class_confusion.csv",
                 "evaluation_summary.json", "dissertation_report.md"):
        assert (tmp_path / name).exists(), f"missing {name}"
    summary = json.loads((tmp_path / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert summary["evaluation_protocol"] == "targeted-v1"  # default label preserved


def test_resume_rejects_stale_prompt_definition(tmp_path):
    expected = {"protocol_hash": "p", "dataset_manifest_hash": "d", "model": "M", "prompt_id": "p1",
                "prompt": "new wording", "target_classes": ["A"], "sensitivity_family": "famA",
                "variant_type": "canonical", "conf_threshold": .3, "iou_threshold": .5, "split": "test"}
    directory = combination_dir(tmp_path, "M", "p1")
    write_json(directory / "status.json", {**expected, "status": "completed"})
    assert compatible_complete(directory, expected)  # identical definition is reusable
    for key, stale in (("prompt", "old wording"), ("target_classes", ["B"]),
                       ("sensitivity_family", "famZ"), ("variant_type", "instruction")):
        with pytest.raises(ValueError, match="Incompatible"):
            compatible_complete(directory, {**expected, key: stale})
    # legacy expectations without prompt-identity keys still validate hash/thresholds only
    legacy = {key: expected[key] for key in ("protocol_hash", "dataset_manifest_hash", "model",
                                             "prompt_id", "conf_threshold", "iou_threshold", "split")}
    assert compatible_complete(directory, legacy)
