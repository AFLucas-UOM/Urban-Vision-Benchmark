import json

import pytest
from PIL import Image

from mtsd_detection.evaluate import (
    clamp_coco_bbox,
    coco_eval,
    load_rgb_image,
    normalize_clamped_detection,
    normalize_detection,
    safe_unified_evaluation,
    validate_coco_results,
    xyxy_to_coco,
)


def _annotation(path, two_objects=False):
    annotations = [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20],
                    "area": 400, "iscrowd": 0, "segmentation": []}]
    if two_objects:
        annotations.append({"id": 2, "image_id": 1, "category_id": 0, "bbox": [50, 50, 10, 10],
                            "area": 100, "iscrowd": 0, "segmentation": []})
    payload = {"info": {}, "licenses": [], "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
               "categories": [{"id": 0, "name": "A", "supercategory": "x"}], "annotations": annotations}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _evaluate(tmp_path, rows, two_objects=False):
    pytest.importorskip("pycocotools")
    ann, pred = tmp_path / "ann.json", tmp_path / "pred.json"
    _annotation(ann, two_objects); pred.write_text(json.dumps(rows), encoding="utf-8")
    return coco_eval(ann, pred)


def test_coordinate_and_framework_normalization_equivalence():
    assert xyxy_to_coco([10, 20, 30, 50]) == [10.0, 20.0, 20.0, 30.0]
    yolo = normalize_detection(1, 0, [10, 20, 30, 50], .9)
    rfdetr = normalize_detection(1, 0, [10, 20, 30, 50], .9)
    assert yolo == rfdetr == {"image_id": 1, "category_id": 0,
                             "bbox": [10.0, 20.0, 20.0, 30.0], "score": .9}


def test_prediction_boxes_are_clamped_and_degenerate_boxes_are_dropped():
    bbox, changed = clamp_coco_bbox([-5, 10, 20, 100], width=100, height=50)
    assert changed and bbox == [0.0, 10.0, 15.0, 40.0]
    row, changed = normalize_clamped_detection(1, 0, [100, 0, 100, 20], .9, 100, 100)
    assert changed and row is None


def test_rgba_inputs_are_converted_to_rgb(tmp_path):
    image_path = tmp_path / "rgba.png"
    Image.new("RGBA", (4, 3), (1, 2, 3, 128)).save(image_path)
    converted = load_rgb_image(image_path)
    assert converted.mode == "RGB" and converted.size == (4, 3)


def test_category_mismatch_rejected(tmp_path):
    payload = _annotation(tmp_path / "ann.json")
    with pytest.raises(ValueError, match="unknown category_id"):
        validate_coco_results(payload, [{"image_id": 1, "category_id": 99,
                                         "bbox": [1, 1, 2, 2], "score": .5}])


def test_unified_failure_is_nonfatal_unless_explicitly_required():
    def fail(): raise ValueError("synthetic failure")
    result = safe_unified_evaluation(fail)
    assert result["unified_eval_status"] == "failed" and "synthetic" in result["unified_eval_error"]
    with pytest.raises(RuntimeError, match="Required unified"):
        safe_unified_evaluation(fail, require=True)


def test_perfect_and_empty_detection(tmp_path):
    perfect = _evaluate(tmp_path, [{"image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20], "score": .9}])
    assert perfect["map50_95"] > .99 and perfect["max_dets"] == [1, 10, 100]
    assert perfect["precision"] > .99 and perfect["recall"] > .99 and perfect["f1"] > .99
    empty = _evaluate(tmp_path, [])
    assert empty["map50_95"] == 0 and empty["evaluated_image_count"] == 1 and empty["f1"] == 0


def test_high_score_false_positive_and_missed_object_reduce_ap(tmp_path):
    false_positive = _evaluate(tmp_path, [
        {"image_id": 1, "category_id": 0, "bbox": [70, 70, 10, 10], "score": .99},
        {"image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20], "score": .9},
    ])
    assert 0 < false_positive["map50"] < 1
    missed = _evaluate(tmp_path, [
        {"image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20], "score": .9},
    ], two_objects=True)
    assert 0 < missed["map50"] < 1


def test_category_exclusion_and_artifact_exports(tmp_path):
    pytest.importorskip("pycocotools")
    annotation = tmp_path / "ann.json"
    predictions = tmp_path / "pred.json"
    artifacts = tmp_path / "artifacts"
    payload = _annotation(annotation)
    payload["categories"].append({"id": 1, "name": "Tourist Sign", "supercategory": "x"})
    payload["annotations"].append({"id": 2, "image_id": 1, "category_id": 1,
                                   "bbox": [50, 50, 10, 10], "area": 100,
                                   "iscrowd": 0, "segmentation": []})
    annotation.write_text(json.dumps(payload), encoding="utf-8")
    predictions.write_text(json.dumps([
        {"image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20], "score": .9},
        {"image_id": 1, "category_id": 1, "bbox": [50, 50, 10, 10], "score": .8},
    ]), encoding="utf-8")
    metrics = coco_eval(annotation, predictions, excluded_category_names=["Tourist Sign"],
                        artifacts_dir=artifacts)
    assert metrics["map50_95"] > .99
    assert metrics["excluded_category_names"] == ["Tourist Sign"]
    assert metrics["excluded_prediction_count"] == 1
    assert metrics["f1"] > .99 and metrics["f1_score_threshold"] == pytest.approx(.9)
    assert {row["category_name"] for row in metrics["per_class"]} == {"A"}
    for name in ("metrics_summary.json", "per_class_metrics.csv", "pr_curves_iou50.csv",
                 "confidence_f1_curve.csv", "pr_curves_iou50.png", "confidence_f1_curve.png"):
        assert (artifacts / name).is_file()
