import json

import pytest

from mtsd_detection.evaluate import coco_eval, normalize_detection, safe_unified_evaluation, validate_coco_results, xyxy_to_coco


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
    empty = _evaluate(tmp_path, [])
    assert empty["map50_95"] == 0 and empty["evaluated_image_count"] == 1


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
