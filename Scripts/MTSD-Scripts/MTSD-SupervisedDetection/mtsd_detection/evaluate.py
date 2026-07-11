from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


IOU_THRESHOLDS = [round(0.50 + 0.05 * index, 2) for index in range(10)]
MAX_DETS = [1, 10, 100]


def safe_unified_evaluation(evaluator, require: bool = False) -> dict[str, Any]:
    try:
        return {"unified_eval_status": "completed", "unified_eval_error": None,
                "unified_test_metrics": evaluator()}
    except Exception as exc:
        if require:
            raise RuntimeError(f"Required unified COCO evaluation failed: {exc}") from exc
        return {"unified_eval_status": "failed", "unified_eval_error": str(exc),
                "unified_test_metrics": {}}


def xyxy_to_coco(box: list[float] | tuple[float, float, float, float]) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in box]
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"Degenerate xyxy box: {box}")
    return [x0, y0, x1 - x0, y1 - y0]


def normalize_detection(image_id: int, category_id: int, xyxy, score: float) -> dict[str, Any]:
    return {"image_id": int(image_id), "category_id": int(category_id),
            "bbox": xyxy_to_coco(xyxy), "score": float(score)}


def validate_coco_results(annotation_payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    image_ids = {int(row["id"]) for row in annotation_payload.get("images", [])}
    category_ids = {int(row["id"]) for row in annotation_payload.get("categories", [])}
    for index, row in enumerate(rows):
        if int(row.get("image_id", -1)) not in image_ids:
            raise ValueError(f"Prediction {index} has unknown image_id {row.get('image_id')}")
        if int(row.get("category_id", -1)) not in category_ids:
            raise ValueError(f"Prediction {index} has unknown category_id {row.get('category_id')}")
        bbox = row.get("bbox", [])
        if len(bbox) != 4 or not all(math.isfinite(float(value)) for value in bbox):
            raise ValueError(f"Prediction {index} has invalid bbox {bbox}")
        if float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            raise ValueError(f"Prediction {index} has non-positive bbox {bbox}")
        score = float(row.get("score", -1))
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Prediction {index} has invalid score {score}")


def coco_eval(annotation_file: Path, predictions_file: Path) -> dict[str, Any]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("Unified evaluation requires pycocotools") from exc
    annotation_payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    rows = json.loads(predictions_file.read_text(encoding="utf-8"))
    validate_coco_results(annotation_payload, rows)
    truth = COCO(str(annotation_file))
    metadata = {"evaluator": "pycocotools.COCOeval", "iou_thresholds": IOU_THRESHOLDS,
                "max_dets": MAX_DETS, "prediction_count": len(rows),
                "evaluated_image_count": len(annotation_payload.get("images", []))}
    if not rows:
        return {"map50_95": 0.0, "map50": 0.0, "map75": 0.0, "ar100": 0.0, **metadata}
    predictions = truth.loadRes(str(predictions_file))
    evaluator = COCOeval(truth, predictions, "bbox")
    evaluator.params.maxDets = MAX_DETS
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    return {"map50_95": float(evaluator.stats[0]), "map50": float(evaluator.stats[1]),
            "map75": float(evaluator.stats[2]), "ar100": float(evaluator.stats[8]), **metadata}


def export_yolo_predictions(checkpoint: Path, coco_dir: Path, split: str, output: Path,
                            image_size: int, device: str) -> dict[str, Any]:
    from ultralytics import YOLO
    annotation = coco_dir / split / "_annotations.coco.json"
    truth = json.loads(annotation.read_text(encoding="utf-8"))
    image_ids = {row["file_name"]: row["id"] for row in truth["images"]}
    rows = []
    model = YOLO(str(checkpoint))
    resolved_device = None if device == "auto" else device
    for result in model.predict(source=str(coco_dir / split), imgsz=image_size, device=resolved_device,
                                conf=0.001, iou=0.7, max_det=100, stream=True, verbose=False):
        image_id = image_ids.get(Path(result.path).name)
        if image_id is None: continue
        for xyxy, score, class_id in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
            x0, y0, x1, y1 = xyxy
            rows.append(normalize_detection(image_id, int(class_id), [x0, y0, x1, y1], float(score)))
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return coco_eval(annotation, output)


def export_rfdetr_predictions(spec, checkpoint: Path, coco_dir: Path, split: str, output: Path,
                              image_size: int, device: str) -> dict[str, Any]:
    from .model_registry import _rfdetr_class
    import torch
    resolved_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model = _rfdetr_class(spec)(pretrain_weights=str(checkpoint), resolution=image_size,
                                gradient_checkpointing=False, device=resolved_device)
    annotation = coco_dir / split / "_annotations.coco.json"
    truth = json.loads(annotation.read_text(encoding="utf-8"))
    rows = []
    for image in truth["images"]:
        detections = model.predict(str(coco_dir / split / image["file_name"]), threshold=0.001)
        xyxy = getattr(detections, "xyxy", [])
        confidence = getattr(detections, "confidence", [])
        class_id = getattr(detections, "class_id", [])
        for box, score, category in zip(xyxy, confidence, class_id):
            x0, y0, x1, y1 = [float(value) for value in box]
            rows.append(normalize_detection(image["id"], int(category), [x0, y0, x1, y1], float(score)))
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return coco_eval(annotation, output)
