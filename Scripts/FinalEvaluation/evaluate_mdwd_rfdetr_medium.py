"""Evaluate a supplied RF-DETR-M checkpoint on MDWD-RFDETR test.

The checkpoint used for the original MDWD scale comparison emits one-based
class IDs, whereas the COCO export is zero-based.  Predictions are therefore
converted from 1..5 to the dataset's 0..4 category IDs before COCO evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Scripts" / "MTSD-Scripts" / "MTSD-SupervisedDetection"))
from mtsd_detection.evaluate import coco_eval, normalize_clamped_detection  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def class_audit(predictions, annotations, categories, threshold: float) -> list[dict]:
    """Class-wise greedy IoU=0.50 matching at one shared score threshold."""
    gt = defaultdict(list)
    for row in annotations:
        gt[(int(row["image_id"]), int(row["category_id"]))].append(row["bbox"])
    pred = defaultdict(list)
    for row in predictions:
        if float(row["score"]) >= threshold:
            pred[(int(row["image_id"]), int(row["category_id"]))].append(row)

    output = []
    for category in categories:
        category_id = int(category["id"])
        tp = fp = fn = 0
        matched_ious = []
        keys = {key for key in gt if key[1] == category_id} | {key for key in pred if key[1] == category_id}
        for key in keys:
            used = set()
            for item in sorted(pred[key], key=lambda row: float(row["score"]), reverse=True):
                best_iou, best_index = 0.0, None
                for index, box in enumerate(gt[key]):
                    if index not in used and (value := iou(item["bbox"], box)) > best_iou:
                        best_iou, best_index = value, index
                if best_index is not None and best_iou >= 0.50:
                    used.add(best_index)
                    tp += 1
                    matched_ious.append(best_iou)
                else:
                    fp += 1
            fn += len(gt[key]) - len(used)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        output.append({
            "class": str(category["name"]), "support": sum(len(boxes) for (__, cid), boxes in gt.items() if cid == category_id),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "MDWD-RF-DETR-M.pt")
    parser.add_argument("--dataset", type=Path, default=ROOT / "Datasets/MDWD/MDWD-RFDETR")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--audit-confidence", type=float, default=0.25,
                        help="Fixed confidence operating point for P/R/F1 and matched IoU.")
    parser.add_argument("--output", type=Path, default=ROOT / "Results/MDWD-Results/RF-DETR/RF-DETR-M-Test-Evaluation")
    args = parser.parse_args()
    checkpoint, annotation = args.checkpoint.resolve(), (args.dataset / args.split / "_annotations.coco.json").resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    truth = json.loads(annotation.read_text(encoding="utf-8"))
    images = {row["file_name"]: row for row in truth["images"]}
    category_ids = {int(row["id"]) for row in truth["categories"]}

    import torch
    from rfdetr import RFDETRMedium

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RFDETRMedium(pretrain_weights=str(checkpoint), resolution=args.resolution, device=device)
    parameter_count = sum(parameter.numel() for parameter in model.model.model.parameters())
    started = time.perf_counter()
    predictions, invalid_class_ids, dropped = [], set(), 0
    for index, (name, image) in enumerate(images.items(), start=1):
        detection = model.predict(str(args.dataset / args.split / name), threshold=0.001)
        for xyxy, score, class_id in zip(detection.xyxy, detection.confidence, detection.class_id):
            category_id = int(class_id) - 1  # checkpoint's verified one-based output convention
            if category_id not in category_ids:
                invalid_class_ids.add(int(class_id))
                continue
            row, _ = normalize_clamped_detection(int(image["id"]), category_id, xyxy, float(score), int(image["width"]), int(image["height"]))
            if row is None:
                dropped += 1
            else:
                predictions.append(row)
        if index % 50 == 0 or index == len(images):
            print(f"Predicted {index}/{len(images)} images")
    elapsed = time.perf_counter() - started
    prediction_path = output / "test_predictions_coco.json"
    prediction_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    metrics = coco_eval(annotation, prediction_path, artifacts_dir=output / "coco_artifacts")
    audit_threshold = float(args.audit_confidence)
    class_rows = class_audit(predictions, truth["annotations"], truth["categories"], audit_threshold)
    audit_tp = sum(row["true_positives"] for row in class_rows)
    audit_fp = sum(row["false_positives"] for row in class_rows)
    audit_fn = sum(row["false_negatives"] for row in class_rows)
    audit_precision = audit_tp / (audit_tp + audit_fp) if audit_tp + audit_fp else 0.0
    audit_recall = audit_tp / (audit_tp + audit_fn) if audit_tp + audit_fn else 0.0
    audit_f1 = 2 * audit_precision * audit_recall / (audit_precision + audit_recall) if audit_precision + audit_recall else 0.0
    with (output / "class_level_fixed_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)
    summary = {
        "model": "RF-DETR-M", "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "model_parameters": parameter_count, "model_parameters_million": parameter_count / 1e6,
        "checkpoint_size_mb": checkpoint.stat().st_size / 1024 ** 2, "dataset": str(args.dataset.resolve()),
        "split": args.split, "resolution": args.resolution, "device": device, "images": len(images),
        "raw_predictions": len(predictions), "dropped_degenerate_predictions": dropped,
        "invalid_raw_class_ids": sorted(invalid_class_ids), "inference_seconds": elapsed,
        "map50": metrics["map50"], "map50_95": metrics["map50_95"],
        "precision": audit_precision, "recall": audit_recall, "f1": audit_f1,
        "fixed_audit_confidence_threshold": audit_threshold, "fixed_audit_iou_threshold": 0.50,
        "fixed_audit_true_positives": audit_tp, "fixed_audit_false_positives": audit_fp,
        "fixed_audit_false_negatives": audit_fn,
        "best_f1_operating_point": {"confidence": metrics["f1_score_threshold"], "precision": metrics["precision"],
                                      "recall": metrics["recall"], "f1": metrics["f1"]},
        "class_level_results": class_rows,
    }
    (output / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
