from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from metrics import evaluate_prompt, iou_xyxy, match_image


def evaluate_targeted(predictions_by_image: dict[str, list[dict]], records: list[dict],
                      target_classes: list[str], iou_threshold: float,
                      map_iou_range: list[float]) -> dict[str, Any]:
    targets = set(target_classes)
    target_gt = {row["image_id"]: [box for box in row["boxes"] if box["class_name"] in targets] for row in records}
    non_target_gt = {row["image_id"]: [box for box in row["boxes"] if box["class_name"] not in targets] for row in records}
    evaluation = evaluate_prompt(predictions_by_image, target_gt, iou_threshold, map_iou_range)
    overlaps, confusion = [], Counter()
    prompt_class_confusion = Counter()
    for record in records:
        image_id = record["image_id"]
        result = match_image(predictions_by_image.get(image_id, []), target_gt[image_id], iou_threshold)
        for pred, is_tp in zip(result["preds_sorted"], result["pred_flags"]):
            if is_tp: continue
            candidates = [(iou_xyxy(pred, box), box) for box in non_target_gt[image_id]]
            best_iou, best_box = max(candidates, default=(0.0, None), key=lambda pair: pair[0])
            row = {"image_id": image_id, "fp_x0": pred["x0"], "fp_y0": pred["y0"],
                   "fp_x1": pred["x1"], "fp_y1": pred["y1"], "fp_score": pred.get("score", 0),
                   "fp_overlap_class": best_box["class_name"] if best_box and best_iou >= iou_threshold else "",
                   "fp_overlap_iou": round(best_iou, 4)}
            overlaps.append(row)
            if row["fp_overlap_class"]: confusion[row["fp_overlap_class"]] += 1
        all_gt_match = match_image(predictions_by_image.get(image_id, []), record["boxes"], iou_threshold)
        for _, matched_gt, _ in all_gt_match["matches"]:
            prompt_class_confusion[matched_gt["class_name"]] += 1
    summary = dict(evaluation["summary"])
    positives = sum(bool(boxes) for boxes in target_gt.values())
    summary.update(target_classes=list(target_classes), target_gt_boxes=sum(len(v) for v in target_gt.values()),
                   positive_images=positives, negative_images=len(records) - positives,
                   fp_overlapping_nontarget=sum(confusion.values()))
    return {"summary": summary, "per_image": evaluation["per_image"], "matches": evaluation["matches"],
            "fp_nontarget_overlap": overlaps, "fp_overlap_counts": dict(confusion),
            "prompt_class_confusion": dict(prompt_class_confusion)}


def deduplicate_union(boxes: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    ordered = sorted(boxes, key=lambda box: (-float(box.get("score", 0)),
                         -max(0, box["x1"] - box["x0"]) * max(0, box["y1"] - box["y0"]), box["x0"]))
    kept = []
    for box in ordered:
        if all(iou_xyxy(box, old) < iou_threshold for old in kept): kept.append(box)
    return kept
