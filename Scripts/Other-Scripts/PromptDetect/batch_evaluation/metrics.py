"""IoU matching and detection metrics for PromptDetect batch evaluation.

Matching is the standard greedy scheme: predictions sorted by score, each
matched to the highest-IoU unmatched ground-truth box (IoU >= threshold).
Extra predictions on an already-matched GT box are counted as *duplicates*
(and as false positives). AP uses all-point interpolation.

Note on confidence: Cosmos and LocateAnything emit no per-box scores (the
backend assigns a constant 1.0), so their AP collapses to the single
precision/recall operating point - reported, but flagged in the summary.
"""

from __future__ import annotations

from collections import defaultdict


def iou_xyxy(a: dict, b: dict) -> float:
    ix0, iy0 = max(a["x0"], b["x0"]), max(a["y0"], b["y0"])
    ix1, iy1 = min(a["x1"], b["x1"]), min(a["y1"], b["y1"])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a["x1"] - a["x0"]) * max(0.0, a["y1"] - a["y0"])
    area_b = max(0.0, b["x1"] - b["x0"]) * max(0.0, b["y1"] - b["y0"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_image(predictions: list[dict], gt_boxes: list[dict], iou_threshold: float) -> dict:
    """Greedy per-image matching.

    Returns {"tp", "fp", "fn", "duplicates", "matches": [(pred, gt, iou)],
             "pred_flags": [bool is_tp per prediction in score order]}.
    """
    preds = sorted(predictions, key=lambda p: -float(p.get("score", 0.0)))
    matched_gt: set[int] = set()
    matches, pred_flags = [], []
    duplicates = 0
    for pred in preds:
        best_iou, best_index = 0.0, None
        for index, gt in enumerate(gt_boxes):
            iou = iou_xyxy(pred, gt)
            if iou > best_iou:
                best_iou, best_index = iou, index
        if best_index is not None and best_iou >= iou_threshold:
            if best_index in matched_gt:
                duplicates += 1
                pred_flags.append(False)
            else:
                matched_gt.add(best_index)
                matches.append((pred, gt_boxes[best_index], best_iou))
                pred_flags.append(True)
        else:
            pred_flags.append(False)
    tp = len(matches)
    fp = len(preds) - tp
    fn = len(gt_boxes) - tp
    return {"tp": tp, "fp": fp, "fn": fn, "duplicates": duplicates,
            "matches": matches, "preds_sorted": preds, "pred_flags": pred_flags}


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # "Accuracy" in the detection sense: TP / (TP + FP + FN) - meaningful only
    # as a combined error rate, reported for completeness.
    accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "accuracy": round(accuracy, 4)}


def average_precision(scored_flags: list[tuple[float, bool]], n_gt: int) -> float:
    """All-point-interpolation AP from (score, is_tp) pairs."""
    if n_gt == 0 or not scored_flags:
        return 0.0
    ordered = sorted(scored_flags, key=lambda item: -item[0])
    tp_cum = fp_cum = 0
    points = []
    for _, is_tp in ordered:
        tp_cum += int(is_tp)
        fp_cum += int(not is_tp)
        points.append((tp_cum / n_gt, tp_cum / (tp_cum + fp_cum)))
    ap, best_precision, previous_recall = 0.0, 0.0, None
    for recall, precision in reversed(points):
        best_precision = max(best_precision, precision)
        if previous_recall is None:
            previous_recall = recall
            segment = recall
        else:
            segment = previous_recall - recall
            previous_recall = recall
        ap += best_precision * segment
    return round(ap, 4)


def evaluate_prompt(
    predictions_by_image: dict[str, list[dict]],
    gt_by_image: dict[str, list[dict]],
    iou_threshold: float,
    map_iou_range: list[float],
) -> dict:
    """Aggregate metrics for one (model, prompt) over all images."""
    totals = defaultdict(int)
    matched_ious: list[float] = []
    all_matches: list[tuple[dict, dict, float]] = []
    per_image_rows = []
    flags_at_iou: dict[float, list[tuple[float, bool]]] = {t: [] for t in map_iou_range}
    n_gt_total = sum(len(v) for v in gt_by_image.values())

    for image_id, gt_boxes in gt_by_image.items():
        preds = predictions_by_image.get(image_id, [])
        result = match_image(preds, gt_boxes, iou_threshold)
        totals["tp"] += result["tp"]
        totals["fp"] += result["fp"]
        totals["fn"] += result["fn"]
        totals["duplicates"] += result["duplicates"]
        matched_ious.extend(iou for _, _, iou in result["matches"])
        all_matches.extend(result["matches"])
        per_image_rows.append({
            "image_id": image_id, "n_gt": len(gt_boxes), "n_pred": len(preds),
            **{k: result[k] for k in ("tp", "fp", "fn", "duplicates")},
            **prf(result["tp"], result["fp"], result["fn"]),
        })
        for threshold in map_iou_range:
            r = result if threshold == iou_threshold else match_image(preds, gt_boxes, threshold)
            flags_at_iou[threshold].extend(
                (float(p.get("score", 0.0)), flag) for p, flag in zip(r["preds_sorted"], r["pred_flags"])
            )

    ap_by_iou = {t: average_precision(flags_at_iou[t], n_gt_total) for t in map_iou_range}
    summary = {
        **{k: int(totals[k]) for k in ("tp", "fp", "fn", "duplicates")},
        **prf(totals["tp"], totals["fp"], totals["fn"]),
        "mean_matched_iou": round(sum(matched_ious) / len(matched_ious), 4) if matched_ious else 0.0,
        "ap50": ap_by_iou.get(0.5, 0.0),
        "map50_95": round(sum(ap_by_iou.values()) / len(ap_by_iou), 4) if ap_by_iou else 0.0,
        "n_gt": n_gt_total,
        "n_images": len(gt_by_image),
    }
    return {"summary": summary, "per_image": per_image_rows, "matches": all_matches}


def confusion_matrix(matches: list[tuple[dict, dict, float]], class_names: list[str]) -> dict:
    """Prompt-label vs GT-class counts over matched boxes (class-aware runs only)."""
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for pred, gt, _ in matches:
        matrix[str(pred.get("prompt", "?"))][str(gt.get("class_name", "?"))] += 1
    return {p: dict(row) for p, row in matrix.items()}
