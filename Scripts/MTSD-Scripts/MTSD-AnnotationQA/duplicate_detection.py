"""Duplicate / high-overlap annotation detection within single images."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import config


def iou_xywh(a: list[float], b: list[float]) -> float:
    """IoU of two COCO [x, y, w, h] boxes."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _label_attributes(annotation: dict[str, Any]) -> dict[str, Any]:
    attrs = annotation.get("attributes") or {}
    return {k: v for k, v in attrs.items() if k not in config.NON_LABEL_ATTRIBUTE_KEYS}


def find_duplicate_pairs(
    annotations: list[dict[str, Any]],
    duplicate_iou: float = config.DUPLICATE_IOU,
    overlap_iou: float = config.OVERLAP_IOU,
) -> list[dict[str, Any]]:
    """All annotation pairs (same image) with IoU >= overlap_iou.

    Each pair record classifies the finding:
        exact_duplicate        IoU >= duplicate_iou, same class, same attributes
        conflicting_duplicate  IoU >= duplicate_iou, class or attributes differ
        high_overlap           overlap_iou <= IoU < duplicate_iou
    """
    pairs = []
    boxed = [a for a in annotations if isinstance(a.get("bbox"), (list, tuple)) and len(a["bbox"]) == 4]
    for a, b in combinations(boxed, 2):
        try:
            iou = iou_xywh([float(v) for v in a["bbox"]], [float(v) for v in b["bbox"]])
        except (TypeError, ValueError):
            continue
        if iou < overlap_iou:
            continue
        attrs_a, attrs_b = _label_attributes(a), _label_attributes(b)
        same_class = a.get("category_id") == b.get("category_id")
        same_attrs = attrs_a == attrs_b
        if iou >= duplicate_iou:
            kind = "exact_duplicate" if (same_class and same_attrs) else "conflicting_duplicate"
        else:
            kind = "high_overlap"
        differing = sorted(
            k for k in set(attrs_a) | set(attrs_b) if attrs_a.get(k) != attrs_b.get(k)
        )
        pairs.append(
            {
                "kind": kind,
                "iou": round(iou, 4),
                "annotation_id_a": a.get("id"),
                "annotation_id_b": b.get("id"),
                "category_id_a": a.get("category_id"),
                "category_id_b": b.get("category_id"),
                "same_class": same_class,
                "same_attributes": same_attrs,
                "differing_attributes": ", ".join(differing),
                "bbox_a": list(a.get("bbox", [])),
                "bbox_b": list(b.get("bbox", [])),
            }
        )
    return pairs
