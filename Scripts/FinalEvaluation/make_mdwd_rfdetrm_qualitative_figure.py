"""Create a representative qualitative RF-DETR-M figure from held-out MDWD test data."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "Datasets/MDWD/MDWD-RFDETR/test"
ANNOTATIONS = TEST_DIR / "_annotations.coco.json"
PREDICTIONS = ROOT / "Results/MDWD-Results/RF-DETR/RF-DETR-M-Test-Evaluation/test_predictions_coco.json"
OUTPUT_DIR = ROOT / "figures"
CONFIDENCE = 0.25
MATCH_IOU = 0.50
CLASS_COLORS = {
    "Mixed Waste": "#151515",          # black
    "Recyclable Material": "#1769AA",  # blue
    "Organic Waste": "#208A43",        # green
    "Orange CMD": "#E67E22",           # orange
    "Other Waste": "#7B3FA1",          # purple
}
ERROR_COLOR = "#C62828"                  # red
# Manually reviewed, varied held-out examples.  These are representative
# rather than extreme: a multi-class success; a clear miss; an isolated FP;
# and a moderately crowded recyclable/organic case.
SELECTED_IDS = {"success": 143, "fn": 48, "difficult": 280}


def iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x0, y0, x1, y1 = max(ax, bx), max(ay, by), min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def match_image(predictions: list[dict], ground_truth: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Class-aware, confidence-sorted greedy matching at the audit definition."""
    available = set(range(len(ground_truth)))
    tp, fp = [], []
    for pred in sorted(predictions, key=lambda row: row["score"], reverse=True):
        best = max((index for index in available if ground_truth[index]["category_id"] == pred["category_id"]),
                   key=lambda index: iou(pred["bbox"], ground_truth[index]["bbox"]), default=None)
        if best is not None and iou(pred["bbox"], ground_truth[best]["bbox"]) >= MATCH_IOU:
            pred = {**pred, "match_iou": iou(pred["bbox"], ground_truth[best]["bbox"])}
            tp.append(pred); available.remove(best)
        else:
            fp.append(pred)
    return tp, fp, [ground_truth[index] for index in sorted(available)]


def candidate_score(kind: str, record: dict) -> tuple:
    tp, fp, fn, gt = record["tp"], record["fp"], record["fn"], record["gt"]
    n_tp, n_fp, n_fn, n_gt = len(tp), len(fp), len(fn), len(gt)
    difficult = any(row["category_name"] in {"Orange CMD", "Other Waste"} for row in gt)
    tp_classes = {row["category_name"] for row in tp}
    error_classes = {row["category_name"] for row in fp + fn}
    image_area = record["image"]["width"] * record["image"]["height"]
    mean_box_fraction = sum(row["bbox"][2] * row["bbox"][3] for row in gt) / max(1, n_gt * image_area)
    # Higher score means objects are smaller / more demanding, but cap it so
    # the figure remains visually interpretable rather than selecting extremes.
    challenge = min(0.075, max(0.0, 0.075 - mean_box_fraction))
    # Prefer compact panels: errors should be evident without hiding many boxes.
    compact = n_gt <= 5 and n_tp + n_fp <= 6
    if kind == "success":
        # Deliberately require a varied correct-detection case, rather than a
        # monoclass scene, for an informative dissertation panel.
        eligible = n_tp >= 3 and n_fp == n_fn == 0 and len(tp_classes) >= 2
        return (eligible, compact, len(tp_classes), challenge, min(n_tp, 4), n_gt)
    if kind == "fn":
        eligible = n_fn >= 1 and n_tp >= 1 and n_fp <= 1
        return (eligible, compact, -abs(n_fn - 1), challenge, n_tp, n_gt)
    if kind == "fp":
        eligible = n_fp >= 1 and n_tp >= 1 and n_fn <= 1 and "Orange CMD" in error_classes
        return (eligible, compact, -abs(n_fp - 1), challenge, n_tp, n_gt)
    # A controlled difficult case: one/two errors involving the weaker classes, not a collapse.
    eligible = difficult and "Other Waste" in error_classes and 1 <= n_fp + n_fn <= 2 and n_tp >= 1
    return (eligible, compact, challenge, n_tp, -(n_fp + n_fn), n_gt)


def choose(records: list[dict]) -> list[tuple[str, dict]]:
    by_id = {record["id"]: record for record in records}
    if all(image_id in by_id for image_id in SELECTED_IDS.values()):
        return [(kind, by_id[image_id]) for kind, image_id in SELECTED_IDS.items()]
    selected, used = [], set()
    for kind in ("success", "fn", "fp", "difficult"):
        ranked = sorted(records, key=lambda row: candidate_score(kind, row), reverse=True)
        choice = next((row for row in ranked if row["id"] not in used and candidate_score(kind, row)[0]), None)
        if choice is None:
            raise RuntimeError(f"Could not find a suitable {kind} panel")
        selected.append((kind, choice)); used.add(choice["id"])
    return selected


def draw_box(axis, bbox, color, linestyle="-", linewidth=2.0, label=None, y_offset=0):
    x, y, width, height = bbox
    axis.add_patch(patches.Rectangle((x, y), width, height, fill=False, edgecolor=color,
                                     linestyle=linestyle, linewidth=linewidth))
    if label:
        # Anchor the score label directly on the upper box edge.
        axis.text(x, max(2, y - y_offset), label, fontsize=7.2, color="white", va="bottom", ha="left",
                  bbox={"facecolor": color, "edgecolor": "none", "boxstyle": "square,pad=0.15", "alpha": 0.92})


def main() -> None:
    truth = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    predictions = [row for row in json.loads(PREDICTIONS.read_text(encoding="utf-8")) if row["score"] >= CONFIDENCE]
    categories = {row["id"]: row["name"] for row in truth["categories"]}
    image_by_id = {row["id"]: row for row in truth["images"]}
    gt_by_image, pred_by_image = defaultdict(list), defaultdict(list)
    for row in truth["annotations"]:
        gt_by_image[row["image_id"]].append({**row, "category_name": categories[row["category_id"]]})
    for row in predictions:
        pred_by_image[row["image_id"]].append({**row, "category_name": categories[row["category_id"]]})

    records = []
    for image_id, image in image_by_id.items():
        tp, fp, fn = match_image(pred_by_image[image_id], gt_by_image[image_id])
        records.append({"id": image_id, "image": image, "gt": gt_by_image[image_id], "tp": tp, "fp": fp, "fn": fn})
    selected = choose(records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.2), constrained_layout=True)
    reasons = {
        "success": "multiple correct detections; no audit errors",
        "fn": "contains a visually interpretable missed ground-truth object",
        "fp": "contains an isolated, visually plausible spurious detection alongside a correct detection",
        "difficult": "complex illegal-dumping scene with mixed waste and non-waste materials",
    }
    summary = ["RF-DETR-M qualitative test examples", f"Fixed audit point: confidence >= {CONFIDENCE:.2f}; class-aware IoU >= {MATCH_IOU:.2f}", "",
               "Legend: solid class colour = correct prediction; dotted red = FP or FN.", ""]
    for axis, (panel, record) in zip(axes.flat, selected):
        image = ImageOps.exif_transpose(Image.open(TEST_DIR / record["image"]["file_name"])).convert("RGB")
        axis.imshow(image)
        displayed_fp = list(record["fp"])
        omitted_overlap_fp = None
        # D9 has two FPs, one directly over its FN marker.  Keep the isolated FP
        # and omit the overlapping marker so the intended FN remains legible.
        if panel == "difficult" and record["fn"] and len(displayed_fp) > 1:
            omitted_overlap_fp = max(
                displayed_fp,
                key=lambda pred: max(iou(pred["bbox"], miss["bbox"]) for miss in record["fn"]),
            )
            displayed_fp.remove(omitted_overlap_fp)
        # Correct detections use only their solid prediction box.  Ground truth
        # is retained solely for unmatched objects (the red FN marker).
        for row in record["fn"]:
            draw_box(axis, row["bbox"], ERROR_COLOR, linestyle=":", linewidth=3.0, label="FN")
        for row in record["tp"]:
            draw_box(axis, row["bbox"], CLASS_COLORS[row["category_name"]], linewidth=2.35,
                     label=f"{row['score']:.2f}")
        for row in displayed_fp:
            draw_box(axis, row["bbox"], ERROR_COLOR, linestyle=":", linewidth=3.0, label="FP", y_offset=11)
        title = {"success": "Correct multi-object detection", "fn": "False-negative case",
                 "fp": "False-positive case", "difficult": "Difficult class case"}[panel]
        axis.set_title(f"{title}  |  TP {len(record['tp'])}, FP {len(displayed_fp)}, FN {len(record['fn'])}",
                       fontsize=12, loc="left", pad=8, fontweight="semibold")
        axis.set_axis_off()
        error_classes = sorted({row["category_name"] for row in displayed_fp + record["fn"]}) or ["None"]
        summary.extend([
            f"{panel.capitalize()}: {record['image']['file_name']} (COCO image ID {record['id']})",
            f"  Rationale: {reasons[panel]}.",
            f"  Displayed TP={len(record['tp'])}; FP={len(displayed_fp)}; FN={len(record['fn'])}; main error class(es): {', '.join(error_classes)}.",
        ])
        if omitted_overlap_fp is not None:
            summary.append("  One FP overlapping the FN marker is omitted from the visual for legibility.")
    taxonomy_legend = fig.legend(
        handles=[patches.Patch(facecolor="none", edgecolor=ERROR_COLOR, linestyle=":", label="False Positive (FP) / False Negative (FN)"),
                 *[patches.Patch(facecolor=color, edgecolor="none", label=name) for name, color in CLASS_COLORS.items()]],
        title="Taxonomy", loc="lower center", bbox_to_anchor=(0.5, -0.060), ncol=6, frameon=False,
        fontsize=12.5, title_fontsize=13.5,
    )
    fig.add_artist(taxonomy_legend)
    fig.savefig(OUTPUT_DIR / "MDWD-RFDETRM-Qualitative.png", dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / "MDWD-RFDETRM-Qualitative.svg", bbox_inches="tight", facecolor="white")
    (OUTPUT_DIR / "MDWD-RFDETRM-Qualitative-summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
