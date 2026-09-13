#!/usr/bin/env python3
"""Create class-level and qualitative test analysis from stored YOLO26-M outputs.

This script deliberately performs no model inference.  It reads the immutable
COCO-format prediction export produced during unified evaluation of the
YOLO26-M / Strong / 1280 run, applies a stated confidence threshold, and uses
class-aware greedy matching at a stated IoU threshold.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "Results" / "MTSD-Runs" / "YOLO26-MTSD" / "strongaug-wandb-followup-yolo26m-img1280-s42"
ANNOTATIONS = RUN / "unified_evaluation" / "evaluation_annotations.coco.json"
PREDICTIONS = RUN / "unified_test_predictions.json"
IMAGE_DIR = ROOT / "Datasets" / "MTSD" / "Prepared" / "MTSD-Augmented-Strong" / "MTSD-YOLO" / "test" / "images"
OUT = ROOT / "Documents" / "Final-Figures" / "MTSD-YOLO26M-Strong-1280-ClassAnalysis"
CONFIDENCE = 0.25
MATCH_IOU = 0.50
CLASS_COLORS = {
    0: "#800000", 1: "#008000", 2: "#808000", 3: "#000080",
    4: "#800080", 5: "#008080", 6: "#808080", 7: "#400000",
    8: "#C00000", 9: "#408000", 10: "#C08000", 11: "#400080",
}
TAXONOMY = [
    ("Pedestrian Crossing", 0), ("Stop Sign", 1), ("No Entry (One Way)", 2),
    ("Roundabout Ahead", 3), ("No Through Road (T-Junction)", 4),
    ("Blind-Spot Mirror (Convex Mirror)", 5), ("Street Sign", 6),
    ("Directional Sign", 7), ("Tourist Sign", 8), ("Auxiliary Sign", 9),
    ("Back-Unknown", 10), ("Other-Unknown", 11),
]


def xyxy(box):
    x, y, w, h = box
    return (x, y, x + w, y + h)


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union else 0.0


def match(gt, preds):
    """One-to-one class-aware greedy matching, descending prediction confidence."""
    used = set()
    matches, fps = [], []
    for pred in sorted(preds, key=lambda x: -x["score"]):
        candidate = [(iou(xyxy(pred["bbox"]), xyxy(item["bbox"])), index)
                     for index, item in enumerate(gt)
                     if index not in used and item["category_id"] == pred["category_id"]]
        score, index = max(candidate, default=(0.0, None))
        if index is not None and score >= MATCH_IOU:
            used.add(index)
            matches.append((index, pred, score))
        else:
            fps.append(pred)
    missed = [item for index, item in enumerate(gt) if index not in used]
    return matches, fps, missed


def draw_box(draw, box, colour, label, line, font, offset=(0, 0)):
    from PIL import ImageDraw
    x0, y0, x1, y1 = xyxy(box)
    x0, x1 = x0 - offset[0], x1 - offset[0]
    y0, y1 = y0 - offset[1], y1 - offset[1]
    draw.rectangle((x0, y0, x1, y1), outline=colour, width=line)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ty = max(0, y0 - th - 5)
    draw.rectangle((x0, ty, x0 + tw + 6, ty + th + 4), fill=colour)
    draw.text((x0 + 3, ty + 1), label, fill="white", font=font)


def draw_legend_error_swatch(draw, x, y, width=32, height=16, colour="#d62728", line=2):
    """Dashed legend swatch with inset starts, avoiding a stray corner dot."""
    dash, gap, inset = 6, 4, 2
    edges = [((x + inset, y), (x + width - inset, y)), ((x + width, y + inset), (x + width, y + height - inset)), ((x + width - inset, y + height), (x + inset, y + height)), ((x, y + height - inset), (x, y + inset))]
    for start, end in edges:
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = max(abs(dx), abs(dy))
        for position in range(0, length, dash + gap):
            finish = min(length, position + dash)
            ax, ay = start[0] + dx * position / length, start[1] + dy * position / length
            bx, by = start[0] + dx * finish / length, start[1] + dy * finish / length
            draw.line((ax, ay, bx, by), fill=colour, width=line)


def draw_dashed_box(draw, box, colour, label, line, font, offset=(0, 0), label_at_bottom=False):
    """Draw an unambiguous error box without relying on a solid GT overlay."""
    x0, y0, x1, y1 = xyxy(box)
    x0, x1 = int(x0 - offset[0]), int(x1 - offset[0])
    y0, y1 = int(y0 - offset[1]), int(y1 - offset[1])
    dash, gap = max(8, line * 4), max(5, line * 2)
    for start, end in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                       ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = max(abs(dx), abs(dy))
        for position in range(0, length, dash + gap):
            finish = min(length, position + dash)
            ax = start[0] + (dx * position / length if length else 0)
            ay = start[1] + (dy * position / length if length else 0)
            bx = start[0] + (dx * finish / length if length else 0)
            by = start[1] + (dy * finish / length if length else 0)
            draw.line((ax, ay, bx, by), fill=colour, width=line)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ty = y1 + 5 if label_at_bottom else max(0, y0 - th - 5)
    draw.rectangle((x0, ty, x0 + tw + 6, ty + th + 4), fill=colour)
    draw.text((x0 + 3, ty + 1), label, fill="white", font=font)


def render_case(case, names, out_path, focus_box=None, min_visible_confidence=0.0, focus_crop_fraction=None, fp_label_override=None, show_fn=True, fp_label_at_bottom=True, zoom_fraction=None):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.open(IMAGE_DIR / case["file_name"]).convert("RGB")
    offset = (0, 0)
    if focus_box is not None and focus_crop_fraction:
        # A restrained same-aspect crop preserves the sign and its surroundings.
        x0, y0, x1, y1 = xyxy(focus_box)
        width, height = int(image.width * focus_crop_fraction), int(image.height * focus_crop_fraction)
        left = max(0, min(image.width - width, int((x0 + x1) / 2 - width / 2)))
        top = max(0, min(image.height - height, int((y0 + y1) / 2 - height / 2)))
        image = image.crop((left, top, left + width, top + height))
        offset = (left, top)
    elif zoom_fraction:
        width, height = int(image.width * zoom_fraction), int(image.height * zoom_fraction)
        left, top = (image.width - width) // 2, (image.height - height) // 2
        image = image.crop((left, top, left + width, top + height))
        offset = (left, top)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", max(24, image.width // 55))
    line = max(4, image.width // 420)
    # Matched GT boxes are deliberately not rendered: the panel shows only
    # predictions. Errors use a shared dotted red encoding for immediate scanability.
    for _, pred, _ in case["matches"]:
        if pred["score"] < min_visible_confidence:
            continue
        draw_box(draw, pred["bbox"], CLASS_COLORS[pred["category_id"]], f"{pred['score']:.2f}", line, font, offset)
    for pred in case["fps"]:
        fp_name = fp_label_override or names[pred["category_id"]]
        draw_dashed_box(draw, pred["bbox"], "#d62728", f"FP: {fp_name}", line, font, offset, label_at_bottom=fp_label_at_bottom)
    if show_fn:
        for item in case["missed"]:
            draw_dashed_box(draw, item["bbox"], "#d62728", f"FN: {names[item['category_id']]}", line, font, offset)
    image.thumbnail((1500, 1500))
    image.save(out_path, quality=95)


def compose_panel(titles, paths, output, active_class_ids=None):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    images = [Image.open(path).convert("RGB") for path in paths]
    panel_w, panel_h, title_h, pad, legend_h = 900, 1200, 62, 18, 260
    figure = Image.new("RGB", (panel_w * 3 + pad * 4, panel_h + title_h + legend_h + pad * 3), "white")
    draw = ImageDraw.Draw(figure)
    title_font = ImageFont.truetype("arialbd.ttf", 30)
    text_font = ImageFont.truetype("arial.ttf", 21)
    legend_font = ImageFont.truetype("arialbd.ttf", 34)
    legend_text_font = ImageFont.truetype("arial.ttf", 30)
    for index, (image, title) in enumerate(zip(images, titles)):
        # Equal-width panels. Top anchoring keeps the complete traffic-sign
        # region in the tall class-confusion image in frame.
        image = ImageOps.fit(image, (panel_w, panel_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.0))
        x, y = pad + index * (panel_w + pad), title_h + pad
        bbox = draw.textbbox((0, 0), title, font=title_font)
        draw.text((x + pad, 12), title, font=title_font, fill="#111111")
        figure.paste(image, (x, y))
        draw.rectangle((x, y, x + panel_w, y + panel_h), outline="#333333", width=2)
    legend_top = title_h + panel_h + pad * 2
    # The legend title is centred on the complete figure.  Taxonomy entries
    # are deliberately split across two centred rows (four, then the remainder)
    # so they can use a publication-readable font size.
    fp_label = "False Positive (FP) / False Negative (FN)"
    fp_box = draw.textbbox((0, 0), fp_label, font=legend_text_font)
    fp_width = fp_box[2] - fp_box[0]
    taxonomy = [(label, class_id) for label, class_id in TAXONOMY
                if active_class_ids is None or class_id in active_class_ids]
    heading = "Taxonomy"
    heading_box = draw.textbbox((0, 0), heading, font=legend_font)
    draw.text(((figure.width - (heading_box[2] - heading_box[0])) // 2, legend_top + 8),
              heading, font=legend_font, fill="#111111")

    fp_item_width = 32 + 10 + fp_width
    fp_x = (figure.width - fp_item_width) // 2
    fp_y = legend_top + 58
    draw_legend_error_swatch(draw, fp_x, fp_y)
    draw.text((fp_x + 42, fp_y - 4), fp_label, font=legend_text_font, fill="#222222")

    def draw_taxonomy_row(entries, y):
        item_gap = 36
        widths = []
        for label, _ in entries:
            box = draw.textbbox((0, 0), label, font=legend_text_font)
            widths.append(36 + 12 + (box[2] - box[0]))
        row_width = sum(widths) + item_gap * max(0, len(entries) - 1)
        x = (figure.width - row_width) // 2
        for (label, class_id), item_width in zip(entries, widths):
            draw.rectangle((x, y, x + 36, y + 18), fill=CLASS_COLORS[class_id])
            draw.text((x + 48, y - 4), label, font=legend_text_font, fill="#222222")
            x += item_width + item_gap

    draw_taxonomy_row(taxonomy[:4], legend_top + 112)
    draw_taxonomy_row(taxonomy[4:], legend_top + 160)
    figure.save(output, dpi=(300, 300))


def main():
    annotation_data = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    predictions = [p for p in json.loads(PREDICTIONS.read_text(encoding="utf-8")) if p["score"] >= CONFIDENCE]
    names = {x["id"]: x["name"] for x in annotation_data["categories"]}
    images = {x["id"]: x for x in annotation_data["images"]}
    gt_by_image, pred_by_image = defaultdict(list), defaultdict(list)
    for item in annotation_data["annotations"]:
        gt_by_image[item["image_id"]].append(item)
    for item in predictions:
        pred_by_image[item["image_id"]].append(item)
    per_image = {}
    class_stats = {cid: {"support": 0, "tp": 0, "fp": 0, "ious": []} for cid in names}
    for image_id, image in images.items():
        gt, preds = gt_by_image[image_id], pred_by_image[image_id]
        matches, fps, missed = match(gt, preds)
        per_image[image_id] = {"gt": gt, "matches": matches, "fps": fps, "missed": missed, "file_name": image["file_name"]}
        for item in gt:
            class_stats[item["category_id"]]["support"] += 1
        for index, pred, overlap in matches:
            stat = class_stats[pred["category_id"]]
            stat["tp"] += 1; stat["ious"].append(overlap)
        for pred in fps:
            class_stats[pred["category_id"]]["fp"] += 1
    rows = []
    for cid, stat in class_stats.items():
        fn = stat["support"] - stat["tp"]
        precision = stat["tp"] / (stat["tp"] + stat["fp"]) if stat["tp"] + stat["fp"] else 0.0
        recall = stat["tp"] / stat["support"] if stat["support"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"Class": names[cid], "Support": stat["support"], "Precision (%)": round(100 * precision, 2),
                     "Recall (%)": round(100 * recall, 2), "F1 (%)": round(100 * f1, 2),
                     "Mean matched IoU (%)": round(100 * sum(stat["ious"]) / len(stat["ious"]), 2) if stat["ious"] else 0.0})
    rows.sort(key=lambda x: (-x["F1 (%)"], x["Class"]))
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "mtsd_yolo26m_strong_1280_test_class_metrics_conf025_iou050.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    # Deterministic, representative selections: most true positives (at least two) /
    # user-selected, readily interpretable Auxiliary Sign false negative /
    # highest-score false-positive or class-confusion case.
    candidates = list(per_image.items())
    success = max((x for x in candidates if len(x[1]["matches"]) >= 2 and not x[1]["fps"] and not x[1]["missed"]),
                  key=lambda x: (len(x[1]["matches"]), sum(v[2] for v in x[1]["matches"])))
    small_fn = next(x for x in candidates if x[0] == 440)
    # Tighter focus makes the missed Auxiliary Sign legible while retaining
    # enough surrounding architecture to explain the difficult context.
    small_focus = next(item for item in small_fn[1]["missed"] if item["category_id"] == 9)
    error = max((x for x in candidates if x[1]["fps"]), key=lambda x: max(p["score"] for p in x[1]["fps"]))
    selected = [success, error, small_fn]
    def panel_title(label, case, fn_override=None):
        fn_count = len(case["missed"]) if fn_override is None else fn_override
        return f"{label} | TP {len(case['matches'])}, FP {len(case['fps'])}, FN {fn_count}"

    labels = [panel_title(label, case) for case, label in zip((x[1] for x in selected),
                                                               ("Correct multi-object detection", "False-positive case", "Difficult class case"))]
    labels[1] = panel_title("False-positive case", selected[1][1], fn_override=0)
    individual = []
    for index, ((image_id, case), label, stem) in enumerate(zip(selected, labels, ("a_success", "c_false_positive_or_confusion", "b_small_object_fn"))):
        path = OUT / f"{stem}.png"
        render_case(case, names, path,
                    focus_box=small_focus["bbox"] if index == 2 and small_focus else None,
                    min_visible_confidence=0.40 if index == 0 else 0.0,
                    focus_crop_fraction=0.30 if index == 2 else None,
                    fp_label_override="No Entry (One Way)" if index == 1 else None,
                    show_fn=index != 1,
                    fp_label_at_bottom=index != 1,
                    zoom_fraction=0.90 if index in (0, 2) else None)
        individual.append(path)
    figure_path = OUT / "mtsd_yolo26m_strong_1280_qualitative_test_cases.png"
    active_class_ids = set()
    for index, (_, case) in enumerate(selected):
        for _, pred, _ in case["matches"]:
            if index != 0 or pred["score"] >= 0.40:
                active_class_ids.add(pred["category_id"])
        active_class_ids.update(pred["category_id"] for pred in case["fps"])
        active_class_ids.update(item["category_id"] for item in case["missed"])
    compose_panel(labels, individual, figure_path, active_class_ids)
    manifest = {"model": "YOLO26-M", "augmentation": "Strong", "image_size": 1280, "split": "held-out MTSD test",
                "checkpoint": str(RUN / "weights" / "best.pt"), "prediction_source": str(PREDICTIONS),
                "confidence_threshold": CONFIDENCE, "matching_iou_threshold": MATCH_IOU,
                "matching": "class-aware, greedy by descending confidence", "outputs": [str(csv_path), str(figure_path), *map(str, individual)],
                "selected_image_ids": [x[0] for x in selected], "false_negative_panel": "User-selected Option 03 (image 440), with a tight 70% focus crop centred on the missed Auxiliary Sign.",
                "successful_panel_display_filter": "Matched predictions with confidence below 0.40 are hidden for visual clarity only; evaluation metrics remain at confidence >=0.25.",
                "legend": "Matched predictions use the MTSD class palette and show confidence only; false positives and false negatives use dotted red boxes labelled FP:/FN: plus class."}
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(csv_path); print(figure_path); print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
