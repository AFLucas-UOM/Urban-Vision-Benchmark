#!/usr/bin/env python3
"""Export ten clear false-negative alternatives from stored MTSD test outputs.

No model is loaded or run. Candidates are class-aware false negatives at the
same confidence >=0.25 and IoU >=0.50 protocol as the class-level analysis.
They are ranked to make human selection easy: one missed object, no unmatched
predictions, few total GT boxes, then a large missed-object area.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from make_mtsd_yolo26m_class_analysis import (ANNOTATIONS, CONFIDENCE, IMAGE_DIR,
                                               MATCH_IOU, OUT, PREDICTIONS, match,
                                               render_case)


OPTIONS = OUT / "easy_false_negative_options"


def make_contact_sheet(items, output: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont
    columns, cell_w, cell_h, title_h, pad = 5, 620, 490, 44, 16
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_w + (columns + 1) * pad,
                                rows * (cell_h + title_h) + (rows + 1) * pad), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("arialbd.ttf", 20)
    for number, item in enumerate(items, 1):
        image = Image.open(item["path"]).convert("RGB")
        image.thumbnail((cell_w, cell_h))
        column, row = (number - 1) % columns, (number - 1) // columns
        x = pad + column * (cell_w + pad)
        y = pad + row * (cell_h + title_h)
        draw.text((x, y), f"Option {number} — {item['class_name']}", fill="#202124", font=font)
        iy = y + title_h
        canvas.paste(image, (x + (cell_w - image.width) // 2, iy + (cell_h - image.height) // 2))
        draw.rectangle((x, iy, x + cell_w, iy + cell_h), outline="#b8b8b8", width=2)
    canvas.save(output, dpi=(300, 300))


def main() -> None:
    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    predictions = [x for x in json.loads(PREDICTIONS.read_text(encoding="utf-8")) if x["score"] >= CONFIDENCE]
    names = {x["id"]: x["name"] for x in annotations["categories"]}
    images = {x["id"]: x for x in annotations["images"]}
    gt, pred = defaultdict(list), defaultdict(list)
    for item in annotations["annotations"]:
        gt[item["image_id"]].append(item)
    for item in predictions:
        pred[item["image_id"]].append(item)
    candidates = []
    for image_id, image in images.items():
        matches, fps, missed = match(gt[image_id], pred[image_id])
        if not missed:
            continue
        # Prefer named traffic-sign categories over the deliberately broad
        # Back-/Other-Unknown labels, which tend to make poor dissertation
        # exemplars even when the box itself is large.
        named_misses = [x for x in missed if x["category_id"] not in {10, 11}]
        if not named_misses:
            continue
        # Use the largest named missed object in each image as the subject.
        focus = max(named_misses, key=lambda x: x["bbox"][2] * x["bbox"][3])
        case = {"gt": gt[image_id], "matches": matches, "fps": fps, "missed": missed,
                "file_name": image["file_name"]}
        candidates.append((image_id, case, focus))
    # Strictly simple cases first; progressively relax only if fewer than ten exist.
    simple = [x for x in candidates if len(x[1]["missed"]) == 1 and not x[1]["fps"] and len(x[1]["gt"]) <= 2]
    remaining = [x for x in candidates if x not in simple]
    rank = lambda x: (len(x[1]["missed"]), len(x[1]["fps"]), len(x[1]["gt"]), -x[2]["bbox"][2] * x[2]["bbox"][3])
    chosen = sorted(simple, key=rank)[:10]
    if len(chosen) < 10:
        chosen.extend(sorted(remaining, key=rank)[:10 - len(chosen)])
    OPTIONS.mkdir(parents=True, exist_ok=True)
    rows, contact_items = [], []
    for number, (image_id, case, focus) in enumerate(chosen, 1):
        class_name = names[focus["category_id"]]
        path = OPTIONS / f"option_{number:02d}_{image_id}_{class_name.replace(' ', '_').replace('/', '-')}.png"
        # A context-preserving focus crop makes the missed target legible.
        render_case(case, names, path, focus_box=focus["bbox"])
        area = focus["bbox"][2] * focus["bbox"][3]
        rows.append({"Option": number, "Image ID": image_id, "Image": case["file_name"], "Missed class": class_name,
                     "Missed bbox area (px²)": round(area, 1), "Total GT boxes": len(case["gt"]),
                     "Matched predictions": len(case["matches"]), "Unmatched predictions": len(case["fps"]),
                     "Annotated image": path.name})
        contact_items.append({"path": path, "class_name": class_name})
    with (OPTIONS / "option_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    make_contact_sheet(contact_items, OPTIONS / "easy_false_negative_contact_sheet.png")
    (OPTIONS / "README.txt").write_text(
        f"Ten held-out-test false-negative candidates. Confidence >= {CONFIDENCE:.2f}; class-aware IoU >= {MATCH_IOU:.2f}.\n"
        "Ranked for visual simplicity before missed-object size. Amber = missed GT; green = matched prediction; red = unmatched prediction.\n",
        encoding="utf-8")
    print(OPTIONS / "easy_false_negative_contact_sheet.png")
    print(OPTIONS / "option_index.csv")


if __name__ == "__main__":
    main()
