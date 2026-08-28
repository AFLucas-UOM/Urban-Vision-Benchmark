#!/usr/bin/env python3
"""Create compact annotation-overview figures for MDWD and MTSD.

One representative image is selected for every class. All ground-truth boxes
in the selected image are drawn, with the panel's target class emphasized.
"""
from pathlib import Path
import math
import random
import itertools
import json

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "Documents" / "Dataset-Annotation-Figures"

DATASETS = {
    "MDWD": {
        "root": ROOT / "Datasets" / "MDWD" / "MDWD-RAW",
        "splits": ["train", "valid", "test"],
        "scan_limit": 5000,
        "preferred": {0: "ARI3129_SDM_83_jpg", 2: "image00025_jpeg", 4: "AFL_NewBatch3_9_jpeg"},
        "display_order": [0, 4, 2, 1, 3],
        "colors": [(214, 65, 65), (239, 125, 34), (55, 158, 83), (142, 78, 190), (42, 111, 219)],
        "classes": ["Mixed Waste", "Orange CMD", "Organic Waste", "Other Waste", "Recyclable Material"],
    },
    "MTSD": {
        "root": ROOT / "Datasets" / "MTSD" / "Prepared" / "MTSD-Unaugmented" / "MTSD-YOLO",
        "splits": ["train", "valid", "test"],
        "scan_limit": 1000,
        "display_order": list(range(12)),
        "draw_all_boxes": False,
        "prefer_single": True,
        "preferred": {
            5: "grp4_e5a15d52-20251116_170059",
            6: "grp10_WhatsApp Image 2025-11-17 at 15.01.53",
            8: "grp10_IMG_20260110_172557_078",
        },
        "colors": [(128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
                    (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0),
                    (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128)],
        "classes": ["Pedestrian Crossing", "Stop Sign", "No Entry (One Way)", "Roundabout Ahead",
                    "No Through Road (T-Junction)", "Blind-Spot Mirror (Convex Mirror)", "Street Sign",
                    "Directional Sign", "Tourist Sign", "Auxiliary Sign", "Back-Unknown", "Other-Unknown"],
    },
}

def font(size, bold=False):
    names = ("segoeuib.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf") if bold else ("segoeui.ttf", "Arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def read_labels(label_path):
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        bits = line.split()
        if len(bits) >= 5:
            rows.append(tuple([int(bits[0])] + [float(v) for v in bits[1:5]]))
    return rows


def collect(dataset):
    root = dataset["root"]
    records = []
    if dataset is DATASETS["MDWD"]:
        for split in dataset["splits"]:
            ann_path = root / split / "_annotations.coco.json"
            if not ann_path.exists():
                continue
            data = json.loads(ann_path.read_text(encoding="utf-8"))
            images = {item["id"]: item for item in data["images"]}
            by_image = {}
            for ann in data["annotations"]:
                image = images.get(ann["image_id"])
                if not image or not 1 <= ann["category_id"] <= len(dataset["classes"]):
                    continue
                x, y, w, h = ann["bbox"]
                by_image.setdefault(image["id"], []).append(
                    (ann["category_id"] - 1,
                     (x + w / 2) / image["width"],
                     (y + h / 2) / image["height"],
                     w / image["width"], h / image["height"]))
            for image_id, labels in by_image.items():
                image = images[image_id]
                records.append((split, root / split / image["file_name"], labels))
        return records
    for split in dataset["splits"]:
        image_dir, label_dir = root / split / "images", root / split / "labels"
        if not image_dir.is_dir():
            continue
        # The repository contains many augmented copies; a bounded scan is
        # sufficient to find clean representative examples and keeps the
        # figure generator quick.
        for image_path in itertools.islice(sorted(image_dir.iterdir()), dataset.get("scan_limit", 1000)):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                labels = read_labels(label_path)
                if labels:
                    records.append((split, image_path, labels))
    return records


def choose(records, class_ids):
    chosen = []
    for cid in class_ids:
        candidates = [(max((w * h for c, _, _, w, h in labels if c == cid), default=0), split, path, labels)
                      for split, path, labels in records if any(c == cid for c, *_ in labels)]
        if not candidates:
            continue
        preferred = CURRENT_DATASET.get("preferred", {}).get(cid)
        if preferred:
            matches = [item for item in candidates if preferred in item[2].name]
            if matches:
                candidates = matches
        # For MTSD, prefer a frame containing exactly one annotated sign so
        # the example is easy to read and the target box is unambiguous.
        if CURRENT_DATASET.get("prefer_single"):
            single = [item for item in candidates if len(item[3]) == 1]
            if single:
                candidates = single
            else:
                # If no one-sign frame exists for a class, use the frame
                # with the fewest annotations before considering box size.
                min_count = min(len(item[3]) for item in candidates)
                candidates = [item for item in candidates if len(item[3]) == min_count]
        # Prefer a sizeable, clearly visible target annotation.
        _, split, path, labels = max(candidates, key=lambda x: x[0])
        chosen.append((cid, split, path, labels))
    return chosen


CURRENT_DATASET = {}


def source_label(split, path):
    name = path.name
    lower = name.lower()
    for marker in ("_jpeg", "_jpg", "_png", "_webp"):
        pos = lower.find(marker)
        if pos >= 0:
            return f"Source: {split} · {name[:pos]}.jpg"
    return f"Source: {split} · {path.stem}.jpg"


def rotate_labels(labels, orientation):
    """Move normalized YOLO boxes into the EXIF-transposed image frame."""
    if orientation == 3:
        return [(c, 1 - cx, 1 - cy, w, h) for c, cx, cy, w, h in labels]
    if orientation == 6:  # 90 degrees clockwise
        return [(c, 1 - cy, cx, h, w) for c, cx, cy, w, h in labels]
    if orientation == 8:  # 90 degrees counter-clockwise
        return [(c, cy, 1 - cx, h, w) for c, cx, cy, w, h in labels]
    return labels


def fit_to_frame(im, labels, frame_size):
    """Place every image in a uniform frame and map boxes through padding."""
    frame_w, frame_h = frame_size
    width, height = im.size
    scale = min(frame_w / width, frame_h / height)
    resized_w = max(1, round(width * scale))
    resized_h = max(1, round(height * scale))
    fitted = im.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    offset_x = (frame_w - resized_w) // 2
    offset_y = (frame_h - resized_h) // 2
    frame = Image.new("RGB", frame_size, "black")
    frame.paste(fitted, (offset_x, offset_y))
    mapped = []
    for c, cx, cy, bw, bh in labels:
        mapped.append((c,
                       (cx * resized_w + offset_x) / frame_w,
                       (cy * resized_h + offset_y) / frame_h,
                       bw * resized_w / frame_w,
                       bh * resized_h / frame_h))
    return frame, mapped


def render(name, dataset):
    global CURRENT_DATASET
    CURRENT_DATASET = dataset
    records = collect(dataset)
    display_order = dataset.get("display_order", list(range(len(dataset["classes"]))))
    chosen = choose(records, display_order)
    cols = 3 if name == "MDWD" else 4
    panel_w, panel_h = (330, 280) if name == "MDWD" else (360, 275)
    gap, title_h = 18, 18
    rows = math.ceil(len(chosen) / cols)
    legend_rows = math.ceil(len(dataset["classes"]) / cols)
    legend_h = 24 + legend_rows * 25
    canvas = Image.new("RGB", (cols * panel_w + (cols + 1) * gap,
                                title_h + rows * panel_h + (rows + 1) * gap + legend_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (cid, split, path, labels) in enumerate(chosen):
        x = gap + (i % cols) * panel_w
        y = title_h + gap + (i // cols) * panel_h
        with Image.open(path) as source:
            im = ImageOps.exif_transpose(source).convert("RGB")
            # The YOLO labels are already expressed in the displayed image
            # orientation; do not rotate them a second time after EXIF fixing.
            frame_h = panel_h - 50
            frame_w = round(frame_h * 1.1)
            im, display_labels = fit_to_frame(im, labels, (frame_w, frame_h))
            px, py = x + (panel_w - im.width) // 2, y + 28
            canvas.paste(im, (px, py))
        od = ImageDraw.Draw(canvas)
        for lc, cx, cy, bw, bh in display_labels:
            if not dataset.get("draw_all_boxes", True) and lc != cid:
                continue
            colour = dataset["colors"][lc]
            x0 = px + (cx - bw / 2) * im.width
            y0 = py + (cy - bh / 2) * im.height
            x1 = px + (cx + bw / 2) * im.width
            y1 = py + (cy + bh / 2) * im.height
            width = 4 if lc == cid else 2
            od.rectangle((x0, y0, x1, y1), outline=colour, width=width)
        source_text = source_label(split, path)
        source_font_size = 12
        source_font = font(source_font_size)
        while od.textbbox((0, 0), source_text, font=source_font)[2] > panel_w - 10 and source_font_size > 8:
            source_font_size -= 1
            source_font = font(source_font_size)
        source_box = od.textbbox((0, 0), source_text, font=source_font)
        source_width = source_box[2] - source_box[0]
        od.text((x + (panel_w - source_width) // 2, y + 5), source_text,
                fill=(25, 35, 50), font=source_font)

    # Taxonomy only: colour swatches and class names, without a heading.
    legend_y = title_h + rows * panel_h + (rows + 1) * gap + 8
    for i, cid in enumerate(display_order):
        label = dataset["classes"][cid]
        panel_x = gap + (i % cols) * panel_w
        legend_font = font(13, bold=True)
        text_box = draw.textbbox((0, 0), label, font=legend_font)
        item_width = 16 + 6 + (text_box[2] - text_box[0])
        lx = panel_x + (panel_w - item_width) // 2
        ly = legend_y + (i // cols) * 25
        draw.rectangle((lx, ly + 3, lx + 16, ly + 17), fill=dataset["colors"][cid])
        draw.text((lx + 22, ly), label, fill=(55, 60, 65), font=legend_font)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{name.lower()}_annotation_overview.png"
    canvas.save(out, quality=95)
    print(out)


if __name__ == "__main__":
    for name, dataset in DATASETS.items():
        render(name, dataset)
