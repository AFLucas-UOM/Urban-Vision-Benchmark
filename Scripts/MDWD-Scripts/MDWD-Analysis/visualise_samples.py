#!/usr/bin/env python3
"""Render annotated MDWD sample images with their bounding boxes.

Draws each image's boxes with a fixed class-to-colour assignment (same
palette order as the charts) and writes the results to
Documents/MDWD-EDA/SampleAnnotationImages/<split>/.

Usage:
    python Scripts/MDWD-Scripts/MDWD-Analysis/visualise_samples.py
    python Scripts/MDWD-Scripts/MDWD-Analysis/visualise_samples.py --per-split 12 --seed 7
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mdwd_eda import config, plotstyle  # noqa: E402
from mdwd_eda.mapper import build_map  # noqa: E402


def load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("segoeui.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_boxes(image_path: Path, boxes, class_names: list[str], out_path: Path) -> None:
    with Image.open(image_path) as img:
        canvas = img.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        line_width = max(2, canvas.width // 320)
        font = load_font(max(12, canvas.width // 55))
        for _, box in boxes.iterrows():
            colour = plotstyle.CATEGORICAL[int(box["class_id"]) % len(plotstyle.CATEGORICAL)]
            x0 = (box["cx"] - box["box_w"] / 2) * canvas.width
            y0 = (box["cy"] - box["box_h"] / 2) * canvas.height
            x1 = (box["cx"] + box["box_w"] / 2) * canvas.width
            y1 = (box["cy"] + box["box_h"] / 2) * canvas.height
            draw.rectangle([x0, y0, x1, y1], outline=colour, width=line_width)
            label = str(box["class_name"])
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            label_y = max(0, y0 - text_h - 6)
            draw.rectangle([x0, label_y, x0 + text_w + 8, label_y + text_h + 6], fill=colour)
            draw.text((x0 + 4, label_y + 2), label, fill="white", font=font)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path, quality=90)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render annotated MDWD samples.")
    parser.add_argument("--variant", default=config.DEFAULT_VARIANT)
    parser.add_argument("--per-split", type=int, default=8, help="Samples per split (default 8).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=config.SAMPLES_DIR)
    args = parser.parse_args()

    variant = config.variant_dir(args.variant)
    if config.detect_format(variant) != "yolo":
        raise SystemExit("visualise_samples.py currently renders the YOLO variants "
                         "(the RF-DETR COCO export carries the same data).")

    dataset_map = build_map(args.variant)
    rng = random.Random(args.seed)
    rendered = 0
    for split in config.SPLITS:
        split_images = dataset_map.images[
            (dataset_map.images["split"] == split) & (dataset_map.images["n_boxes"] > 0)
        ]
        if split_images.empty:
            continue
        chosen = split_images.sample(
            n=min(args.per_split, len(split_images)), random_state=rng.randint(0, 2**31)
        )
        for _, row in chosen.iterrows():
            image_path = variant / split / "images" / row["file_name"]
            boxes = dataset_map.boxes[
                (dataset_map.boxes["split"] == split)
                & (dataset_map.boxes["file_name"] == row["file_name"])
            ]
            out_path = args.out_dir / split / f"annotated_{row['file_name']}"
            draw_boxes(image_path, boxes, dataset_map.class_names, out_path)
            rendered += 1
    print(f"Rendered {rendered} annotated samples to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
