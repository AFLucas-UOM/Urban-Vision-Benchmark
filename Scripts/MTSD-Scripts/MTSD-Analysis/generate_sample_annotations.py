#!/usr/bin/env python3

import json
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image, ImageDraw, ImageFont


# <repo>/Scripts/MTSD-Scripts/MTSD-Analysis/generate_sample_annotations.py
BASE_DIR = Path(__file__).resolve().parents[3]
DATASET_DIR = BASE_DIR / "Datasets" / "MTSD"
ANNOTATIONS_DIR = DATASET_DIR / "Annotations"
OUTPUT_DIR = BASE_DIR / "Documents" / "MTSD-EDA" / "SampleAnnotationImages"
SAMPLE_SIZE = 10
GROUP_PATTERN = "GRP-*"
ANNOTATION_JSON = "merged_input.json"
IMAGE_DIR = "Images"


def load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def resolve_image_path(image_dir: Path, image_name: str) -> Optional[Path]:
    direct = image_dir / image_name
    if direct.exists():
        return direct

    image_key = image_name.lower()
    lookup = {path.name.lower(): path for path in image_dir.iterdir() if path.is_file()}
    stem_lookup = {path.stem.lower(): path for path in image_dir.iterdir() if path.is_file()}
    return lookup.get(image_key) or stem_lookup.get(Path(image_key).stem)


def normalize_group_name(group: str) -> str:
    group_name = str(group).strip()
    if group_name.upper().startswith("GRP-"):
        return f"GRP-{group_name.split('-', 1)[1]}"
    return f"GRP-{group_name}"


def candidate_image_dirs(group_dir: Path) -> List[Path]:
    return [
        group_dir / IMAGE_DIR,
        group_dir / "merged_images",
        group_dir,
    ]


def find_image_dir(group_dir: Path) -> Optional[Path]:
    for image_dir in candidate_image_dirs(group_dir):
        if image_dir.exists():
            return image_dir
    return None


def find_annotation_json(group_dir: Path) -> Optional[Path]:
    candidates = [
        group_dir / ANNOTATION_JSON,
        group_dir / "merged_inputs.json",
        group_dir / "Images" / "Ignore" / ANNOTATION_JSON,
        group_dir / "Images" / "Ignore" / "merged_inputs.json",
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def collect_xml_regions(xml_path: Path) -> List[dict]:
    root = ET.parse(xml_path).getroot()
    regions = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            left = float(box.findtext("xmin", "0"))
            top = float(box.findtext("ymin", "0"))
            right = float(box.findtext("xmax", "0"))
            bottom = float(box.findtext("ymax", "0"))
        except ValueError:
            continue
        if right <= left or bottom <= top:
            continue
        regions.append(
            {
                "absolute_box": (left, top, right, bottom),
                "labels": [obj.findtext("name", "Unlabeled")],
            }
        )
    return regions


def collect_regions(task: dict) -> List[dict]:
    grouped: Dict[str, dict] = {}

    for annotation in task.get("annotations") or []:
        for result in annotation.get("result") or []:
            value = result.get("value") or {}
            region_id = result.get("id")
            if not region_id or "x" not in value or "y" not in value:
                continue

            region = grouped.setdefault(
                region_id,
                {
                    "x": value["x"],
                    "y": value["y"],
                    "width": value["width"],
                    "height": value["height"],
                    "labels": [],
                },
            )

            if result.get("type") == "rectanglelabels":
                region["labels"].extend(value.get("rectanglelabels") or [])
            elif result.get("type") == "choices":
                choices = value.get("choices") or []
                if choices:
                    region["labels"].append(f"{result.get('from_name', 'choice')}: {', '.join(choices)}")

    return list(grouped.values())


def draw_label(draw: ImageDraw.ImageDraw, box: tuple, text: str, font: ImageFont.ImageFont) -> None:
    left, top, right, _ = box
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    padding_x = 6
    padding_y = 4
    bg_left = left
    bg_top = max(0, top - text_height - (padding_y * 2) - 2)
    bg_right = left + text_width + (padding_x * 2)
    bg_bottom = bg_top + text_height + (padding_y * 2)

    draw.rectangle((bg_left, bg_top, bg_right, bg_bottom), fill=(0, 0, 0))
    draw.text((bg_left + padding_x, bg_top + padding_y), text, fill=(255, 255, 255), font=font)


def annotate_image(image_path: Path, regions: Iterable[dict], output_path: Path) -> None:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        font_size = max(16, image.width // 60)
        font = load_font(font_size)
        stroke_width = max(3, image.width // 400)

        for region in regions:
            if "absolute_box" in region:
                box = region["absolute_box"]
            else:
                left = (region["x"] / 100.0) * image.width
                top = (region["y"] / 100.0) * image.height
                right = left + (region["width"] / 100.0) * image.width
                bottom = top + (region["height"] / 100.0) * image.height
                box = (left, top, right, bottom)

            draw.rectangle(box, outline=(255, 0, 0), width=stroke_width)
            label_text = " | ".join(region["labels"]) if region["labels"] else "Unlabeled"
            draw_label(draw, box, label_text, font)

        image.save(output_path, quality=95)


def main() -> None:
    random_generator = random.SystemRandom()
    OUTPUT_DIR.mkdir(exist_ok=True)

    summaries = []
    for group_dir in sorted(DATASET_DIR.glob(GROUP_PATTERN)):
        group_name = normalize_group_name(group_dir.name)
        json_path = find_annotation_json(group_dir)
        xml_dir = ANNOTATIONS_DIR / group_name
        image_dir = find_image_dir(group_dir)
        if image_dir is None:
            continue

        candidates = []
        missing_images = []
        if xml_dir.exists():
            for xml_path in sorted(xml_dir.glob("*.xml")):
                regions = collect_xml_regions(xml_path)
                if not regions:
                    continue
                image_path = resolve_image_path(image_dir, xml_path.with_suffix("").name)
                if image_path is None:
                    missing_images.append(xml_path.name)
                    continue
                candidates.append((image_path, regions))
        elif json_path is not None:
            with json_path.open() as handle:
                tasks = json.load(handle)

            for task in tasks:
                image_name = Path(task.get("data", {}).get("image", "")).name
                if not image_name:
                    continue

                regions = collect_regions(task)
                if not regions:
                    continue

                image_path = resolve_image_path(image_dir, image_name)
                if image_path is None:
                    missing_images.append(image_name)
                    continue

                candidates.append((image_path, regions))

        if not candidates:
            continue

        sample_count = min(SAMPLE_SIZE, len(candidates))
        sampled = random_generator.sample(candidates, sample_count)
        group_output_dir = OUTPUT_DIR / group_dir.name
        group_output_dir.mkdir(parents=True, exist_ok=True)

        for existing in group_output_dir.iterdir():
            if existing.is_file():
                existing.unlink()

        for image_path, regions in sampled:
            output_path = group_output_dir / image_path.name
            annotate_image(image_path, regions, output_path)

        summaries.append(
            {
                "group": group_dir.name,
                "available": len(candidates),
                "sampled": sample_count,
                "missing_images": len(missing_images),
            }
        )

    summary_path = OUTPUT_DIR / "sample_annotation_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summaries, handle, indent=2)


if __name__ == "__main__":
    main()
