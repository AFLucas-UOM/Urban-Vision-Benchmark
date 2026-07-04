#!/usr/bin/env python3
"""Prepare a GRP-* Pascal VOC annotation folder for Label Studio import."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import uuid
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = ROOT / "Datasets"
DEFAULT_CONFIG = ROOT / "Documents" / "mtsd_config.xml"
DEFAULT_GROUP = "GRP-1"
DEFAULT_DATASET = DATASETS_ROOT / DEFAULT_GROUP
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMPORT_MODE_PROMPT = "prompt"
IMPORT_MODE_QA = "qa"
IMPORT_MODE_RAW = "raw"
LABEL_ALIASES = {
    "Stop": "Stop Sign",
}

ATTRIBUTE_MAP = {
    "Viewing Angle": "view_angle",
    "Mounting Type": "mounting",
    "Condition Assessment": "condition",
    "Sign Condition": "condition",
    "Shape Annotation": "sign_shape",
    "Sign Shape": "sign_shape",
}

ATTRIBUTE_TITLES = {
    "view_angle": "Viewing Angle",
    "mounting": "Mounting Type",
    "condition": "Sign Condition",
    "sign_shape": "Sign Shape",
}


CHOICE_NAME_BY_VALUE = {
    "Front": "view_angle",
    "Side": "view_angle",
    "Back": "view_angle",
    "Pole-Mounted": "mounting",
    "Wall-Mounted": "mounting",
    "Good": "condition",
    "Weathered": "condition",
    "Heavily Damaged": "condition",
    "Circular": "sign_shape",
    "Quadrangle": "sign_shape",
    "Octagonal": "sign_shape",
    "Pentagon": "sign_shape",
    "Damaged-Unknown": "sign_shape",
}


def stable_id(*parts: object) -> str:
    return uuid.uuid5(uuid.NAMESPACE_URL, "::".join(str(p) for p in parts)).hex[:10]


def read_labelmap(path: Path) -> list[dict[str, str]]:
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(":")
        if len(fields) < 2:
            continue
        name, rgb = fields[0], fields[1]
        if name == "background":
            continue
        parts = [int(part) for part in rgb.split(",")]
        color = "#{:02X}{:02X}{:02X}".format(*parts)
        labels.append({"name": name, "color": color})
    return labels


def local_file_url(root: Path, image_path: Path) -> str:
    rel = image_path.relative_to(root).as_posix()
    return f"/data/local-files/?d={rel}"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def numeric_text(node: ET.Element | None, child: str, default: float = 0.0) -> float:
    if node is None:
        return default
    raw = node.findtext(child)
    if raw is None or raw == "":
        return default
    return float(raw)


def parse_attributes(obj: ET.Element) -> dict[str, str]:
    attrs = {}
    for attr in obj.findall("attributes/attribute"):
        name = (attr.findtext("name") or "").strip()
        value = (attr.findtext("value") or "").strip()
        if name:
            attrs[name] = value
    return attrs


def rectangle_result(
    *,
    image_name: str,
    box_index: int,
    label: str,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: int,
    height: int,
    rotation: float,
) -> tuple[str, dict[str, Any]]:
    x = xmin / width * 100
    y = ymin / height * 100
    box_width = (xmax - xmin) / width * 100
    box_height = (ymax - ymin) / height * 100
    region_id = stable_id(image_name, box_index, label, xmin, ymin, xmax, ymax)
    return region_id, {
        "id": region_id,
        "from_name": "sign_type",
        "to_name": "image",
        "type": "rectanglelabels",
        "origin": "manual",
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "value": {
            "x": x,
            "y": y,
            "width": box_width,
            "height": box_height,
            "rotation": rotation,
            "rectanglelabels": [label],
        },
    }


def choice_result(
    *,
    region_id: str,
    from_name: str,
    choice: str,
    rect_value: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    return {
        "id": region_id,
        "from_name": from_name,
        "to_name": "image",
        "type": "choices",
        "origin": "manual",
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "value": {
            "x": rect_value["x"],
            "y": rect_value["y"],
            "width": rect_value["width"],
            "height": rect_value["height"],
            "rotation": rect_value.get("rotation", 0),
            "choices": [choice],
        },
    }


def detect_format(annotation_files: list[Path]) -> str:
    if not annotation_files:
        return "none"
    sample = annotation_files[0]
    if sample.suffix.lower() == ".json":
        data = json.loads(sample.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict) and {"images", "annotations", "categories"}.issubset(data):
            return "coco_json"
    if sample.suffix.lower() == ".xml":
        root = ET.parse(sample).getroot()
        if root.tag == "annotation" and root.find("object/bndbox") is not None:
            return "pascal_voc_xml"
    return "unknown"


def load_images(images_dir: Path) -> tuple[list[Path], dict[str, Path], dict[str, Path]]:
    images = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    by_name = {p.name: p for p in images}
    by_stem = {p.stem: p for p in images}
    return images, by_name, by_stem


def first_existing(candidates: list[Path], *, kind: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    candidate_list = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find {kind}. Checked:\n{candidate_list}")


def resolve_images_dir(dataset_dir: Path, group: str) -> Path:
    return first_existing(
        [
            dataset_dir / "Images",
            dataset_dir / group,
            dataset_dir / "merged_images",
        ],
        kind=f"image directory for {group}",
    )


def resolve_annotations_dir(dataset_dir: Path, group: str) -> Path:
    return first_existing(
        [
            DATASETS_ROOT / "Annotations" / group,
            dataset_dir / "Annotations" / group,
            dataset_dir / "Annotations",
        ],
        kind=f"annotation directory for {group}",
    )


def compact_group_name(group: str) -> str:
    return group.replace("-", "")


def find_qa_annotation_files(annotations_dir: Path | None) -> list[Path]:
    if annotations_dir is None or not annotations_dir.exists():
        return []
    group = annotations_dir.name
    final_qa_dir = annotations_dir / "Final-QA"
    preferred = [
        final_qa_dir / f"QA-{compact_group_name(group)}.json",
        final_qa_dir / f"QA-{group}.json",
    ]
    final_qa_matches = sorted(final_qa_dir.glob("QA-*.json")) if final_qa_dir.exists() else []
    seen = set()
    files = []
    for candidate in [*preferred, *final_qa_matches]:
        if candidate.is_file() and candidate not in seen:
            files.append(candidate)
            seen.add(candidate)
    return files


def resolve_xml_annotation_files(annotations_dir: Path) -> list[Path]:
    fiverr_dir = annotations_dir / "Fiverr-Annotations"
    search_dir = fiverr_dir if fiverr_dir.exists() else annotations_dir
    return sorted(path for path in search_dir.rglob("*.xml") if path.is_file())


def choose_import_mode(requested_mode: str, qa_files: list[Path]) -> str:
    if requested_mode in {IMPORT_MODE_QA, IMPORT_MODE_RAW}:
        if requested_mode == IMPORT_MODE_QA and not qa_files:
            print("No Final-QA/QA-GRPX.json file was found; falling back to RAW import.")
            return IMPORT_MODE_RAW
        return requested_mode

    if not qa_files:
        print("No Final-QA/QA-GRPX.json file was found; using RAW import.")
        return IMPORT_MODE_RAW

    if not sys.stdin.isatty():
        print(f"Found {qa_files[0].name}; using QA import. Pass --import-mode raw to ignore it.")
        return IMPORT_MODE_QA

    print(f"Found QA annotations: {qa_files[0]}")
    print("Choose Label Studio import mode:")
    print("  1. QA  - load pre-labelled QA JSON annotations")
    print("  2. RAW - ignore QA JSON and create a fresh project")
    while True:
        choice = input("Import mode [QA/raw]: ").strip().lower()
        if choice in {"", "1", "q", "qa"}:
            return IMPORT_MODE_QA
        if choice in {"2", "r", "raw"}:
            return IMPORT_MODE_RAW
        print("Please enter QA or RAW.")


def resolve_imageset_path(dataset_dir: Path) -> Path | None:
    candidates = [
        dataset_dir / "Export" / "ImageSets.txt",
        dataset_dir / "Annotaiton-Export" / "ImageSets.txt",
        dataset_dir / "Annotation-Export" / "ImageSets.txt",
        dataset_dir / "ImageSets" / "ImageSets.txt",
        dataset_dir / "ImageSets" / "Main" / "default.txt",
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def resolve_labelmap_path(dataset_dir: Path) -> Path | None:
    candidates = [
        dataset_dir / "Export" / "LabelMap.txt",
        dataset_dir / "Annotaiton-Export" / "LabelMap.txt",
        dataset_dir / "Annotation-Export" / "LabelMap.txt",
        dataset_dir / "labelmap.txt",
        dataset_dir / "LabelMap.txt",
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def imageset_entries(path: Path) -> list[str]:
    if not path or not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_config(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Could not find Label Studio config: {path}")
    return path.read_text(encoding="utf-8")


def labels_from_config(config: str) -> set[str]:
    root = ET.fromstring(config)
    return {label.attrib["value"] for label in root.findall(".//Label") if label.attrib.get("value")}


def choice_values_from_config(config: str) -> dict[str, set[str]]:
    root = ET.fromstring(config)
    values = {}
    for choices in root.findall(".//Choices"):
        name = choices.attrib.get("name")
        if not name:
            continue
        values[name] = {choice.attrib["value"] for choice in choices.findall("./Choice") if choice.attrib.get("value")}
    return values


def parse_voc_xml(
    xml_path: Path,
    *,
    label_names: set[str],
    image_by_name: dict[str, Path],
    image_by_stem: dict[str, Path],
    report: dict[str, Any],
) -> tuple[Path | None, list[dict[str, Any]]]:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        report["invalid_xml"].append({"file": str(xml_path), "reason": str(exc)})
        return None, []

    filename = Path(root.findtext("filename") or f"{xml_path.stem}.jpg").name
    image_path = image_by_name.get(filename) or image_by_stem.get(Path(filename).stem) or image_by_stem.get(xml_path.stem)
    if image_path is None:
        report["annotations_without_images"].append({"annotation": str(xml_path), "filename": filename})
        return None, []

    width = int(numeric_text(root.find("size"), "width"))
    height = int(numeric_text(root.find("size"), "height"))
    if width <= 0 or height <= 0:
        report["invalid_xml"].append({"file": str(xml_path), "reason": "missing or invalid image size"})
        return image_path, []

    results = []
    for index, obj in enumerate(root.findall("object"), start=1):
        source_label = (obj.findtext("name") or "").strip()
        label = LABEL_ALIASES.get(source_label, source_label)
        box = obj.find("bndbox")
        if not label:
            report["invalid_objects"].append({"file": str(xml_path), "object": index, "reason": "missing label"})
            continue
        if label not in label_names:
            report["unknown_labels"].append({"file": str(xml_path), "object": index, "label": source_label})
        if box is None:
            report["invalid_objects"].append({"file": str(xml_path), "object": index, "reason": "missing bndbox"})
            continue
        try:
            xmin = numeric_text(box, "xmin")
            ymin = numeric_text(box, "ymin")
            xmax = numeric_text(box, "xmax")
            ymax = numeric_text(box, "ymax")
        except ValueError as exc:
            report["invalid_objects"].append({"file": str(xml_path), "object": index, "reason": f"non-numeric bndbox: {exc}"})
            continue
        if xmax <= xmin or ymax <= ymin:
            report["invalid_objects"].append({"file": str(xml_path), "object": index, "reason": "non-positive bndbox area"})
            continue
        if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
            report["out_of_bounds_boxes"].append(
                {
                    "file": str(xml_path),
                    "object": index,
                    "box": [xmin, ymin, xmax, ymax],
                    "image_size": [width, height],
                }
            )

        attrs = parse_attributes(obj)
        rotation = float(attrs.get("rotation", "0") or 0)
        region_id, rect = rectangle_result(
            image_name=image_path.name,
            box_index=index,
            label=label,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            width=width,
            height=height,
            rotation=rotation,
        )
        results.append(rect)
        for source_name, value in attrs.items():
            if source_name == "rotation":
                continue
            from_name = ATTRIBUTE_MAP.get(source_name)
            if not from_name:
                report["unmapped_attributes"].append(
                    {"file": str(xml_path), "object": index, "attribute": source_name, "value": value}
                )
                continue
            report["attribute_values"][from_name][value] += 1
            results.append(
                choice_result(
                    region_id=region_id,
                    from_name=from_name,
                    choice=value,
                    rect_value=rect["value"],
                    width=width,
                    height=height,
                )
            )
    return image_path, results


def parse_annotation_attributes(
    ann: dict[str, Any],
    *,
    choice_values: dict[str, set[str]],
    report: dict[str, Any],
) -> dict[str, str]:
    raw = ann.get("attributes") or ann.get("attribute") or {}
    if not isinstance(raw, dict):
        return {}

    attrs = {}
    for source_name, raw_value in raw.items():
        if raw_value in (None, ""):
            continue
        if str(source_name).strip() in {
            "label_studio_region_id",
            "label_studio_annotation_id",
            "origin",
            "rotation",
        }:
            continue
        value = str(raw_value).strip()
        from_name = ATTRIBUTE_MAP.get(str(source_name).strip()) or str(source_name).strip()
        if from_name not in choice_values:
            from_name = CHOICE_NAME_BY_VALUE.get(value, from_name)
        if from_name not in choice_values:
            report["unmapped_attributes"].append(
                {"annotation": ann.get("id"), "attribute": source_name, "value": value}
            )
            continue
        if value not in choice_values[from_name]:
            report["unknown_attribute_values"].append(
                {"annotation": ann.get("id"), "from_name": from_name, "value": value}
            )
        attrs[from_name] = value
    return attrs


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_width * inter_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def parse_coco_json(
    json_path: Path,
    *,
    label_names: set[str],
    choice_values: dict[str, set[str]],
    image_by_name: dict[str, Path],
    report: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    categories = {category["id"]: category["name"] for category in data.get("categories", [])}
    image_rows_by_id = {image["id"]: image for image in data.get("images", [])}
    image_names_by_id = {image_id: Path(row.get("file_name", "")).name for image_id, row in image_rows_by_id.items()}

    seen_image_names = Counter(image_names_by_id.values())
    duplicate_image_names = {name for name, count in seen_image_names.items() if name and count > 1}
    report["duplicate_coco_images"] = sorted(duplicate_image_names)

    results_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_boxes_by_image: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = defaultdict(list)
    for ann in data.get("annotations", []):
        image_name = image_names_by_id.get(ann.get("image_id"))
        if not image_name:
            report["annotations_without_images"].append({"annotation": ann.get("id"), "filename": None})
            continue
        image_path = image_by_name.get(image_name)
        if image_path is None:
            report["annotations_without_images"].append({"annotation": ann.get("id"), "filename": image_name})
            continue

        source_label = categories.get(ann.get("category_id"), "")
        label = LABEL_ALIASES.get(source_label, source_label)
        if label in CHOICE_NAME_BY_VALUE:
            report["orphan_attribute_categories"].append(
                {"annotation": ann.get("id"), "filename": image_name, "category": source_label}
            )
            continue
        if label not in label_names:
            report["unknown_labels"].append({"annotation": ann.get("id"), "filename": image_name, "label": source_label})
            continue

        bbox = ann.get("bbox") or []
        if len(bbox) != 4:
            report["invalid_objects"].append({"annotation": ann.get("id"), "reason": "missing or invalid COCO bbox"})
            continue
        xmin, ymin, box_width, box_height = (float(v) for v in bbox)
        xmax = xmin + box_width
        ymax = ymin + box_height
        width = int(image_rows_by_id[ann["image_id"]].get("width") or 0)
        height = int(image_rows_by_id[ann["image_id"]].get("height") or 0)
        if width <= 0 or height <= 0 or xmax <= xmin or ymax <= ymin:
            report["invalid_objects"].append({"annotation": ann.get("id"), "reason": "invalid COCO bbox or image size"})
            continue
        if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
            report["out_of_bounds_boxes"].append(
                {
                    "annotation": ann.get("id"),
                    "box": [xmin, ymin, xmax, ymax],
                    "image_size": [width, height],
                }
            )

        pixel_box = (xmin, ymin, xmax, ymax)
        duplicate = None
        if image_name in duplicate_image_names:
            duplicate = next(
                (
                    seen_box
                    for seen_label, seen_box in seen_boxes_by_image[image_name]
                    if seen_label == label and box_iou(pixel_box, seen_box) >= 0.98
                ),
                None,
            )
        if duplicate:
            report["duplicate_coco_annotations"].append(
                {
                    "annotation": ann.get("id"),
                    "filename": image_name,
                    "label": label,
                    "box": [xmin, ymin, xmax, ymax],
                    "matched_box": list(duplicate),
                }
            )
            continue
        seen_boxes_by_image[image_name].append((label, pixel_box))

        region_id, rect = rectangle_result(
            image_name=image_name,
            box_index=ann.get("id", len(results_by_image[image_name]) + 1),
            label=label,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            width=width,
            height=height,
            rotation=0,
        )
        rect["meta"] = {
            "source_annotation_format": "coco_json",
            "source_annotation": {
                key: value
                for key, value in ann.items()
                if key not in {"bbox", "category_id", "image_id"}
            },
            "source_category": source_label,
        }
        results_by_image[image_name].append(rect)

        for from_name, value in parse_annotation_attributes(ann, choice_values=choice_values, report=report).items():
            report["attribute_values"][from_name][value] += 1
            results_by_image[image_name].append(
                choice_result(
                    region_id=region_id,
                    from_name=from_name,
                    choice=value,
                    rect_value=rect["value"],
                    width=width,
                    height=height,
                )
            )

    return dict(results_by_image)


def result_percent_box(result: dict[str, Any]) -> tuple[float, float, float, float]:
    value = result["value"]
    x1 = float(value["x"])
    y1 = float(value["y"])
    x2 = x1 + float(value["width"])
    y2 = y1 + float(value["height"])
    return x1, y1, x2, y2


def enrich_results_with_xml_attributes(
    results_by_image: dict[str, list[dict[str, Any]]],
    xml_files: list[Path],
    *,
    label_names: set[str],
    choice_values: dict[str, set[str]],
    image_by_name: dict[str, Path],
    image_by_stem: dict[str, Path],
    report: dict[str, Any],
    min_iou: float = 0.90,
) -> None:
    candidates_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    attribute_report: dict[str, Any] = {
        "invalid_xml": [],
        "invalid_objects": [],
        "unknown_labels": [],
        "unmapped_attributes": [],
        "out_of_bounds_boxes": [],
        "annotations_without_images": [],
        "attribute_values": defaultdict(Counter),
    }

    for xml_path in xml_files:
        image_path, raw_results = parse_voc_xml(
            xml_path,
            label_names=label_names,
            image_by_name=image_by_name,
            image_by_stem=image_by_stem,
            report=attribute_report,
        )
        if image_path is None:
            continue

        attrs_by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
        rects = []
        for result in raw_results:
            if result["type"] == "rectanglelabels":
                rects.append(result)
            elif result["type"] == "choices":
                attrs_by_region[result["id"]].append(result)

        for rect in rects:
            candidates_by_image[image_path.name].append(
                {
                    "label": rect["value"]["rectanglelabels"][0],
                    "box": result_percent_box(rect),
                    "attributes": attrs_by_region.get(rect["id"], []),
                    "used": False,
                }
            )

    enriched = 0
    unmatched = 0
    skipped_unknown_values = []
    for image_name, results in results_by_image.items():
        candidates = candidates_by_image.get(image_name, [])
        for rect in [result for result in results if result["type"] == "rectanglelabels"]:
            label = rect["value"]["rectanglelabels"][0]
            rect_box = result_percent_box(rect)
            existing_choice_names = {
                result["from_name"]
                for result in results
                if result["type"] == "choices" and result["id"] == rect["id"]
            }
            best_candidate = None
            best_iou = 0.0
            for candidate in candidates:
                if candidate["used"] or candidate["label"] != label:
                    continue
                iou = box_iou(rect_box, candidate["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_candidate = candidate
            if best_candidate is None or best_iou < min_iou:
                unmatched += 1
                continue

            best_candidate["used"] = True
            added_for_region = 0
            for attr in best_candidate["attributes"]:
                from_name = attr["from_name"]
                if from_name in existing_choice_names:
                    continue
                values = attr["value"].get("choices") or []
                if not values:
                    continue
                value = values[0]
                if from_name in choice_values and value not in choice_values[from_name]:
                    skipped_unknown_values.append(
                        {"image": image_name, "from_name": from_name, "value": value, "label": label}
                    )
                    continue
                results.append(
                    choice_result(
                        region_id=rect["id"],
                        from_name=from_name,
                        choice=value,
                        rect_value=rect["value"],
                        width=int(rect["original_width"]),
                        height=int(rect["original_height"]),
                    )
                )
                report["attribute_values"][from_name][value] += 1
                existing_choice_names.add(from_name)
                added_for_region += 1
            if added_for_region:
                enriched += 1

    report["qa_xml_attribute_enrichment"] = {
        "xml_file_count": len(xml_files),
        "regions_enriched": enriched,
        "regions_without_xml_attribute_match": unmatched,
        "skipped_unknown_attribute_values": skipped_unknown_values,
    }


def build_config(group: str, labels: list[dict[str, str]], attribute_values: dict[str, Counter[str]]) -> str:
    hotkeys = list("1234567890qwertyuiopasdfghjklzxcvbnm")
    label_lines = []
    for index, label in enumerate(labels):
        hotkey = f' hotkey="{hotkeys[index]}"' if index < len(hotkeys) else ""
        label_lines.append(
            f'      <Label value="{escape_xml(label["name"])}" background="{label["color"]}"{hotkey}/>'
        )

    attr_blocks = []
    for from_name in ("view_angle", "mounting", "condition", "sign_shape"):
        values = sorted(v for v in attribute_values.get(from_name, {}) if v)
        if not values:
            continue
        choice_lines = "\n".join(f'    <Choice value="{escape_xml(value)}"/>' for value in values)
        attr_blocks.append(
            f'''  <Choices name="{from_name}" toName="image" perRegion="true" choice="single" required="false" showInline="true">
    <Header value="{ATTRIBUTE_TITLES[from_name]}"/>
{choice_lines}
  </Choices>'''
        )

    return f'''<View>
  <Header value="{escape_xml(group)} Traffic Signs Review"/>

  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>

  <RectangleLabels name="sign_type" toName="image" showInline="true">
{chr(10).join(label_lines)}
  </RectangleLabels>

{chr(10).join(attr_blocks)}
</View>
'''


def escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default=DEFAULT_GROUP, help="Dataset group name, for example GRP-1.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, help="Defaults to <dataset-dir>/labelstudio_output.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Shared Label Studio config XML.")
    parser.add_argument("--annotations-file", type=Path, help="Use a single annotation file, for example a COCO JSON export.")
    parser.add_argument(
        "--import-mode",
        choices=[IMPORT_MODE_PROMPT, IMPORT_MODE_QA, IMPORT_MODE_RAW],
        default=IMPORT_MODE_PROMPT,
        help="Choose QA pre-labelled annotations when available, or RAW fresh-project generation.",
    )
    args = parser.parse_args()

    group = args.group
    default_dataset = DATASETS_ROOT / group
    dataset_dir = (args.dataset_dir if args.dataset_dir != DEFAULT_DATASET else default_dataset).resolve()
    explicit_annotations_file = args.annotations_file.resolve() if args.annotations_file else None
    images_dir = resolve_images_dir(dataset_dir, group)
    annotations_dir = resolve_annotations_dir(dataset_dir, group)
    qa_files = find_qa_annotation_files(annotations_dir)
    selected_mode = choose_import_mode(args.import_mode, qa_files) if not explicit_annotations_file else IMPORT_MODE_QA
    qa_annotations_file = qa_files[0].resolve() if selected_mode == IMPORT_MODE_QA and qa_files else None
    annotations_file = explicit_annotations_file or qa_annotations_file
    imageset_path = resolve_imageset_path(dataset_dir)
    labelmap_path = resolve_labelmap_path(dataset_dir)
    config_path = args.config.resolve()
    config = read_config(config_path)
    label_names = labels_from_config(config)
    choice_values = choice_values_from_config(config)
    out_dir = (args.output_dir or dataset_dir / "labelstudio_output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images, image_by_name, image_by_stem = load_images(images_dir)
    source_xml_files = resolve_xml_annotation_files(annotations_dir)
    xml_files = [] if annotations_file else source_xml_files
    annotation_files = [annotations_file] if annotations_file else xml_files
    entries = imageset_entries(imageset_path)

    report: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "group": group,
        "import_mode": selected_mode,
        "annotation_format": detect_format(annotation_files),
        "images_dir": str(images_dir),
        "annotations_dir": str(annotations_dir) if annotations_dir else None,
        "xml_annotations_dir": str((annotations_dir / "Fiverr-Annotations") if (annotations_dir / "Fiverr-Annotations").exists() else annotations_dir),
        "annotations_file": str(annotations_file) if annotations_file else None,
        "explicit_annotations_file": str(explicit_annotations_file) if explicit_annotations_file else None,
        "qa_annotation_files": [str(path) for path in qa_files],
        "selected_qa_annotation_file": str(qa_annotations_file) if qa_annotations_file else None,
        "imageset_path": str(imageset_path) if imageset_path else None,
        "config_path": str(config_path),
        "labelmap_path": str(labelmap_path) if labelmap_path else None,
        "image_count": len(images),
        "annotation_file_count": len(annotation_files),
        "imageset_entry_count": len(entries),
        "label_count": len(label_names),
        "labels": sorted(label_names),
        "invalid_xml": [],
        "invalid_objects": [],
        "unknown_labels": [],
        "unknown_attribute_values": [],
        "unmapped_attributes": [],
        "orphan_attribute_categories": [],
        "duplicate_coco_images": [],
        "duplicate_coco_annotations": [],
        "qa_xml_attribute_enrichment": None,
        "out_of_bounds_boxes": [],
        "annotations_without_images": [],
        "images_without_annotations": [],
        "images_missing_from_imageset": [],
        "imageset_entries_without_images": [],
        "annotation_files_missing_from_imageset": [],
        "attribute_values": defaultdict(Counter),
    }

    results_by_image: dict[str, list[dict[str, Any]]] = {}
    annotation_stems = set()
    label_counts: Counter[str] = Counter()
    region_count = 0

    if annotations_file:
        if detect_format([annotations_file]) != "coco_json":
            raise SystemExit(f"Unsupported annotations file format: {annotations_file}")
        results_by_image = parse_coco_json(
            annotations_file,
            label_names=label_names,
            choice_values=choice_values,
            image_by_name=image_by_name,
            report=report,
        )
        if qa_annotations_file:
            enrich_results_with_xml_attributes(
                results_by_image,
                source_xml_files,
                label_names=label_names,
                choice_values=choice_values,
                image_by_name=image_by_name,
                image_by_stem=image_by_stem,
                report=report,
            )
        annotation_stems = set(Path(name).stem for name in results_by_image)
    else:
        for xml_path in xml_files:
            annotation_stems.add(xml_path.stem)
            image_path, results = parse_voc_xml(
                xml_path,
                label_names=label_names,
                image_by_name=image_by_name,
                image_by_stem=image_by_stem,
                report=report,
            )
            if image_path is None:
                continue
            results_by_image[image_path.name] = results

    for results in results_by_image.values():
        for result in results:
            if result["type"] == "rectanglelabels":
                label_counts.update(result["value"]["rectanglelabels"])
                region_count += 1

    for image_path in images:
        if image_path.name not in results_by_image:
            report["images_without_annotations"].append(image_path.name)

    image_stems = {p.stem for p in images}
    imageset_stems = {Path(entry).name for entry in entries}
    report["images_missing_from_imageset"] = sorted(image_stems - imageset_stems)
    report["imageset_entries_without_images"] = sorted(imageset_stems - image_stems)
    report["annotation_files_missing_from_imageset"] = sorted(annotation_stems - imageset_stems)

    tasks = []
    for image_path in images:
        task = {
            "data": {
                "image": local_file_url(ROOT, image_path),
                "image_filename": image_path.name,
            },
            "meta": {
                "source_image": str(image_path.relative_to(ROOT)),
                "mime_type": mimetypes.guess_type(image_path.name)[0] or "image/jpeg",
            },
            "annotations": [],
        }
        results = results_by_image.get(image_path.name, [])
        if results:
            task["annotations"].append({"result": results})
        tasks.append(task)

    report["task_count"] = len(tasks)
    report["annotated_task_count"] = sum(1 for task in tasks if task["annotations"])
    report["box_annotation_count"] = region_count
    report["label_counts"] = dict(sorted(label_counts.items()))
    report["attribute_values"] = {key: dict(sorted(counter.items())) for key, counter in report["attribute_values"].items()}
    report["valid_for_import"] = not (
        report["invalid_xml"]
        or report["invalid_objects"]
        or report["annotations_without_images"]
        or report["images_without_annotations"]
        or report["imageset_entries_without_images"]
        or report["unknown_labels"]
    )

    selected_task_path = out_dir / f"{group}.{selected_mode}.tasks.json"
    task_path = out_dir / f"{group}.tasks.json"
    selected_task_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    task_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    (out_dir / "config.xml").write_bytes(config_path.read_bytes())
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = {
        "group": group,
        "import_mode": selected_mode,
        "task_file": display_path(task_path),
        "mode_task_file": display_path(selected_task_path),
        "config_file": display_path(out_dir / "config.xml"),
        "qa_annotation_file": display_path(qa_annotations_file) if qa_annotations_file else None,
    }
    (out_dir / "import_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        f"# {group} Label Studio Import Validation",
        "",
        f"- Import mode: `{report['import_mode'].upper()}`",
        f"- Detected annotation format: `{report['annotation_format']}`",
        f"- Images: {report['image_count']}",
        f"- Annotation files: {report['annotation_file_count']}",
        f"- ImageSets entries: {report['imageset_entry_count']}",
        f"- Tasks: {report['task_count']}",
        f"- Tasks with annotations: {report['annotated_task_count']}",
        f"- Bounding boxes: {report['box_annotation_count']}",
        f"- Valid for import: {report['valid_for_import']}",
        "",
        "## Issues",
        f"- Invalid XML files: {len(report['invalid_xml'])}",
        f"- Invalid objects skipped: {len(report['invalid_objects'])}",
        f"- Images without annotations: {len(report['images_without_annotations'])}",
        f"- Annotations without images: {len(report['annotations_without_images'])}",
        f"- ImageSets entries without images: {len(report['imageset_entries_without_images'])}",
        f"- Images missing from ImageSets: {len(report['images_missing_from_imageset'])}",
        f"- Annotation files missing from ImageSets: {len(report['annotation_files_missing_from_imageset'])}",
        f"- Unknown labels: {len(report['unknown_labels'])}",
        f"- Unknown attribute values: {len(report['unknown_attribute_values'])}",
        f"- Unmapped attributes: {len(report['unmapped_attributes'])}",
        f"- Orphan attribute-category annotations skipped: {len(report['orphan_attribute_categories'])}",
        f"- Duplicate COCO image entries merged by filename: {len(report['duplicate_coco_images'])}",
        f"- Duplicate COCO annotations skipped: {len(report['duplicate_coco_annotations'])}",
        f"- Out-of-bounds boxes imported but reported: {len(report['out_of_bounds_boxes'])}",
        "",
        "## Label Counts",
    ]
    for label, count in report["label_counts"].items():
        lines.append(f"- {label}: {count}")
    (out_dir / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {selected_task_path}")
    print(f"Wrote {task_path}")
    print(f"Wrote {out_dir / 'config.xml'}")
    print(f"Wrote {out_dir / 'import_manifest.json'}")
    print(f"Wrote {out_dir / 'validation_report.md'}")
    print(f"{report['image_count']} images, {report['annotation_file_count']} annotation files, {region_count} boxes")


if __name__ == "__main__":
    main()
