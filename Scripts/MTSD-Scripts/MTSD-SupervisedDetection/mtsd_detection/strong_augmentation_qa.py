from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .dataset_validation import validate_variant
from .utils import sha256_file

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VISUAL_CATEGORIES = ("photometric", "geometric", "mosaic", "copy_paste")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_yolo(path: Path, width: int, height: int) -> list[tuple[int, list[float]]]:
    rows: list[tuple[int, list[float]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_text, cx_text, cy_text, width_text, height_text = line.split()
        cx = float(cx_text) * width
        cy = float(cy_text) * height
        box_width = float(width_text) * width
        box_height = float(height_text) * height
        rows.append((
            int(class_text),
            [
                cx - box_width / 2,
                cy - box_height / 2,
                cx + box_width / 2,
                cy + box_height / 2,
            ],
        ))
    return rows


def _coco_xyxy(annotation: dict[str, Any]) -> list[float]:
    x, y, width, height = [float(value) for value in annotation["bbox"]]
    return [x, y, x + width, y + height]


def _close(left: Iterable[float], right: Iterable[float], tolerance: float) -> bool:
    return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right))


def _resize_matrix(operation: dict[str, Any] | None) -> np.ndarray:
    if not operation:
        return np.eye(3, dtype=np.float64)
    scale_x, scale_y = operation["scale_xy"]
    return np.asarray([
        [float(scale_x), 0.0, 0.0],
        [0.0, float(scale_y), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _project_box(box: Iterable[float], matrix: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = [float(value) for value in box]
    corners = np.asarray([
        [x0, y0, 1.0],
        [x1, y0, 1.0],
        [x1, y1, 1.0],
        [x0, y1, 1.0],
    ], dtype=np.float64)
    projected = (matrix @ corners.T).T
    return projected[:, :2] / projected[:, 2:3]


def _intersection_bounds(
    points: np.ndarray,
    width: int,
    height: int,
) -> tuple[list[float], float]:
    rectangle = np.asarray([
        [0.0, 0.0],
        [float(width), 0.0],
        [float(width), float(height)],
        [0.0, float(height)],
    ], dtype=np.float32)
    full_area = abs(float(cv2.contourArea(points.astype(np.float32))))
    visible_area, intersection = cv2.intersectConvexConvex(
        points.astype(np.float32),
        rectangle,
    )
    if intersection is None or visible_area <= 0:
        return [0.0, 0.0, 0.0, 0.0], 0.0
    intersection = np.asarray(intersection, dtype=np.float64).reshape((-1, 2))
    bounds = [
        float(intersection[:, 0].min()),
        float(intersection[:, 1].min()),
        float(intersection[:, 0].max()),
        float(intersection[:, 1].max()),
    ]
    visible_fraction = min(1.0, float(visible_area) / max(full_area, 1e-12))
    return bounds, visible_fraction


def _row_matrices(row: dict[str, Any]) -> dict[str, np.ndarray]:
    if row.get("mosaic_source_positions"):
        return {
            str(position["source_image"]): np.asarray(position["matrix"], dtype=np.float64)
            for position in row["mosaic_source_positions"]
        }
    resize = next(
        (operation for operation in row["applied_operations"] if operation["name"] == "pre_resize"),
        None,
    )
    geometry = next(
        (operation for operation in row["applied_operations"]
         if operation["name"] == "bbox_aware_geometric"),
        None,
    )
    matrix = _resize_matrix(resize)
    if geometry:
        matrix = np.asarray(geometry["matrix"], dtype=np.float64) @ matrix
    return {str(source): matrix for source in row["source_images"]}


def _candidate_key(name: str, category: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{category}:{name}".encode("utf-8")).digest()[:8],
        "big",
    )


def _keep_lowest(
    heap: list[tuple[int, str, dict[str, Any]]],
    key: int,
    name: str,
    row: dict[str, Any],
    limit: int,
) -> None:
    entry = (-key, name, row)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif key < -heap[0][0]:
        heapq.heapreplace(heap, entry)


def _risk_score(row: dict[str, Any]) -> float:
    visibility = [
        float(box.get("visible_fraction", 1.0))
        for box in row.get("transformed_boxes", [])
    ]
    displacement = 0.0
    for box in row.get("transformed_boxes", []):
        if box.get("copy_paste") or not box.get("projected_polygon"):
            continue
        original = box["original_bbox_xyxy"]
        output = box["output_bbox_xyxy"]
        original_diagonal = max(
            math.hypot(original[2] - original[0], original[3] - original[1]),
            1.0,
        )
        displacement = max(
            displacement,
            math.hypot(
                (output[0] + output[2] - original[0] - original[2]) / 2,
                (output[1] + output[3] - original[1] - original[3]) / 2,
            ) / original_diagonal,
        )
    return (
        1000.0 * len(row.get("clipped_boxes", []))
        + 250.0 * len(row.get("discarded_boxes", []))
        + 100.0 * (1.0 - min(visibility, default=1.0))
        + displacement
    )


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, float, int, int]:
    width, height = size
    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    return resized, scale, (width - resized.width) // 2, (height - resized.height) // 2


def _draw_boxes(
    canvas: Image.Image,
    boxes: list[tuple[int, list[float]]],
    class_names: list[str],
    scale: float,
    offset_x: int,
    offset_y: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    label_font = _font(15)
    palette = (
        "#ff3b30", "#007aff", "#34c759", "#ff9500", "#af52de", "#00a6a6",
        "#e91e63", "#795548", "#5c6bc0", "#8bc34a", "#ff5722", "#00897b",
    )
    for class_id, box in boxes:
        x0, y0, x1, y1 = [
            offset_x + float(box[index]) * scale if index % 2 == 0
            else offset_y + float(box[index]) * scale
            for index in range(4)
        ]
        color = palette[class_id % len(palette)]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        label = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        text_box = draw.textbbox((x0, y0), label, font=label_font)
        text_width = text_box[2] - text_box[0] + 6
        text_height = text_box[3] - text_box[1] + 4
        label_y = max(0, y0 - text_height)
        draw.rectangle((x0, label_y, x0 + text_width, label_y + text_height), fill=color)
        draw.text((x0 + 3, label_y + 1), label, fill="white", font=label_font)


def _source_panel(
    row: dict[str, Any],
    source_paths: dict[str, Path],
    class_names: list[str],
    size: tuple[int, int],
) -> Image.Image:
    sources = list(dict.fromkeys(str(value) for value in row["source_images"]))
    columns = 2 if len(sources) > 1 else 1
    rows = math.ceil(len(sources) / columns)
    tile_width = size[0] // columns
    tile_height = size[1] // rows
    panel = Image.new("RGB", size, "#202124")
    grouped: dict[str, list[tuple[int, list[float]]]] = defaultdict(list)
    for box in row.get("original_boxes", []):
        grouped[str(box["source_image"])].append(
            (int(box["class_id"]), [float(value) for value in box["bbox_xyxy"]])
        )
    draw = ImageDraw.Draw(panel)
    title_font = _font(14)
    for index, source in enumerate(sources):
        source_path = source_paths[source]
        with Image.open(source_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        fitted, scale, inner_x, inner_y = _fit(image, (tile_width, tile_height - 24))
        tile_x = index % columns * tile_width
        tile_y = index // columns * tile_height
        panel.paste(fitted, (tile_x + inner_x, tile_y + 24 + inner_y))
        _draw_boxes(
            panel,
            grouped[source],
            class_names,
            scale,
            tile_x + inner_x,
            tile_y + 24 + inner_y,
        )
        draw.rectangle((tile_x, tile_y, tile_x + tile_width, tile_y + 23), fill="#111315")
        draw.text((tile_x + 5, tile_y + 4), source, fill="white", font=title_font)
    return panel


def _generated_panel(
    image_path: Path,
    boxes: list[tuple[int, list[float]]],
    class_names: list[str],
    size: tuple[int, int],
) -> Image.Image:
    with Image.open(image_path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    panel = Image.new("RGB", size, "#202124")
    fitted, scale, offset_x, offset_y = _fit(image, size)
    panel.paste(fitted, (offset_x, offset_y))
    _draw_boxes(panel, boxes, class_names, scale, offset_x, offset_y)
    return panel


def _render_sample(
    row: dict[str, Any],
    source_paths: dict[str, Path],
    generated_root: Path,
    coco_by_name: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
    class_names: list[str],
    output: Path,
) -> None:
    panel_size = (700, 520)
    source = _source_panel(row, source_paths, class_names, panel_size)
    image_row, annotations = coco_by_name[row["generated_filename"]]
    boxes = [(int(annotation["category_id"]), _coco_xyxy(annotation)) for annotation in annotations]
    generated = _generated_panel(
        generated_root / image_row["file_name"],
        boxes,
        class_names,
        panel_size,
    )
    header = 42
    canvas = Image.new("RGB", (panel_size[0] * 2, panel_size[1] + header), "white")
    canvas.paste(source, (0, header))
    canvas.paste(generated, (panel_size[0], header))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), "BEFORE: source annotation", fill="#111111", font=_font(18))
    draw.text(
        (panel_size[0] + 8, 10),
        f"AFTER: {row['augmentation_category']} | {row['generated_filename']}",
        fill="#111111",
        font=_font(18),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90, subsampling=0)


def _contact_sheet(images: list[Path], output: Path) -> None:
    tile_size = (460, 200)
    columns = 5
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), "#17191c")
    for index, path in enumerate(images):
        with Image.open(path) as raw:
            thumb, _, offset_x, offset_y = _fit(raw.convert("RGB"), tile_size)
        x = index % columns * tile_size[0] + offset_x
        y = index // columns * tile_size[1] + offset_y
        sheet.paste(thumb, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90, subsampling=0)


def _write_html(
    output: Path,
    sections: dict[str, list[Path]],
    report_name: str,
) -> None:
    blocks = []
    for name, images in sections.items():
        blocks.append(f"<h2>{name.replace('_', ' ').title()}</h2>")
        blocks.append("<div class='grid'>")
        for image in images:
            relative = image.relative_to(output.parent).as_posix()
            blocks.append(
                f"<a href='{relative}'><img loading='lazy' src='{relative}' "
                f"alt='{image.stem}'><span>{image.stem}</span></a>"
            )
        blocks.append("</div>")
    output.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{report_name}</title><style>"
        "body{font-family:Arial,sans-serif;margin:24px;background:#f5f6f8;color:#17191c}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px}"
        "a{background:white;border:1px solid #d7d9dd;padding:8px;color:#17191c;text-decoration:none}"
        "img{display:block;width:100%;height:auto}span{display:block;margin-top:6px;font-size:13px}"
        "</style></head><body>"
        f"<h1>{report_name}</h1><p>Left: source annotation. Right: generated sample.</p>"
        + "".join(blocks)
        + "</body></html>\n",
        encoding="utf-8",
    )


def _add_error(
    errors: list[dict[str, Any]],
    code: str,
    generated: str,
    detail: str,
    limit: int = 250,
) -> None:
    if len(errors) < limit:
        errors.append({"code": code, "generated_filename": generated, "detail": detail})


def run_strong_qa(
    dataset_root: Path,
    unaugmented_root: Path,
    output_root: Path,
    config: dict[str, Any],
    *,
    samples_per_category: int = 25,
    risk_samples: int = 25,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Immutable QA output already exists: {output_root}")
    output_root.mkdir(parents=True)
    prep_path = dataset_root / "prep_manifest.json"
    split_path = dataset_root / "split_manifest.csv"
    audit_path = dataset_root / "augmentation_manifest.jsonl"
    for required in (prep_path, split_path, audit_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    class_names = [str(value) for value in prep["class_names"]]
    class_count = len(class_names)
    strong_config = config["strong_augmentation"]
    box_filter = strong_config["box_filter"]
    split_rows = _read_csv(split_path)
    split_by_name = {row["out_name"]: row["split"] for row in split_rows}
    split_by_hash = {row["source_image_sha256"]: row["split"] for row in split_rows}
    source_paths = {row["out_name"]: Path(row["source_image_path"]) for row in split_rows}
    source_hashes = {row["out_name"]: row["source_image_sha256"] for row in split_rows}

    coco_path = dataset_root / "MTSD-COCO" / "train" / "_annotations.coco.json"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    coco_by_name = {
        str(image["file_name"]): (image, annotations_by_image[int(image["id"])])
        for image in coco["images"]
    }

    baseline_validation = validate_variant(
        dataset_root,
        strict=True,
        policy=config["validation"],
    )
    errors: list[dict[str, Any]] = []
    mismatch_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    discard_reasons: Counter[str] = Counter()
    total_boxes_before = 0
    total_boxes_after = 0
    total_clipped = 0
    total_discarded = 0
    total_rejected = 0
    generated_count = 0
    empty_generated = 0
    invalid_classes = 0
    invalid_boxes = 0
    out_of_bounds_boxes = 0
    visual_heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = {
        category: [] for category in VISUAL_CATEGORIES
    }
    highest_risk: list[tuple[float, str, dict[str, Any]]] = []
    generated_names: set[str] = set()
    transformation_checks = Counter()

    with audit_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            generated = str(row["generated_filename"])
            category = str(row["augmentation_category"])
            generated_count += 1
            generated_names.add(generated)
            category_counts[category] += 1
            total_boxes_before += len(row.get("original_boxes", []))
            total_boxes_after += len(row.get("transformed_boxes", []))
            total_clipped += len(row.get("clipped_boxes", []))
            total_discarded += len(row.get("discarded_boxes", []))
            total_rejected += len(row.get("rejected_attempts", []))
            empty_generated += int(not row.get("transformed_boxes"))
            for operation in row["applied_operations"]:
                operation_counts[str(operation["name"])] += 1
            for discarded in row.get("discarded_boxes", []):
                reason = str(discarded.get("reason") or "missing_reason")
                discard_reasons[reason] += 1
                if reason == "missing_reason":
                    _add_error(errors, "discard_reason_missing", generated, json.dumps(discarded))

            if row.get("split") != "train":
                _add_error(errors, "generated_non_train_split", generated, str(row.get("split")))
            if row.get("recipe_version") != "strong-offline-v2":
                _add_error(errors, "recipe_mismatch", generated, str(row.get("recipe_version")))
            if len(row.get("source_images", [])) != len(row.get("source_image_hashes", [])):
                _add_error(errors, "source_audit_incomplete", generated, "source/hash counts differ")
            for source, source_hash in zip(row["source_images"], row["source_image_hashes"]):
                if split_by_name.get(source) != "train":
                    _add_error(errors, "source_split_leakage", generated, f"{source}: {split_by_name.get(source)}")
                if split_by_hash.get(source_hash) != "train":
                    _add_error(errors, "source_hash_leakage", generated, str(source_hash))
                if source_hashes.get(source) != source_hash:
                    _add_error(errors, "source_hash_mismatch", generated, str(source))

            image_path = dataset_root / "MTSD-YOLO" / "train" / "images" / generated
            coco_image_path = dataset_root / "MTSD-COCO" / "train" / generated
            try:
                with Image.open(image_path) as raw:
                    raw.verify()
                with Image.open(image_path) as raw:
                    actual_size = ImageOps.exif_transpose(raw).size
                    actual_mode = raw.mode
            except Exception as exc:
                _add_error(errors, "generated_image_unreadable", generated, str(exc))
                continue
            expected_size = (int(row["image_width"]), int(row["image_height"]))
            if actual_size != expected_size:
                _add_error(errors, "generated_dimension_mismatch", generated, f"{actual_size} != {expected_size}")
            if actual_mode not in {"RGB", "L"}:
                _add_error(errors, "unexpected_generated_color_mode", generated, actual_mode)
            expected_hash = str(row.get("generated_image_hash") or "")
            if not expected_hash or sha256_file(image_path) != expected_hash:
                _add_error(errors, "generated_hash_mismatch", generated, str(image_path))
            if not coco_image_path.is_file() or sha256_file(coco_image_path) != expected_hash:
                _add_error(errors, "coco_generated_hash_mismatch", generated, str(coco_image_path))

            coco_entry = coco_by_name.get(generated)
            if coco_entry is None:
                _add_error(errors, "generated_missing_from_coco", generated, "")
                continue
            image_row, coco_annotations = coco_entry
            if (int(image_row["width"]), int(image_row["height"])) != expected_size:
                _add_error(errors, "coco_dimension_mismatch", generated, json.dumps(image_row))
            yolo_path = dataset_root / "MTSD-YOLO" / "train" / "labels" / f"{Path(generated).stem}.txt"
            yolo_boxes = _read_yolo(yolo_path, *expected_size)
            transformed = row.get("transformed_boxes", [])
            if not (len(yolo_boxes) == len(coco_annotations) == len(transformed) == int(row["box_count"])):
                mismatch_counts["annotation_count"] += 1
                _add_error(
                    errors,
                    "annotation_count_mismatch",
                    generated,
                    f"yolo={len(yolo_boxes)} coco={len(coco_annotations)} audit={len(transformed)}",
                )
            for index, (yolo_box, annotation) in enumerate(zip(yolo_boxes, coco_annotations)):
                coco_box = _coco_xyxy(annotation)
                if yolo_box[0] != int(annotation["category_id"]) or not _close(yolo_box[1], coco_box, 0.11):
                    mismatch_counts["yolo_coco"] += 1
                    _add_error(errors, "yolo_coco_mismatch", generated, f"box {index}")

            matrices = _row_matrices(row)
            originals = {
                (
                    str(box["source_image"]),
                    box.get("source_annotation_id"),
                    int(box["class_id"]),
                ): [float(value) for value in box["bbox_xyxy"]]
                for box in row.get("original_boxes", [])
            }
            paste_rows = list(row.get("copy_paste_source_annotations", []))
            for index, transformed_box in enumerate(transformed):
                output_box = [float(value) for value in transformed_box["output_bbox_xyxy"]]
                class_before = int(transformed_box["class_id_before"])
                class_after = int(transformed_box["class_id_after"])
                if not 0 <= class_after < class_count:
                    invalid_classes += 1
                    _add_error(errors, "invalid_class", generated, f"box {index}: {class_after}")
                if class_before != class_after:
                    _add_error(errors, "unexpected_class_remap", generated, f"{class_before} -> {class_after}")
                x0, y0, x1, y1 = output_box
                width, height = expected_size
                if x1 <= x0 or y1 <= y0:
                    invalid_boxes += 1
                    _add_error(errors, "degenerate_box", generated, f"box {index}: {output_box}")
                if x0 < -1e-6 or y0 < -1e-6 or x1 > width + 1e-6 or y1 > height + 1e-6:
                    out_of_bounds_boxes += 1
                    _add_error(errors, "out_of_bounds_box", generated, f"box {index}: {output_box}")
                if (
                    x1 - x0 < float(box_filter["min_width_px"]) - 1e-6
                    or y1 - y0 < float(box_filter["min_height_px"]) - 1e-6
                    or (x1 - x0) * (y1 - y0) < float(box_filter["min_area_px"]) - 1e-6
                ):
                    invalid_boxes += 1
                    _add_error(errors, "tiny_box_retained", generated, f"box {index}: {output_box}")
                visible = float(transformed_box.get("visible_fraction", 1.0))
                if visible < float(box_filter["visible_area_threshold"]) - 1e-6:
                    invalid_boxes += 1
                    _add_error(errors, "visibility_threshold_violation", generated, f"box {index}: {visible}")
                if index < len(coco_annotations):
                    annotation = coco_annotations[index]
                    if class_after != int(annotation["category_id"]) or not _close(
                        output_box, _coco_xyxy(annotation), 1e-4
                    ):
                        mismatch_counts["audit_coco"] += 1
                        _add_error(errors, "audit_coco_mismatch", generated, f"box {index}")

                source_key = (
                    str(transformed_box["source_image"]),
                    transformed_box.get("source_annotation_id"),
                    class_before,
                )
                if source_key not in originals:
                    _add_error(errors, "source_annotation_not_audited", generated, str(source_key))
                if transformed_box.get("copy_paste"):
                    matching_pastes = [
                        paste for paste in paste_rows
                        if (
                            str(paste["source_image"]) == source_key[0]
                            and paste.get("source_annotation_id") == source_key[1]
                            and _close(output_box, paste["output_bbox_xyxy"], 1e-6)
                        )
                    ]
                    if not matching_pastes:
                        _add_error(errors, "copy_paste_box_mismatch", generated, f"box {index}")
                    transformation_checks["copy_paste"] += 1
                    continue
                matrix = matrices.get(source_key[0])
                if matrix is None:
                    _add_error(errors, "missing_transform_matrix", generated, source_key[0])
                    continue
                projected = _project_box(transformed_box["original_bbox_xyxy"], matrix)
                recorded_polygon = np.asarray(transformed_box.get("projected_polygon", []), dtype=np.float64)
                if recorded_polygon.shape != (4, 2) or not np.allclose(
                    projected, recorded_polygon, atol=2e-3, rtol=0.0
                ):
                    mismatch_counts["projected_polygon"] += 1
                    _add_error(errors, "projected_polygon_mismatch", generated, f"box {index}")
                expected_box, expected_visible = _intersection_bounds(
                    projected, width, height,
                )
                if not _close(output_box, expected_box, 2e-3):
                    mismatch_counts["transform_box"] += 1
                    _add_error(errors, "transformed_box_mismatch", generated, f"box {index}")
                if abs(visible - expected_visible) > 2e-3:
                    mismatch_counts["visible_fraction"] += 1
                    _add_error(errors, "visible_fraction_mismatch", generated, f"box {index}")
                transformation_checks[
                    "mosaic" if row.get("mosaic_source_positions")
                    else "geometric" if any(
                        operation["name"] == "bbox_aware_geometric"
                        for operation in row["applied_operations"]
                    )
                    else "pre_resize_or_identity"
                ] += 1

            clipped_keys = {
                (
                    str(box["source_image"]),
                    box.get("source_annotation_id"),
                    tuple(box["output_bbox_xyxy"]),
                )
                for box in row.get("clipped_boxes", [])
            }
            for clipped in row.get("clipped_boxes", []):
                visible = float(clipped.get("visible_fraction", 0.0))
                if not float(box_filter["visible_area_threshold"]) <= visible < 1.0:
                    _add_error(errors, "invalid_clipped_visibility", generated, str(visible))
            for transformed_box in transformed:
                if float(transformed_box.get("visible_fraction", 1.0)) < 0.999999:
                    key = (
                        str(transformed_box["source_image"]),
                        transformed_box.get("source_annotation_id"),
                        tuple(transformed_box["output_bbox_xyxy"]),
                    )
                    if key not in clipped_keys:
                        _add_error(errors, "clipped_box_not_recorded", generated, str(key))

            geometry_operation = next(
                (operation for operation in row["applied_operations"]
                 if operation["name"] == "bbox_aware_geometric"),
                None,
            )
            if geometry_operation and geometry_operation.get("horizontal_flip"):
                _add_error(errors, "horizontal_flip_used", generated, "disabled strong recipe")
            if category == "mosaic":
                positions = row.get("mosaic_source_positions", [])
                if len(positions) != 4 or len({position["source_image"] for position in positions}) != 4:
                    _add_error(errors, "invalid_mosaic_sources", generated, f"count={len(positions)}")
                if any(split_by_name.get(position["source_image"]) != "train" for position in positions):
                    _add_error(errors, "mosaic_source_leakage", generated, "")
            if category == "copy_paste":
                pastes = row.get("copy_paste_source_annotations", [])
                if not pastes:
                    _add_error(errors, "copy_paste_audit_empty", generated, "")
                if any(split_by_name.get(paste["source_image"]) != "train" for paste in pastes):
                    _add_error(errors, "copy_paste_source_leakage", generated, "")

            if category in visual_heaps:
                _keep_lowest(
                    visual_heaps[category],
                    _candidate_key(generated, category),
                    generated,
                    row,
                    samples_per_category,
                )
            risk = _risk_score(row)
            risk_entry = (risk, generated, row)
            if len(highest_risk) < risk_samples:
                heapq.heappush(highest_risk, risk_entry)
            elif risk > highest_risk[0][0]:
                heapq.heapreplace(highest_risk, risk_entry)

    for split in ("valid", "test"):
        for base in (
            dataset_root / "MTSD-YOLO" / split / "images",
            dataset_root / "MTSD-COCO" / split,
        ):
            for path in base.glob("*"):
                if path.is_file() and (
                    path.name in generated_names or "_augstrong" in path.stem
                ):
                    _add_error(errors, "generated_derivative_split_leakage", path.name, split)

    unaug_split = unaugmented_root / "split_manifest.csv"
    split_manifest_unchanged = (
        unaug_split.is_file()
        and sha256_file(split_path) == sha256_file(unaug_split)
        and prep.get("split_manifest_sha256") == sha256_file(split_path)
    )
    if not split_manifest_unchanged:
        _add_error(errors, "original_split_manifest_changed", "", str(unaug_split))

    for finding in baseline_validation["findings"]:
        if finding["severity"] == "fatal":
            _add_error(
                errors,
                f"strict_validator:{finding['code']}",
                "",
                finding["detail"],
            )

    required_operations = {
        "brightness", "contrast", "saturation_or_color", "gamma",
        "gaussian_blur", "gaussian_noise", "jpeg_compression", "motion_blur",
        "bbox_aware_geometric", "four_image_mosaic", "rare_class_copy_paste",
    }
    missing_operations = sorted(required_operations - set(operation_counts))
    for operation in missing_operations:
        _add_error(errors, "required_operation_not_observed", "", operation)

    automated_pass = (
        not errors
        and baseline_validation["ok"]
        and generated_count > 0
        and all(category_counts[category] >= samples_per_category for category in VISUAL_CATEGORIES)
    )

    visual_root = output_root / "visual_qa"
    sections: dict[str, list[Path]] = {}
    selected_rows: dict[str, list[dict[str, Any]]] = {
        category: [
            entry[2]
            for entry in sorted(heap, key=lambda item: (-item[0], item[1]))
        ]
        for category, heap in visual_heaps.items()
    }
    selected_rows["highest_risk"] = [
        entry[2]
        for entry in sorted(highest_risk, key=lambda item: (-item[0], item[1]))
    ]
    generated_root = dataset_root / "MTSD-COCO" / "train"
    for category, rows in selected_rows.items():
        images: list[Path] = []
        for index, row in enumerate(rows, 1):
            output = visual_root / category / f"{index:02d}_{Path(row['generated_filename']).stem}.jpg"
            _render_sample(
                row,
                source_paths,
                generated_root,
                coco_by_name,
                class_names,
                output,
            )
            images.append(output)
        sections[category] = images
        _contact_sheet(images, visual_root / f"contact_sheet_{category}.jpg")
    html_path = visual_root / "index.html"
    _write_html(html_path, sections, "MTSD Strong Offline Augmentation Bounding-Box QA")

    report = {
        "schema_version": "mtsd-strong-augmentation-qa-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root.resolve()),
        "dataset_version": prep.get("dataset_version"),
        "prep_manifest_path": str(prep_path.resolve()),
        "prep_manifest_sha256": sha256_file(prep_path),
        "split_manifest_path": str(split_path.resolve()),
        "split_manifest_sha256": sha256_file(split_path),
        "augmentation_manifest_path": str(audit_path.resolve()),
        "augmentation_manifest_sha256": sha256_file(audit_path),
        "recipe_version": prep.get("augmentation", {}).get("recipe_version"),
        "global_seed": strong_config["seed"],
        "total_generated_images": generated_count,
        "total_boxes_before_augmentation": total_boxes_before,
        "total_boxes_after_augmentation": total_boxes_after,
        "boxes_clipped": total_clipped,
        "boxes_discarded": total_discarded,
        "discard_reasons": dict(sorted(discard_reasons.items())),
        "empty_generated_samples": empty_generated,
        "empty_samples_rejected_or_retried": total_rejected,
        "images_by_augmentation_type": dict(sorted(category_counts.items())),
        "operations_observed": dict(sorted(operation_counts.items())),
        "transformation_checks": dict(sorted(transformation_checks.items())),
        "yolo_coco_mismatches": int(sum(mismatch_counts.values())),
        "mismatch_counts": dict(sorted(mismatch_counts.items())),
        "split_leakage_findings": sum(
            1 for error in errors if "leakage" in error["code"]
        ),
        "original_split_manifest_unchanged": split_manifest_unchanged,
        "invalid_classes": invalid_classes,
        "invalid_boxes": invalid_boxes,
        "out_of_bounds_boxes": out_of_bounds_boxes,
        "strict_validator": baseline_validation,
        "automated_errors": errors,
        "automated_error_count": len(errors),
        "automated_status": "pass" if automated_pass else "fail",
        "transformation_unit_tests": {"status": "pending", "report": None},
        "visual_qa": {
            "review_status": "pending",
            "samples_per_category": samples_per_category,
            "additional_high_risk_samples": risk_samples,
            "root": str(visual_root.resolve()),
            "html_report": str(html_path.resolve()),
            "contact_sheets": {
                category: str((visual_root / f"contact_sheet_{category}.jpg").resolve())
                for category in selected_rows
            },
        },
        "final_status": "pending_visual_review" if automated_pass else "fail",
    }
    report_path = output_root / "qa_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def finalize_report(
    report_path: Path,
    visual_decision: str,
    tests_passed: bool,
    test_report: Path | None,
    notes: str,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if visual_decision not in {"pass", "fail"}:
        raise ValueError("visual_decision must be pass or fail")
    report["visual_qa"].update({
        "review_status": visual_decision,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "review_notes": notes,
    })
    report["transformation_unit_tests"] = {
        "status": "pass" if tests_passed else "fail",
        "report": str(test_report.resolve()) if test_report else None,
    }
    passed = (
        report.get("automated_status") == "pass"
        and visual_decision == "pass"
        and tests_passed
    )
    report["final_status"] = "pass" if passed else "fail"
    report["finalized_at"] = datetime.now(timezone.utc).isoformat()
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = ["finalize_report", "run_strong_qa"]
