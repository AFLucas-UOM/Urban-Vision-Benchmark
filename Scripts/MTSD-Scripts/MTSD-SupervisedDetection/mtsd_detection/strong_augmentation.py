from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from .utils import fingerprint, sha256_file

Box = dict[str, Any]
Record = dict[str, Any]


def deterministic_seed(global_seed: int, *parts: object) -> int:
    material = ":".join([str(global_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def _box_payload(box: Box) -> dict[str, Any]:
    return {
        "class_id": int(box["class_id"]),
        "bbox_xyxy": [
            round(float(box["x0"]), 6),
            round(float(box["y0"]), 6),
            round(float(box["x1"]), 6),
            round(float(box["y1"]), 6),
        ],
        **({"source_annotation_id": box["source_annotation_id"]}
           if box.get("source_annotation_id") is not None else {}),
    }


def _load_rgb(record: Record) -> np.ndarray:
    with Image.open(record["source_path"]) as raw:
        return np.asarray(ImageOps.exif_transpose(raw).convert("RGB"))


def _load_rgb_scaled(
    record: Record,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
    image = _load_rgb(record)
    height, width = image.shape[:2]
    maximum = int(config.get("output", {}).get("max_long_edge", 0) or 0)
    if maximum <= 0 or max(width, height) <= maximum:
        return image, np.eye(3, dtype=np.float64), None
    scale = maximum / max(width, height)
    output_width = max(1, int(round(width * scale)))
    output_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_AREA)
    scale_x, scale_y = output_width / width, output_height / height
    matrix = np.asarray([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1],
    ], dtype=np.float64)
    operation = {
        "name": "pre_resize",
        "source_size": [width, height],
        "output_size": [output_width, output_height],
        "scale_xy": [round(scale_x, 10), round(scale_y, 10)],
        "max_long_edge": maximum,
    }
    return resized, matrix, operation


def _save_rgb(image: np.ndarray, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(
        path,
        format="JPEG",
        quality=quality,
        subsampling=0,
        optimize=False,
        progressive=False,
    )


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) / 2.0


def _clip_polygon(points: np.ndarray, width: int, height: int) -> np.ndarray:
    polygon = [(float(x), float(y)) for x, y in points]

    def clip_edge(
        values: list[tuple[float, float]],
        inside,
        intersection,
    ) -> list[tuple[float, float]]:
        if not values:
            return []
        output: list[tuple[float, float]] = []
        previous = values[-1]
        previous_inside = inside(previous)
        for current in values:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(intersection(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersection(previous, current))
            previous, previous_inside = current, current_inside
        return output

    def vertical(boundary: float):
        def intersect(a, b):
            denominator = b[0] - a[0]
            ratio = 0.0 if abs(denominator) < 1e-12 else (boundary - a[0]) / denominator
            return boundary, a[1] + ratio * (b[1] - a[1])
        return intersect

    def horizontal(boundary: float):
        def intersect(a, b):
            denominator = b[1] - a[1]
            ratio = 0.0 if abs(denominator) < 1e-12 else (boundary - a[1]) / denominator
            return a[0] + ratio * (b[0] - a[0]), boundary
        return intersect

    polygon = clip_edge(polygon, lambda p: p[0] >= 0.0, vertical(0.0))
    polygon = clip_edge(polygon, lambda p: p[0] <= width, vertical(float(width)))
    polygon = clip_edge(polygon, lambda p: p[1] >= 0.0, horizontal(0.0))
    polygon = clip_edge(polygon, lambda p: p[1] <= height, horizontal(float(height)))
    return np.asarray(polygon, dtype=np.float64).reshape((-1, 2))


def transform_boxes(
    boxes: Iterable[Box],
    matrix: np.ndarray,
    output_size: tuple[int, int],
    box_filter: dict[str, Any],
    source_name: str,
    class_remap: dict[int, int] | None = None,
) -> tuple[list[Box], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Transform axis-aligned boxes through a homography and audit every outcome."""
    width, height = output_size
    visible_min = float(box_filter["visible_area_threshold"])
    min_width = float(box_filter["min_width_px"])
    min_height = float(box_filter["min_height_px"])
    min_area = float(box_filter["min_area_px"])
    kept: list[Box] = []
    transformed: list[dict[str, Any]] = []
    clipped: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []
    remap = class_remap or {}
    for index, box in enumerate(boxes):
        original = np.asarray([
            [box["x0"], box["y0"]],
            [box["x1"], box["y0"]],
            [box["x1"], box["y1"]],
            [box["x0"], box["y1"]],
        ], dtype=np.float64)
        homogeneous = np.column_stack([original, np.ones(4)])
        projected = (matrix @ homogeneous.T).T
        valid_denominator = np.abs(projected[:, 2]) > 1e-12
        if not valid_denominator.all():
            discarded.append({
                "source_image": source_name,
                "source_box_index": index,
                "original": _box_payload(box),
                "reason": "invalid_perspective_projection",
            })
            continue
        polygon = projected[:, :2] / projected[:, 2:3]
        full_area = _polygon_area(polygon)
        visible_polygon = _clip_polygon(polygon, width, height)
        visible_area = _polygon_area(visible_polygon)
        visible_fraction = 0.0 if full_area <= 1e-9 else min(1.0, visible_area / full_area)
        if len(visible_polygon):
            x0 = max(0.0, float(visible_polygon[:, 0].min()))
            y0 = max(0.0, float(visible_polygon[:, 1].min()))
            x1 = min(float(width), float(visible_polygon[:, 0].max()))
            y1 = min(float(height), float(visible_polygon[:, 1].max()))
        else:
            x0 = y0 = x1 = y1 = 0.0
        out_width, out_height = x1 - x0, y1 - y0
        reason = None
        if visible_fraction < visible_min:
            reason = "visible_area_below_threshold"
        elif out_width <= 0 or out_height <= 0:
            reason = "degenerate_after_clip"
        elif out_width < min_width:
            reason = "width_below_minimum"
        elif out_height < min_height:
            reason = "height_below_minimum"
        elif out_width * out_height < min_area:
            reason = "area_below_minimum"
        audit = {
            "source_image": source_name,
            "source_box_index": index,
            "source_annotation_id": box.get("source_annotation_id"),
            "class_id_before": int(box["class_id"]),
            "class_id_after": int(remap.get(int(box["class_id"]), int(box["class_id"]))),
            "original_bbox_xyxy": _box_payload(box)["bbox_xyxy"],
            "projected_polygon": np.round(polygon, 6).tolist(),
            "visible_fraction": round(visible_fraction, 6),
            "output_bbox_xyxy": [round(x0, 6), round(y0, 6), round(x1, 6), round(y1, 6)],
        }
        if reason:
            discarded.append({**audit, "reason": reason})
            continue
        output = {
            "class_id": audit["class_id_after"],
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "source_annotation_id": box.get("source_annotation_id"),
            "source_image": source_name,
            "visible_fraction": visible_fraction,
        }
        kept.append(output)
        transformed.append(audit)
        if visible_fraction < 0.999999:
            clipped.append(audit)
    return kept, transformed, clipped, discarded


def validate_horizontal_flip_config(config: dict[str, Any], class_count: int) -> dict[int, int]:
    flip = config.get("horizontal_flip", {})
    if not flip.get("enabled", False):
        return {}
    if not flip.get("remap_validated", False):
        raise ValueError("Horizontal flip requires horizontal_flip.remap_validated=true")
    sensitive = {int(value) for value in flip.get("direction_sensitive_class_ids", [])}
    remap = {int(key): int(value) for key, value in flip.get("class_remap", {}).items()}
    if not sensitive or not sensitive.issubset(remap):
        raise ValueError("Horizontal flip requires an explicit remap for every direction-sensitive class")
    if any(not 0 <= key < class_count or not 0 <= value < class_count for key, value in remap.items()):
        raise ValueError("Horizontal flip class remap contains an invalid class ID")
    if any(remap.get(value) != key for key, value in remap.items()):
        raise ValueError("Horizontal flip class remap must be symmetric")
    return remap


def _apply_gamma(image: Image.Image, gamma: float) -> Image.Image:
    table = [round(255 * ((index / 255) ** gamma)) for index in range(256)]
    return image.point(table * 3)


def _motion_blur(values: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(values, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def apply_photometric(
    image: np.ndarray,
    rng: random.Random,
    np_rng: np.random.Generator,
    config: dict[str, Any],
    moderate: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    ops: list[dict[str, Any]] = []
    pil = Image.fromarray(image, mode="RGB")
    factor_strength = 0.55 if moderate else 1.0
    for name, enhancer in (
        ("brightness", ImageEnhance.Brightness),
        ("contrast", ImageEnhance.Contrast),
        ("saturation_or_color", ImageEnhance.Color),
    ):
        rule = config[name]
        if rng.random() <= float(rule.get("probability", 0.75)):
            sampled = rng.uniform(float(rule["min"]), float(rule["max"]))
            factor = 1.0 + (sampled - 1.0) * factor_strength
            pil = enhancer(pil).enhance(factor)
            ops.append({"name": name, "factor": round(factor, 6)})
    gamma_rule = config["gamma"]
    if rng.random() <= float(gamma_rule.get("probability", 0.75)):
        sampled = rng.uniform(float(gamma_rule["min"]), float(gamma_rule["max"]))
        gamma = 1.0 + (sampled - 1.0) * factor_strength
        pil = _apply_gamma(pil, gamma)
        ops.append({"name": "gamma", "gamma": round(gamma, 6)})
    values = np.asarray(pil, dtype=np.uint8)
    probability_scale = 0.5 if moderate else 1.0
    illumination = config.get("illumination", {})
    if rng.random() < float(illumination.get("partial_shadow_probability", 0.0)) * probability_scale:
        height, width = values.shape[:2]
        mask = np.zeros((height, width), dtype=np.float32)
        margin = max(width, height)
        points = np.asarray([
            [rng.randint(-margin // 3, width), rng.randint(-margin // 3, height)],
            [rng.randint(0, width + margin // 3), rng.randint(-margin // 3, height)],
            [rng.randint(0, width + margin // 3), rng.randint(0, height + margin // 3)],
            [rng.randint(-margin // 3, width), rng.randint(0, height + margin // 3)],
        ], dtype=np.int32)
        cv2.fillConvexPoly(mask, points, 1.0)
        softness = max(9, int(min(width, height) * 0.08) | 1)
        mask = cv2.GaussianBlur(mask, (softness, softness), 0)
        factor = rng.uniform(0.62 if not moderate else 0.78, 0.88)
        values = np.clip(values.astype(np.float32) * (1.0 - mask[..., None] * (1.0 - factor)), 0, 255).astype(np.uint8)
        ops.append({"name": "partial_shadow", "factor": round(factor, 6), "polygon": points.tolist()})
    if rng.random() < float(illumination.get("gradient_probability", 0.0)) * probability_scale:
        height, width = values.shape[:2]
        angle = rng.uniform(0, math.tau)
        yy, xx = np.mgrid[0:height, 0:width]
        projection = (xx / max(width - 1, 1) - 0.5) * math.cos(angle) + \
                     (yy / max(height - 1, 1) - 0.5) * math.sin(angle)
        strength = rng.uniform(0.10, 0.25 if not moderate else 0.16)
        field = np.clip(1.0 + projection * 2.0 * strength, 0.65, 1.35)
        values = np.clip(values.astype(np.float32) * field[..., None], 0, 255).astype(np.uint8)
        ops.append({"name": "illumination_gradient", "angle_radians": round(angle, 6),
                    "strength": round(strength, 6)})
    if rng.random() < float(illumination.get("local_change_probability", 0.0)) * probability_scale:
        height, width = values.shape[:2]
        cx, cy = rng.uniform(0.15, 0.85) * width, rng.uniform(0.15, 0.85) * height
        radius = rng.uniform(0.18, 0.42) * max(width, height)
        yy, xx = np.mgrid[0:height, 0:width]
        mask = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / max(2.0 * radius ** 2, 1.0))
        factor = rng.uniform(0.72, 1.28)
        values = np.clip(values.astype(np.float32) * (1.0 + mask[..., None] * (factor - 1.0)), 0, 255).astype(np.uint8)
        ops.append({"name": "local_illumination", "centre": [round(cx, 3), round(cy, 3)],
                    "radius": round(radius, 3), "factor": round(factor, 6)})
    if rng.random() < float(illumination.get("haze_probability", 0.0)) * probability_scale:
        alpha = rng.uniform(0.05, 0.16 if not moderate else 0.10)
        values = np.clip(values.astype(np.float32) * (1.0 - alpha) + 235.0 * alpha, 0, 255).astype(np.uint8)
        ops.append({"name": "mild_haze", "alpha": round(alpha, 6)})
    blur = config["gaussian_blur"]
    if rng.random() < float(blur["probability"]) * probability_scale:
        sigma = rng.uniform(0.1, float(blur["sigma_max"]) * (0.7 if moderate else 1.0))
        values = np.asarray(Image.fromarray(values).filter(ImageFilter.GaussianBlur(sigma)))
        ops.append({"name": "gaussian_blur", "sigma": round(sigma, 6)})
    motion = config["motion_blur"]
    if rng.random() < float(motion["probability"]) * probability_scale:
        kernel = int(rng.choice(list(motion["kernel_sizes"])))
        values = _motion_blur(values, kernel)
        ops.append({"name": "motion_blur", "direction": "horizontal", "kernel_size": kernel})
    noise = config["gaussian_noise"]
    if rng.random() < float(noise["probability"]) * probability_scale:
        sigma = rng.uniform(0.002, float(noise["sigma_max"]) * (0.7 if moderate else 1.0))
        noisy = values.astype(np.float32) / 255.0
        noisy += np_rng.normal(0.0, sigma, noisy.shape)
        values = np.clip(noisy * 255.0, 0, 255).astype(np.uint8)
        ops.append({"name": "gaussian_noise", "sigma": round(sigma, 6)})
    jpeg = config["jpeg_compression"]
    if rng.random() < float(jpeg["probability"]) * probability_scale:
        low = int(jpeg["quality_min"])
        high = int(jpeg["quality_max"])
        if moderate:
            low = max(low, 65)
        quality = rng.randint(low, high)
        success, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(values, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
             cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444],
        )
        if success:
            values = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            ops.append({"name": "jpeg_compression", "quality": quality, "subsampling": "4:4:4"})
    if not ops:
        factor = rng.uniform(0.92, 1.08)
        values = np.asarray(ImageEnhance.Brightness(Image.fromarray(values)).enhance(factor))
        ops.append({"name": "brightness", "factor": round(factor, 6), "fallback": True})
    return values, ops


def sample_geometric_matrix(
    width: int,
    height: int,
    rng: random.Random,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    angle = rng.uniform(*[float(v) for v in config["rotation_degrees"]])
    scale = rng.uniform(*[float(v) for v in config["scale_range"]])
    shear_x = math.tan(math.radians(rng.uniform(-float(config["shear_degrees"]), float(config["shear_degrees"]))))
    shear_y = math.tan(math.radians(rng.uniform(-float(config["shear_degrees"]), float(config["shear_degrees"]))))
    tx = rng.uniform(-float(config["translation_fraction"]), float(config["translation_fraction"])) * width
    ty = rng.uniform(-float(config["translation_fraction"]), float(config["translation_fraction"])) * height
    cx, cy = width / 2.0, height / 2.0
    centre = np.asarray([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
    uncentre = np.asarray([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    radians = math.radians(angle)
    rotate_scale = np.asarray([
        [scale * math.cos(radians), -scale * math.sin(radians), 0],
        [scale * math.sin(radians), scale * math.cos(radians), 0],
        [0, 0, 1],
    ], dtype=np.float64)
    shear = np.asarray([[1, shear_x, 0], [shear_y, 1, 0], [0, 0, 1]], dtype=np.float64)
    translate = np.asarray([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    matrix = translate @ centre @ rotate_scale @ shear @ uncentre
    parameters: dict[str, Any] = {
        "rotation_degrees": round(angle, 6),
        "scale": round(scale, 6),
        "shear_x_degrees": round(math.degrees(math.atan(shear_x)), 6),
        "shear_y_degrees": round(math.degrees(math.atan(shear_y)), 6),
        "translation_px": [round(tx, 6), round(ty, 6)],
    }
    perspective = config.get("mild_perspective", {})
    if perspective.get("enabled", False):
        distortion = float(perspective.get("distortion_fraction", 0.02))
        source = np.asarray([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        destination = source.copy()
        destination[:, 0] += np.asarray([rng.uniform(-distortion, distortion) * width for _ in range(4)])
        destination[:, 1] += np.asarray([rng.uniform(-distortion, distortion) * height for _ in range(4)])
        perspective_matrix = cv2.getPerspectiveTransform(source, destination).astype(np.float64)
        matrix = perspective_matrix @ matrix
        parameters["perspective_destination_corners"] = np.round(destination, 6).tolist()
    crop = config.get("random_crop", {})
    if crop.get("enabled", False) and rng.random() < float(crop.get("probability", 0.0)):
        retained = rng.uniform(float(crop.get("min_retained_fraction", 0.85)), 1.0)
        crop_width, crop_height = width * retained, height * retained
        crop_x = rng.uniform(0.0, width - crop_width)
        crop_y = rng.uniform(0.0, height - crop_height)
        crop_matrix = np.asarray([
            [width / crop_width, 0, -crop_x * width / crop_width],
            [0, height / crop_height, -crop_y * height / crop_height],
            [0, 0, 1],
        ], dtype=np.float64)
        matrix = crop_matrix @ matrix
        parameters["random_crop_xywh"] = [
            round(crop_x, 6), round(crop_y, 6),
            round(crop_width, 6), round(crop_height, 6),
        ]
    parameters["matrix"] = np.round(matrix, 10).tolist()
    return matrix, parameters


def _horizontal_flip_matrix(width: int) -> np.ndarray:
    return np.asarray([[-1.0, 0.0, float(width)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def _original_box_audit(records: Iterable[Record]) -> list[dict[str, Any]]:
    return [
        {"source_image": record["out_name"], **_box_payload(box)}
        for record in records
        for box in record["clean_boxes"]
    ]


def generate_photometric_sample(
    record: Record,
    rng: random.Random,
    np_rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[Box], dict[str, Any]]:
    image, resize_matrix, resize_operation = _load_rgb_scaled(record, config)
    boxes, transformed, clipped, discarded = transform_boxes(
        record["clean_boxes"],
        resize_matrix,
        (image.shape[1], image.shape[0]),
        config["box_filter"],
        record["out_name"],
    )
    image, photo_operations = apply_photometric(
        image, rng, np_rng, config["photometric"],
    )
    operations = ([resize_operation] if resize_operation else []) + photo_operations
    return image, boxes, {
        "operations": operations,
        "original_boxes": _original_box_audit([record]),
        "transformed_boxes": transformed,
        "clipped_boxes": clipped,
        "discarded_boxes": discarded,
        "sources": [record],
    }


def generate_geometric_sample(
    record: Record,
    rng: random.Random,
    np_rng: np.random.Generator,
    config: dict[str, Any],
    class_count: int,
) -> tuple[np.ndarray, list[Box], dict[str, Any]]:
    image, resize_matrix, resize_operation = _load_rgb_scaled(record, config)
    height, width = image.shape[:2]
    geometry = config["geometric"]
    matrix, parameters = sample_geometric_matrix(width, height, rng, geometry)
    remap: dict[int, int] = {}
    flip_cfg = geometry.get("horizontal_flip", {})
    if flip_cfg.get("enabled", False):
        remap = validate_horizontal_flip_config(geometry, class_count)
        if rng.random() < float(flip_cfg.get("probability", 0.0)):
            matrix = _horizontal_flip_matrix(width) @ matrix
            parameters["horizontal_flip"] = True
            parameters["matrix"] = np.round(matrix, 10).tolist()
        else:
            remap = {}
            parameters["horizontal_flip"] = False
    warped = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    validity = cv2.warpPerspective(
        np.full((height, width), 255, dtype=np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32) / 255.0
    invalid_fraction = float(np.mean(validity < 0.999))
    if invalid_fraction > float(geometry.get("maximum_border_fraction", 1.0)):
        raise ValueError(f"geometric_border_fraction={invalid_fraction:.6f}")
    if np.any(validity < 0.999):
        # Reflection avoids a flat artificial border, while heavy blur on only
        # the newly exposed region prevents duplicated signs or scene structure
        # from becoming plausible-looking unlabelled objects.
        reflected_background = cv2.GaussianBlur(
            warped, (0, 0), sigmaX=max(8.0, min(width, height) * 0.018),
        )
        blend = np.clip(validity, 0.0, 1.0)[..., None]
        warped = np.clip(
            warped.astype(np.float32) * blend +
            reflected_background.astype(np.float32) * (1.0 - blend),
            0, 255,
        ).astype(np.uint8)
    composite_matrix = matrix @ resize_matrix
    kept, transformed, clipped, discarded = transform_boxes(
        record["clean_boxes"], composite_matrix, (width, height), config["box_filter"],
        record["out_name"], class_remap=remap,
    )
    operations: list[dict[str, Any]] = (
        [resize_operation] if resize_operation else []
    ) + [{"name": "bbox_aware_geometric", **parameters}]
    operations[0]["border_fraction"] = round(invalid_fraction, 6)
    if rng.random() < float(geometry.get("photometric_probability", 0.65)):
        warped, photo_ops = apply_photometric(
            warped, rng, np_rng, config["photometric"], moderate=True,
        )
        operations.extend(photo_ops)
    return warped, kept, {
        "operations": operations,
        "original_boxes": _original_box_audit([record]),
        "transformed_boxes": transformed,
        "clipped_boxes": clipped,
        "discarded_boxes": discarded,
        "sources": [record],
    }


def _sample_mosaic_sources(anchor: Record, records: list[Record], rng: random.Random) -> list[Record]:
    alternatives = [record for record in records if record["out_name"] != anchor["out_name"]]
    if len(records) < 4:
        raise ValueError("Mosaic requires at least four training images")
    return [anchor, *rng.sample(alternatives, 3)]


def generate_mosaic_sample(
    anchor: Record,
    records: list[Record],
    rng: random.Random,
    np_rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[Box], dict[str, Any]]:
    mosaic = config["mosaic"]
    canvas_width, canvas_height = [int(value) for value in mosaic["canvas_size"]]
    centre_min, centre_max = [float(value) for value in mosaic["centre_range"]]
    centre_x = int(rng.uniform(centre_min, centre_max) * canvas_width)
    centre_y = int(rng.uniform(centre_min, centre_max) * canvas_height)
    rectangles = [
        (0, 0, centre_x, centre_y),
        (centre_x, 0, canvas_width, centre_y),
        (0, centre_y, centre_x, canvas_height),
        (centre_x, centre_y, canvas_width, canvas_height),
    ]
    sources = _sample_mosaic_sources(anchor, records, rng)
    canvas = np.full((canvas_height, canvas_width, 3), 114, dtype=np.uint8)
    kept_all: list[Box] = []
    transformed_all: list[dict[str, Any]] = []
    clipped_all: list[dict[str, Any]] = []
    discarded_all: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    for quadrant, (record, (x0, y0, x1, y1)) in enumerate(zip(sources, rectangles)):
        source, resize_matrix, resize_operation = _load_rgb_scaled(record, config)
        source_height, source_width = source.shape[:2]
        target_width, target_height = x1 - x0, y1 - y0
        scale = max(target_width / source_width, target_height / source_height)
        resized_width = max(target_width, int(round(source_width * scale)))
        resized_height = max(target_height, int(round(source_height * scale)))
        resized = cv2.resize(source, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        crop_x = rng.randint(0, max(0, resized_width - target_width))
        crop_y = rng.randint(0, max(0, resized_height - target_height))
        canvas[y0:y1, x0:x1] = resized[crop_y:crop_y + target_height, crop_x:crop_x + target_width]
        placement_matrix = np.asarray([
            [scale, 0, x0 - crop_x],
            [0, scale, y0 - crop_y],
            [0, 0, 1],
        ], dtype=np.float64)
        matrix = placement_matrix @ resize_matrix
        kept, transformed, clipped, discarded = transform_boxes(
            record["clean_boxes"], matrix, (canvas_width, canvas_height),
            config["box_filter"], record["out_name"],
        )
        kept_all.extend(kept)
        transformed_all.extend(transformed)
        clipped_all.extend(clipped)
        discarded_all.extend(discarded)
        positions.append({
            "quadrant": quadrant,
            "source_image": record["out_name"],
            "source_image_hash": record["source_image_sha256"],
            "destination_xyxy": [x0, y0, x1, y1],
            "scale": round(scale, 8),
            "source_pre_resize": resize_operation,
            "source_crop_xywh": [crop_x, crop_y, target_width, target_height],
            "matrix": np.round(matrix, 10).tolist(),
        })
    tiny_area = float(mosaic.get("tiny_object_area_px", config["box_filter"]["min_area_px"]))
    tiny_count = sum((box["x1"] - box["x0"]) * (box["y1"] - box["y0"]) < tiny_area
                     for box in kept_all)
    tiny_fraction = tiny_count / len(kept_all) if kept_all else 1.0
    if tiny_fraction > float(mosaic.get("max_tiny_object_fraction", 1.0)):
        raise ValueError(f"mosaic_tiny_object_fraction={tiny_fraction:.6f}")
    operations: list[dict[str, Any]] = [{
        "name": "four_image_mosaic",
        "canvas_size": [canvas_width, canvas_height],
        "centre": [centre_x, centre_y],
        "source_positions": positions,
        "tiny_object_fraction": round(tiny_fraction, 6),
    }]
    if rng.random() < float(mosaic.get("photometric_probability", 0.35)):
        canvas, photo_ops = apply_photometric(
            canvas, rng, np_rng, config["photometric"], moderate=True,
        )
        operations.extend(photo_ops)
    return canvas, kept_all, {
        "operations": operations,
        "original_boxes": _original_box_audit(sources),
        "transformed_boxes": transformed_all,
        "clipped_boxes": clipped_all,
        "discarded_boxes": discarded_all,
        "mosaic_source_positions": positions,
        "sources": sources,
    }


def class_frequencies(records: Iterable[Record], class_count: int) -> list[int]:
    counts = [0] * class_count
    for record in records:
        for box in record["clean_boxes"]:
            counts[int(box["class_id"])] += 1
    return counts


def identify_rare_classes(frequencies: list[int], config: dict[str, Any]) -> set[int]:
    absolute = int(config.get("rare_class_max_instances", 0))
    positive = sorted(count for count in frequencies if count > 0)
    quantile = float(config.get("rare_class_frequency_quantile", 0.0))
    quantile_limit = 0
    if positive and quantile > 0:
        quantile_limit = int(np.quantile(np.asarray(positive), quantile, method="higher"))
    active_limits = [limit for limit in (absolute, quantile_limit) if limit > 0]
    limit = min(active_limits) if active_limits else 0
    return {class_id for class_id, count in enumerate(frequencies) if 0 < count <= limit}


def _iou(candidate: Box, other: Box) -> float:
    x0 = max(candidate["x0"], other["x0"])
    y0 = max(candidate["y0"], other["y0"])
    x1 = min(candidate["x1"], other["x1"])
    y1 = min(candidate["y1"], other["y1"])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if intersection <= 0:
        return 0.0
    area_a = (candidate["x1"] - candidate["x0"]) * (candidate["y1"] - candidate["y0"])
    area_b = (other["x1"] - other["x0"]) * (other["y1"] - other["y0"])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def _context_alpha(height: int, width: int, feather_fraction: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    edge_distance = np.minimum.reduce([xx + 1, width - xx, yy + 1, height - yy]).astype(np.float32)
    feather = max(2.0, min(height, width) * feather_fraction)
    return np.clip(edge_distance / feather, 0.0, 1.0)


def generate_copy_paste_sample(
    destination: Record,
    records: list[Record],
    rare_classes: set[int],
    frequencies: list[int],
    rng: random.Random,
    np_rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[Box], dict[str, Any]]:
    copy_cfg = config["copy_paste"]
    candidates = [
        (record, box)
        for record in records
        for box in record["clean_boxes"]
        if int(box["class_id"]) in rare_classes and record["out_name"] != destination["out_name"]
    ]
    if not candidates:
        raise ValueError("copy_paste_has_no_rare_class_candidates")
    weights = [1.0 / max(1, frequencies[int(box["class_id"])]) for _, box in candidates]
    image, destination_matrix, destination_resize = _load_rgb_scaled(destination, config)
    image = image.copy()
    height, width = image.shape[:2]
    boxes, transformed, clipped, discarded = transform_boxes(
        destination["clean_boxes"],
        destination_matrix,
        (width, height),
        config["box_filter"],
        destination["out_name"],
    )
    source_records: dict[str, Record] = {destination["out_name"]: destination}
    paste_audit: list[dict[str, Any]] = []
    paste_count = rng.randint(1, int(copy_cfg["maximum_paste_count"]))
    for _ in range(paste_count):
        source_record, source_box_original = rng.choices(candidates, weights=weights, k=1)[0]
        source, source_matrix, source_resize = _load_rgb_scaled(source_record, config)
        source_height, source_width = source.shape[:2]
        source_box = {
            **source_box_original,
            "x0": source_box_original["x0"] * source_matrix[0, 0],
            "y0": source_box_original["y0"] * source_matrix[1, 1],
            "x1": source_box_original["x1"] * source_matrix[0, 0],
            "y1": source_box_original["y1"] * source_matrix[1, 1],
        }
        box_width = source_box["x1"] - source_box["x0"]
        box_height = source_box["y1"] - source_box["y0"]
        context = float(copy_cfg["context_fraction"])
        crop_x0 = max(0, int(math.floor(source_box["x0"] - box_width * context)))
        crop_y0 = max(0, int(math.floor(source_box["y0"] - box_height * context)))
        crop_x1 = min(source_width, int(math.ceil(source_box["x1"] + box_width * context)))
        crop_y1 = min(source_height, int(math.ceil(source_box["y1"] + box_height * context)))
        patch = source[crop_y0:crop_y1, crop_x0:crop_x1]
        if patch.size == 0:
            continue
        destination_scale = min(width / source_width, height / source_height)
        scale = destination_scale * rng.uniform(
            *[float(value) for value in copy_cfg["scale_range"]]
        )
        source_sign_width = source_box["x1"] - source_box["x0"]
        source_sign_height = source_box["y1"] - source_box["y0"]
        maximum_width = width * float(copy_cfg["maximum_sign_width_fraction"])
        maximum_height = height * float(copy_cfg["maximum_sign_height_fraction"])
        scale = min(
            scale,
            maximum_width / max(source_sign_width, 1e-9),
            maximum_height / max(source_sign_height, 1e-9),
        )
        target_width = max(2, int(round(patch.shape[1] * scale)))
        target_height = max(2, int(round(patch.shape[0] * scale)))
        fit = min(1.0, (width - 2) / target_width, (height - 2) / target_height)
        scale *= fit
        target_width = max(2, int(round(patch.shape[1] * scale)))
        target_height = max(2, int(round(patch.shape[0] * scale)))
        patch = cv2.resize(patch, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        local_box = {
            "x0": (source_box["x0"] - crop_x0) * scale,
            "y0": (source_box["y0"] - crop_y0) * scale,
            "x1": (source_box["x1"] - crop_x0) * scale,
            "y1": (source_box["y1"] - crop_y0) * scale,
        }
        local_width = local_box["x1"] - local_box["x0"]
        local_height = local_box["y1"] - local_box["y0"]
        filter_config = config["box_filter"]
        reason = None
        if local_width < float(filter_config["min_width_px"]):
            reason = "width_below_minimum"
        elif local_height < float(filter_config["min_height_px"]):
            reason = "height_below_minimum"
        elif local_width * local_height < float(filter_config["min_area_px"]):
            reason = "area_below_minimum"
        if reason:
            discarded.append({
                "source_image": source_record["out_name"],
                "source_annotation_id": source_box_original.get("source_annotation_id"),
                "class_id_before": int(source_box["class_id"]),
                "class_id_after": int(source_box["class_id"]),
                "original_bbox_xyxy": _box_payload(source_box_original)["bbox_xyxy"],
                "candidate_dimensions": [
                    round(local_width, 6),
                    round(local_height, 6),
                ],
                "candidate_area": round(local_width * local_height, 6),
                "visible_fraction": 1.0,
                "copy_paste": True,
                "reason": reason,
            })
            continue
        placement = None
        for _placement_attempt in range(int(copy_cfg["placement_attempts"])):
            paste_x = rng.randint(0, max(0, width - target_width))
            source_centre_y = (source_box["y0"] + source_box["y1"]) / 2 / max(source_height, 1)
            centre_y = int(np.clip(
                (source_centre_y + rng.uniform(-float(copy_cfg["vertical_jitter_fraction"]),
                                               float(copy_cfg["vertical_jitter_fraction"]))) * height,
                target_height / 2,
                height - target_height / 2,
            ))
            paste_y = int(np.clip(centre_y - target_height / 2, 0, height - target_height))
            candidate = {
                "class_id": int(source_box["class_id"]),
                "x0": paste_x + local_box["x0"], "y0": paste_y + local_box["y0"],
                "x1": paste_x + local_box["x1"], "y1": paste_y + local_box["y1"],
            }
            context_candidate = {
                "x0": paste_x, "y0": paste_y,
                "x1": paste_x + target_width, "y1": paste_y + target_height,
            }
            sign_clear = all(
                _iou(candidate, existing) <= float(copy_cfg["maximum_iou"])
                for existing in boxes
            )
            context_clear = all(
                _iou(context_candidate, existing) <=
                float(copy_cfg["maximum_context_iou"])
                for existing in boxes
            )
            if sign_clear and context_clear:
                placement = paste_x, paste_y, candidate
                break
        if placement is None:
            continue
        paste_x, paste_y, candidate = placement
        alpha = _context_alpha(target_height, target_width, float(copy_cfg["edge_feather_fraction"]))
        region = image[paste_y:paste_y + target_height, paste_x:paste_x + target_width].astype(np.float32)
        image[paste_y:paste_y + target_height, paste_x:paste_x + target_width] = np.clip(
            patch.astype(np.float32) * alpha[..., None] + region * (1.0 - alpha[..., None]),
            0, 255,
        ).astype(np.uint8)
        pasted = {
            **candidate,
            "source_annotation_id": source_box.get("source_annotation_id"),
            "source_image": source_record["out_name"],
            "visible_fraction": 1.0,
        }
        boxes.append(pasted)
        audit = {
            "source_image": source_record["out_name"],
            "source_image_hash": source_record["source_image_sha256"],
            "source_annotation_id": source_box_original.get("source_annotation_id"),
            "class_id": int(source_box["class_id"]),
            "source_bbox_xyxy": _box_payload(source_box_original)["bbox_xyxy"],
            "source_pre_resize": source_resize,
            "source_context_crop_xyxy": [crop_x0, crop_y0, crop_x1, crop_y1],
            "scale": round(scale, 8),
            "placement_xy": [paste_x, paste_y],
            "output_bbox_xyxy": [round(candidate[key], 6) for key in ("x0", "y0", "x1", "y1")],
            "maximum_iou_with_existing": round(max((_iou(candidate, existing) for existing in boxes[:-1]), default=0.0), 6),
        }
        paste_audit.append(audit)
        transformed.append({
            "source_image": source_record["out_name"],
            "source_annotation_id": source_box_original.get("source_annotation_id"),
            "class_id_before": int(source_box["class_id"]),
            "class_id_after": int(source_box["class_id"]),
            "original_bbox_xyxy": _box_payload(source_box_original)["bbox_xyxy"],
            "output_bbox_xyxy": audit["output_bbox_xyxy"],
            "visible_fraction": 1.0,
            "copy_paste": True,
        })
        source_records[source_record["out_name"]] = source_record
    if not paste_audit:
        rejected_reasons = Counter(
            row["reason"] for row in discarded if row.get("copy_paste")
        )
        detail = ",".join(
            f"{key}={value}" for key, value in sorted(rejected_reasons.items())
        )
        raise ValueError(
            "copy_paste_placement_rejected"
            + (f":{detail}" if detail else "")
        )
    operations: list[dict[str, Any]] = (
        [destination_resize] if destination_resize else []
    ) + [{"name": "rare_class_copy_paste", "pastes": paste_audit}]
    if rng.random() < float(copy_cfg.get("photometric_probability", 0.45)):
        image, photo_ops = apply_photometric(
            image, rng, np_rng, config["photometric"], moderate=True,
        )
        operations.extend(photo_ops)
    sources = list(source_records.values())
    return image, boxes, {
        "operations": operations,
        "original_boxes": _original_box_audit(sources),
        "transformed_boxes": transformed,
        "clipped_boxes": clipped,
        "discarded_boxes": discarded,
        "copy_paste_sources": paste_audit,
        "sources": sources,
    }


def _select_category(copy_index: int, rng: random.Random, config: dict[str, Any]) -> str:
    mosaic_probability = float(config["mosaic"]["probability"])
    copy_probability = float(config["copy_paste"]["probability"]) \
        if config["copy_paste"].get("enabled", False) else 0.0
    if mosaic_probability + copy_probability > 1.0:
        raise ValueError("mosaic.probability + copy_paste.probability must not exceed 1")
    draw = rng.random()
    if draw < mosaic_probability:
        return "mosaic"
    if draw < mosaic_probability + copy_probability:
        return "copy_paste"
    return "photometric" if copy_index % 2 else "geometric"


def _generated_name(record: Record, copy_index: int) -> str:
    return f"{Path(record['out_name']).stem}_augstrong{copy_index}.jpg"


def _write_yolo_label(path: Path, boxes: list[Box], width: int, height: int) -> None:
    lines = []
    for box in boxes:
        centre_x = (box["x0"] + box["x1"]) / 2.0 / width
        centre_y = (box["y0"] + box["y1"]) / 2.0 / height
        box_width = (box["x1"] - box["x0"]) / width
        box_height = (box["y1"] - box["y0"]) / height
        lines.append(f"{int(box['class_id'])} {centre_x:.8f} {centre_y:.8f} {box_width:.8f} {box_height:.8f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _draw_qa(
    image: np.ndarray,
    boxes: list[Box],
    class_names: list[str],
    path: Path,
) -> None:
    canvas = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    line_width = max(2, round(min(canvas.size) / 400))
    for box in boxes:
        colour = (255, 52 + int(box["class_id"]) * 13 % 180, 32)
        coordinates = [box["x0"], box["y0"], box["x1"], box["y1"]]
        draw.rectangle(coordinates, outline=colour, width=line_width)
        label = class_names[int(box["class_id"])]
        text_box = draw.textbbox((box["x0"], box["y0"]), label)
        draw.rectangle(text_box, fill=colour)
        draw.text((box["x0"], box["y0"]), label, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="JPEG", quality=90, subsampling=0)


def _size_bin(box: Box) -> str:
    area = (box["x1"] - box["x0"]) * (box["y1"] - box["y0"])
    if area < 32 ** 2:
        return "small_lt_32sq"
    if area < 96 ** 2:
        return "medium_32sq_to_96sq"
    return "large_ge_96sq"


def _audit_csv_row(audit: dict[str, Any]) -> dict[str, Any]:
    nested = (
        "source_images", "source_image_hashes", "applied_operations", "original_boxes",
        "transformed_boxes", "clipped_boxes", "discarded_boxes",
        "mosaic_source_positions", "copy_paste_source_annotations", "rejected_attempts",
    )
    result = {key: value for key, value in audit.items() if key not in nested}
    result.update({key: json.dumps(audit.get(key, []), sort_keys=True, separators=(",", ":"))
                   for key in nested})
    return result


def _validate_box_list(
    boxes: list[Box],
    width: int,
    height: int,
    class_count: int,
    box_filter: dict[str, Any],
) -> None:
    for box in boxes:
        if not 0 <= int(box["class_id"]) < class_count:
            raise ValueError(f"invalid class ID {box['class_id']}")
        if not (0 <= box["x0"] < box["x1"] <= width and 0 <= box["y0"] < box["y1"] <= height):
            raise ValueError(f"box outside output: {box}")
        box_width = box["x1"] - box["x0"]
        box_height = box["y1"] - box["y0"]
        if box_width < float(box_filter["min_width_px"]):
            raise ValueError(f"retained_box_width_below_minimum:{box_width}")
        if box_height < float(box_filter["min_height_px"]):
            raise ValueError(f"retained_box_height_below_minimum:{box_height}")
        if box_width * box_height < float(box_filter["min_area_px"]):
            raise ValueError(f"retained_box_area_below_minimum:{box_width * box_height}")


def _materialize_strong_sample(args: tuple[Any, ...]) -> dict[str, Any]:
    (
        variant, anchor, copy_index, forced_category, train_records,
        rare_classes, frequencies, config, class_count,
    ) = args
    sample_seed = deterministic_seed(
        config["seed"], anchor["out_name"], copy_index, config["recipe_version"],
    )
    category_rng = random.Random(sample_seed)
    requested_category = forced_category or _select_category(copy_index, category_rng, config)
    rejected_attempts: list[dict[str, Any]] = []
    output = None
    maximum_attempts = int(config["rejection"]["maximum_attempts"])
    category = requested_category
    for attempt in range(maximum_attempts):
        attempt_seed = deterministic_seed(sample_seed, attempt)
        rng = random.Random(attempt_seed)
        np_rng = np.random.default_rng(attempt_seed ^ 0xA5A5A5A5A5A5A5A5)
        try:
            if category == "photometric":
                output = generate_photometric_sample(anchor, rng, np_rng, config)
            elif category == "geometric":
                output = generate_geometric_sample(
                    anchor, rng, np_rng, config, class_count,
                )
            elif category == "mosaic":
                output = generate_mosaic_sample(
                    anchor, train_records, rng, np_rng, config,
                )
            elif category == "copy_paste":
                output = generate_copy_paste_sample(
                    anchor, train_records, rare_classes, frequencies,
                    rng, np_rng, config,
                )
            else:
                raise ValueError(f"Unknown augmentation category: {category}")
            image, boxes, audit_info = output
            if not boxes and bool(config["rejection"]["reject_empty_samples"]):
                raise ValueError("empty_augmented_sample")
            _validate_box_list(
                boxes,
                image.shape[1],
                image.shape[0],
                class_count,
                config["box_filter"],
            )
            break
        except ValueError as exc:
            rejected_attempts.append({
                "attempt": attempt + 1,
                "category": category,
                "seed": attempt_seed,
                "reason": str(exc),
            })
            output = None
    if output is None:
        category = "mosaic"
        fallback_seed = deterministic_seed(sample_seed, "fallback-mosaic")
        rng = random.Random(fallback_seed)
        np_rng = np.random.default_rng(fallback_seed ^ 0x5A5A5A5A5A5A5A5A)
        image, boxes, audit_info = generate_mosaic_sample(
            anchor, train_records, rng, np_rng, config,
        )
        if not boxes and bool(config["rejection"]["reject_empty_samples"]):
            raise RuntimeError(f"Strong augmentation exhausted attempts for {anchor['out_name']}")
        _validate_box_list(
            boxes,
            image.shape[1],
            image.shape[0],
            class_count,
            config["box_filter"],
        )
        audit_info["operations"].insert(0, {
            "name": "category_fallback",
            "requested_category": requested_category,
            "actual_category": category,
        })
    else:
        image, boxes, audit_info = output
    generated = _generated_name(anchor, copy_index)
    yolo = variant / "MTSD-YOLO"
    coco = variant / "MTSD-COCO"
    image_path = yolo / "train" / "images" / generated
    _save_rgb(image, image_path, int(config["output"]["jpeg_quality"]))
    label_path = yolo / "train" / "labels" / f"{Path(generated).stem}.txt"
    _write_yolo_label(label_path, boxes, image.shape[1], image.shape[0])
    coco_image = coco / "train" / generated
    coco_image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, coco_image)
    generated_hash = sha256_file(image_path)
    if generated_hash != sha256_file(coco_image):
        raise RuntimeError(f"YOLO/COCO generated image hash mismatch: {generated}")
    return {
        "anchor": anchor,
        "copy_index": copy_index,
        "sample_seed": sample_seed,
        "requested_category": requested_category,
        "category": category,
        "rejected_attempts": rejected_attempts,
        "boxes": boxes,
        "audit_info": audit_info,
        "generated": generated,
        "image_path": image_path,
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "generated_hash": generated_hash,
    }


def generate_strong_augmentation(
    variant: Path,
    records: list[Record],
    config: dict[str, Any],
    class_names: list[str],
    *,
    category_plan: list[str] | None = None,
    maximum_generated: int | None = None,
    source_pool: list[Record] | None = None,
) -> dict[str, Any]:
    """Generate deterministic strong train-only samples into an existing variant."""
    train_records = sorted(
        (source_pool if source_pool is not None else [record for record in records if record["split"] == "train"]),
        key=lambda record: record["out_name"],
    )
    anchors = sorted([record for record in records if record["split"] == "train"],
                     key=lambda record: record["out_name"])
    if not anchors:
        raise ValueError("Strong augmentation requires training records")
    validate_horizontal_flip_config(config["geometric"], len(class_names))
    frequencies = class_frequencies(train_records, len(class_names))
    rare_classes = identify_rare_classes(frequencies, config["copy_paste"])
    yolo, coco = variant / "MTSD-YOLO", variant / "MTSD-COCO"
    coco_path = coco / "train" / "_annotations.coco.json"
    payload = json.loads(coco_path.read_text(encoding="utf-8"))
    next_image_id = max((row["id"] for row in payload["images"]), default=0) + 1
    next_annotation_id = max((row["id"] for row in payload["annotations"]), default=0) + 1
    copies = int(config["copies_per_image"])
    sample_specs: list[tuple[Record, int, str | None]] = []
    if category_plan:
        for index, category in enumerate(category_plan):
            sample_specs.append((anchors[index % len(anchors)], index + 1, category))
    else:
        sample_specs = [(record, copy_index, None)
                        for record in anchors for copy_index in range(1, copies + 1)]
    if maximum_generated is not None:
        sample_specs = sample_specs[:maximum_generated]
    manifest_jsonl = variant / "augmentation_manifest.jsonl"
    manifest_csv = variant / "augmentation_manifest.csv"
    manifest_jsonl.parent.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    class_after: Counter[int] = Counter()
    copy_paste_added: Counter[int] = Counter()
    boxes_before = 0
    boxes_after = 0
    boxes_clipped = 0
    boxes_discarded = 0
    size_distribution: Counter[str] = Counter()
    rejected_total = 0
    qa_counts: Counter[str] = Counter()
    qa_limit = int(config.get("visual_qa", {}).get("samples_per_category", 8))
    generated_hashes: list[str] = []
    worker_count = max(1, int(config.get("output", {}).get("workers", 1)))
    cv2.setNumThreads(max(1, cv2.getNumberOfCPUs() // worker_count))
    worker_args = [
        (
            variant, anchor, copy_index, forced_category, train_records,
            rare_classes, frequencies, config, len(class_names),
        )
        for anchor, copy_index, forced_category in sample_specs
    ]
    with manifest_jsonl.open("w", encoding="utf-8", newline="\n") as jsonl:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            materialized = executor.map(_materialize_strong_sample, worker_args)
            for result in materialized:
                anchor = result["anchor"]
                copy_index = result["copy_index"]
                sample_seed = result["sample_seed"]
                requested_category = result["requested_category"]
                category = result["category"]
                rejected_attempts = result["rejected_attempts"]
                boxes = result["boxes"]
                audit_info = result["audit_info"]
                generated = result["generated"]
                image_width = result["image_width"]
                image_height = result["image_height"]
                generated_hash = result["generated_hash"]
                payload["images"].append({
                    "id": next_image_id,
                    "file_name": generated,
                    "width": image_width,
                    "height": image_height,
                    "mtsd_group": anchor["group"],
                    "augmented_from": [record["out_name"] for record in audit_info["sources"]],
                    "augmentation_recipe": config["recipe_version"],
                    "augmentation_category": category,
                })
                for box in boxes:
                    width, height = box["x1"] - box["x0"], box["y1"] - box["y0"]
                    payload["annotations"].append({
                        "id": next_annotation_id,
                        "image_id": next_image_id,
                        "category_id": int(box["class_id"]),
                        "bbox": [
                            round(float(box["x0"]), 6), round(float(box["y0"]), 6),
                            round(float(width), 6), round(float(height), 6),
                        ],
                        "area": round(float(width * height), 6),
                        "iscrowd": 0,
                        "segmentation": [],
                        "augmentation_source_image": box.get("source_image"),
                        "augmentation_source_annotation_id": box.get("source_annotation_id"),
                    })
                    next_annotation_id += 1
                    class_after[int(box["class_id"])] += 1
                    size_distribution[_size_bin(box)] += 1
                next_image_id += 1
                sources = audit_info["sources"]
                audit = {
                    "recipe_version": config["recipe_version"],
                    "global_seed": int(config["seed"]),
                    "random_seed": sample_seed,
                    "seed_material": f"{config['seed']}:{anchor['out_name']}:{copy_index}:{config['recipe_version']}",
                    "generated_filename": generated,
                    "source_images": [record["out_name"] for record in sources],
                    "source_image_hashes": [record["source_image_sha256"] for record in sources],
                    "split": "train",
                    "augmentation_category": category,
                    "requested_category": requested_category,
                    "applied_operations": audit_info["operations"],
                    "original_boxes": audit_info["original_boxes"],
                    "transformed_boxes": audit_info["transformed_boxes"],
                    "clipped_boxes": audit_info["clipped_boxes"],
                    "discarded_boxes": audit_info["discarded_boxes"],
                    "mosaic_source_positions": audit_info.get("mosaic_source_positions", []),
                    "copy_paste_source_annotations": audit_info.get("copy_paste_sources", []),
                    "rejected_attempts": rejected_attempts,
                    "image_width": image_width,
                    "image_height": image_height,
                    "box_count": len(boxes),
                    "generated_image_hash": generated_hash,
                }
                jsonl.write(json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n")
                csv_rows.append(_audit_csv_row(audit))
                category_counts[category] += 1
                copy_paste_added.update(
                    int(row["class_id"])
                    for row in audit_info.get("copy_paste_sources", [])
                )
                boxes_before += len(audit_info["original_boxes"])
                boxes_after += len(boxes)
                boxes_clipped += len(audit_info["clipped_boxes"])
                boxes_discarded += len(audit_info["discarded_boxes"])
                rejected_total += len(rejected_attempts)
                generated_hashes.append(generated_hash)
                if qa_counts[category] < qa_limit:
                    with Image.open(result["image_path"]) as qa_source:
                        qa_image = np.asarray(qa_source.convert("RGB"))
                    _draw_qa(
                        qa_image, boxes, class_names,
                        variant / "visual_qa" / category / generated,
                    )
                    qa_counts[category] += 1
    coco_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if csv_rows:
        with manifest_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    else:
        manifest_csv.write_text("", encoding="utf-8")
    class_before = class_frequencies(train_records, len(class_names))
    summary = {
        "recipe_version": config["recipe_version"],
        "recipe_fingerprint": fingerprint(config),
        "seed": int(config["seed"]),
        "original_training_images": len(train_records),
        "generated_images": len(csv_rows),
        "generated_images_by_category": dict(sorted(category_counts.items())),
        "boxes_before_augmentation": boxes_before,
        "boxes_after_augmentation": boxes_after,
        "boxes_clipped": boxes_clipped,
        "boxes_discarded": boxes_discarded,
        "generated_object_size_distribution": dict(sorted(size_distribution.items())),
        "class_distribution_before": {
            class_names[index]: count for index, count in enumerate(class_before)
        },
        "class_distribution_after_generated_only": {
            class_names[index]: class_after[index] for index in range(len(class_names))
        },
        "copy_paste_added_by_class": {
            class_names[index]: copy_paste_added[index]
            for index in range(len(class_names))
        },
        "class_distribution_after_copy_paste_additions": {
            class_names[index]: class_before[index] + copy_paste_added[index]
            for index in range(len(class_names))
        },
        "rare_classes": [class_names[index] for index in sorted(rare_classes)],
        "rejected_augmentation_attempts": rejected_total,
        "empty_augmented_samples": sum(int(row["box_count"]) == 0 for row in csv_rows),
        "generated_hash_set_fingerprint": fingerprint(generated_hashes),
        "visual_qa_samples_by_category": dict(sorted(qa_counts.items())),
    }
    qa_root = variant / "visual_qa"
    qa_root.mkdir(parents=True, exist_ok=True)
    (qa_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    return {
        "rows": len(csv_rows),
        "boxes": boxes_after,
        "summary": summary,
        "augmentation_manifest_jsonl": str(manifest_jsonl),
        "augmentation_manifest_csv": str(manifest_csv),
        "augmentation_manifest_jsonl_sha256": sha256_file(manifest_jsonl),
        "augmentation_manifest_csv_sha256": sha256_file(manifest_csv),
    }


__all__ = [
    "apply_photometric",
    "class_frequencies",
    "deterministic_seed",
    "generate_copy_paste_sample",
    "generate_geometric_sample",
    "generate_mosaic_sample",
    "generate_photometric_sample",
    "generate_strong_augmentation",
    "identify_rare_classes",
    "sample_geometric_matrix",
    "transform_boxes",
    "validate_horizontal_flip_config",
]
