"""Statistics tables for a mapped MDWD variant.

Each function returns a tidy DataFrame ready to be written as a CSV; the
``summary_dict`` collects the headline numbers for ``eda_summary.json``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .mapper import DatasetMap


def split_overview(dataset_map: DatasetMap) -> pd.DataFrame:
    """Images, boxes, unique sources and per-image box stats per split."""
    images, boxes = dataset_map.images, dataset_map.boxes
    rows = []
    for split in config.SPLITS:
        split_images = images[images["split"] == split]
        split_boxes = boxes[boxes["split"] == split]
        if split_images.empty:
            continue
        per_image = split_images["n_boxes"]
        rows.append(
            {
                "split": split,
                "images": len(split_images),
                "unique_source_images": split_images["source_stem"].nunique(),
                "boxes": len(split_boxes),
                "boxes_per_image_mean": round(per_image.mean(), 3),
                "boxes_per_image_median": per_image.median(),
                "boxes_per_image_max": per_image.max(),
                "images_without_boxes": int((per_image == 0).sum()),
            }
        )
    total_images = images["n_boxes"]
    rows.append(
        {
            "split": "TOTAL",
            "images": len(images),
            "unique_source_images": images["source_stem"].nunique(),
            "boxes": len(boxes),
            "boxes_per_image_mean": round(total_images.mean(), 3),
            "boxes_per_image_median": total_images.median(),
            "boxes_per_image_max": total_images.max(),
            "images_without_boxes": int((total_images == 0).sum()),
        }
    )
    return pd.DataFrame(rows)


def class_distribution(dataset_map: DatasetMap) -> pd.DataFrame:
    """Instance counts per class, overall and per split."""
    boxes = dataset_map.boxes
    if boxes.empty:
        return pd.DataFrame(columns=["class_name", "instances", "share_pct", *config.SPLITS])
    table = (
        boxes.groupby("class_name").size().rename("instances").sort_values(ascending=False).reset_index()
    )
    table["share_pct"] = (table["instances"] / table["instances"].sum() * 100).round(2)
    per_split = boxes.pivot_table(index="class_name", columns="split", aggfunc="size", fill_value=0)
    for split in config.SPLITS:
        table[split] = table["class_name"].map(per_split.get(split, pd.Series(dtype=int))).fillna(0).astype(int)
    return table


def bbox_geometry(dataset_map: DatasetMap) -> pd.DataFrame:
    """Distribution summary of box area (image fraction) and aspect ratio."""
    boxes = dataset_map.boxes
    rows = []
    for metric, series in (
        ("area_frac", boxes["area_frac"].dropna()),
        ("aspect_ratio", boxes["aspect_ratio"].dropna()),
        ("pixel_area", boxes["pixel_area"].dropna()),
    ):
        if series.empty:
            continue
        quantiles = series.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        rows.append(
            {
                "metric": metric,
                "mean": series.mean(),
                "p05": quantiles[0.05],
                "p25": quantiles[0.25],
                "median": quantiles[0.5],
                "p75": quantiles[0.75],
                "p95": quantiles[0.95],
                "min": series.min(),
                "max": series.max(),
            }
        )
    return pd.DataFrame(rows).round(5)


def size_class(pixel_area: float) -> str:
    """COCO object-size class from a box's pixel area."""
    if pixel_area <= config.COCO_SMALL_MAX_AREA:
        return "small"
    if pixel_area <= config.COCO_MEDIUM_MAX_AREA:
        return "medium"
    return "large"


def size_class_distribution(dataset_map: DatasetMap) -> pd.DataFrame:
    """COCO small/medium/large counts per class."""
    boxes = dataset_map.boxes.dropna(subset=["pixel_area"]).copy()
    if boxes.empty:
        return pd.DataFrame(columns=["class_name", "small", "medium", "large"])
    boxes["size_class"] = boxes["pixel_area"].map(size_class)
    table = boxes.pivot_table(index="class_name", columns="size_class", aggfunc="size", fill_value=0)
    for column in ("small", "medium", "large"):
        if column not in table:
            table[column] = 0
    return table[["small", "medium", "large"]].reset_index()


def resolution_distribution(dataset_map: DatasetMap) -> pd.DataFrame:
    """Image count per (width, height) resolution."""
    images = dataset_map.images.dropna(subset=["width", "height"])
    if images.empty:
        return pd.DataFrame(columns=["width", "height", "images"])
    table = (
        images.groupby(["width", "height"]).size().rename("images").sort_values(ascending=False).reset_index()
    )
    table["width"] = table["width"].astype(int)
    table["height"] = table["height"].astype(int)
    return table


def instances_per_image_histogram(dataset_map: DatasetMap) -> pd.DataFrame:
    """Image count per boxes-per-image value, per split."""
    images = dataset_map.images
    table = images.pivot_table(index="n_boxes", columns="split", aggfunc="size", fill_value=0)
    table["all_splits"] = table.sum(axis=1)
    return table.reset_index()


def class_cooccurrence(dataset_map: DatasetMap) -> pd.DataFrame:
    """How often two classes appear in the same image (diagonal = image count)."""
    boxes = dataset_map.boxes
    names = sorted(dataset_map.class_names)
    matrix = pd.DataFrame(0, index=names, columns=names, dtype=int)
    if boxes.empty:
        return matrix
    per_image = boxes.groupby("file_name")["class_name"].agg(set)
    for classes in per_image:
        for a in classes:
            for b in classes:
                matrix.loc[a, b] += 1
    return matrix


def summary_dict(dataset_map: DatasetMap) -> dict:
    """Headline numbers for eda_summary.json."""
    images, boxes = dataset_map.images, dataset_map.boxes
    per_split = {
        split: {
            "images": int((images["split"] == split).sum()),
            "boxes": int((boxes["split"] == split).sum()),
        }
        for split in config.SPLITS
    }
    return {
        "variant": dataset_map.variant,
        "format": dataset_map.format,
        "classes": dataset_map.class_names,
        "n_images": int(len(images)),
        "n_unique_source_images": int(images["source_stem"].nunique()) if not images.empty else 0,
        "n_boxes": int(len(boxes)),
        "per_split": per_split,
        "boxes_per_image_mean": round(float(images["n_boxes"].mean()), 3) if not images.empty else None,
        "median_box_area_frac": round(float(boxes["area_frac"].median()), 5) if not boxes.empty else None,
        "issue_counts": dataset_map.issue_counts,
    }


def to_native(value):
    """Recursively convert numpy scalars for json.dump."""
    if isinstance(value, dict):
        return {k: to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
