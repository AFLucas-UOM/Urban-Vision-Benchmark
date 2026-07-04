"""Loading and validation of the QA-verified COCO annotation exports.

Only the ``QA-GRP*.json`` files under ``Datasets/MTSD/Annotations/*/Final-QA`` are
ever read - they are the verified ground truth.  The per-file COCO ids are
re-indexed into globally unique ids; the source group is kept purely as
provenance (it identifies which QA file a record came from and where the
image lives on disk, and carries no analytical meaning).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config

ATTRIBUTE_KEYS = ("view_angle", "mounting", "condition", "sign_shape", "origin")

# COCO size convention (areas in squared pixels).
COCO_SMALL_MAX = 32**2
COCO_MEDIUM_MAX = 96**2


def discover_qa_files(annotations_root: Path = config.ANNOTATIONS_ROOT) -> list[Path]:
    """Return every QA-verified COCO export, sorted by group number."""
    def sort_key(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else 10**9, path.name)

    return sorted(annotations_root.glob(config.QA_ANNOTATION_GLOB), key=sort_key)


def load_qa_dataset(
    annotations_root: Path = config.ANNOTATIONS_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge all QA files into ``(images, annotations, categories)`` frames.

    Category ids are identical across files (a shared label map); image and
    annotation ids are re-indexed to be globally unique.
    """
    qa_files = discover_qa_files(annotations_root)
    if not qa_files:
        raise FileNotFoundError(
            f"No QA annotation files found under {annotations_root} "
            f"(expected {config.QA_ANNOTATION_GLOB})"
        )

    image_rows: list[dict] = []
    annotation_rows: list[dict] = []
    categories: dict[int, str] = {}
    next_image_uid = 1
    next_annotation_uid = 1

    for qa_file in qa_files:
        group = qa_file.parents[1].name  # Annotations/<GRP-x>/Final-QA/QA-*.json
        with open(qa_file, encoding="utf-8-sig") as handle:
            payload = json.load(handle)

        for category in payload.get("categories", []):
            existing = categories.setdefault(category["id"], category["name"])
            if existing != category["name"]:
                raise ValueError(
                    f"Category id {category['id']} maps to both "
                    f"'{existing}' and '{category['name']}' ({qa_file.name})"
                )

        uid_by_local_id: dict[int, int] = {}
        for image in payload.get("images", []):
            uid = next_image_uid
            next_image_uid += 1
            uid_by_local_id[image["id"]] = uid
            image_rows.append(
                {
                    "image_uid": uid,
                    "file_name": image.get("file_name"),
                    "width": image.get("width"),
                    "height": image.get("height"),
                    "source_group": group,
                    "source_file": qa_file.name,
                    "source_image": image.get("source_image"),
                }
            )

        for annotation in payload.get("annotations", []):
            x, y, w, h = (annotation.get("bbox") or [None] * 4)[:4]
            attributes = annotation.get("attributes") or {}
            annotation_rows.append(
                {
                    "annotation_uid": next_annotation_uid,
                    "image_uid": uid_by_local_id.get(annotation.get("image_id")),
                    "category_id": annotation.get("category_id"),
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_w": w,
                    "bbox_h": h,
                    "iscrowd": annotation.get("iscrowd", 0),
                    **{key: attributes.get(key) for key in ATTRIBUTE_KEYS},
                }
            )
            next_annotation_uid += 1

    images = pd.DataFrame(image_rows)
    annotations = pd.DataFrame(annotation_rows)
    category_frame = (
        pd.DataFrame(
            [{"category_id": cid, "category_name": name} for cid, name in categories.items()]
        )
        .sort_values("category_id")
        .reset_index(drop=True)
    )

    annotations = annotations.merge(category_frame, on="category_id", how="left")
    annotations = _add_bbox_features(annotations, images)
    return images, annotations, category_frame


def _add_bbox_features(annotations: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Attach absolute and image-relative bounding-box geometry."""
    dims = images.set_index("image_uid")[["width", "height"]].rename(
        columns={"width": "image_width", "height": "image_height"}
    )
    merged = annotations.join(dims, on="image_uid")

    merged["bbox_area"] = merged["bbox_w"] * merged["bbox_h"]
    merged["bbox_aspect"] = merged["bbox_w"] / merged["bbox_h"]
    merged["bbox_cx"] = merged["bbox_x"] + merged["bbox_w"] / 2
    merged["bbox_cy"] = merged["bbox_y"] + merged["bbox_h"] / 2
    merged["rel_width"] = merged["bbox_w"] / merged["image_width"]
    merged["rel_height"] = merged["bbox_h"] / merged["image_height"]
    merged["rel_area"] = merged["bbox_area"] / (merged["image_width"] * merged["image_height"])
    merged["rel_cx"] = merged["bbox_cx"] / merged["image_width"]
    merged["rel_cy"] = merged["bbox_cy"] / merged["image_height"]
    merged["size_class"] = pd.cut(
        merged["bbox_area"],
        bins=[0, COCO_SMALL_MAX, COCO_MEDIUM_MAX, float("inf")],
        labels=["small (<32² px)", "medium (32²-96² px)", "large (>96² px)"],
        right=True,
    )
    return merged


def annotation_quality_report(
    images: pd.DataFrame,
    annotations: pd.DataFrame,
    inventory: pd.DataFrame | None = None,
    boundary_tolerance: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """Run consistency checks and return one DataFrame of offenders per check.

    Checks: degenerate boxes, out-of-bounds boxes, sub-pixel-thin boxes,
    duplicated boxes inside one image, orphan annotations, images without
    annotations, missing auxiliary attributes, and (when an image inventory
    is supplied) mismatches between the QA files and the images on disk.
    """
    report: dict[str, pd.DataFrame] = {}
    detail_columns = [
        "annotation_uid", "image_uid", "category_name",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "image_width", "image_height",
    ]

    degenerate = annotations[(annotations["bbox_w"] <= 0) | (annotations["bbox_h"] <= 0)]
    report["degenerate_boxes"] = degenerate[detail_columns]

    out_of_bounds = annotations[
        (annotations["bbox_x"] < -boundary_tolerance)
        | (annotations["bbox_y"] < -boundary_tolerance)
        | (annotations["bbox_x"] + annotations["bbox_w"] > annotations["image_width"] + boundary_tolerance)
        | (annotations["bbox_y"] + annotations["bbox_h"] > annotations["image_height"] + boundary_tolerance)
    ]
    report["out_of_bounds_boxes"] = out_of_bounds[detail_columns]

    tiny = annotations[
        ((annotations["bbox_w"] < 4) | (annotations["bbox_h"] < 4))
        & (annotations["bbox_w"] > 0)
        & (annotations["bbox_h"] > 0)
    ]
    report["suspiciously_small_boxes"] = tiny[detail_columns]

    rounded = annotations.assign(
        _key=annotations[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].round(0).astype(str).agg("|".join, axis=1)
    )
    duplicated_mask = rounded.duplicated(subset=["image_uid", "category_id", "_key"], keep=False)
    report["duplicate_boxes"] = annotations[duplicated_mask][detail_columns]

    orphans = annotations[~annotations["image_uid"].isin(images["image_uid"])]
    report["orphan_annotations"] = orphans[["annotation_uid", "image_uid", "category_name"]]

    annotated_uids = set(annotations["image_uid"].dropna())
    unannotated = images[~images["image_uid"].isin(annotated_uids)]
    report["images_without_annotations"] = unannotated[["image_uid", "file_name", "source_group"]]

    attribute_missing = annotations[
        annotations[list(ATTRIBUTE_KEYS[:4])].isna().any(axis=1)
    ]
    report["missing_attributes"] = attribute_missing[
        ["annotation_uid", "image_uid", "category_name", *ATTRIBUTE_KEYS[:4]]
    ]

    if inventory is not None:
        annotated_groups = sorted(images["source_group"].unique())
        disk = inventory[inventory["group"].isin(annotated_groups)]
        disk_names = set(disk["filename"])
        qa_names = set(images["file_name"])
        report["qa_entries_missing_on_disk"] = images[
            ~images["file_name"].isin(disk_names)
        ][["image_uid", "file_name", "source_group"]]
        report["disk_images_missing_in_qa"] = disk[
            ~disk["filename"].isin(qa_names)
        ][["group", "filename", "relative_path"]]

    return report


def class_imbalance_metrics(annotations: pd.DataFrame) -> dict[str, float]:
    """Summary metrics describing how skewed the class distribution is."""
    counts = annotations["category_name"].value_counts().sort_values(ascending=False)
    share = counts / counts.sum()
    cumulative = share.cumsum()

    # Gini coefficient over class frequencies (0 = perfectly balanced).
    sorted_counts = counts.sort_values().to_numpy(dtype=float)
    n = len(sorted_counts)
    index = range(1, n + 1)
    gini = float(
        (2 * sum(i * c for i, c in zip(index, sorted_counts)) - (n + 1) * sorted_counts.sum())
        / (n * sorted_counts.sum())
    )

    return {
        "num_classes": int(n),
        "imbalance_ratio": float(counts.iloc[0] / counts.iloc[-1]),
        "gini_coefficient": round(gini, 4),
        "top1_share_pct": round(float(share.iloc[0]) * 100, 2),
        "top3_share_pct": round(float(cumulative.iloc[min(2, n - 1)]) * 100, 2),
        "classes_for_80pct": int((cumulative < 0.80).sum() + 1),
    }
