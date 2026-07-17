"""Annotation-level analysis of the QA-verified COCO exports.

Every function here is a pure computation: it takes the frames produced by
``coco.load_qa_dataset`` (plus, for the geographic analysis, the image
inventory) and returns DataFrames ready for display and CSV export. Nothing
in this module reads or writes annotation files or images.

Terminology, used consistently in every output:

- *images*      - entries in the QA image lists (one per capture);
- *annotations* - bounding boxes (one per traffic-sign instance);
- *classes*     - traffic-sign categories from the shared label map.

Attribute values are always discovered from the loaded data - the module
never assumes a fixed value set. ``PREFERRED_VALUE_ORDER`` only provides a
display ordering for values that happen to be present; unknown extra values
are appended in frequency order rather than dropped.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import coco, geo

#: Attributes required on every QA-verified box.
CORE_ATTRIBUTES = ("view_angle", "mounting", "condition", "sign_shape")

#: Display ordering for known values (semantic, not alphabetical). Values not
#: listed here are appended by descending frequency; the sets are NOT treated
#: as exhaustive.
PREFERRED_VALUE_ORDER: dict[str, list[str]] = {
    "condition": ["Good", "Weathered", "Heavily Damaged"],
    "view_angle": ["Front", "Side", "Back"],
    "mounting": ["Pole-Mounted", "Wall-Mounted"],
    "size_class": ["small (<32² px)", "medium (32²-96² px)", "large (>96² px)"],
}

#: Heuristic for fallback/unknown attribute values (e.g. ``Damaged-Unknown``).
_FALLBACK_PATTERN = re.compile(r"unknown|n/?a\b|unclassified|illegible", re.IGNORECASE)


def ordered_values(annotations: pd.DataFrame, attribute: str) -> list:
    """Values of ``attribute`` present in the data, in display order.

    Preferred (semantic) ordering first for the values that exist, then any
    remaining discovered values by descending frequency.
    """
    counts = annotations[attribute].value_counts()
    preferred = [v for v in PREFERRED_VALUE_ORDER.get(attribute, []) if v in counts.index]
    rest = [v for v in counts.index if v not in preferred]
    return preferred + rest


# ---------------------------------------------------------------------------
# 1. Overall annotation summary
# ---------------------------------------------------------------------------

def per_image_counts(images: pd.DataFrame, annotations: pd.DataFrame) -> pd.Series:
    """Bounding boxes per QA image, **including zero** for unannotated images."""
    counts = annotations.groupby("image_uid").size()
    return counts.reindex(images["image_uid"], fill_value=0).rename("annotations")


def sign_count_buckets(images: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """Images bucketed by how many signs they contain (0, 1, 2, 3, >3).

    Unit: images. Percentages use all QA images as the denominator.
    """
    counts = per_image_counts(images, annotations)
    buckets = pd.cut(
        counts,
        bins=[-0.5, 0.5, 1.5, 2.5, 3.5, np.inf],
        labels=["0 signs", "1 sign", "2 signs", "3 signs", ">3 signs"],
    )
    table = buckets.value_counts().reindex(
        ["0 signs", "1 sign", "2 signs", "3 signs", ">3 signs"]
    )
    return pd.DataFrame(
        {
            "images": table.astype(int),
            "pct_of_all_images": (table / len(counts) * 100).round(2),
        }
    ).rename_axis("signs_in_image")


def _distribution_stats(series: pd.Series) -> dict[str, float]:
    return {
        "mean": round(float(series.mean()), 3),
        "median": float(series.median()),
        "std": round(float(series.std()), 3),
        "min": int(series.min()),
        "q25": float(series.quantile(0.25)),
        "q75": float(series.quantile(0.75)),
        "max": int(series.max()),
    }


def overview_table(
    qa_files: list,
    images: pd.DataFrame,
    annotations: pd.DataFrame,
    categories: pd.DataFrame,
) -> pd.DataFrame:
    """Headline annotation summary as a tidy Metric/Value/Unit table."""
    counts_all = per_image_counts(images, annotations)
    annotated = int((counts_all > 0).sum())
    unannotated = int((counts_all == 0).sum())
    observed_classes = int(annotations["category_name"].nunique())

    rows: list[tuple[str, str, str]] = [
        ("QA annotation files processed", f"{len(qa_files)}", "files"),
        ("MTSD groups represented", f"{images['source_group'].nunique()}", "groups"),
        ("Images listed in QA exports", f"{len(images):,}", "images"),
        ("Annotated images", f"{annotated:,} ({annotated / len(images):.1%})", "images"),
        ("Images without annotations", f"{unannotated:,} ({unannotated / len(images):.1%})", "images"),
        ("Total annotations (bounding boxes)", f"{len(annotations):,}", "annotations"),
        ("Traffic-sign classes in label map", f"{len(categories)}", "classes"),
        ("Classes observed in annotations", f"{observed_classes}", "classes"),
    ]
    for label, series in (
        ("all QA images", counts_all),
        ("annotated images only", counts_all[counts_all > 0]),
    ):
        stats = _distribution_stats(series)
        rows.append(
            (
                f"Annotations per image ({label})",
                f"mean {stats['mean']}, median {stats['median']:g}, std {stats['std']}, "
                f"min {stats['min']}, q25 {stats['q25']:g}, q75 {stats['q75']:g}, "
                f"max {stats['max']}",
                "annotations/image",
            )
        )
    for bucket, row in sign_count_buckets(images, annotations).iterrows():
        rows.append(
            (
                f"Images with {bucket}",
                f"{int(row['images']):,} ({row['pct_of_all_images']:.2f}%)",
                "images",
            )
        )
    return pd.DataFrame(rows, columns=["Metric", "Value", "Unit"])


# ---------------------------------------------------------------------------
# 2. Traffic-sign class distribution
# ---------------------------------------------------------------------------

def class_distribution(
    annotations: pd.DataFrame, categories: pd.DataFrame
) -> pd.DataFrame:
    """Complete per-class table (includes label-map classes with 0 boxes).

    ``pct_of_annotations`` uses all annotations as denominator;
    ``pct_of_annotated_images`` uses images with at least one annotation.
    """
    ann_counts = annotations.groupby("category_id").size()
    img_counts = annotations.groupby("category_id")["image_uid"].nunique()
    annotated_images = annotations["image_uid"].nunique()

    table = categories.copy()
    table["annotations"] = table["category_id"].map(ann_counts).fillna(0).astype(int)
    table["pct_of_annotations"] = (
        table["annotations"] / len(annotations) * 100
    ).round(2)
    table["images_with_class"] = (
        table["category_id"].map(img_counts).fillna(0).astype(int)
    )
    table["pct_of_annotated_images"] = (
        table["images_with_class"] / annotated_images * 100
    ).round(2)
    table = table.sort_values("annotations", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", table.index + 1)
    return table


# ---------------------------------------------------------------------------
# 3. Attribute distributions
# ---------------------------------------------------------------------------

def _absent_mask(annotations: pd.DataFrame, attribute: str) -> pd.Series:
    """Rows whose JSON ``attributes`` dict did not contain ``attribute`` at all."""
    if "absent_attribute_keys" not in annotations.columns:
        return pd.Series(False, index=annotations.index)
    return annotations["absent_attribute_keys"].fillna("").str.split(",").apply(
        lambda keys: attribute in keys
    )


def attribute_distribution(
    annotations: pd.DataFrame, attribute: str
) -> pd.DataFrame:
    """Per-value table for one attribute, missing/null reported as rows.

    ``value_kind`` distinguishes regular values, fallback values (matched by
    a name heuristic, e.g. ``Damaged-Unknown``), keys present with a null
    value, and keys absent from the JSON entirely. Percentages use all
    annotations as the denominator so the rows sum to 100%.
    """
    annotated_images = annotations["image_uid"].nunique()
    total = len(annotations)
    rows = []
    for value in ordered_values(annotations, attribute):
        subset = annotations[annotations[attribute] == value]
        rows.append(
            {
                "value": value,
                "value_kind": "fallback" if _FALLBACK_PATTERN.search(str(value)) else "regular",
                "annotations": len(subset),
                "pct_of_annotations": round(len(subset) / total * 100, 2),
                "images": int(subset["image_uid"].nunique()),
                "pct_of_annotated_images": round(
                    subset["image_uid"].nunique() / annotated_images * 100, 2
                ),
                "classes": int(subset["category_name"].nunique()),
            }
        )

    absent = _absent_mask(annotations, attribute)
    null_mask = annotations[attribute].isna() & ~absent
    for label, mask in (("(null value)", null_mask), ("(attribute absent)", absent)):
        subset = annotations[mask]
        rows.append(
            {
                "value": label,
                "value_kind": "null" if label == "(null value)" else "missing",
                "annotations": len(subset),
                "pct_of_annotations": round(len(subset) / total * 100, 2),
                "images": int(subset["image_uid"].nunique()),
                "pct_of_annotated_images": round(
                    subset["image_uid"].nunique() / annotated_images * 100, 2
                )
                if annotated_images
                else 0.0,
                "classes": int(subset["category_name"].nunique()),
            }
        )
    frame = pd.DataFrame(rows)
    frame.insert(0, "attribute", attribute)
    return frame


# ---------------------------------------------------------------------------
# 4./5. Co-occurrence matrices
# ---------------------------------------------------------------------------

def crosstab_matrices(
    annotations: pd.DataFrame, index: str, column: str
) -> dict[str, pd.DataFrame]:
    """Raw count, row-% and column-% matrices for two annotation fields.

    Rows with a null in either field are excluded from the matrices (their
    count is reported by the caller); percentages are computed after that
    exclusion. Class rows are ordered by class frequency; attribute rows and
    columns use the semantic display order.
    """
    valid = annotations.dropna(subset=[index, column])
    counts = pd.crosstab(valid[index], valid[column])

    if index == "category_name":
        row_order = valid["category_name"].value_counts().index.tolist()
    else:
        row_order = ordered_values(valid, index)
    col_order = ordered_values(valid, column)
    counts = counts.reindex(index=row_order, columns=col_order, fill_value=0)

    row_pct = counts.div(counts.sum(axis=1), axis=0).mul(100).round(2)
    col_pct = counts.div(counts.sum(axis=0), axis=1).mul(100).round(2)
    return {
        "counts": counts,
        "row_pct": row_pct,
        "col_pct": col_pct,
        "excluded_nulls": len(annotations) - len(valid),
    }


# ---------------------------------------------------------------------------
# 7. Bounding-box summaries
# ---------------------------------------------------------------------------

def bbox_summary(
    annotations: pd.DataFrame, by: str = "category_name"
) -> pd.DataFrame:
    """Per-group bounding-box geometry summary (relative units where possible)."""
    size_labels = annotations["size_class"].cat.categories.tolist()
    rows = []
    grouped = annotations.dropna(subset=[by]).groupby(by, observed=True)
    for name, sub in grouped:
        sizes = sub["size_class"].value_counts(normalize=True)
        rows.append(
            {
                by: name,
                "annotations": len(sub),
                "median_bbox_w_px": round(float(sub["bbox_w"].median()), 1),
                "median_bbox_h_px": round(float(sub["bbox_h"].median()), 1),
                "median_aspect_w_h": round(float(sub["bbox_aspect"].median()), 3),
                "median_rel_width": round(float(sub["rel_width"].median()), 4),
                "median_rel_height": round(float(sub["rel_height"].median()), 4),
                "median_rel_area_pct": round(float(sub["rel_area"].median()) * 100, 4),
                "mean_rel_area_pct": round(float(sub["rel_area"].mean()) * 100, 4),
                **{
                    f"pct_{label.split(' ')[0]}": round(float(sizes.get(label, 0.0)) * 100, 2)
                    for label in size_labels
                },
            }
        )
    frame = pd.DataFrame(rows).sort_values("annotations", ascending=False)
    if by != "category_name":
        order = ordered_values(annotations, by)
        frame = frame.set_index(by).reindex(order).reset_index()
    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Geographic joins and locality aggregation
# ---------------------------------------------------------------------------

def annotated_image_geo(
    images: pd.DataFrame,
    annotations: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """One row per QA image with its GPS fix (if any) and annotation count.

    The join key is ``(source_group, file_name)`` against the image
    inventory's ``(group, filename)``; ``validate='one_to_one'`` guarantees
    the join can never duplicate records. ``gps_valid`` additionally requires
    the fix to fall inside the Malta bounding box.
    """
    counts = annotations.groupby("image_uid").size().rename("annotation_count")
    merged = images.merge(
        inventory[
            ["group", "filename", "has_gps", "gps_latitude", "gps_longitude"]
        ],
        left_on=["source_group", "file_name"],
        right_on=["group", "filename"],
        how="left",
        validate="one_to_one",
    ).drop(columns=["group", "filename"])
    merged["matched_on_disk"] = merged["has_gps"].notna()
    merged["has_gps"] = merged["has_gps"].fillna(False).astype(bool)
    merged = merged.join(counts, on="image_uid")
    merged["annotation_count"] = merged["annotation_count"].fillna(0).astype(int)
    merged["gps_valid"] = (
        merged["has_gps"]
        & merged["gps_latitude"].notna()
        & merged["gps_longitude"].notna()
        & geo.within_malta(merged)
    )
    return merged


def gps_coverage_table(image_geo: pd.DataFrame) -> pd.DataFrame:
    """How much of the annotated corpus the geographic analysis can cover."""
    annotated = image_geo[image_geo["annotation_count"] > 0]
    with_gps = annotated[annotated["gps_valid"]]
    ann_total = int(image_geo["annotation_count"].sum())
    ann_with_gps = int(with_gps["annotation_count"].sum())
    outside = int((annotated["has_gps"] & ~annotated["gps_valid"]).sum())

    rows = [
        ("Annotated images", f"{len(annotated):,}", "images"),
        (
            "Annotated images with valid GPS",
            f"{len(with_gps):,} ({len(with_gps) / len(annotated):.1%})",
            "images",
        ),
        (
            "Annotated images without usable GPS",
            f"{len(annotated) - len(with_gps):,} "
            f"({(len(annotated) - len(with_gps)) / len(annotated):.1%})",
            "images",
        ),
        (
            "  of which GPS present but outside Malta bounds",
            f"{outside:,}",
            "images",
        ),
        ("Annotations in geographic analysis", f"{ann_with_gps:,} ({ann_with_gps / ann_total:.1%})", "annotations"),
        (
            "Annotations excluded (no valid GPS)",
            f"{ann_total - ann_with_gps:,} ({(ann_total - ann_with_gps) / ann_total:.1%})",
            "annotations",
        ),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value", "Unit"])


def annotation_geo(
    annotations: pd.DataFrame, image_geo: pd.DataFrame
) -> pd.DataFrame:
    """Annotations joined to their image's GPS fix and nearest locality.

    Only images with a valid in-Malta fix are included; locality assignment
    is nearest-centroid (see ``geo.assign_localities``), not point-in-polygon.
    """
    located = image_geo[image_geo["gps_valid"]].copy()
    located = geo.assign_localities(located)
    keys = [
        "image_uid", "gps_latitude", "gps_longitude",
        "locality", "district", "locality_distance_km",
    ]
    return annotations.merge(located[keys], on="image_uid", how="inner")


def locality_annotation_summary(
    geo_annotations: pd.DataFrame,
    image_geo: pd.DataFrame,
    min_annotations: int = 30,
) -> pd.DataFrame:
    """Ranked per-locality summary of annotation volume and sign condition.

    ``geotagged_images`` counts every image with a valid fix assigned to the
    locality (annotated or not); the remaining columns count annotations.
    ``low_sample`` flags localities below ``min_annotations`` - their
    percentage columns are reported but should not be ranked on.
    """
    located_images = geo.assign_localities(image_geo[image_geo["gps_valid"]].copy())
    images_per_locality = located_images.groupby("locality").size()

    condition_values = ordered_values(geo_annotations, "condition")
    rows = []
    for locality, sub in geo_annotations.groupby("locality"):
        row = {
            "locality": locality,
            "district": sub["district"].iloc[0],
            "geotagged_images": int(images_per_locality.get(locality, 0)),
            "annotated_images": int(sub["image_uid"].nunique()),
            "annotations": len(sub),
            "distinct_classes": int(sub["category_name"].nunique()),
        }
        for value in condition_values:
            n = int((sub["condition"] == value).sum())
            row[f"n_{value}"] = n
            row[f"pct_{value}"] = round(n / len(sub) * 100, 2)
        row["low_sample"] = len(sub) < min_annotations
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("annotations", ascending=False)
        .reset_index(drop=True)
    )
