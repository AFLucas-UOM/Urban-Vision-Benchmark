"""Integrity checks over a mapped MDWD variant.

Consumes the ``DatasetMap`` produced by :mod:`mdwd_eda.mapper` and condenses
its issue list into a per-type summary table plus a machine-readable dict.
"""

from __future__ import annotations

import pandas as pd

from .mapper import DatasetMap

# Every issue type the mapper can emit, with a short human explanation.
ISSUE_DESCRIPTIONS = {
    "missing_split": "expected split folder/annotation file is absent",
    "unreadable_image": "image file exists but PIL cannot read its header",
    "missing_image": "annotation references an image file absent from disk",
    "missing_label": "image has no matching YOLO label file",
    "orphan_label": "label file has no matching image",
    "orphan_annotation": "COCO annotation references an unknown image id",
    "empty_annotation": "image has an annotation entry but zero boxes",
    "malformed_label_row": "label row does not parse as 'class cx cy w h'",
    "invalid_class_id": "class id outside the declared vocabulary",
    "out_of_range_box": "box coordinates fall outside the image bounds",
    "duplicate_filename": "same file name appears more than once",
    "source_in_multiple_splits": "augmented copies of one source image appear in more than one split (leakage)",
}


def integrity_summary(dataset_map: DatasetMap) -> pd.DataFrame:
    """One row per issue type: count + description (zero rows included)."""
    counts = dataset_map.issue_counts
    rows = [
        {
            "issue": issue,
            "count": counts.get(issue, 0),
            "description": description,
        }
        for issue, description in ISSUE_DESCRIPTIONS.items()
    ]
    return pd.DataFrame(rows)


def issues_table(dataset_map: DatasetMap) -> pd.DataFrame:
    """Full issue list as a DataFrame (empty frame when the dataset is clean)."""
    if not dataset_map.issues:
        return pd.DataFrame(columns=["type", "split", "file", "detail"])
    return pd.DataFrame(dataset_map.issues)


def is_clean(dataset_map: DatasetMap, ignore: tuple[str, ...] = ()) -> bool:
    """True when no issues (outside *ignore*) were recorded."""
    return all(issue["type"] in ignore for issue in dataset_map.issues)
