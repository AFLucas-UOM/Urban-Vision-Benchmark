"""Attribute schema loading and per-annotation validation for MTSD QA files."""

from __future__ import annotations

import difflib
from typing import Any

import config


def load_attribute_schema() -> dict[str, list[str]]:
    """Required attributes and their controlled vocabularies.

    Read from the attribute-classification config when available so the audit
    and the training pipeline can never disagree; falls back to a static copy.
    """
    try:
        import yaml

        cfg = yaml.safe_load(config.ATTRCLS_CONFIG_PATH.read_text(encoding="utf-8"))
        schema = {
            name: list(spec["classes"])
            for name, spec in cfg["attributes"].items()
            if isinstance(spec, dict) and spec.get("classes")
        }
        if schema:
            return schema
    except Exception:
        pass
    return dict(config.FALLBACK_ATTRIBUTE_SCHEMA)


def is_missing_value(value: Any) -> bool:
    """True for null / empty / whitespace / textual-null attribute values."""
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"none", "null", "nan", "n/a", "-"}


def classify_attribute_value(
    attribute: str, value: Any, vocabulary: list[str]
) -> tuple[str, str | None]:
    """Classify one attribute value.

    Returns (status, suggestion):
        status in {"ok", "missing", "known_drop_value", "case_mismatch",
                   "close_misspelling", "unknown_value"}
        suggestion is the canonical value to fix to, when one can be inferred.
    """
    if is_missing_value(value):
        return "missing", None
    text = str(value).strip()
    if text in vocabulary:
        return "ok", None
    if text in config.KNOWN_DROP_VALUES.get(attribute, []):
        return "known_drop_value", None

    lower_map = {v.lower(): v for v in vocabulary}
    if text.lower() in lower_map:
        return "case_mismatch", lower_map[text.lower()]

    close = difflib.get_close_matches(text, vocabulary, n=1, cutoff=config.SUGGESTION_CUTOFF)
    if close:
        return "close_misspelling", close[0]
    return "unknown_value", None


def bbox_problems(
    bbox: Any, image_width: float | None, image_height: float | None
) -> list[str]:
    """Structural validity checks for a COCO [x, y, w, h] bbox."""
    problems = []
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return ["bbox_not_4_numbers"]
    try:
        x, y, w, h = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return ["bbox_not_numeric"]
    if w <= 0 or h <= 0:
        problems.append("bbox_non_positive_size")
    if x < -1 or y < -1:
        problems.append("bbox_negative_origin")
    if image_width and image_height:
        if x + w > image_width + 1 or y + h > image_height + 1:
            problems.append("bbox_outside_image")
    return problems
