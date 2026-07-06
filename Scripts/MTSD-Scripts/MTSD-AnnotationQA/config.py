"""Shared paths and thresholds for the MTSD annotation QA tooling.

Group discovery is dynamic: any QA-style JSON found under the annotations
root is audited, so future groups (GRP-4, GRP-5, ...) are picked up
automatically once their Final-QA export exists. Nothing here hardcodes the
current group list.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

# Repository layout ----------------------------------------------------------
# <repo>/Scripts/MTSD-Scripts/MTSD-AnnotationQA/config.py
PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parents[2]

MTSD_ROOT = BASE_DIR / "Datasets" / "MTSD"
ANNOTATIONS_ROOT = MTSD_ROOT / "Annotations"

OUTPUTS_DIR = PACKAGE_DIR / "outputs"
LATEST_POINTER = OUTPUTS_DIR / "LATEST.txt"

# QA-file discovery -----------------------------------------------------------
# A JSON file is treated as a QA annotation export when any of these hold.
QA_DIR_NAMES = {"final-qa", "qa"}
QA_FILENAME_PREFIXES = ("qa-", "qa_")
# Files that are never audited (backups and editor artefacts).
SKIP_SUFFIX_TOKENS = (".bak", ".pre-migration-", ".pre-qa-fix-", ".tmp")

# Attribute schema ------------------------------------------------------------
# The controlled vocabulary is read from the attribute-classification config
# (single source of truth); this fallback copy is used only if that file is
# missing or unparseable.
ATTRCLS_CONFIG_PATH = (
    BASE_DIR / "Scripts" / "MTSD-Scripts" / "AttributeClassification" / "config" / "default.yaml"
)
FALLBACK_ATTRIBUTE_SCHEMA = {
    "view_angle": ["Front", "Back", "Side"],
    "mounting": ["Pole-Mounted", "Wall-Mounted"],
    "condition": ["Good", "Weathered", "Heavily Damaged"],
    "sign_shape": ["Circular", "Quadrangle", "Triangular", "Octagonal", "Pentagon"],
}
# Attribute values that are known/tolerated but flagged for review (they are
# dropped by the training pipeline rather than being part of the vocabulary).
KNOWN_DROP_VALUES = {"sign_shape": ["Damaged-Unknown"]}

# Metadata keys inside `attributes` that are provenance, not label attributes.
NON_LABEL_ATTRIBUTE_KEYS = {
    "label_studio_region_id",
    "label_studio_annotation_id",
    "rotation",
    "origin",
}

# Duplicate detection ---------------------------------------------------------
DUPLICATE_IOU = 0.95     # >= this IoU: duplicate candidate
OVERLAP_IOU = 0.75       # >= this IoU (but < DUPLICATE_IOU): high-overlap warning

# Path conventions --------------------------------------------------------------
# source_image values are expected to be repo-relative and to start with this
# prefix (forward or back slashes both accepted).
EXPECTED_SOURCE_PREFIX = "Datasets/MTSD/"

# Fuzzy-suggestion threshold for misspelt attribute values (difflib ratio).
SUGGESTION_CUTOFF = 0.8


def new_audit_dir() -> Path:
    """Create and return a fresh timestamped audit output folder."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    audit_dir = OUTPUTS_DIR / f"audit-{stamp}"
    audit_dir.mkdir(parents=True, exist_ok=True)
    return audit_dir


def latest_audit_dir() -> Path | None:
    """Return the most recent audit folder (via pointer file, else by name)."""
    if LATEST_POINTER.exists():
        candidate = Path(LATEST_POINTER.read_text(encoding="utf-8").strip())
        if candidate.is_dir():
            return candidate
    audits = sorted(OUTPUTS_DIR.glob("audit-*"))
    return audits[-1] if audits else None
