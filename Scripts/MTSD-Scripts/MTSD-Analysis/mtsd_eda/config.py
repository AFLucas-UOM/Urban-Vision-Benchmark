"""Shared filesystem paths and scan settings for the MTSD analysis tooling."""

from __future__ import annotations

from pathlib import Path

# Repository layout ----------------------------------------------------------
# <repo>/Scripts/MTSD-Scripts/MTSD-Analysis/mtsd_eda/config.py
PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parents[3]

DATASETS_ROOT = BASE_DIR / "Datasets" / "MTSD"
ANNOTATIONS_ROOT = DATASETS_ROOT / "Annotations"

EDA_DIR = BASE_DIR / "Documents" / "MTSD-EDA"
CSV_DIR = EDA_DIR / "GeneratedCSVs"
FIGURES_DIR = EDA_DIR / "Figures"

INVENTORY_CSV = CSV_DIR / "image_inventory.csv"

# Image discovery ------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tif", ".tiff"}

# Directory names that never contain dataset captures.
EXCLUDED_DIR_NAMES = {
    "Ignore",
    "Export",
    "labelstudio_output",
    "LabelStudioData",
    "Annotations",
}

# QA-verified COCO exports are the only annotation ground truth.
QA_ANNOTATION_GLOB = "GRP-*/Final-QA/QA-*.json"


def ensure_output_dirs() -> None:
    """Create the EDA output directories if they do not exist yet."""
    for directory in (CSV_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def discover_group_dirs(datasets_root: Path = DATASETS_ROOT) -> list[Path]:
    """Return the dataset group directories (GRP-*) sorted by group number."""

    def sort_key(path: Path) -> tuple[int, str]:
        try:
            return (int(path.name.split("-")[-1]), path.name)
        except ValueError:
            return (10**9, path.name)

    return sorted(
        (p for p in datasets_root.glob("GRP-*") if p.is_dir()),
        key=sort_key,
    )
