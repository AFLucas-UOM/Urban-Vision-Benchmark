"""Shared filesystem paths and scan settings for the MDWD analysis tooling."""

from __future__ import annotations

from pathlib import Path

# Repository layout ----------------------------------------------------------
# <repo>/Scripts/MDWD-Scripts/MDWD-Analysis/mdwd_eda/config.py
PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parents[3]

DATASETS_ROOT = BASE_DIR / "Datasets" / "MDWD"

# EDA outputs mirror the MTSD convention (Documents/MTSD-EDA/...).
EDA_DIR = BASE_DIR / "Documents" / "MDWD-EDA"
CSV_DIR = EDA_DIR / "GeneratedCSVs"
FIGURES_DIR = EDA_DIR / "Figures"
SAMPLES_DIR = EDA_DIR / "SampleAnnotationImages"
SUMMARY_JSON = EDA_DIR / "eda_summary.json"

# Dataset variants -----------------------------------------------------------
# All variants are exports of the same Roboflow project (v20); the YOLO ones
# share identical splits/labels, MDWD-RFDETR carries the same data as COCO.
DEFAULT_VARIANT = "MDWD-YOLO26"
SPLITS = ("train", "valid", "test")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# COCO size-class thresholds, in squared pixels of the (640x640) export.
COCO_SMALL_MAX_AREA = 32**2
COCO_MEDIUM_MAX_AREA = 96**2


def discover_variants(datasets_root: Path = DATASETS_ROOT) -> list[Path]:
    """Return MDWD-* dataset variant directories, default variant first."""
    variants = sorted(
        (p for p in datasets_root.glob("MDWD-*") if p.is_dir()),
        key=lambda p: (p.name != DEFAULT_VARIANT, p.name),
    )
    return variants


def variant_dir(name: str = DEFAULT_VARIANT) -> Path:
    """Resolve a variant name (e.g. ``MDWD-YOLO26``) to its directory."""
    path = DATASETS_ROOT / name
    if not path.is_dir():
        known = ", ".join(p.name for p in discover_variants()) or "none found"
        raise FileNotFoundError(
            f"MDWD variant {name!r} not found at {path} (available: {known})"
        )
    return path


def detect_format(variant: Path) -> str:
    """Return ``"yolo"`` or ``"coco"`` based on the files actually present."""
    if (variant / "data.yaml").exists():
        return "yolo"
    if any((variant / split / "_annotations.coco.json").exists() for split in SPLITS):
        return "coco"
    raise ValueError(
        f"Could not detect the annotation format of {variant}: no data.yaml "
        f"and no <split>/_annotations.coco.json found."
    )


def ensure_output_dirs() -> None:
    """Create the EDA output directories if they do not exist yet."""
    for directory in (CSV_DIR, FIGURES_DIR, SAMPLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
