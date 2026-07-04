#!/usr/bin/env python3
"""Scan dataset images and report true EXIF/GPS tag coverage."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import ExifTags, Image, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".mpo"}
EXIF_TAGS = ExifTags.TAGS
GPS_TAGS = ExifTags.GPSTAGS
# <repo>/Scripts/MTSD-Scripts/MTSD-Analysis/check_gps_tags.py
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parents[2]
DATASET_ROOT = BASE_DIR / "Datasets" / "MTSD"
CSV_DIR = BASE_DIR / "Documents" / "MTSD-EDA" / "GeneratedCSVs"


def discover_groups(root: Path) -> list[Path]:
    """Return dataset group folders under the root."""
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name.upper().startswith("GRP-"))


def iter_group_images(group_dir: Path):
    """Yield supported image files inside a group's Images folder."""
    image_dir = group_dir / "Images"
    if not image_dir.is_dir():
        image_dir = group_dir / "merged_images"
    if not image_dir.is_dir():
        image_dir = group_dir
    return sorted(
        path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def to_decimal_coordinate(values, ref):
    """Convert EXIF GPS rationals to decimal degrees."""
    if not values or len(values) != 3:
        return None

    def _to_float(part):
        if isinstance(part, tuple):
            numerator, denominator = part
            return float(numerator) / float(denominator) if denominator else 0.0
        if hasattr(part, "numerator") and hasattr(part, "denominator"):
            return float(part.numerator) / float(part.denominator) if part.denominator else 0.0
        return float(part)

    degrees = _to_float(values[0])
    minutes = _to_float(values[1])
    seconds = _to_float(values[2])
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in {"S", "W"}:
        decimal *= -1
    return decimal


def decode_exif(exif):
    """Decode raw EXIF tag IDs into readable names."""
    if not exif:
        return {}
    decoded = {}
    for tag_id, value in exif.items():
        tag_name = EXIF_TAGS.get(tag_id, tag_id)
        if tag_name == "GPSInfo" and isinstance(value, dict):
            decoded[tag_name] = {GPS_TAGS.get(k, k): v for k, v in value.items()}
        else:
            decoded[tag_name] = value
    return decoded


def extract_exif_gps(image_obj):
    """Return booleans for EXIF presence and usable GPS coordinates."""
    raw_exif = image_obj.getexif()
    exif = decode_exif(raw_exif)
    gps = {}

    if hasattr(raw_exif, "get_ifd"):
        try:
            gps = {GPS_TAGS.get(k, k): v for k, v in raw_exif.get_ifd(ExifTags.IFD.GPSInfo).items()}
        except Exception:
            gps = {}
    elif isinstance(exif.get("GPSInfo"), dict):
        gps = exif.get("GPSInfo", {})

    latitude = to_decimal_coordinate(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef")) if gps else None
    longitude = to_decimal_coordinate(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef")) if gps else None
    has_gps = latitude is not None and longitude is not None

    return {
        "has_exif": bool(exif),
        "has_gps": has_gps,
        "latitude": latitude,
        "longitude": longitude,
    }


def scan_dataset(root: Path) -> list[dict]:
    """Scan dataset and return per-image GPS/EXIF results."""
    rows: list[dict] = []
    for group_dir in discover_groups(root):
        for image_path in iter_group_images(group_dir):
            row = {
                "group": group_dir.name,
                "filename": image_path.name,
                "path": str(image_path),
                "is_corrupted": False,
                "has_exif": False,
                "has_gps": False,
                "latitude": None,
                "longitude": None,
                "error": "",
            }
            try:
                with Image.open(image_path) as image:
                    metadata = extract_exif_gps(image)
                row.update(metadata)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                row["is_corrupted"] = True
                row["error"] = str(exc)
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict[str, int | float]:
    """Build summary counts from per-image rows."""
    total = len(rows)
    corrupted = sum(1 for row in rows if row["is_corrupted"])
    valid = total - corrupted
    has_exif = sum(1 for row in rows if not row["is_corrupted"] and row["has_exif"])
    has_gps = sum(1 for row in rows if not row["is_corrupted"] and row["has_gps"])
    no_gps = sum(1 for row in rows if not row["is_corrupted"] and not row["has_gps"])
    exif_but_no_gps = sum(
        1 for row in rows if not row["is_corrupted"] and row["has_exif"] and not row["has_gps"]
    )
    no_exif = sum(1 for row in rows if not row["is_corrupted"] and not row["has_exif"])

    return {
        "total_images": total,
        "valid_images": valid,
        "corrupted_images": corrupted,
        "images_with_exif": has_exif,
        "images_with_gps": has_gps,
        "images_with_no_gps": no_gps,
        "images_with_exif_but_no_gps": exif_but_no_gps,
        "images_with_no_exif": no_exif,
        "gps_pct_of_valid": round((has_gps / valid) * 100, 2) if valid else 0.0,
        "exif_pct_of_valid": round((has_exif / valid) * 100, 2) if valid else 0.0,
    }


def write_csv(rows: list[dict], output_csv: Path) -> None:
    """Write per-image scan results to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "filename",
        "path",
        "is_corrupted",
        "has_exif",
        "has_gps",
        "latitude",
        "longitude",
        "error",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATASET_ROOT, help="Datasets root containing GRP-* folders.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=CSV_DIR / "gps_tag_audit.csv",
        help="Optional per-image audit CSV output.",
    )
    return parser.parse_args()


def main() -> None:
    """Scan the dataset and print a concise GPS audit."""
    args = parse_args()
    rows = scan_dataset(args.root.resolve())
    summary = summarize(rows)
    write_csv(rows, args.output_csv.resolve())

    print("GPS tag audit complete")
    print(f"- Total images: {summary['total_images']}")
    print(f"- Valid images: {summary['valid_images']}")
    print(f"- Corrupted images: {summary['corrupted_images']}")
    print(f"- Images with EXIF: {summary['images_with_exif']}")
    print(f"- Images with GPS: {summary['images_with_gps']}")
    print(f"- Images with no GPS: {summary['images_with_no_gps']}")
    print(f"- Images with EXIF but no GPS: {summary['images_with_exif_but_no_gps']}")
    print(f"- Images with no EXIF: {summary['images_with_no_exif']}")
    print(f"- EXIF coverage of valid images: {summary['exif_pct_of_valid']:.2f}%")
    print(f"- GPS coverage of valid images: {summary['gps_pct_of_valid']:.2f}%")
    print(f"- Audit CSV: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
