"""Image discovery and EXIF metadata extraction for the MTSD image collection.

The scan walks every ``Datasets/MTSD/GRP-*`` folder, reads image headers with
Pillow (no full decode), extracts the EXIF/GPS tags that matter for the EDA,
and computes a 64-bit difference hash for near-duplicate detection.  Results
are cached to ``Documents/MTSD-EDA/GeneratedCSVs/image_inventory.csv`` so the
notebook and the mapper can reload them instantly.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile
from PIL.ExifTags import IFD

from . import config

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None

# Allow slightly truncated phone captures to be read instead of raising.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# EXIF tag ids (https://exiv2.org/tags.html)
TAG_MAKE = 0x010F
TAG_MODEL = 0x0110
TAG_SOFTWARE = 0x0131
TAG_DATETIME = 0x0132
TAG_ORIENTATION = 0x0112
TAG_EXPOSURE_TIME = 0x829A
TAG_F_NUMBER = 0x829D
TAG_ISO = 0x8827
TAG_DATETIME_ORIGINAL = 0x9003
TAG_BRIGHTNESS = 0x9203
TAG_EXPOSURE_BIAS = 0x9204
TAG_FLASH = 0x9209
TAG_FOCAL_LENGTH = 0x920A
TAG_FOCAL_35MM = 0xA405
TAG_LENS_MODEL = 0xA434
GPS_LAT_REF, GPS_LAT = 1, 2
GPS_LON_REF, GPS_LON = 3, 4
GPS_ALT_REF, GPS_ALT = 5, 6
GPS_SPEED_REF, GPS_SPEED = 12, 13
GPS_IMG_DIRECTION = 17

INVENTORY_COLUMNS = [
    "group", "filename", "relative_path", "extension", "format",
    "width", "height", "megapixels", "aspect_ratio", "orientation",
    "file_size_bytes", "file_size_mb",
    "camera_make", "camera_model", "lens_model", "software",
    "focal_length_mm", "focal_length_35mm", "f_number",
    "exposure_time_s", "iso", "exposure_bias_ev", "brightness_ev", "flash_fired",
    "datetime_original",
    "has_exif", "has_gps",
    "gps_latitude", "gps_longitude", "gps_altitude_m",
    "gps_img_direction", "gps_speed_kmh",
    "dhash",
    "is_corrupted", "corruption_error",
]


def discover_images(datasets_root: Path = config.DATASETS_ROOT) -> list[tuple[str, Path]]:
    """Return ``(group, path)`` pairs for every capture in the collection."""
    records: list[tuple[str, Path]] = []
    for group_dir in config.discover_group_dirs(datasets_root):
        for path in sorted(group_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in config.IMAGE_EXTENSIONS:
                continue
            if any(part in config.EXCLUDED_DIR_NAMES for part in path.relative_to(group_dir).parts[:-1]):
                continue
            records.append((group_dir.name, path))
    return records


def _to_float(value) -> float | None:
    """Convert EXIF rationals/ints to float, tolerating invalid entries."""
    try:
        result = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return result if result == result else None  # filter NaN


def _dms_to_decimal(dms, ref: str | None) -> float | None:
    """Convert an EXIF degrees/minutes/seconds tuple to signed decimal degrees."""
    try:
        degrees, minutes, seconds = (float(part) for part in dms)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref and str(ref).strip().upper() in {"S", "W"}:
        decimal = -decimal
    return decimal


def _parse_exif_datetime(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value).strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def exposure_fraction(seconds: float | None) -> str | None:
    """Render an exposure time as the conventional 1/N notation."""
    if seconds is None or seconds <= 0:
        return None
    if seconds >= 1:
        return f"{seconds:g} s"
    return f"1/{Fraction(seconds).limit_denominator(8000).denominator}"


def extract_image_record(group: str, path: Path, compute_hash: bool = True) -> dict:
    """Read one image header and return a flat metadata record."""
    record: dict = {column: None for column in INVENTORY_COLUMNS}
    record.update(
        group=group,
        filename=path.name,
        relative_path=path.relative_to(config.BASE_DIR).as_posix(),
        extension=path.suffix.lower(),
        file_size_bytes=path.stat().st_size,
        is_corrupted=False,
        has_exif=False,
        has_gps=False,
        flash_fired=None,
    )
    record["file_size_mb"] = round(record["file_size_bytes"] / 1_048_576, 4)

    try:
        with Image.open(path) as img:
            record["format"] = img.format
            width, height = img.size

            exif = img.getexif()
            if len(exif) > 0:
                record["has_exif"] = True
                record["camera_make"] = _clean_str(exif.get(TAG_MAKE))
                record["camera_model"] = _clean_str(exif.get(TAG_MODEL))
                record["software"] = _clean_str(exif.get(TAG_SOFTWARE))

                # Stored orientation: 5-8 mean the pixels are rotated 90 deg.
                orientation_tag = exif.get(TAG_ORIENTATION)
                if orientation_tag in (5, 6, 7, 8):
                    width, height = height, width

                ifd = exif.get_ifd(IFD.Exif)
                record["exposure_time_s"] = _to_float(ifd.get(TAG_EXPOSURE_TIME))
                record["f_number"] = _to_float(ifd.get(TAG_F_NUMBER))
                record["iso"] = _to_float(ifd.get(TAG_ISO))
                record["focal_length_mm"] = _to_float(ifd.get(TAG_FOCAL_LENGTH))
                record["focal_length_35mm"] = _to_float(ifd.get(TAG_FOCAL_35MM))
                record["exposure_bias_ev"] = _to_float(ifd.get(TAG_EXPOSURE_BIAS))
                record["brightness_ev"] = _to_float(ifd.get(TAG_BRIGHTNESS))
                record["lens_model"] = _clean_str(ifd.get(TAG_LENS_MODEL))
                flash = ifd.get(TAG_FLASH)
                if flash is not None:
                    record["flash_fired"] = bool(int(flash) & 0x1)

                taken = _parse_exif_datetime(
                    ifd.get(TAG_DATETIME_ORIGINAL) or exif.get(TAG_DATETIME)
                )
                record["datetime_original"] = taken

                gps = exif.get_ifd(IFD.GPSInfo)
                if gps:
                    latitude = _dms_to_decimal(gps.get(GPS_LAT), gps.get(GPS_LAT_REF))
                    longitude = _dms_to_decimal(gps.get(GPS_LON), gps.get(GPS_LON_REF))
                    if latitude is not None and longitude is not None and (latitude, longitude) != (0.0, 0.0):
                        record["has_gps"] = True
                        record["gps_latitude"] = latitude
                        record["gps_longitude"] = longitude
                        altitude = _to_float(gps.get(GPS_ALT))
                        if altitude is not None:
                            ref = gps.get(GPS_ALT_REF, b"\x00")
                            below = (ref[0] if isinstance(ref, bytes) and ref else ref) == 1
                            record["gps_altitude_m"] = -altitude if below else altitude
                        record["gps_img_direction"] = _to_float(gps.get(GPS_IMG_DIRECTION))
                        speed = _to_float(gps.get(GPS_SPEED))
                        if speed is not None:
                            unit = str(gps.get(GPS_SPEED_REF, "K")).upper()
                            factor = {"K": 1.0, "M": 1.609344, "N": 1.852}.get(unit, 1.0)
                            record["gps_speed_kmh"] = speed * factor

            record["width"] = width
            record["height"] = height
            record["megapixels"] = round(width * height / 1_000_000, 3)
            record["aspect_ratio"] = round(width / height, 4) if height else None
            if width and height:
                record["orientation"] = (
                    "square" if width == height else ("landscape" if width > height else "portrait")
                )

            if compute_hash:
                record["dhash"] = _dhash(img)
    except Exception as error:  # noqa: BLE001 - any unreadable file is recorded, not fatal
        record["is_corrupted"] = True
        record["corruption_error"] = f"{type(error).__name__}: {error}"

    return record


def _clean_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\x00", "").strip()
    return text or None


def _dhash(img: Image.Image, hash_size: int = 8) -> str:
    """64-bit difference hash on a fast DCT-downscaled decode."""
    img.draft("L", (hash_size * 8, hash_size * 8))
    small = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(small.getdata())
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            bits = (bits << 1) | (left > right)
    return f"{bits:016x}"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Bit distance between two hex-encoded dhash values."""
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


def build_inventory(
    force: bool = False,
    compute_hash: bool = True,
    workers: int = 12,
    cache_csv: Path = config.INVENTORY_CSV,
) -> pd.DataFrame:
    """Scan the dataset (or reuse the cache) and return the image inventory."""
    if cache_csv.exists() and not force:
        return load_inventory(cache_csv)

    targets = discover_images()
    iterator = (
        tqdm(targets, desc="Scanning images", unit="img") if tqdm is not None else targets
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(extract_image_record, group, path, compute_hash) for group, path in targets]
        records = []
        for future, _ in zip(futures, iterator):
            records.append(future.result())

    inventory = pd.DataFrame.from_records(records, columns=INVENTORY_COLUMNS)
    inventory = inventory.sort_values(["group", "filename"], key=_natural_key).reset_index(drop=True)

    config.ensure_output_dirs()
    inventory.to_csv(cache_csv, index=False)
    return inventory


def load_inventory(cache_csv: Path = config.INVENTORY_CSV) -> pd.DataFrame:
    """Load a cached inventory CSV with proper dtypes."""
    inventory = pd.read_csv(
        cache_csv,
        parse_dates=["datetime_original"],
        dtype={"dhash": "string"},
    )
    for column in ("has_exif", "has_gps", "is_corrupted"):
        inventory[column] = inventory[column].astype(bool)
    return inventory


def _natural_key(series: pd.Series) -> pd.Series:
    """Sort GRP-2 before GRP-10 by using the numeric suffix where present."""
    def key(value: str):
        tail = str(value).split("-")[-1]
        return int(tail) if tail.isdigit() else value

    return series.map(key)


def find_near_duplicates(inventory: pd.DataFrame, max_distance: int = 2) -> pd.DataFrame:
    """Return pairs of images whose dhash values differ by <= ``max_distance`` bits.

    Exact matches are found via grouping; near matches use a bucketed
    comparison on the two hash halves, which keeps the pair count tractable.
    """
    hashed = inventory.dropna(subset=["dhash"]).reset_index(drop=True)
    pairs: list[dict] = []
    seen: set[tuple[int, int]] = set()

    buckets: dict[str, list[int]] = {}
    for idx, value in enumerate(hashed["dhash"]):
        # any pair within Hamming distance 2 of a 64-bit hash shares at least
        # one identical 32-bit half, so bucketing halves finds all candidates
        for half in (value[:8], value[8:]):
            buckets.setdefault(half, []).append(idx)

    for bucket in buckets.values():
        for pos, idx_a in enumerate(bucket):
            for idx_b in bucket[pos + 1:]:
                key = (min(idx_a, idx_b), max(idx_a, idx_b))
                if key in seen:
                    continue
                seen.add(key)
                distance = hamming_distance(hashed.at[idx_a, "dhash"], hashed.at[idx_b, "dhash"])
                if distance <= max_distance:
                    row_a, row_b = hashed.iloc[idx_a], hashed.iloc[idx_b]
                    pairs.append(
                        {
                            "file_a": row_a["relative_path"],
                            "file_b": row_b["relative_path"],
                            "group_a": row_a["group"],
                            "group_b": row_b["group"],
                            "hamming_distance": distance,
                            "same_file_size": bool(row_a["file_size_bytes"] == row_b["file_size_bytes"]),
                        }
                    )

    return pd.DataFrame(pairs, columns=[
        "file_a", "file_b", "group_a", "group_b", "hamming_distance", "same_file_size",
    ]).sort_values(["hamming_distance", "file_a"]).reset_index(drop=True)
