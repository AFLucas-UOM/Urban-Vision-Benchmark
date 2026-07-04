#!/usr/bin/env python3

import argparse
import csv
import json
import math
import statistics
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_IMAGES_DIR = BASE_DIR / "Datasets"
DEFAULT_ANNOTATIONS_DIR = DEFAULT_IMAGES_DIR / "Annotations"
DEFAULT_CSV_OUTPUT = BASE_DIR / "Documents" / "MTSD-EDA" / "GeneratedCSVs" / "image_annotation_stats.csv"
DEFAULT_HISTOGRAM_CSV_OUTPUT = BASE_DIR / "Documents" / "MTSD-EDA" / "GeneratedCSVs" / "image_annotation_histogram.csv"

GROUP_PATTERN = "GRP-*"
IMAGE_DIR_NAME = "Images"
ANNOTATION_FILENAMES = ("merged_inputs.json", "merged_input.json")
IMAGE_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count images in Datasets/GRP-*/Images, then summarise available "
            "annotations from Datasets/Annotations/GRP-* XML or Label Studio JSON per group."
        )
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="Path to the local Datasets folder.",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=DEFAULT_ANNOTATIONS_DIR,
        help="Path to the Datasets/Annotations folder containing GRP-* XML annotations.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
        help="Where to write the CSV summary.",
    )
    parser.add_argument(
        "--histogram-csv-output",
        type=Path,
        default=DEFAULT_HISTOGRAM_CSV_OUTPUT,
        help="Where to write the per-group annotation-count histogram CSV.",
    )
    parser.add_argument(
        "--seconds-per-image",
        type=float,
        default=20.0,
        help="Estimated fixed handling/review seconds per image for workload estimates.",
    )
    parser.add_argument(
        "--seconds-per-rectangle",
        type=float,
        default=40.0,
        help="Estimated seconds to draw and classify one rectangle annotation.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Print only; do not write a CSV file.",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def normalize_group_name(group: str) -> str:
    group_name = str(group).strip()
    if group_name.upper().startswith("GRP-"):
        return f"GRP-{group_name.split('-', 1)[1]}"
    return f"GRP-{group_name}"


def find_annotation_file(group_dir: Path) -> Optional[Path]:
    candidates = []
    for filename in ANNOTATION_FILENAMES:
        candidates.extend(
            [
                group_dir / filename,
                group_dir / "Images" / "Ignore" / filename,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def image_files_for_group(group_dir: Path) -> List[Path]:
    image_dir = group_dir / IMAGE_DIR_NAME
    if not image_dir.exists():
        image_dir = group_dir / "merged_images"
    search_dir = image_dir if image_dir.exists() else group_dir
    return sorted(path for path in search_dir.rglob("*") if is_image_file(path))


def image_name_from_task(task: Dict[str, Any]) -> str:
    image_value = str((task.get("data") or {}).get("image") or "")
    return Path(image_value).name


def count_rectangle_annotations(task: Dict[str, Any]) -> int:
    region_ids = set()
    rectangles_without_id = 0

    for annotation in task.get("annotations") or []:
        for result in annotation.get("result") or []:
            if result.get("type") != "rectanglelabels":
                continue
            value = result.get("value") or {}
            if not value.get("rectanglelabels"):
                continue

            region_id = result.get("id")
            if region_id:
                region_ids.add(region_id)
            else:
                rectangles_without_id += 1

    return len(region_ids) + rectangles_without_id


def count_result_items(task: Dict[str, Any]) -> int:
    return sum(len(annotation.get("result") or []) for annotation in task.get("annotations") or [])


def percentile(values: List[int], percentile_value: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (percentile_value / 100.0)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = rank - lower_index
    return lower_value + ((upper_value - lower_value) * weight)


def build_count_stats(counts: List[int], prefix: str) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        f"{prefix}_mean": 0.0,
        f"{prefix}_median": 0.0,
        f"{prefix}_min": 0,
        f"{prefix}_max": 0,
        f"{prefix}_std_dev": 0.0,
        f"{prefix}_p25": 0.0,
        f"{prefix}_p50": 0.0,
        f"{prefix}_p75": 0.0,
        f"{prefix}_p90": 0.0,
        f"{prefix}_p95": 0.0,
    }
    if not counts:
        return stats

    stats[f"{prefix}_mean"] = statistics.mean(counts)
    stats[f"{prefix}_median"] = statistics.median(counts)
    stats[f"{prefix}_min"] = min(counts)
    stats[f"{prefix}_max"] = max(counts)
    stats[f"{prefix}_std_dev"] = statistics.pstdev(counts)
    for percentile_value in (25, 50, 75, 90, 95):
        stats[f"{prefix}_p{percentile_value}"] = percentile(counts, percentile_value)
    return stats


def build_distribution_stats(counts: List[int], prefix: str) -> Dict[str, Any]:
    total = len(counts)
    count_1 = sum(1 for count in counts if count == 1)
    count_2 = sum(1 for count in counts if count == 2)
    count_3 = sum(1 for count in counts if count == 3)
    count_4_plus = sum(1 for count in counts if count >= 4)

    def pct(value: int) -> float:
        return (value / total * 100.0) if total else 0.0

    return {
        f"{prefix}_1_AnnotationImages": count_1,
        f"{prefix}_1_annotation_pct": pct(count_1),
        f"{prefix}_2_AnnotationImages": count_2,
        f"{prefix}_2_annotation_pct": pct(count_2),
        f"{prefix}_3_AnnotationImages": count_3,
        f"{prefix}_3_annotation_pct": pct(count_3),
        f"{prefix}_4_plus_AnnotationImages": count_4_plus,
        f"{prefix}_4_plus_annotation_pct": pct(count_4_plus),
    }


def load_tasks(annotation_path: Path) -> List[Dict[str, Any]]:
    with annotation_path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{annotation_path} does not contain a list of Label Studio tasks")
    return [task for task in payload if isinstance(task, dict)]


def count_xml_objects(xml_path: Path) -> int:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return 0
    return len(root.findall("object"))


def summarise_group(
    group_dir: Path,
    annotations_dir: Path,
    seconds_per_image: float,
    seconds_per_rectangle: float,
) -> Dict[str, Any]:
    image_paths = image_files_for_group(group_dir)
    image_names = {path.name.lower() for path in image_paths}
    rectangle_counts_by_image = {path.name.lower(): 0 for path in image_paths}
    rectangle_counts_by_stem = {path.stem.lower(): path.name.lower() for path in image_paths}
    group_name = normalize_group_name(group_dir.name)
    xml_annotation_dir = annotations_dir / group_name
    annotation_path = find_annotation_file(group_dir)

    summary: Dict[str, Any] = {
        "group": group_name,
        "images": len(image_paths),
        "annotation_file": "",
        "json_tasks": 0,
        "tasks_matching_images": 0,
        "json_tasks_without_matching_image": 0,
        "images_with_rectangles": 0,
        "label_studio_annotation_objects": 0,
        "result_items": 0,
        "rectangle_annotations": 0,
        "zero_AnnotationImages": len(image_paths),
        "avg_rectangles_per_image": 0.0,
        "avg_rectangles_per_annotated_image": 0.0,
        "estimated_seconds": 0.0,
        "estimated_hours": 0.0,
        "estimated_minutes_per_image": 0.0,
        "projected_rectangles_for_all_images": 0.0,
        "projected_total_hours_for_all_images": 0.0,
        "_all_image_counts": [],
        "_annotated_image_counts": [],
    }

    if xml_annotation_dir.exists():
        summary["annotation_file"] = str(xml_annotation_dir.relative_to(BASE_DIR))
        for xml_path in sorted(xml_annotation_dir.glob("*.xml")):
            image_key = rectangle_counts_by_stem.get(xml_path.stem.lower())
            if image_key is None:
                summary["json_tasks_without_matching_image"] += 1
                continue
            rectangle_count = count_xml_objects(xml_path)
            rectangle_counts_by_image[image_key] += rectangle_count
            summary["tasks_matching_images"] += 1
            summary["json_tasks"] += 1
            summary["rectangle_annotations"] += rectangle_count
            if rectangle_count > 0:
                summary["images_with_rectangles"] += 1

        all_counts = list(rectangle_counts_by_image.values())
        annotated_counts = [count for count in all_counts if count > 0]
        summary["zero_AnnotationImages"] = sum(1 for count in all_counts if count == 0)
        summary.update(build_count_stats(all_counts, "all_images_annotations"))
        summary.update(build_count_stats(annotated_counts, "annotated_images_annotations"))
        summary.update(build_distribution_stats(all_counts, "all_images"))
        if summary["images"]:
            summary["avg_rectangles_per_image"] = summary["rectangle_annotations"] / summary["images"]
        if summary["images_with_rectangles"]:
            summary["avg_rectangles_per_annotated_image"] = (
                summary["rectangle_annotations"] / summary["images_with_rectangles"]
            )
        summary["estimated_seconds"] = (summary["images"] * seconds_per_image) + (
            summary["rectangle_annotations"] * seconds_per_rectangle
        )
        summary["estimated_hours"] = summary["estimated_seconds"] / 3600.0
        if summary["images"]:
            summary["estimated_minutes_per_image"] = summary["estimated_seconds"] / 60.0 / summary["images"]
        if summary["tasks_matching_images"]:
            observed_rate = summary["rectangle_annotations"] / summary["tasks_matching_images"]
            summary["projected_rectangles_for_all_images"] = observed_rate * summary["images"]
            projected_seconds = (summary["images"] * seconds_per_image) + (
                summary["projected_rectangles_for_all_images"] * seconds_per_rectangle
            )
            summary["projected_total_hours_for_all_images"] = projected_seconds / 3600.0
        summary["_all_image_counts"] = all_counts
        summary["_annotated_image_counts"] = annotated_counts
        return summary

    summary["annotation_file"] = annotation_path.name if annotation_path else ""
    if annotation_path is None:
        all_counts = list(rectangle_counts_by_image.values())
        summary.update(build_count_stats(all_counts, "all_images_annotations"))
        summary.update(build_count_stats([], "annotated_images_annotations"))
        summary.update(build_distribution_stats(all_counts, "all_images"))
        summary["estimated_seconds"] = (summary["images"] * seconds_per_image)
        summary["estimated_hours"] = summary["estimated_seconds"] / 3600.0
        if summary["images"]:
            summary["estimated_minutes_per_image"] = summary["estimated_seconds"] / 60.0 / summary["images"]
        summary["projected_total_hours_for_all_images"] = summary["estimated_hours"]
        summary["_all_image_counts"] = all_counts
        return summary

    tasks = load_tasks(annotation_path)
    summary["json_tasks"] = len(tasks)

    annotated_image_names = set()
    for task in tasks:
        task_image_name = image_name_from_task(task)
        task_matches_image = bool(task_image_name and task_image_name.lower() in image_names)
        if task_matches_image:
            summary["tasks_matching_images"] += 1
        else:
            summary["json_tasks_without_matching_image"] += 1
            continue

        image_key = task_image_name.lower()
        annotation_objects = task.get("annotations") or []
        rectangle_count = count_rectangle_annotations(task)
        rectangle_counts_by_image[image_key] += rectangle_count

        summary["label_studio_annotation_objects"] += len(annotation_objects)
        summary["result_items"] += count_result_items(task)
        summary["rectangle_annotations"] += rectangle_count

        if rectangle_count > 0 and task_image_name:
            annotated_image_names.add(task_image_name.lower())

    summary["images_with_rectangles"] = len(annotated_image_names)
    if summary["images"]:
        summary["avg_rectangles_per_image"] = summary["rectangle_annotations"] / summary["images"]
    if summary["images_with_rectangles"]:
        summary["avg_rectangles_per_annotated_image"] = (
            summary["rectangle_annotations"] / summary["images_with_rectangles"]
        )

    all_counts = list(rectangle_counts_by_image.values())
    annotated_counts = [count for count in all_counts if count > 0]
    summary["zero_AnnotationImages"] = sum(1 for count in all_counts if count == 0)
    summary.update(build_count_stats(all_counts, "all_images_annotations"))
    summary.update(build_count_stats(annotated_counts, "annotated_images_annotations"))
    summary.update(build_distribution_stats(all_counts, "all_images"))
    summary["estimated_seconds"] = (summary["images"] * seconds_per_image) + (
        summary["rectangle_annotations"] * seconds_per_rectangle
    )
    summary["estimated_hours"] = summary["estimated_seconds"] / 3600.0
    if summary["images"]:
        summary["estimated_minutes_per_image"] = summary["estimated_seconds"] / 60.0 / summary["images"]
    if summary["tasks_matching_images"]:
        observed_rate = summary["rectangle_annotations"] / summary["tasks_matching_images"]
        summary["projected_rectangles_for_all_images"] = observed_rate * summary["images"]
        projected_seconds = (summary["images"] * seconds_per_image) + (
            summary["projected_rectangles_for_all_images"] * seconds_per_rectangle
        )
        summary["projected_total_hours_for_all_images"] = projected_seconds / 3600.0
    summary["_all_image_counts"] = all_counts
    summary["_annotated_image_counts"] = annotated_counts
    return summary


def format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def print_table(rows: Iterable[Dict[str, Any]], columns: List[str]) -> None:
    rows = list(rows)
    widths = {
        column: max(len(column), *(len(format_number(row[column])) for row in rows))
        for column in columns
    }

    header = "  ".join(column.ljust(widths[column]) for column in columns)
    separator = "  ".join("-" * widths[column] for column in columns)
    print(header)
    print(separator)
    for row in rows:
        print("  ".join(format_number(row[column]).ljust(widths[column]) for column in columns))


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def build_histogram_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    histogram_rows = []
    for row in rows:
        counts = row["_all_image_counts"]
        total_images = len(counts)
        for annotation_count, image_count in sorted(Counter(counts).items()):
            histogram_rows.append(
                {
                    "group": row["group"],
                    "annotations_per_image": annotation_count,
                    "image_count": image_count,
                    "image_percentage": (image_count / total_images * 100.0) if total_images else 0.0,
                }
            )
    return histogram_rows


def write_histogram_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    columns = ["group", "annotations_per_image", "image_count", "image_percentage"]
    write_csv(path, build_histogram_rows(rows), columns)


def build_total_row(
    rows: List[Dict[str, Any]],
    seconds_per_image: float,
    seconds_per_rectangle: float,
) -> Dict[str, Any]:
    all_counts = [count for row in rows for count in row["_all_image_counts"]]
    annotated_counts = [count for count in all_counts if count > 0]
    total = {
        "group": "TOTAL",
        "images": sum(row["images"] for row in rows),
        "annotation_file": "",
        "json_tasks": sum(row["json_tasks"] for row in rows),
        "tasks_matching_images": sum(row["tasks_matching_images"] for row in rows),
        "json_tasks_without_matching_image": sum(row["json_tasks_without_matching_image"] for row in rows),
        "images_with_rectangles": sum(row["images_with_rectangles"] for row in rows),
        "label_studio_annotation_objects": sum(row["label_studio_annotation_objects"] for row in rows),
        "result_items": sum(row["result_items"] for row in rows),
        "rectangle_annotations": sum(row["rectangle_annotations"] for row in rows),
        "zero_AnnotationImages": sum(row["zero_AnnotationImages"] for row in rows),
        "avg_rectangles_per_image": 0.0,
        "avg_rectangles_per_annotated_image": 0.0,
        "estimated_seconds": 0.0,
        "estimated_hours": 0.0,
        "estimated_minutes_per_image": 0.0,
        "projected_rectangles_for_all_images": 0.0,
        "projected_total_hours_for_all_images": 0.0,
        "_all_image_counts": all_counts,
        "_annotated_image_counts": annotated_counts,
    }
    if total["images"]:
        total["avg_rectangles_per_image"] = total["rectangle_annotations"] / total["images"]
    if total["images_with_rectangles"]:
        total["avg_rectangles_per_annotated_image"] = (
            total["rectangle_annotations"] / total["images_with_rectangles"]
        )
    total.update(build_count_stats(all_counts, "all_images_annotations"))
    total.update(build_count_stats(annotated_counts, "annotated_images_annotations"))
    total.update(build_distribution_stats(all_counts, "all_images"))
    total["estimated_seconds"] = (total["images"] * seconds_per_image) + (
        total["rectangle_annotations"] * seconds_per_rectangle
    )
    total["estimated_hours"] = total["estimated_seconds"] / 3600.0
    if total["images"]:
        total["estimated_minutes_per_image"] = total["estimated_seconds"] / 60.0 / total["images"]
    if total["tasks_matching_images"]:
        observed_rate = total["rectangle_annotations"] / total["tasks_matching_images"]
        total["projected_rectangles_for_all_images"] = observed_rate * total["images"]
        projected_seconds = (total["images"] * seconds_per_image) + (
            total["projected_rectangles_for_all_images"] * seconds_per_rectangle
        )
        total["projected_total_hours_for_all_images"] = projected_seconds / 3600.0
    return total


def visible_row(row: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
    return {column: row[column] for column in columns}


def print_workload_summary(total: Dict[str, Any], seconds_per_image: float, seconds_per_rectangle: float) -> None:
    print("\nWorkload estimate")
    print("-----------------")
    print(
        "Assumption: "
        f"{seconds_per_image:.0f}s fixed review/handling per image + "
        f"{seconds_per_rectangle:.0f}s per rectangle."
    )
    print(f"Total images: {total['images']}")
    print(f"Total annotated images: {total['images_with_rectangles']}")
    print(f"Known rectangles from existing JSON: {total['rectangle_annotations']}")
    print(f"Known annotated workload: {total['estimated_hours']:.2f} hours")
    print(
        "Projected full-dataset rectangles from observed rate: "
        f"{total['projected_rectangles_for_all_images']:.0f}"
    )
    print(
        "Projected full-dataset workload: "
        f"{total['projected_total_hours_for_all_images']:.2f} hours"
    )
    print(
        "Estimated known-workload average: "
        f"{total['estimated_minutes_per_image']:.2f} minutes per image"
    )


def print_histogram_summary(rows: List[Dict[str, Any]]) -> None:
    histogram_rows = build_histogram_rows(rows)
    if not histogram_rows:
        return

    print("\nHistogram: annotations per image")
    print("--------------------------------")
    print_table(histogram_rows, ["group", "annotations_per_image", "image_count", "image_percentage"])


def main() -> None:
    args = parse_args()
    images_dir = args.images_dir.resolve()
    annotations_dir = args.annotations_dir.resolve()
    if not images_dir.exists():
        raise SystemExit(f"Images folder does not exist: {images_dir}")

    group_dirs = [
        path
        for path in sorted(images_dir.glob(GROUP_PATTERN))
        if path.is_dir() and path.name.lower() != "Samples"
    ]
    rows = [
        summarise_group(group_dir, annotations_dir, args.seconds_per_image, args.seconds_per_rectangle)
        for group_dir in group_dirs
    ]
    total_row = build_total_row(rows, args.seconds_per_image, args.seconds_per_rectangle)
    rows_with_total = rows + [total_row]

    columns = [
        "group",
        "images",
        "annotation_file",
        "json_tasks",
        "tasks_matching_images",
        "json_tasks_without_matching_image",
        "images_with_rectangles",
        "zero_AnnotationImages",
        "rectangle_annotations",
        "avg_rectangles_per_image",
        "avg_rectangles_per_annotated_image",
        "all_images_annotations_mean",
        "all_images_annotations_median",
        "all_images_annotations_min",
        "all_images_annotations_max",
        "all_images_annotations_std_dev",
        "all_images_annotations_p25",
        "all_images_annotations_p50",
        "all_images_annotations_p75",
        "all_images_annotations_p90",
        "all_images_annotations_p95",
        "annotated_images_annotations_mean",
        "annotated_images_annotations_median",
        "annotated_images_annotations_min",
        "annotated_images_annotations_max",
        "annotated_images_annotations_std_dev",
        "annotated_images_annotations_p25",
        "annotated_images_annotations_p50",
        "annotated_images_annotations_p75",
        "annotated_images_annotations_p90",
        "annotated_images_annotations_p95",
        "all_images_1_AnnotationImages",
        "all_images_1_annotation_pct",
        "all_images_2_AnnotationImages",
        "all_images_2_annotation_pct",
        "all_images_3_AnnotationImages",
        "all_images_3_annotation_pct",
        "all_images_4_plus_AnnotationImages",
        "all_images_4_plus_annotation_pct",
        "estimated_hours",
        "estimated_minutes_per_image",
        "projected_rectangles_for_all_images",
        "projected_total_hours_for_all_images",
        "label_studio_annotation_objects",
        "result_items",
    ]

    print_table(rows_with_total, columns)
    print_workload_summary(total_row, args.seconds_per_image, args.seconds_per_rectangle)
    print_histogram_summary(rows_with_total)
    if not args.no_csv:
        write_csv(args.csv_output, [visible_row(row, columns) for row in rows_with_total], columns)
        write_histogram_csv(args.histogram_csv_output, rows_with_total)
        print(f"\nCSV written to: {args.csv_output}")
        print(f"Histogram CSV written to: {args.histogram_csv_output}")


if __name__ == "__main__":
    main()
