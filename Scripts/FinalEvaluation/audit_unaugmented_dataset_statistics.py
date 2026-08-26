"""Recompute MDWD and MTSD split statistics from unaugmented dataset files only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def duplicate_groups(values: dict[str, list[dict]]) -> dict[str, list[dict]]:
    return {value: rows for value, rows in values.items() if len(rows) > 1}


def dhash(path: Path) -> int:
    """Return a compact perceptual hash for candidate source-image comparisons."""
    with Image.open(path) as image:
        pixels = image.convert("L").resize((9, 8)).getdata()
    values = list(pixels)
    return sum(
        (values[row * 9 + col] > values[row * 9 + col + 1]) << (row * 8 + col)
        for row in range(8)
        for col in range(8)
    )


def coco_audit(dataset: str, root: Path, *, source_manifest: Path | None = None) -> dict:
    records, issues = [], []
    hashes, source_keys = defaultdict(list), defaultdict(list)
    class_names = None
    for split in SPLITS:
        split_dir = root / split
        annotation_path = split_dir / "_annotations.coco.json"
        if not annotation_path.exists():
            issues.append({"severity": "critical", "type": "missing_annotation_file", "detail": str(annotation_path)})
            continue
        coco = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = {item["id"]: item["name"] for item in coco["categories"]}
        if class_names is None:
            class_names = categories
        elif categories != class_names:
            issues.append({"severity": "critical", "type": "class_schema_mismatch", "detail": split})
        images = {item["id"]: item for item in coco["images"]}
        annotations_by_image = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations_by_image[annotation["image_id"]].append(annotation)
            if annotation["image_id"] not in images:
                issues.append({"severity": "critical", "type": "orphan_annotation", "detail": f"{split}: annotation {annotation.get('id')}"})
            if annotation["category_id"] not in categories:
                issues.append({"severity": "critical", "type": "unknown_category", "detail": f"{split}: annotation {annotation.get('id')}"})
        disk_images = {path.name: path for path in split_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
        coco_names = {image["file_name"] for image in images.values()}
        missing_files = sorted(coco_names - disk_images.keys())
        unreferenced_files = sorted(disk_images.keys() - coco_names)
        for name in missing_files:
            issues.append({"severity": "critical", "type": "annotation_references_missing_image", "detail": f"{split}: {name}"})
        for name in unreferenced_files:
            issues.append({"severity": "high", "type": "image_missing_annotation_record", "detail": f"{split}: {name}"})
        for image_id, image in images.items():
            name = image["file_name"]
            annotations = annotations_by_image[image_id]
            if not annotations:
                issues.append({"severity": "medium", "type": "image_with_zero_objects", "detail": f"{split}: {name}"})
            if name in disk_images:
                digest = sha256(disk_images[name])
                hashes[digest].append({"split": split, "file": name})
            # Roboflow's suffix is an export identifier; removing it produces the source filename key.
            source_key = re.sub(r"\.rf\.[0-9a-f]{32}(?=\.[^.]+$)", "", name, flags=re.IGNORECASE)
            source_keys[source_key.casefold()].append({"split": split, "file": name, "path": str(disk_images.get(name, ""))})
            records.append({
                "split": split,
                "file": name,
                "objects": len(annotations),
                "classes": Counter(categories[ann["category_id"]] for ann in annotations),
            })

    duplicate_hashes = duplicate_groups(hashes)
    duplicate_source_keys = duplicate_groups(source_keys)
    for label, groups in (("exact_image_duplicate", duplicate_hashes),):
        for value, rows in groups.items():
            splits = sorted({row["split"] for row in rows})
            issues.append({"severity": "critical" if len(splits) > 1 else "medium", "type": label, "detail": f"{value}: {rows}"})

    collision_groups, leakage_groups = 0, 0
    for key, rows in duplicate_source_keys.items():
        if len({row["split"] for row in rows}) < 2:
            continue
        collision_groups += 1
        for row in rows:
            row["dhash"] = dhash(Path(row["path"]))
        distances = [
            (left["dhash"] ^ right["dhash"]).bit_count()
            for index, left in enumerate(rows)
            for right in rows[index + 1 :]
            if left["split"] != right["split"]
        ]
        minimum = min(distances)
        if minimum <= 5:
            leakage_groups += 1
            issues.append({"severity": "critical", "type": "probable_source_image_leakage_across_splits", "detail": f"{key}: minimum dHash distance {minimum}; {rows}"})
        else:
            issues.append({"severity": "medium", "type": "cross_split_filename_collision_not_visually_similar", "detail": f"{key}: minimum dHash distance {minimum}; {rows}"})

    result = assemble(dataset, records, class_names or {}, issues, duplicate_hashes, duplicate_source_keys, source_manifest)
    result["cross_split_source_filename_collision_groups"] = collision_groups
    result["probable_source_image_leakage_groups_across_splits"] = leakage_groups
    return result


def assemble(dataset, records, categories, issues, duplicate_hashes, duplicate_source_keys, source_manifest):
    class_order = [categories[key] for key in sorted(categories)]
    splits = {}
    for split in SPLITS:
        split_records = [row for row in records if row["split"] == split]
        classes = Counter()
        for row in split_records:
            classes.update(row["classes"])
        objects = sum(classes.values())
        splits[split] = {
            "images": len(split_records),
            "objects": objects,
            "class_counts": {name: classes[name] for name in class_order},
            "class_percentages": {name: round(100 * classes[name] / objects, 4) if objects else 0.0 for name in class_order},
            "class_count_reconciles": objects == sum(classes.values()),
        }
    all_images = sum(item["images"] for item in splits.values())
    all_objects = sum(item["objects"] for item in splits.values())
    result = {
        "dataset": dataset,
        "included_source": "unaugmented COCO split directories only",
        "splits": splits,
        "total_unique_images": all_images,
        "total_annotated_objects": all_objects,
        "reconciliation": {
            "split_images_sum": all_images,
            "split_objects_sum": all_objects,
            "image_total_reconciles": all_images == len(records),
            "object_total_reconciles": all_objects == sum(sum(row["classes"].values()) for row in records),
        },
        "duplicate_exact_image_groups": len(duplicate_hashes),
        "duplicate_source_filename_groups": len(duplicate_source_keys),
        "issues": issues,
    }
    if source_manifest:
        with source_manifest.open(encoding="utf-8-sig", newline="") as handle:
            manifest = list(csv.DictReader(handle))
        manifest_by_out = {row["out_name"]: row for row in manifest}
        missing_manifest = sorted({row["file"] for row in records} - manifest_by_out.keys())
        extra_manifest = sorted(manifest_by_out.keys() - {row["file"] for row in records})
        source_hashes = defaultdict(list)
        for row in manifest:
            source_hashes[row["source_image_sha256"]].append({"split": row["split"], "file": row["out_name"]})
        source_dupes = duplicate_groups(source_hashes)
        cross_split_source_dupes = {key: rows for key, rows in source_dupes.items() if len({row["split"] for row in rows}) > 1}
        result["manifest_cross_check"] = {
            "rows": len(manifest),
            "missing_manifest_rows_for_coco_images": missing_manifest,
            "manifest_rows_missing_from_coco_images": extra_manifest,
            "source_image_hash_duplicate_groups": len(source_dupes),
            "source_image_leakage_groups_across_splits": len(cross_split_source_dupes),
        }
        if missing_manifest or extra_manifest:
            issues.append({"severity": "critical", "type": "coco_manifest_image_mismatch", "detail": f"missing={len(missing_manifest)}, extra={len(extra_manifest)}"})
        for key, rows in cross_split_source_dupes.items():
            issues.append({"severity": "critical", "type": "source_image_leakage_across_splits", "detail": f"{key}: {rows}"})
    return result


def markdown(result: dict) -> str:
    lines = [f"# {result['dataset']} unaugmented dataset statistics audit", "", f"Source included: `{result['included_source']}`.", "", "| Split | Images | Objects | Class total check |", "|---|---:|---:|---|" ]
    for split, values in result["splits"].items():
        lines.append(f"| {split.title()} | {values['images']:,} | {values['objects']:,} | {'PASS' if values['class_count_reconciles'] else 'FAIL'} |")
    lines.extend([f"| **Total** | **{result['total_unique_images']:,}** | **{result['total_annotated_objects']:,}** | **{'PASS' if result['reconciliation']['image_total_reconciles'] and result['reconciliation']['object_total_reconciles'] else 'FAIL'}** |", ""])
    for split, values in result["splits"].items():
        lines.extend([f"## {split.title()} class distribution", "", "| Class | Objects | % of split objects |", "|---|---:|---:|"])
        lines.extend(f"| {name} | {count:,} | {values['class_percentages'][name]:.4f}% |" for name, count in values["class_counts"].items())
        lines.append(f"| **Total** | **{values['objects']:,}** | **100.0000%** |\n")
    lines.extend(["## Integrity checks", "", f"- Exact-image duplicate groups: {result['duplicate_exact_image_groups']}", f"- Normalized source-filename duplicate groups: {result['duplicate_source_filename_groups']}"])
    lines.extend([f"- Cross-split source-filename collision groups: {result['cross_split_source_filename_collision_groups']}", f"- Probable source-image leakage groups (dHash distance ≤5): {result['probable_source_image_leakage_groups_across_splits']}"])
    if "manifest_cross_check" in result:
        check = result["manifest_cross_check"]
        lines.extend([f"- Source-image hash duplicate groups in manifest: {check['source_image_hash_duplicate_groups']}", f"- Source-image leakage groups across splits: {check['source_image_leakage_groups_across_splits']}", f"- COCO images absent from manifest: {len(check['missing_manifest_rows_for_coco_images'])}", f"- Manifest images absent from COCO: {len(check['manifest_rows_missing_from_coco_images'])}"])
    if "source_qa_annotation_cross_check" in result:
        check = result["source_qa_annotation_cross_check"]
        lines.extend([f"- Final-QA source files checked: {check['qa_files']}", f"- Final-QA image total matches prepared COCO: {check['matches_prepared_coco_images']}", f"- Final-QA annotation total matches prepared COCO: {check['matches_prepared_coco_annotations']}", f"- Final-QA per-class totals match prepared COCO: {check['matches_prepared_coco_per_class']}"])
    lines.append(f"- Findings requiring attention: {len(result['issues'])}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path("Documents/Final-Reports/Dataset-Statistics-Audit"))
    args = parser.parse_args()
    root, output = args.root.resolve(), (args.root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mdwd = coco_audit("MDWD", root / "Datasets/MDWD/MDWD-RAW")
    mtsd = coco_audit("MTSD", root / "Datasets/MTSD/Prepared/MTSD-Unaugmented/MTSD-COCO", source_manifest=root / "Datasets/MTSD/Prepared/MTSD-Unaugmented/split_manifest.csv")
    qa_files = sorted((root / "Datasets/MTSD/Annotations").glob("GRP-*/Final-QA/QA-*.json"))
    qa_images = qa_annotations = 0
    qa_schemas = set()
    qa_class_counts = Counter()
    for qa_file in qa_files:
        qa = json.loads(qa_file.read_text(encoding="utf-8"))
        qa_images += len(qa["images"])
        qa_annotations += len(qa["annotations"])
        qa_schemas.add(tuple((item["id"], item["name"]) for item in qa["categories"]))
        categories = {item["id"]: item["name"] for item in qa["categories"]}
        qa_class_counts.update(categories[item["category_id"]] for item in qa["annotations"])
    prepared_class_counts = Counter()
    for split in mtsd["splits"].values():
        prepared_class_counts.update(split["class_counts"])
    mtsd["source_qa_annotation_cross_check"] = {
        "qa_files": len(qa_files),
        "source_images": qa_images,
        "source_annotations": qa_annotations,
        "category_schemas": len(qa_schemas),
        "matches_prepared_coco_images": qa_images == mtsd["total_unique_images"],
        "matches_prepared_coco_annotations": qa_annotations == mtsd["total_annotated_objects"],
        "matches_prepared_coco_per_class": qa_class_counts == prepared_class_counts,
    }
    if qa_images != mtsd["total_unique_images"] or qa_annotations != mtsd["total_annotated_objects"] or qa_class_counts != prepared_class_counts or len(qa_schemas) != 1:
        mtsd["issues"].append({"severity": "critical", "type": "source_qa_cross_check_failure", "detail": "Prepared COCO totals or class schema disagree with Final-QA source files"})
    results = {"mdwd": mdwd, "mtsd": mtsd}
    (output / "audit.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output / "MDWD_unaugmented_statistics.md").write_text(markdown(mdwd), encoding="utf-8")
    (output / "MTSD_unaugmented_statistics.md").write_text(markdown(mtsd), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
