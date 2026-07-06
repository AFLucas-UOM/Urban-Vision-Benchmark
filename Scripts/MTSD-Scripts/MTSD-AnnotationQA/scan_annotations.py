#!/usr/bin/env python3
"""Audit every MTSD QA annotation file for schema, duplicate and reference issues.

Discovers QA JSON exports dynamically under the annotations root (current and
future groups alike), then checks:

  1. missing required attributes (class label, view_angle, mounting,
     condition, sign_shape - schema read from the AttributeClassification
     config so the audit matches the training pipeline);
  2. invalid attribute values (null/empty, unknown, case mismatches and
     close misspellings against the controlled vocabulary);
  3. duplicate annotations (same-image pairs with IoU above configurable
     thresholds, split into exact duplicates, conflicting duplicates and
     high-overlap warnings);
  4. image/reference problems (missing or old-convention source_image paths,
     unreadable images with --verify-images, malformed JSON, orphan
     annotations, invalid bboxes).

Read-only: annotation files are never modified. Reports are written to a
timestamped folder under outputs/ (skipped entirely with --dry-run).

Usage:
    python scan_annotations.py                 # full audit, writes reports
    python scan_annotations.py --dry-run       # print findings only
    python scan_annotations.py --verify-images # also open every image header
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config  # noqa: E402
from annotation_schema import bbox_problems, classify_attribute_value, load_attribute_schema  # noqa: E402
from duplicate_detection import find_duplicate_pairs  # noqa: E402


def is_qa_json(path: Path) -> bool:
    name = path.name.lower()
    if any(token in name for token in config.SKIP_SUFFIX_TOKENS):
        return False
    if path.suffix.lower() != ".json":
        return False
    return (
        path.parent.name.lower() in config.QA_DIR_NAMES
        or name.startswith(config.QA_FILENAME_PREFIXES)
    )


def discover_qa_files(annotations_root: Path) -> list[Path]:
    return sorted(p for p in annotations_root.rglob("*.json") if is_qa_json(p))


def group_of(qa_path: Path, annotations_root: Path) -> str:
    try:
        return qa_path.relative_to(annotations_root).parts[0]
    except ValueError:
        return qa_path.parent.name


def audit_file(
    qa_path: Path,
    group: str,
    schema: dict[str, list[str]],
    args: argparse.Namespace,
    rows: dict[str, list[dict]],
    stats: Counter,
) -> None:
    rel = str(qa_path.relative_to(config.BASE_DIR))
    try:
        data = json.loads(qa_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        rows["reference"].append({
            "group": group, "qa_file": rel, "image": "", "annotation_id": "",
            "problem": "malformed_json", "detail": f"{type(exc).__name__}: {exc}",
        })
        stats[f"{group}/malformed_json"] += 1
        return

    categories = {c.get("id"): c.get("name") for c in data.get("categories", [])}
    images = {im.get("id"): im for im in data.get("images", [])}
    annotations_by_image: dict = defaultdict(list)
    stats[f"{group}/images"] += len(images)
    stats[f"{group}/annotations"] += len(data.get("annotations", []))

    # --- image / reference checks -------------------------------------------
    for image in data.get("images", []):
        file_name = image.get("file_name", "")
        source_rel = str(image.get("source_image", "") or "").replace("\\", "/")
        if not source_rel:
            rows["reference"].append({
                "group": group, "qa_file": rel, "image": file_name, "annotation_id": "",
                "problem": "missing_source_image_field", "detail": "",
            })
            stats[f"{group}/missing_source_field"] += 1
            continue
        if not source_rel.startswith(config.EXPECTED_SOURCE_PREFIX):
            rows["reference"].append({
                "group": group, "qa_file": rel, "image": file_name, "annotation_id": "",
                "problem": "old_path_convention", "detail": source_rel,
            })
            stats[f"{group}/old_path_convention"] += 1
        source_path = config.BASE_DIR / source_rel
        if not source_path.exists():
            rows["reference"].append({
                "group": group, "qa_file": rel, "image": file_name, "annotation_id": "",
                "problem": "source_image_missing_on_disk", "detail": source_rel,
            })
            stats[f"{group}/missing_image_file"] += 1
        elif args.verify_images:
            try:
                from PIL import Image

                with Image.open(source_path) as img:
                    img.size  # header read only
            except Exception as exc:
                rows["reference"].append({
                    "group": group, "qa_file": rel, "image": file_name, "annotation_id": "",
                    "problem": "unreadable_image", "detail": f"{type(exc).__name__}: {exc}",
                })
                stats[f"{group}/unreadable_image"] += 1

    # --- per-annotation checks ------------------------------------------------
    for annotation in data.get("annotations", []):
        ann_id = annotation.get("id")
        image = images.get(annotation.get("image_id"))
        image_name = image.get("file_name", "") if image else ""
        if image is None:
            rows["reference"].append({
                "group": group, "qa_file": rel, "image": "", "annotation_id": ann_id,
                "problem": "orphan_annotation", "detail": f"image_id={annotation.get('image_id')}",
            })
            stats[f"{group}/orphan_annotation"] += 1

        # class label
        if annotation.get("category_id") not in categories:
            rows["invalid"].append({
                "group": group, "qa_file": rel, "image": image_name, "annotation_id": ann_id,
                "attribute": "category_id", "value": annotation.get("category_id"),
                "status": "unknown_category", "suggestion": "",
            })
            stats[f"{group}/unknown_category"] += 1

        # bbox structure
        for problem in bbox_problems(
            annotation.get("bbox"),
            image.get("width") if image else None,
            image.get("height") if image else None,
        ):
            rows["reference"].append({
                "group": group, "qa_file": rel, "image": image_name, "annotation_id": ann_id,
                "problem": problem, "detail": str(annotation.get("bbox")),
            })
            stats[f"{group}/{problem}"] += 1

        # required attributes
        attrs = annotation.get("attributes")
        if not isinstance(attrs, dict):
            for attribute in schema:
                rows["missing"].append({
                    "group": group, "qa_file": rel, "image": image_name, "annotation_id": ann_id,
                    "attribute": attribute, "value": "<attributes dict absent>",
                    "current_attributes": "", "suggestion": "",
                })
                stats[f"{group}/missing_attribute"] += 1
            continue
        for attribute, vocabulary in schema.items():
            if attribute not in attrs:
                rows["missing"].append({
                    "group": group, "qa_file": rel, "image": image_name, "annotation_id": ann_id,
                    "attribute": attribute, "value": "<key absent>",
                    "current_attributes": json.dumps(attrs, default=str), "suggestion": "",
                })
                stats[f"{group}/missing_attribute"] += 1
                continue
            status, suggestion = classify_attribute_value(attribute, attrs[attribute], vocabulary)
            if status == "ok":
                continue
            target = "missing" if status == "missing" else "invalid"
            rows[target].append({
                "group": group, "qa_file": rel, "image": image_name, "annotation_id": ann_id,
                "attribute": attribute, "value": attrs[attribute],
                **({"current_attributes": json.dumps(attrs, default=str)} if target == "missing" else {}),
                "status": status, "suggestion": suggestion or "",
            })
            stats[f"{group}/attr_{status}"] += 1

        if image is not None:
            annotations_by_image[annotation["image_id"]].append(annotation)

    # --- duplicates -------------------------------------------------------------
    for image_id, image_annotations in annotations_by_image.items():
        if len(image_annotations) < 2:
            continue
        for pair in find_duplicate_pairs(image_annotations, args.iou_dup, args.iou_overlap):
            image = images[image_id]
            pair.update({
                "group": group, "qa_file": rel,
                "image": image.get("file_name", ""),
                "source_image": image.get("source_image", ""),
                "category_a": categories.get(pair["category_id_a"], ""),
                "category_b": categories.get(pair["category_id_b"], ""),
            })
            rows["duplicates"].append(pair)
            stats[f"{group}/dup_{pair['kind']}"] += 1


def write_reports(audit_dir: Path, rows: dict, stats: Counter, schema: dict, meta: dict) -> None:
    def write_csv(name: str, records: list[dict]) -> None:
        path = audit_dir / name
        if not records:
            path.write_text("", encoding="utf-8")
            return
        fieldnames = sorted({k for r in records for k in r}, key=lambda k: (k != "group", k))
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    write_csv("missing_attributes.csv", rows["missing"])
    write_csv("invalid_attributes.csv", rows["invalid"])
    write_csv("duplicate_candidates.csv", rows["duplicates"])
    write_csv("reference_problems.csv", rows["reference"])

    groups = sorted({key.split("/")[0] for key in stats})
    per_group = []
    metrics = sorted({key.split("/", 1)[1] for key in stats})
    for group in groups:
        per_group.append({"group": group, **{m: stats.get(f"{group}/{m}", 0) for m in metrics}})
    write_csv("per_group_stats.csv", per_group)

    summary = {
        **meta,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema": schema,
        "totals": {
            "missing_attribute_findings": len(rows["missing"]),
            "invalid_attribute_findings": len(rows["invalid"]),
            "duplicate_pair_findings": len(rows["duplicates"]),
            "reference_problem_findings": len(rows["reference"]),
        },
        "per_group": per_group,
    }
    (audit_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )

    lines = [
        "# MTSD annotation QA audit",
        f"\nGenerated: {summary['generated_at']}  ",
        f"QA files audited: {meta['qa_files_audited']}  ",
        f"Thresholds: duplicate IoU >= {meta['duplicate_iou']}, overlap IoU >= {meta['overlap_iou']}",
        "\n| Finding | Count |", "| --- | --- |",
    ]
    for key, value in summary["totals"].items():
        lines.append(f"| {key.replace('_', ' ')} | {value} |")
    lines.append("\n## Per-group counts\n")
    if per_group:
        header = ["group"] + metrics
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for row in per_group:
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in header) + " |")
    lines.append(
        "\nNext steps: review findings visually with `python review_app.py`, then "
        "apply confirmed decisions with `python apply_fixes.py --decisions "
        f"{audit_dir.name}/reviewed_decisions.json --apply`.\n"
    )
    (audit_dir / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit MTSD QA annotation files (read-only).")
    parser.add_argument("--annotations-root", type=Path, default=config.ANNOTATIONS_ROOT)
    parser.add_argument("--iou-dup", type=float, default=config.DUPLICATE_IOU,
                        help=f"IoU >= this is a duplicate candidate (default {config.DUPLICATE_IOU}).")
    parser.add_argument("--iou-overlap", type=float, default=config.OVERLAP_IOU,
                        help=f"IoU >= this is flagged as high overlap (default {config.OVERLAP_IOU}).")
    parser.add_argument("--verify-images", action="store_true",
                        help="Also open every referenced image header (slower).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print findings summary only; write no report files.")
    args = parser.parse_args()

    qa_files = discover_qa_files(args.annotations_root)
    if not qa_files:
        print(f"No QA annotation JSONs found under {args.annotations_root}")
        return 1
    schema = load_attribute_schema()
    print(f"Annotations root : {args.annotations_root}")
    print(f"QA files found   : {len(qa_files)}")
    for path in qa_files:
        print(f"  - {path.relative_to(config.BASE_DIR)}")
    print(f"Attribute schema : { {k: len(v) for k, v in schema.items()} }")

    rows: dict[str, list[dict]] = {"missing": [], "invalid": [], "duplicates": [], "reference": []}
    stats: Counter = Counter()
    for qa_path in qa_files:
        audit_file(qa_path, group_of(qa_path, args.annotations_root), schema, args, rows, stats)

    print("\n================ Findings ================")
    print(f"Missing attributes   : {len(rows['missing'])}")
    print(f"Invalid attributes   : {len(rows['invalid'])}")
    print(f"Duplicate candidates : {len(rows['duplicates'])} "
          f"({sum(1 for r in rows['duplicates'] if r['kind'] == 'exact_duplicate')} exact, "
          f"{sum(1 for r in rows['duplicates'] if r['kind'] == 'conflicting_duplicate')} conflicting, "
          f"{sum(1 for r in rows['duplicates'] if r['kind'] == 'high_overlap')} high-overlap)")
    print(f"Reference problems   : {len(rows['reference'])}")

    if args.dry_run:
        print("\nDry run: no report files written.")
        return 0

    audit_dir = config.new_audit_dir()
    meta = {
        "annotations_root": str(args.annotations_root),
        "qa_files_audited": len(qa_files),
        "qa_files": [str(p.relative_to(config.BASE_DIR)) for p in qa_files],
        "duplicate_iou": args.iou_dup,
        "overlap_iou": args.iou_overlap,
        "verify_images": args.verify_images,
    }
    write_reports(audit_dir, rows, stats, schema, meta)
    config.LATEST_POINTER.write_text(str(audit_dir) + "\n", encoding="utf-8")
    print(f"\nReports written to: {audit_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
