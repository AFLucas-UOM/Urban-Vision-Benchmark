#!/usr/bin/env python3
"""Final, dissertation-ready dataset integrity report for MDWD and MTSD.

Read-only. Reuses the repository's existing audit machinery so this report can
never disagree with the working tools:

  * MDWD (and the prepared MTSD detection dataset, once built): the
    ``mdwd_eda`` mapper/checks - missing/unreadable images, missing/orphan
    labels, malformed rows, invalid class ids, out-of-range boxes, empty
    annotations, duplicate filenames, augmented-source split leakage, plus
    class/image/annotation distributions per split.
  * MTSD QA annotations: the MTSD-AnnotationQA scanner - missing/invalid
    attributes, duplicate & near-duplicate boxes, reference problems
    (groups discovered dynamically).

Statuses: FAIL = broken data that would corrupt training/evaluation
(missing files, invalid classes/boxes, malformed annotations);
WARNING = quality issues to acknowledge (duplicates, leakage, empty
annotations, vocabulary drop-values); PASS = clean.

Outputs (default; override with --output-dir):
    Documents/Final-Reports/dataset_integrity_report.md
    Documents/Final-Reports/dataset_integrity_report.json
    Documents/Final-Reports/dataset_integrity_issues.csv

Usage:
    python Scripts/FinalEvaluation/final_dataset_integrity_check.py --dataset all
    python Scripts/FinalEvaluation/final_dataset_integrity_check.py --dataset MDWD --max-images 500 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path = SCRIPT_DIR) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "Datasets").exists() and (parent / "Scripts").exists():
            return parent
    raise RuntimeError("Could not locate project root.")


ROOT = find_project_root()
MDWD_ANALYSIS = ROOT / "Scripts" / "MDWD-Scripts" / "MDWD-Analysis"
ANNOTATION_QA = ROOT / "Scripts" / "MTSD-Scripts" / "MTSD-AnnotationQA"
MTSD_PREPARED = ROOT / "Datasets" / "MTSD" / "Prepared" / "MTSD-YOLO"
DEFAULT_OUTPUT_DIR = ROOT / "Documents" / "Final-Reports"

# Issue-type severity map (anything not listed is WARNING).
FAIL_ISSUES = {
    "missing_split", "unreadable_image", "missing_image", "missing_label",
    "orphan_label", "orphan_annotation", "malformed_label_row", "invalid_class_id",
    "out_of_range_box", "malformed_json", "missing_source_field",
    "source_image_missing_on_disk", "unknown_category", "bbox_not_4_numbers",
    "bbox_not_numeric", "bbox_non_positive_size", "bbox_negative_origin",
    "bbox_outside_image", "missing_attribute",
}
WARN_ISSUES = {
    "empty_annotation", "duplicate_filename", "source_in_multiple_splits",
    "attr_missing", "attr_unknown_value", "attr_case_mismatch",
    "attr_close_misspelling", "attr_known_drop_value", "old_path_convention",
    "dup_exact_duplicate", "dup_conflicting_duplicate", "dup_high_overlap",
    "box_clipped_to_image", "image_without_valid_boxes", "degenerate_box_after_clip",
}


def severity_of(issue: str) -> str:
    if issue in FAIL_ISSUES:
        return "FAIL"
    if issue in WARN_ISSUES:
        return "WARNING"
    return "WARNING"


# ---------------------------------------------------------------------------
# MDWD (and prepared-MTSD) via mdwd_eda
# ---------------------------------------------------------------------------

def check_mdwd(max_images: int | None) -> dict:
    if str(MDWD_ANALYSIS) not in sys.path:
        sys.path.insert(0, str(MDWD_ANALYSIS))
    from mdwd_eda import stats as mdwd_stats
    from mdwd_eda.mapper import build_map

    dataset_map = build_map(max_images=max_images)
    summary = mdwd_stats.to_native(mdwd_stats.summary_dict(dataset_map))
    issues = [{"dataset": "MDWD", "issue": i["type"],
               "severity": severity_of(i["type"]),
               "where": str(i.get("file", i.get("path", ""))),
               "split": i.get("split", ""), "detail": str(i.get("detail", ""))}
              for i in dataset_map.issues]
    class_by_split = mdwd_stats.class_distribution(dataset_map).to_dict(orient="records")
    return {
        "name": "MDWD (variant MDWD-YOLO26)",
        "format": dataset_map.format,
        "sampled": max_images is not None,
        "summary": summary,
        "class_distribution_by_split": class_by_split,
        "issues": issues,
        "notes": [
            "Counts are export-level (train is ~10x offline-augmented); "
            f"unique source images: {summary['n_unique_source_images']}.",
            "source_in_multiple_splits = augmented copies of one capture in more "
            "than one split (leakage) - list in the issues CSV.",
        ],
    }


def check_prepared_mtsd_detection() -> dict | None:
    """Light YOLO-layout audit of Datasets/MTSD/Prepared/MTSD-YOLO, if built."""
    if not MTSD_PREPARED.is_dir():
        return None
    import yaml

    class_names = list((yaml.safe_load((MTSD_PREPARED / "data.yaml").read_text(encoding="utf-8")) or {}).get("names", []))
    issues, per_split = [], {}
    class_counts: dict[str, Counter] = {}
    for split in ("train", "valid", "test"):
        images_dir, labels_dir = MTSD_PREPARED / split / "images", MTSD_PREPARED / split / "labels"
        image_stems = {p.stem for p in images_dir.iterdir() if p.is_file()} if images_dir.is_dir() else set()
        label_stems = {p.stem for p in labels_dir.glob("*.txt")} if labels_dir.is_dir() else set()
        counts = Counter()
        boxes = 0
        for label_path in (labels_dir.glob("*.txt") if labels_dir.is_dir() else []):
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    class_id = int(parts[0])
                    values = [float(v) for v in parts[1:5]]
                    assert len(parts) == 5 and 0 <= class_id < len(class_names)
                    assert all(0 <= v <= 1 for v in values) and values[2] > 0 and values[3] > 0
                    counts[class_names[class_id]] += 1
                    boxes += 1
                except (AssertionError, ValueError, IndexError):
                    issues.append({"dataset": "MTSD-prepared", "issue": "malformed_label_row",
                                   "severity": "FAIL", "where": str(label_path.relative_to(ROOT)),
                                   "split": split, "detail": line[:60]})
        for stem in sorted(image_stems - label_stems):
            issues.append({"dataset": "MTSD-prepared", "issue": "missing_label", "severity": "FAIL",
                           "where": stem, "split": split, "detail": ""})
        for stem in sorted(label_stems - image_stems):
            issues.append({"dataset": "MTSD-prepared", "issue": "orphan_label", "severity": "FAIL",
                           "where": stem, "split": split, "detail": ""})
        per_split[split] = {"images": len(image_stems), "labels": len(label_stems), "boxes": boxes}
        class_counts[split] = counts
    return {
        "name": "MTSD prepared detection dataset (MTSD-YOLO)",
        "format": "yolo",
        "summary": {"per_split": per_split, "classes": class_names},
        "class_distribution_by_split": [
            {"class_name": name, **{s: class_counts[s].get(name, 0) for s in per_split}}
            for name in class_names],
        "issues": issues,
        "notes": ["Split leakage is impossible by construction here (group-prefixed "
                  "unique captures, no augmented duplicates); provenance in prep_manifest.json."],
    }


# ---------------------------------------------------------------------------
# MTSD QA annotations via MTSD-AnnotationQA
# ---------------------------------------------------------------------------

def check_mtsd_qa() -> dict:
    if str(ANNOTATION_QA) not in sys.path:
        sys.path.insert(0, str(ANNOTATION_QA))
    import config as qa_config
    from annotation_schema import load_attribute_schema
    from scan_annotations import audit_file, discover_qa_files, group_of

    qa_files = discover_qa_files(qa_config.ANNOTATIONS_ROOT)
    schema = load_attribute_schema()
    rows = {"missing": [], "invalid": [], "duplicates": [], "reference": []}
    stats: Counter = Counter()
    shim = SimpleNamespace(iou_dup=qa_config.DUPLICATE_IOU, iou_overlap=qa_config.OVERLAP_IOU,
                           verify_images=False)
    for qa_path in qa_files:
        audit_file(qa_path, group_of(qa_path, qa_config.ANNOTATIONS_ROOT), schema, shim, rows, stats)

    issues = []
    for row in rows["missing"]:
        issues.append({"dataset": "MTSD-QA", "issue": "missing_attribute", "severity": "FAIL",
                       "where": f"{row['group']}/{row['image']}#{row['annotation_id']}",
                       "split": row["group"], "detail": f"{row['attribute']}={row['value']}"})
    for row in rows["invalid"]:
        issue = f"attr_{row.get('status', 'unknown_value')}"
        issues.append({"dataset": "MTSD-QA", "issue": issue, "severity": severity_of(issue),
                       "where": f"{row['group']}/{row['image']}#{row['annotation_id']}",
                       "split": row["group"],
                       "detail": f"{row['attribute']}={row['value']} suggestion={row.get('suggestion', '')}"})
    for row in rows["duplicates"]:
        issue = f"dup_{row['kind']}"
        issues.append({"dataset": "MTSD-QA", "issue": issue, "severity": severity_of(issue),
                       "where": f"{row['group']}/{row['image']}",
                       "split": row["group"],
                       "detail": f"ann {row['annotation_id_a']} vs {row['annotation_id_b']} IoU={row['iou']}"})
    for row in rows["reference"]:
        issues.append({"dataset": "MTSD-QA", "issue": row["problem"],
                       "severity": severity_of(row["problem"]),
                       "where": f"{row['group']}/{row['image']}",
                       "split": row["group"], "detail": str(row.get("detail", ""))})

    per_group = {}
    for key, value in sorted(stats.items()):
        group, metric = key.split("/", 1)
        per_group.setdefault(group, {})[metric] = value
    return {
        "name": "MTSD QA annotations (Final-QA COCO exports)",
        "format": "coco+attributes",
        "summary": {"qa_files": [str(p.relative_to(ROOT)) for p in qa_files], "per_group": per_group,
                    "attribute_schema": schema},
        "issues": issues,
        "notes": ["Duplicate thresholds: exact/conflicting IoU >= "
                  f"{qa_config.DUPLICATE_IOU}, high-overlap >= {qa_config.OVERLAP_IOU}.",
                  "Fixable via the MTSD-AnnotationQA review/apply workflow."],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def section_status(section_or_issues) -> str:
    if isinstance(section_or_issues, dict):
        if section_or_issues.get("pending"):
            return "PENDING"
        issues = section_or_issues["issues"]
    else:
        issues = section_or_issues
    if any(i["severity"] == "FAIL" for i in issues):
        return "FAIL"
    if any(i["severity"] == "WARNING" for i in issues):
        return "WARNING"
    return "PASS"


def write_reports(sections: list[dict], output_dir: Path, generated_at: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_issues = [issue for section in sections for issue in section["issues"]]
    overall = ("FAIL" if any(i["severity"] == "FAIL" for i in all_issues)
               else "WARNING" if all_issues else "PASS")

    payload = {"generated_at": generated_at, "overall": overall,
               "sections": [{**s, "status": section_status(s),
                             "issue_counts": dict(Counter(i["issue"] for i in s["issues"]))}
                            for s in sections]}
    (output_dir / "dataset_integrity_report.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

    with (output_dir / "dataset_integrity_issues.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "issue", "severity", "split", "where", "detail"])
        writer.writeheader()
        writer.writerows(all_issues)

    lines = ["# Dataset integrity report",
             f"\nGenerated: {generated_at}  |  Overall: **{overall}**",
             "\nFAIL = data that would corrupt training/evaluation; WARNING = quality "
             "issues to acknowledge in the dissertation; full row-level list in "
             "`dataset_integrity_issues.csv`.\n"]
    for section in payload["sections"]:
        lines.append(f"\n## {section['name']} — **{section['status']}**\n")
        if section.get("sampled"):
            lines.append("> Sample mode (`--max-images`) — counts are partial.\n")
        for note in section.get("notes", []):
            lines.append(f"> {note}\n")
        if section["issue_counts"]:
            lines.append("| Issue | Severity | Count |")
            lines.append("| --- | --- | --- |")
            for issue, count in sorted(section["issue_counts"].items()):
                lines.append(f"| {issue} | {severity_of(issue)} | {count} |")
        else:
            lines.append("No issues found.")
        summary = section.get("summary", {})
        per_split = summary.get("per_split") or (summary.get("summary", {}) or {}).get("per_split")
        if per_split:
            lines.append("\n| Split | Images | Boxes |")
            lines.append("| --- | --- | --- |")
            for split, values in per_split.items():
                lines.append(f"| {split} | {values.get('images', '-')} | {values.get('boxes', '-')} |")
        table = section.get("class_distribution_by_split")
        if table:
            columns = [c for c in table[0] if c != "class_name"]
            lines.append("\n| Class | " + " | ".join(columns) + " |")
            lines.append("|" + "---|" * (len(columns) + 1))
            for row in table:
                lines.append("| " + str(row.get("class_name", "?")) + " | "
                             + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    lines.append("\n*Read-only report - no dataset files were modified.*\n")
    (output_dir / "dataset_integrity_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Final dataset integrity check (read-only).")
    parser.add_argument("--dataset", choices=["MDWD", "MTSD", "all"], default="all")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Per-split cap for the MDWD scan (sample mode; cross-split checks skipped).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Print statuses; write nothing.")
    args = parser.parse_args()

    generated_at = datetime.now().isoformat(timespec="seconds")
    sections = []
    if args.dataset in ("MDWD", "all"):
        print("Checking MDWD..." + (" (sample mode)" if args.max_images else ""))
        sections.append(check_mdwd(args.max_images))
    if args.dataset in ("MTSD", "all"):
        print("Checking MTSD QA annotations...")
        sections.append(check_mtsd_qa())
        prepared = check_prepared_mtsd_detection()
        if prepared:
            print("Checking prepared MTSD detection dataset...")
            sections.append(prepared)
        else:
            sections.append({"name": "MTSD prepared detection dataset (MTSD-YOLO)",
                             "format": "-", "summary": {}, "pending": True,
                             "issues": [],
                             "notes": ["PENDING - not built yet; run "
                                       "Prepare-MTSD-Detection-Dataset.ipynb first."]})

    for section in sections:
        status = section_status(section)
        counts = Counter(i["severity"] for i in section["issues"])
        print(f"  {section['name']:<52} {status:<8} "
              f"(FAIL {counts.get('FAIL', 0)}, WARN {counts.get('WARNING', 0)})")

    if args.dry_run:
        print("\nDry run: no report files written.")
        return 0
    write_reports(sections, args.output_dir, generated_at)
    print(f"\nReports written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
