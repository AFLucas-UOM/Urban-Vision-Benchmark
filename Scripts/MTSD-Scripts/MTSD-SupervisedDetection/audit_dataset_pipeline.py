#!/usr/bin/env python3
"""Read-only MTSD dataset-pipeline state audit.

One command that answers: where does the MTSD dataset-creation pipeline stand
right now, and what still blocks final preparation? It reconciles

- Final-QA group discovery (active files, excluded backups, invalid files),
- class-vocabulary consistency across every group,
- the configured approved scope (config/default.yaml),
- the QA gate (config/qa_gate.yaml) and the newest annotation audit,
- the attribute-classification crop manifest's group scope and splits,
- prepared-dataset presence and (when present) strict validation,
- sampled source-image existence per group,

and emits a machine-readable JSON report plus a human-readable Markdown
summary. Strictly read-only over all inputs; with --dry-run nothing is
written at all, otherwise reports go to a fresh timestamped folder under
Documents/Final-Reports/MTSD-Dataset-Audit/.

Usage:
    python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/audit_dataset_pipeline.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import yaml

from mtsd_detection.annotation_sources import CLASS_NAMES, parse_qa
from mtsd_detection.config import load_config
from mtsd_detection.dataset_validation import validate_prepared
from mtsd_detection.qa_gate import load_qa_gate
from update_qa_gate import (compute_outstanding, compute_unaudited,
                            discover_final_qa_groups, discover_latest_audit)

AUDIT_ROOT = HERE.parent / "MTSD-AnnotationQA" / "outputs"


def audit(config_path: Path | None, sample_images: int = 5) -> dict:
    config = load_config(config_path or HERE / "config" / "default.yaml")
    repo_root = Path(config["repo_root"])
    annotations_root = Path(config["dataset"]["annotations_root"])
    report: dict = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "read_only": True, "blockers": [], "warnings": []}

    # 1. Discovery -----------------------------------------------------------
    discovery = discover_final_qa_groups(annotations_root)
    valid = dict(discovery["valid"])
    invalid = dict(discovery["invalid"])
    known = set(valid) | set(invalid)
    report["discovery"] = {
        "valid_groups": sorted(valid, key=lambda g: int(g.split("-")[-1])),
        "invalid_groups": invalid,
        "groups_without_final_qa": sorted(
            p.name for p in annotations_root.glob("GRP-*")
            if p.is_dir() and p.name not in known),
    }
    if invalid:
        report["blockers"].append(f"Invalid Final-QA files: {invalid}")

    # 2. Vocabulary consistency + sampled image existence ---------------------
    vocab_ok, image_checks = {}, {}
    for group, info in valid.items():
        qa_path = Path(info["path"]) if info.get("path") else None
        if qa_path is None:
            continue
        try:
            records, meta = parse_qa(qa_path, repo_root)
            vocab_ok[group] = True
        except ValueError as exc:
            vocab_ok[group] = False
            report["blockers"].append(f"{group}: {exc}")
            continue
        missing = [str(r["source_path"]) for r in records[:sample_images]
                   if not Path(r["source_path"]).is_file()]
        image_checks[group] = {"sampled": min(sample_images, len(records)),
                               "missing": missing}
        if missing:
            report["blockers"].append(
                f"{group}: sampled source images missing: {missing}")
    report["vocabulary_consistent"] = all(vocab_ok.values()) if vocab_ok else False
    report["expected_class_names"] = CLASS_NAMES
    report["sampled_image_existence"] = image_checks

    # 3. Scope reconciliation --------------------------------------------------
    approved = list(config["annotations"].get("approved_groups", []))
    discovered = report["discovery"]["valid_groups"]
    pending = [g for g in discovered if g not in approved]
    removed = [g for g in approved if g not in discovered]
    report["scope"] = {"approved_groups": approved,
                       "discovered_valid_groups": discovered,
                       "pending_unapproved_groups": pending,
                       "approved_but_missing_groups": removed,
                       "unexpected_group_policy":
                           config["annotations"].get("unexpected_group_policy")}
    if pending:
        report["blockers"].append(
            f"Final-QA groups outside approved scope (need audit + gate "
            f"refresh, never auto-approval): {pending}")
    if removed:
        report["blockers"].append(f"Approved groups without Final-QA: {removed}")

    # 4. QA gate + newest audit -------------------------------------------------
    gate = load_qa_gate(Path(config["annotations"]["qa_gate_file"]), repo_root)
    latest = discover_latest_audit(AUDIT_ROOT)
    outstanding = compute_outstanding(latest) if latest.get("found") else {}
    unaudited = compute_unaudited(discovered, latest) if latest.get("found") else discovered
    report["qa_gate"] = {"resolution_status": gate.get("resolution_status"),
                         "gate_audit_path": gate.get("audit_path"),
                         "gate_audited_groups": gate.get("audited_groups"),
                         "newest_audit_path": latest.get("path"),
                         "newest_audit_timestamp": latest.get("timestamp"),
                         "newest_audit_groups": latest.get("audited_groups"),
                         "outstanding_findings": outstanding,
                         "unaudited_discovered_groups": unaudited}
    if not gate.get("resolved"):
        report["blockers"].append(
            f"QA gate is {gate.get('resolution_status')}; outstanding findings "
            f"in newest audit: {outstanding or 'unknown'}; unaudited groups: "
            f"{unaudited or 'none'}")

    # 5. Attribute-crop manifest cross-check ------------------------------------
    attr_manifest_path = (repo_root / "Scripts/MTSD-Scripts/AttributeClassification"
                          / "outputs/manifests/manifest.json")
    if attr_manifest_path.is_file():
        with open(attr_manifest_path, encoding="utf-8-sig") as handle:
            attr_manifest = json.load(handle)
        attr_groups = sorted(attr_manifest.get("groups", {}),
                             key=lambda g: int(g.split("-")[-1]))
        splits: dict[str, int] = {}
        for record in attr_manifest.get("records", []):
            splits[record["split"]] = splits.get(record["split"], 0) + 1
        will_ingest = [g for g in discovered if g not in attr_groups]
        report["attribute_manifest"] = {
            "ingested_groups": attr_groups,
            "updated_at": attr_manifest.get("updated_at"),
            "split_sizes": splits,
            "groups_pending_auto_ingest": will_ingest,
            "split_policy": "per-source-image salted hash "
                            "(mtsd-attr-split-v1); crops of one photo share "
                            "one split by construction",
        }
        if will_ingest:
            report["warnings"].append(
                f"Attribute manifest auto-discovers groups: the next manifest "
                f"refresh/training run will ingest {will_ingest}, changing the "
                f"dataset under the completed size-ablation runs (trained on "
                f"{attr_groups}). Decide the final attribute scope explicitly "
                f"before any further attribute training.")
    else:
        report["attribute_manifest"] = {"present": False}

    # 6. Prepared datasets -------------------------------------------------------
    prepared_root = Path(config["dataset"]["prepared_root"])
    if prepared_root.exists() and any(prepared_root.glob("MTSD-*")):
        validation = validate_prepared(prepared_root, strict=True,
                                       policy=config["validation"])
        report["prepared"] = {"present": True, "ok": validation["ok"],
                              "fatal_findings": [f for v in validation.get("variants", {}).values()
                                                 for f in v["findings"] if f["severity"] == "fatal"]
                              + [f for f in validation.get("findings", [])]}
        if not validation["ok"]:
            report["blockers"].append("Prepared dataset failed strict validation")
    else:
        report["prepared"] = {"present": False}
        report["warnings"].append(
            "No prepared MTSD dataset exists yet (Datasets/MTSD/Prepared). "
            "Prompt-based MTSD evaluation and supervised training wait on "
            "final preparation.")

    report["ready_for_final_preparation"] = not report["blockers"]
    return report


def render_markdown(report: dict) -> str:
    lines = ["# MTSD dataset-pipeline state audit",
             f"Generated {report['generated_at']} (read-only).", ""]
    verdict = ("READY: no blockers - final preparation may proceed."
               if report["ready_for_final_preparation"]
               else "NOT READY for final preparation.")
    lines += [f"**{verdict}**", ""]
    if report["blockers"]:
        lines.append("## Blockers")
        lines += [f"- {b}" for b in report["blockers"]] + [""]
    if report["warnings"]:
        lines.append("## Warnings")
        lines += [f"- {w}" for w in report["warnings"]] + [""]
    scope = report["scope"]
    lines += ["## Scope",
              f"- approved: {', '.join(scope['approved_groups']) or '(none)'}",
              f"- discovered valid: {', '.join(scope['discovered_valid_groups'])}",
              f"- pending (unapproved): {', '.join(scope['pending_unapproved_groups']) or '(none)'}",
              ""]
    gate = report["qa_gate"]
    lines += ["## QA gate",
              f"- status: {gate['resolution_status']}",
              f"- newest audit: {gate['newest_audit_path']} ({gate['newest_audit_timestamp']})",
              f"- outstanding findings: {gate['outstanding_findings']}",
              f"- unaudited discovered groups: {gate['unaudited_discovered_groups'] or 'none'}",
              ""]
    attr = report.get("attribute_manifest", {})
    if attr.get("ingested_groups"):
        lines += ["## Attribute crop manifest",
                  f"- ingested groups: {', '.join(attr['ingested_groups'])}",
                  f"- split sizes: {attr['split_sizes']}",
                  f"- pending auto-ingest: {', '.join(attr['groups_pending_auto_ingest']) or '(none)'}",
                  ""]
    prepared = report.get("prepared", {})
    lines += ["## Prepared datasets",
              ("- present and strictly valid" if prepared.get("ok")
               else "- present but INVALID" if prepared.get("present")
               else "- not built yet"),
              ""]
    lines += ["## Vocabulary", f"- consistent across groups: "
              f"{report['vocabulary_consistent']}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--sample-images", type=int, default=5,
                        help="Source images sampled per group for existence checks")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the summary only; write no report files")
    args = parser.parse_args(argv)
    report = audit(args.config, args.sample_images)
    markdown = render_markdown(report)
    print(markdown)
    if args.dry_run:
        print("(dry run: no report files written)")
        return 0 if report["ready_for_final_preparation"] else 1
    repo_root = HERE.parents[2]
    out_dir = (repo_root / "Documents/Final-Reports/MTSD-Dataset-Audit"
               / f"audit-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "pipeline_state.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    (out_dir / "pipeline_state.md").write_text(markdown, encoding="utf-8")
    print(f"Reports written to: {out_dir}")
    return 0 if report["ready_for_final_preparation"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
