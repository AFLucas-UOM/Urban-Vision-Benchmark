#!/usr/bin/env python3
"""Refresh the MTSD QA gate and approved Final-QA group scope from disk.

Discovers which MTSD groups currently have a valid Final-QA annotation JSON,
then refreshes two files so nobody has to hand-edit them when a new group
gains QA approval:

  * config/default.yaml  (annotations.approved_groups)
  * config/qa_gate.yaml  (audit metadata + resolution_status)

Discovery is independent of the group list already written in either file -
the whole point is to remove manual editing. The resulting scope is written
back as an explicit, locked list (group_scope itself is never touched: this
script always produces a locked, reproducible scope for whatever value
group_scope already has).

This script never runs dataset preparation, training or PromptDetect
evaluation, never launches the annotation-QA audit itself, and never creates
backup copies of the YAML files it rewrites.

Usage (from the repository root or from this directory):
    python update_qa_gate.py --dry-run
    python update_qa_gate.py --apply
Running it with neither flag behaves like --dry-run.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mtsd_detection.annotation_sources import _validate_categories  # noqa: E402
from mtsd_detection.utils import find_repo_root, sha256_file  # noqa: E402

DEFAULT_CONFIG_PATH = HERE / "config" / "default.yaml"
DEFAULT_QA_GATE_PATH = HERE / "config" / "qa_gate.yaml"
DEFAULT_ANNOTATIONS_ROOT_REL = "Datasets/MTSD/Annotations"
DEFAULT_AUDIT_ROOT_REL = "Scripts/MTSD-Scripts/MTSD-AnnotationQA/outputs"
AUDIT_SUMMARY_NAME = "audit_summary.json"
REVIEWED_DECISIONS_NAME = "reviewed_decisions.json"
BACKUP_MARKERS = (".bak", "pre-migration", "pre-qa-fix", "backup", ".orig", "_old")


def _group_number(name: str) -> int:
    return int(name.rsplit("-", 1)[-1])


def _is_backup_name(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in BACKUP_MARKERS)


def _resolve(value: Path | None, default: Path, repo_root: Path) -> Path:
    candidate = value if value is not None else default
    return candidate if candidate.is_absolute() else (repo_root / candidate)


def _relativize(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _read_annotations_config(path: Path) -> dict[str, Any]:
    """Tolerant read of just what this script needs from default.yaml.

    Deliberately lighter than mtsd_detection.config.load_config: that loader
    requires the full training schema and a non-empty approved_groups for
    explicit scope, which would block the one thing this script exists to
    do - populate approved_groups for the first time. Malformed YAML syntax
    or a missing annotations/group_scope section is still treated as a
    structural configuration error.
    """
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"could not parse YAML: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("annotations"), dict):
        raise ValueError("missing top-level 'annotations' section")
    if "group_scope" not in data["annotations"]:
        raise ValueError("annotations section is missing 'group_scope'")
    return data


# --- Final-QA group discovery ---------------------------------------------------

def _load_qa_payload(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Soft validation: returns (payload, None) if usable, (None, reason) if not.

    Raises ValueError only for the canonical-vocabulary check (duplicate
    category IDs/names, or a class list that disagrees with the canonical
    12-class MTSD vocabulary) - a data-integrity problem serious enough that
    the real pipeline (mtsd_detection.annotation_sources.discover_sources)
    also aborts on it rather than silently excluding the group.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return None, f"invalid JSON ({exc})"
    if not isinstance(payload, dict):
        return None, "QA file is not a JSON object"
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            return None, f"missing or non-list field {key!r}"
    categories = payload["categories"]
    try:
        ids = [int(item["id"]) for item in categories]
        names = [str(item["name"]) for item in categories]
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"malformed categories entry ({exc})"
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate category IDs in categories block")
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate category names in categories block")
    _validate_categories(categories, path)  # raises ValueError on vocabulary mismatch
    return payload, None


def discover_final_qa_groups(annotations_root: Path) -> dict[str, dict]:
    """Returns {"valid": {group: {path, sha256, images, boxes}}, "invalid": {group: reason}}.

    Raises ValueError for structural problems (multiple active Final-QA JSONs
    for one group, or a canonical class-vocabulary mismatch) rather than
    silently excluding the group.
    """
    valid: dict[str, dict] = {}
    invalid: dict[str, str] = {}
    group_dirs = sorted(
        (p for p in annotations_root.glob("GRP-*") if p.is_dir()),
        key=lambda p: _group_number(p.name),
    )
    for group_dir in group_dirs:
        qa_dir = group_dir / "Final-QA"
        if not qa_dir.is_dir():
            continue
        candidates = sorted(p for p in qa_dir.glob("QA-GRP*.json") if not _is_backup_name(p.name))
        if not candidates:
            continue
        if len(candidates) > 1:
            raise ValueError(
                f"{group_dir.name}: multiple active Final-QA JSON files found: "
                f"{[c.name for c in candidates]}"
            )
        path = candidates[0]
        payload, reason = _load_qa_payload(path)
        if payload is None:
            invalid[group_dir.name] = reason
            continue
        valid[group_dir.name] = {
            "path": path, "sha256": sha256_file(path),
            "images": len(payload["images"]), "boxes": len(payload["annotations"]),
        }
    return {"valid": valid, "invalid": invalid}


# --- newest annotation-QA audit -------------------------------------------------

def discover_latest_audit(audit_root: Path) -> dict[str, Any]:
    """Finds the newest audit-*/audit_summary.json by its own generated_at field.

    Falls back to directory mtime only for an entry whose summary lacks a
    parseable generated_at. Raises ValueError if an audit_summary.json exists
    but cannot be parsed - such a file claims to be the authoritative record
    for that audit run, so silently skipping it would risk picking a stale
    audit without saying so.
    """
    if not audit_root.is_dir():
        return {"found": False, "path": None, "timestamp": None,
                "audited_groups": [], "finding_counts": {}, "review": {}}
    candidates: list[tuple[datetime, Path, dict[str, Any]]] = []
    for entry in sorted(p for p in audit_root.glob("audit-*") if p.is_dir()):
        summary_path = entry / AUDIT_SUMMARY_NAME
        if not summary_path.is_file():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Audit summary is unparseable: {summary_path} ({exc})") from exc
        generated_at = payload.get("generated_at")
        sort_key: datetime | None = None
        if generated_at:
            try:
                sort_key = datetime.fromisoformat(str(generated_at))
            except ValueError:
                sort_key = None
        if sort_key is None:
            sort_key = datetime.fromtimestamp(summary_path.stat().st_mtime)
        candidates.append((sort_key, entry, payload))
    if not candidates:
        return {"found": False, "path": None, "timestamp": None,
                "audited_groups": [], "finding_counts": {}, "review": {}}
    candidates.sort(key=lambda item: item[0])
    _, entry, payload = candidates[-1]
    audited_groups = sorted(
        {m.group(0) for qa_path in payload.get("qa_files", [])
         for m in [re.search(r"GRP-\d+", str(qa_path).replace("\\", "/"))] if m},
        key=_group_number,
    )
    totals = payload.get("totals", {})
    raw_duplicate_count = totals.get("duplicate_pair_findings")
    reviewed_duplicate_count = count_reviewed_duplicate_candidates(entry)
    if raw_duplicate_count is None:
        unresolved_duplicate_count = None
    else:
        unresolved_duplicate_count = max(0, int(raw_duplicate_count) - reviewed_duplicate_count)
    finding_counts = {
        "invalid_attribute_values": totals.get("invalid_attribute_findings"),
        "duplicate_candidates": unresolved_duplicate_count,
        "missing_attributes": totals.get("missing_attribute_findings"),
        "reference_problems": totals.get("reference_problem_findings"),
    }
    return {"found": True, "path": entry, "timestamp": payload.get("generated_at"),
            "audited_groups": audited_groups, "finding_counts": finding_counts,
            "review": {
                "raw_duplicate_candidates": raw_duplicate_count,
                "reviewed_duplicate_candidates": reviewed_duplicate_count,
                "unresolved_duplicate_candidates": unresolved_duplicate_count,
            }}


def count_reviewed_duplicate_candidates(audit_dir: Path) -> int:
    """Count duplicate-candidate rows explicitly reviewed as keep-both."""
    duplicates_path = audit_dir / "duplicate_candidates.csv"
    decisions_path = audit_dir / REVIEWED_DECISIONS_NAME
    if not duplicates_path.is_file() or not decisions_path.is_file():
        return 0
    try:
        import csv

        with duplicates_path.open(encoding="utf-8", newline="") as handle:
            duplicate_pairs = {
                (
                    row["qa_file"],
                    frozenset({str(row["annotation_id_a"]), str(row["annotation_id_b"])}),
                )
                for row in csv.DictReader(handle)
            }
        decisions = json.loads(decisions_path.read_text(encoding="utf-8")).get("decisions", [])
    except Exception:
        return 0
    reviewed_pairs = {
        (
            decision.get("qa_file"),
            frozenset({
                str(decision.get("annotation_id_a")),
                str(decision.get("annotation_id_b")),
            }),
        )
        for decision in decisions
        if decision.get("type") == "keep_both"
    }
    return len(duplicate_pairs & reviewed_pairs)


# --- resolution status + acknowledgement ----------------------------------------

def compute_unaudited(proposed_scope: list[str], audit: dict[str, Any]) -> list[str]:
    if not audit["found"]:
        return list(proposed_scope)
    return [g for g in proposed_scope if g not in audit["audited_groups"]]


def compute_outstanding(audit: dict[str, Any]) -> dict[str, int | None]:
    if not audit["found"]:
        return {}
    return {k: v for k, v in audit["finding_counts"].items() if v is None or v > 0}


_FINDING_LABELS = {
    "invalid_attribute_values": "invalid attribute value",
    "duplicate_candidates": "duplicate candidate",
    "missing_attributes": "missing attribute",
    "reference_problems": "reference problem",
}


def build_acknowledgement(audit: dict[str, Any], unaudited: list[str], outstanding: dict) -> str:
    if not audit["found"]:
        return ("No usable annotation-QA audit was found; the QA gate remains "
                "unresolved until an audit runs and this refresh is re-applied.")
    if unaudited:
        names = ", ".join(unaudited)
        plural = len(unaudited) > 1
        has_have = "have" if plural else "has"
        is_are = "are" if plural else "is"
        return f"{names} {has_have} a valid Final-QA JSON but {is_are} not included in the latest annotation audit."
    if outstanding:
        pieces = []
        for key, value in outstanding.items():
            label = _FINDING_LABELS.get(key, key.replace("_", " "))
            if value is None:
                pieces.append(f"an unknown number of {label}s")
            else:
                pieces.append(f"{value} {label}{'' if value == 1 else 's'}")
        joined = pieces[0] if len(pieces) == 1 else ", ".join(pieces[:-1]) + f", and {pieces[-1]}"
        return f"Latest audit covers all approved groups, but {joined} remain unresolved."
    return "Latest audit covers all approved Final-QA groups and reports no unresolved critical findings."


# --- YAML rendering / patching (no backups, minimal-diff writes) ----------------

def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    dumped = yaml.safe_dump({"k": value}, default_flow_style=False).strip()
    return dumped[len("k: "):]


def _flow_list(items: list[str]) -> str:
    return yaml.safe_dump(list(items), default_flow_style=True, explicit_end=False).strip()


def render_qa_gate(gate: dict[str, Any]) -> str:
    lines = [
        f"gate_version: {_scalar(gate['gate_version'])}",
        f"audit_path: {_scalar(gate['audit_path'])}",
        f"audit_timestamp: {_scalar(gate['audit_timestamp'])}",
        f"audited_groups: {_flow_list(gate['audited_groups'])}",
        f"approved_scope: {_flow_list(gate['approved_scope'])}",
        "finding_counts:",
        f"  invalid_attribute_values: {_scalar(gate['finding_counts']['invalid_attribute_values'])}",
        f"  duplicate_candidates: {_scalar(gate['finding_counts']['duplicate_candidates'])}",
        f"resolution_status: {_scalar(gate['resolution_status'])}",
        f"approved_by: {_scalar(gate['approved_by'])}",
        f"acknowledgement: {_scalar(gate['acknowledgement'])}",
    ]
    return "\n".join(lines) + "\n"


def patch_approved_groups(raw_text: str, groups: list[str]) -> str:
    """Replaces only `approved_groups: [...]` inside the `annotations:` block."""
    section_pat = re.compile(r"^annotations:\s*$", re.MULTILINE)
    match = section_pat.search(raw_text)
    if not match:
        raise ValueError("default.yaml has no top-level 'annotations:' section")
    start = match.end()
    next_top = re.search(r"^\S", raw_text[start:], re.MULTILINE)
    end = start + next_top.start() if next_top else len(raw_text)
    block = raw_text[start:end]
    key_pat = re.compile(r"^(\s*)approved_groups:\s*\[[^\]]*\]\s*$", re.MULTILINE)
    if not key_pat.search(block):
        raise ValueError("default.yaml annotations section has no flow-style 'approved_groups: [...]' field")
    new_block = key_pat.sub(lambda m: f"{m.group(1)}approved_groups: {_flow_list(groups)}", block, count=1)
    return raw_text[:start] + new_block + raw_text[end:]


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(tmp_name, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.remove(tmp_name)
        raise


# --- report assembly + CLI -------------------------------------------------------

def build_report(*, mode: str, discovery: dict, current_scope: list[str], proposed_scope: list[str],
                 group_scope_value: Any, audit: dict, unaudited: list[str], outstanding: dict,
                 resolution_status: str, acknowledgement: str, files_to_modify: list[str],
                 gate: dict, repo_root: Path) -> dict:
    return {
        "mode": mode,
        "valid_final_qa_groups": [
            {"group": g, "images": meta["images"], "boxes": meta["boxes"]}
            for g, meta in sorted(discovery["valid"].items(), key=lambda kv: _group_number(kv[0]))
        ],
        "invalid_final_qa_groups": [
            {"group": g, "reason": reason}
            for g, reason in sorted(discovery["invalid"].items(), key=lambda kv: _group_number(kv[0]))
        ],
        "current_approved_groups": current_scope,
        "proposed_approved_groups": proposed_scope,
        "newly_added_groups": sorted(set(proposed_scope) - set(current_scope), key=_group_number),
        "removed_groups": sorted(set(current_scope) - set(proposed_scope), key=_group_number),
        "group_scope_config_value": group_scope_value,
        "audit": {
            "found": audit["found"],
            "path": _relativize(audit["path"], repo_root) if audit["found"] else None,
            "timestamp": audit["timestamp"],
            "audited_groups": audit["audited_groups"],
            "review": audit.get("review", {}),
        },
        "approved_but_unaudited_groups": unaudited,
        "invalid_attribute_value_count": gate["finding_counts"]["invalid_attribute_values"],
        "duplicate_candidate_count": gate["finding_counts"]["duplicate_candidates"],
        "other_unresolved_findings": {k: v for k, v in outstanding.items()
                                      if k not in ("invalid_attribute_values", "duplicate_candidates")},
        "proposed_resolution_status": resolution_status,
        "proposed_acknowledgement": acknowledgement,
        "files_that_would_be_modified": files_to_modify,
    }


def _print_report(report: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2, default=str))
        return
    print(f"MTSD QA gate refresh - {report['mode']}\n")
    print("Valid Final-QA groups discovered:")
    for row in report["valid_final_qa_groups"]:
        print(f"  {row['group']}: {row['images']} images, {row['boxes']} boxes")
    if not report["valid_final_qa_groups"]:
        print("  (none)")
    print("\nInvalid Final-QA groups (excluded):")
    for row in report["invalid_final_qa_groups"]:
        print(f"  {row['group']}: {row['reason']}")
    if not report["invalid_final_qa_groups"]:
        print("  (none)")
    print(f"\nCurrent approved_groups : {report['current_approved_groups']}")
    print(f"Proposed approved_groups: {report['proposed_approved_groups']}")
    print(f"Newly added groups      : {report['newly_added_groups'] or '(none)'}")
    print(f"Removed groups          : {report['removed_groups'] or '(none)'}")
    audit = report["audit"]
    print(f"\nNewest audit found : {audit['found']}")
    print(f"Audit path         : {audit['path']}")
    print(f"Audit timestamp    : {audit['timestamp']}")
    print(f"Audited groups     : {audit['audited_groups']}")
    review = audit.get("review") or {}
    if review:
        print(f"Raw duplicate candidates     : {review.get('raw_duplicate_candidates')}")
        print(f"Reviewed duplicate candidates: {review.get('reviewed_duplicate_candidates')}")
    print(f"Approved but unaudited groups: {report['approved_but_unaudited_groups'] or '(none)'}")
    print(f"Invalid attribute value count: {report['invalid_attribute_value_count']}")
    print(f"Duplicate candidate count    : {report['duplicate_candidate_count']}")
    if report["other_unresolved_findings"]:
        print(f"Other unresolved findings    : {report['other_unresolved_findings']}")
    print(f"\nProposed resolution_status: {report['proposed_resolution_status']}")
    print(f"Proposed acknowledgement  : {report['proposed_acknowledgement']}")
    files = report.get("files_written", report["files_that_would_be_modified"])
    label = "Files written" if "files_written" in report else "Files that would be modified"
    print(f"\n{label}: {files or '(none)'}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh the MTSD QA gate and approved Final-QA group scope from disk "
                    "(config/default.yaml + config/qa_gate.yaml). Never runs preparation, "
                    "training or PromptDetect evaluation, and never launches the audit itself."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="Print the proposed refresh; write nothing (default behaviour).")
    mode.add_argument("--apply", action="store_true",
                      help="Write the discovered scope/gate to both YAML files.")
    parser.add_argument("--annotations-root", type=Path, default=None,
                        help=f"Override the Final-QA search root (default: {DEFAULT_ANNOTATIONS_ROOT_REL}).")
    parser.add_argument("--audit-root", type=Path, default=None,
                        help=f"Override the annotation-QA audit output root (default: {DEFAULT_AUDIT_ROOT_REL}).")
    parser.add_argument("--default-config", type=Path, default=None,
                        help=f"Override the default.yaml path (default: {DEFAULT_CONFIG_PATH}).")
    parser.add_argument("--qa-gate", type=Path, default=None,
                        help=f"Override the qa_gate.yaml path (default: {DEFAULT_QA_GATE_PATH}).")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text.")
    return parser


def _fail(message: str, as_json: bool) -> int:
    if as_json:
        print(json.dumps({"error": message}, indent=2))
    else:
        print(f"ERROR: {message}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    apply_changes = bool(args.apply)

    repo_root = find_repo_root(HERE)
    default_config_path = _resolve(args.default_config, DEFAULT_CONFIG_PATH, repo_root)
    annotations_root = _resolve(args.annotations_root, repo_root / DEFAULT_ANNOTATIONS_ROOT_REL, repo_root)
    audit_root = _resolve(args.audit_root, repo_root / DEFAULT_AUDIT_ROOT_REL, repo_root)

    try:
        config = _read_annotations_config(default_config_path)
    except FileNotFoundError as exc:
        return _fail(f"Configuration file not found: {exc}", args.json)
    except ValueError as exc:
        return _fail(f"Malformed configuration YAML ({default_config_path}): {exc}", args.json)
    raw_default_text = default_config_path.read_text(encoding="utf-8")
    current_scope = list(config["annotations"].get("approved_groups") or [])
    group_scope_value = config["annotations"].get("group_scope")

    default_qa_gate = repo_root / config["annotations"].get("qa_gate_file", "") \
        if config["annotations"].get("qa_gate_file") else DEFAULT_QA_GATE_PATH
    qa_gate_path = _resolve(args.qa_gate, default_qa_gate, repo_root)

    try:
        discovery = discover_final_qa_groups(annotations_root)
    except ValueError as exc:
        return _fail(f"Structural annotation error: {exc}", args.json)

    try:
        audit = discover_latest_audit(audit_root)
    except ValueError as exc:
        return _fail(f"Unusable annotation-QA audit: {exc}", args.json)

    proposed_scope = sorted(discovery["valid"], key=_group_number)
    if apply_changes and not proposed_scope:
        return _fail(
            f"Refusing to apply: no valid Final-QA groups were discovered under "
            f"{annotations_root}; approved_groups would become empty.", args.json,
        )

    unaudited = compute_unaudited(proposed_scope, audit)
    outstanding = compute_outstanding(audit)
    resolved = bool(audit["found"] and not unaudited and not outstanding)
    resolution_status = "resolved" if resolved else "unresolved"
    acknowledgement = build_acknowledgement(audit, unaudited, outstanding)

    gate = {
        "gate_version": "qa-gate-v1",
        "audit_path": _relativize(audit["path"], repo_root) if audit["found"] else None,
        "audit_timestamp": audit["timestamp"],
        "audited_groups": audit["audited_groups"],
        "approved_scope": proposed_scope,
        "finding_counts": {
            "invalid_attribute_values": audit["finding_counts"].get("invalid_attribute_values") if audit["found"] else None,
            "duplicate_candidates": audit["finding_counts"].get("duplicate_candidates") if audit["found"] else None,
        },
        "resolution_status": resolution_status,
        "approved_by": None,
        "acknowledgement": acknowledgement,
    }

    try:
        new_default_text = patch_approved_groups(raw_default_text, proposed_scope)
        parsed_check = yaml.safe_load(new_default_text)
        if parsed_check.get("annotations", {}).get("approved_groups") != proposed_scope:
            raise ValueError("patched default.yaml failed self-check")
    except ValueError as exc:
        return _fail(f"Could not safely patch default.yaml: {exc}", args.json)
    new_gate_text = render_qa_gate(gate)

    current_gate_text = qa_gate_path.read_text(encoding="utf-8") if qa_gate_path.is_file() else None
    files_to_modify = []
    if new_default_text != raw_default_text:
        files_to_modify.append(str(default_config_path))
    if current_gate_text != new_gate_text:
        files_to_modify.append(str(qa_gate_path))

    report = build_report(
        mode="apply" if apply_changes else "dry-run", discovery=discovery, current_scope=current_scope,
        proposed_scope=proposed_scope, group_scope_value=group_scope_value, audit=audit, unaudited=unaudited,
        outstanding=outstanding, resolution_status=resolution_status, acknowledgement=acknowledgement,
        files_to_modify=files_to_modify, gate=gate, repo_root=repo_root,
    )

    if not apply_changes:
        _print_report(report, args.json)
        return 0

    if files_to_modify:
        _atomic_write_text(default_config_path, new_default_text)
        _atomic_write_text(qa_gate_path, new_gate_text)

    try:
        reloaded_default = _read_annotations_config(default_config_path)
    except ValueError as exc:
        return _fail(f"Post-write validation failed: default.yaml did not parse ({exc})", args.json)
    try:
        reloaded_gate = yaml.safe_load(qa_gate_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _fail(f"Post-write validation failed: qa_gate.yaml did not parse ({exc})", args.json)
    if reloaded_default["annotations"]["approved_groups"] != proposed_scope:
        return _fail("Post-write validation failed: default.yaml approved_groups mismatch", args.json)
    if reloaded_gate.get("approved_scope") != proposed_scope:
        return _fail("Post-write validation failed: qa_gate.yaml approved_scope mismatch", args.json)

    report["files_written"] = files_to_modify
    _print_report(report, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
