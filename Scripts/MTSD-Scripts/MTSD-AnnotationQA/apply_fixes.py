#!/usr/bin/env python3
"""Apply reviewed annotation QA decisions to the MTSD QA JSON files.

This is the ONLY component of the QA workflow that modifies annotation files,
and it is deliberately guarded:

* dry-run is the default - nothing is written without ``--apply``;
* ``--apply`` asks for interactive confirmation (skip with ``--yes``);
* every file is backed up to ``<name>.json.pre-qa-fix-<timestamp>.bak`` first;
* only the targeted fields change: ``set_attribute`` touches exactly one
  value inside one annotation's ``attributes``; ``delete_annotation`` removes
  exactly one annotation object (never its image entry);
* if the stored ``old_value`` no longer matches the file, the decision is
  skipped and reported (the file changed since the review);
* the rewritten file is re-parsed and its object counts re-checked before the
  original is replaced;
* an ``applied_changes-<timestamp>.json`` log records exactly what was done.

Note: files are re-serialised with ``indent=2``, so the first apply may
reformat whitespace produced by earlier export tools. Content is preserved
verbatim (same keys, same order, same values apart from the applied fixes)
and the pre-edit file remains in the ``.bak`` backup and in git history.

Usage:
    python apply_fixes.py --decisions outputs/audit-<ts>/reviewed_decisions.json --dry-run
    python apply_fixes.py --decisions outputs/audit-<ts>/reviewed_decisions.json --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config  # noqa: E402

ACTIONABLE = {"set_attribute", "delete_annotation"}


def load_decisions(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    decisions = payload.get("decisions", payload if isinstance(payload, list) else [])
    actionable = [d for d in decisions if d.get("type") in ACTIONABLE]
    informational = len(decisions) - len(actionable)
    print(f"Loaded {len(decisions)} decisions ({len(actionable)} actionable, "
          f"{informational} informational e.g. keep-both).")
    return actionable


def plan_changes(decisions: list[dict]) -> dict[str, list[dict]]:
    by_file: dict[str, list[dict]] = defaultdict(list)
    for decision in decisions:
        by_file[decision["qa_file"]].append(decision)
    return by_file


def annotation_by_id(data: dict, annotation_id) -> dict | None:
    for annotation in data.get("annotations", []):
        if annotation.get("id") == annotation_id or str(annotation.get("id")) == str(annotation_id):
            return annotation
    return None


def apply_to_file(qa_rel: str, decisions: list[dict], apply: bool, stamp: str) -> list[dict]:
    """Return a change log for one QA file; writes only when apply=True."""
    qa_path = config.BASE_DIR / qa_rel
    log: list[dict] = []
    if not qa_path.exists():
        print(f"  !! {qa_rel}: file not found - {len(decisions)} decision(s) skipped")
        return [{"qa_file": qa_rel, "status": "file_missing", "skipped": len(decisions)}]

    data = json.loads(qa_path.read_text(encoding="utf-8-sig"))
    n_annotations_before = len(data.get("annotations", []))
    n_images_before = len(data.get("images", []))
    planned_deletes = 0

    for decision in decisions:
        annotation = annotation_by_id(data, decision["annotation_id"])
        entry = {"qa_file": qa_rel, **{k: decision[k] for k in ("type", "annotation_id") }}
        if annotation is None:
            entry.update(status="skipped_annotation_not_found")
            print(f"  !! annotation {decision['annotation_id']}: not found - skipped")
        elif decision["type"] == "set_attribute":
            attribute = decision["attribute"]
            attrs = annotation.setdefault("attributes", {})
            current = attrs.get(attribute)
            expected_old = decision.get("old_value")
            if expected_old not in (None, "", "<key absent>") and str(current) != str(expected_old):
                entry.update(status="skipped_value_drifted", expected=expected_old, found=current)
                print(f"  !! annotation {decision['annotation_id']}.{attribute}: file has "
                      f"'{current}', review saw '{expected_old}' - skipped (re-audit)")
            else:
                attrs[attribute] = decision["new_value"]
                entry.update(status="applied" if apply else "would_apply",
                             attribute=attribute, old=current, new=decision["new_value"])
                print(f"  {'*' if apply else '-'} annotation {decision['annotation_id']}: "
                      f"{attribute}: '{current}' -> '{decision['new_value']}'")
        elif decision["type"] == "delete_annotation":
            data["annotations"] = [
                a for a in data["annotations"]
                if not (a.get("id") == annotation.get("id"))
            ]
            planned_deletes += 1
            entry.update(status="applied" if apply else "would_apply",
                         reason=decision.get("reason", ""))
            print(f"  {'*' if apply else '-'} annotation {decision['annotation_id']}: DELETED "
                  f"({decision.get('reason', 'no reason recorded')})")
        log.append(entry)

    if apply and any(e["status"] == "applied" for e in log):
        # sanity: images untouched, deletions accounted for
        assert len(data.get("images", [])) == n_images_before, "image entries must never change"
        expected = n_annotations_before - planned_deletes
        actual = len(data.get("annotations", []))
        if actual != expected:
            raise RuntimeError(
                f"{qa_rel}: annotation count mismatch after edits "
                f"(expected {expected}, found {actual}) - file NOT written."
            )
        backup = qa_path.with_name(qa_path.name + f".pre-qa-fix-{stamp}.bak")
        if not backup.exists():
            shutil.copy2(qa_path, backup)
        serialised = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        json.loads(serialised)  # must re-parse before we overwrite anything
        qa_path.write_text(serialised, encoding="utf-8")
        print(f"  => wrote {qa_rel} (backup: {backup.name})")
    return log


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply reviewed MTSD annotation fixes (guarded).")
    parser.add_argument("--decisions", type=Path, required=True,
                        help="reviewed_decisions.json produced by review_app.py "
                             "(relative paths are resolved against outputs/).")
    parser.add_argument("--apply", action="store_true",
                        help="Actually modify the QA files (default: dry-run).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Explicit dry-run (same as omitting --apply).")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the interactive confirmation when using --apply.")
    args = parser.parse_args()

    decisions_path = args.decisions
    if not decisions_path.exists() and (config.OUTPUTS_DIR / decisions_path).exists():
        decisions_path = config.OUTPUTS_DIR / decisions_path
    if not decisions_path.exists():
        print(f"Decisions file not found: {args.decisions}")
        return 1

    apply = args.apply and not args.dry_run
    decisions = load_decisions(decisions_path)
    if not decisions:
        print("Nothing actionable to apply.")
        return 0
    by_file = plan_changes(decisions)
    print(f"Mode: {'APPLY' if apply else 'DRY-RUN (no files will be modified)'}")
    print(f"Files affected: {len(by_file)}")

    if apply and not args.yes:
        answer = input(
            f"\nAbout to modify {len(by_file)} annotation file(s) "
            f"({len(decisions)} change(s)); timestamped .bak backups will be created.\n"
            f"Type YES to continue: "
        ).strip()
        if answer != "YES":
            print("Aborted - nothing was modified.")
            return 1

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    full_log: list[dict] = []
    for qa_rel, file_decisions in sorted(by_file.items()):
        print(f"\n{qa_rel} ({len(file_decisions)} decision(s)):")
        full_log.extend(apply_to_file(qa_rel, file_decisions, apply, stamp))

    applied = sum(1 for e in full_log if e.get("status") == "applied")
    would = sum(1 for e in full_log if e.get("status") == "would_apply")
    skipped = sum(1 for e in full_log if str(e.get("status", "")).startswith("skipped"))
    print(f"\nSummary: {applied} applied, {would} would apply (dry-run), {skipped} skipped.")

    log_path = decisions_path.parent / f"applied_changes-{stamp}{'':s}.json"
    log_path.write_text(json.dumps({
        "mode": "apply" if apply else "dry-run",
        "decisions_file": str(decisions_path),
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "changes": full_log,
    }, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Change log: {log_path}")
    if applied:
        print("\nReminder: the AttributeClassification manifest stores QA SHA-256 hashes - "
              "regenerate it before the next training round "
              "(python -m mtsd_attr.data_manifest).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
