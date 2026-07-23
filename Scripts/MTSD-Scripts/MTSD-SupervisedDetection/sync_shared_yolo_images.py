#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mtsd_detection.manifests import sha256_file  # noqa: E402


DEFAULT_PREPARED = ROOT / "Datasets" / "MTSD" / "Prepared"
DEFAULT_REPORT_ROOT = ROOT / "Results" / "MTSD-Results" / "Dataset-QA"
SPLITS = ("train", "valid", "test")


def find_mismatches(source_variant: Path, target_variant: Path) -> list[dict[str, Any]]:
    rows = []
    for split in SPLITS:
        source_dir = source_variant / "MTSD-YOLO" / split / "images"
        target_dir = target_variant / "MTSD-YOLO" / split / "images"
        if not source_dir.is_dir() or not target_dir.is_dir():
            raise FileNotFoundError(f"Missing YOLO image directory for {split}")
        for target in sorted(path for path in target_dir.iterdir() if path.is_file()):
            source = source_dir / target.name
            if not source.is_file():
                raise FileNotFoundError(f"Shared source image is missing: {source}")
            source_hash = sha256_file(source)
            target_hash = sha256_file(target)
            if source_hash != target_hash:
                rows.append({
                    "split": split,
                    "name": target.name,
                    "source": str(source.resolve()),
                    "target": str(target.resolve()),
                    "source_sha256": source_hash,
                    "target_sha256_before": target_hash,
                    "source_bytes": source.stat().st_size,
                    "target_bytes_before": target.stat().st_size,
                })
    return rows


def apply_sync(rows: list[dict[str, Any]], target_variant: Path) -> None:
    allowed_root = (target_variant / "MTSD-YOLO").resolve()
    for row in rows:
        source = Path(row["source"])
        target = Path(row["target"])
        if not target.resolve().is_relative_to(allowed_root):
            raise PermissionError(f"Refusing target outside derived YOLO tree: {target}")
        if not source.is_file() or not target.is_file():
            raise FileNotFoundError(f"Sync endpoint disappeared: {source} -> {target}")
        if sha256_file(source) != row["source_sha256"] or sha256_file(target) != row["target_sha256_before"]:
            raise RuntimeError(f"Sync endpoint changed after audit: {target}")
        # Prepared YOLO files may be hard links to GDPR-safe sources. Unlinking
        # first guarantees the copy cannot modify another link's underlying data.
        target.unlink()
        shutil.copy2(source, target)
        if sha256_file(target) != row["source_sha256"]:
            raise RuntimeError(f"Post-copy hash mismatch: {target}")
        row["target_sha256_after"] = row["source_sha256"]
        row["target_bytes_after"] = target.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description="Synchronize shared YOLO originals after trainer JPEG repair")
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--source-variant", default="MTSD-Augmented")
    parser.add_argument("--target-variant", default="MTSD-Unaugmented")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    prepared = args.prepared_root.resolve()
    source_variant = (prepared / args.source_variant).resolve()
    target_variant = (prepared / args.target_variant).resolve()
    if not source_variant.is_relative_to(prepared) or not target_variant.is_relative_to(prepared):
        raise PermissionError("Variant paths must stay inside the prepared dataset root")
    mismatches = find_mismatches(source_variant, target_variant)
    before_count = len(mismatches)
    if args.apply:
        apply_sync(mismatches, target_variant)
    remaining = find_mismatches(source_variant, target_variant) if args.apply else mismatches
    report = {
        "status": "applied" if args.apply else "dry_run",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_variant": str(source_variant),
        "target_variant": str(target_variant),
        "mismatch_count_before": before_count,
        "mismatch_count_after": len(remaining),
        "source_policy": "Exact YOLO bytes used by the completed augmented baseline",
        "source_dataset_modified": False,
        "rows": mismatches,
    }
    report_path = args.report or (
        DEFAULT_REPORT_ROOT / f"shared-yolo-sync-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{'apply' if args.apply else 'dry-run'}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "source_variant", "target_variant",
                                                    "mismatch_count_before", "mismatch_count_after")}, indent=2))
    print(f"report: {report_path}")
    return 0 if not args.apply or not remaining else 1


if __name__ == "__main__":
    raise SystemExit(main())
