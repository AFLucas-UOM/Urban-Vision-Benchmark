#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mtsd_detection.config import load_config
from mtsd_detection.strong_augmentation_qa import finalize_report, run_strong_qa


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Independent, dataset-wide QA for MTSD strong offline augmentation",
    )
    result.add_argument("--config", type=Path, default=HERE / "config" / "default.yaml")
    result.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Datasets/MTSD/Prepared/MTSD-Augmented-Strong"),
    )
    result.add_argument(
        "--unaugmented-root",
        type=Path,
        default=Path("Datasets/MTSD/Prepared/MTSD-Unaugmented"),
    )
    result.add_argument("--output", type=Path)
    result.add_argument("--samples-per-category", type=int, default=25)
    result.add_argument("--risk-samples", type=int, default=25)
    result.add_argument("--finalize-report", type=Path)
    result.add_argument("--visual-decision", choices=("pass", "fail"))
    result.add_argument("--tests-passed", action="store_true")
    result.add_argument("--test-report", type=Path)
    result.add_argument("--review-notes", default="")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.finalize_report:
        if args.visual_decision is None:
            raise ValueError("--finalize-report requires --visual-decision")
        report = finalize_report(
            args.finalize_report,
            args.visual_decision,
            args.tests_passed,
            args.test_report,
            args.review_notes,
        )
    else:
        if args.output is None:
            raise ValueError("--output is required for a new immutable QA run")
        if args.samples_per_category < 25 or args.risk_samples < 25:
            raise ValueError("Final QA requires at least 25 samples per category and 25 risk samples")
        report = run_strong_qa(
            args.dataset_root,
            args.unaugmented_root,
            args.output,
            load_config(args.config),
            samples_per_category=args.samples_per_category,
            risk_samples=args.risk_samples,
        )
    print(json.dumps({
        "report": str((args.finalize_report or args.output / "qa_report.json").resolve()),
        "automated_status": report["automated_status"],
        "final_status": report["final_status"],
        "automated_error_count": report["automated_error_count"],
        "visual_qa": report["visual_qa"],
    }, indent=2))
    return 0 if report["final_status"] in {"pass", "pending_visual_review"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
