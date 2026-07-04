#!/usr/bin/env python3
"""MTSD GDPR Compliance - detect and redact privacy-sensitive image regions.

Detects human faces, vehicle licence plates, and QR codes, then blurs them.
**Originals are never modified by `preview`**: results go to a separate
preview tree that mirrors the input structure, alongside before/after
comparison sheets and a detection report for manual verification.

Detection backends:
  sam31    Meta SAM 3.1 open-vocabulary detection (needs the mtsd-base conda
           env; faces/plates/QR via text prompts on one shared model, plus
           OpenCV's exact QR detector)
  classic  dedicated detectors (YuNet faces, local plate ONNX, OpenCV QR)
  auto     sam31 when importable, otherwise classic (announced, never silent)

Typical workflow:

  1. Preview run (writes only flagged images + comparisons + report):
       conda activate mtsd-base
       python redact.py preview

  2. Inspect  <output>/_comparisons/  and  detection_summary.csv

  3. Only after manual verification, replace the originals (with backups):
       python redact.py apply

Examples:
  python redact.py preview --backend classic --method pixelate
  python redact.py preview --input ..\\..\\Datasets\\GRP-1 --limit 50
  python redact.py preview --sam-prompt "licence_plate=car number plate"
  python redact.py apply --yes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gdpr_compliance import pipeline, reporting
from gdpr_compliance.detectors import (BACKEND_DEFAULTS, DETECTOR_REGISTRY,
                                       build_detectors, resolve_backend)
from gdpr_compliance.detectors.sam31 import DEFAULT_CATEGORIES

# <repo>/Scripts/Other-Scripts/GDPR-Compliance/redact.py
BASE_DIR = SCRIPT_DIR.parents[2]
DEFAULT_INPUT = BASE_DIR / "Datasets" / "MTSD"
DEFAULT_OUTPUT = BASE_DIR / "GDPR-Compliance-Preview"


def _parse_sam_prompts(overrides: list[str] | None) -> dict:
    """Merge `label=prompt` CLI overrides into the default category table."""
    categories = dict(DEFAULT_CATEGORIES)
    for item in overrides or []:
        label, _, prompt = item.partition("=")
        label, prompt = label.strip(), prompt.strip()
        if not prompt or label not in categories:
            raise SystemExit(
                f"--sam-prompt expects label=prompt with label in "
                f"{sorted(categories)}; got '{item}'")
        old_prompt, pad, conf = categories[label]
        categories[label] = (prompt, pad, conf)
    return categories


def cmd_preview(args: argparse.Namespace) -> None:
    backend = resolve_backend(args.backend)
    detector_names = (
        [n.strip() for n in args.detectors.split(",") if n.strip()]
        if args.detectors else list(BACKEND_DEFAULTS[backend])
    )
    sam_options = {
        "version": args.sam_version,
        "checkpoint": args.sam_checkpoint,
        "device": args.device,
        "categories": _parse_sam_prompts(args.sam_prompt),
    }
    detectors = build_detectors(detector_names, sam_options=sam_options)
    if args.yolo_weights:
        from gdpr_compliance.detectors.yolo import YoloDetector
        detectors.append(YoloDetector(args.yolo_weights, label=args.yolo_label))

    print(f"Input     : {args.input}")
    print(f"Output    : {args.output}  (originals are NOT modified)")
    print(f"Backend   : {backend}")
    print(f"Detectors : {', '.join(d.name for d in detectors)}")
    print(f"Redaction : {args.method} (size-scaled)")

    report = pipeline.run_preview(
        input_root=args.input,
        output_root=args.output,
        detectors=detectors,
        backend=backend,
        method=args.method,
        make_comparisons=not args.no_comparisons,
        copy_clean=args.copy_clean,
        jpeg_quality=args.jpeg_quality,
        limit=args.limit,
    )

    print()
    print(f"Scanned   : {report['images_scanned']:,} images")
    print(f"Flagged   : {report['images_flagged']:,} images "
          f"({report['total_detections']:,} regions redacted)")
    print(f"Clean     : {report['images_clean']:,} images"
          + ("" if args.copy_clean else " (not copied - use --copy-clean for a full mirror)"))
    if report["images_failed"]:
        print(f"Failed    : {report['images_failed']:,} images (see {reporting.REPORT_NAME})")
    print()
    print("Review the results before touching any originals:")
    print(f"  comparisons : {args.output / pipeline.COMPARISONS_DIR_NAME}")
    print(f"  report      : {args.output / reporting.REPORT_NAME}")
    print(f"  summary     : {args.output / reporting.SUMMARY_NAME}")
    print("When satisfied, replace originals with:  python redact.py apply")


def cmd_apply(args: argparse.Namespace) -> None:
    report_path = args.output / reporting.REPORT_NAME
    if not report_path.exists():
        raise SystemExit(f"No {reporting.REPORT_NAME} found in {args.output} - "
                         "run the 'preview' step first.")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    print(f"About to REPLACE originals under: {args.input}")
    print(f"with redacted versions from     : {args.output}")
    print(f"(preview of {report.get('generated', '?')}, "
          f"{report.get('images_flagged', '?')} flagged images, "
          f"backend {report.get('backend', '?')})")
    print("Backups of replaced originals   : "
          + ("ON  -> " + str(args.output / pipeline.BACKUP_DIR_NAME)
             if not args.no_backup else "OFF"))

    if not args.yes:
        answer = input("\nType 'REPLACE' to proceed: ").strip()
        if answer != "REPLACE":
            print("Aborted - nothing was changed.")
            return

    replaced, missing = pipeline.apply_results(
        output_root=args.output, input_root=args.input, backup=not args.no_backup)
    print(f"\nReplaced  : {replaced:,} originals")
    if missing:
        print(f"Missing   : {missing:,} originals not found (renamed or moved?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    prev = sub.add_parser("preview",
                          help="Detect + redact into the preview directory (originals untouched).")
    prev.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                      help=f"Image tree to scan (default: {DEFAULT_INPUT})")
    prev.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                      help=f"Preview output tree (default: {DEFAULT_OUTPUT})")
    prev.add_argument("--backend", choices=["auto", "sam31", "classic"], default="auto",
                      help="Detection backend (default: auto = sam31 if importable)")
    prev.add_argument("--detectors", default=None,
                      help="Comma-separated detector names overriding the backend "
                           f"default set (available: {', '.join(sorted(DETECTOR_REGISTRY))})")
    prev.add_argument("--method", choices=["gaussian", "pixelate"], default="gaussian",
                      help="Redaction style (default: gaussian)")
    prev.add_argument("--jpeg-quality", type=int, default=95,
                      help="JPEG quality for redacted output (default: 95)")
    prev.add_argument("--limit", type=int, default=None,
                      help="Process only the first N images (for quick trials).")
    prev.add_argument("--copy-clean", action="store_true",
                      help="Also copy images without detections (full mirror).")
    prev.add_argument("--no-comparisons", action="store_true",
                      help="Skip before/after comparison sheets.")
    prev.add_argument("--device", default=None,
                      help="SAM 3.1 device override (default: cuda if available)")
    prev.add_argument("--sam-version", choices=["sam3.1", "sam3"], default="sam3.1",
                      help="SAM checkpoint version (default: sam3.1)")
    prev.add_argument("--sam-checkpoint", default=None,
                      help="Explicit SAM checkpoint path (default: HF cache download)")
    prev.add_argument("--sam-prompt", action="append", metavar="LABEL=PROMPT",
                      help="Override a SAM text prompt, e.g. "
                           '"licence_plate=car number plate" (repeatable)')
    prev.add_argument("--yolo-weights", type=Path, default=None,
                      help="Optional Ultralytics YOLO weights for an extra detector.")
    prev.add_argument("--yolo-label", default="yolo_region",
                      help="Label for YOLO detections (default: yolo_region)")
    prev.set_defaults(func=cmd_preview)

    app = sub.add_parser("apply",
                         help="Replace originals with verified redacted versions.")
    app.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                     help=f"Original image tree (default: {DEFAULT_INPUT})")
    app.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                     help=f"Preview tree produced by 'preview' (default: {DEFAULT_OUTPUT})")
    app.add_argument("--no-backup", action="store_true",
                     help="Do not back up originals before replacing them.")
    app.add_argument("--yes", action="store_true",
                     help="Skip the interactive confirmation prompt.")
    app.set_defaults(func=cmd_apply)

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.func(arguments)
