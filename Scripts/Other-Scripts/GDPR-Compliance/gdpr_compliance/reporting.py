"""Report writers: detection_report.json and detection_summary.csv."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import TOOL_NAME
from .detections import Detection

REPORT_NAME = "detection_report.json"
SUMMARY_NAME = "detection_summary.csv"


@dataclass
class ImageResult:
    """Everything the reports need to say about one processed image."""

    source: str                       # input image path (as given)
    relative_path: str                # path relative to the input root (posix)
    preview: str | None = None        # redacted copy, when one was written
    comparison: str | None = None     # before/after sheet, when one was written
    redaction_method: str | None = None
    detections_raw: list[Detection] = field(default_factory=list)
    detections_padded: list[Detection] = field(default_factory=list)
    error: str | None = None

    @property
    def had_detections(self) -> bool:
        return bool(self.detections_raw)


def write_reports(output_root: Path, results: list[ImageResult], *,
                  input_root: Path, backend: str, detector_names: list[str],
                  method: str) -> dict:
    """Write both report files and return the JSON payload."""
    flagged = sum(1 for r in results if r.had_detections)
    errors = sum(1 for r in results if r.error)

    payload = {
        "tool": TOOL_NAME,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "backend": backend,
        "detectors": detector_names,
        "redaction_method": method,
        "images_scanned": len(results),
        "images_flagged": flagged,
        "images_clean": len(results) - flagged - errors,
        "images_failed": errors,
        "total_detections": sum(len(r.detections_raw) for r in results),
        "images": [
            {
                "source": r.source,
                "preview": r.preview,
                "comparison": r.comparison,
                "had_detections": r.had_detections,
                "redaction_method": r.redaction_method if r.had_detections else None,
                "detections": [
                    {
                        "label": raw.label,
                        "confidence": round(float(raw.score), 4),
                        "detector": raw.detector,
                        "box": raw.box_dict(),
                        "box_padded": padded.box_dict(),
                    }
                    for raw, padded in zip(r.detections_raw, r.detections_padded)
                ],
                **({"error": r.error} if r.error else {}),
            }
            for r in results
            if r.had_detections or r.error
        ],
    }

    (output_root / REPORT_NAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    with open(output_root / SUMMARY_NAME, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "preview", "detections",
                         "faces", "licence_plates", "qr_codes", "other", "error"])
        for r in results:
            if not (r.had_detections or r.error):
                continue
            labels = [d.label for d in r.detections_raw]
            known = {"face", "licence_plate", "qr_code"}
            writer.writerow([
                r.relative_path,
                r.preview or "",
                len(labels),
                labels.count("face"),
                labels.count("licence_plate"),
                labels.count("qr_code"),
                sum(1 for l in labels if l not in known),
                r.error or "",
            ])
    return payload
