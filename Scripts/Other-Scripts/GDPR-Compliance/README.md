# MTSD GDPR Compliance

Privacy redaction for dataset images: detects and blurs regions that could
identify people or vehicles before the images are used further.

Sensitive categories:

- **human faces**
- **vehicle licence plates**
- **QR codes**

## Detection backends

| Backend | What runs | Environment |
| --- | --- | --- |
| `sam31` | **Meta SAM 3.1** open-vocabulary detection — faces, plates, and QR codes via text prompts on one shared model — plus OpenCV's exact QR decoder | `mtsd-base` conda env (torch + `sam3` package, CUDA) |
| `classic` | Dedicated per-category detectors: YuNet face ONNX, local plate ONNX (`models/number_plate_model_2025.onnx`, OpenCV-DNN), OpenCV multi-QR | any env with `opencv-python` |
| `auto` (default) | `sam31` when the `sam3` package is importable, otherwise `classic` — the fallback is announced, never silent | — |

SAM 3.1 specifics (deliberately explicit):

- The model builds through Meta's official `sam3` package
  (`build_sam3_image_model` + `Sam3Processor.set_text_prompt`); the
  `sam3.1_multiplex.pt` checkpoint auto-downloads from the **gated**
  `facebook/sam3.1` Hugging Face repo (access already granted on this
  machine; override with `--sam-checkpoint <path>`).
- The model is loaded **once per run** and reused; each image is encoded
  once and prompted once per category.
- Prompted concept detection is open-vocabulary: recall depends on the
  prompt (defaults: "human face", "vehicle licence plate", "QR code
  sticker"; override with `--sam-prompt label=prompt`). **QR codes are the
  weakest category for concept prompting**, so the default `sam31` set also
  runs OpenCV's exact QR detector — this is a documented design decision,
  not a silent fallback.
- Per-category confidence thresholds are defined in
  `gdpr_compliance/detectors/sam31.py` (`DEFAULT_CATEGORIES`).

The detector set is **modular**: subclass `BaseDetector`
([`gdpr_compliance/detectors/base.py`](gdpr_compliance/detectors/base.py)),
register it in `DETECTOR_REGISTRY`
([`gdpr_compliance/detectors/__init__.py`](gdpr_compliance/detectors/__init__.py)),
and it becomes selectable via `--detectors`. A ready-made `YoloDetector`
wrapper accepts any Ultralytics YOLO weights (`--yolo-weights plates.pt`).

## Safety model

**Originals are never modified by `preview`.** Results go to a separate tree
that mirrors the input structure:

```text
GDPR-Compliance-Preview/
  GRP-1/Images/<name>.jpeg               # redacted copy (EXIF kept, orientation baked)
  _comparisons/GRP-1/Images/<name>.jpg   # before (boxes + confidence) | after sheet
  detection_report.json                  # per-detection: label, confidence, detector,
                                         #   box before AND after padding, method, errors
  detection_summary.csv                  # one row per flagged image, for triage
```

Only the separate `apply` command touches originals. It:

- requires typing `REPLACE` (skip with `--yes`),
- backs up every replaced original to
  `GDPR-Compliance-Preview/_replaced_originals/` (disable with `--no-backup`).

## Usage

```powershell
# 1. Preview over the whole dataset with SAM 3.1 (originals untouched)
conda activate mtsd-base
python .\Scripts\Other-Scripts\GDPR-Compliance\redact.py preview

# quick trial on a subset
python .\Scripts\Other-Scripts\GDPR-Compliance\redact.py preview --input .\Datasets\MTSD\GRP-1 --limit 50

# classical detectors only (no torch needed)
python .\Scripts\Other-Scripts\GDPR-Compliance\redact.py preview --backend classic

# pixelation instead of Gaussian blur; custom plate prompt
python .\Scripts\Other-Scripts\GDPR-Compliance\redact.py preview --method pixelate --sam-prompt "licence_plate=car number plate"

# full redacted mirror (also copies images without detections)
python .\Scripts\Other-Scripts\GDPR-Compliance\redact.py preview --copy-clean

# 2. Review  GDPR-Compliance-Preview\_comparisons\  and  detection_summary.csv

# 3. Replace originals (interactive confirmation; backups on by default)
python .\Scripts\Other-Scripts\GDPR-Compliance\redact.py apply
```

## Notes

- Redaction is Gaussian blur whose kernel scales with the region
  (~half of the region's shorter side, floor 15 px), applied after the
  detector-specific padding (10–20 % per side), so the sensitive region is
  always fully covered with margin.
- Output JPEGs are re-encoded at quality 95 with the original EXIF block
  (GPS, timestamps, camera) preserved; the orientation tag is normalised
  because rotation is baked into the pixels.
- Detection runs once per image with all detectors loaded up front; nothing
  is reloaded per image.
- `plates_haar` (OpenCV Haar cascade) remains registered purely as a
  dependency-free point of comparison — expect false positives on text and
  roof lines.
