# Robustness slice analysis — new canonical-prompt extension

Generated: 2026-09-14T17:47:15  |  Commit: `2a158e051662`

Performance across data slices, computed **only from stored predictions and ground truth** (no model was run). Full machine-readable results: `robustness_slice_results.csv/.json`; thresholds and matching rules: `robustness_slice_config.json`; gaps: `insufficient_or_missing_inputs.md`.

> **New analysis, separate from the retained robustness report.** This output contains SAM 3 canonical-prompt image-condition and object-size slices. It does not alter or supersede any earlier retained report. The requested VJEPA crop-condition branch is documented below as unavailable because per-crop predictions were not retained.

## PromptDetect-SAM3-canonical-MDWD - status: **final**

| Model | Run | Split | Metric | Value | Images | Objects |
| --- | --- | --- | --- | --- | --- | --- |
| SAM 3 canonical macro | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | test | macro_f1 | 0.5676 | 369 | 828 |
| SAM 3 canonical macro | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | test | macro_recall | 0.638 | 369 | 828 |
| SAM 3 — mdwd-mixed-waste | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | test | f1 | 0.8516 | 369 | 334 |
| SAM 3 — mdwd-mixed-waste | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | test | recall | 0.9281 | 369 | 334 |
| SAM 3 — mdwd-orange-cmd | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | test | f1 | 0.8312 | 369 | 78 |
| SAM 3 — mdwd-orange-cmd | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | test | recall | 0.8205 | 369 | 78 |
| SAM 3 — mdwd-organic-waste | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | test | f1 | 0.5269 | 369 | 178 |
| SAM 3 — mdwd-organic-waste | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | test | recall | 0.7697 | 369 | 178 |
| SAM 3 — mdwd-recyclable-material | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | test | f1 | 0.0606 | 369 | 238 |
| SAM 3 — mdwd-recyclable-material | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | test | recall | 0.0336 | 369 | 238 |

Canonical SAM 3 macro metrics using detector-aligned slice definitions:

| Dimension | Slice | Metric | Value | Images | Target boxes | Support |
| --- | --- | --- | --- | --- | --- | --- |
| brightness | low | macro_f1 | 0.5594 | 123 | 294 | ok |
| brightness | medium | macro_f1 | 0.5941 | 123 | 279 | ok |
| brightness | high | macro_f1 | 0.5073 | 123 | 255 | ok |
| contrast | low | macro_f1 | 0.6564 | 123 | 238 | ok |
| contrast | medium | macro_f1 | 0.5302 | 123 | 296 | ok |
| contrast | high | macro_f1 | 0.5567 | 123 | 294 | ok |
| sharpness | blurred | macro_f1 | 0.62 | 123 | 241 | ok |
| sharpness | intermediate | macro_f1 | 0.5433 | 123 | 276 | ok |
| sharpness | sharp | macro_f1 | 0.5377 | 123 | 311 | ok |
| object_density | single-object | macro_f1 | 0.6321 | 142 | 138 | ok |
| object_density | low-clutter | macro_f1 | 0.5691 | 159 | 321 | ok |
| object_density | high-clutter | macro_f1 | 0.5171 | 68 | 369 | ok |
| object_size | small | macro_recall | 0.5298 | 13 | 49 | ok |
| object_size | medium | macro_recall | 0.6576 | 144 | 281 | ok |
| object_size | large | macro_recall | 0.7074 | 265 | 488 | ok |

Object density is an image-level split and therefore reports macro-F1. Object size is a target-object split and reports macro-recall because false positives cannot be uniquely assigned to a ground-truth size bin. Size-family cells below the minimum support are excluded from that bin's macro-average.

Slices deviating >= 0.03 from their aggregate (sufficient support only):

| Model | Split | Slice | Metric | Value | Aggregate | Delta | Support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SAM 3 — mdwd-orange-cmd | test | object_size=small | recall | 0.381 | 0.8205 | -0.4395 | 21 |
| SAM 3 — mdwd-mixed-waste | test | object_size=small | recall | 0.6786 | 0.9281 | -0.2495 | 28 |
| SAM 3 — mdwd-orange-cmd | test | brightness=high | f1 | 0.5926 | 0.8312 | -0.2386 | 123 |
| SAM 3 — mdwd-organic-waste | test | contrast=medium | f1 | 0.3704 | 0.5269 | -0.1565 | 123 |
| SAM 3 — mdwd-orange-cmd | test | sharpness=sharp | f1 | 0.7037 | 0.8312 | -0.1275 | 123 |
| SAM 3 — mdwd-orange-cmd | test | object_density=high-clutter | f1 | 0.7059 | 0.8312 | -0.1253 | 68 |
| SAM 3 canonical macro | test | object_size=small | macro_recall | 0.5298 | 0.638 | -0.1082 | 49 |
| SAM 3 — mdwd-organic-waste | test | brightness=medium | f1 | 0.4337 | 0.5269 | -0.0932 | 123 |
| SAM 3 — mdwd-organic-waste | test | sharpness=intermediate | f1 | 0.4387 | 0.5269 | -0.0882 | 123 |
| SAM 3 — mdwd-organic-waste | test | object_density=high-clutter | f1 | 0.4503 | 0.5269 | -0.0766 | 68 |
| SAM 3 — mdwd-orange-cmd | test | contrast=high | f1 | 0.7556 | 0.8312 | -0.0756 | 123 |
| SAM 3 — mdwd-organic-waste | test | object_density=low-clutter | f1 | 0.4576 | 0.5269 | -0.0693 | 159 |
| SAM 3 — mdwd-recyclable-material | test | brightness=low | f1 | 0.0 | 0.0606 | -0.0606 | 123 |
| SAM 3 — mdwd-recyclable-material | test | brightness=high | f1 | 0.0 | 0.0606 | -0.0606 | 123 |
| SAM 3 — mdwd-mixed-waste | test | contrast=medium | f1 | 0.791 | 0.8516 | -0.0606 | 123 |
| SAM 3 — mdwd-recyclable-material | test | contrast=high | f1 | 0.0 | 0.0606 | -0.0606 | 123 |
| SAM 3 — mdwd-recyclable-material | test | object_density=single-object | f1 | 0.0 | 0.0606 | -0.0606 | 142 |
| SAM 3 canonical macro | test | brightness=high | macro_f1 | 0.5073 | 0.5676 | -0.0603 | 123 |
| SAM 3 canonical macro | test | object_density=high-clutter | macro_f1 | 0.5171 | 0.5676 | -0.0505 | 68 |
| SAM 3 — mdwd-mixed-waste | test | brightness=low | f1 | 0.8067 | 0.8516 | -0.0449 | 123 |

> 2 slice rows have support below 15 and are marked `insufficient_support`.

## PromptDetect-SAM3-canonical-MTSD - status: **final**

| Model | Run | Split | Metric | Value | Images | Objects |
| --- | --- | --- | --- | --- | --- | --- |
| SAM 3 canonical macro | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | test | macro_f1 | 0.3985 | 747 | 711 |
| SAM 3 canonical macro | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | test | macro_recall | 0.6727 | 747 | 711 |
| SAM 3 — mtsd-blind-spot-mirror | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | test | f1 | 0.5476 | 747 | 160 |
| SAM 3 — mtsd-blind-spot-mirror | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | test | recall | 0.4313 | 747 | 160 |
| SAM 3 — mtsd-no-entry | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | test | f1 | 0.292 | 747 | 211 |
| SAM 3 — mtsd-no-entry | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | test | recall | 0.8199 | 747 | 211 |
| SAM 3 — mtsd-pedestrian-crossing | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | test | f1 | 0.6222 | 747 | 143 |
| SAM 3 — mtsd-pedestrian-crossing | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | test | recall | 0.5874 | 747 | 143 |
| SAM 3 — mtsd-roundabout-ahead | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | test | f1 | 0.1248 | 747 | 67 |
| SAM 3 — mtsd-roundabout-ahead | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | test | recall | 0.6866 | 747 | 67 |
| SAM 3 — mtsd-stop-sign | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | test | f1 | 0.406 | 747 | 130 |
| SAM 3 — mtsd-stop-sign | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | test | recall | 0.8385 | 747 | 130 |

Canonical SAM 3 macro metrics using detector-aligned slice definitions:

| Dimension | Slice | Metric | Value | Images | Target boxes | Support |
| --- | --- | --- | --- | --- | --- | --- |
| brightness | low | macro_f1 | 0.3796 | 250 | 247 | ok |
| brightness | medium | macro_f1 | 0.4165 | 248 | 232 | ok |
| brightness | high | macro_f1 | 0.3911 | 249 | 232 | ok |
| contrast | low | macro_f1 | 0.4052 | 249 | 222 | ok |
| contrast | medium | macro_f1 | 0.4005 | 250 | 247 | ok |
| contrast | high | macro_f1 | 0.3697 | 248 | 242 | ok |
| sharpness | blurred | macro_f1 | 0.402 | 249 | 203 | ok |
| sharpness | intermediate | macro_f1 | 0.4045 | 249 | 236 | ok |
| sharpness | sharp | macro_f1 | 0.3902 | 249 | 272 | ok |
| object_density | single-object | macro_f1 | 0.4339 | 296 | 192 | ok |
| object_density | low-clutter | macro_f1 | 0.3773 | 348 | 323 | ok |
| object_density | high-clutter | macro_f1 | 0.3461 | 103 | 196 | ok |
| object_size | small | macro_recall | 0.1944 | 25 | 33 | ok |
| object_size | medium | macro_recall | 0.4397 | 76 | 96 | ok |
| object_size | large | macro_recall | 0.784 | 496 | 538 | ok |

Object density is an image-level split and therefore reports macro-F1. Object size is a target-object split and reports macro-recall because false positives cannot be uniquely assigned to a ground-truth size bin. Size-family cells below the minimum support are excluded from that bin's macro-average.

Slices deviating >= 0.03 from their aggregate (sufficient support only):

| Model | Split | Slice | Metric | Value | Aggregate | Delta | Support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SAM 3 — mtsd-pedestrian-crossing | test | object_size=small | recall | 0.0 | 0.5874 | -0.5874 | 15 |
| SAM 3 canonical macro | test | object_size=small | macro_recall | 0.1944 | 0.6727 | -0.4783 | 33 |
| SAM 3 — mtsd-no-entry | test | object_size=small | recall | 0.3889 | 0.8199 | -0.431 | 18 |
| SAM 3 — mtsd-blind-spot-mirror | test | object_size=medium | recall | 0.0435 | 0.4313 | -0.3878 | 23 |
| SAM 3 — mtsd-blind-spot-mirror | test | object_density=high-clutter | f1 | 0.2609 | 0.5476 | -0.2867 | 103 |
| SAM 3 canonical macro | test | object_size=medium | macro_recall | 0.4397 | 0.6727 | -0.233 | 96 |
| SAM 3 — mtsd-pedestrian-crossing | test | object_size=medium | recall | 0.4048 | 0.5874 | -0.1826 | 42 |
| SAM 3 — mtsd-blind-spot-mirror | test | contrast=high | f1 | 0.375 | 0.5476 | -0.1726 | 248 |
| SAM 3 — mtsd-blind-spot-mirror | test | object_density=low-clutter | f1 | 0.4194 | 0.5476 | -0.1282 | 348 |
| SAM 3 — mtsd-pedestrian-crossing | test | contrast=low | f1 | 0.5 | 0.6222 | -0.1222 | 249 |
| SAM 3 — mtsd-roundabout-ahead | test | object_density=single-object | f1 | 0.0105 | 0.1248 | -0.1143 | 296 |
| SAM 3 — mtsd-blind-spot-mirror | test | brightness=low | f1 | 0.4638 | 0.5476 | -0.0838 | 250 |
| SAM 3 — mtsd-stop-sign | test | brightness=high | f1 | 0.3333 | 0.406 | -0.0727 | 249 |
| SAM 3 — mtsd-pedestrian-crossing | test | object_density=high-clutter | f1 | 0.5631 | 0.6222 | -0.0591 | 103 |
| SAM 3 — mtsd-no-entry | test | sharpness=intermediate | f1 | 0.2365 | 0.292 | -0.0555 | 249 |
| SAM 3 — mtsd-no-entry | test | object_density=high-clutter | f1 | 0.2372 | 0.292 | -0.0548 | 103 |
| SAM 3 canonical macro | test | object_density=high-clutter | macro_f1 | 0.3461 | 0.3985 | -0.0524 | 103 |
| SAM 3 — mtsd-blind-spot-mirror | test | sharpness=sharp | f1 | 0.5 | 0.5476 | -0.0476 | 249 |
| SAM 3 — mtsd-stop-sign | test | sharpness=blurred | f1 | 0.3636 | 0.406 | -0.0424 | 249 |
| SAM 3 — mtsd-roundabout-ahead | test | brightness=low | f1 | 0.0923 | 0.1248 | -0.0325 | 250 |

> 5 slice rows have support below 15 and are marked `insufficient_support`.

## Figures

- `Documents/Final-Figures/Robustness-Slices-New-Analysis/latest/mdwd_sam3_canonical_macro_f1_by_brightness.png`
- `Documents/Final-Figures/Robustness-Slices-New-Analysis/latest/mdwd_sam3_canonical_macro_f1_by_contrast.png`
- `Documents/Final-Figures/Robustness-Slices-New-Analysis/latest/mdwd_sam3_canonical_macro_f1_by_sharpness.png`
- `Documents/Final-Figures/Robustness-Slices-New-Analysis/latest/mdwd_sam3_canonical_macro_f1_by_object_density.png`
- `Documents/Final-Figures/Robustness-Slices-New-Analysis/latest/mdwd_sam3_canonical_macro_recall_by_object_size.png`
- `Documents/Final-Figures/Robustness-Slices-New-Analysis/latest/mtsd_sam3_canonical_macro_f1_by_brightness.png`
- `Documents/Final-Figures/Robustness-Slices-New-Analysis/latest/mtsd_sam3_canonical_macro_f1_by_contrast.png`
- `Documents/Final-Figures/Robustness-Slices-New-Analysis/latest/mtsd_sam3_canonical_macro_f1_by_sharpness.png`
- `Documents/Final-Figures/Robustness-Slices-New-Analysis/latest/mtsd_sam3_canonical_macro_f1_by_object_density.png`
- `Documents/Final-Figures/Robustness-Slices-New-Analysis/latest/mtsd_sam3_canonical_macro_recall_by_object_size.png`

## Requested branch not produced

- VJEPA 2.1-L LoRA crop-condition slices: all 1,890 crop records and crop images still exist, but no per-crop prediction or logit records were retained. Only aggregate metrics/confusion matrices and the checkpoint remain. Per instruction, this branch was stopped: no terciles were computed and no detector thresholds were reused.

*Read-only analysis - no datasets, checkpoints or previous results were modified.*
