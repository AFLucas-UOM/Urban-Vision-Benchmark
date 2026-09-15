# Robustness slice analysis — new canonical-prompt extension

Generated: 2026-09-14T17:10:03  |  Commit: `2a158e051662`

Performance across data slices, computed **only from stored predictions and ground truth** (no model was run). Full machine-readable results: `robustness_slice_results.csv/.json`; thresholds and matching rules: `robustness_slice_config.json`; gaps: `insufficient_or_missing_inputs.md`.

> **New analysis, separate from the retained robustness report.** This output contains SAM 3 canonical-prompt image-condition slices only. It does not alter or supersede any earlier retained report. The requested VJEPA crop-condition branch is documented below as unavailable because per-crop predictions were not retained.

## PromptDetect-SAM3-canonical-MDWD - status: **final**

| Model | Run | Split | Metric | Value | Images | Objects |
| --- | --- | --- | --- | --- | --- | --- |
| SAM 3 canonical macro | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | test | macro_f1 | 0.5676 | 369 | 828 |
| SAM 3 | mdwd-mixed-waste | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | test | f1 | 0.8516 | 369 | 334 |
| SAM 3 | mdwd-orange-cmd | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | test | f1 | 0.8312 | 369 | 78 |
| SAM 3 | mdwd-organic-waste | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | test | f1 | 0.5269 | 369 | 178 |
| SAM 3 | mdwd-recyclable-material | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | test | f1 | 0.0606 | 369 | 238 |

Canonical SAM 3 macro-F1 by the retained supervised-detection test terciles:

| Dimension | Slice | Macro-F1 | Images | Target boxes | Support |
| --- | --- | --- | --- | --- | --- |
| brightness | low | 0.5594 | 123 | 294 | ok |
| brightness | medium | 0.5941 | 123 | 279 | ok |
| brightness | high | 0.5073 | 123 | 255 | ok |
| contrast | low | 0.6564 | 123 | 238 | ok |
| contrast | medium | 0.5302 | 123 | 296 | ok |
| contrast | high | 0.5567 | 123 | 294 | ok |
| sharpness | blurred | 0.62 | 123 | 241 | ok |
| sharpness | intermediate | 0.5433 | 123 | 276 | ok |
| sharpness | sharp | 0.5377 | 123 | 311 | ok |

Slices deviating >= 0.03 from their aggregate (sufficient support only):

| Model | Split | Slice | Metric | Value | Aggregate | Delta | Support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SAM 3 | mdwd-orange-cmd | test | brightness=high | f1 | 0.5926 | 0.8312 | -0.2386 | 123 |
| SAM 3 | mdwd-organic-waste | test | contrast=medium | f1 | 0.3704 | 0.5269 | -0.1565 | 123 |
| SAM 3 | mdwd-orange-cmd | test | sharpness=sharp | f1 | 0.7037 | 0.8312 | -0.1275 | 123 |
| SAM 3 | mdwd-organic-waste | test | brightness=medium | f1 | 0.4337 | 0.5269 | -0.0932 | 123 |
| SAM 3 | mdwd-organic-waste | test | sharpness=intermediate | f1 | 0.4387 | 0.5269 | -0.0882 | 123 |
| SAM 3 | mdwd-orange-cmd | test | contrast=high | f1 | 0.7556 | 0.8312 | -0.0756 | 123 |
| SAM 3 | mdwd-recyclable-material | test | brightness=low | f1 | 0.0 | 0.0606 | -0.0606 | 123 |
| SAM 3 | mdwd-recyclable-material | test | brightness=high | f1 | 0.0 | 0.0606 | -0.0606 | 123 |
| SAM 3 | mdwd-mixed-waste | test | contrast=medium | f1 | 0.791 | 0.8516 | -0.0606 | 123 |
| SAM 3 | mdwd-recyclable-material | test | contrast=high | f1 | 0.0 | 0.0606 | -0.0606 | 123 |
| SAM 3 canonical macro | test | brightness=high | macro_f1 | 0.5073 | 0.5676 | -0.0603 | 123 |
| SAM 3 | mdwd-mixed-waste | test | brightness=low | f1 | 0.8067 | 0.8516 | -0.0449 | 123 |
| SAM 3 | mdwd-recyclable-material | test | sharpness=intermediate | f1 | 0.0204 | 0.0606 | -0.0402 | 123 |
| SAM 3 canonical macro | test | contrast=medium | macro_f1 | 0.5302 | 0.5676 | -0.0374 | 123 |
| SAM 3 | mdwd-mixed-waste | test | sharpness=sharp | f1 | 0.8194 | 0.8516 | -0.0322 | 123 |
| SAM 3 | mdwd-organic-waste | test | brightness=high | f1 | 0.557 | 0.5269 | +0.0301 | 123 |
| SAM 3 | mdwd-mixed-waste | test | contrast=low | f1 | 0.8852 | 0.8516 | +0.0336 | 123 |
| SAM 3 | mdwd-mixed-waste | test | sharpness=intermediate | f1 | 0.8881 | 0.8516 | +0.0365 | 123 |
| SAM 3 | mdwd-organic-waste | test | contrast=high | f1 | 0.5694 | 0.5269 | +0.0425 | 123 |
| SAM 3 | mdwd-mixed-waste | test | contrast=high | f1 | 0.9017 | 0.8516 | +0.0501 | 123 |

## PromptDetect-SAM3-canonical-MTSD - status: **final**

| Model | Run | Split | Metric | Value | Images | Objects |
| --- | --- | --- | --- | --- | --- | --- |
| SAM 3 canonical macro | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | test | macro_f1 | 0.3985 | 747 | 711 |
| SAM 3 | mtsd-blind-spot-mirror | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | test | f1 | 0.5476 | 747 | 160 |
| SAM 3 | mtsd-no-entry | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | test | f1 | 0.292 | 747 | 211 |
| SAM 3 | mtsd-pedestrian-crossing | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | test | f1 | 0.6222 | 747 | 143 |
| SAM 3 | mtsd-roundabout-ahead | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | test | f1 | 0.1248 | 747 | 67 |
| SAM 3 | mtsd-stop-sign | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | test | f1 | 0.406 | 747 | 130 |

Canonical SAM 3 macro-F1 by the retained supervised-detection test terciles:

| Dimension | Slice | Macro-F1 | Images | Target boxes | Support |
| --- | --- | --- | --- | --- | --- |
| brightness | low | 0.3796 | 250 | 247 | ok |
| brightness | medium | 0.4165 | 248 | 232 | ok |
| brightness | high | 0.3911 | 249 | 232 | ok |
| contrast | low | 0.4052 | 249 | 222 | ok |
| contrast | medium | 0.4005 | 250 | 247 | ok |
| contrast | high | 0.3697 | 248 | 242 | ok |
| sharpness | blurred | 0.402 | 249 | 203 | ok |
| sharpness | intermediate | 0.4045 | 249 | 236 | ok |
| sharpness | sharp | 0.3902 | 249 | 272 | ok |

Slices deviating >= 0.03 from their aggregate (sufficient support only):

| Model | Split | Slice | Metric | Value | Aggregate | Delta | Support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SAM 3 | mtsd-blind-spot-mirror | test | contrast=high | f1 | 0.375 | 0.5476 | -0.1726 | 248 |
| SAM 3 | mtsd-pedestrian-crossing | test | contrast=low | f1 | 0.5 | 0.6222 | -0.1222 | 249 |
| SAM 3 | mtsd-blind-spot-mirror | test | brightness=low | f1 | 0.4638 | 0.5476 | -0.0838 | 250 |
| SAM 3 | mtsd-stop-sign | test | brightness=high | f1 | 0.3333 | 0.406 | -0.0727 | 249 |
| SAM 3 | mtsd-no-entry | test | sharpness=intermediate | f1 | 0.2365 | 0.292 | -0.0555 | 249 |
| SAM 3 | mtsd-blind-spot-mirror | test | sharpness=sharp | f1 | 0.5 | 0.5476 | -0.0476 | 249 |
| SAM 3 | mtsd-stop-sign | test | sharpness=blurred | f1 | 0.3636 | 0.406 | -0.0424 | 249 |
| SAM 3 | mtsd-roundabout-ahead | test | brightness=low | f1 | 0.0923 | 0.1248 | -0.0325 | 250 |
| SAM 3 | mtsd-blind-spot-mirror | test | contrast=medium | f1 | 0.5176 | 0.5476 | -0.03 | 250 |
| SAM 3 | mtsd-blind-spot-mirror | test | sharpness=intermediate | f1 | 0.5783 | 0.5476 | +0.0307 | 249 |
| SAM 3 | mtsd-no-entry | test | brightness=high | f1 | 0.3277 | 0.292 | +0.0357 | 249 |
| SAM 3 | mtsd-stop-sign | test | brightness=medium | f1 | 0.4528 | 0.406 | +0.0468 | 248 |
| SAM 3 | mtsd-no-entry | test | sharpness=blurred | f1 | 0.3446 | 0.292 | +0.0526 | 249 |
| SAM 3 | mtsd-stop-sign | test | sharpness=intermediate | f1 | 0.4615 | 0.406 | +0.0555 | 249 |
| SAM 3 | mtsd-blind-spot-mirror | test | brightness=medium | f1 | 0.6055 | 0.5476 | +0.0579 | 248 |
| SAM 3 | mtsd-pedestrian-crossing | test | contrast=medium | f1 | 0.6869 | 0.6222 | +0.0647 | 250 |
| SAM 3 | mtsd-blind-spot-mirror | test | contrast=low | f1 | 0.6796 | 0.5476 | +0.132 | 249 |

## Figures

- `Documents/Final-Figures/Robustness-Slices-New-Analysis/20260914-171000/mdwd_sam3_canonical_macro_f1_by_brightness.png`
- `Documents/Final-Figures/Robustness-Slices-New-Analysis/20260914-171000/mdwd_sam3_canonical_macro_f1_by_contrast.png`
- `Documents/Final-Figures/Robustness-Slices-New-Analysis/20260914-171000/mdwd_sam3_canonical_macro_f1_by_sharpness.png`
- `Documents/Final-Figures/Robustness-Slices-New-Analysis/20260914-171000/mtsd_sam3_canonical_macro_f1_by_brightness.png`
- `Documents/Final-Figures/Robustness-Slices-New-Analysis/20260914-171000/mtsd_sam3_canonical_macro_f1_by_contrast.png`
- `Documents/Final-Figures/Robustness-Slices-New-Analysis/20260914-171000/mtsd_sam3_canonical_macro_f1_by_sharpness.png`

## Requested branch not produced

- VJEPA 2.1-L LoRA crop-condition slices: all 1,890 crop records and crop images still exist, but no per-crop prediction or logit records were retained. Only aggregate metrics/confusion matrices and the checkpoint remain. Per instruction, this branch was stopped: no terciles were computed and no detector thresholds were reused.

*Read-only analysis - no datasets, checkpoints or previous results were modified.*
