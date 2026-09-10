# Robustness slice analysis

Generated: 2026-07-10T12:03:57  |  Commit: `15853d305771`

Performance across data slices, computed **only from stored predictions and ground truth** (no model was run). Full machine-readable results: `robustness_slice_results.csv/.json`; thresholds and matching rules: `robustness_slice_config.json`; gaps: `insufficient_or_missing_inputs.md`.

## MDWD-detection - status: **final**

| Model | Run | Split | Metric | Value | Images | Objects |
| --- | --- | --- | --- | --- | --- | --- |
| yolo11l@YOLO11-EUVIP | yolo11l__YOLO11-EUVIP__test__original | test | precision | 0.8919 | 369 | 1106 |
| yolo11l@YOLO11-EUVIP | yolo11l__YOLO11-EUVIP__test__original | test | recall | 0.8653 | 369 | 1106 |
| yolo11l@YOLO11-EUVIP | yolo11l__YOLO11-EUVIP__test__original | test | f1 | 0.8784 | 369 | 1106 |
| yolo11l@YOLO11-EUVIP | yolo11l__YOLO11-EUVIP__test__original | test | mean_matched_iou | 0.9094 | 369 | 1106 |
| yolo11l@YOLO11-EUVIP | yolo11l__YOLO11-EUVIP__valid__original | valid | precision | 0.9148 | 369 | 1072 |
| yolo11l@YOLO11-EUVIP | yolo11l__YOLO11-EUVIP__valid__original | valid | recall | 0.861 | 369 | 1072 |
| yolo11l@YOLO11-EUVIP | yolo11l__YOLO11-EUVIP__valid__original | valid | f1 | 0.8871 | 369 | 1072 |
| yolo11l@YOLO11-EUVIP | yolo11l__YOLO11-EUVIP__valid__original | valid | mean_matched_iou | 0.9152 | 369 | 1072 |
| yolo12m@YOLO12-EUVIP | yolo12m__YOLO12-EUVIP__test__original | test | precision | 0.9037 | 369 | 1106 |
| yolo12m@YOLO12-EUVIP | yolo12m__YOLO12-EUVIP__test__original | test | recall | 0.8743 | 369 | 1106 |
| yolo12m@YOLO12-EUVIP | yolo12m__YOLO12-EUVIP__test__original | test | f1 | 0.8888 | 369 | 1106 |
| yolo12m@YOLO12-EUVIP | yolo12m__YOLO12-EUVIP__test__original | test | mean_matched_iou | 0.9068 | 369 | 1106 |
| yolo12m@YOLO12-EUVIP | yolo12m__YOLO12-EUVIP__valid__original | valid | precision | 0.8836 | 369 | 1072 |
| yolo12m@YOLO12-EUVIP | yolo12m__YOLO12-EUVIP__valid__original | valid | recall | 0.8853 | 369 | 1072 |
| yolo12m@YOLO12-EUVIP | yolo12m__YOLO12-EUVIP__valid__original | valid | f1 | 0.8844 | 369 | 1072 |
| yolo12m@YOLO12-EUVIP | yolo12m__YOLO12-EUVIP__valid__original | valid | mean_matched_iou | 0.9068 | 369 | 1072 |
| yolo26l@YOLO26-DGX | yolo26l__YOLO26-DGX__test__original | test | precision | 0.9341 | 369 | 1106 |
| yolo26l@YOLO26-DGX | yolo26l__YOLO26-DGX__test__original | test | recall | 0.8843 | 369 | 1106 |
| yolo26l@YOLO26-DGX | yolo26l__YOLO26-DGX__test__original | test | f1 | 0.9085 | 369 | 1106 |
| yolo26l@YOLO26-DGX | yolo26l__YOLO26-DGX__test__original | test | mean_matched_iou | 0.9175 | 369 | 1106 |
| yolo26l@YOLO26-DGX | yolo26l__YOLO26-DGX__valid__original | valid | precision | 0.9296 | 369 | 1072 |
| yolo26l@YOLO26-DGX | yolo26l__YOLO26-DGX__valid__original | valid | recall | 0.8741 | 369 | 1072 |
| yolo26l@YOLO26-DGX | yolo26l__YOLO26-DGX__valid__original | valid | f1 | 0.901 | 369 | 1072 |
| yolo26l@YOLO26-DGX | yolo26l__YOLO26-DGX__valid__original | valid | mean_matched_iou | 0.9236 | 369 | 1072 |
| yolo26l@YOLO26-EUVIP | yolo26l__YOLO26-EUVIP__test__original | test | precision | 0.9454 | 369 | 1106 |
| yolo26l@YOLO26-EUVIP | yolo26l__YOLO26-EUVIP__test__original | test | recall | 0.8761 | 369 | 1106 |
| yolo26l@YOLO26-EUVIP | yolo26l__YOLO26-EUVIP__test__original | test | f1 | 0.9094 | 369 | 1106 |
| yolo26l@YOLO26-EUVIP | yolo26l__YOLO26-EUVIP__test__original | test | mean_matched_iou | 0.9215 | 369 | 1106 |
| yolo26l@YOLO26-EUVIP | yolo26l__YOLO26-EUVIP__valid__original | valid | precision | 0.9173 | 369 | 1072 |
| yolo26l@YOLO26-EUVIP | yolo26l__YOLO26-EUVIP__valid__original | valid | recall | 0.8797 | 369 | 1072 |
| yolo26l@YOLO26-EUVIP | yolo26l__YOLO26-EUVIP__valid__original | valid | f1 | 0.8981 | 369 | 1072 |
| yolo26l@YOLO26-EUVIP | yolo26l__YOLO26-EUVIP__valid__original | valid | mean_matched_iou | 0.9233 | 369 | 1072 |

Slices deviating >= 0.03 from their aggregate (sufficient support only):

| Model | Split | Slice | Metric | Value | Aggregate | Delta | Support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| yolo12m@YOLO12-EUVIP | test | object_size=small | recall | 0.3786 | 0.8743 | -0.4957 | 103 |
| yolo26l@YOLO26-EUVIP | test | object_size=small | recall | 0.4175 | 0.8761 | -0.4586 | 103 |
| yolo11l@YOLO11-EUVIP | valid | object_size=small | recall | 0.413 | 0.861 | -0.448 | 92 |
| yolo26l@YOLO26-DGX | test | object_size=small | recall | 0.4369 | 0.8843 | -0.4474 | 103 |
| yolo12m@YOLO12-EUVIP | valid | object_size=small | recall | 0.4565 | 0.8853 | -0.4288 | 92 |
| yolo11l@YOLO11-EUVIP | test | object_size=small | recall | 0.4369 | 0.8653 | -0.4284 | 103 |
| yolo26l@YOLO26-DGX | valid | object_size=small | recall | 0.4457 | 0.8741 | -0.4284 | 92 |
| yolo26l@YOLO26-EUVIP | valid | object_size=small | recall | 0.4783 | 0.8797 | -0.4014 | 92 |
| yolo11l@YOLO11-EUVIP | test | object_position=near-edge | recall | 0.6737 | 0.8653 | -0.1916 | 285 |
| yolo26l@YOLO26-DGX | valid | object_position=near-edge | recall | 0.6965 | 0.8741 | -0.1776 | 313 |
| yolo12m@YOLO12-EUVIP | test | object_position=near-edge | recall | 0.6982 | 0.8743 | -0.1761 | 285 |
| yolo26l@YOLO26-EUVIP | test | object_position=near-edge | recall | 0.7018 | 0.8761 | -0.1743 | 285 |
| yolo11l@YOLO11-EUVIP | valid | object_position=near-edge | recall | 0.6869 | 0.861 | -0.1741 | 313 |
| yolo26l@YOLO26-EUVIP | valid | object_position=near-edge | recall | 0.7093 | 0.8797 | -0.1704 | 313 |
| yolo26l@YOLO26-DGX | test | object_position=near-edge | recall | 0.7158 | 0.8843 | -0.1685 | 285 |
| yolo12m@YOLO12-EUVIP | valid | object_position=near-edge | recall | 0.722 | 0.8853 | -0.1633 | 313 |
| yolo12m@YOLO12-EUVIP | test | object_size=small | mean_matched_iou | 0.7631 | 0.9068 | -0.1437 | 103 |
| yolo11l@YOLO11-EUVIP | test | class=Other Waste | recall | 0.7266 | 0.8653 | -0.1387 | 278 |
| yolo11l@YOLO11-EUVIP | test | object_size=small | mean_matched_iou | 0.7754 | 0.9094 | -0.134 | 103 |
| yolo26l@YOLO26-DGX | test | object_size=small | mean_matched_iou | 0.791 | 0.9175 | -0.1265 | 103 |

## MTSD-attributes - status: **historical snapshot**

> Historical GRP-1..GRP-3 snapshot; a final-scope round is still pending.


Slices deviating >= 0.03 from their aggregate (sufficient support only):

| Model | Split | Slice | Metric | Value | Aggregate | Delta | Support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vjepa_lora | test | condition_class=Heavily Damaged | f1 | 0.3182 | 0.5948 | -0.2767 | 30 |
| vjepa | test | condition_class=Heavily Damaged | f1 | 0.2887 | 0.511 | -0.2224 | 30 |
| dinov3 | test | condition_class=Heavily Damaged | f1 | 0.3462 | 0.5534 | -0.2073 | 30 |
| convnext_frozen | test | condition_class=Heavily Damaged | f1 | 0.4043 | 0.5673 | -0.1631 | 30 |
| vjepa | test | condition_class=Weathered | f1 | 0.3681 | 0.511 | -0.1429 | 85 |
| convnext_frozen | test | mounting_class=Wall-Mounted | f1 | 0.6621 | 0.8026 | -0.1405 | 59 |
| convnext | test | condition_class=Heavily Damaged | f1 | 0.5882 | 0.7113 | -0.1231 | 30 |
| dinov3_lora | test | condition_class=Weathered | f1 | 0.6161 | 0.7346 | -0.1185 | 85 |
| convnext_frozen | test | condition_class=Weathered | f1 | 0.449 | 0.5673 | -0.1183 | 85 |
| vjepa | test | sign_shape_class=Octagonal | f1 | 0.7006 | 0.8189 | -0.1183 | 61 |
| dinov3 | test | mounting_class=Wall-Mounted | f1 | 0.7302 | 0.8458 | -0.1156 | 59 |
| convnext | test | mounting_class=Wall-Mounted | f1 | 0.7748 | 0.8734 | -0.0986 | 59 |
| vjepa | test | mounting_class=Wall-Mounted | f1 | 0.7714 | 0.8672 | -0.0958 | 59 |
| convnext | test | condition_class=Weathered | f1 | 0.6199 | 0.7113 | -0.0914 | 85 |
| convnext_frozen | test | view_angle_class=Side | f1 | 0.7603 | 0.8426 | -0.0823 | 118 |
| dinov3_lora | test | mounting_class=Wall-Mounted | f1 | 0.8167 | 0.8959 | -0.0793 | 59 |
| vjepa_lora | test | mounting_class=Wall-Mounted | f1 | 0.8346 | 0.9054 | -0.0707 | 59 |
| dinov3 | test | view_angle_class=Side | f1 | 0.8034 | 0.8649 | -0.0615 | 118 |
| dinov3 | test | condition_class=Weathered | f1 | 0.4925 | 0.5534 | -0.0609 | 85 |
| convnext | test | view_angle_class=Side | f1 | 0.8485 | 0.9054 | -0.0569 | 118 |

> 6 slice rows have support below 15 and are marked `insufficient_support`.

## PromptDetect-MDWD - status: **pilot**

> Pilot-scale run: pipeline validation only, not dissertation evidence.

| Model | Run | Split | Metric | Value | Images | Objects |
| --- | --- | --- | --- | --- | --- | --- |
| Cosmos Reason2 2B | MDWD/20260710-011505 | garbage bag | test | precision | 1.0 | 10 | 23 |
| Cosmos Reason2 2B | MDWD/20260710-011505 | garbage bag | test | recall | 0.3478 | 10 | 23 |
| Cosmos Reason2 2B | MDWD/20260710-011505 | garbage bag | test | f1 | 0.5161 | 10 | 23 |
| Cosmos Reason2 2B | MDWD/20260710-011505 | garbage bag | test | mean_matched_iou | 0.8921 | 10 | 23 |
| Cosmos Reason2 2B | MDWD/20260710-011505 | garbage bag | test | fp_per_image | 0.0 | 10 | 23 |
| Cosmos Reason2 2B | MDWD/20260710-011505 | garbage bag | test | fn_per_image | 1.5 | 10 | 23 |
| SAM 3.1 | MDWD/20260710-011505 | garbage bag | test | precision | 0.9167 | 10 | 23 |
| SAM 3.1 | MDWD/20260710-011505 | garbage bag | test | recall | 0.9565 | 10 | 23 |
| SAM 3.1 | MDWD/20260710-011505 | garbage bag | test | f1 | 0.9362 | 10 | 23 |
| SAM 3.1 | MDWD/20260710-011505 | garbage bag | test | mean_matched_iou | 0.9308 | 10 | 23 |
| SAM 3.1 | MDWD/20260710-011505 | garbage bag | test | fp_per_image | 0.2 | 10 | 23 |
| SAM 3.1 | MDWD/20260710-011505 | garbage bag | test | fn_per_image | 0.1 | 10 | 23 |

Slices deviating >= 0.03 from their aggregate (sufficient support only):

| Model | Split | Slice | Metric | Value | Aggregate | Delta | Support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cosmos Reason2 2B | test | object_size=large | recall | 0.381 | 0.3478 | +0.0332 | 21 |
| SAM 3.1 | test | object_size=large | recall | 1.0 | 0.9565 | +0.0435 | 21 |

> 98 slice rows have support below 15 and are marked `insufficient_support`.

## Figures

- `Documents/Final-Figures/Robustness-Slices/20260710-120353/mdwd_recall_by_object_size_test.png`
- `Documents/Final-Figures/Robustness-Slices/20260710-120353/mdwd_f1_by_brightness_test.png`
- `Documents/Final-Figures/Robustness-Slices/20260710-120353/mdwd_f1_by_sharpness_test.png`
- `Documents/Final-Figures/Robustness-Slices/20260710-120353/mdwd_per_class_f1_test.png`
- `Documents/Final-Figures/Robustness-Slices/20260710-120353/mtsd_attr_macro_f1_by_head.png`
- `Documents/Final-Figures/Robustness-Slices/20260710-120353/mtsd_attr_condition_per_class_f1.png`
- `Documents/Final-Figures/Robustness-Slices/20260710-120353/mtsd_attr_view_angle_per_class_f1.png`
- `Documents/Final-Figures/Robustness-Slices/20260710-120353/promptdetect_model_prompt_f1.png`
- `Documents/Final-Figures/Robustness-Slices/20260710-120353/mdwd_slice_degradation_test.png`

*Read-only analysis - no datasets, checkpoints or previous results were modified.*
