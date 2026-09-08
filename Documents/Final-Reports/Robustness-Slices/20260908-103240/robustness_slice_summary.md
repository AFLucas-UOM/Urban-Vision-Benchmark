# Robustness slice analysis

Generated: 2026-09-08T10:32:54  |  Commit: `ff0bc51f590b`

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
| convnext_base_finetune | test | condition_class=Heavily Damaged | f1 | 0.4094 | 0.6502 | -0.2409 | 110 |
| lingbot_vitl_frozen | test | condition_class=Heavily Damaged | f1 | 0.3519 | 0.5578 | -0.206 | 110 |
| vjepa21_vitl_frozen | test | condition_class=Heavily Damaged | f1 | 0.3623 | 0.5601 | -0.1978 | 110 |
| lingbot_vitb_frozen | test | condition_class=Heavily Damaged | f1 | 0.3353 | 0.5306 | -0.1953 | 110 |
| dinov3_vitb_frozen | test | condition_class=Heavily Damaged | f1 | 0.388 | 0.5767 | -0.1888 | 110 |
| convnext_large_lora | test | condition_class=Heavily Damaged | f1 | 0.5048 | 0.6891 | -0.1844 | 110 |
| vjepa21_vitb_frozen | test | condition_class=Heavily Damaged | f1 | 0.3755 | 0.5568 | -0.1813 | 110 |
| convnext_large_frozen | test | condition_class=Weathered | f1 | 0.3844 | 0.5538 | -0.1694 | 392 |
| convnext_base_frozen | test | condition_class=Heavily Damaged | f1 | 0.3647 | 0.5339 | -0.1692 | 110 |
| convnext_base_lora | test | condition_class=Heavily Damaged | f1 | 0.5436 | 0.7022 | -0.1586 | 110 |
| dinov3_vitl_frozen | test | condition_class=Heavily Damaged | f1 | 0.4264 | 0.5835 | -0.1572 | 110 |
| dinov3_vitb_lora | test | condition_class=Heavily Damaged | f1 | 0.5463 | 0.6924 | -0.1461 | 110 |
| dinov3_vitl_lora | test | condition_class=Heavily Damaged | f1 | 0.5951 | 0.7282 | -0.133 | 110 |
| convnext_large_frozen | test | condition_class=Heavily Damaged | f1 | 0.4225 | 0.5538 | -0.1312 | 110 |
| convnext_large_finetune | test | condition_class=Heavily Damaged | f1 | 0.5654 | 0.6955 | -0.13 | 110 |
| lingbot_vitb_lora | test | condition_class=Heavily Damaged | f1 | 0.5842 | 0.7101 | -0.126 | 110 |
| vjepa21_vitb_lora | test | condition_class=Heavily Damaged | f1 | 0.5933 | 0.7126 | -0.1193 | 110 |
| lingbot_vitl_lora | test | condition_class=Heavily Damaged | f1 | 0.5915 | 0.7055 | -0.114 | 110 |
| convnext_base_frozen | test | condition_class=Weathered | f1 | 0.424 | 0.5339 | -0.1098 | 392 |
| vjepa21_vitl_frozen | test | condition_class=Weathered | f1 | 0.4641 | 0.5601 | -0.096 | 392 |

## PromptDetect-MDWD - status: **final**

| Model | Run | Split | Metric | Value | Images | Objects |
| --- | --- | --- | --- | --- | --- | --- |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | precision | 0.8537 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | recall | 0.0949 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | f1 | 0.1709 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | mean_matched_iou | 0.8639 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | fp_per_image | 0.0488 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | fn_per_image | 2.7127 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | precision | 0.8587 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | recall | 0.2197 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | f1 | 0.3499 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | mean_matched_iou | 0.8478 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | fp_per_image | 0.1084 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | fn_per_image | 2.3388 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | precision | 0.85 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | recall | 0.2152 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | f1 | 0.3434 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | mean_matched_iou | 0.8446 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | fp_per_image | 0.1138 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | fn_per_image | 2.3523 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | precision | 0.9146 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | recall | 0.3969 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | f1 | 0.5536 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | mean_matched_iou | 0.8532 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | fp_per_image | 0.1111 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | fn_per_image | 1.8076 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | precision | 0.9165 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | recall | 0.4367 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | f1 | 0.5915 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | mean_matched_iou | 0.8568 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | fp_per_image | 0.1192 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | fn_per_image | 1.6883 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | precision | 0.8764 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | recall | 0.2758 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | f1 | 0.4195 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | mean_matched_iou | 0.8524 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | fp_per_image | 0.1165 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | fn_per_image | 2.1707 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | precision | 0.8824 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | recall | 0.3933 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | f1 | 0.5441 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | mean_matched_iou | 0.8588 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | fp_per_image | 0.1572 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | fn_per_image | 1.8184 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | precision | 0.8365 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | recall | 0.0787 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | f1 | 0.1438 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | mean_matched_iou | 0.8644 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | fp_per_image | 0.0461 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | fn_per_image | 2.7615 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | precision | 0.8377 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | recall | 0.1447 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | f1 | 0.2467 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | mean_matched_iou | 0.8423 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | fp_per_image | 0.084 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | fn_per_image | 2.5637 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | precision | 0.8901 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | recall | 0.1537 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | f1 | 0.2621 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | mean_matched_iou | 0.8378 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | fp_per_image | 0.0569 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | fn_per_image | 2.5366 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | precision | 0.8595 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | recall | 0.094 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | f1 | 0.1695 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | mean_matched_iou | 0.8517 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | fp_per_image | 0.0461 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | fn_per_image | 2.7154 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | precision | 0.85 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | recall | 0.1844 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | f1 | 0.3031 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | mean_matched_iou | 0.8571 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | fp_per_image | 0.0976 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | fn_per_image | 2.4444 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | precision | 0.8598 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | recall | 0.1664 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | f1 | 0.2788 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | mean_matched_iou | 0.851 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | fp_per_image | 0.0813 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | fn_per_image | 2.4986 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | precision | 0.8953 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | recall | 0.0696 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | f1 | 0.1292 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | mean_matched_iou | 0.8292 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | fp_per_image | 0.0244 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | fn_per_image | 2.7886 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | precision | 0.8301 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | recall | 0.1148 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | f1 | 0.2017 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | mean_matched_iou | 0.8446 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | fp_per_image | 0.0705 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | fn_per_image | 2.6531 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | precision | 0.7877 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | recall | 0.1275 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | f1 | 0.2195 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | mean_matched_iou | 0.8377 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | fp_per_image | 0.103 | 369 | 1106 |
| Cosmos Reason2 2B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | fn_per_image | 2.6152 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | precision | 0.8952 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | recall | 0.4014 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | f1 | 0.5543 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | mean_matched_iou | 0.8497 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | fp_per_image | 0.1409 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | fn_per_image | 1.794 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | precision | 0.8536 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | recall | 0.3689 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | f1 | 0.5152 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | mean_matched_iou | 0.8518 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | fp_per_image | 0.1897 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | fn_per_image | 1.8916 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | precision | 0.8381 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | recall | 0.3743 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | f1 | 0.5175 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | mean_matched_iou | 0.8543 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | fp_per_image | 0.2168 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | fn_per_image | 1.8753 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | precision | 0.846 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | recall | 0.3626 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | f1 | 0.5076 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | mean_matched_iou | 0.847 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | fp_per_image | 0.1978 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | fn_per_image | 1.9106 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | precision | 0.8941 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | recall | 0.4503 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | f1 | 0.5989 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | mean_matched_iou | 0.8496 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | fp_per_image | 0.1599 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | fn_per_image | 1.6477 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | precision | 0.8735 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | recall | 0.1998 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | f1 | 0.3252 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | mean_matched_iou | 0.8466 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | fp_per_image | 0.0867 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | fn_per_image | 2.3984 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | precision | 0.8831 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | recall | 0.3553 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | f1 | 0.5068 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | mean_matched_iou | 0.8574 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | fp_per_image | 0.1409 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | fn_per_image | 1.9322 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | precision | 0.8915 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | recall | 0.4159 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | f1 | 0.5672 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | mean_matched_iou | 0.8575 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | fp_per_image | 0.1518 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | fn_per_image | 1.7507 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | precision | 0.9038 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | recall | 0.425 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | f1 | 0.5781 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | mean_matched_iou | 0.854 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | fp_per_image | 0.1355 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | fn_per_image | 1.7236 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | precision | 0.8912 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | recall | 0.3924 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | f1 | 0.5449 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | mean_matched_iou | 0.8592 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | fp_per_image | 0.1436 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | fn_per_image | 1.8211 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | precision | 0.8874 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | recall | 0.3065 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | f1 | 0.4556 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | mean_matched_iou | 0.861 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | fp_per_image | 0.1165 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | fn_per_image | 2.0786 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | precision | 0.8986 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | recall | 0.2803 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | f1 | 0.4273 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | mean_matched_iou | 0.8614 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | fp_per_image | 0.0949 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | fn_per_image | 2.1572 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | precision | 0.8921 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | recall | 0.2767 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | f1 | 0.4224 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | mean_matched_iou | 0.8603 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | fp_per_image | 0.1003 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | fn_per_image | 2.168 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | precision | 0.8787 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | recall | 0.3797 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | f1 | 0.5303 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | mean_matched_iou | 0.8568 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | fp_per_image | 0.1572 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | fn_per_image | 1.8591 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | precision | 0.866 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | recall | 0.368 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | f1 | 0.5165 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | mean_matched_iou | 0.8569 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | fp_per_image | 0.1707 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | fn_per_image | 1.8943 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | precision | 0.9 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | recall | 0.358 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | f1 | 0.5123 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | mean_matched_iou | 0.8568 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | fp_per_image | 0.1192 | 369 | 1106 |
| Cosmos Reason2 8B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | fn_per_image | 1.9241 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | precision | 0.5874 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | recall | 0.4647 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | f1 | 0.5189 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | mean_matched_iou | 0.8872 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | fp_per_image | 0.9783 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | fn_per_image | 1.6043 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | precision | 0.6408 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | recall | 0.2839 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | f1 | 0.3935 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | mean_matched_iou | 0.8801 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | fp_per_image | 0.477 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | fn_per_image | 2.1463 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | precision | 0.643 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | recall | 0.2948 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | f1 | 0.4042 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | mean_matched_iou | 0.8821 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | fp_per_image | 0.4905 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | fn_per_image | 2.1138 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | precision | 0.46 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | recall | 0.4467 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | f1 | 0.4532 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | mean_matched_iou | 0.8727 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | fp_per_image | 1.5718 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | fn_per_image | 1.6585 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | precision | 0.4314 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | recall | 0.6311 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | f1 | 0.5125 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | mean_matched_iou | 0.8772 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | fp_per_image | 2.4932 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | fn_per_image | 1.1057 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | precision | 0.3009 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | recall | 0.1203 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | f1 | 0.1718 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | mean_matched_iou | 0.8738 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | fp_per_image | 0.8374 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | fn_per_image | 2.6369 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | precision | 0.3606 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | recall | 0.4819 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | f1 | 0.4125 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | mean_matched_iou | 0.8793 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | fp_per_image | 2.561 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | fn_per_image | 1.5528 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | precision | 0.5952 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | recall | 0.6076 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | f1 | 0.6013 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | mean_matched_iou | 0.8845 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | fp_per_image | 1.2385 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | fn_per_image | 1.1762 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | precision | 0.6765 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | recall | 0.6221 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | f1 | 0.6481 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | mean_matched_iou | 0.8832 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | fp_per_image | 0.8916 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | fn_per_image | 1.1328 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | precision | 0.5862 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | recall | 0.6058 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | f1 | 0.5958 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | mean_matched_iou | 0.8836 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | fp_per_image | 1.2818 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | fn_per_image | 1.1816 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | precision | 0.4635 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | recall | 0.2414 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | f1 | 0.3175 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | mean_matched_iou | 0.8794 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | fp_per_image | 0.8374 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | fn_per_image | 2.2737 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | precision | 0.4586 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | recall | 0.1854 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | f1 | 0.264 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | mean_matched_iou | 0.8782 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | fp_per_image | 0.6558 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | fn_per_image | 2.4417 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | precision | 0.5648 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | recall | 0.1655 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | f1 | 0.2559 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | mean_matched_iou | 0.8851 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | fp_per_image | 0.3821 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | fn_per_image | 2.5014 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | precision | 0.5269 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | recall | 0.4873 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | f1 | 0.5063 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | mean_matched_iou | 0.8842 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | fp_per_image | 1.3117 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | fn_per_image | 1.5366 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | precision | 0.6014 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | recall | 0.4584 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | f1 | 0.5203 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | mean_matched_iou | 0.8857 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | fp_per_image | 0.9106 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | fn_per_image | 1.6233 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | precision | 0.5572 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | recall | 0.4711 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | f1 | 0.5105 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | mean_matched_iou | 0.8814 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | fp_per_image | 1.122 | 369 | 1106 |
| LocateAnything 3B | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | fn_per_image | 1.5854 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | precision | 0.8668 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | recall | 0.4412 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | f1 | 0.5848 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | mean_matched_iou | 0.8873 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | fp_per_image | 0.2033 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | fn_per_image | 1.6748 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | precision | 0.8173 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | recall | 0.2911 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | f1 | 0.4293 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | mean_matched_iou | 0.8761 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | fp_per_image | 0.1951 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | fn_per_image | 2.1247 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | precision | 0.8273 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | recall | 0.2902 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | f1 | 0.4297 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | mean_matched_iou | 0.878 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | fp_per_image | 0.1816 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | fn_per_image | 2.1274 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | precision | 0.9139 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | recall | 0.2016 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | f1 | 0.3304 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | mean_matched_iou | 0.895 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | fp_per_image | 0.0569 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | fn_per_image | 2.393 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | precision | 1.0 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | recall | 0.0036 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | f1 | 0.0072 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | mean_matched_iou | 0.9843 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | fp_per_image | 0.0 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | fn_per_image | 2.9864 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | precision | 1.0 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | recall | 0.0136 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | f1 | 0.0268 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | mean_matched_iou | 0.9255 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | fp_per_image | 0.0 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | fn_per_image | 2.9566 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | precision | 0.9685 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | recall | 0.1112 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | f1 | 0.1995 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | mean_matched_iou | 0.9115 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | fp_per_image | 0.0108 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | fn_per_image | 2.664 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | precision | 0.9328 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | recall | 0.3011 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | f1 | 0.4552 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | mean_matched_iou | 0.8902 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | fp_per_image | 0.065 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | fn_per_image | 2.0949 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | precision | 0.9075 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | recall | 0.4168 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | f1 | 0.5713 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | mean_matched_iou | 0.8949 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | fp_per_image | 0.1274 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | fn_per_image | 1.748 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | precision | 0.8846 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | recall | 0.0208 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | f1 | 0.0406 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | mean_matched_iou | 0.9197 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | fp_per_image | 0.0081 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | fn_per_image | 2.935 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | precision | 0.8854 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | recall | 0.0769 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | f1 | 0.1414 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | mean_matched_iou | 0.9195 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | fp_per_image | 0.0298 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | fn_per_image | 2.7669 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | precision | 0.8816 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | recall | 0.0606 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | f1 | 0.1134 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | mean_matched_iou | 0.9229 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | fp_per_image | 0.0244 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | fn_per_image | 2.8157 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | precision | 0.875 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | recall | 0.038 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | f1 | 0.0728 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | mean_matched_iou | 0.9309 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | fp_per_image | 0.0163 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | fn_per_image | 2.8835 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | precision | 0.8784 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | recall | 0.4376 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | f1 | 0.5842 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | mean_matched_iou | 0.8878 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | fp_per_image | 0.1816 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | fn_per_image | 1.6856 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | precision | 0.9688 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | recall | 0.1962 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | f1 | 0.3263 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | mean_matched_iou | 0.9048 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | fp_per_image | 0.019 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | fn_per_image | 2.4092 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | precision | 0.9444 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | recall | 0.292 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | f1 | 0.4461 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | mean_matched_iou | 0.8989 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | fp_per_image | 0.0515 | 369 | 1106 |
| SAM 3 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | fn_per_image | 2.122 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | precision | 0.8758 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | recall | 0.472 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | f1 | 0.6134 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | mean_matched_iou | 0.8878 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | fp_per_image | 0.2005 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black bag containing mixed household waste | test | fn_per_image | 1.5827 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | precision | 0.799 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | recall | 0.2911 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | f1 | 0.4268 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | mean_matched_iou | 0.8761 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | fp_per_image | 0.2195 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black garbage bag | test | fn_per_image | 2.1247 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | precision | 0.806 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | recall | 0.2893 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | f1 | 0.4258 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | mean_matched_iou | 0.8791 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | fp_per_image | 0.2087 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | black refuse bag | test | fn_per_image | 2.1301 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | precision | 0.912 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | recall | 0.2342 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | f1 | 0.3727 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | mean_matched_iou | 0.8943 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | fp_per_image | 0.0678 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all black mixed-waste bags | test | fn_per_image | 2.2954 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | precision | 1.0 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | recall | 0.0054 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | f1 | 0.0108 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | mean_matched_iou | 0.9635 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | fp_per_image | 0.0 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all grey recycling bags | test | fn_per_image | 2.981 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | precision | 1.0 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | recall | 0.0136 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | f1 | 0.0268 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | mean_matched_iou | 0.9255 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | fp_per_image | 0.0 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all orange waste bags | test | fn_per_image | 2.9566 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | precision | 0.9658 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | recall | 0.1275 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | f1 | 0.2252 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | mean_matched_iou | 0.9131 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | fp_per_image | 0.0136 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | find all white organic-waste bags | test | fn_per_image | 2.6152 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | precision | 0.9161 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | recall | 0.3553 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | f1 | 0.5121 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | mean_matched_iou | 0.8872 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | fp_per_image | 0.0976 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey bag containing recyclable household material | test | fn_per_image | 1.9322 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | precision | 0.91 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | recall | 0.4295 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | f1 | 0.5835 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | mean_matched_iou | 0.8952 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | fp_per_image | 0.1274 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey household recycling bag | test | fn_per_image | 1.71 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | precision | 0.9091 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | recall | 0.0271 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | f1 | 0.0527 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | mean_matched_iou | 0.9214 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | fp_per_image | 0.0081 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | grey recycling bag | test | fn_per_image | 2.916 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | precision | 0.8932 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | recall | 0.0832 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | f1 | 0.1522 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | mean_matched_iou | 0.9129 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | fp_per_image | 0.0298 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange bag containing domestic waste | test | fn_per_image | 2.748 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | precision | 0.8816 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | recall | 0.0606 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | f1 | 0.1134 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | mean_matched_iou | 0.9217 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | fp_per_image | 0.0244 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange garbage bag | test | fn_per_image | 2.8157 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | precision | 0.8868 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | recall | 0.0425 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | f1 | 0.0811 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | mean_matched_iou | 0.9249 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | fp_per_image | 0.0163 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | orange refuse bag | test | fn_per_image | 2.8699 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | precision | 0.8682 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | recall | 0.4467 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | f1 | 0.5899 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | mean_matched_iou | 0.8877 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | fp_per_image | 0.2033 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white bag containing organic household waste | test | fn_per_image | 1.6585 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | precision | 0.966 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | recall | 0.2315 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | f1 | 0.3735 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | mean_matched_iou | 0.9031 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | fp_per_image | 0.0244 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white domestic organic-waste bag | test | fn_per_image | 2.3035 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | precision | 0.9349 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | recall | 0.3246 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | f1 | 0.4819 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | mean_matched_iou | 0.8972 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | fp_per_image | 0.0678 | 369 | 1106 |
| SAM 3.1 | MDWD/20260821-170445-optimized-v2-bounded-sensitivity | white organic-waste bag | test | fn_per_image | 2.0244 | 369 | 1106 |

Slices deviating >= 0.03 from their aggregate (sufficient support only):

| Model | Split | Slice | Metric | Value | Aggregate | Delta | Support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SAM 3 | test | contrast=high | precision | 0.0 | 1.0 | -1.0 | 123 |
| SAM 3 | test | contrast=low | precision | 0.0 | 1.0 | -1.0 | 123 |
| SAM 3 | test | object_density=high-clutter | precision | 0.0 | 1.0 | -1.0 | 68 |
| SAM 3 | test | object_density=single-object | precision | 0.0 | 1.0 | -1.0 | 142 |
| SAM 3 | test | sharpness=sharp | precision | 0.0 | 1.0 | -1.0 | 123 |
| SAM 3.1 | test | contrast=high | precision | 0.0 | 1.0 | -1.0 | 123 |
| SAM 3.1 | test | contrast=low | precision | 0.0 | 1.0 | -1.0 | 123 |
| SAM 3.1 | test | object_density=high-clutter | precision | 0.0 | 1.0 | -1.0 | 68 |
| SAM 3.1 | test | object_density=single-object | precision | 0.0 | 1.0 | -1.0 | 142 |
| SAM 3.1 | test | sharpness=sharp | precision | 0.0 | 1.0 | -1.0 | 123 |
| SAM 3 | test | object_size=medium | mean_matched_iou | 0.0 | 0.9843 | -0.9843 | 409 |
| SAM 3 | test | object_size=small | mean_matched_iou | 0.0 | 0.9843 | -0.9843 | 103 |
| SAM 3 | test | class=Mixed Waste | mean_matched_iou | 0.0 | 0.9843 | -0.9843 | 334 |
| SAM 3 | test | class=Orange CMD | mean_matched_iou | 0.0 | 0.9843 | -0.9843 | 78 |
| SAM 3 | test | class=Organic Waste | mean_matched_iou | 0.0 | 0.9843 | -0.9843 | 178 |
| SAM 3 | test | class=Recyclable Material | mean_matched_iou | 0.0 | 0.9843 | -0.9843 | 238 |
| SAM 3.1 | test | object_size=medium | mean_matched_iou | 0.0 | 0.9635 | -0.9635 | 409 |
| SAM 3.1 | test | object_size=small | mean_matched_iou | 0.0 | 0.9635 | -0.9635 | 103 |
| SAM 3.1 | test | class=Mixed Waste | mean_matched_iou | 0.0 | 0.9635 | -0.9635 | 334 |
| SAM 3.1 | test | class=Orange CMD | mean_matched_iou | 0.0 | 0.9635 | -0.9635 | 78 |

## PromptDetect-MTSD - status: **final**

| Model | Run | Split | Metric | Value | Images | Objects |
| --- | --- | --- | --- | --- | --- | --- |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | precision | 0.4407 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | recall | 0.1522 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | f1 | 0.2262 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | mean_matched_iou | 0.9009 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | fp_per_image | 0.5114 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | fn_per_image | 2.245 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | precision | 0.5337 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | recall | 0.1643 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | f1 | 0.2513 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | mean_matched_iou | 0.8992 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | fp_per_image | 0.3802 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | fn_per_image | 2.2129 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | precision | 0.6723 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | recall | 0.1795 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | f1 | 0.2833 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | mean_matched_iou | 0.8711 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | fp_per_image | 0.2316 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | fn_per_image | 2.1727 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | precision | 0.66 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | recall | 0.2002 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | f1 | 0.3072 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | mean_matched_iou | 0.8773 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | fp_per_image | 0.2731 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | fn_per_image | 2.1178 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | precision | 0.1954 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | recall | 0.1026 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | f1 | 0.1346 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | mean_matched_iou | 0.8936 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | fp_per_image | 1.1191 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | fn_per_image | 2.3762 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | precision | 0.7077 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | recall | 0.2619 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | f1 | 0.3823 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | mean_matched_iou | 0.8887 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | fp_per_image | 0.2865 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | fn_per_image | 1.9545 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | precision | 0.7303 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | recall | 0.2245 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | f1 | 0.3434 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | mean_matched_iou | 0.8872 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | fp_per_image | 0.2195 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | fn_per_image | 2.0535 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | precision | 0.7188 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | recall | 0.2417 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | f1 | 0.3617 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | mean_matched_iou | 0.8946 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | fp_per_image | 0.2503 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | fn_per_image | 2.008 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | precision | 0.7065 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | recall | 0.2361 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | f1 | 0.3539 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | mean_matched_iou | 0.8953 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | fp_per_image | 0.2597 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | fn_per_image | 2.0228 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | precision | 0.7106 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | recall | 0.2396 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | f1 | 0.3584 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | mean_matched_iou | 0.8972 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | fp_per_image | 0.2584 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | fn_per_image | 2.0134 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | precision | 0.7214 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | recall | 0.2422 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | f1 | 0.3626 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | mean_matched_iou | 0.8955 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | fp_per_image | 0.2477 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | fn_per_image | 2.0067 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | precision | 0.7219 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | recall | 0.227 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | f1 | 0.3454 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | mean_matched_iou | 0.8998 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | fp_per_image | 0.2316 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | fn_per_image | 2.0469 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | precision | 0.5117 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | recall | 0.1552 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | f1 | 0.2382 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | mean_matched_iou | 0.8936 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | fp_per_image | 0.3922 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | fn_per_image | 2.2369 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | precision | 0.72 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | recall | 0.2366 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | f1 | 0.3562 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | mean_matched_iou | 0.8997 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | fp_per_image | 0.2436 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | fn_per_image | 2.0214 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | precision | 0.726 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | recall | 0.2558 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | f1 | 0.3783 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | mean_matched_iou | 0.9044 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | fp_per_image | 0.2557 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | fn_per_image | 1.9705 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | precision | 0.725 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | recall | 0.2452 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | f1 | 0.3665 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | mean_matched_iou | 0.8981 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | fp_per_image | 0.2463 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | fn_per_image | 1.9987 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | precision | 0.7664 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | recall | 0.2305 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | f1 | 0.3545 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | mean_matched_iou | 0.9089 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | fp_per_image | 0.1861 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | fn_per_image | 2.0375 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | precision | 0.7635 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | recall | 0.2578 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | f1 | 0.3855 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | mean_matched_iou | 0.907 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | fp_per_image | 0.2115 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | fn_per_image | 1.9652 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | precision | 0.7345 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | recall | 0.2462 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | f1 | 0.3688 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | mean_matched_iou | 0.8979 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | fp_per_image | 0.2356 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | fn_per_image | 1.996 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | precision | 0.7662 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | recall | 0.227 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | f1 | 0.3502 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | mean_matched_iou | 0.8911 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | fp_per_image | 0.1834 | 747 | 1978 |
| Cosmos Reason2 2B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | fn_per_image | 2.0469 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | precision | 0.5478 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | recall | 0.1623 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | f1 | 0.2504 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | mean_matched_iou | 0.9084 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | fp_per_image | 0.3548 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | fn_per_image | 2.2182 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | precision | 0.6336 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | recall | 0.1259 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | f1 | 0.21 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | mean_matched_iou | 0.8963 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | fp_per_image | 0.1928 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | fn_per_image | 2.3146 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | precision | 0.74 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | recall | 0.2144 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | f1 | 0.3324 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | mean_matched_iou | 0.8946 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | fp_per_image | 0.1995 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | fn_per_image | 2.0803 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | precision | 0.6152 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | recall | 0.1309 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | f1 | 0.2159 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | mean_matched_iou | 0.8783 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | fp_per_image | 0.2169 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | fn_per_image | 2.3012 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | precision | 0.5039 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | recall | 0.0647 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | f1 | 0.1147 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | mean_matched_iou | 0.875 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | fp_per_image | 0.1687 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | fn_per_image | 2.4766 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | precision | 0.7694 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | recall | 0.2007 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | f1 | 0.3184 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | mean_matched_iou | 0.9029 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | fp_per_image | 0.1593 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | fn_per_image | 2.1165 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | precision | 0.8254 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | recall | 0.227 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | f1 | 0.3561 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | mean_matched_iou | 0.9164 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | fp_per_image | 0.1272 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | fn_per_image | 2.0469 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | precision | 0.7899 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | recall | 0.2204 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | f1 | 0.3447 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | mean_matched_iou | 0.9103 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | fp_per_image | 0.1553 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | fn_per_image | 2.0643 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | precision | 0.7385 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | recall | 0.2199 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | f1 | 0.3389 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | mean_matched_iou | 0.8986 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | fp_per_image | 0.2062 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | fn_per_image | 2.0656 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | precision | 0.7757 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | recall | 0.1714 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | f1 | 0.2807 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | mean_matched_iou | 0.8906 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | fp_per_image | 0.1312 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | fn_per_image | 2.1941 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | precision | 0.8122 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | recall | 0.2295 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | f1 | 0.3579 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | mean_matched_iou | 0.9075 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | fp_per_image | 0.1406 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | fn_per_image | 2.0402 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | precision | 0.8715 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | recall | 0.1749 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | f1 | 0.2914 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | mean_matched_iou | 0.928 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | fp_per_image | 0.0683 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | fn_per_image | 2.1847 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | precision | 0.5952 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | recall | 0.1375 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | f1 | 0.2234 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | mean_matched_iou | 0.8909 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | fp_per_image | 0.2477 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | fn_per_image | 2.2838 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | precision | 0.7869 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | recall | 0.2073 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | f1 | 0.3281 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | mean_matched_iou | 0.9088 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | fp_per_image | 0.1486 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | fn_per_image | 2.0991 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | precision | 0.7117 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | recall | 0.1198 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | f1 | 0.2051 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | mean_matched_iou | 0.8849 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | fp_per_image | 0.1285 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | fn_per_image | 2.3307 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | precision | 0.8368 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | recall | 0.2644 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | f1 | 0.4018 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | mean_matched_iou | 0.9175 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | fp_per_image | 0.1365 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | fn_per_image | 1.9478 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | precision | 0.8054 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | recall | 0.1047 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | f1 | 0.1852 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | mean_matched_iou | 0.8899 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | fp_per_image | 0.0669 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | fn_per_image | 2.3708 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | precision | 0.868 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | recall | 0.1795 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | f1 | 0.2974 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | mean_matched_iou | 0.9269 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | fp_per_image | 0.0723 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | fn_per_image | 2.1727 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | precision | 0.829 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | recall | 0.1936 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | f1 | 0.3139 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | mean_matched_iou | 0.9142 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | fp_per_image | 0.1058 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | fn_per_image | 2.1352 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | precision | 0.7849 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | recall | 0.1421 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | f1 | 0.2406 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | mean_matched_iou | 0.8978 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | fp_per_image | 0.1031 | 747 | 1978 |
| Cosmos Reason2 8B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | fn_per_image | 2.2718 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | precision | 0.1529 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | recall | 0.1021 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | f1 | 0.1225 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | mean_matched_iou | 0.9442 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | fp_per_image | 1.498 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | fn_per_image | 2.3775 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | precision | 0.1918 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | recall | 0.2053 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | f1 | 0.1983 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | mean_matched_iou | 0.9243 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | fp_per_image | 2.2905 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | fn_per_image | 2.1044 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | precision | 0.2868 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | recall | 0.3155 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | f1 | 0.3004 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | mean_matched_iou | 0.9288 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | fp_per_image | 2.0776 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | fn_per_image | 1.8126 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | precision | 0.1935 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | recall | 0.1547 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | f1 | 0.172 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | mean_matched_iou | 0.9081 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | fp_per_image | 1.7068 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | fn_per_image | 2.2383 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | precision | 0.0831 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | recall | 0.0799 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | f1 | 0.0815 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | mean_matched_iou | 0.9357 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | fp_per_image | 2.3333 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | fn_per_image | 2.4364 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | precision | 0.2344 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | recall | 0.3923 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | f1 | 0.2934 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | mean_matched_iou | 0.9179 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | fp_per_image | 3.3936 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | fn_per_image | 1.6091 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | precision | 0.269 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | recall | 0.2573 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | f1 | 0.263 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | mean_matched_iou | 0.9195 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | fp_per_image | 1.8514 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | fn_per_image | 1.9665 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | precision | 0.2994 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | recall | 0.2796 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | f1 | 0.2892 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | mean_matched_iou | 0.9331 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | fp_per_image | 1.7323 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | fn_per_image | 1.9076 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | precision | 0.2635 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | recall | 0.094 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | f1 | 0.1386 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | mean_matched_iou | 0.9138 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | fp_per_image | 0.6961 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | fn_per_image | 2.3989 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | precision | 0.2197 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | recall | 0.362 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | f1 | 0.2734 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | mean_matched_iou | 0.9158 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | fp_per_image | 3.4043 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | fn_per_image | 1.6894 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | precision | 0.2482 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | recall | 0.3746 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | f1 | 0.2985 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | mean_matched_iou | 0.9333 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | fp_per_image | 3.0054 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | fn_per_image | 1.656 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | precision | 0.2464 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | recall | 0.3756 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | f1 | 0.2976 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | mean_matched_iou | 0.9297 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | fp_per_image | 3.0415 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | fn_per_image | 1.6533 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | precision | 0.1641 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | recall | 0.134 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | f1 | 0.1475 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | mean_matched_iou | 0.9317 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | fp_per_image | 1.8072 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | fn_per_image | 2.2932 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | precision | 0.3465 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | recall | 0.4211 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | f1 | 0.3802 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | mean_matched_iou | 0.9287 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | fp_per_image | 2.1031 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | fn_per_image | 1.5328 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | precision | 0.2174 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | recall | 0.271 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | f1 | 0.2413 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | mean_matched_iou | 0.9214 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | fp_per_image | 2.5823 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | fn_per_image | 1.9304 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | precision | 0.3222 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | recall | 0.2492 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | f1 | 0.2811 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | mean_matched_iou | 0.9295 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | fp_per_image | 1.3882 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | fn_per_image | 1.988 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | precision | 0.2297 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | recall | 0.4474 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | f1 | 0.3035 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | mean_matched_iou | 0.9164 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | fp_per_image | 3.9732 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | fn_per_image | 1.4632 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | precision | 0.2582 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | recall | 0.3933 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | f1 | 0.3118 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | mean_matched_iou | 0.9233 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | fp_per_image | 2.992 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | fn_per_image | 1.6064 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | precision | 0.2479 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | recall | 0.4429 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | f1 | 0.3179 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | mean_matched_iou | 0.9226 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | fp_per_image | 3.5569 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | fn_per_image | 1.4752 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | precision | 0.2596 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | recall | 0.458 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | f1 | 0.3314 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | mean_matched_iou | 0.9209 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | fp_per_image | 3.4592 | 747 | 1978 |
| LocateAnything 3B | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | fn_per_image | 1.4351 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | precision | 0.8478 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | recall | 0.0394 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | f1 | 0.0754 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | mean_matched_iou | 0.9585 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | fp_per_image | 0.0187 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | fn_per_image | 2.5435 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | precision | 0.4705 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | recall | 0.455 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | f1 | 0.4626 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | mean_matched_iou | 0.9089 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | fp_per_image | 1.3561 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | fn_per_image | 1.4431 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | precision | 0.8425 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | recall | 0.3028 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | f1 | 0.4455 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | mean_matched_iou | 0.9406 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | fp_per_image | 0.1499 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | fn_per_image | 1.8461 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | precision | 0.7795 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | recall | 0.2538 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | f1 | 0.3829 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | mean_matched_iou | 0.9288 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | fp_per_image | 0.1901 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | fn_per_image | 1.9759 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | precision | 0.5409 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | recall | 0.1304 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | f1 | 0.2102 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | mean_matched_iou | 0.9385 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | fp_per_image | 0.2932 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | fn_per_image | 2.3025 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | precision | 0.8102 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | recall | 0.4014 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | f1 | 0.5368 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | mean_matched_iou | 0.9312 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | fp_per_image | 0.249 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | fn_per_image | 1.585 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | precision | 0.8485 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | recall | 0.4019 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | f1 | 0.5455 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | mean_matched_iou | 0.9221 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | fp_per_image | 0.1901 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | fn_per_image | 1.5837 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | precision | 0.6817 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | recall | 0.3357 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | f1 | 0.4499 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | mean_matched_iou | 0.9388 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | fp_per_image | 0.415 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | fn_per_image | 1.759 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | precision | 0.8425 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | recall | 0.0541 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | f1 | 0.1017 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | mean_matched_iou | 0.9328 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | fp_per_image | 0.0268 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | fn_per_image | 2.5047 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | precision | 0.7597 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | recall | 0.4267 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | f1 | 0.5465 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | mean_matched_iou | 0.9142 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | fp_per_image | 0.3574 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | fn_per_image | 1.5181 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | precision | 0.7034 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | recall | 0.4651 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | f1 | 0.56 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | mean_matched_iou | 0.9273 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | fp_per_image | 0.5194 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | fn_per_image | 1.4163 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | precision | 0.5375 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | recall | 0.4242 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | f1 | 0.4741 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | mean_matched_iou | 0.9048 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | fp_per_image | 0.9665 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | fn_per_image | 1.5248 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | precision | 0.2483 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | recall | 0.0728 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | f1 | 0.1126 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | mean_matched_iou | 0.8906 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | fp_per_image | 0.5837 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | fn_per_image | 2.4552 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | precision | 0.7448 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | recall | 0.2523 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | f1 | 0.3769 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | mean_matched_iou | 0.949 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | fp_per_image | 0.2289 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | fn_per_image | 1.9799 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | precision | 0.7348 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | recall | 0.1835 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | f1 | 0.2937 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | mean_matched_iou | 0.9257 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | fp_per_image | 0.1754 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | fn_per_image | 2.162 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | precision | 0.8673 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | recall | 0.1785 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | f1 | 0.296 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | mean_matched_iou | 0.932 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | fp_per_image | 0.0723 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | fn_per_image | 2.1754 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | precision | 0.7027 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | recall | 0.5187 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | f1 | 0.5969 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | mean_matched_iou | 0.9282 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | fp_per_image | 0.581 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | fn_per_image | 1.2744 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | precision | 0.6603 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | recall | 0.5051 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | f1 | 0.5723 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | mean_matched_iou | 0.9121 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | fp_per_image | 0.6881 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | fn_per_image | 1.3106 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | precision | 0.5616 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | recall | 0.4889 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | f1 | 0.5227 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | mean_matched_iou | 0.911 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | fp_per_image | 1.0107 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | fn_per_image | 1.3534 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | precision | 0.7455 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | recall | 0.3109 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | f1 | 0.4388 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | mean_matched_iou | 0.9437 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | fp_per_image | 0.2811 | 747 | 1978 |
| SAM 3 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | fn_per_image | 1.8246 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | precision | 0.8265 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | recall | 0.041 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | f1 | 0.078 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | mean_matched_iou | 0.9595 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | fp_per_image | 0.0228 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | blind-spot convex mirror | test | fn_per_image | 2.5395 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | precision | 0.4663 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | recall | 0.4788 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | f1 | 0.4724 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | mean_matched_iou | 0.9078 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | fp_per_image | 1.4511 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | convex traffic mirror used to improve road visibility | test | fn_per_image | 1.3802 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | precision | 0.8439 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | recall | 0.317 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | f1 | 0.4609 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | mean_matched_iou | 0.9376 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | fp_per_image | 0.1553 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all no entry signs | test | fn_per_image | 1.8086 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | precision | 0.7714 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | recall | 0.2594 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | f1 | 0.3882 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | mean_matched_iou | 0.9282 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | fp_per_image | 0.2035 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all pedestrian crossing signs | test | fn_per_image | 1.9612 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | precision | 0.5036 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | recall | 0.1411 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | f1 | 0.2204 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | mean_matched_iou | 0.9429 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | fp_per_image | 0.3681 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roadside blind-spot mirrors | test | fn_per_image | 2.2744 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | precision | 0.8014 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | recall | 0.408 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | f1 | 0.5407 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | mean_matched_iou | 0.9297 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | fp_per_image | 0.2677 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all roundabout ahead signs | test | fn_per_image | 1.5676 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | precision | 0.8431 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | recall | 0.4075 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | f1 | 0.5494 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | mean_matched_iou | 0.9209 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | fp_per_image | 0.2008 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | find all stop signs | test | fn_per_image | 1.5689 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | precision | 0.667 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | recall | 0.3382 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | f1 | 0.4488 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | mean_matched_iou | 0.9375 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | fp_per_image | 0.4471 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | no entry sign | test | fn_per_image | 1.7523 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | precision | 0.811 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | recall | 0.0521 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | f1 | 0.0979 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | mean_matched_iou | 0.9361 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | fp_per_image | 0.0321 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | pedestrian crossing sign | test | fn_per_image | 2.51 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | precision | 0.7562 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | recall | 0.4312 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | f1 | 0.5493 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | mean_matched_iou | 0.913 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | fp_per_image | 0.3681 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign for a pedestrian crossing | test | fn_per_image | 1.506 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | precision | 0.6908 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | recall | 0.4676 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | f1 | 0.5577 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | mean_matched_iou | 0.9274 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | fp_per_image | 0.5542 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign indicating no entry | test | fn_per_image | 1.4096 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | precision | 0.5308 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | recall | 0.4181 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | f1 | 0.4678 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | mean_matched_iou | 0.9028 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | fp_per_image | 0.9786 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | road sign instructing drivers to stop | test | fn_per_image | 1.5408 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | precision | 0.2367 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | recall | 0.0834 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | f1 | 0.1234 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | mean_matched_iou | 0.8993 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | fp_per_image | 0.7122 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roadside convex mirror for blind spots | test | fn_per_image | 2.427 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | precision | 0.7197 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | recall | 0.2674 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | f1 | 0.39 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | mean_matched_iou | 0.9474 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | fp_per_image | 0.2758 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | roundabout ahead sign | test | fn_per_image | 1.9398 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | precision | 0.731 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | recall | 0.1704 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | f1 | 0.2763 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | mean_matched_iou | 0.9209 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | fp_per_image | 0.166 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | sign indicating a pedestrian crossing | test | fn_per_image | 2.1968 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | precision | 0.8564 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | recall | 0.178 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | f1 | 0.2947 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | mean_matched_iou | 0.931 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | fp_per_image | 0.079 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | stop sign | test | fn_per_image | 2.1767 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | precision | 0.6844 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | recall | 0.5404 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | f1 | 0.604 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | mean_matched_iou | 0.9253 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | fp_per_image | 0.66 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating an upcoming roundabout | test | fn_per_image | 1.2169 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | precision | 0.6496 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | recall | 0.5126 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | f1 | 0.573 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | mean_matched_iou | 0.9105 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | fp_per_image | 0.7323 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign indicating that vehicles must stop | test | fn_per_image | 1.2905 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | precision | 0.5535 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | recall | 0.5096 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | f1 | 0.5307 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | mean_matched_iou | 0.9083 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | fp_per_image | 1.0884 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | traffic sign prohibiting vehicles from entering | test | fn_per_image | 1.2985 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | precision | 0.7287 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | recall | 0.3327 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | f1 | 0.4568 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | mean_matched_iou | 0.9424 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | fp_per_image | 0.328 | 747 | 1978 |
| SAM 3.1 | MTSD/20260822-102455-optimized-v2-bounded-sensitivity | warning sign for a roundabout ahead | test | fn_per_image | 1.7671 | 747 | 1978 |

Slices deviating >= 0.03 from their aggregate (sufficient support only):

| Model | Split | Slice | Metric | Value | Aggregate | Delta | Support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SAM 3.1 | test | object_size=small | mean_matched_iou | 0.0 | 0.9595 | -0.9595 | 193 |
| SAM 3.1 | test | class=Auxiliary Sign | mean_matched_iou | 0.0 | 0.9595 | -0.9595 | 120 |
| SAM 3.1 | test | class=Directional Sign | mean_matched_iou | 0.0 | 0.9595 | -0.9595 | 108 |
| SAM 3.1 | test | class=No Entry (One Way) | mean_matched_iou | 0.0 | 0.9595 | -0.9595 | 211 |
| SAM 3.1 | test | class=No Through Road (T-Junction) | mean_matched_iou | 0.0 | 0.9595 | -0.9595 | 61 |
| SAM 3.1 | test | class=Pedestrian Crossing | mean_matched_iou | 0.0 | 0.9595 | -0.9595 | 143 |
| SAM 3.1 | test | class=Roundabout Ahead | mean_matched_iou | 0.0 | 0.9595 | -0.9595 | 67 |
| SAM 3.1 | test | class=Stop Sign | mean_matched_iou | 0.0 | 0.9595 | -0.9595 | 130 |
| SAM 3.1 | test | class=Street Sign | mean_matched_iou | 0.0 | 0.9595 | -0.9595 | 94 |
| SAM 3 | test | object_size=small | mean_matched_iou | 0.0 | 0.9585 | -0.9585 | 193 |
| SAM 3 | test | class=Auxiliary Sign | mean_matched_iou | 0.0 | 0.9585 | -0.9585 | 120 |
| SAM 3 | test | class=Directional Sign | mean_matched_iou | 0.0 | 0.9585 | -0.9585 | 108 |
| SAM 3 | test | class=No Entry (One Way) | mean_matched_iou | 0.0 | 0.9585 | -0.9585 | 211 |
| SAM 3 | test | class=No Through Road (T-Junction) | mean_matched_iou | 0.0 | 0.9585 | -0.9585 | 61 |
| SAM 3 | test | class=Pedestrian Crossing | mean_matched_iou | 0.0 | 0.9585 | -0.9585 | 143 |
| SAM 3 | test | class=Roundabout Ahead | mean_matched_iou | 0.0 | 0.9585 | -0.9585 | 67 |
| SAM 3 | test | class=Stop Sign | mean_matched_iou | 0.0 | 0.9585 | -0.9585 | 130 |
| SAM 3 | test | class=Street Sign | mean_matched_iou | 0.0 | 0.9585 | -0.9585 | 94 |
| LocateAnything 3B | test | object_size=small | mean_matched_iou | 0.0 | 0.9442 | -0.9442 | 193 |
| LocateAnything 3B | test | class=Roundabout Ahead | mean_matched_iou | 0.0 | 0.9442 | -0.9442 | 67 |

> 200 slice rows have support below 15 and are marked `insufficient_support`.

## Figures

- `Documents/Final-Figures/Robustness-Slices/20260908-103240/mdwd_recall_by_object_size_test.png`
- `Documents/Final-Figures/Robustness-Slices/20260908-103240/mdwd_f1_by_brightness_test.png`
- `Documents/Final-Figures/Robustness-Slices/20260908-103240/mdwd_f1_by_sharpness_test.png`
- `Documents/Final-Figures/Robustness-Slices/20260908-103240/mdwd_per_class_f1_test.png`
- `Documents/Final-Figures/Robustness-Slices/20260908-103240/mtsd_attr_macro_f1_by_head.png`
- `Documents/Final-Figures/Robustness-Slices/20260908-103240/mtsd_attr_condition_per_class_f1.png`
- `Documents/Final-Figures/Robustness-Slices/20260908-103240/mtsd_attr_view_angle_per_class_f1.png`
- `Documents/Final-Figures/Robustness-Slices/20260908-103240/promptdetect_model_prompt_f1.png`
- `Documents/Final-Figures/Robustness-Slices/20260908-103240/mdwd_slice_degradation_test.png`

*Read-only analysis - no datasets, checkpoints or previous results were modified.*
