# V-JEPA attribute robustness: isolated crops remain strongest for shape and mounting

Generated 2026-09-14T16:13:46+00:00 from deterministic inference over all 1,890 retained MTSD test crops.

> **New inference-only analysis, separate from the retained robustness report.** The saved epoch-12 checkpoint was loaded; nothing was retrained, the manifest was not refreshed, and the test split was not changed.

## Executive summary

- Rerun overall mean macro-F1: **90.00%**; absolute difference from the retained result was 6.66e-04.
- The weakest physical slice was **Original object size / Small** at 75.34%; the strongest was **Crop brightness / Medium** at 90.79%.
- Semantic slices use only other ground-truth attributes. A head is never sliced by its own target, and object density/image position are intentionally absent.

## Rerun aggregate performance and reconciliation

| Head | Macro-F1 | Retained | Absolute difference | Support |
| --- | ---: | ---: | ---: | ---: |
| Viewing angle | 92.97% | 92.97% | 0.00e+00 | 1,890 |
| Mounting | 95.15% | 95.26% | 1.15e-03 | 1,890 |
| Condition | 74.49% | 74.70% | 2.10e-03 | 1,890 |
| Sign shape | 97.38% | 97.32% | 5.91e-04 | 1,890 |
| Overall mean | 90.00% | 90.06% | 6.66e-04 | 1,890 |

The strictly deterministic CUDA rerun exactly reproduced viewing-angle macro-F1. The other heads differed slightly from the retained historical bundle; the largest absolute difference was 0.21 percentage points (condition). The source of this small numerical/prediction discrepancy was not isolated. All slice results use the newly saved, internally consistent per-crop outputs rather than mixing in retained aggregates.

## Original object size

| Slice | Viewing angle | Mounting | Condition | Sign shape | Overall mean | Crops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small | 83.77% | 79.38% | 41.92% | 96.28% | 75.34% | 93 |
| Medium | 87.98% | 95.06% | 67.72% | 96.79% | 86.89% | 451 |
| Large | 94.12% | 95.98% | 74.69% | 97.91% | 90.67% | 1,346 |

Size is based on native-coordinate `bbox_width × bbox_height` before crop padding or resizing, using the detector audit's COCO thresholds.

## Crop brightness

| Slice | Viewing angle | Mounting | Condition | Sign shape | Overall mean | Crops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Low | 90.63% | 92.82% | 70.93% | 96.02% | 87.60% | 630 |
| Medium | 94.31% | 96.63% | 75.21% | 97.01% | 90.79% | 630 |
| High | 92.00% | 94.71% | 75.42% | 98.53% | 90.17% | 630 |

Test-crop tercile thresholds: 104.2858 / 120.1720. Statistics use classifier-input pixels after deterministic padding/resizing and before normalisation.

## Crop contrast

| Slice | Viewing angle | Mounting | Condition | Sign shape | Overall mean | Crops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Low | 91.94% | 94.65% | 72.01% | 95.93% | 88.63% | 630 |
| Medium | 92.46% | 96.34% | 75.85% | 96.72% | 90.34% | 630 |
| High | 91.21% | 94.44% | 75.28% | 99.66% | 90.15% | 630 |

Test-crop tercile thresholds: 37.0250 / 49.6186. Statistics use classifier-input pixels after deterministic padding/resizing and before normalisation.

## Crop sharpness

| Slice | Viewing angle | Mounting | Condition | Sign shape | Overall mean | Crops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Blurred | 87.13% | 91.04% | 61.53% | 96.30% | 84.00% | 630 |
| Intermediate | 92.94% | 96.21% | 75.95% | 97.19% | 90.57% | 630 |
| Sharp | 95.45% | 97.22% | 74.65% | 78.76% | 86.52% | 630 |

Test-crop tercile thresholds: 43.2855 / 293.3200. Statistics use classifier-input pixels after deterministic padding/resizing and before normalisation.

## Semantic-condition robustness

The complete cross-attribute matrix is in `attribute_robustness_slices.csv` and the compact heatmap. The lowest sufficient-support head-level cells are:

| Prediction head | Sliced by | GT value | Macro-F1 | Support |
| --- | --- | --- | ---: | ---: |
| Condition | Sign shape | Pentagon | 29.63% | 36 |
| Mounting | Sign shape | Pentagon | 50.00% | 36 |
| Condition | Sign shape | Triangular | 69.07% | 281 |
| Condition | Mounting | Wall-Mounted | 70.07% | 261 |
| Condition | Viewing angle | Back | 71.39% | 738 |
| Condition | Viewing angle | Side | 71.66% | 453 |
| Condition | Sign shape | Octagonal | 72.90% | 212 |
| Condition | Sign shape | Quadrangle | 74.43% | 615 |

## Support and interpretation

Every row requires support ≥ 15. There are **0** rows below this threshold; they remain in the CSV but are marked `insufficient_support` and excluded from figures and overall slice means.

Separately, **23 head-level rows** contain at least one target class with fewer than 15 examples; **5** contain an entirely absent target class. Their fixed-vocabulary macro-F1 values are retained, but `interpretation_flag`, `class_support`, and `low_support_classes` identify composition-sensitive cells in the CSV. For example, the Sharp sign-shape slice contains no Pentagon examples, which mechanically contributes a zero for that class.

## Protocol boundaries

- Original size uses the retained manifest's native GT bounding box, not the padded crop.
- Brightness, contrast and sharpness use the actual deterministic 384×384 classifier input geometry before tensor normalisation; their thresholds are specific to these 1,890 crops and are not detector thresholds.
- Macro-F1 uses every class in the saved head vocabulary with `zero_division=0`, matching the retained evaluator.
- Semantic slices are observational correlations among attributes, not controlled corruptions or causal effects.
- Isolated GT crops provide no defensible object-density or image-position test.

## Outputs

- `per_crop_predictions_logits.csv`: one aligned row per test crop, including native bbox, derived properties, true/predicted labels, indices and every class logit.
- `attribute_robustness_slices.csv`: aggregate, physical and semantic result rows.
- `run_config.json`, `robustness_thresholds.json`, and `robustness_protocol.json`: exact identities, hashes, preprocessing and rules.
- `missing_or_low_support.md`: input audit and low-support rows.

Figures:

- `Documents/Final-Figures/Attribute-Robustness/20260914-180650/vjepa21_vitl_lora_macro_f1_by_original_object_size.png`
- `Documents/Final-Figures/Attribute-Robustness/20260914-180650/vjepa21_vitl_lora_crop_property_robustness.png`
- `Documents/Final-Figures/Attribute-Robustness/20260914-180650/vjepa21_vitl_lora_semantic_condition_robustness.png`

*Read-only evaluation of retained data/model inputs; only new analysis artifacts were written.*
