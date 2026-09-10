# MDWD split-leakage sensitivity analysis

Generated: 2026-07-09T23:46:31  |  Overall: **NEGLIGIBLE**

Purpose: measure whether the cross-split source-image leakage in the
MDWD Roboflow v20 export materially inflated the reported validation/test
detection metrics. Machine-readable outputs sit next to this report
(`leaked_source_identities.csv`, `affected_evaluation_images.csv`,
`subset_counts.csv`, `metric_comparison.csv/.json`).

## 1. Leakage definition and source-identity normalisation

Roboflow names each augmented copy `<source_stem>_<ext>.rf.<hash>.jpg`;
the source identity of a file is the part of its stem before `.rf.`
(`mdwd_eda.mapper.source_stem`, reused verbatim). A source identity is
**leaked** when its augmented derivatives occur in more than one of
train/valid/test. Limitation: the original file extension is folded into
the stem (`_jpg` vs `_jpeg`), so one photograph exported under two
different extensions would count as two identities; this matches the
EDA's definition and errs on the side of distinct sources.

## 2. Leakage audit (independently recomputed)

- Leaked source identities: **30**
- Direction breakdown: train+test: 16, train+valid: 11, valid+test: 3
- Affected validation images: **14** of 369
- Affected test images: **19** of 369
- Identity check: per-split unique stems 2891 + 369 + 368 = 3628 vs 3598 deduplicated -> excess 30, expected from leaked stems 30 (consistent)
- Cross-check vs EDA `integrity_issues.csv`: **MATCH** (EDA 30 vs recomputed 30; only-recomputed 0, only-EDA 0, split-set mismatches 0)

## 3. Clean evaluation subsets

An evaluation image is excluded when its source stem also occurs in any
other split (train or the other evaluation split); valid<->test leaks are
excluded from both clean subsets, so no evaluated image shares a source
with any other split.

| Split | Subset | Images | Source stems | GT boxes | Removed | % removed |
| --- | --- | --- | --- | --- | --- | --- |
| valid | original | 369 | 369 | 1072 | 0 | 0.0 |
| valid | clean | 355 | 355 | 1030 | 14 | 3.79 |
| test | original | 369 | 368 | 1106 | 0 | 0.0 |
| test | clean | 350 | 349 | 1066 | 19 | 5.15 |

## 4. Other integrity issues (separate from leakage)

- Out-of-range boxes: 15 (by split: {'train': 15}); empty annotations: 5 (by split: {'train': 5}).
- In evaluation splits (valid/test): 0; overlapping the clean subsets: 0. They therefore do not influence the evaluation metrics or the leakage sensitivity subsets.
- Loader behaviour: the `mdwd_eda` audit flags any coordinate outside [0, 1] strictly; Ultralytics' training loader tolerates 1% overshoot and rejects the whole image beyond that (coordinate > 1.01 or value < -0.01). Of the 15 flagged label lines, 0 would cause Ultralytics to drop the image from training entirely; the rest train with the slightly out-of-bounds box as-is. Empty label files act as background (negative) images during training.

## 5. Metric sensitivity (original vs leakage-excluded)

Two metric sources per run: `UL` = Ultralytics `model.val()` (same
protocol as the original benchmarks: imgsz=640, conf=0.001,
iou=0.7, batch=32); `SUP` = class-aware greedy matching
reusing PromptDetect's `batch_evaluation/metrics.py` (counts at working
confidence 0.25; AP over all predictions). SUP absolute
values are not comparable to UL (different matcher); interpret only the
original-vs-clean delta within each source. SUP clean-subset metrics are
computed by filtering the original run's predictions, so their delta is a
pure subset-composition effect.

### yolo11l (YOLO11-EUVIP)

Checkpoint: `Results\MDWD-Runs\YOLO11-EUVIP\E004_yolo11l_rfv20_img640_b32_e100_adamw_lr0p001\weights\best.pt`

| Split | Subset | Images | UL P | UL R | UL F1 | UL mAP50 | UL mAP50-95 | SUP F1 | SUP mIoU | SUP FP | SUP FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid | original | 369 | 0.9430 | 0.8614 | 0.9004 | 0.8993 | 0.7614 | 0.8871 | 0.9058 | 86 | 149 |
| valid | clean | 355 | 0.9429 | 0.8665 | 0.9031 | 0.9005 | 0.7662 | 0.8887 | 0.9081 | 83 | 140 |
| test | original | 369 | 0.9564 | 0.8352 | 0.8917 | 0.8865 | 0.7490 | 0.8784 | 0.9014 | 116 | 149 |
| test | clean | 350 | 0.9581 | 0.8340 | 0.8918 | 0.8842 | 0.7476 | 0.8795 | 0.9020 | 106 | 146 |

- valid: delta mAP50 +0.0012 (+0.12 pp), delta mAP50-95 +0.0048 (+0.48 pp) -> **negligible**
- test: delta mAP50 -0.0023 (-0.23 pp), delta mAP50-95 -0.0015 (-0.15 pp) -> **negligible**
- Reported benchmark (Results\MDWD-Results\YOLO11\Model-Size-Comparison\yolo11_multi_model_template_summary.csv): val mAP50 0.8993 / mAP50-95 0.7614; test mAP50 0.8865 / mAP50-95 0.7490. Reported val metrics come from the best training epoch, so small differences vs this standalone re-validation are expected.
- Determinism check (test): 1 of 350 clean-subset images had different prediction counts between the clean run and the filtered original run (batch-composition jitter; SUP deltas use the filtered original predictions and are unaffected).

### yolo12m (YOLO12-EUVIP)

Checkpoint: `Results\MDWD-Runs\YOLO12-EUVIP\E003_yolo12m_rfv20_img640_b32_e100_adamw_lr0p001\weights\best.pt`

| Split | Subset | Images | UL P | UL R | UL F1 | UL mAP50 | UL mAP50-95 | SUP F1 | SUP mIoU | SUP FP | SUP FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid | original | 369 | 0.9416 | 0.8759 | 0.9076 | 0.9147 | 0.7656 | 0.8844 | 0.8970 | 125 | 123 |
| valid | clean | 355 | 0.9423 | 0.8771 | 0.9086 | 0.9155 | 0.7671 | 0.8833 | 0.8972 | 123 | 118 |
| test | original | 369 | 0.9573 | 0.8573 | 0.9045 | 0.9030 | 0.7584 | 0.8888 | 0.8978 | 103 | 139 |
| test | clean | 350 | 0.9570 | 0.8584 | 0.9050 | 0.9036 | 0.7596 | 0.8903 | 0.8982 | 97 | 133 |

- valid: delta mAP50 +0.0008 (+0.08 pp), delta mAP50-95 +0.0014 (+0.14 pp) -> **negligible**
- test: delta mAP50 +0.0005 (+0.05 pp), delta mAP50-95 +0.0011 (+0.11 pp) -> **negligible**
- Reported benchmark (Results\MDWD-Results\YOLO12\Model-Size-Comparison\yolo12_multi_model_template_summary.csv): val mAP50 0.9147 / mAP50-95 0.7656; test mAP50 0.9030 / mAP50-95 0.7584. Reported val metrics come from the best training epoch, so small differences vs this standalone re-validation are expected.

### yolo26l (YOLO26-DGX)

Checkpoint: `Results\MDWD-Runs\YOLO26-DGX\E004_yolo26l_rfv20_img640_b32_e100_adamw_lr0p001\weights\best.pt`

| Split | Subset | Images | UL P | UL R | UL F1 | UL mAP50 | UL mAP50-95 | SUP F1 | SUP mIoU | SUP FP | SUP FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid | original | 369 | 0.9487 | 0.8793 | 0.9127 | 0.9100 | 0.7948 | 0.9010 | 0.9179 | 71 | 135 |
| valid | clean | 355 | 0.9481 | 0.8820 | 0.9139 | 0.9119 | 0.7971 | 0.9011 | 0.9187 | 70 | 128 |
| test | original | 369 | 0.9570 | 0.8701 | 0.9115 | 0.9014 | 0.7810 | 0.9085 | 0.9123 | 69 | 128 |
| test | clean | 350 | 0.9557 | 0.8682 | 0.9099 | 0.9004 | 0.7806 | 0.9070 | 0.9131 | 68 | 125 |

- valid: delta mAP50 +0.0020 (+0.20 pp), delta mAP50-95 +0.0023 (+0.23 pp) -> **negligible**
- test: delta mAP50 -0.0010 (-0.10 pp), delta mAP50-95 -0.0004 (-0.04 pp) -> **negligible**
- Reported benchmark (Results\MDWD-Results\YOLO26-DGX\Model-Size-Comparison\yolo26_multi_model_template_summary.csv): val mAP50 0.9100 / mAP50-95 0.7948; test mAP50 0.9014 / mAP50-95 0.7810. Reported val metrics come from the best training epoch, so small differences vs this standalone re-validation are expected.

### yolo26l (YOLO26-EUVIP)

Checkpoint: `Results\MDWD-Runs\YOLO26-EUVIP\E004_yolo26l_rfv20_img640_b32_e100_adamw_lr0p001\weights\best.pt`

| Split | Subset | Images | UL P | UL R | UL F1 | UL mAP50 | UL mAP50-95 | SUP F1 | SUP mIoU | SUP FP | SUP FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid | original | 369 | 0.9578 | 0.8732 | 0.9136 | 0.9152 | 0.7951 | 0.8981 | 0.9160 | 85 | 129 |
| valid | clean | 355 | 0.9564 | 0.8741 | 0.9134 | 0.9172 | 0.7976 | 0.8971 | 0.9163 | 85 | 123 |
| test | original | 369 | 0.9718 | 0.8698 | 0.9180 | 0.8976 | 0.7824 | 0.9094 | 0.9150 | 56 | 137 |
| test | clean | 350 | 0.9709 | 0.8686 | 0.9169 | 0.8952 | 0.7814 | 0.9094 | 0.9154 | 54 | 132 |

- valid: delta mAP50 +0.0020 (+0.20 pp), delta mAP50-95 +0.0025 (+0.25 pp) -> **negligible**
- test: delta mAP50 -0.0024 (-0.24 pp), delta mAP50-95 -0.0011 (-0.11 pp) -> **negligible**
- Reported benchmark (Results\MDWD-Results\YOLO26\Model-Size-Comparison\yolo26_multi_model_template_summary.csv): val mAP50 0.9152 / mAP50-95 0.7951; test mAP50 0.8976 / mAP50-95 0.7824. Reported val metrics come from the best training epoch, so small differences vs this standalone re-validation are expected.

### Delta classification thresholds

Absolute percentage-point change in mAP50 / mAP50-95 (clean - original);
the headline label per model/split uses the larger |delta| of the two:
negligible <= 0.5 pp < small <= 1.0 pp < moderate <= 2.0 pp < substantial.
Rationale: the clean subsets remove ~4-5% of evaluation images, so
composition noise alone can move mAP by a few tenths of a point;
only shifts clearly above typical seed-to-seed run variance can be
attributed to leakage inflation.

## 6. Method notes and limitations

- No stored per-image predictions existed in the repository (the original
  notebooks ran `model.val()` without `save_json`), so metrics were
  recomputed by checkpoint inference gated behind `--run-inference`.
- Subset evaluation uses a scratch data.yaml pointing at .txt lists of
  absolute image paths inside the untouched dataset; Ultralytics'
  dataset-cache writer is patched to a no-op so no `labels.cache` is
  created or modified anywhere under `Datasets/`.
- RF-DETR is out of scope here: it was trained/evaluated with a different
  framework (COCO evaluator inside the rfdetr package), its benchmark is
  incomplete (single N-scale run), and no Ultralytics-compatible
  re-validation path exists for it.
- The SUP matcher is greedy and per-class (reused from PromptDetect batch
  evaluation); its absolute AP differs from Ultralytics' by construction.
- Environment: ultralytics 8.4.60, torch 2.10.0.dev20251013+cu130, device NVIDIA GeForce RTX 4090.

*Read-only analysis - no dataset, checkpoint, notebook, or historical result files were modified. All outputs live under `Documents/Final-Reports/MDWD-Leakage-Analysis/`. Re-run via `python Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py`.*
