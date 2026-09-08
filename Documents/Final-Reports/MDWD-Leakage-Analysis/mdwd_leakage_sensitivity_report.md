# MDWD split-leakage sensitivity analysis

Generated: 2026-09-08T10:29:48  |  Overall: **AUDIT-ONLY (metrics pending)**

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

PENDING - run with `--run-inference` to populate this section, e.g.:
```
python Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py --run-inference --models best-per-family
```

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

*Read-only analysis - no dataset, checkpoint, notebook, or historical result files were modified. All outputs live under `Documents/Final-Reports/MDWD-Leakage-Analysis/`. Re-run via `python Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py`.*
