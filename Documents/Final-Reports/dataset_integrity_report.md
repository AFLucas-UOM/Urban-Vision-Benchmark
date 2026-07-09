# Dataset integrity report

Generated: 2026-07-09T20:11:41  |  Overall: **FAIL**

FAIL = data that would corrupt training/evaluation; WARNING = quality issues to acknowledge in the dissertation; full row-level list in `dataset_integrity_issues.csv`.


## MDWD (variant MDWD-YOLO26) — **FAIL**

> Counts are export-level (train is ~10x offline-augmented); unique source images: 3598.

> source_in_multiple_splits = augmented copies of one capture in more than one split (leakage) - list in the issues CSV.

| Issue | Severity | Count |
| --- | --- | --- |
| empty_annotation | WARNING | 5 |
| out_of_range_box | FAIL | 15 |
| source_in_multiple_splits | WARNING | 30 |

| Split | Images | Boxes |
| --- | --- | --- |
| train | 29487 | 209334 |
| valid | 369 | 1072 |
| test | 369 | 1106 |

| Class | instances | share_pct | train | valid | test |
|---|---|---|---|---|---|
| Mixed Waste | 60034 | 28.38 | 59426 | 274 | 334 |
| Recyclable Material | 52096 | 24.63 | 51595 | 263 | 238 |
| Other Waste | 43727 | 20.67 | 43188 | 261 | 278 |
| Organic Waste | 40783 | 19.28 | 40409 | 196 | 178 |
| Orange CMD | 14872 | 7.03 | 14716 | 78 | 78 |

## MTSD QA annotations (Final-QA COCO exports) — **WARNING**

> Duplicate thresholds: exact/conflicting IoU >= 0.95, high-overlap >= 0.75.

> Fixable via the MTSD-AnnotationQA review/apply workflow.

| Issue | Severity | Count |
| --- | --- | --- |
| attr_known_drop_value | WARNING | 5 |
| dup_exact_duplicate | WARNING | 1 |
| dup_high_overlap | WARNING | 4 |

## MTSD prepared detection dataset (MTSD-YOLO) — **PENDING**

> PENDING - not built yet; run Prepare-MTSD-Detection-Dataset.ipynb first.

No issues found.

*Read-only report - no dataset files were modified.*
