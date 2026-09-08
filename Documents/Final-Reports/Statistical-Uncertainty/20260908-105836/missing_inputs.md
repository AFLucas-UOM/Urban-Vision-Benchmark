# Bootstrap uncertainty - missing or pending inputs

Generated: 2026-09-08T10:58:42

- MTSD attributes: cross-head mean macro-F1 CIs and paired variant comparisons need per-crop prediction dumps (crops aligned across heads/variants), which the training pipeline does not store. Add a per-sample prediction export to `mtsd_attr` evaluation and rerun the final-scope attribute round to enable them.
- MDWD detection: confidence-ranked AP/mAP intervals are deliberately not bootstrapped here - the official mAP protocol evaluates a ranking over the whole split, and a defensible image-level AP bootstrap over the full conf-0.001 prediction set is computationally heavy in pure Python. P/R/F1/matched-IoU intervals above use the documented operating point instead; headline mAP values remain the Ultralytics point estimates.
- MTSD supervised detection: PENDING - no training runs exist yet.
