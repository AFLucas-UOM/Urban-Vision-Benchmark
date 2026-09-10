# Bootstrap uncertainty - missing or pending inputs

Generated: 2026-07-10T12:04:59

- MTSD attributes convnext/sign_shape/Pentagon: support 11 < 20 - per-class CI skipped (insufficient support).
- MTSD attributes convnext_frozen/sign_shape/Pentagon: support 11 < 20 - per-class CI skipped (insufficient support).
- MTSD attributes dinov3/sign_shape/Pentagon: support 11 < 20 - per-class CI skipped (insufficient support).
- MTSD attributes dinov3_lora/sign_shape/Pentagon: support 11 < 20 - per-class CI skipped (insufficient support).
- MTSD attributes vjepa/sign_shape/Pentagon: support 11 < 20 - per-class CI skipped (insufficient support).
- MTSD attributes vjepa_lora/sign_shape/Pentagon: support 11 < 20 - per-class CI skipped (insufficient support).
- MTSD attributes: cross-head mean macro-F1 CIs and paired variant comparisons need per-crop prediction dumps (crops aligned across heads/variants), which the training pipeline does not store. Add a per-sample prediction export to `mtsd_attr` evaluation and rerun the final-scope attribute round to enable them.
- PromptDetect MDWD run `MDWD/20260710-011505` is a PILOT (10 images): intervals are extremely wide by construction and validate the pipeline only. Final CIs need the full fixed-protocol evaluation.
- PromptDetect MTSD: no batch-evaluation runs - run run_batch_eval.py first; CIs pending.
- MDWD detection: confidence-ranked AP/mAP intervals are deliberately not bootstrapped here - the official mAP protocol evaluates a ranking over the whole split, and a defensible image-level AP bootstrap over the full conf-0.001 prediction set is computationally heavy in pure Python. P/R/F1/matched-IoU intervals above use the documented operating point instead; headline mAP values remain the Ultralytics point estimates.
- MTSD supervised detection: PENDING - no training runs exist yet.
