# MTSD attribute bootstrap audit

**Result: PASS.** The existing uncertainty rows correspond to all 18 non-smoke final metric files.

## Validity

A confusion matrix is a sufficient statistic for an unpaired marginal per-head resample: each test crop contributes exactly one `(true, predicted)` cell. Re-expanding the counts therefore preserves accuracy, macro-F1, each class F1, class support and the 1,890-crop unit count. It cannot restore crop identity or align outcomes across heads/models.

The report is labelled **historical snapshot** solely because `discover_attribute_variants()` assigns that status to all non-smoke metric folders. It does not mean the 18 runs are superseded; the current Evaluation chapter cites these same files as the final 1,890-crop evidence.

## Limits retained

- No CI for the four-head mean macro-F1: joint crop-level outcomes across heads are not retained.
- No paired model-vs-model bootstrap: aligned per-crop predictions are not retained.
- No source-image-clustered bootstrap: source-image identifiers are not retained with predictions.

## Documentation check

The current Evaluation table correctly contains 18 data rows (DINOv3, V-JEPA, ConvNeXt and LingBot-Vision configurations across the listed frozen, LoRA and full-fine-tuning strategies).

## Files

- `final_18_configurations.csv`: exact model/backbone, adaptation, run and metrics file for all final configurations.
- `final_attribute_head_ci_table.csv`: requested 72-row, four-head accuracy and macro-F1 table with CIs.
- `supported_per_class_f1_bootstrap.csv`: existing per-class CIs where support >=20.
- `configuration_provenance_audit.csv`: per-head identity and reconstruction checks.
