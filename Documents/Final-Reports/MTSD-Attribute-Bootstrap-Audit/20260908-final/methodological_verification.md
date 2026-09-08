# Final methodological verification

## Result: pass

An independent, read-only reproduction checked all **378** retained marginal
confidence intervals against the current 18 non-smoke `test_metrics.json`
files: 72 head-level accuracy/macro-F1 intervals and 234 supported per-class
F1 intervals. All 378 reproduce exactly to the four decimal places stored in
`bootstrap_results.csv`.

Settings reproduced: test crop as the resampling unit, 2,000 resamples with
replacement, `random.Random(42)`, and 95% percentile intervals. The check is
recorded in `exact_ci_reproduction.csv` and `exact_ci_reproduction.json`.

The 18 configurations also have identical ground-truth support vectors,
confirming the common 1,890-crop marginal test scope:

| Head | Class supports |
| --- | --- |
| Viewing angle | 699, 738, 453 |
| Mounting | 1,629, 261 |
| Physical condition | 1,388, 392, 110 |
| Sign shape | 746, 615, 281, 212, 36 |

## What is methodologically sound

For each head separately, the confusion matrix fully determines accuracy,
macro-F1 and each per-class F1. Re-expanding each cell count into individual
`(true label, predicted label)` crop outcomes is therefore valid for an
unpaired, marginal test-crop bootstrap. No inference, retraining, threshold
selection or test-set optimisation occurred.

## What it cannot support

It remains invalid to bootstrap the four-head mean macro-F1, paired
model-versus-model differences, or source-image-clustered intervals from these
matrices. Those quantities require aligned crop-level predictions and,
respectively, source-image identifiers; neither is retained in the metrics
files.

The label **historical snapshot** is a status label hard-coded by the discovery
helper for non-smoke metrics folders. It does not indicate stale or superseded
results in this case.
