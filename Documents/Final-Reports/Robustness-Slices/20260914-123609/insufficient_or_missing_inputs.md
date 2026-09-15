# Robustness slices - missing or insufficient inputs

Generated: 2026-09-14T12:36:10

- MTSD supervised detection: skipped 5 prediction exports without a completed run record or compatible COCO ground truth.
- MTSD attribute classification: object-size / brightness / blur / capture-group slices need per-crop predictions, which the training pipeline does not store (only aggregate metrics + confusion matrices). To enable them, extend `mtsd_attr` evaluation to dump per-crop predictions and rerun the (final) attribute round - do not rerun just for this analysis while the annotation scope is still open.

Slices marked `insufficient_support` (support < 15): 6 of 4975 rows (retained in the CSV/JSON, never highlighted).
