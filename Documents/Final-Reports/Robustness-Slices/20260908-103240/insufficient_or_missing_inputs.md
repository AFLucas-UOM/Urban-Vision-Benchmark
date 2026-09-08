# Robustness slices - missing or insufficient inputs

Generated: 2026-09-08T10:32:54

- MDWD detection: capture-group and geographic slices are not applicable - the Roboflow export carries no capture-group or GPS metadata.
- MTSD supervised detection: PENDING - Results/MTSD-Runs and Results/MTSD-Results are empty. Train via Scripts/MTSD-Scripts/MTSD-SupervisedNotebooks first.
- MTSD attribute classification: object-size / brightness / blur / capture-group slices need per-crop predictions, which the training pipeline does not store (only aggregate metrics + confusion matrices). To enable them, extend `mtsd_attr` evaluation to dump per-crop predictions and rerun the (final) attribute round - do not rerun just for this analysis while the annotation scope is still open.

Slices marked `insufficient_support` (support < 15): 200 of 12810 rows (retained in the CSV/JSON, never highlighted).
