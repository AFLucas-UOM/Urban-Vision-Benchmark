# Robustness slices - missing or insufficient inputs

Generated: 2026-07-10T12:03:57

- MDWD detection: capture-group and geographic slices are not applicable - the Roboflow export carries no capture-group or GPS metadata.
- MTSD supervised detection: PENDING - Results/MTSD-Runs and Results/MTSD-Results are empty. Train via Scripts/MTSD-Scripts/MTSD-SupervisedNotebooks first.
- MTSD attribute classification: object-size / brightness / blur / capture-group slices need per-crop predictions, which the training pipeline does not store (only aggregate metrics + confusion matrices). To enable them, extend `mtsd_attr` evaluation to dump per-crop predictions and rerun the (final) attribute round - do not rerun just for this analysis while the annotation scope is still open.
- PromptDetect MDWD run `MDWD/20260710-011505` is a PILOT (10 images) - slices are reported for pipeline validation only and are not dissertation evidence.
- PromptDetect MTSD: no batch-evaluation runs under `Results/PromptDetect/BatchEvaluation/MTSD` - run `Scripts/Other-Scripts/PromptDetect/batch_evaluation/run_batch_eval.py --dataset MTSD --split test --prompts ... --models ...` first.

Slices marked `insufficient_support` (support < 15): 104 of 820 rows (retained in the CSV/JSON, never highlighted).
