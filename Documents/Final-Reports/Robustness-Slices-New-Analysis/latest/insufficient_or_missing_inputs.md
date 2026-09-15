# Robustness slices - missing or insufficient inputs

Generated: 2026-09-14T17:47:15

- VJEPA 2.1-L LoRA crop-condition slices: all 1,890 crop records and crop images still exist, but no per-crop prediction or logit records were retained. Only aggregate metrics/confusion matrices and the checkpoint remain. Per instruction, this branch was stopped: no terciles were computed and no detector thresholds were reused.

Slices marked `insufficient_support` (support < 15): 7 of 187 rows (retained in the CSV/JSON, never highlighted).
