# Existing MTSD supervised-detection experiment audit

Generated: 2026-08-12T12:32:22.516691+00:00
Repository commit at audit time: `168260f936f8eb9edea828089182c825c5d7a020`

This is a read-only inventory. Existing run folders, checkpoints, datasets, W&B metadata, and result exports were not modified.

## Scope and evidence

- Canonical run-directory rows: 36.
- Orchestration-state-only rows: 2; these capture attempts whose state exists but whose canonical run record is absent or not complete.
- Local W&B run directories inspected: 35.
- Status counts: {'completed': 31, 'failed': 1, 'partial': 4, 'pending': 2}.
- Augmentation-regime counts: {'Good': 20, 'NoAug': 3, 'Strong': 11, 'Unknown': 4}.
- `val_*` fields are preferred for selection/ablation. `test_*` fields are retained only when an existing run record contains a test/unified-test evaluation.
- `early_stopping_epoch` is recorded as the last completed epoch when the run used the configured early-stopping policy; `best_epoch` is the best recorded validation epoch.
- Library versions are recorded from the reproducible `mtsd-base` environment available locally; historical run records did not persist a separate Ultralytics/RF-DETR version field.

## Prepared MTSD variants

| Regime | Dataset version | Source train | Physical train | Multiplier | Valid | Test | Recipe | Train-only evidence |
|---|---|---:|---:|---:|---:|---:|---|---|
| Good | `mtsd-qa-v1-aug` | 5986 | 23944 | 4.0 | 749 | 747 | `photometric-v1+motion-blur-v1` | True |
| NoAug | `mtsd-qa-v1-noaug` | 5986 | 5986 | 1.0 | 749 | 747 | `none` | True |
| Strong | `mtsd-qa-v1-strong-offline-v2` | 5986 | 23944 | 4.0 | 749 | 747 | `strong-offline-v2` | True |

The three split manifests were compared by source image hash/name: train/valid/test membership is identical across NoAug, Good, and Strong. Validation and test physical image counts remain the original 749/747; only training is expanded. The current prepared corpora contain 5,986 source training rows (5,985 unique source hashes) and 23,944 physical training images for Good/Strong, a 4.0× exposure multiplier. Exact batches and optimizer steps depend on the run-specific physical/effective batch and are included in the CSV.

## Verified missing/failed patterns relevant to the requested sweep

- Strong YOLO11 at 1280: S and M are complete; N is absent.
- Strong YOLO26 at 1280: S and M are complete; N is absent.
- Strong YOLO12: S at 960 is complete; the prior M at 960 attempt was stopped as impractically slow. No common 960 N/S/M matrix exists.
- Strong RF-DETR native: N at 384 is complete; S at 512 and M at 576 are absent. Strong S@640 and M@704 are separate resolution follow-ups and are retained.
- The CSV contains the evidence used to decide whether an apparent filename match is actually a completed equivalent; filenames alone were not used as completion proof.

## Output

Machine-readable inventory: `existing_mtSD_experiment_audit.csv`

The inventory is regenerated with `Scripts/MTSD-Scripts/MTSD-SupervisedDetection/audit_experiments.py`.
