# MTSD Supervised Detection

This is the canonical, importable route for MTSD detection preparation, training, unified evaluation, and reporting. The older notebooks are thin interactive front ends; they must not maintain separate split or training logic.

Safe checks from the repository root:

```powershell
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --dry-run
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --validate-prepared --strict
```

Preparation creates `MTSD-Augmented` and `MTSD-Unaugmented` from one split assignment, each with YOLO and COCO layouts plus hashed manifests. Final mode requires the explicit GRP-1 through GRP-11 lock and a resolved `config/qa_gate.yaml`. Auto discovery is development-only. Raw XML requires both fallback flags, creates a distinct mixed version, and is forbidden in final mode.

Preparation now creates a horizontal motion-blurred third training copy (`aug3`) automatically using a kernel size of 9 and blur weight of 0.85. For a dataset that was already prepared with only `aug1` and `aug2`, add `aug3` to both YOLO and COCO layouts with:

```powershell
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/add_mtsd_motion_blur_aug3.py
```

The command is deterministic and safe to rerun; it updates the augmentation and preparation manifests and validates the resulting dataset.

Training requires `--matrix`, `--models`, or `--smoke-test`. Final dissertation training additionally requires `--final` and the augmented variant. Full runs use W&B project `MSc-MTSD-SupervisedDetection`; run directories and resume state are immutable and fingerprint-checked. The clean augmentation ablation is YOLO11m/12m/26m, with YOLO26l optional; RF-DETR is descriptive because its internal online augmentation is not fully controlled.

Unified COCO evaluation clamps predictions to the source image, discards boxes that remain degenerate, and converts RF-DETR inputs to RGB. The default evaluation policy drops `Tourist Sign` from both ground truth and predictions while leaving the training taxonomy unchanged for causal comparability. Each evaluation exports predictions, aggregate COCO metrics, per-class AP/AR, IoU-0.50 PR curves, and one micro-averaged confidence-swept F1 threshold. Rerun a completed matrix without retraining with:

```powershell
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/rerun_unified_evaluation.py --device cuda
```

Overlapping native-image tiles can be evaluated with class-aware NMS using `run_tiled_inference.py`. Registered clean-ablation and 960-pixel pilot runs use `run_followup_experiments.py`; its 1280 stage is conditional on the decision thresholds recorded in that script and its output manifest.

## Keeping the QA gate current

`config/default.yaml` (`annotations.approved_groups`) and `config/qa_gate.yaml` (audit metadata + `resolution_status`) must always agree, and both drift whenever a group gains or loses a Final-QA JSON or a new annotation audit runs. Refresh them from disk instead of hand-editing either file:

```powershell
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/update_qa_gate.py --dry-run
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/update_qa_gate.py --apply
```

The refresh re-discovers every group with exactly one valid, non-backup Final-QA JSON (rejecting a class-vocabulary mismatch or more than one active file for a group as a structural error), reads the newest `MTSD-AnnotationQA/outputs/audit-*/audit_summary.json` by its own `generated_at` timestamp, and writes a locked `approved_groups`/`approved_scope` list plus a recomputed `resolution_status` and `acknowledgement` - never a fabricated audit path/timestamp when no audit is usable. It never touches `group_scope`, `unexpected_group_policy`, or any other configuration field, never creates `.bak` files, and never prepares data, trains, or launches the audit itself. A newly QA'd group is added to the scope automatically but stays blocked from `--final` use until it is covered by a refreshed audit with zero unresolved findings.
