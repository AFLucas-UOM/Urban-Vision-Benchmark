# MTSD Supervised Detection

This is the canonical, importable route for MTSD detection preparation, training, unified evaluation, and reporting. The older notebooks are thin interactive front ends; they must not maintain separate split or training logic.

Safe checks from the repository root:

```powershell
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --dry-run
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --validate-prepared --strict
```

Preparation creates `MTSD-Augmented` and `MTSD-Unaugmented` from one split assignment, each with YOLO and COCO layouts plus hashed manifests. Final mode requires the explicit GRP-1/2/3/5/6 lock and a resolved `config/qa_gate.yaml`; GRP-6 is listed explicitly but remains pending refreshed-audit approval. Auto discovery is development-only. Raw XML requires both fallback flags, creates a distinct mixed version, and is forbidden in final mode.

Training requires `--matrix`, `--models`, or `--smoke-test`. Final dissertation training additionally requires `--final` and the augmented variant. Full runs use W&B project `MSc-MTSD-SupervisedDetection`; run directories and resume state are immutable and fingerprint-checked. The clean augmentation ablation is YOLO11m/12m/26m, with YOLO26l optional; RF-DETR is descriptive because its internal online augmentation is not fully controlled.

## Keeping the QA gate current

`config/default.yaml` (`annotations.approved_groups`) and `config/qa_gate.yaml` (audit metadata + `resolution_status`) must always agree, and both drift whenever a group gains or loses a Final-QA JSON or a new annotation audit runs. Refresh them from disk instead of hand-editing either file:

```powershell
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/update_qa_gate.py --dry-run
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/update_qa_gate.py --apply
```

The refresh re-discovers every group with exactly one valid, non-backup Final-QA JSON (rejecting a class-vocabulary mismatch or more than one active file for a group as a structural error), reads the newest `MTSD-AnnotationQA/outputs/audit-*/audit_summary.json` by its own `generated_at` timestamp, and writes a locked `approved_groups`/`approved_scope` list plus a recomputed `resolution_status` and `acknowledgement` - never a fabricated audit path/timestamp when no audit is usable. It never touches `group_scope`, `unexpected_group_policy`, or any other configuration field, never creates `.bak` files, and never prepares data, trains, or launches the audit itself. A newly QA'd group is added to the scope automatically but stays blocked from `--final` use until it is covered by a refreshed audit with zero unresolved findings.
