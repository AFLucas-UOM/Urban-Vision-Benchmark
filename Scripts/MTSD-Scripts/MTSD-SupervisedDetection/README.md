# MTSD Supervised Detection

This is the canonical, importable route for MTSD detection preparation, training, unified evaluation, and reporting. The older notebooks are thin interactive front ends; they must not maintain separate split or training logic.

Safe checks from the repository root:

```powershell
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --dry-run
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --validate-prepared --strict
```

Preparation creates `MTSD-Augmented` and `MTSD-Unaugmented` from one split assignment, each with YOLO and COCO layouts plus hashed manifests. Final mode requires the explicit GRP-1/2/3/5/6 lock and a resolved `config/qa_gate.yaml`; GRP-6 is listed explicitly but remains pending refreshed-audit approval. Auto discovery is development-only. Raw XML requires both fallback flags, creates a distinct mixed version, and is forbidden in final mode.

Training requires `--matrix`, `--models`, or `--smoke-test`. Final dissertation training additionally requires `--final` and the augmented variant. Full runs use W&B project `MSc-MTSD-SupervisedDetection`; run directories and resume state are immutable and fingerprint-checked. The clean augmentation ablation is YOLO11m/12m/26m, with YOLO26l optional; RF-DETR is descriptive because its internal online augmentation is not fully controlled.
