# MTSD Supervised Detection

This is the canonical, importable route for MTSD detection preparation, training, unified evaluation, and reporting. The older notebooks are thin interactive front ends; they must not maintain separate split or training logic.

Safe checks from the repository root:

```powershell
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --dry-run
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --validate-prepared --strict
```

Preparation assigns the original images to train, validation, and test before creating any augmented copy. Every prepared variant uses that fixed assignment and contains matching YOLO and COCO layouts plus hashed manifests. Final mode requires the explicit GRP-1 through GRP-11 lock and a resolved `config/qa_gate.yaml`. Auto discovery is development-only. Raw XML requires both fallback flags, creates a distinct mixed version, and is forbidden in final mode.

## Offline augmentation recipes

Three explicit offline preparation choices are available:

- `--offline-augmentation none` writes `Prepared/MTSD-Unaugmented`.
- `--offline-augmentation mild` writes `Prepared/MTSD-Augmented-Mild`.
- `--offline-augmentation strong` writes `Prepared/MTSD-Augmented-Strong`.

The legacy `--dataset-variant augmented` command remains compatible and still writes `Prepared/MTSD-Augmented` using the original `photometric-v1+motion-blur-v1` recipe. Its implementation and parameters are unchanged. That mild recipe creates three training copies, with the third copy using horizontal motion blur with kernel size 9 and blur weight 0.85. For a legacy dataset prepared with only `aug1` and `aug2`, add `aug3` to both YOLO and COCO layouts with:

```powershell
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/add_mtsd_motion_blur_aug3.py
```

The command is deterministic and safe to rerun; it updates the augmentation and preparation manifests and validates the resulting dataset.

The `strong-offline-v2` recipe produces three deterministic training copies per original by default. Its probabilistic schedule targets approximately 40 percent four-image mosaics, 20 percent rare-class copy-paste samples, and a balanced mixture of photometric/camera degradation and bbox-aware geometry for the remainder. Non-mosaic strong outputs use a configurable 1920-pixel long-edge cap before augmentation, retaining two-times spatial headroom for 960 training while avoiding repeated 3–4K processing and trainer-side downsampling; this resize and its bbox scale are recorded in each audit row. Geometry transforms every box through the same homography, then clips and filters it using visible-area and minimum-size thresholds. Mosaic uses four training sources only. Copy-paste uses a feathered bbox-and-context patch and records the source annotation and placement. MixUp and horizontal flip are disabled by default. Horizontal flip cannot be enabled without a validated symmetric class-remapping table for direction-sensitive classes.

All strong settings are explicit in `config/default.yaml`, including probabilities, ranges, box filters, rejection limits, mosaic canvas size, rare-class selection, and visual QA sample counts. Augmented data is generated only after the split and only in `train`; validation and test contain original images only. The JSONL audit records source hashes, operations, box outcomes, rejected attempts, and generated hashes for every output.

Build and strictly validate the strong dataset:

```powershell
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --prepare-only --offline-augmentation strong --rebuild --final
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --validate-prepared --offline-augmentation strong --strict
```

Run the small strong-augmentation smoke test:

```powershell
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --augmentation-smoke-test --augmentation-smoke-samples 2 --rebuild
```

Smoke output is written to `Results/MTSD-Results/Dataset-QA/Strong-Augmentation-Smoke`. Full visual QA is under `Prepared/MTSD-Augmented-Strong/visual_qa`, separated into photometric, geometric, mosaic, and copy-paste examples, with `summary.json` containing box, class, rejection, and object-size counts.

Training requires `--matrix`, `--models`, or `--smoke-test`. Final dissertation training additionally requires `--final` and an augmented recipe. YOLO11-M at 960 with the strong dataset is:

```powershell
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --final --offline-augmentation strong --models yolo11m --image-size 960 --device cuda --require-unified-eval --yolo-online-augmentation disabled --run-label mtsd-strong-yolo11m-960
```

`training.yolo_online_augmentation` remains `disabled`; selecting an offline recipe never enables Ultralytics HSV, mosaic, MixUp, flips, or geometric transforms. Full runs use W&B project `MSc-MTSD-SupervisedDetection`; run directories and resume state are immutable and fingerprint-checked. The clean augmentation ablation is YOLO11m/12m/26m, with YOLO26l optional; RF-DETR is descriptive because its internal online augmentation is not fully controlled.

Mosaic can reduce already-small signs when four scenes share one canvas, so tiny-object filtering and rejection counts must be reported with results. Bbox-only copy-paste cannot reproduce true object contours or scene depth; feathered context reduces rectangular edges but does not make it equivalent to mask-based compositing. These limitations mean strong-vs-mild comparisons should be interpreted as a combined offline-policy intervention unless mosaic and copy-paste are ablated separately.

Unified COCO evaluation clamps predictions to the source image, discards boxes that remain degenerate, and converts RF-DETR inputs to RGB. The default evaluation policy drops `Tourist Sign` from both ground truth and predictions while leaving the training taxonomy unchanged for causal comparability. Each evaluation exports predictions, aggregate COCO metrics, per-class AP/AR, IoU-0.50 PR curves, and one micro-averaged confidence-swept F1 threshold. Rerun a completed matrix without retraining with:

```powershell
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/rerun_unified_evaluation.py --device cuda
```

Overlapping native-image tiles can be evaluated with class-aware NMS using `run_tiled_inference.py`. Registered clean-ablation and 960-pixel pilot runs use `run_followup_experiments.py`; its 1280 stage is conditional on the decision thresholds recorded in that script and its output manifest.

The original 640 runs used Ultralytics physical batch 32 with its nominal optimizer batch 64 (two accumulation steps). Resolution pilots preserve that effective optimizer batch with physical batch 16 at 960 and 8 at 1280, avoiding a change in optimization semantics while fitting GPU memory.

## Final strong-augmentation experiment gate

Before controlled training, run the independent dataset-wide audit. It opens and hashes every generated image, recalculates transformed polygons and clipped boxes from the recorded matrices, compares audit/YOLO/COCO annotations, checks source hashes and split membership, and creates 25 before/after examples per category plus 25 highest-risk examples:

```powershell
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_strong_augmentation_qa.py --output Results/MTSD-Results/Dataset-QA/Strong-Augmentation-Final-QA-20260724-R2 --samples-per-category 25 --risk-samples 25
```

The report remains `pending_visual_review` until its contact sheets have been inspected and the focused transformation tests pass. Finalisation records both decisions in `qa_report.json`:

```powershell
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_strong_augmentation_qa.py --finalize-report Results/MTSD-Results/Dataset-QA/Strong-Augmentation-Final-QA-20260724-R2/qa_report.json --visual-decision pass --tests-passed --test-report Results/MTSD-Results/Dataset-QA/strong-full-tests-20260724-R2-final.xml --review-notes "Boxes align in all reviewed contact sheets."
```

The controlled queue refuses to start unless that report is a final pass and still hashes to the current prepared dataset. It uses only repository checkpoints, disables W&B, blocks non-loopback network access in every child process, preserves the 64-image YOLO optimiser batch, runs sequentially, and writes immutable command/state/telemetry records. YOLO12-S 960 receives a training-only two-epoch feasibility run with a strict one-hour process-tree timeout; exactly one full YOLO12-S path is then selected:

```powershell
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_strong_aug_experiment_queue.py --preflight-only
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_strong_aug_experiment_queue.py --launch
```

RF-DETR-N remains at its verified native 384 resolution and prior batch 32. RF-DETR 1.3.0 does not expose a public switch that removes its training-time random horizontal flip and random crop/resize composition, so that run is explicitly marked descriptive rather than a clean causal offline-augmentation ablation.

## Keeping the QA gate current

`config/default.yaml` (`annotations.approved_groups`) and `config/qa_gate.yaml` (audit metadata + `resolution_status`) must always agree, and both drift whenever a group gains or loses a Final-QA JSON or a new annotation audit runs. Refresh them from disk instead of hand-editing either file:

```powershell
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/update_qa_gate.py --dry-run
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/update_qa_gate.py --apply
```

The refresh re-discovers every group with exactly one valid, non-backup Final-QA JSON (rejecting a class-vocabulary mismatch or more than one active file for a group as a structural error), reads the newest `MTSD-AnnotationQA/outputs/audit-*/audit_summary.json` by its own `generated_at` timestamp, and writes a locked `approved_groups`/`approved_scope` list plus a recomputed `resolution_status` and `acknowledgement` - never a fabricated audit path/timestamp when no audit is usable. It never touches `group_scope`, `unexpected_group_policy`, or any other configuration field, never creates `.bak` files, and never prepares data, trains, or launches the audit itself. A newly QA'd group is added to the scope automatically but stays blocked from `--final` use until it is covered by a refreshed audit with zero unresolved findings.
