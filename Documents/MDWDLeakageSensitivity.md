# MDWD Split-Leakage Sensitivity Analysis

**Status: implemented, executed, complete (2026-07-09).** Read-only, non-destructive.
Full report: [`Final-Reports/MDWD-Leakage-Analysis/mdwd_leakage_sensitivity_report.md`](Final-Reports/MDWD-Leakage-Analysis/mdwd_leakage_sensitivity_report.md).

---

## Why this test exists

The MDWD dataset is a Roboflow v20 export where every original photograph is
offline-augmented into ~10 copies before being split into train/valid/test.
The existing MDWD EDA (`Scripts/MDWD-Scripts/MDWD-Analysis/mdwd_eda`) flags
this as a dataset-integrity issue: because splitting happens *after*
augmentation, it is possible for augmented copies of the **same source
photograph** to land in more than one split. If that happens, a model can see
near-duplicates of a validation or test image during training, which would
artificially inflate the reported validation/test detection metrics — a
classic train/test leakage problem.

`Documents/MDWD-EDA/GeneratedCSVs/integrity_issues.csv` already reported this:
**30 source images** have augmented derivatives spread across more than one
split. Before treating the completed **YOLO11/12/26** benchmark tables as final
dissertation evidence, this needed a direct answer to one question: **does this
leakage actually move the reported numbers, or is it a cosmetic dataset-hygiene
footnote?** RF-DETR nano is outside this sensitivity analysis because it uses a
different evaluator and has not been normalised into the consolidated result
tables.

Retraining anything, regenerating the splits, or touching the completed
benchmark runs was explicitly out of scope — those results are the
dissertation's experimental record and must not be altered. So this had to be
answered by an independent, read-only sensitivity test layered on top of the
existing data and checkpoints.

## How it was done

Tool: [`Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py`](../Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py)
(registered as the `MDWD-Leakage-Sensitivity` target in
`Scripts/Automation/workflow_targets.json`).

### 1. Independent re-derivation of the leakage (not just trusting the EDA)

Roboflow names every augmented copy `<source_stem>_<ext>.rf.<hash>.jpg`; the
source identity of a file is the part of its stem before `.rf.`
(`mdwd_eda.mapper.source_stem`, reused rather than reimplemented). The script
re-groups all 30,225 images by this source stem and independently flags any
source whose derivatives appear in more than one split — without reading
`integrity_issues.csv` at all — then cross-checks the two results against
each other, plus a completely independent arithmetic identity check:

```
per-split unique source stems (train 2891 + valid 369 + test 368) = 3628
globally unique source stems (deduplicated)                       = 3598
excess                                                              = 30  ✓ matches leaked-stem count
```

Both cross-checks came back **MATCH** / **consistent** — the 30-source figure
is confirmed independently, not merely copied from the earlier audit.

**Result:** 30 leaked source identities — 16 train↔test, 11 train↔valid, 3
valid↔test — affecting **14 of 369 validation images (3.8%)** and **19 of 369
test images (5.2%)**. Full lists: `leaked_source_identities.csv` and
`affected_evaluation_images.csv` in the output folder below.

### 2. Building "clean" evaluation subsets (no dataset files touched)

A validation or test image is excluded from the clean subset if its source
stem also appears in *any other split* (train, or the other evaluation
split); a valid↔test leak is excluded from **both** clean subsets. This
yields:

| Split | Original | Clean | Removed |
|---|---|---|---|
| valid | 369 images / 1,072 boxes | 355 images / 1,030 boxes | 14 images (3.8%) |
| test | 369 images / 1,106 boxes | 350 images / 1,066 boxes | 19 images (5.2%) |

Crucially, **no image was moved, copied, or deleted anywhere on disk.** The
script writes a scratch `data.yaml` plus `.txt` files listing the *absolute
paths* of the images to keep, inside a new `MDWD-Leakage-Analysis/scratch/`
folder — Ultralytics reads images directly from the untouched
`Datasets/MDWD/MDWD-YOLO26/{valid,test}/images/` folders and resolves labels
by its normal `images/`→`labels/` path substitution. A monkey-patch also
disables Ultralytics' `labels.cache` writer for the run, so not even a cache
file was created or overwritten inside `Datasets/`. Verified by comparing
every `labels.cache` timestamp under `Datasets/MDWD/` before and after the
run — unchanged — and by `git status` showing zero modifications under
`Datasets/`, `Results/MDWD-Runs/`, or `Results/MDWD-Results/`.

### 3. Re-evaluating trained checkpoints, original vs. clean

No stored per-image predictions existed anywhere (the original training
notebooks called `model.val()` without `save_json`/`save_txt`), so getting
clean-subset metrics required running inference — no shortcut was available.
Four headline YOLO checkpoints were re-evaluated using the **exact original
protocol** (`imgsz=640, conf=0.001, iou=0.7,
batch=32`), on four subsets: original valid, clean valid, original test,
clean test:

- `yolo11l` (YOLO11-EUVIP)
- `yolo12m` (YOLO12-EUVIP)
- `yolo26l` (YOLO26-EUVIP) — the dissertation's headline model
- `yolo26l` (YOLO26-DGX) — the DGX re-run of the same configuration

That is 4 checkpoints × 2 splits × 2 subsets = **16 Ultralytics validation
runs**, all on the local RTX 4090, all writing only into the new scratch
folder. As a built-in sanity check, every *original*-subset re-evaluation was
compared against the already-published benchmark CSVs in
`Results/MDWD-Results/*/Model-Size-Comparison/` — **all four reproduced the
published mAP50/mAP50-95 to 4 decimal places**, confirming the re-evaluation
pipeline exactly matches the original protocol before trusting its clean-subset
numbers.

A second, independent metric source was layered on top by reusing
[`Scripts/Other-Scripts/PromptDetect/batch_evaluation/metrics.py`](../Scripts/Other-Scripts/PromptDetect/batch_evaluation/metrics.py)
(greedy IoU matching, per-class) rather than writing new matching logic — this
gives F1 / mean matched IoU / FP / FN at a fixed 0.25 confidence, values
Ultralytics' own `val()` output does not expose directly.

### 4. Validation

13 deterministic self-tests (`--self-test`, no filesystem access) exercise
the normalisation and leakage-classification logic against synthetic cases:
same-source augmentations within one split (correctly *not* flagged),
train+valid / train+test / valid+test / all-three-split leaks (each correctly
categorised and excluded from the right clean subset), visually similar but
distinct stems (`IMG_100` vs `IMG_1000`), and the documented edge case where
Roboflow folds the original file extension into the stem (`10_jpeg` vs
`10_jpg` are treated as distinct source photographs — a known, stated
limitation, not a bug). All 13 passed.

## What the results show

Full per-model tables are in the report; headline numbers:

| Model | Split | mAP50 Δ (clean − original) | mAP50-95 Δ | Verdict |
|---|---|---|---|---|
| yolo11l | valid | +0.12 pp | +0.48 pp | negligible |
| yolo11l | test | −0.23 pp | −0.15 pp | negligible |
| yolo12m | valid | +0.08 pp | +0.14 pp | negligible |
| yolo12m | test | +0.05 pp | +0.11 pp | negligible |
| yolo26l (EUVIP) | valid | +0.20 pp | +0.25 pp | negligible |
| yolo26l (EUVIP) | test | −0.24 pp | −0.11 pp | negligible |
| yolo26l (DGX) | valid | +0.19 pp | +0.23 pp | negligible |
| yolo26l (DGX) | test | −0.10 pp | −0.04 pp | negligible |

Thresholds (documented in the report): negligible ≤ 0.5pp < small ≤ 1.0pp <
moderate ≤ 2.0pp < substantial, chosen because the clean subsets remove only
~4–5% of evaluation images, so composition noise alone can move mAP by a few
tenths of a point.

**Every model/split combination falls in the negligible band**, and the
pattern is not one-sided: validation deltas are mostly slightly *positive*
after removing leaked images, while test deltas hover around zero in both
directions. If the leaked images had been meaningfully inflating scores by
letting the model "cheat" on near-duplicates it had memorised, removing them
would be expected to *consistently lower* the metrics — instead the direction
is mixed and the magnitude is within normal run-to-run noise.

**Conclusion: the leakage is real (independently confirmed) but does not
materially change the MDWD benchmark results.** The completed MDWD detection
benchmark stands as reported; the leakage is documented as a limitation, not
retracted or adjusted.

### Separately audited: out-of-range boxes and empty annotations

The same integrity CSV also flagged 15 out-of-range bounding boxes and 5
empty-annotation images. These are unrelated to split leakage (they are a
label-quality issue, not a cross-split contamination issue) and were audited
separately: **all 20 are confined to the training split**, none touch
validation or test, and re-checking them against Ultralytics' actual training
loader tolerance (1% coordinate overshoot allowed before an image is dropped)
shows **0 of the 15** would have been silently dropped from training — they
train with a marginally out-of-bounds box as-is, and the 5 empty-label images
simply act as background/negative examples. Neither issue could have
influenced any reported validation/test metric.

## Outputs

All outputs are new, non-destructive, and read-only with respect to the
dataset and completed benchmark runs:

```
Documents/Final-Reports/MDWD-Leakage-Analysis/
├── mdwd_leakage_sensitivity_report.md   # full dissertation-ready report (start here)
├── leaked_source_identities.csv         # 30 leaked stems, split pairs, per-split counts
├── affected_evaluation_images.csv       # exact filenames excluded from each clean subset
├── subset_counts.csv                    # original vs clean image/box counts
├── metric_comparison.csv                # per-model per-split per-subset metrics + deltas
├── metric_comparison.json               # same, machine-readable, plus full audit payload
└── scratch/                             # generated data.yaml, image lists, cached predictions
```

## Reproducing this analysis

```powershell
# Self-test the leakage logic (fast, no filesystem access)
conda run -n MDWD python Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py --self-test

# Audit only (independent leakage re-derivation + clean-subset construction; ~1-2 min)
conda run -n MDWD python Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py

# Full sensitivity run (re-evaluates the best checkpoint per model family; ~5-10 min on GPU)
conda run -n MDWD python Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py --run-inference --models best-per-family

# Or via the root launcher / workflow runner
python launch_uvb.py
.\Scripts\Automation\run_urban_workflows.ps1 -Target MDWD-Leakage-Sensitivity -Args "--run-inference --models best-per-family"
```

## Recommended dissertation wording

> An integrity audit of the augmented Roboflow export identified 30 source
> images (of 3,598) whose augmented derivatives appear in more than one split
> (16 train–test, 11 train–validation, 3 validation–test), affecting 14 of
> 369 validation and 19 of 369 test images. To quantify the impact, four
> headline YOLO checkpoints were re-evaluated on leakage-excluded
> validation (355 images) and test (350 images) subsets, constructed so that
> no evaluated image shares a source photograph with any other split. Across
> all evaluated YOLO configurations and both splits, mAP@50 and mAP@50–95 changed by at most 0.48
> percentage points, below typical run-to-run training variance, and
> validation scores marginally increased after exclusion. The cross-split
> leakage therefore did not materially inflate the reported YOLO results, and
> those comparisons are reported on the original splits, with this
> sensitivity analysis retained as supplementary evidence. A further 15
> marginally out-of-range bounding boxes and 5 empty annotations were
> confined to the training split and could not affect evaluation metrics.
