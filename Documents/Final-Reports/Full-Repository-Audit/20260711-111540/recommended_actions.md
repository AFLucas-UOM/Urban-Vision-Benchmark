# Recommended actions (2026-07-11 full repository audit)

Ordered by priority. Nothing below was performed automatically unless marked **done**.

## Blocking (before MTSD dataset preparation)

1. **Fix the EXIF-orientation gap in offline augmentation (F02).**
   `Scripts/MTSD-Scripts/MTSD-SupervisedDetection/mtsd_detection/augmentation.py` must apply
   `PIL.ImageOps.exif_transpose()` to the loaded image before the photometric ops (QA labels are in
   EXIF-applied coordinate space, so transposing the pixels aligns them; the current code would emit
   augmented copies rotated relative to their labels for the 570+ EXIF-oriented images).
   Add a regression test with an EXIF `Orientation=6` fixture. Also confirm (once prepared) that the
   RF-DETR/COCO loader applies EXIF the same way Ultralytics does for the *unaugmented* hard-linked
   originals — if it does not, the COCO variant needs EXIF-normalised image copies instead of hard links.

2. **Close the QA gate over the full 5-group scope (F03).** Adjudicate the 8 invalid
   `Damaged-Unknown` `sign_shape` values (GRP-2 ann 1529; GRP-3 ann 1078; GRP-5 anns 66, 1114, 1443;
   GRP-6 anns 165, 913, 1067) and the 5 duplicate-candidate pairs (GRP-2 ×3 high-overlap,
   GRP-5 ×1 high-overlap + 1 exact: anns 1033/1040), rerun
   `scan_annotations.py` (writes a new `audit-<ts>` output), then
   `update_qa_gate.py --apply` and set `resolution_status: resolved` with `approved_by`.
   This audit deliberately did **not** touch annotations or the gate.

## High priority (repository hygiene)

3. **Untrack the Label Studio database (F01).**
   `git rm --cached "Datasets/MTSD/LabelStudioShared/labelstudio_output/LabelStudioData/label_studio.sqlite3"`
   then commit. The ignore rule (`label_studio.sqlite3`, `*.sqlite3`) was **added to `.gitignore` (done)**.
   The file stays on disk. Optional: rewrite history to purge the 36.7 MB blob (repo is private; low urgency).

4. **Refresh the living health report (F06).** Run
   `python Scripts/Automation/verify_repository_health.py` (no `--dry-run`) so the stored report
   includes the new `qa_gate_scope_consistency` check. Left to you because the audit was instructed
   not to overwrite existing reports.

## Medium priority

5. **Quantify or absorb the 9 extension-normalised leaked sources (F04).** Optionally extend
   `mdwd_leakage_sensitivity.py` with an `--identity ext-normalized` mode (39 identities instead of 30)
   and regenerate the clean-subset comparison, or add one sentence to
   `Documents/MDWDLeakageSensitivity.md`'s limitation paragraph quantifying it
   (9 extra photos: IMG_4299/4396/4398/4403/4405/4406/4407/4410/4419; ≤9 extra affected eval images —
   the NEGLIGIBLE verdict is very unlikely to flip). Do not edit the stored generated report.

6. **Extend the health checker (F07).** (a) markdown-link check → `Documents/**/*.md` recursive;
   (b) add a "tracked file > N MB or *.sqlite3/db" check; (c) broaden the secret scan to HF (`hf_…`)
   and OpenAI (`sk-…`) token shapes.

## Low priority / when the time comes

7. **GRP-8..11 layout (F08).** Before annotating them, rename `merged_images/` → `Images/`
   (or teach the tooling both names). Rerun the raw-inventory hash pass over them at that point.
8. **Inference-benchmark pointer**: registry note for pending MTSD detection models still points to
   the notebooks; point it at `run_mtsd_supervised.py` in the next touch of that file.
9. **EDA polygon-line interpretation (F05)**: optionally document in the MDWD EDA README that
   `out_of_range_box` rows in `integrity_issues.csv` are polygon-format lines, not true bad boxes.

## Explicitly NOT to be done automatically (policy)

- Editing Final-QA annotation values or deleting duplicate annotations.
- Resolving/relaxing the MTSD QA gate.
- Regenerating prepared datasets, retraining, or touching completed run records
  (`Results/**`, `AttributeClassification/outputs/**`, W&B).
- Rewriting the stored MDWD leakage or EDA generated records.
