# Methodology — writing notes

Companion to `methodology.tex` (commit `cee7dc6a`, drafted 11 July 2026 under the final-state assumption).

## Placeholder ledger — every `\TBD{...}` in methodology.tex

Fill each from the named final artefact once it exists. Search the .tex for `\TBD{` — the macro renders as **[TBD: id]**.

| Placeholder id | Meaning | Source that will provide it |
|---|---|---|
| `mdwd-collection-period` | MDWD capture date range | EUVIP paper / collection log (not in repo) |
| `mdwd-capture-devices` | MDWD capture devices | EUVIP paper / collection log |
| `mtsd-collection-period` | MTSD capture date range (11 groups) | final MTSD EDA `temporal_summary.csv` |
| `mtsd-final-annotated-images` (×2) | final annotated-image total | final `QA-GRP*.json` count over 11 groups |
| `mtsd-final-instances` (×2) | final instance total | same |
| `mtsd-final-exif-coverage` / `mtsd-final-gps-coverage` | EXIF/GPS coverage % | final EDA `gps_tag_audit.csv` / EXIF summary |
| `grp4-img`, `grp4-inst`, `grp8-*`, `grp9-*`, `grp10-*`, `grp11-*` | per-group annotated images/instances | final Final-QA JSONs for GRP-4/8/9/10/11 |
| `cls-*` (12 counts + 12 pct) | final detection-class distribution | regenerated final EDA `class_frequencies.csv` |
| `va-*`, `mt-*`, `cond-*`, `sh-*` (13 values) | final attribute distribution | regenerated final EDA `attribute_summary.csv` |
| `mtsd-prepared-train`, `-train-aug`, `-valid`, `-test` | prepared split sizes | prepared-dataset manifest (`Datasets/MTSD/Prepared/.../manifest`) ; note train-aug = 3 × train under copies_per_image: 2 |
| `attr-final-crops` | final crop-manifest size | refreshed `outputs/manifests/manifest.json` |
| `rfdetr-n-params`, `rfdetr-s-params`, `rfdetr-m-params` | RF-DETR parameter counts | runtime introspection of final checkpoints (`sum(p.numel())`) |

Known already-annotated groups were entered as fixed values (GRP-1: 724/1,960; GRP-2: 630/1,621; GRP-3: 617/1,844; GRP-5: 657/1,841; GRP-6: 470/1,106; GRP-7: 710/1,891) — re-verify these do not change during the final QA-resolution pass (duplicate deletions would lower instance counts slightly).

## Final-state assumptions vs current repository reality

The chapter is written past-tense per the drafting instruction. At commit `cee7dc6a` the following are **not yet true** and must become true before this chapter is honest:

1. GRP-4 and GRP-8–11 have no Final-QA JSONs (6 of 11 groups annotated).
2. `qa_gate.yaml` is `unresolved`; recorded audit predates GRP-6/7; 15 drop-value + 5 duplicate findings pending review.
3. `Datasets/MTSD/Prepared/` not built; MTSD detection training not run; `Results/MTSD-*` empty.
4. Final attribute round not run (completed round = GRP-1–3 snapshot; parameter counts and architecture carry over, dataset counts do not).
5. PromptDetect full protocol not executed (5-image pilot only).
6. Inference benchmark, final robustness slices, final bootstrap, final failure-case sampling not executed over final runs.
7. Final MTSD EDA regeneration pending (current CSVs are a pre-GRP-5 snapshot — deliberately not cited anywhere in the chapter).

## Hardware-restriction handling

- Chapter mentions only the RTX 4090 workstation; the restricted platform name appears nowhere in the chapter, tables or these notes' companion audit (referred to there as "other hardware" in conflict 3).
- Consequence: the second YOLO26 MDWD suite in `Results/MDWD-Runs/` cannot be cited as final evidence in Chapter 5 unless re-run on the workstation, or the environment section is amended. **Supervisor decision required** (this affects Evaluation, not Methodology wording).
- MDWD env torch is a nightly build (`2.10.0.dev20251013+cu130`); chapter reports "PyTorch 2.10.0 (CUDA 13.0)". If the final runs re-execute under a stable release, update Table `tab:environment`.

## Verification gaps to close before submission

1. **V-JEPA pooling**: confirm mean-pool over tokens in `backbones/vjepa_backbone.py` (chapter states mean-pooled token features).
2. **YOLO checkpoint rule**: chapter says "best validation fitness" (Ultralytics default fitness = 0.1·mAP50 + 0.9·mAP50-95); confirm no fitness override in the final runs.
3. **RF-DETR internal augmentation**: chapter asserts it cannot be disabled equivalently (basis: rfdetr 1.3.0 training pipeline; multi_scale/expanded_scales off in configs). Document the exact residual transform list from the rfdetr source for the appendix if challenged.
4. **MDWD reviewer process**: "reviewer consolidation" is described generically; add reviewer count/protocol from the EUVIP paper when available. The "~20% visible-object rule" was **not** included (unverifiable in repo).
5. **Batch sizes 4/8**: optional in the benchmark CLI; if the final benchmark ran batch 1 only, delete the optional-batch sentence.
6. **Attribute bootstrap**: if the final round does not emit per-sample predictions, revise §Statistical Analysis to the confusion-matrix reconstruction wording (see audit conflict 6).
7. **`multirow` package**: Table `tab:ablation` uses `\multirow` — ensure the dissertation preamble loads `multirow` (test wrapper did).
8. **TikZ**: Figure `fig:multihead` needs `\usepackage{tikz}` with `positioning` library (`\usetikzlibrary{positioning}`).

## Structural decisions

- Mandated 14-section skeleton kept exactly; Annotation-Effort and Operational Analysis placed as a subsection of §Evaluation Framework (no extra principal section allowed).
- EDA is described inside §Dataset Development (subsection per dataset) per the topic list.
- The MDWD leakage audit is described as method only (no findings), in §Annotation and QA.
- No results, rankings or metric values appear anywhere; dataset descriptive statistics only.
- Citations reuse `background_literature_references.bib` keys only (no new entries needed): ultralytics2024yolo11, tian2025yolov12, jocher2026yolo26, robinson2025rfdetr, meta2025sam3, meta2026sam31, nvidia2025cosmosreason2, nvidia2026locateanything, simeoni2025dinov3, murlabadia2026vjepa21, liu2022convnet, hu2022lora, lin2014microsoft, padilla2021comparative, sokolova2009systematic, guo2017calibration, gdpr2016.

## Supervisor decisions that may still be required

1. Which YOLO26 MDWD suite is headline evidence given the single-workstation environment statement (re-run vs amend).
2. Whether GRP-4/8–11 annotation will genuinely complete (otherwise this chapter must be rewritten to the actual final scope — the placeholders make the rewrite mechanical).
3. Whether YOLO26l stays in the ablation (config lists it as optional; chapter includes it only in the matrix, not the ablation trio).
4. Whether the MDWD collection-details placeholders are filled from the EUVIP paper or dropped.
