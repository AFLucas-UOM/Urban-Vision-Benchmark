# Methodology — writing notes

Companion to `methodology.tex` (commit `cee7dc6a`, drafted 11 July 2026 under the final-state assumption).

## Note 2026-07-24 (repository review, not a chapter revision)

Point 3 of the "Revision 2026-07-18" entry below states *"MTSD training itself not yet run (`Results/MTSD-Runs` empty)"* — this is now **stale**. The configured 13-model dissertation matrix ran to completion 19–22 July 2026, followed by a clean augmentation ablation and resolution follow-up experiments (640/960/1280 px) and a tiled-inference pilot. Full results: root `Summary.md` §2.2. `methodology.tex` itself makes no claim about MTSD training status (it describes the protocol/procedure only, not results), so no chapter edit was needed for this; a Results/Evaluation chapter, when drafted, should cite `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/` as the headline evidence source. A further MTSD augmentation recipe and additional training were in active, uncommitted development at the time of this note and are out of scope here.

## Placeholder ledger — every `\TBD{...}` in methodology.tex

Fill each from the named final artefact once it exists. Search the .tex for `\TBD{` — the macro renders as **[TBD: id]**.

| Placeholder id | Meaning | Source that will provide it |
|---|---|---|
| `mdwd-collection-period` | MDWD capture date range | EUVIP paper / collection log (not in repo) |
| `mdwd-capture-devices` | MDWD capture devices | EUVIP paper / collection log |
| `mtsd-collection-period` | MTSD capture date range (11 groups) | final MTSD EDA `temporal_summary.csv` |
| `mtsd-imgs-final` (×2; renamed from `mtsd-final-annotated-images` 2026-07-17 for table width) | final annotated-image total | final `QA-GRP*.json` count over 11 groups |
| `mtsd-inst-final` (×2; renamed from `mtsd-final-instances`) | final instance total | same |
| `mtsd-final-exif-coverage` / `mtsd-final-gps-coverage` | EXIF/GPS coverage % | final EDA `gps_tag_audit.csv` / EXIF summary |
| `grp4-img`, `grp4-inst`, `grp8-*`, `grp9-*`, `grp10-*`, `grp11-*` | per-group annotated images/instances | final Final-QA JSONs for GRP-4/8/9/10/11 |
| `cls-*` (12 counts + 12 pct) | final detection-class distribution | regenerated final EDA `class_frequencies.csv` |
| `va-*`, `mt-*`, `cond-*`, `sh-*` (13 values) | final attribute distribution | regenerated final EDA `attribute_summary.csv` |
| `mtsd-prepared-train`, `-train-aug`, `-valid`, `-test` | prepared split sizes | prepared-dataset manifest (`Datasets/MTSD/Prepared/.../manifest`) ; note train-aug = 3 × train under copies_per_image: 2 |
| `attr-final-crops` | final crop-manifest size | refreshed `outputs/manifests/manifest.json` |
| `rfdetr-n`, `rfdetr-s`, `rfdetr-m` (renamed from `rfdetr-*-params`) | RF-DETR parameter counts | runtime introspection of final checkpoints (`sum(p.numel())`) |

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

## Revision 2026-07-18 (commit 0c344db)

Chapter revised against the current repository state (inspected 18 July 2026):

1. **MTSD annotation/QA complete.** All eleven groups have Final-QA; terminal audit `audit-20260718-113454` reports 0 invalid values / 0 duplicate candidates; `qa_gate.yaml` **resolved** over GRP-1..11. The Damaged-Unknown claim ("no value survives") verified directly: 0 of 20,509 annotations.
2. **Counts corrected and filled from primary artefacts.** Raw/annotated totals are now **7,482 images / 20,509 instances** (GRP-10 = 779 images, not 789; GRP-5 = 1,839 instances, not 1,841; ~10 images and 27 annotations removed in the final QA resolution pass). The stale figure 7,492 was removed from this chapter and the Introduction. Group, class-distribution and attribute-distribution tables filled by direct count over `QA-GRP*.json` (18 July 2026). The on-disk EDA CSVs still reflect the 17 July pre-resolution snapshot (7,492/20,536, 31 Damaged-Unknown) — **regenerate the final EDA** so its CSVs reproduce the chapter's numbers.
3. **Prepared datasets built** (`mtsd-qa-v1-aug` / `mtsd-qa-v1-noaug`, prepared 18 July 2026): splits 5,986/749/747 images (16,535/1,996/1,978 boxes), 23,944 augmented train images; validation findings (6 clipped boxes, 2 boxless GRP-7 images, 1,273 EXIF-normalised COCO copies) documented in §Dataset Preparation. MTSD training itself **not yet run** (`Results/MTSD-Runs` empty).
4. **Augmentation recipe updated** to `photometric-v1+motion-blur-v1`: `copies_per_image: 3`; copy 3 = deterministic horizontal motion blur (kernel 9, weight 0.85). Chapter text, table `tab:mtsd-aug` and the ablation wording updated; optional YOLO26-L ablation extension noted.
5. **Prompt-sensitivity protocol documented** (new §Prompt-Sensitivity Protocol, label `subsec:prompt-sensitivity`): prompt-sensitivity-v1, 9 families x 4 variants = 36 prompts, family validation, spread/degradation statistics, prediction-consistency analysis, dual-stage execution. Source: `batch_evaluation/prompt_protocols/prompt_sensitivity_protocol.yaml` + `Documents/PromptDetect-Prompt-Sensitivity-Protocol.md`.
6. **Statistical-analysis wording corrected** (closes verification gap 6): the attribute bootstrap reconstructs per-crop outcomes from stored confusion matrices (verified in `bootstrap_uncertainty.py`); the former "per-crop prediction records" sentence was wrong and is fixed, with the cross-head-correlation caveat added.
7. **MDWD QA:** polygon-format adjudication sentence added (integrity-report FAIL reconciled per Summary.md 2026-07-11 audit note) — closes scope-audit conflict 5.
8. **New §Automation, Verification and Test Infrastructure** under Reproducibility: workflow registry/runner (training gate, dry-run, headless notebooks), repository health verifier (local pre-commit/pre-push; CI deliberately omitted), dataset-pipeline audit script, pytest suites incl. consumer-compatibility tests, managed application launcher. Label Studio import-preparation and QA-Formatter tooling now described in §MTSD QA.
9. **Placeholder ledger status:** filled — mtsd-collection-period (Nov 2025–Jan 2026), EXIF/GPS coverage (72.6% timestamps / 57.9% decodable GPS, computed 18 July over all 7,482 files), all grp-*/cls-*/va-mt-cond-sh values, mtsd-prepared-* splits, attr-final-crops (stated as the 13,862-crop eight-group snapshot pending the scope decision), rfdetr-n (30,157,870 from the E001 training log). **Still open:** `mdwd-collection-period`, `mdwd-capture-devices` (EUVIP paper), `rfdetr-s`, `rfdetr-m` (models not yet instantiated).
10. **Not documented on purpose:** the 2026-07-16/17 interpretability workflow was never committed and no longer exists in the repository; it must not be described in the chapter.

## Revision 2026-07-16/17 (commit 2d70be27)

Chapter revised for the current attribute-classification state:

1. **Four backbone families.** LingBot-Vision (robbyant/lingbot-vision-vit-base / -vit-large, official `lingbot_vision` loader pinned to upstream commit `151e4632`) added alongside DINOv3, V-JEPA 2.1 and ConvNeXt. Verified from the implementation and model card: RoPE position embeddings, 4 register tokens, 512 px pretraining global crops, evaluated at 224 px, mean of `x_norm_patchtokens` as pooled readout, embed dims 768 (ViT-B) / 1024 (ViT-L), frozen + LoRA (fused `qkv`) only, no full fine-tuning. Described strictly as a backbone inside the shared multi-head classifier.
2. **Two experiment rounds separated.** Historical six-variant round (GRP-1–3 snapshot; table retained, marked superseded) vs the 16-variant size/adaptation matrix, which COMPLETED real (non-smoke) training + test evaluation 2026-07-13 on the eight-group snapshot (GRP-1–3, 5–9; 13,862 crops: 11,025/1,397/1,440). All parameter counts in the new table were taken from `outputs/reports/size_ablation.csv` (run-time introspection), not vendor figures.
3. **Gradient accumulation** documented: effective batch 32 everywhere; 16×2 for ViT-L LoRA variants and ConvNeXt-Large fine-tune.
4. **New citation used:** fu2026lingbot (in the shared .bib).
5. **Verification gap 1 closed:** V-JEPA pooling confirmed as `tokens.mean(dim=1)` in `mtsd_attr/backbones/vjepa_backbone.py` (both backends).
6. **Placeholder ledger unchanged** except: `attr-final-crops` remains open — NOTE the crop manifest auto-discovers Final-QA groups, so the next manifest refresh will ingest GRP-10/11 (QA landed 14–16 July) and change the dataset under the completed 16-variant matrix. **Research decision required:** either freeze the attribute scope at the eight-group snapshot (report it as such) or re-run the matrix on the final scope; the chapter currently states the completed snapshot explicitly and defers the final-scope decision.
7. **Repository reality as of 17 July 2026:** 10 of 11 groups have Final-QA (GRP-4 outstanding); fresh annotation audit (audit-20260716-212105) over all 10 groups found 30 invalid attribute values (all `Damaged-Unknown` sign shapes) + 18 duplicate candidates (1 exact, 1 conflicting, 16 high-overlap); QA gate unresolved; detection approved scope still GRP-1/2/3/5/6; `Datasets/MTSD/Prepared/` still not built.
