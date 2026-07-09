# Urban-Vision-Benchmark — Full Repository Status Summary

> **Purpose:** temporary, in-depth context document (live scan updated 2026-07-09) so that
> another model/person can evaluate what remains for the MSc dissertation. Safe to
> delete. Status labels used throughout: **implemented** (built and executed),
> **scaffolded** (built, verified light, but not executed for real),
> **pending** (not built / not run), **requires manual review** (needs a human decision).

---

## 0. Live scan and repository hygiene update (2026-07-09)

**Status: implemented and verified.** The repository hygiene pass was completed
without deleting datasets, result folders, notebooks, checkpoints, crops, or
trained outputs. A fresh live health check is clean:

```powershell
python Scripts\Automation\verify_repository_health.py
```

Result:

```text
Repository health: PASS
All checks PASS, 0 findings.
```

Reports regenerated:

- `Documents/Final-Reports/repository_health_check.md`
- `Documents/Final-Reports/repository_health_check.json`

Live scan results that supersede older MTSD statements elsewhere in this file:

- All 11 MTSD raw-capture groups are now populated (**7,492 images** total).
  QA-approved JSONs currently exist for **GRP-1, GRP-2, GRP-3 and GRP-5**:
  **2,628 images / 7,266 annotations** across 12 detection classes. GRP-4 and
  GRP-6--GRP-11 have images but no Final-QA JSON and remain excluded from
  experiments.
- The fresh annotation audit (`audit-20260709-201340`) found **5 invalid
  `sign_shape=Damaged-Unknown` values** and **5 duplicate candidates** (one
  exact duplicate, four high-overlap), with no missing attributes or broken
  image references. These need visual review before a new MTSD training round.
- The attribute-classification manifest and completed six-model comparison are
  a **GRP-1--GRP-3 snapshot** (manifest last refreshed 2026-07-04). They do
  not include the newly QA-approved GRP-5 data. Preserve those results as a
  historical experiment. Do not rerun just for GRP-5; wait until the final MTSD
  annotation scope is fixed, then refresh the manifest and run one separately
  labelled final attribute round.
- `Datasets/MTSD/LabelStudioShared/.../label_studio.sqlite3` is a new,
  untracked 33 MB local Label Studio database. It is not covered by the
  present ignore rules, so it is an accidental-commit and privacy risk. Do not
  add it to Git; add a narrow ignore rule after confirming the directory is
  purely local application state.

Files changed by the hygiene pass:

- `Scripts/MTSD-Scripts/AttributeClassification/outputs/manifests/manifest.json`
- `Scripts/Automation/verify_repository_health.py`
- `Scripts/Other-Scripts/GDPR-Compliance/README.md`
- `Scripts/Other-Scripts/GDPR-Compliance/redact.py`
- `Requirements/CondaEnvironments/README.md`
- `Scripts/MTSD-Scripts/AttributeClassification/README.md`
- `Scripts/MTSD-Scripts/AttributeClassification/inference/gradio_compare.py`
- `Documents/Other/CloudflareTunnel.md`
- `Documents/Other/CleanModelCache.md`
- `Documents/Final-Reports/repository_health_check.md`
- `Documents/Final-Reports/repository_health_check.json`

What was fixed:

- The attribute-classification manifest had **5,273 stale `source_image` paths**
  of the old form `Datasets/GRP-*/Images/...`; all were updated to the current
  MTSD layout `Datasets/MTSD/GRP-*/Images/...`.
- The GDPR redaction docs and CLI examples now point at the current
  `Datasets/MTSD/GRP-1` layout. The preview output tree is documented as
  relative to the selected input root (`./GRP-1/...`) rather than as a dataset
  source path.
- The Conda environments README link to the model-cache cleanup document was
  fixed from the old `Documents/CleanModelCache.md` location to
  `Documents/Other/CleanModelCache.md`.
- Machine-specific documentation examples using local Windows user-profile
  paths or local Anaconda interpreter paths were replaced with portable
  conda/Python commands, `%USERPROFILE%`, or explicit placeholders.
- The repository health checker now has a narrow documented allowlist for
  intentionally retained historical/provenance references:
  - QA `source_file` fields in `Datasets/MTSD/Annotations/.../Final-QA/QA-GRP*.json`
    remain as provenance of the original export/migration source.
  - old MDWD EDA notebook references remain as historical/legacy context.
  - attribute-classification `experiment_log.json` checkpoint paths remain as
    historical run records.
  - migration docs/scripts may still show the old `Datasets\GRP-*\Images\...`
    shape because they document the exact legacy form the migration script fixes.

No remaining warnings are intentionally outstanding in the generated health
report. Historical references are retained only where they are explicitly
documented provenance or migration examples.

---

## 1. Repository overview

**Urban-Vision-Benchmark** is the central private repository for an MSc dissertation
(University of Malta) on **computer vision for urban waste and infrastructure
monitoring in Malta**. It unifies two research tracks:

1. **MDWD — Maltese Domestic Waste Dataset**: street-level kerbside waste photos,
   object detection with 5 classes (`Mixed Waste`, `Orange CMD`, `Organic Waste`,
   `Other Waste`, `Recyclable Material`).
2. **MTSD — Maltese Traffic Sign Dataset**: 7,492 smartphone-collected traffic-sign
   photos across 11 collection groups. Four groups currently have QA-verified
   COCO annotations (2,628 images / 7,266 boxes; 12 sign classes) whose boxes
   carry **auxiliary attributes** (view angle, mounting, condition, shape).

The comparative angle of the dissertation: classical supervised detectors
(YOLO11/12/26, RF-DETR) vs prompt-based foundation models (SAM 3/3.1, Cosmos
Reason2, LocateAnything) vs representation-learning attribute classifiers
(DINOv3, V-JEPA 2.1, ConvNeXt under frozen/LoRA/fine-tune adaptation).

The repo is a **private** GitHub repository (`AFLucas-UOM/Urban-Vision-Benchmark`);
raw imagery (~250 GB), model weights, W&B folders and bulk crops are kept out of
git; code, configs, QA annotations, numeric results and EDA artefacts are tracked.

## 2. Folder structure

```
Datasets/
  MDWD/                    4 exports of Roboflow project v20 (same data):
                           MDWD-YOLO11/12/26 (YOLO) + MDWD-RFDETR (COCO);
                           train 29,487 (10x augmented) / valid 369 / test 369
  MTSD/GRP-1..GRP-11/      collection groups (raw Images/); 7,492 images total:
                           GRP-1: 724, 2: 630, 3: 617, 4: 875, 5: 657,
                           6: 470, 7: 710, 8: 703, 9: 715, 10: 789, 11: 602
  MTSD/Annotations/        GRP-*/Final-QA/QA-GRP*.json — QA-verified COCO annotations
                           (currently GRP-1/2/3/5; 2,628 images, 7,266 boxes, 12 classes)
  MTSD/Prepared/           NOT YET CREATED — output target of the dataset prep notebook
Documents/                 workflow docs + generated EDA outputs
  MTSD-EDA/                21 figures, ~30 CSVs, interactive HTML capture map
                           (`MTSD_mapped.html` with image popups)
  MDWD-EDA/                10 figures, 11 CSVs, eda_summary.json, annotated samples
Models/                    stock pretrained weights (YOLO11/12/26 n-l, RF-DETR n/s/m) [untracked]
Requirements/              5 pip requirement files per workstream
Results/
  MDWD-Results/            consolidated benchmark tables/plots (YOLO11/12/26, YOLO26-DGX, RF-DETR)
  MDWD-Runs/               archived training runs incl. weights [weights/media untracked]
  MTSD-Results/, MTSD-Runs/  EMPTY — reserved for the MTSD supervised detection track
  PromptDetect/BatchEvaluation/  batch-evaluation outputs (one 5-image MDWD pilot exists;
                                 full dissertation evaluations still pending)
Scripts/
  MDWD-Scripts/
    MDWD-Analysis/         mdwd_eda package + run_eda.py + visualise_samples.py
                           + MDWD-EDA.ipynb (+2 legacy notebooks)
    MDWD-SupervisedNotebooks/  YOLO12/YOLO26/RF-DETR benchmark notebooks (RF-DETR nano executed; small/medium templates)
  MTSD-Scripts/
    AttributeClassification/   config-driven multi-attribute pipeline (mtsd_attr pkg,
                               6 training scripts, run_all.py, outputs/, inference/gradio_compare.py)
    MTSD-Analysis/         mtsd_eda package + MTSD-EDA.ipynb + audit scripts
    MTSD-AnnotationQA/     NEW — audit -> review -> guarded-apply annotation QA workflow
    MTSD-SupervisedNotebooks/  Prepare-MTSD-Detection-Dataset + YOLO12/YOLO26/RF-DETR (not executed)
    LabelStudio/           annotation workflow tooling + QA formatter
    update_annotation_paths.ps1   (applied 2026-07-04)
  Other-Scripts/
    GDPR-Compliance/       face/plate detection + redaction previews
    Inference-Benchmark/   deployment-speed benchmark across supervised,
                           attribute, and prompt-based models
    PromptDetect/          Gradio app (SAM 3/3.1, Cosmos Reason2 2B/8B/32B, LocateAnything 3B)
      batch_evaluation/    NEW — GT-scored batch evaluation (CLI + Gradio)
```

## 3. MDWD status

| Aspect | Status |
| --- | --- |
| Dataset | **implemented** — Roboflow v20 exports on disk in 4 variants (YOLO + COCO), 30,225 images / 211,512 boxes / 3,598 unique source images |
| EDA | **implemented & executed** — `mdwd_eda` package + `run_eda.py` (11 tables, 10 figures, summary JSON in `Documents/MDWD-EDA/`) + `MDWD-EDA.ipynb` (26 cells, mirrors MTSD EDA style; created clean, not yet executed top-to-bottom by the user) |
| YOLO11/12/26 benchmarks | **implemented & executed** — suites in `Results/MDWD-Runs/{YOLO11,YOLO12,YOLO26}-EUVIP` + `YOLO26-DGX` (n/s/m/l), consolidated CSVs in `Results/MDWD-Results/*/Model-Size-Comparison/` |
| RF-DETR benchmark | **partially implemented & executed** — RF-DETR **nano** has a completed checkpoint and training log; small/medium configs are archived templates without completed checkpoints. Its metrics are not yet normalised into the consolidated MDWD result/table pipeline |
| Known issues | EDA integrity findings: 30 source images have augmented copies in >1 split (train/val/test leakage), 15 out-of-range boxes, 5 empty annotations (`Documents/MDWD-EDA/GeneratedCSVs/integrity_issues.csv`). **RESOLVED FOR THE CONSOLIDATED YOLO TABLES (2026-07-09)**: leakage sensitivity test (`Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py`) independently re-confirmed the 30 leaked sources and measured the effect as **negligible** (max Δ 0.48pp mAP50-95 across the four evaluated YOLO checkpoint configurations/splits) — see [`Documents/MDWDLeakageSensitivity.md`](Documents/MDWDLeakageSensitivity.md). RF-DETR is outside this Ultralytics-based sensitivity analysis. Headline YOLO tables need no adjustment; retain a dissertation-limitations caveat |

## 3a. MDWD split-leakage sensitivity analysis (2026-07-09) — RESOLVED

**Status: implemented and executed.** Full write-up:
[`Documents/MDWDLeakageSensitivity.md`](Documents/MDWDLeakageSensitivity.md); machine-readable
outputs in `Documents/Final-Reports/MDWD-Leakage-Analysis/`.

The 30 leaked source identities flagged by the MDWD EDA were independently
re-derived from scratch (not just read from the existing CSV) via
`Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py`, cross-checked against
`integrity_issues.csv` (**MATCH**) and an independent arithmetic identity
check (**consistent**: 2891+369+368 unique per-split source stems vs 3598
deduplicated = excess of exactly 30). Leakage-excluded ("clean") valid/test
subsets were built (14/369 valid images and 19/369 test images excluded;
0 dataset files touched — verified via unchanged `labels.cache` timestamps
and clean `git status` under `Datasets/`), and four headline YOLO checkpoints
(yolo11l, yolo12m, yolo26l EUVIP, yolo26l DGX) were re-evaluated with
the original protocol on original vs. clean subsets (16 Ultralytics val
runs). Every original-subset re-evaluation reproduced the published benchmark
CSV values to 4 decimal places, confirming pipeline fidelity before trusting
the clean-subset numbers.

**Result: negligible effect** — every evaluated YOLO configuration/split delta falls within ±0.48
percentage points of mAP50/mAP50-95, with validation deltas mostly slightly
positive after exclusion (the opposite of what memorisation-driven inflation
would produce). The completed MDWD benchmark tables require no adjustment;
the leakage is documented as a limitations-section caveat only. The 15
out-of-range boxes and 5 empty annotations were audited separately, confirmed
confined to the training split (0 in evaluation splits), and confirmed not to
cause any image to be dropped by Ultralytics' training loader.

## 4. MTSD status

- **Dataset**: all groups GRP-1..11 have raw images (**7,492** total). QA-approved
  COCO JSONs are present for GRP-1, GRP-2, GRP-3 and GRP-5 (**2,628 images / 7,266
  annotations**); GRP-4 and GRP-6..11 remain unannotated. GRP-3's previously-missing
  219 images remain restored (617 annotated images on disk).
- **Annotation schema**: COCO-style QA JSONs; absolute-pixel bboxes; per-box
  `attributes`: `view_angle` {Front, Back, Side}, `mounting` {Pole-Mounted,
  Wall-Mounted}, `condition` {Good, Weathered, Heavily Damaged}, `sign_shape`
  {Circular, Quadrangle, Triangular, Octagonal, Pentagon} (+ provenance keys).
  12 detection classes, identical across groups.
- **Annotation QA (new tool, `Scripts/MTSD-Scripts/MTSD-AnnotationQA/`)**:
  **implemented**; the current audit, `audit-20260709-201340`, covers all four
  QA groups and found **5 `Damaged-Unknown` `sign_shape` values** (configured as
  drop values), plus **5 duplicate candidates** (1 exact, 4 high-overlap), with
  0 missing attributes and 0 reference problems. The visual review (Gradio) and
  guarded apply steps require manual decisions (`review_app.py`, then
  `apply_fixes.py --apply`). Re-scan and refresh the manifest after any edit.
- **Attribute classification**: **implemented & executed for the historical
  GRP-1--GRP-3 snapshot only** — full training round (2026-07-03/04) for 6
  variants. Test macro-F1 (mean over 4 heads): dinov3_lora **0.873**, convnext
  fine-tuned **0.864**, vjepa_lora **0.840**, dinov3 0.783, vjepa 0.770,
  convnext_frozen 0.763 (see `outputs/reports/comparison.md`). `condition` is
  weakest throughout (macro-F1 0.51–0.73). The current manifest excludes
  GRP-5, so retain these outputs as provenance. Do not rerun attribute
  classification only for GRP-5; wait until the final annotation scope is fixed,
  then refresh the manifest and run one distinctly labelled final round. W&B
  project `MSc-MTSD-Attributes`.

  **Model comparison design:** all variants are crop-level, four-head classifiers
  (view angle, mounting, condition, sign shape) sharing one visual backbone and
  four independent linear heads. The six executed variants are:

  | Variant | Backbone / pretraining | Adaptation | Feature dim. | Trainable params |
  |---|---|---|---:|---:|
  | `dinov3` | DINOv3 ViT-B/16, self-supervised, Hugging Face gated checkpoint | Frozen backbone + linear probe | 768 | 9,997 |
  | `dinov3_lora` | Same DINOv3 | LoRA rank 8 on `q_proj`/`v_proj` + heads | 768 | 304,909 |
  | `vjepa` | V-JEPA 2.1 ViT-L, video self-supervised; still crop repeated into a 2-frame pseudo-clip | Frozen backbone + linear probe | 1,024 | 13,325 |
  | `vjepa_lora` | Same V-JEPA | LoRA rank 8 on fused `qkv` + heads | 1,024 | 799,757 |
  | `convnext_frozen` | ConvNeXt-Tiny, ImageNet-1K supervised pretraining | Frozen backbone + linear probe | 768 | 9,997 |
  | `convnext` | Same ConvNeXt-Tiny | Full end-to-end fine-tuning | 768 | 27,828,589 |

  The comparison therefore covers frozen probing, parameter-efficient adaptation
  and full fine-tuning. It is not a pure backbone ranking: the LoRA variants have
  different adapter capacities, and DINOv3/V-JEPA full fine-tuning is not tested.
  The strongest current claim is an adaptation-strategy comparison, not that one
  pretraining family is universally superior.

  **Training/evaluation details:** the executed runs used seed 42, AdamW,
  class-weighted masked cross-entropy, equal head-loss weights, 32-image batches,
  150-epoch maximum, five warm-up epochs, cosine decay, and early stopping
  (patience 10); best checkpoints were selected using validation mean macro-F1
  and evaluated once on the test split. `Documentation.md`, the attribute README,
  and the exported comparison tables now label these metrics as the historical
  GRP-1--GRP-3 snapshot and record the executed schedule.
 - **EDA**: **implemented & executed** — `MTSD-EDA.ipynb` + `mtsd_eda`, 21 figures
   and ~30 CSVs in `Documents/MTSD-EDA/`. The interactive GPS atlas
   (`MTSD_mapped.html`) was regenerated 2026-07-07 after the dataset move to
   `Datasets/MTSD/...`; point popups now show image previews, capture metadata,
   `Open image` / `Copy path` controls, visible image-missing diagnostics, and
   a non-interactive lower heatmap pane so markers remain clickable in `Both`
   mode. The numeric EDA CSV/notebook outputs still contain the pre-GRP-5
   1,971-image / 5,457-box annotation snapshot; rerun the EDA after the final
   annotation scope is frozen before citing current MTSD annotation statistics.
- **Supervised detection**: **scaffolded, not executed** — `Prepare-MTSD-Detection-Dataset.ipynb`
  (QA groups → `Datasets/MTSD/Prepared/{MTSD-YOLO,MTSD-COCO}`, overwrite-protected).
  The earlier 1,752-image dry-run is obsolete because GRP-5 is now QA-approved;
  regenerate the prepared dataset after QA review and record the resulting split
  counts. The YOLO12/YOLO26/RF-DETR notebooks (`RUN_TRAINING=False`, W&B project
  `MTSD-Supervised-Detection`, outputs → `Results/MTSD-Runs|MTSD-Results`). **No MTSD
  detection training has been run**; `Results/MTSD-*` are empty.

## 5. PromptDetect status

- **Models supported** (registry in `backend.py`): SAM 3, SAM 3.1 (Meta native pkg,
  boxes+masks+real confidences), Cosmos Reason2 2B/8B/32B (Qwen3-VL, boxes only, no
  per-box confidence), LocateAnything 3B (separate `mtsd-la` env worker, boxes only).
- **Manual UI** (`app.py`): **implemented** — single-image, prompt-comparison,
  file-batch and folder-summary tabs plus analytics; *no ground-truth scoring*.
  Untouched by this update.
- **Batch evaluation** (`batch_evaluation/`): **implemented, dry-run verified, and
  pilot-executed** — CLI + Gradio; MDWD (YOLO GT) and MTSD (prepared dataset, or
  QA-annotation fallback via `--split all`) loaders verified; heavy-model gate and
  prompt limits (0–15, with zero prompts allowed only for dry-run/config checks)
  verified. Only Cosmos Reason2 32B is gated by `--allow-heavy`; 8B is allowed
  normally. A real MDWD test pilot (`20260709-234655`) ran SAM 3 and Cosmos
  Reason2 2B on **5 images / 11 GT boxes** with six waste prompts. It verifies
  model loading and scoring, but is far too small and prompt-exploratory for
  dissertation comparison; retain it as a smoke/pilot, not headline evidence.
- **Metrics produced**: P/R/F1/accuracy, AP@50, mAP@50:95, mean matched IoU, FP/FN,
  duplicate detections, per-image/per-prompt/per-model tables, prompt-vs-class
  confusion matrix, plots, optional GT-vs-pred overlays.
- **Limitations**: Cosmos/LocateAnything have no per-box confidence → AP collapses
  to one P/R point (compare on F1); box-based scoring only (no mask GT); SAM 3/3.1
  weights are gated on HF and download on first load.

## 6. gradio_compare status

Side-by-side inference of attribute-classification checkpoints on one uploaded
sign crop (`AttributeClassification/inference/gradio_compare.py`). Changes made
today (**implemented, import- and build-verified**): smoke checkpoints are now
**excluded by default** everywhere (module default, UI initial list, checkbox
default=False, explanatory tooltip), and the results panel now shows a **dynamic
comparison title** built from the actually-loaded checkpoints in the pattern
`<model>_<mode> vs <model>_<mode>` (e.g. `DINOv3_linearprobe vs DINOv3_lora`;
modes derived from checkpoint metadata: frozen+linear-probe → `linearprobe`,
lora → `lora`, finetune → `finetuned`; smoke checkpoints suffixed `(smoke)`).
Discovery currently sees 6 real + 6 smoke checkpoints. Remaining check: a quick
human glance at the running UI (not launched here).

## 7. W&B / experiment tracking

- Projects: `MSc-MDWD-EUVIP26` (MDWD detection, executed), `MSc-MTSD-Attributes`
  (attribute classification, executed), `MTSD-Supervised-Detection` (reserved for
  the MTSD detection notebooks, unused so far).
- Entity: `WANDB_ENTITY` env var, default `mark-bugeja-university-of-malta`.
- Credentials: repository-root `.env` (git-ignored) with `WANDB_API_KEY`; loaded by
  notebooks/scripts; never printed or committed. Local W&B run folders live beside
  the notebooks/pipelines and are git-ignored (cloud is the mirror).

## 8. Results and model outputs — preservation rules

- `Results/MDWD-Runs/**` and `Results/MDWD-Results/**` are the completed MDWD
  experimental record — **never overwrite**; the notebooks refuse to reuse an
  existing run directory.
- `Scripts/MTSD-Scripts/AttributeClassification/outputs/**` holds the completed
  attribute round (checkpoints/metrics/reports/crops) — treat as read-only history;
  `reset_outputs.py` archives rather than deletes.
- New tooling writes only to fresh timestamped folders
  (`MTSD-AnnotationQA/outputs/audit-<ts>/`, `Results/PromptDetect/BatchEvaluation/<ds>/<ts>/`).
- Annotation edits happen only via `apply_fixes.py --apply` (typed confirmation,
  `.bak` backups, targeted changes, change log).

## 9. Reproducibility notes

- **Environments** (conda; `Requirements/CondaEnvironments/`): `MDWD`
  (detection notebooks + MDWD EDA), `mtsd-attrcls` (attribute pipeline +
  gradio_compare), `mtsd-base` (PromptDetect app/batch eval, GDPR, annotation QA
  review app), `mtsd-la` (LocateAnything worker). Requirements files mirror these.
- **Paths**: everything resolves the repo root by walking up from the file
  (pathlib); no absolute paths in code.
- **How to run things**:
  - MDWD EDA: `python Scripts/MDWD-Scripts/MDWD-Analysis/run_eda.py` (or the notebook).
  - MTSD dataset prep: run `Prepare-MTSD-Detection-Dataset.ipynb` (config cell;
    overwrite-protected; ~2 GB output).
  - MTSD/MDWD supervised training: notebooks in the respective
    `*-SupervisedNotebooks/` folders; flip `RUN_TRAINING=True` in the final cell.
  - Annotation QA: `scan_annotations.py [--dry-run]` → `review_app.py` →
    `apply_fixes.py --decisions ... --apply`.
  - MTSD atlas: `python Scripts/MTSD-Scripts/MTSD-Analysis/mtsd_mapper.py [--rescan]`;
    open `Documents/MTSD-EDA/MTSD_mapped.html` via Live Server from the repo root
    so relative image links resolve.
  - PromptDetect batch eval: `run_batch_eval.py --dataset ... --prompts ... --models ...
    [--max-images N] [--dry-run]` or `gradio_batch_eval.py`; only Cosmos 32B
    needs `--allow-heavy`.
  - Inference-speed benchmark: `python Scripts/Other-Scripts/Inference-Benchmark/inference_speed_benchmark.py
    --dataset ... --task ... --models ... [--max-images N] [--dry-run]`; only Cosmos
    32B needs `--allow-heavy`.
- Dry-run/sample modes exist for: annotation scan, fix application, batch
  evaluation, EDA (`--max-images`, `--out-tag`), prep notebook (read-only cells).

## 10. Dissertation relevance

- **MDWD** grounds the domestic-waste-monitoring use case: a realistic, augmented
  street-level detection benchmark with completed YOLO11/12/26 model-family/size
  ablations. RF-DETR nano is an additional completed run, but it must be
  normalised into the final table before being used in a quantitative
  cross-family conclusion; small/medium were not run.
- **MTSD** grounds infrastructure monitoring: detection (scaffolded) plus a unique
  **multi-attribute condition-assessment** layer (view/mounting/condition/shape).
  The completed adaptation study (frozen probe vs LoRA vs full fine-tune across
  DINOv3/V-JEPA/ConvNeXt) is valid evidence for its recorded GRP-1--GRP-3 data
  snapshot; a final-scope rerun is needed before generalising it to any expanded
  MTSD corpus.
- **PromptDetect batch evaluation** operationalises the third axis: can zero-shot,
  prompt-based foundation models replace or complement supervised training for
  these municipal tasks? The new evaluator produces the quantitative comparison.
- **Annotation QA tooling** documents dataset quality control — methodology-chapter
  material (annotation protocol, QA audit findings, correction workflow).

## 11. What is complete ✅

- [x] Unified private repo, git history, path conventions, `.gitignore` policy
- [x] MDWD dataset exports (4 variants) + full EDA (package, notebook, artefacts)
- [x] MDWD supervised benchmarks: YOLO11/12/26 (n/s/m/l), runs + consolidated results + W&B; RF-DETR nano has a completed checkpoint/log, while small/medium remain unexecuted templates
- [x] MTSD raw captures in all 11 groups (7,492 images); Final-QA exists for
      GRP-1/2/3/5 (2,628 images / 7,266 boxes); GRP-3 image gap resolved
- [x] MTSD EDA (package, notebook, 21 figures, GPS atlas with clickable image popups)
- [x] MTSD attribute-classification round: 6 variants trained and compared on the
      recorded GRP-1--GRP-3 snapshot (best: dinov3_lora, macro-F1 0.873)
- [x] Annotation path migration (source_image → new layout)
- [x] Annotation QA **audit** executed on the current four QA groups: 5 duplicate
      candidates and 5 configured drop-value findings, with no missing attributes
      or broken references
- [x] gradio_compare cleanup (smokes off by default, dynamic `<model>_<mode> vs ...` title)
- [x] Inference-speed/deployment benchmark tooling (`Scripts/Other-Scripts/Inference-Benchmark/`, dry-run verified;
      see `Documents/Final-Reports/inference_speed_benchmark_report.md`)
- [x] PromptDetect manual UI (5 tabs, 6 models)
- [x] PromptDetect GT-scored MDWD pilot: SAM 3 + Cosmos Reason2 2B, 5 test
      images / 11 GT boxes / 6 prompts; pipeline and model loading verified only
- [x] Repository hygiene pass (2026-07-07): stale MTSD manifest paths fixed,
      documentation path examples made portable, broken Markdown link repaired,
      provenance allowlist added, health check now **PASS** with 0 findings
- [x] Documentation: root README, per-tool READMEs, workflow docs, this Summary
- [x] MDWD split-leakage sensitivity analysis (2026-07-09): 30 leaked sources
      independently re-confirmed, clean-subset re-evaluation shows negligible
      effect (max Δ 0.48pp) — see `Documents/MDWDLeakageSensitivity.md`

## 12. What is still pending ⏳

- [ ] **MTSD annotation cleanup** (requires manual review): review 5 duplicate
      candidates plus 5 `Damaged-Unknown` drop values in
      `audit-20260709-201340` using `review_app.py`; apply only confirmed
      changes, then re-scan until the accepted status is documented.
- [ ] **MTSD detection dataset preparation**: run the prep notebook (after cleanup)
- [ ] **MTSD supervised detection training**: YOLO12/YOLO26/RF-DETR notebooks (not run;
      `Results/MTSD-*` empty) — decide epochs/augmentation for the current
      ~2.6k-image QA-approved scale
- [ ] **PromptDetect batch evaluations**: a 5-image MDWD pilot has run, but the
      dissertation still needs a fixed, adequately sampled MDWD test evaluation
      and MTSD evaluation after QA/preparation. Use a documented prompt set and
      retain the pilot separately; Cosmos 32B remains opt-in.
- [ ] **Inference-speed benchmarks** (tooling ready, not executed): MDWD detection,
      attribute classifiers, PromptDetect models; MTSD detection blocked on training
- [ ] **RF-DETR nano metric export**: extract/validate the completed run's COCO
      metrics into the same traceable result format as the YOLO tables, or exclude
      RF-DETR from quantitative cross-family tables and state why. Do not imply an
      RF-DETR n/s/m ablation: only nano was run.
- [ ] **Further MTSD annotation**: GRP-4 and GRP-6..11 contain 4,864 unannotated
      images. Decide explicitly whether any belong in dissertation scope; annotate
      them only under a documented expansion protocol, then regenerate EDA/manifests.
- [ ] **Final MTSD attribute round, deferred**: do not rerun only to add GRP-5.
      Wait until the intended annotation scope is fixed, then refresh the manifest
      and run one distinctly labelled final attribute experiment.
- [ ] **MDWD-EDA.ipynb / MTSD supervised notebooks**: execute top-to-bottom once, so
      outputs/cell numbers exist for the dissertation record
- [ ] **Dissertation figures/tables**: cross-track comparison table (supervised vs
      prompt-based), per-class detection results, attribute confusion matrices,
      dataset datasheets; decide which EDA figures go in

## 13. Suggested next steps (practical, in priority order)

1. **Protect the local Label Studio state**: do not commit
   `Datasets/MTSD/LabelStudioShared/.../label_studio.sqlite3`. After confirming it
   is only local application state, add a narrow `.gitignore` rule and commit that
   rule separately from research outputs.
2. ~~Finish the MDWD leakage sensitivity test~~ **DONE (2026-07-09)**: measured
   negligible effect (max Δ 0.48pp); see `Documents/MDWDLeakageSensitivity.md`.
   The completed MDWD benchmark stands unchanged with a documented caveat — no
   further action needed here.
3. **Clean the current MTSD QA findings** (review → apply confirmed decisions →
   re-scan → regenerate manifest). Everything downstream (dataset preparation,
   detection training and MTSD batch evaluation) consumes these files; do it before
   any new MTSD runs.
4. **Run the MTSD prep notebook, then the three detection notebooks** (YOLO12 →
   YOLO26 → RF-DETR). Risk: roughly 2.1k training images with the MDWD protocol (100
   epochs, no augmentation) may underfit — consider enabling the prep notebook's
   photometric augmentation or raising epochs, and record whichever choice you make
   as a protocol deviation in the methodology.
5. **Promote PromptDetect from pilot to evidence.** The 5-image MDWD pilot has
   validated SAM 3 and Cosmos Reason2 2B end-to-end. Now fix 3–5 prompts per
   dataset, run an adequately sampled MDWD test evaluation, then run the MTSD
   evaluation after QA/preparation. Budget GPU time for Cosmos 2B/8B; keep 32B
   optional and explicitly opt in with `--allow-heavy`.
6. **Defer the final MTSD attribute-classification rerun** until the final
   annotation scope is fixed. The GRP-1--GRP-3 results are already usable as a
   historical/provenance experiment; do not burn time rerunning just for GRP-5 if
   more groups may be added.
7. **Methodology chapter content now available**: dataset construction + QA workflow
   (LabelStudio → Final-QA → audit tool findings), EDA statistics, training
   protocols (identical-protocol ablation design), evaluation metrics definitions
   (IoU matching, macro-F1 rationale), reproducibility measures (seeds, manifests,
   W&B), and the MDWD leakage sensitivity methodology (independent leakage
   re-derivation, clean-subset construction, checkpoint re-evaluation). Evaluation
   chapter: MDWD YOLO ablation results are final (leakage-checked); RF-DETR nano
   still needs a normalised metric export before a cross-family quantitative claim.
   The existing MTSD
   attribute results are final only for their GRP-1--GRP-3 snapshot, while the
   final-scope attribute rerun, detection and prompt-based results remain pending.
8. **Risks to watch**: annotation edits change QA SHA-256 hashes (manifest must be
   regenerated or training aborts); the stale manifest is already a data-snapshot
   risk; GRP-4 and GRP-6..11 annotation effort is the biggest remaining manual
   cost; SAM 3/3.1 HF gating and the `mtsd-la` env are the usual PromptDetect setup
   friction; keep the repo private (street imagery, GDPR).

## 14. Good next tasks for another coding agent

These are good copy-paste prompts for Claude Code, GPT-5.X Codex, or another
coding agent. Keep this list short: only tasks that unblock dissertation evidence
or prevent a costly rerun belong here.

### High-value prompt 1 — prepare an MTSD annotation cleanup runbook

```text
Do not apply annotation fixes yet. Inspect the latest
Scripts/MTSD-Scripts/MTSD-AnnotationQA/outputs/audit-*/ reports and produce a
step-by-step runbook for manually reviewing and applying MTSD annotation QA
fixes. Include exact commands, expected files, backup behavior, how to re-scan,
and how to regenerate the AttributeClassification manifest afterward. Preserve
all datasets, notebooks, results, and trained outputs.
```

### High-value prompt 2 — dry-run MTSD detection preparation safely

```text
Inspect Scripts/MTSD-Scripts/MTSD-SupervisedNotebooks/Prepare-MTSD-Detection-Dataset.ipynb
and any helper code. Do not execute cells that write the prepared dataset unless
there is an explicit dry-run/read-only mode. Summarize the current expected
input QA files, output folders, split policy, class mapping, overwrite protection,
and what commands/manual steps are needed before real MTSD detection training.
Do not modify notebooks unless a broken path must be fixed.
```

### High-value prompt 3 — make a dissertation handoff checklist

```text
Create or update a concise Documents/Final-Reports/dissertation_handoff_checklist.md
that turns Summary.md into a checklist for the remaining dissertation work:
annotation cleanup, MTSD detection, PromptDetect batch eval, inference-speed
benchmarks, MDWD leakage discussion, final tables/figures, and commit/push.
Use existing files as sources; do not invent results.
```

### High-value prompt 4 — make a PromptDetect evaluation runbook

```text
Inspect the existing MDWD pilot at
Results/PromptDetect/BatchEvaluation/MDWD/20260709-234655 and the batch evaluator.
Write a reproducible evaluation runbook: a fixed prompt set, a defensible image
sample, model selection, exact commands for MDWD and later MTSD, and how to keep
pilot outputs separate from dissertation results. Do not run heavy models or
change evaluation logic.
```

---

## 15. Dissertation priority filter

This section trims the recommendation list down to work that meaningfully changes
the validity, defensibility or clarity of the dissertation. Small engineering
polish is deliberately excluded.

### Must do before final results

1. ~~Resolve the MDWD leakage question.~~ **DONE (2026-07-09)**: independently
   re-derived leakage count matches (30 sources), clean-subset re-evaluation of
   four headline YOLO checkpoints shows a negligible effect (max Δ
   0.48pp mAP50-95). Use the recommended limitations-section wording in
   `Documents/MDWDLeakageSensitivity.md`; no metric adjustment needed.
2. **Finish MTSD QA decisions before new MTSD detection/prompt runs.** Review the
   current duplicate/drop-value findings, apply only confirmed fixes, re-scan, and
   regenerate downstream prepared datasets/manifests.
3. **Run MTSD supervised detection and complete PromptDetect batch evaluation.**
   A 5-image MDWD PromptDetect pilot exists, but the full fixed-protocol MDWD and
   MTSD evaluations remain core evidence gaps.
4. **Freeze the final methodology snapshot after MTSD scope is decided.** The
   attribute-classification docs now describe the six-variant historical
   GRP-1--GRP-3 comparison and current GRP-1/2/3/5 QA scope. After the final
   annotation decision, update the one explicit scope note and run the deferred
   final attribute round.

### Defer on purpose

1. **GRP-5-only attribute retraining.** Do not do this now. Wait until all groups
   intended for the dissertation are QA-ready, then run one final labelled
   attribute round. Keep the GRP-1--GRP-3 six-model results as historical/provenance
   evidence.
2. **Condition-head improvement experiments.** The weak `condition` head should be
   discussed, but extra class-merging or reweighting experiments are only worth it
   if attribute classification becomes a headline contribution rather than a
   supporting study.
3. **Further MTSD annotation beyond GRP-1/2/3/5.** Valuable, but it is a scope
   decision, not a coding task. Only expand if the dissertation timeline supports
   annotation, QA, EDA refresh and re-running affected experiments.

### Nice only if time remains

1. **Bootstrap confidence intervals or multi-seed reruns.** Useful for stronger
   statistical language, but not required if the dissertation frames results as a
   controlled empirical benchmark and avoids overclaiming significance.
2. **Single provenance index across every experiment.** Nice for handoff, but the
   existing run folders, manifests, W&B logs and final tables are enough if they
   are cited carefully.
3. **Automated manifest freshness check.** Helpful later, but not worth prioritising
   over the manual QA, MTSD detection, and PromptDetect evaluation work still
   remaining.
