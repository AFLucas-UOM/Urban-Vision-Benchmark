# Urban-Vision-Benchmark — Full Repository Status Summary

> **Purpose:** temporary, in-depth context document (generated 2026-07-06) so that
> another model/person can evaluate what remains for the MSc dissertation. Safe to
> delete. Status labels used throughout: **implemented** (built and executed),
> **scaffolded** (built, verified light, but not executed for real),
> **pending** (not built / not run), **requires manual review** (needs a human decision).

---

## 1. Repository overview

**Urban-Vision-Benchmark** is the central private repository for an MSc dissertation
(University of Malta) on **computer vision for urban waste and infrastructure
monitoring in Malta**. It unifies two research tracks:

1. **MDWD — Maltese Domestic Waste Dataset**: street-level kerbside waste photos,
   object detection with 5 classes (`Mixed Waste`, `Orange CMD`, `Organic Waste`,
   `Other Waste`, `Recyclable Material`).
2. **MTSD — Maltese Traffic Sign Dataset**: smartphone-collected traffic-sign photos
   in collection groups, with QA-verified COCO annotations (12 sign classes) whose
   boxes carry **auxiliary attributes** (view angle, mounting, condition, shape).

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
  MTSD/GRP-1..GRP-11/      collection groups (raw Images/); GRP-1: 725, GRP-2: 631,
                           GRP-3: 617, GRP-4: 875, GRP-5: 658 images; GRP-6..11 empty so far
  MTSD/Annotations/        GRP-*/Final-QA/QA-GRP*.json — QA-verified COCO annotations
                           (currently GRP-1..3; 1,971 images, 5,457 boxes, 12 classes)
  MTSD/Prepared/           NOT YET CREATED — output target of the dataset prep notebook
Documents/                 workflow docs + generated EDA outputs
  MTSD-EDA/                21 figures, ~30 CSVs, interactive HTML capture map
  MDWD-EDA/                10 figures, 11 CSVs, eda_summary.json, annotated samples
Models/                    stock pretrained weights (YOLO11/12/26 n-l, RF-DETR n/s/m) [untracked]
Requirements/              5 pip requirement files per workstream
Results/
  MDWD-Results/            consolidated benchmark tables/plots (YOLO11/12/26, YOLO26-DGX, RF-DETR)
  MDWD-Runs/               archived training runs incl. weights [weights/media untracked]
  MTSD-Results/, MTSD-Runs/  EMPTY — reserved for the MTSD supervised detection track
  PromptDetect/BatchEvaluation/  output target of the new batch evaluator (created on first run)
Scripts/
  MDWD-Scripts/
    MDWD-Analysis/         mdwd_eda package + run_eda.py + visualise_samples.py
                           + MDWD-EDA.ipynb (+2 legacy notebooks)
    MDWD-SupervisedNotebooks/  YOLO12/YOLO26/RF-DETR benchmark notebooks (executed)
  MTSD-Scripts/
    AttributeClassification/   config-driven multi-attribute pipeline (mtsd_attr pkg,
                               6 training scripts, run_all.py, outputs/, inference/gradio_compare.py)
    MTSD-Analysis/         mtsd_eda package + MTSD-EDA.ipynb + audit scripts
    MTSD-AnnotationQA/     NEW — audit -> review -> guarded-apply annotation QA workflow
    MTSD-SupervisedNotebooks/  Prepare-MTSD-Detection-Dataset + YOLO12/YOLO26/RF-DETR (not executed)
    LabelStudio/           annotation workflow tooling + QA formatter
    update_annotation_paths.ps1   (applied 2026-07-04)
  Other-Scripts/
    CondaEnvironments/     4 env YAMLs + per-platform setup scripts
    GDPR-Compliance/       face/plate detection + redaction previews
    PromptDetect/          Gradio app (SAM 3/3.1, Cosmos Reason2 2B/8B/32B, LocateAnything 3B)
      batch_evaluation/    NEW — GT-scored batch evaluation (CLI + Gradio)
```

## 3. MDWD status

| Aspect | Status |
| --- | --- |
| Dataset | **implemented** — Roboflow v20 exports on disk in 4 variants (YOLO + COCO), 30,225 images / 211,512 boxes / 3,598 unique source images |
| EDA | **implemented & executed** — `mdwd_eda` package + `run_eda.py` (11 tables, 10 figures, summary JSON in `Documents/MDWD-EDA/`) + `MDWD-EDA.ipynb` (26 cells, mirrors MTSD EDA style; created clean, not yet executed top-to-bottom by the user) |
| YOLO11/12/26 benchmarks | **implemented & executed** — suites in `Results/MDWD-Runs/{YOLO11,YOLO12,YOLO26}-EUVIP` + `YOLO26-DGX` (n/s/m/l), consolidated CSVs in `Results/MDWD-Results/*/Model-Size-Comparison/` |
| RF-DETR benchmark | **implemented & executed** — `RF-DETR-EUVIP` run archive + `Results/MDWD-Results/RF-DETR/Model-Size-Comparison/` |
| Known issues | EDA integrity findings, **requires manual review**: 30 source images have augmented copies in >1 split (train/val/test leakage), 15 out-of-range boxes, 5 empty annotations (`Documents/MDWD-EDA/GeneratedCSVs/integrity_issues.csv`). Leakage should at least be quantified/discussed in the dissertation; ideally the affected val/test images are replaced or excluded and key models re-evaluated |

## 4. MTSD status

- **Dataset**: groups GRP-1..5 populated (GRP-4/5 = 1,533 images not yet annotated);
  GRP-6..11 exist as empty scaffolding. **GRP-3's previously-missing 219 images were
  restored — all 617 annotated images are now on disk** (verified 2026-07-06).
- **Annotation schema**: COCO-style QA JSONs; absolute-pixel bboxes; per-box
  `attributes`: `view_angle` {Front, Back, Side}, `mounting` {Pole-Mounted,
  Wall-Mounted}, `condition` {Good, Weathered, Heavily Damaged}, `sign_shape`
  {Circular, Quadrangle, Triangular, Octagonal, Pentagon} (+ provenance keys).
  12 detection classes, identical across groups.
- **Annotation QA (new tool, `Scripts/MTSD-Scripts/MTSD-AnnotationQA/`)**:
  **implemented**; audit executed 2026-07-06 (dry-run) and found, across GRP-1..3:
  **42 duplicate pairs (29 exact, 10 conflicting-attribute, 3 high-overlap), 7
  missing attribute values, 4 known-drop values** (`Damaged-Unknown`), 0 reference
  problems. The visual review (Gradio) and guarded apply steps are **scaffolded and
  verified light** but the actual cleanup **requires manual review** (run
  `review_app.py`, then `apply_fixes.py --apply`). After applying, regenerate the
  AttrCls manifest.
- **Attribute classification**: **implemented & executed** — full training round
  (2026-07-03/04) for 6 variants. Test macro-F1 (mean over 4 heads): dinov3_lora
  **0.873**, convnext fine-tuned **0.864**, vjepa_lora **0.840**, dinov3 0.783,
  vjepa 0.770, convnext_frozen 0.763 (see `outputs/reports/comparison.md`).
  `condition` is the weakest head everywhere (macro-F1 0.51–0.73). W&B project
  `MSc-MTSD-Attributes`.
- **EDA**: **implemented & executed** — `MTSD-EDA.ipynb` + `mtsd_eda`, 21 figures
  and ~30 CSVs in `Documents/MTSD-EDA/`, interactive GPS map.
- **Supervised detection**: **scaffolded, not executed** — `Prepare-MTSD-Detection-Dataset.ipynb`
  (QA groups → `Datasets/MTSD/Prepared/{MTSD-YOLO,MTSD-COCO}`, overwrite-protected,
  read-only logic dry-tested: 1,752 usable images, split 1401/175/176, 4,746 boxes —
  numbers will shift slightly now GRP-3 is complete and after QA cleanup) and the
  YOLO12/YOLO26/RF-DETR notebooks (`RUN_TRAINING=False`, W&B project
  `MTSD-Supervised-Detection`, outputs → `Results/MTSD-Runs|MTSD-Results`). **No MTSD
  detection training has been run**; `Results/MTSD-*` are empty.

## 5. PromptDetect status

- **Models supported** (registry in `backend.py`): SAM 3, SAM 3.1 (Meta native pkg,
  boxes+masks+real confidences), Cosmos Reason2 2B/8B/32B (Qwen3-VL, boxes only, no
  per-box confidence), LocateAnything 3B (separate `mtsd-la` env worker, boxes only).
- **Manual UI** (`app.py`): **implemented** — single-image, prompt-comparison,
  file-batch and folder-summary tabs plus analytics; *no ground-truth scoring*.
  Untouched by this update.
- **Batch evaluation** (`batch_evaluation/`, NEW): **scaffolded & dry-run verified**
  — CLI + Gradio; MDWD (YOLO GT) and MTSD (prepared dataset, or QA-annotation
  fallback via `--split all`) loaders verified; heavy-model gate and prompt limits
  (1–15) verified. **No real model evaluation has been executed yet.**
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

- **Environments** (conda; `Scripts/Other-Scripts/CondaEnvironments/`): `MDWD`
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
  - PromptDetect batch eval: `run_batch_eval.py --dataset ... --prompts ... --models ...
    [--max-images N] [--dry-run]` or `gradio_batch_eval.py`; heavy Cosmos variants
    need `--allow-heavy`.
- Dry-run/sample modes exist for: annotation scan, fix application, batch
  evaluation, EDA (`--max-images`, `--out-tag`), prep notebook (read-only cells).

## 10. Dissertation relevance

- **MDWD** grounds the domestic-waste-monitoring use case: a realistic, augmented
  street-level detection benchmark with a completed model-family/size ablation
  (YOLO11 vs 12 vs 26 vs RF-DETR, n→l scales, identical protocol).
- **MTSD** grounds infrastructure monitoring: detection (scaffolded) plus a unique
  **multi-attribute condition-assessment** layer (view/mounting/condition/shape)
  with a completed adaptation study (frozen probe vs LoRA vs full fine-tune across
  DINOv3/V-JEPA/ConvNeXt) — directly relevant to sign-maintenance auditing.
- **PromptDetect batch evaluation** operationalises the third axis: can zero-shot,
  prompt-based foundation models replace or complement supervised training for
  these municipal tasks? The new evaluator produces the quantitative comparison.
- **Annotation QA tooling** documents dataset quality control — methodology-chapter
  material (annotation protocol, QA audit findings, correction workflow).

## 11. What is complete ✅

- [x] Unified private repo, git history, path conventions, `.gitignore` policy
- [x] MDWD dataset exports (4 variants) + full EDA (package, notebook, artefacts)
- [x] MDWD supervised benchmarks: YOLO11/12/26 (n/s/m/l) + RF-DETR (n/s/m), runs + consolidated results + W&B
- [x] MTSD groups 1–3 QA-annotated (1,971 images / 5,457 boxes); GRP-3 image gap resolved
- [x] MTSD EDA (package, notebook, 21 figures, GPS map)
- [x] MTSD attribute-classification round: 6 variants trained, compared (best: dinov3_lora, macro-F1 0.873)
- [x] Annotation path migration (source_image → new layout)
- [x] Annotation QA **audit** executed: findings quantified (42 dup pairs, 11 attribute issues)
- [x] gradio_compare cleanup (smokes off by default, dynamic `<model>_<mode> vs ...` title)
- [x] Inference-speed/deployment benchmark tooling (`Scripts/FinalBenchmarks/`, dry-run verified;
      see `Documents/Final-Reports/inference_speed_benchmark_report.md`)
- [x] PromptDetect manual UI (5 tabs, 6 models)
- [x] Documentation: root README, per-tool READMEs, workflow docs, this Summary

## 12. What is still pending ⏳

- [ ] **MTSD annotation cleanup** (requires manual review): review the 42 duplicate
      pairs + 11 attribute findings in `review_app.py`, apply, re-scan until clean,
      then regenerate the AttrCls manifest (`python -m mtsd_attr.data_manifest`)
- [ ] **MTSD detection dataset preparation**: run the prep notebook (after cleanup)
- [ ] **MTSD supervised detection training**: YOLO12/YOLO26/RF-DETR notebooks (not run;
      `Results/MTSD-*` empty) — decide epochs/augmentation for the ~1.8k-image scale
- [ ] **PromptDetect batch evaluations** (not run): e.g. SAM 3/3.1 + LocateAnything on
      MTSD (prompts per sign type) and MDWD test (waste prompts); Cosmos variants opt-in
- [ ] **Inference-speed benchmarks** (tooling ready, not executed): MDWD detection,
      attribute classifiers, PromptDetect models; MTSD detection blocked on training
- [ ] **MDWD split-leakage remediation/discussion** (30 leaked sources, 15 bad boxes)
- [ ] **GRP-4/GRP-5 annotation** (1,533 images collected, unannotated) and further groups
      if in scope; re-run AttrCls/EDA as groups join
- [ ] **Attribute classifier improvement for `condition`** (macro-F1 ≤0.73): class
      imbalance (4,126 Good vs 332 Heavily Damaged) — consider re-weighting/merging
- [ ] **MDWD-EDA.ipynb / MTSD supervised notebooks**: execute top-to-bottom once, so
      outputs/cell numbers exist for the dissertation record
- [ ] **Dissertation figures/tables**: cross-track comparison table (supervised vs
      prompt-based), per-class detection results, attribute confusion matrices,
      dataset datasheets; decide which EDA figures go in
- [ ] Commit + push the new tooling (MTSD-AnnotationQA, batch_evaluation,
      gradio_compare changes, Summary.md are currently uncommitted)

## 13. Suggested next steps (practical, in priority order)

1. **Clean the MTSD annotations this week** (review → apply → re-scan → regenerate
   manifest). Everything downstream (prep notebook, detection training, attribute
   re-training, batch eval on MTSD) consumes these files; do it before any new runs.
2. **Run the MTSD prep notebook, then the three detection notebooks** (YOLO12 →
   YOLO26 → RF-DETR). Risk: ~1.4k training images with the MDWD protocol (100
   epochs, no augmentation) may underfit — consider enabling the prep notebook's
   photometric augmentation or raising epochs, and record whichever choice you make
   as a protocol deviation in the methodology.
3. **Run a small PromptDetect batch eval first** (`--max-images 25`, SAM 3 only) to
   validate the pipeline end-to-end, then the full MTSD test-split and MDWD
   test-split evaluations with 3–5 prompts per dataset. Budget GPU time for Cosmos
   2B; keep 8B/32B optional.
4. **Quantify the MDWD leakage impact**: re-evaluate the best YOLO26 checkpoint on
   val/test with the 30 leaked source images excluded; if metrics barely move, a
   dissertation footnote suffices — otherwise regenerate splits (risk: invalidates
   comparability with existing runs, so prefer exclusion-based re-evaluation).
5. **Fix `condition` head weakness** before presenting attribute results: try class
   re-weighting already in the config, or merge Weathered/Heavily Damaged into
   "Degraded" as a sensitivity analysis. Document the imbalance either way.
6. **Methodology chapter content now available**: dataset construction + QA workflow
   (LabelStudio → Final-QA → audit tool findings), EDA statistics, training
   protocols (identical-protocol ablation design), evaluation metrics definitions
   (IoU matching, macro-F1 rationale), reproducibility measures (seeds, manifests,
   W&B). Evaluation chapter: MDWD ablation results are final; MTSD attribute results
   are final pending annotation-fix sensitivity; detection + prompt-based results pending.
7. **Risks to watch**: annotation edits change QA SHA-256 hashes (manifest must be
   regenerated or training aborts); GRP-4/5 annotation effort is the biggest
   remaining manual cost; SAM 3/3.1 HF gating and the `mtsd-la` env are the usual
   PromptDetect setup friction; keep the repo private (street imagery, GDPR).
