# Urban-Vision-Benchmark — Repository Status and Review

> **Purpose:** the authoritative status ledger for the MSc dissertation repository — what exists, what has been executed, what remains, and the findings of the most recent full repository review. Companion to [README.md](README.md), which introduces the project for readers unfamiliar with it.
>
> **Last full review:** 24 July 2026 (working tree at commit `3457155` plus the 24 July MTSD evaluation artefacts). Status labels: **executed** (run to completion with recorded artefacts), **implemented** (built and verified; execution stated separately), **in progress**, **pending**, **decision required** (needs an explicit human/research decision).
>
> **In flight, not yet incorporated below:** a `strong-offline-v2` MTSD augmentation recipe and further training runs are being actively developed in the working tree as of this review (uncommitted changes under `Scripts/MTSD-Scripts/MTSD-SupervisedDetection/`). This document deliberately does not describe that work — it is unfinished and unreviewed — so the MTSD numbers below (all sourced from committed, completed runs) will be superseded once it lands. Re-review after it is committed.

---

## 1. Project snapshot

Two purpose-built Maltese street-level datasets, three experimental paradigms, one fixed comparative protocol.

| Fact | Value | Primary source |
| --- | --- | --- |
| MDWD unique source photographs | 3,598 (Roboflow v20; five classes) | integrity report + `eda_summary.json` |
| MDWD export splits | 29,487 train (~10× offline-aug) / 369 valid / 369 test | v20 exports |
| MTSD images / groups | **7,482** across GRP-1…11 (Nov 2025 – Jan 2026) | on-disk folders = Final-QA exports (verified identical 18 Jul) |
| MTSD annotated instances | **20,509** over 12 classes, each with 4 attributes | direct count over `QA-GRP*.json` (18 Jul) |
| MTSD QA state | **Complete — gate resolved 18 Jul 2026** over all 11 groups (`audit-20260718-113454`: 0 invalid values, 0 unresolved duplicates; 0 `Damaged-Unknown` remain) | `config/qa_gate.yaml` |
| MTSD prepared datasets | `mtsd-qa-v1-aug` / `mtsd-qa-v1-noaug` (built 18 Jul): 5,986/749/747 images (16,535/1,996/1,978 boxes); augmented train = 23,944 images | `prep_manifest.json` (both variants) |
| Augmentation recipe | `photometric-v1+motion-blur-v1`, **3 copies/train image** (2 photometric + 1 horizontal motion blur, kernel 9, weight 0.85) | `config/default.yaml` + manifests |
| Attribute crop manifest | 13,862 crops (11,025/1,397/1,440) — eight-group snapshot GRP-1–3, 5–9 | `outputs/manifests/manifest.json` (refreshed 13 Jul) |
| MTSD 13-model matrix | **Completed** 19–22 Jul 2026 (~74h); best model RF-DETR-M, mAP50-95 **0.686** (11-class, Tourist Sign excluded from scoring only) | `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/model_metrics.csv` |

## 2. Experimental status by track

### 2.1 MDWD supervised detection — **executed**

- YOLO11/12/26 benchmark suites (n–l) archived under `Results/MDWD-Runs/{YOLO11,YOLO12,YOLO26}-EUVIP` (+ a second YOLO26 suite, `YOLO26-DGX`, run on different hardware — see finding R6), consolidated tables under `Results/MDWD-Results/*/Model-Size-Comparison/`.
- RF-DETR **nano** executed (checkpoint + log; 30,157,870 parameters introspected from the training log); small/medium not executed, and nano's metrics are not yet normalised into the consolidated table format (finding R5).
- Split-leakage sensitivity analysis executed and **resolved**: leakage independently re-derived (30 sources; 39 under extension-normalised identity), clean-subset re-evaluation shows a negligible effect (max Δ 0.48 pp mAP50/mAP50-95). Details: [Documents/MDWDLeakageSensitivity.md](Documents/MDWDLeakageSensitivity.md). The 15 "out-of-range box" findings were adjudicated as polygon-format label lines misread by the checker — no true invalid coordinate exists.

### 2.2 MTSD supervised detection — **executed (13-model matrix + follow-ups); resolution question open**

- The QA gate was resolved on 18 July 2026 over all eleven groups, and both prepared dataset variants were built the same day under the strict pipeline (hash manifests, shared split manifest, strict validation; 6 marginally out-of-bounds boxes clipped; 2 boxless GRP-7 images retained as background; 1,273 image copies EXIF-orientation-normalised).
- **The configured 13-model dissertation matrix ran to completion** (`Results/MTSD-Runs/Supervised-Matrix/20260719-003416/`, all 13 `status: "completed"`, no scope drift from the registry — YOLO11/12/26 n/s/m, YOLO26-L, RF-DETR n/s/m — closing finding R4 in favour of the config/registry scope, not the earlier n/m/l brief). Started 2026-07-18T22:34Z, finished 2026-07-22T00:48Z (~74h13m wall clock); RF-DETR alone (n/s/m) consumed ~42h of that, dominated by RF-DETR-m at 18h13m. Every model trained on `mtsd-qa-v1-aug` (the aug3 photometric+motion-blur recipe) at 640 px, 100 epochs max, effective batch 32, AdamW, seed 42, under the resumable executor with per-run fingerprints. W&B project `MSc-MTSD-SupervisedDetection`, group `dissertation-augmented-v1`.
- **Headline unified-evaluation ranking** (`Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/model_metrics.csv`, 747 test images, 11 of 12 classes — see below), by mAP50-95:

  | Rank | Model | mAP50-95 | mAP50 | Precision | Recall |
  | --- | --- | --- | --- | --- | --- |
  | 1 | RF-DETR-M (576 px) | 0.686 | 0.784 | 0.926 | 0.704 |
  | 2 | RF-DETR-S (512 px) | 0.668 | 0.773 | 0.930 | 0.682 |
  | 3 | RF-DETR-N (384 px) | 0.623 | 0.715 | 0.906 | 0.655 |
  | 4 | YOLO26-L | 0.620 | 0.686 | 0.853 | 0.631 |
  | 5 | YOLO26-M | 0.613 | 0.672 | 0.886 | 0.611 |
  | 6 | YOLO11-M | 0.601 | 0.659 | 0.916 | 0.598 |
  | 7–10 | YOLO26-S / YOLO12-S / YOLO12-M / YOLO11-S | 0.575–0.590 | — | — | — |
  | 11–13 | YOLO12-N / YOLO11-N / YOLO26-N | 0.544–0.547 | — | — | — |

  RF-DETR sweeps the top three ranks — even RF-DETR-N at 384 px outperforms every YOLO variant including YOLO26-L. Among YOLO families, YOLO26 ≥ YOLO11 > YOLO12 at matched scale, and the ordering is monotonic in model size within every family.
- **Evaluation policy — `Tourist Sign` excluded from scoring, not from training.** Every model still trains on the full 12-class taxonomy; the standalone unified-evaluation step drops `Tourist Sign` from both ground truth and predictions (`excluded_categories: ["Tourist Sign"]` in every evaluation manifest) because it has only 33 instances (0.16% of all annotations) and its AP is unstable across splits/models (RF-DETR native per-class AP50-95 for this class alone swung between 0.01 and 0.66 across models). This keeps training taxonomy and comparative evaluation scope decoupled and documented (`Scripts/MTSD-Scripts/MTSD-SupervisedDetection/README.md`).
- **The in-training unified-evaluation step failed for all three RF-DETR runs** (`"Invalid image shape. Expected 3 channels (RGB), but got 4 channels."` — a small number of source JPEGs decode to non-RGB modes). Training itself was unaffected. The fix (`load_rgb_image()` forcing EXIF-transpose + RGB conversion, added to `mtsd_detection/evaluate.py`) plus two data-level repairs — a one-off JPEG End-Of-Image marker repair (scratchpad script, used once and removed) and `sync_shared_yolo_images.py` (found and applied 548 byte-level mismatches between the Augmented and Unaugmented prepared trees, verified 0 remaining) — unblocked a **standalone re-evaluation** (`rerun_unified_evaluation.py`, new: re-scores already-trained checkpoints without retraining) that produced the canonical `20260722-all13` results above.
- **Follow-up experiment 1 — clean augmentation ablation** (YOLO11m/12m/26m, unaugmented vs augmented, RF-DETR excluded as its online augmentation is not controllable): effect is small and mixed — Δ mAP50-95 of −0.0098 (YOLO11m), −0.0054 (YOLO12m), +0.0032 (YOLO26m) relative to the augmented baseline. The aug3 motion-blur recipe is not a major driver of the headline numbers once RF-DETR is set aside.
- **Follow-up experiment 2 — resolution scaling (decision required, R12):** the main matrix fixed 640 px throughout, and `map_small`/`ar_small` were consistently its weakest metrics (e.g. YOLO11-M: `map_small=0.071` vs `map_large=0.769`). A 960 px pilot (YOLO11-S, YOLO11-M) and a follow-on 1280 px run (YOLO11-S, YOLO11-M, YOLO26-S) both showed **monotonic, substantial mAP50-95 gains with resolution** — YOLO11-S: 0.575 (640) → 0.635 (960) → 0.663 (1280); YOLO11-M: 0.601 → 0.656 → 0.684 — with small-object AR roughly doubling. The automated go/no-go script (`run_followup_experiments.py`) never finished: it was manually stopped mid-960-pilot (`experiment_manifest.json`: `"status": "stopped_by_user"`, reason "YOLO12m 960 was progressing but impractically slow"), so the 1280 px runs were launched by hand outside that script and the matrix was never systematically extended to 960/1280. **Open decision:** whether the resolution gain justifies re-running some or all of the 13-model matrix at a higher resolution, given RF-DETR-scale training already costs ~42h at 640 px.
- **Tiled-inference pilot (YOLO11-M, RF-DETR-M @ 1280 px tiles, 20% overlap):** substantially improves small-object recall/AP (`ar_small` +0.27 to +0.28) but **reduces overall mAP50-95 sharply** (YOLO11-M −0.272, RF-DETR-M −0.366) because tile-boundary duplicate detections collapse precision (RF-DETR-M: 0.925 → 0.346) despite class-aware NMS merging. Not currently a net win over single-pass evaluation; recorded as evidence, not adopted.
- All artefacts above are under `Results/MTSD-Runs/{Supervised-Matrix,RF-DETR-MTSD,YOLO11-MTSD,YOLO12-MTSD,YOLO26-MTSD}/` and `Results/MTSD-Results/{Unified-Evaluation-NoTourist,Followup-Experiments,Tiled-Evaluation-NoTourist,Dataset-QA}/`.

### 2.3 Attribute classification — **executed (snapshot); final-scope decision required**

- **16-variant size/adaptation matrix completed 13 July 2026** on the eight-group snapshot (13,862 crops): DINOv3, V-JEPA 2.1, ConvNeXt, LingBot-Vision × base/large × frozen probe + family-appropriate adaptation (LoRA r=8, or full fine-tune for ConvNeXt). Consolidated report: `AttributeClassification/outputs/reports/size_ablation.{csv,md}`. Top test mean macro-F1: `dinov3_vitl_lora` **0.894**, `lingbot_vitl_lora` 0.892, `vjepa21_vitl_lora` 0.888. The condition head remains the weakest throughout (class imbalance).
- The historical six-variant round (GRP-1–3 snapshot) is preserved immutably and contributes no headline results.
- **Decision required (R2):** the crop-manifest tool auto-discovers Final-QA groups, so the next refresh ingests GRP-4/10/11 and changes the dataset under the completed matrix — freeze the eight-group snapshot as the reported scope, or re-run the matrix once on the final eleven-group scope.

### 2.4 Prompt-based localisation (PromptDetect) — **implemented; full runs pending**

- Six models through one backend: SAM 3 / SAM 3.1 (boxes+masks+real confidences), LocateAnything-3B (isolated `mtsd-la` worker), Cosmos Reason2 2B/8B (32B opt-in via `--allow-heavy`).
- Two versioned protocols, both implemented and verified: `dissertation-v1` (targeted, taxonomy-aware evaluation; 6 MDWD + 13 MTSD prompts) and `prompt-sensitivity-v1` (9 controlled paraphrase families × 4 variants = 36 prompts, spread/degradation statistics plus prediction-consistency analysis). Default two-stage execution runs both; atomic per-model×prompt persistence with strict resume guards.
- Executed so far: two small MDWD pilots (Jul 9–10) and one MTSD smoke sensitivity run — pipeline verification only, **not** dissertation evidence. The full fixed-protocol MDWD and MTSD test-split evaluations are the largest outstanding evidence gap.

### 2.5 Deployment benchmark and evidence tooling

- Inference-speed benchmark (all paradigms, seeded 50-image samples, cold-start/preprocess/forward/postprocess separated): **implemented, dry-run verified, not executed**.
- FinalEvaluation tooling (integrity check, robustness slices, bootstrap CIs, effort report, failure-case sampler, table export, offline evidence dashboard): **implemented and executed over available evidence**; outputs regenerate as final runs land. Attribute CIs are reconstructed exactly from stored confusion matrices (documented; per-sample dumps would additionally enable paired variant comparisons).

## 3. Dissertation documentation — synchronised 18 July 2026; MTSD results not yet incorporated

`Documentation/` holds the LaTeX chapters with dated writing notes and evidence audits. On 18 July the Introduction, Background, Literature Review and Methodology were reconciled against the repository: final MTSD counts (7,482/20,509, all-groups QA), the three-copy augmentation recipe, the prepared-dataset facts, the 16-variant matrix, both prompt protocols, the corrected attribute-bootstrap description, and the automation/verification infrastructure are now documented; Objective 2 was widened to both datasets. Remaining placeholders: MDWD collection period/devices (EUVIP paper) and RF-DETR small/medium parameter counts (not yet instantiated).

The Methodology chapter describes the training **protocol** only (no results section exists yet in any chapter), so the §2.2 training completion above does not require a chapter edit. However, `methodology_writing_notes.md`'s 18 July revision log explicitly states *"MTSD training itself not yet run (`Results/MTSD-Runs` empty)"* — now stale — and has been appended with a dated correction note (not rewritten) pointing to this section. When a Results/Evaluation chapter is drafted, its evidence hierarchy should be `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/` for the headline table and the Followup-Experiments/Tiled-Evaluation directories for the ablation and resolution discussion.

---

## 4. Repository review findings — 18 July 2026 (updated 24 July 2026: R4 and R8 resolved, R12–R13 added)

Full-repository review performed as part of the documentation pass. Findings are status flags, not a changelog — resolved items stay listed with their resolution recorded rather than being deleted, so the ledger remains a complete audit trail. Ordered by importance.

- **R1 — Stale MTSD EDA artefacts (regenerate before citing).** Every CSV under `Documents/MTSD-EDA/GeneratedCSVs/` predates the final QA-resolution pass: they report 7,492 images / 20,536 annotations, 31 `Damaged-Unknown` shapes and GRP-10 = 789, whereas the current Final-QA truth is 7,482 / 20,509, 0 `Damaged-Unknown`, GRP-10 = 779. Additionally, `gps_tag_audit.csv` contains dead paths for GRP-8…11 (files renamed during Label Studio import). Re-run the MTSD EDA (and the GPS atlas) over the final annotation state so the generated artefacts reproduce the numbers now cited in the Methodology chapter.
- **R2 — Attribute-study scope decision.** The completed 16-variant matrix is valid for its recorded eight-group manifest; the manifest auto-refresh will silently ingest GRP-4/10/11 next time it runs (any `run_all.py` invocation refreshes it). Decide: freeze and report the snapshot, or re-run the matrix once on the final scope. Until decided, avoid running `run_all.py` without `--plan`.
- **R3 — Prompt evaluations and inference benchmark are the remaining evidence gaps.** Both are fully implemented; neither has produced final evidence. They gate RQ2 and RQ4 (`Scripts/FinalEvaluation/config/research_questions.yaml` marks them pilot/pending). MTSD supervised detection (§2.2) is no longer a gap for RQ1-equivalent MTSD evidence — `research_questions.yaml` has no dedicated MTSD-detection question yet and should gain one, or fold MTSD into RQ1's scope, once the resolution question (R12) is settled.
- **R4 — Training-plan vs configured matrix discrepancy — RESOLVED 22 July 2026.** The executed 13-model matrix used the registry/`config/default.yaml` scope (YOLO11/12/26 n/s/m + YOLO26-L + RF-DETR n/s/m), not the earlier "n, m, l" brief; all 13 completed with no mid-matrix scope changes. No `yolo11l`, `yolo12l` or `rfdetr-l` was registered or trained. Recorded here for provenance; no further action.
- **R5 — RF-DETR (MDWD) remains a partial result.** Only nano executed; metrics not yet normalised into the consolidated MDWD tables, so cross-family MDWD claims are currently YOLO-internal. Either export/validate nano into the standard format and run small/medium, or scope the MDWD cross-family claim accordingly.
- **R6 — Hardware provenance of the second YOLO26 MDWD suite.** `Results/MDWD-Runs/YOLO26-DGX` was trained on different hardware, while the Methodology's environment section names only the RTX 4090 workstation. Decide which suite is headline evidence (re-run vs amend the environment statement); this affects the Evaluation chapter, not the pipelines.
- **R7 — MDWD source-count conflict with the EUVIP paper.** The paper cites 3,697 source images / 11,461 instances; the repository v20 release contains 3,598 unique sources and no artefact reproducing 11,461. The dissertation currently reports the repository-verified figures; reconcile with the paper (or state which release each figure describes) before submission.
- **R8 — Stale counts in secondary documentation — RESOLVED 24 July 2026.** `Scripts/MTSD-Scripts/README.md`, `Scripts/MTSD-Scripts/AttributeClassification/README.md` and `Documents/MTSD-Supervised-Detection-Automation.md` quoted pre-final counts (3,098/8,372, five-group scope); all three now state the current eleven-group QA-resolved scope (7,482/20,509) while preserving the old figures as explicitly dated historical values. Dated audit/notes files under `Documentation/` continue to retain old numbers legitimately as historical records (unchanged, by design). Root README.md and this file are current.
- **R9 — QA gate approver field.** `config/qa_gate.yaml` has `approved_by: null`. Record the human approver (and optionally the decision note) for provenance before the first `--final` training run consumes the gate.
- **R10 — Annotation-hour values unrecorded.** `FinalEvaluation/config/annotation_effort_manual.yaml` still reports all manual hours as *not recorded*; the effort report will honestly say so unless values are filled in.
- **R11 — Minor hygiene.** (a) Local Label Studio state (`label_studio.sqlite3` under GRP-5/6) is covered by the `Datasets/MTSD/GRP-*/` ignore rule — confirmed not tracked; no action needed beyond awareness. (b) The 2026-07-16/17 interpretability workflow was never committed and no longer exists; it is correctly absent from the Methodology chapter — do not cite it. (c) Two GRP-7 images legitimately contain no annotations and are retained as background training images (documented in the Methodology).
- **R12 — MTSD resolution scaling: decision required.** 640→960→1280 px pilots show monotonic, substantial mAP50-95 gains (YOLO11-M: 0.601→0.656→0.684) and roughly double small-object AR, but the automated decision script was manually stopped mid-pilot and the 13-model matrix was never systematically extended past 640 px. Given RF-DETR-scale training already costs ~42h at 640 px for just three sizes, re-running some or all of the matrix at a higher resolution is a real time/evidence-quality trade-off that needs an explicit decision before further training is scheduled (relevant to the strong-augmentation work currently in progress, since a resolution decision made now affects what that work should target).
- **R13 — MTSD tiled-inference pilot: informative negative result, not a fix.** Overlapping 1280 px tiled inference (YOLO11-M, RF-DETR-M) roughly doubles small-object AR but collapses precision via tile-boundary duplicates, cutting overall mAP50-95 by 0.27–0.37. This is evidence the small-object weakness is real (R12) but tiling with the current merge settings is not the remedy; record it as a tested-and-rejected approach rather than re-attempting it unchanged.

## 5. Priority next steps

1. **Decide the MTSD resolution question (R12)** before scheduling further training — this also bears on the in-progress strong-augmentation work, so settling it first avoids training the new recipe at a resolution that gets revisited immediately after.
2. **Regenerate the final MTSD EDA (R1)** so every cited statistic has a matching generated artefact.
3. **Decide the attribute-study scope (R2)**; if re-running, use a distinctly labelled final round and preserve the snapshot results.
4. **Execute the PromptDetect fixed-protocol evaluations** (both datasets, both protocols) and then the **inference-speed benchmark** (R3) — the last two evidence gaps for RQ2/RQ4.
5. **Close the remaining provenance items**: RF-DETR normalisation or scoping (R5), YOLO26 suite decision (R6), EUVIP count reconciliation (R7), gate approver (R9), annotation hours (R10).
6. **Refresh the evidence layer** (robustness slices, bootstrap, effort report, dashboard, dissertation tables) after each of the above lands — all tools are read-only and re-runnable, and should also be re-run once the in-progress strong-augmentation training completes and is committed.
7. **Draft the Results/Evaluation chapter** for MTSD supervised detection once R12 is settled — the unified-evaluation, follow-up-ablation and tiled-inference evidence (§2.2) is ready to cite.

---

*This document supersedes all previous versions of Summary.md; historical status entries were retired once their content was verified as either completed (and recorded above) or stale (and flagged in §4).*
