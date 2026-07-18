# Urban-Vision-Benchmark — Repository Status and Review

> **Purpose:** the authoritative status ledger for the MSc dissertation repository — what exists, what has been executed, what remains, and the findings of the most recent full repository review. Companion to [README.md](README.md), which introduces the project for readers unfamiliar with it.
>
> **Last full review:** 18 July 2026 (working tree at commit `0c344db` plus the 18 July QA/preparation artefacts). Status labels: **executed** (run to completion with recorded artefacts), **implemented** (built and verified; execution stated separately), **in progress**, **pending**, **decision required** (needs an explicit human/research decision).

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

## 2. Experimental status by track

### 2.1 MDWD supervised detection — **executed**

- YOLO11/12/26 benchmark suites (n–l) archived under `Results/MDWD-Runs/{YOLO11,YOLO12,YOLO26}-EUVIP` (+ a second YOLO26 suite, `YOLO26-DGX`, run on different hardware — see finding R6), consolidated tables under `Results/MDWD-Results/*/Model-Size-Comparison/`.
- RF-DETR **nano** executed (checkpoint + log; 30,157,870 parameters introspected from the training log); small/medium not executed, and nano's metrics are not yet normalised into the consolidated table format (finding R5).
- Split-leakage sensitivity analysis executed and **resolved**: leakage independently re-derived (30 sources; 39 under extension-normalised identity), clean-subset re-evaluation shows a negligible effect (max Δ 0.48 pp mAP50/mAP50-95). Details: [Documents/MDWDLeakageSensitivity.md](Documents/MDWDLeakageSensitivity.md). The 15 "out-of-range box" findings were adjudicated as polygon-format label lines misread by the checker — no true invalid coordinate exists.

### 2.2 MTSD supervised detection — **in progress (training starting)**

- The QA gate was resolved on 18 July 2026 over all eleven groups, and both prepared dataset variants were built the same day under the strict pipeline (hash manifests, shared split manifest, strict validation; 6 marginally out-of-bounds boxes clipped; 2 boxless GRP-7 images retained as background; 1,273 image copies EXIF-orientation-normalised).
- The augmented training set now includes the **third (aug3) motion-blur copy** — a deliberate strengthening of the earlier two-copy photometric recipe (`add_mtsd_motion_blur_aug3.py` retrofits/refreshes it deterministically).
- Training is about to begin for the configured **13-model dissertation matrix** — YOLO11/12/26 at n/s/m, YOLO26-L, RF-DETR n/s/m — under the shared protocol (100 epochs max, 640 px, effective batch 32, AdamW, seed 42, resumable executor with per-run fingerprints), plus the **3-model augmentation ablation** (YOLO11m/12m/26m on augmented vs unaugmented; YOLO26-L optional). See finding R4 on the size-scope discrepancy.
- `Results/MTSD-Runs` / `MTSD-Results` are empty until these runs land. W&B: `MSc-MTSD-SupervisedDetection`.

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

## 3. Dissertation documentation — synchronised 18 July 2026

`Documentation/` holds the LaTeX chapters with dated writing notes and evidence audits. On 18 July the Introduction, Background, Literature Review and Methodology were reconciled against the repository: final MTSD counts (7,482/20,509, all-groups QA), the three-copy augmentation recipe, the prepared-dataset facts, the 16-variant matrix, both prompt protocols, the corrected attribute-bootstrap description, and the automation/verification infrastructure are now documented; Objective 2 was widened to both datasets. Remaining placeholders: MDWD collection period/devices (EUVIP paper) and RF-DETR small/medium parameter counts (not yet instantiated).

---

## 4. Repository review findings — 18 July 2026

Full-repository review performed as part of the documentation pass. **Nothing below has been modified; each item awaits the author's decision.** Ordered by importance.

- **R1 — Stale MTSD EDA artefacts (regenerate before citing).** Every CSV under `Documents/MTSD-EDA/GeneratedCSVs/` predates the final QA-resolution pass: they report 7,492 images / 20,536 annotations, 31 `Damaged-Unknown` shapes and GRP-10 = 789, whereas the current Final-QA truth is 7,482 / 20,509, 0 `Damaged-Unknown`, GRP-10 = 779. Additionally, `gps_tag_audit.csv` contains dead paths for GRP-8…11 (files renamed during Label Studio import). Re-run the MTSD EDA (and the GPS atlas) over the final annotation state so the generated artefacts reproduce the numbers now cited in the Methodology chapter.
- **R2 — Attribute-study scope decision.** The completed 16-variant matrix is valid for its recorded eight-group manifest; the manifest auto-refresh will silently ingest GRP-4/10/11 next time it runs (any `run_all.py` invocation refreshes it). Decide: freeze and report the snapshot, or re-run the matrix once on the final scope. Until decided, avoid running `run_all.py` without `--plan`.
- **R3 — Prompt evaluations and inference benchmark are the remaining evidence gaps.** Both are fully implemented; neither has produced final evidence. They gate RQ2 and RQ4 (`Scripts/FinalEvaluation/config/research_questions.yaml` marks them pilot/pending).
- **R4 — Training-plan vs configured matrix discrepancy.** The current plan mentions YOLO sizes *n, m, l*; the registry and `config/default.yaml` define the matrix as n/s/m per YOLO family + YOLO26-L + RF-DETR n/s/m — no `yolo11l`, `yolo12l` or `rfdetr-l` is registered. Reconcile intent vs configuration *before* launching the matrix (adding registry entries later restarts nothing, but mid-matrix scope changes complicate the "identical protocol" claim).
- **R5 — RF-DETR (MDWD) remains a partial result.** Only nano executed; metrics not yet normalised into the consolidated MDWD tables, so cross-family MDWD claims are currently YOLO-internal. Either export/validate nano into the standard format and run small/medium, or scope the MDWD cross-family claim accordingly.
- **R6 — Hardware provenance of the second YOLO26 MDWD suite.** `Results/MDWD-Runs/YOLO26-DGX` was trained on different hardware, while the Methodology's environment section names only the RTX 4090 workstation. Decide which suite is headline evidence (re-run vs amend the environment statement); this affects the Evaluation chapter, not the pipelines.
- **R7 — MDWD source-count conflict with the EUVIP paper.** The paper cites 3,697 source images / 11,461 instances; the repository v20 release contains 3,598 unique sources and no artefact reproducing 11,461. The dissertation currently reports the repository-verified figures; reconcile with the paper (or state which release each figure describes) before submission.
- **R8 — Stale counts in secondary documentation.** `Scripts/MTSD-Scripts/README.md`, `Scripts/MTSD-Scripts/AttributeClassification/README.md` and `Documents/MTSD-Supervised-Detection-Automation.md` still quote pre-final counts (3,098/8,372, five-group scope, or 7,492). Dated audit/notes files under `Documentation/` retain old numbers legitimately as historical records. Root README.md and this file are now current.
- **R9 — QA gate approver field.** `config/qa_gate.yaml` has `approved_by: null`. Record the human approver (and optionally the decision note) for provenance before the first `--final` training run consumes the gate.
- **R10 — Annotation-hour values unrecorded.** `FinalEvaluation/config/annotation_effort_manual.yaml` still reports all manual hours as *not recorded*; the effort report will honestly say so unless values are filled in.
- **R11 — Minor hygiene.** (a) Local Label Studio state (`label_studio.sqlite3` under GRP-5/6) is covered by the `Datasets/MTSD/GRP-*/` ignore rule — confirmed not tracked; no action needed beyond awareness. (b) The 2026-07-16/17 interpretability workflow was never committed and no longer exists; it is correctly absent from the Methodology chapter — do not cite it. (c) Two GRP-7 images legitimately contain no annotations and are retained as background training images (documented in the Methodology).

## 5. Priority next steps

1. **Reconcile the matrix scope (R4), then launch MTSD supervised training** — smoke test first, then the resumable dissertation matrix and the augmentation ablation with `--final`. Do not alter the 100-epoch/640 px/batch-32 protocol mid-matrix.
2. **Regenerate the final MTSD EDA (R1)** so every cited statistic has a matching generated artefact.
3. **Decide the attribute-study scope (R2)**; if re-running, use a distinctly labelled final round and preserve the snapshot results.
4. **Execute the PromptDetect fixed-protocol evaluations** (both datasets, both protocols) and then the **inference-speed benchmark** (R3) — the last two evidence gaps for RQ2/RQ4.
5. **Close the provenance items**: RF-DETR normalisation or scoping (R5), YOLO26 suite decision (R6), EUVIP count reconciliation (R7), secondary-doc count refresh (R8), gate approver (R9), annotation hours (R10).
6. **Refresh the evidence layer** (robustness slices, bootstrap, effort report, dashboard, dissertation tables) after each of the above lands — all tools are read-only and re-runnable.

---

*This document supersedes all previous versions of Summary.md; historical status entries were retired once their content was verified as either completed (and recorded above) or stale (and flagged in §4).*
