# Urban Vision Benchmark

**A comparative benchmark of vision-based perception paradigms for urban waste and infrastructure monitoring in Malta.**

This repository is the complete experimental record of an MSc dissertation at the **University of Malta** (2026). It contains the code, configurations, annotation records, quality-assurance tooling and numeric results behind two purpose-built Maltese street-level datasets and three families of computer-vision experiments run on them:

1. **Supervised object detection** — three YOLO generations (YOLO11, YOLO12, YOLO26) and the RF-DETR detection transformer, trained and compared under one fixed protocol.
2. **Prompt-based, zero-shot localisation** — SAM 3 / SAM 3.1, LocateAnything-3B and NVIDIA Cosmos Reason2 (2B/8B/32B), evaluated against fixed municipal ground truth from natural-language prompts alone, including a dedicated prompt-sensitivity protocol.
3. **Multi-attribute representation learning** — a shared-backbone, four-head traffic-sign attribute classifier comparing frozen probing, LoRA and full fine-tuning across DINOv3, V-JEPA 2.1, ConvNeXt and LingBot-Vision at two model sizes (16 variants).

Everything is driven by version-controlled configuration, fixed seeds (42 throughout), SHA-256-hashed dataset manifests and immutable run directories, so that every reported number traces back to a recorded artefact.

> **Data availability & privacy.** The raw street-level imagery contains personal data (faces, number plates) and is **not distributed** with this repository; it is stored locally under GDPR-compliant handling, and a guarded redaction workflow (face/plate/QR blurring with manual preview) governs anything designated for sharing. The repository publishes code, configuration, annotation records and numeric results only.

---

## The datasets

### MDWD — Maltese Domestic Waste Dataset

Street-level photographs of domestic waste presented for kerbside collection in Malta, annotated for object detection with **five operational classes** aligned with the national collection streams: `Mixed Waste`, `Organic Waste`, `Recyclable Material`, `Orange CMD`, `Other Waste`.

| Property | Value |
| --- | --- |
| Unique source photographs | 3,598 |
| Working release | Roboflow project v20, 640×640 |
| Export splits (images) | train 29,487 (~10× offline-augmented) / valid 369 / test 369 |
| Export formats | YOLO (`MDWD-YOLO11/12/26`) and COCO (`MDWD-RFDETR`) — four variants of one annotation state |
| Integrity | dataset-integrity checker + independent split-leakage sensitivity analysis (effect measured **negligible**, max Δ 0.48 pp mAP50-95 — [Documents/MDWDLeakageSensitivity.md](Documents/MDWDLeakageSensitivity.md)) |

### MTSD — Maltese Traffic Sign Dataset

Smartphone photographs of Maltese traffic-sign and roadside infrastructure, collected in eleven groups (GRP-1…GRP-11) between November 2025 and January 2026. **Annotation and quality assurance are complete over all eleven groups** (QA gate resolved 18 July 2026).

| Property | Value |
| --- | --- |
| Images | 7,482 (72.6% with EXIF timestamps, 57.9% with GPS coordinates) |
| Annotated instances | 20,509 bounding boxes |
| Detection classes | 12 (regulatory, warning, navigational and municipal signage, incl. explicit `Back-Unknown` / `Other-Unknown` categories) |
| Per-box attributes | viewing angle (Front/Side/Back) · mounting (Pole/Wall) · condition (Good/Weathered/Heavily Damaged) · shape (Circular/Quadrangle/Triangular/Octagonal/Pentagon) |
| Canonical record | one QA-verified COCO JSON per group (`Datasets/MTSD/Annotations/GRP-*/Final-QA/`) |
| Prepared datasets | `mtsd-qa-v1-aug` and `mtsd-qa-v1-noaug` (built 18 July 2026): deterministic 80/10/10 source-level split — 5,986 train / 749 valid / 747 test images; the augmented variant adds **three offline copies per training image** (two photometric + one motion-blur, recipe `photometric-v1+motion-blur-v1`) for 23,944 training images |

Every annotated sign carries all four attributes, which turns the detection inventory into a maintenance-oriented condition-assessment resource — a combination no established public traffic-sign dataset provides.

---

## Experimental tracks and status

| Track | Scope | Status (18 July 2026) |
| --- | --- | --- |
| MDWD supervised detection | YOLO11/12/26 (n–l suites) + RF-DETR | **Executed**; runs archived, leakage-checked (RF-DETR nano executed; small/medium pending) |
| MTSD supervised detection | 13-model matrix: YOLO11/12/26 (n/s/m), YOLO26-L, RF-DETR (n/s/m) + 3-model augmentation ablation | **Prepared datasets built; training starting** (resumable matrix executor, QA-gated) |
| Prompt-based localisation | SAM 3/3.1, LocateAnything-3B, Cosmos Reason2 2B/8B (32B opt-in) under `dissertation-v1` + `prompt-sensitivity-v1` (36 prompts, 9 paraphrase families) | Protocols implemented and pilot-verified; full fixed-protocol runs pending |
| Attribute classification | 16-variant size/adaptation matrix over 4 backbone families × 2 sizes × frozen/adapted | **Executed** 13 July 2026 on the eight-group snapshot (13,862 crops); best variant `dinov3_vitl_lora`, test mean macro-F1 0.894 |
| Deployment benchmark | latency / throughput / memory across all paradigms | Implemented, dry-run verified; execution pending |

The MTSD training matrix consumes the newly built augmented set, whose third (motion-blur) copy strengthens the earlier two-copy photometric recipe; the controlled ablation (YOLO11m/12m/26m on augmented vs unaugmented data) isolates the augmentation's causal effect.

---

## Repository structure

```
Urban-Vision-Benchmark/
├── Datasets/
│   ├── MDWD/                          # Waste-detection exports (YOLO11/12/26 + RF-DETR variants)
│   └── MTSD/                          # GRP-1..11 raw groups, Annotations/ (Final-QA), Prepared/
├── Documentation/                     # Dissertation chapters (LaTeX) + writing notes/audits
│   ├── 1. Introduction/
│   ├── 2-3. Background-LiteratureReview/
│   └── 4. Methodology/
├── Documents/                         # Workflow guides + generated EDA/report artefacts
│   ├── MDWD-EDA/  MTSD-EDA/           # EDA figures, CSVs, interactive GPS atlas
│   ├── Final-Reports/                 # Integrity, leakage, health, audit, dashboard outputs
│   ├── Final-Figures/  Final-Tables/  # Dissertation-ready exports
│   └── *.md                           # Protocol and workflow documentation
├── Models/                            # Stock pretrained checkpoints (untracked, re-downloadable)
├── Requirements/                      # Pip requirements + conda environment definitions
├── Results/
│   ├── MDWD-Runs/  MDWD-Results/      # Archived MDWD runs + consolidated tables
│   ├── MTSD-Runs/  MTSD-Results/      # MTSD supervised outputs (matrix in progress)
│   └── PromptDetect/                  # Batch prompt-evaluation runs
├── Scripts/
│   ├── Automation/                    # Workflow registry/runner + repository health verifier
│   ├── FinalEvaluation/               # Read-only evidence tooling (see below)
│   ├── MDWD-Scripts/                  # MDWD EDA + supervised training notebooks
│   ├── MTSD-Scripts/
│   │   ├── AttributeClassification/   # 16-variant multi-attribute pipeline
│   │   ├── LabelStudio/               # Annotation import/setup/QA-formatting tooling
│   │   ├── MTSD-Analysis/             # EDA package, GPS atlas, audit utilities
│   │   ├── MTSD-AnnotationQA/         # Audit → visual review → guarded-apply QA pipeline
│   │   ├── MTSD-SupervisedDetection/  # QA-gated preparation + resumable training matrix
│   │   └── MTSD-SupervisedNotebooks/  # Thin read-only notebook front ends
│   └── Other-Scripts/
│       ├── GDPR-Compliance/           # Face/plate/QR redaction with preview-and-verify
│       ├── Inference-Benchmark/       # Cross-paradigm inference-speed benchmark
│       └── PromptDetect/              # Prompt-based evaluation backend, protocols, apps
├── tests/                             # Launcher lifecycle tests (pipeline tests live per-package)
├── launch_uvb.py / launch_uvb.ps1     # Managed launcher for all interactive tools
├── README.md                          # This file
└── Summary.md                         # Detailed status, review findings and next steps
```

---

## Getting started

### Environments

Four isolated conda environments cover mutually incompatible dependency stacks (definitions and setup scripts in [Requirements/CondaEnvironments/](Requirements/CondaEnvironments/); pip layers in [Requirements/](Requirements/)):

| Environment | Used by | Key pins |
| --- | --- | --- |
| `MDWD` | Detection training (YOLO + RF-DETR) and MDWD EDA | Ultralytics 8.4.x, `rfdetr` 1.3.0, NumPy < 2 |
| `mtsd-attrcls` | Attribute-classification pipeline | Transformers 5.x, PEFT, timm, `lingbot-vision` |
| `mtsd-base` | PromptDetect (SAM 3/3.1, Cosmos), GDPR redaction, QA review app | Meta `sam3` package, Transformers 5.x |
| `mtsd-la` | LocateAnything worker (HTTP subprocess) | Transformers 4.57.x |

### Secrets

Create a git-ignored `.env` at the repository root for Weights & Biases logging (not required for offline work):

```
WANDB_MODE=online
WANDB_API_KEY=<your key>
```

W&B projects: `MSc-MDWD-EUVIP26` (MDWD detection), `MSc-MTSD-SupervisedDetection` (MTSD detection), `MSc-MTSD-Attributes` (attribute classification). W&B is a monitoring mirror; the authoritative record is the repository artefacts.

### The UVB launcher

`python launch_uvb.py` opens a searchable menu over every safe tool in the repository — EDA, audits, review apps, evaluators, reports — resolving the right conda environment per tool. Web applications run as **managed processes**: the launcher polls readiness before opening a browser, offers reopen/restart/stop for running apps, keeps per-app logs under `.uvb_launcher_logs/`, and never starts training, applies annotation fixes or deletes anything. Heavy/mutating commands sit behind explicit confirmation.

### The workflow runner

Batch workflows are registered as named targets in [Scripts/Automation/](Scripts/Automation/README.md) and executed via `run_urban_workflows.ps1`: every run is logged, notebooks execute headlessly into timestamped copies (sources never modified), `-DryRun` previews any target, and training targets refuse to run without an explicit `-AllowTraining` flag. `verify_repository_health.py` is the local pre-commit/pre-push health check (structure, notebook validity, compilation, stale paths, credential leaks, link integrity).

---

## Key workflows

```bash
# MDWD exploratory analysis
python Scripts/MDWD-Scripts/MDWD-Analysis/run_eda.py

# MTSD annotation QA (read-only audit → visual review → guarded apply)
python Scripts/MTSD-Scripts/MTSD-AnnotationQA/scan_annotations.py --dry-run
python Scripts/MTSD-Scripts/MTSD-AnnotationQA/review_app.py
python Scripts/MTSD-Scripts/MTSD-AnnotationQA/apply_fixes.py --decisions <file> --apply

# MTSD supervised detection (QA-gated; refuses on unresolved audit findings)
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --dry-run
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --prepare-only --dataset-variant both --final
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --validate-prepared --strict
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --matrix dissertation --final

# Attribute classification (16-variant matrix; read-only planning first)
python Scripts/MTSD-Scripts/AttributeClassification/run_all.py --plan --profile size_ablation_all
python Scripts/MTSD-Scripts/AttributeClassification/run_all.py --profile size_ablation_all

# Prompt-based evaluation (both protocols, resumable, atomic per model×prompt)
python Scripts/Other-Scripts/PromptDetect/batch_evaluation/run_dissertation_protocol.py --dataset both --split test --dry-run
python Scripts/Other-Scripts/PromptDetect/batch_evaluation/run_dissertation_protocol.py --dataset both --split test --final

# Cross-paradigm inference benchmark
python Scripts/Other-Scripts/Inference-Benchmark/inference_speed_benchmark.py --dataset MDWD --task detection --dry-run
```

### Dissertation evidence tooling — [Scripts/FinalEvaluation/](Scripts/FinalEvaluation/README.md)

A read-only analysis layer over stored results (nothing here trains or silently re-runs inference): dataset-integrity sign-off, the MDWD leakage-sensitivity analysis, robustness slice analysis (twelve slice dimensions from stored predictions), seeded bootstrap confidence intervals, an annotation/operational-effort report (no composite scores), failure-case sampling, final table/figure export with per-cell provenance, and a fully offline HTML evidence dashboard mapping each research question to its evidence. All tools have `--self-test` and `--dry-run` modes and write to fresh timestamped folders.

---

## Reproducibility and data governance

- **Seeds and determinism**: seed 42 everywhere; deterministic execution enabled where frameworks support it; splits assigned at source-image level by deterministic (salted-hash) algorithms that are never re-salted.
- **Immutable artefacts**: prepared datasets and annotation exports carry SHA-256 manifests; run directories are name-encoded, never overwritten, and export their effective configuration verbatim.
- **QA gating**: MTSD dataset preparation is blocked by a machine-readable QA gate that names the audit it trusts; preparation refuses to consume unaudited annotation state.
- **Testing**: pytest suites cover the preparation, QA, augmentation, splitting, attribute and prompt-evaluation packages, including consumer-compatibility tests that run real framework data loaders against synthetic prepared datasets; every attribute variant has a smoke-test mode.
- **Privacy**: raw imagery is untracked and access-restricted; the GDPR redaction workflow (automatic face/plate/QR detection, manual preview, guarded apply) governs any imagery designated for sharing; GPS metadata is used only in aggregate.

---

## Dissertation documentation

The LaTeX chapters live under [Documentation/](Documentation/) (Introduction; Background; Literature Review; Methodology), each accompanied by dated writing notes and evidence audits that map every claim to its repository artefact. Chapter state as of 18 July 2026: dataset facts synchronised with the final eleven-group MTSD scope, the three-copy augmentation recipe, the completed 16-variant attribute matrix and both prompt protocols.

## Citation

> A. F. Lucas, *A Comparative Study of Vision-Based Perception Paradigms for Urban Waste and Infrastructure Monitoring* (working title), MSc dissertation, University of Malta, 2026. Repository: `AFLucas-UOM/Urban-Vision-Benchmark`.

Please also see [Summary.md](Summary.md) for the detailed status ledger, the current repository review findings and the prioritised remaining work.
