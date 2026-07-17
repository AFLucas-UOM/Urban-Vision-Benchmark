# Urban Vision Benchmark

**Central experimental repository for an MSc dissertation on computer vision for
urban waste and infrastructure monitoring in Malta** (University of Malta).

The repository unifies two research tracks — domestic waste detection and
traffic-sign understanding — in one organised, reproducible place: datasets,
training notebooks and scripts, configurations, Weights & Biases run records,
results, analysis tooling and supporting documentation.

> **Privacy note:** this is a **private** research repository. The datasets
> contain street-level imagery collected in Malta, and models/results are
> unpublished dissertation material. It is not intended for public release
> unless explicitly cleaned and licensed later.

---

## The two datasets

### MDWD — Maltese Domestic Waste Dataset

Street-level photographs of domestic waste put out for kerbside collection in
Malta, annotated for object detection with **5 classes**: `Mixed Waste`,
`Orange CMD`, `Organic Waste`, `Other Waste`, `Recyclable Material`.

The working exports (Roboflow project v20, 640×640, ~10× augmentation on the
training split) live under [Datasets/MDWD/](Datasets/MDWD/) in four
framework-specific variants of the same data:

| Variant | Format | Splits (images) |
| --- | --- | --- |
| `MDWD-YOLO11` / `MDWD-YOLO12` / `MDWD-YOLO26` | YOLO (`data.yaml` + txt labels) | train 29,487 / valid 369 / test 369 |
| `MDWD-RFDETR` | COCO (`_annotations.coco.json`) | same splits |

### MTSD — Maltese Traffic Sign Dataset

Photographs of Maltese traffic signs collected in **11 groups**
([Datasets/MTSD/GRP-1 … GRP-11](Datasets/MTSD/)), each group holding raw
captures under `Images/`. QA-verified annotations currently exist for
**GRP-1, GRP-2, GRP-3, GRP-5 and GRP-6** under
[Datasets/MTSD/Annotations/](Datasets/MTSD/Annotations/): the five
COCO-style JSONs (`GRP-*/Final-QA/QA-GRP*.json`) contain **3,098 annotated
images / 8,372 boxes**. Their boxes carry **per-sign auxiliary attributes** —
viewing angle, mounting type, condition, and sign shape — used for the
multi-attribute classification experiments. The completed attribute-model
comparison is a **historical GRP-1–GRP-3 snapshot** (1,971 images / 5,273
retained crops); GRP-5 is deliberately deferred until the final annotation
scope is fixed. Remaining groups join automatically once their `Final-QA`
JSON exists.

---

## The two experimental tracks

1. **Domestic waste detection (MDWD).** Supervised object-detection
    benchmarks across model families and sizes: **YOLO11, YOLO12, YOLO26**
    (n/s/m/l) and **RF-DETR nano**. RF-DETR small/medium configurations are
    archived as templates but have no completed checkpoints, so they are not
    reported as executed results. Runs are produced from the notebooks in
    [Scripts/MDWD-Scripts/MDWD-SupervisedNotebooks/](Scripts/MDWD-Scripts/MDWD-SupervisedNotebooks/),
    with runs archived under [Results/MDWD-Runs/](Results/MDWD-Runs/) and
    consolidated metrics under [Results/MDWD-Results/](Results/MDWD-Results/).
    The RF-DETR nano checkpoint/log exists, but its metrics are not yet in the
    standard consolidated table format and should not be used for a quantitative
    cross-family conclusion until exported and validated.

2. **Traffic-sign multi-attribute classification (MTSD).** Representation
   learning on QA-approved sign crops with four heads (view angle, mounting,
   condition, shape) across backbone/adaptation variants — **DINOv3**
   (frozen/LoRA), **V-JEPA 2.1** (frozen/LoRA) and **ConvNeXt**
   (frozen/fine-tuned) — in
   [Scripts/MTSD-Scripts/AttributeClassification/](Scripts/MTSD-Scripts/AttributeClassification/).
   Supporting tooling covers annotation QA, Label Studio conversion, GDPR
   redaction, promptable-detection experiments, and inference-speed benchmarking.

---

## Repository structure

```
Urban-Vision-Benchmark/
├── Datasets/
│   ├── MDWD/                      # Waste-detection exports (YOLO11/12/26 + RF-DETR)
│   └── MTSD/                      # Traffic-sign groups GRP-1..11 + Annotations/
├── Documents/                     # Workflow docs + generated EDA outputs
│   ├── MTSD-EDA/                  #   MTSD EDA figures/CSVs/HTML map
│   ├── MDWD-EDA/                  #   MDWD EDA figures/CSVs/summary
│   ├── Final-Reports/             #   Health checks, integrity + leakage sensitivity reports
│   └── *.md                       #   LabelStudio, QA formatter, PromptDetect, leakage analysis, etc.
├── Models/                        # Stock pretrained weights (YOLO11/12/26, RF-DETR)
├── Requirements/                  # pip requirements per workstream
├── Results/
│   ├── MDWD-Results/              # Consolidated benchmark tables/plots per model family
│   ├── MDWD-Runs/                 # Archived training runs (weights, curves, configs)
│   ├── MTSD-Results/              # (reserved for MTSD result exports)
│   ├── MTSD-Runs/                 # (reserved for MTSD run archives)
│   └── PromptDetect/              # Batch prompt-evaluation outputs
└── Scripts/
    ├── MDWD-Scripts/
    │   ├── MDWD-Analysis/         # MDWD EDA package + runner (see its README)
    │   └── MDWD-SupervisedNotebooks/  # YOLO26 / YOLO12 / RF-DETR training notebooks
    ├── FinalEvaluation/           # Dataset integrity + MDWD leakage sensitivity reports
    ├── MTSD-Scripts/              # (see its README)
    │   ├── AttributeClassification/   # Multi-attribute training pipeline + configs
    │   ├── LabelStudio/           # Annotation-workflow tooling
    │   ├── MTSD-Analysis/         # MTSD EDA package + notebook + audit scripts
    │   ├── MTSD-AnnotationQA/     # Audit/review/apply workflow for Final-QA JSONs
    │   └── update_annotation_paths.ps1
    └── Other-Scripts/
        ├── GDPR-Compliance/       # Face/plate detection + redaction pipeline
        ├── Inference-Benchmark/   # Inference-speed benchmark scripts
        └── PromptDetect/          # SAM 3 / Cosmos Reason2 / LocateAnything app + batch eval
```

### What each top-level folder is for

- **Datasets/** — the data itself. Raw images and heavy exports are *not*
  tracked in git (see [.gitignore](.gitignore)); the irreplaceable small files
  are: MDWD `data.yaml`/Roboflow READMEs and the MTSD `Final-QA` QA JSONs.
- **Documents/** — human-readable workflow documentation plus the generated
  EDA artefacts that the dissertation figures draw from.
- **Models/** — stock pretrained checkpoints used to initialise training
  (re-downloadable; untracked). Trained weights live inside their run folders
  under `Results/MDWD-Runs/**/weights/` and in W&B artifacts.
- **Requirements/** — pip requirement files per workstream (attribute
  classification, traditional training, PromptDetect, LocateAnything, general
  tooling).
- **Results/** — the experimental record. Run folders keep everything
  (weights, curves, val images); git tracks the numeric record (CSV/YAML/JSON)
  while weights and per-run media stay local-only.
- **Scripts/** — all code, grouped by track (`MDWD-Scripts`, `MTSD-Scripts`)
  plus shared tooling (`Other-Scripts`).

---

## MDWD workstream

### EDA / analysis — [Scripts/MDWD-Scripts/MDWD-Analysis/](Scripts/MDWD-Scripts/MDWD-Analysis/README.md)

`run_eda.py` maps a dataset variant (YOLO or COCO auto-detected), runs
integrity checks (missing/orphan labels, invalid class ids, out-of-range
boxes, unreadable images, cross-split leakage of augmented sources, …) and
writes statistics tables, ten publication-style figures and
`eda_summary.json` to `Documents/MDWD-EDA/`. `visualise_samples.py` renders
annotated sample images. Two `DatasetVisualisation*.ipynb` notebooks are kept
as **legacy** provenance from the pre-reorganisation repo.

```bash
python Scripts/MDWD-Scripts/MDWD-Analysis/run_eda.py                 # full EDA
python Scripts/MDWD-Scripts/MDWD-Analysis/visualise_samples.py       # annotated samples
```

### Supervised detection notebooks — [Scripts/MDWD-Scripts/MDWD-SupervisedNotebooks/](Scripts/MDWD-Scripts/MDWD-SupervisedNotebooks/)

`YOLO26.ipynb`, `YOLO12.ipynb` and `RF-DETR.ipynb` each run a multi-size
benchmark suite (dataset check/download → environment check → per-size
training → val/test evaluation → W&B logging → summary CSV export):

- Datasets resolve to `Datasets/MDWD/<variant>` from the repository root
  (found by walking up from the working directory).
- New runs are created under `Results/MDWD-Runs/<suite>/E###_<config>/`;
  a run directory that already exists is **never overwritten** (the notebook
  raises instead).
- Consolidated metrics/config exports go to
  `Results/MDWD-Results/<model>/Model-Size-Comparison/`.
- W&B project: `MSc-MDWD-EUVIP26`. Historical local W&B files and scratch
  `runs/` from earlier sessions remain in the notebook folder (untracked).
- Existing archives: `YOLO11-EUVIP`, `YOLO12-EUVIP`, `YOLO26-EUVIP`,
  `YOLO26-DGX` (DGX machine runs), `RF-DETR-EUVIP`, and an `[OLD] YOLO26-Sweep`
  hyper-parameter sweep, all under `Results/MDWD-Runs/`.

### Split-leakage sensitivity analysis — [Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py](Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py)

Because MDWD splits the data *after* ~10x offline augmentation, the EDA flags
30 source photographs whose augmented copies land in more than one split.
This read-only tool independently re-derives that leakage (cross-checked
against the EDA's `integrity_issues.csv`), builds leakage-excluded valid/test
subsets without touching any dataset file, and re-evaluates the best
checkpoint per model family (YOLO11/12/26) on original vs. clean subsets to
measure the actual effect on reported metrics — result: negligible (max Δ
0.48pp mAP50-95). Full write-up:
[Documents/MDWDLeakageSensitivity.md](Documents/MDWDLeakageSensitivity.md);
outputs in `Documents/Final-Reports/MDWD-Leakage-Analysis/`.

```bash
python Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py --self-test   # logic self-tests
python Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py               # audit only
python Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py --run-inference --models best-per-family
```

## MTSD workstream

### Attribute classification — [Scripts/MTSD-Scripts/AttributeClassification/](Scripts/MTSD-Scripts/AttributeClassification/README.md)

Config-driven pipeline (single source of truth:
[config/default.yaml](Scripts/MTSD-Scripts/AttributeClassification/config/default.yaml);
all relative paths resolve against the repository root):

- `mtsd_attr/` package: manifest building (QA JSONs → padded sign crops →
  deterministic hash-based train/val/test splits), multi-head model, shared
  training loop (generic gradient accumulation), variant metadata/profiles
  (`variants.py`), evaluation and size-ablation reporting.
- Four backbone families — DINOv3 (ViT-B/L), V-JEPA 2.1 (ViT-B/L), ConvNeXt
  (Tiny legacy, Base/Large), LingBot-Vision (ViT-B/L) — as a 16-variant
  model-size ablation (`--profile size_ablation_all`) on top of the
  preserved historical six-variant round.
- Generic entry point `train_variant.py --variant <key>` (historical thin
  scripts `train_dinov3.py`, `train_vjepa.py`, ... kept as wrappers),
  orchestrated by `run_all.py` (`--profile` / `--plan` / `--list-variants` /
  `--list-profiles` / `--smoke-test`).
- Outputs under `outputs/` (crops, checkpoints, metrics, reports, logs,
  `experiment_log.jsonl`); `reset_outputs.py` archives/clears them safely.
- W&B project: `MSc-MTSD-Attributes` (disabled automatically for smoke tests).
- Detailed methodology in
  [Documentation.md](Scripts/MTSD-Scripts/AttributeClassification/Documentation.md).

### EDA / analysis — [Scripts/MTSD-Scripts/MTSD-Analysis/](Scripts/MTSD-Scripts/MTSD-Analysis/)

`mtsd_eda` package + `MTSD-EDA.ipynb` generate the image inventory, EXIF/GPS
profiling, annotation statistics and 21 figures under
`Documents/MTSD-EDA/`; `mtsd_mapper.py` builds the interactive
`MTSD_mapped.html` capture atlas. Atlas point popups show the image preview,
filename, group, GPS position, capture metadata when available, plus `Open image`
and `Copy path`; the density layer is non-interactive so points remain clickable
in `Both` mode. Open the HTML through Live Server from the repository root so
relative links to `Datasets/MTSD/...` resolve correctly. `check_gps_tags.py`,
`count_image_annotation_stats.py` and `generate_sample_annotations.py` are
standalone audit utilities.

### Supervised detection — [Scripts/MTSD-Scripts/MTSD-SupervisedDetection/](Scripts/MTSD-Scripts/MTSD-SupervisedDetection/README.md)

The canonical CLI/package implements Final-QA discovery, the shared deterministic
80/10/10 split, hashed augmented and unaugmented YOLO/COCO exports, strict dataset
validation, and a resumable 13-model YOLO11/12/26 + RF-DETR training matrix.
The explicit candidate lock currently resolves **3,098 images** into
**2,479 train / 310 valid / 309 test**. Training uses W&B project
`MSc-MTSD-SupervisedDetection`; no MTSD detection training has been executed.
The four notebooks under `MTSD-SupervisedNotebooks/` are now thin read-only front
ends to this package.

```bash
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --dry-run
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --prepare-only --dataset-variant both --final
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --validate-prepared --strict
```

The final command intentionally refuses until `config/qa_gate.yaml` is resolved:
the recorded audit has five invalid values and five duplicate candidates and
predates GRP-6. Thus GRP-6 is explicitly configured, not silently discovered,
but is not yet methodologically approved for final execution.

### Annotation & compliance tooling

- **Label Studio workflow** ([Scripts/MTSD-Scripts/LabelStudio/](Scripts/MTSD-Scripts/LabelStudio/),
  guide: [Documents/LabelStudioWorkflow.md](Documents/LabelStudioWorkflow.md)):
  launchers, import preparation, project setup, and the QA formatter that
  converts Label Studio exports into the Final-QA COCO format.
- **Annotation path migration**
  ([Scripts/MTSD-Scripts/update_annotation_paths.ps1](Scripts/MTSD-Scripts/README.md)):
  one-off, re-runnable migration of legacy `source_image` paths to the new
  layout (dry-run + timestamped backups). Already applied on 2026-07-04.
- **Annotation QA** ([Scripts/MTSD-Scripts/MTSD-AnnotationQA/](Scripts/MTSD-Scripts/MTSD-AnnotationQA/)):
  audit, visual review and guarded-apply workflow for duplicate boxes and
  attribute issues in the Final-QA JSONs.
- **GDPR redaction** ([Scripts/Other-Scripts/GDPR-Compliance/](Scripts/Other-Scripts/GDPR-Compliance/)):
  face/licence-plate detection and blurring previews before any imagery is
  shared; preview/apply paths are guarded so stale previews are not applied.
- **PromptDetect** ([Scripts/Other-Scripts/PromptDetect/](Scripts/Other-Scripts/PromptDetect/),
  docs: [Documents/PromptDetect.md](Documents/PromptDetect.md) and
  [Documents/PromptDetect-Dissertation-Protocol.md](Documents/PromptDetect-Dissertation-Protocol.md)):
  Gradio app for promptable detection with SAM 3/3.1, Cosmos Reason2 and
  LocateAnything, plus
  an exploratory class-agnostic evaluator and a fixed target-aware dissertation
  protocol with resumable per-model×prompt persistence.
- **Inference benchmark** ([Scripts/Other-Scripts/Inference-Benchmark/](Scripts/Other-Scripts/Inference-Benchmark/)):
  dry-run verified inference-speed benchmark covering supervised detectors,
  prompt models and attribute classifiers. Only Cosmos Reason2 32B requires
  `--allow-heavy`; 8B is allowed normally.

---

## Dissertation evidence tooling — [Scripts/FinalEvaluation/](Scripts/FinalEvaluation/README.md)

Read-only analysis and reporting layer over the completed experiments (nothing
here trains or silently runs inference):

- `final_dataset_integrity_check.py` — MDWD + MTSD integrity sign-off report.
- `mdwd_leakage_sensitivity.py` — the completed split-leakage sensitivity
  analysis (effect measured negligible; see
  [Documents/MDWDLeakageSensitivity.md](Documents/MDWDLeakageSensitivity.md)).
- `export_dissertation_tables.py` / `failure_case_sampler.py` — final tables
  and qualitative error samples.
- `robustness_slice_analysis.py` — performance by object size, brightness,
  blur, clutter, position, class, attribute head and prompt, computed from
  **stored** predictions only; low-support slices are marked, never highlighted.
- `annotation_effort_report.py` — dataset/annotation/QA/training/operational
  effort per paradigm; human hours come only from
  `config/annotation_effort_manual.yaml` (missing = *not recorded*, never
  estimated); no composite ranking is produced.
- `bootstrap_uncertainty.py` — seeded percentile-bootstrap confidence
  intervals and paired model/prompt comparisons over stored sample-level
  outcomes; pending experiments produce pending reports, not reruns.
- `build_dissertation_dashboard.py` — static, fully offline HTML evidence
  dashboard (research-question matrix, experiment status badges, tables,
  figures, robustness, uncertainty, effort, failure cases, limitations, and a
  machine-readable `data/evidence_index.json`); missing evidence renders as
  PENDING. Build with `--overwrite` for the stable
  `Documents/Final-Reports/Dissertation-Dashboard/latest/` path.

Outputs go to timestamped folders under `Documents/Final-Reports/`,
`Documents/Final-Figures/` and `Documents/Final-Tables/`; previous outputs are
never overwritten by default. Pilot, smoke-test, historical-snapshot and final
results stay explicitly labelled throughout.

---

## Setup

### UVB launcher

Use the root launcher to find the repository's safe audit tools, analysis
commands, Gradio applications and report folders without memorising paths:

```powershell
python launch_uvb.py
.\launch_uvb.ps1
python launch_uvb.py --list
python launch_uvb.py --doctor
python launch_uvb.py --dry-run
python launch_uvb.py promptdetect
```

The menu is grouped into local web applications, annotation/QA, dataset
analysis, evaluation, maintenance, and documentation. It detects the
repository root from the launcher location and uses `conda run` for the
documented `MDWD`, `mtsd-attrcls`, `mtsd-base`, and `mtsd-la` environments.
If Conda is missing, local web applications are refused with an actionable
warning; safe commands may use the current Python interpreter. Heavy model
tools require an explicit confirmation.

**Local web applications are managed processes.** The launcher starts each
app in its own process group, polls the local HTTP port until the server is
actually ready (configurable `--ready-timeout`; no fixed sleep), and opens
the browser only after a successful start — a failed or timed-out startup
opens no tab and prints the app's log tail. Closing a browser tab does NOT
stop the server (that cannot be detected reliably); instead the main menu
shows a *Running local web applications* entry from which each app can be
reopened in the browser, restarted, or stopped. Stopping, `Ctrl+C`, menu
quit and errors all terminate the app's complete launcher-owned process
tree (including `conda run` intermediaries and the LocateAnything worker).
If the same app is already running, the launcher offers reopen/restart/stop
instead of launching a duplicate; if an unrelated process owns the port, the
launcher reports it and either picks a verified free port (all managed apps
accept `--port`) or refuses — it never terminates a process it did not
start. Per-app startup logs live under `.uvb_launcher_logs/`; leftover PIDs
from a crashed session are re-verified against their recorded command line
before the launcher ever offers to stop them.

Navigate the interactive menu with **Up/Down** arrows and **Enter**. Use
**Esc** or **Backspace** to return, `/` to search by tool name or purpose, and
`q` to quit from the workspace menu. `j`/`k` also move the selection when
arrow keys are inconvenient.

The launcher never starts training, never auto-applies annotation fixes, never
deletes anything, and never opens or modifies `label_studio.sqlite3`. The Label
Studio menu item prints the documented workflow because the existing startup
script can stop a port-8080 process and writes local Label Studio state.

To add a tool, add one `Tool(...)` entry to `TOOLS` in `launch_uvb.py`: use a
repository-relative script path, working directory, environment, safe default
arguments, and — for a local web application — a port, a `--port`-style
`port_arg` and a `no_browser_arg` so the launcher can manage it. Keep
mutating or heavy commands behind `confirm=True` and prefer their documented
dry-run flags. Launcher lifecycle tests live in `tests/test_launch_uvb.py`.

### Environments (conda, recommended)

Per-workstream environments are defined in
[Requirements/CondaEnvironments/](Requirements/CondaEnvironments/)
with per-platform setup scripts:

| Env | Used by |
| --- | --- |
| `MDWD` | MDWD notebooks + MDWD EDA |
| `mtsd-attrcls` | AttributeClassification training/inference |
| `mtsd-base` | PromptDetect app/batch eval, GDPR, annotation QA review app, general MTSD tooling |
| `mtsd-la` | LocateAnything worker |

### Requirements files

[Requirements/](Requirements/) holds the pip layer per workstream
(`requirements-attribute-classification.txt`,
`requirements-traditional-training.txt`, `requirements-promptdetect.txt`,
`requirements-locate-anything.txt`, `requirements-general-tooling.txt`).

### Secrets / W&B

Create a `.env` at the repository root (git-ignored):

```
WANDB_MODE=online
WANDB_API_KEY=<your key>
```

Notebooks and training scripts load it automatically; without a key, W&B
falls back to an existing login, and the attribute pipeline can run fully
offline (`wandb.enabled: false` in its YAML).

### Dataset paths

All code resolves data relative to the repository root — no absolute paths.
Expected locations: `Datasets/MDWD/<variant>/` and
`Datasets/MTSD/GRP-*/Images/` + `Datasets/MTSD/Annotations/`. Cloning the
repo without the (untracked) images gives you all code, annotations and the
numeric results; the imagery must be restored from local/off-repo storage.

---

## Reproducibility notes

- Seeds are fixed in the configs (`seed: 42` for attribute classification;
  deterministic hash-salted splits that must never be re-salted mid-project).
- MDWD run names encode their full configuration
  (`E001_yolo26n_rfv20_img640_b32_e100_adamw_lr0p001`), and each run folder
  keeps `args.yaml` / run-config JSON alongside the metrics.
- Existing run directories are never overwritten by the notebooks; delete or
  rename manually if a re-run is truly intended.
- W&B mirrors runs, configs and artifacts for both tracks
  (`MSc-MDWD-EUVIP26`, `MSc-MTSD-Attributes`).
- Git tracks code, configs, annotations and numeric results; weights, raw
  imagery, W&B folders and bulk crops are local-only by design.

## Status

**Under active MSc research development** (2026). Current state: MDWD
detection benchmarks (YOLO11/12/26 plus RF-DETR nano) executed with archived runs and
a completed split-leakage sensitivity check (negligible effect, see
[Documents/MDWDLeakageSensitivity.md](Documents/MDWDLeakageSensitivity.md));
MTSD groups 1/2/3/5/6 have Final-QA files, with the candidate final lock awaiting
human QA approval; EDA and a six-variant
attribute-classification round executed on the historical GRP-1–GRP-3
snapshot; the canonical MTSD supervised pipeline and fixed PromptDetect
dissertation protocol are implemented and dry-run verified but have not been
executed for final evidence. A five-image exploratory MDWD PromptDetect pilot
exists (SAM 3 + Cosmos Reason2 2B; not dissertation evidence). Annotation-QA
review remains the gate before final MTSD preparation and evaluation.

## Citation

Dissertation citation to be added on submission:

> A. F. Lucas, *[dissertation title TBC]*, MSc dissertation, University of
> Malta, 2026. Repository: `AFLucas-UOM/Urban-Vision-Benchmark` (private).
