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
captures under `Images/`. QA-verified annotations exist for **GRP-1–GRP-3**
under [Datasets/MTSD/Annotations/](Datasets/MTSD/Annotations/):
COCO-style JSONs (`GRP-*/Final-QA/QA-GRP*.json`, 1,971 annotated images) whose
boxes carry **per-sign auxiliary attributes** — viewing angle, mounting type,
condition, and sign shape — used for the multi-attribute classification
experiments. Remaining groups join automatically once their `Final-QA` JSON
exists.

> Known data gap: GRP-3's `Images/` folder currently contains 398 files while
> its QA JSON annotates 617 images (219 annotated images not on disk).

---

## The two experimental tracks

1. **Domestic waste detection (MDWD).** Supervised object-detection
   benchmarks across model families and sizes: **YOLO11, YOLO12, YOLO26**
   (n/s/m/l) and **RF-DETR** (nano/small/medium), run from the notebooks in
   [Scripts/MDWD-Scripts/MDWD-SupervisedNotebooks/](Scripts/MDWD-Scripts/MDWD-SupervisedNotebooks/),
   with runs archived under [Results/MDWD-Runs/](Results/MDWD-Runs/) and
   consolidated metrics under [Results/MDWD-Results/](Results/MDWD-Results/).

2. **Traffic-sign multi-attribute classification (MTSD).** Representation
   learning on QA-approved sign crops with four heads (view angle, mounting,
   condition, shape) across backbone/adaptation variants — **DINOv3**
   (frozen/LoRA), **V-JEPA 2.1** (frozen/LoRA) and **ConvNeXt**
   (frozen/fine-tuned) — in
   [Scripts/MTSD-Scripts/AttributeClassification/](Scripts/MTSD-Scripts/AttributeClassification/).
   Supporting tooling covers annotation (Label Studio), GDPR redaction, and
   promptable-detection experiments.

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
│   └── *.md                       #   LabelStudio, QA formatter, PromptDetect, etc.
├── Models/                        # Stock pretrained weights (YOLO11/12/26, RF-DETR)
├── Requirements/                  # pip requirements per workstream
├── Results/
│   ├── MDWD-Results/              # Consolidated benchmark tables/plots per model family
│   ├── MDWD-Runs/                 # Archived training runs (weights, curves, configs)
│   ├── MTSD-Results/              # (reserved for MTSD result exports)
│   └── MTSD-Runs/                 # (reserved for MTSD run archives)
└── Scripts/
    ├── MDWD-Scripts/
    │   ├── MDWD-Analysis/         # MDWD EDA package + runner (see its README)
    │   └── MDWD-SupervisedNotebooks/  # YOLO26 / YOLO12 / RF-DETR training notebooks
    ├── MTSD-Scripts/              # (see its README)
    │   ├── AttributeClassification/   # Multi-attribute training pipeline + configs
    │   ├── LabelStudio/           # Annotation-workflow tooling
    │   ├── MTSD-Analysis/         # MTSD EDA package + notebook + audit scripts
    │   └── update_annotation_paths.ps1
    └── Other-Scripts/
        ├── CondaEnvironments/     # Reproducible conda env YAMLs + setup scripts
        ├── GDPR-Compliance/       # Face/plate detection + redaction pipeline
        └── PromptDetect/          # SAM 3 / Cosmos Reason2 / LocateAnything app
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

## MTSD workstream

### Attribute classification — [Scripts/MTSD-Scripts/AttributeClassification/](Scripts/MTSD-Scripts/AttributeClassification/README.md)

Config-driven pipeline (single source of truth:
[config/default.yaml](Scripts/MTSD-Scripts/AttributeClassification/config/default.yaml);
all relative paths resolve against the repository root):

- `mtsd_attr/` package: manifest building (QA JSONs → padded sign crops →
  deterministic hash-based train/val/test splits), multi-head model, shared
  training loop, evaluation.
- `train_dinov3.py` / `train_dinov3_lora.py` / `train_vjepa.py` /
  `train_vjepa_lora.py` / `train_convnext.py` / `train_convnext_finetuned.py`,
  orchestrated by `run_all.py` (`--smoke-test` supported).
- Outputs under `outputs/` (crops, checkpoints, metrics, reports, logs,
  `experiment_log.jsonl`); `reset_outputs.py` archives/clears them safely.
- W&B project: `MSc-MTSD-Attributes` (disabled automatically for smoke tests).
- Detailed methodology in
  [Documentation.md](Scripts/MTSD-Scripts/AttributeClassification/Documentation.md).

### EDA / analysis — [Scripts/MTSD-Scripts/MTSD-Analysis/](Scripts/MTSD-Scripts/MTSD-Analysis/)

`mtsd_eda` package + `MTSD-EDA.ipynb` generate the image inventory, EXIF/GPS
profiling, annotation statistics and 21 figures under
`Documents/MTSD-EDA/`; `mtsd_mapper.py` builds the interactive
`MTSD_mapped.html` capture map; `check_gps_tags.py`,
`count_image_annotation_stats.py` and `generate_sample_annotations.py` are
standalone audit utilities.

### Annotation & compliance tooling

- **Label Studio workflow** ([Scripts/MTSD-Scripts/LabelStudio/](Scripts/MTSD-Scripts/LabelStudio/),
  guide: [Documents/LabelStudioWorkflow.md](Documents/LabelStudioWorkflow.md)):
  launchers, import preparation, project setup, and the QA formatter that
  converts Label Studio exports into the Final-QA COCO format.
- **Annotation path migration**
  ([Scripts/MTSD-Scripts/update_annotation_paths.ps1](Scripts/MTSD-Scripts/README.md)):
  one-off, re-runnable migration of legacy `source_image` paths to the new
  layout (dry-run + timestamped backups). Already applied on 2026-07-04.
- **GDPR redaction** ([Scripts/Other-Scripts/GDPR-Compliance/](Scripts/Other-Scripts/GDPR-Compliance/)):
  face/licence-plate detection and blurring previews before any imagery is
  shared.
- **PromptDetect** ([Scripts/Other-Scripts/PromptDetect/](Scripts/Other-Scripts/PromptDetect/),
  doc: [Documents/PromptDetect.md](Documents/PromptDetect.md)): Gradio app for
  promptable detection with SAM 3/3.1, Cosmos Reason2 and LocateAnything.

---

## Setup

### Environments (conda, recommended)

Per-workstream environments are defined in
[Requirements/CondaEnvironments/](Requirements/CondaEnvironments/README.md)
with per-platform setup scripts:

| Env | Used by |
| --- | --- |
| `MDWD` | MDWD notebooks + MDWD EDA |
| `mtsd-attrcls` | AttributeClassification training/inference |
| `mtsd-base` | PromptDetect app, GDPR `sam31` backend, general MTSD tooling |
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
detection benchmarks (YOLO11/12/26, RF-DETR) executed with archived runs;
MTSD groups 1–3 QA-annotated with EDA and attribute-classification pipeline
operational (full training round in progress); MTSD detection-track results
folders reserved; further MTSD group annotation ongoing.

## Citation

Dissertation citation to be added on submission:

> A. F. Lucas, *[dissertation title TBC]*, MSc dissertation, University of
> Malta, 2026. Repository: `AFLucas-UOM/Urban-Vision-Benchmark` (private).
