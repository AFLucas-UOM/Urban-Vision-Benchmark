# Urban Vision Benchmark

**Benchmarking supervised, prompt-based and representation-learning approaches to street-level urban monitoring in Malta.**

This repository is the reproducible research artefact for an MSc dissertation at the [University of Malta](https://www.um.edu.mt/). It introduces two locally collected datasets and evaluates three computer-vision paradigms for domestic-waste monitoring and traffic-sign assessment.

## Repository status

This is a **private** dissertation repository being prepared for public release. The root README is the maintained project entry point; timestamped tables and reports under `Documents/Final-Tables/` and `Documents/Final-Reports/` are the supporting evidence. Local datasets, downloaded weights, transient working directories and superseded experiment artefacts are intentionally excluded from version control.

## At a glance

| Component | Scope |
| --- | --- |
| **MDWD** | 3,598 source photographs of domestic waste; five operational waste classes |
| **MTSD** | 7,482 street-level photographs; 20,509 traffic-sign boxes; 12 sign classes and four maintenance-relevant attributes |
| **Supervised detection** | YOLO11, YOLO12, YOLO26 and RF-DETR |
| **Zero-shot localisation** | SAM 3, SAM 3.1, LocateAnything-3B and Cosmos Reason2 2B/8B |
| **Attribute classification** | DINOv3, V-JEPA 2.1, ConvNeXt and LingBot-Vision under frozen, LoRA and full fine-tuning regimes |

The central question is not only which model scores highest, but which paradigm provides the best balance of accuracy, adaptation effort, prompt robustness and operational usefulness for local-authority workflows.

## Results summary

| Experiment | Best recorded result | Main observation |
| --- | --- | --- |
| **MDWD supervised detection** | **YOLO26-L:** 0.782 test mAP50–95 | YOLO26-L was the strongest stored run; smaller YOLO variants offered competitive accuracy at lower cost. |
| **MTSD supervised detection** | **YOLO26-M:** 0.738 test mAP50–95 in the high-resolution follow-up (11-class evaluation) | RF-DETR-M led the original common 13-model evaluation at 0.686; later controlled runs showed that input resolution was a major performance driver. |
| **MTSD attribute classification** | **V-JEPA 2.1 ViT-L + LoRA:** 0.901 test mean macro-F1, 95% CI [0.891, 0.910] | LoRA consistently improved frozen representations while updating less than 0.3% of the winning model's parameters. Condition remained the hardest attribute (0.747 macro-F1). |
| **MDWD prompt localisation** | **SAM 3.1:** 0.526 macro mean F1 across four prompt families | Strongest zero-shot result, but wording sensitivity remained substantial (49.0% mean relative best-to-worst degradation). |
| **MTSD prompt localisation** | **Cosmos Reason2 8B:** 0.305 macro mean F1 across five prompt families | The strongest prompt model was also the least sensitive of those tested on MTSD, but prompt-based localisation remained less reliable for this specialised task. |

These metrics come from task-specific protocols and should not be compared as if they were interchangeable: detection uses mAP, attribute classification uses crop-level macro-F1, and prompt localisation reports targeted F1 over controlled paraphrase families. The complete evidence is available in the [MDWD detection table](Documents/Final-Tables/20260712-213548/mdwd_detection_results.csv), [MTSD detection report](Documents/Final-Reports/MTSD-SupervisedDetection/mtsd_main_scale_comparison.md), [attribute ablation report](Scripts/MTSD-Scripts/AttributeClassification/outputs/reports/size_ablation.md), prompt-sensitivity reports for [MDWD](Results/PromptDetect/BatchEvaluation/MDWD/20260821-170445-optimized-v2-bounded-sensitivity/prompt_sensitivity_report.md) and [MTSD](Results/PromptDetect/BatchEvaluation/MTSD/20260822-102455-optimized-v2-bounded-sensitivity/prompt_sensitivity_report.md), and the [completed cross-suite inference benchmark](Documents/Final-Tables/20260910-inference-benchmark/all_model_inference_benchmark.md).

Additional findings include:

- Increasing MTSD input resolution produced larger and more consistent gains than the original offline-augmentation ablation.
- Prompt wording can change both accuracy and the set of objects returned, making prompt selection an operational variable rather than a cosmetic one.
- For MTSD attributes, mounting and sign shape were comparatively easy; visual condition was consistently the limiting head.

## Datasets

### Maltese Domestic Waste Dataset (MDWD)

MDWD contains street-level photographs of waste presented for kerbside collection. Its five classes align with Malta's collection streams: `Mixed Waste`, `Organic Waste`, `Recyclable Material`, `Orange CMD` and `Other Waste`.

| Property | Value |
| --- | --- |
| Unique source images | 3,598 |
| Working export | Roboflow v20, 640 × 640 |
| Exported splits | 29,487 augmented train / 369 validation / 369 test images |
| Formats | YOLO and COCO, derived from the same annotation state |

### Maltese Traffic Sign Dataset (MTSD)

MTSD contains smartphone imagery collected across Malta between November 2025 and January 2026. Every retained sign is localised and labelled for viewing angle, mounting, physical condition and shape.

| Property | Value |
| --- | --- |
| Source images | 7,482 |
| Annotated instances | 20,509 |
| Detection classes | 12 |
| Attributes | view angle, mounting, condition, sign shape |
| Canonical split | 5,986 train / 749 validation / 747 test source images |
| Attribute study | 19,253 retained crops across all 11 QA-approved groups |

The canonical annotations are the QA-approved COCO files under `Datasets/MTSD/Annotations/GRP-*/Final-QA/`. Prepared datasets use deterministic source-level splits, preventing crops or augmented copies of one photograph from crossing partitions.

## Reproducibility

- Seed **42** is used throughout; deterministic execution is enabled where supported.
- Dataset splits are assigned at source-image level using fixed salted hashes.
- Prepared datasets and annotation exports carry SHA-256 manifests.
- Run directories are immutable and store their effective configuration, environment and evaluation artefacts.
- Dataset preparation is blocked unless the corresponding MTSD annotation QA gate has passed.
- Final evaluation tools are read-only over stored predictions and write to new timestamped directories.
- Raw imagery, downloaded weights, local scratch space and superseded smoke or trial outputs are excluded from the release snapshot.

The repository preserves code, protocols, configurations, QA records, aggregate results and dissertation-ready exports. Weights & Biases is used as a monitoring mirror; repository artefacts are the authoritative evidence.

## Getting started

Clone the repository and create the environment required by the workflow you want to reproduce. Environment definitions and platform-specific setup scripts are in [Requirements/CondaEnvironments](Requirements/CondaEnvironments/README.md).

| Environment | Primary use |
| --- | --- |
| `MDWD` | YOLO/RF-DETR training and MDWD analysis |
| `mtsd-attrcls` | Multi-head attribute classification |
| `mtsd-base` | MTSD detection, SAM/Cosmos evaluation, QA and privacy tools |
| `mtsd-la` | Isolated LocateAnything worker |

Optional W&B logging is configured through a git-ignored root `.env`:

```dotenv
WANDB_MODE=online
WANDB_API_KEY=<your-key>
WANDB_ENTITY=<your-entity>
```

For an interactive entry point:

```bash
python launch_uvb.py
```

The launcher selects the appropriate conda environment, manages local web applications and logs, and requires confirmation before heavy or mutating actions.

### Representative workflows

```bash
# Check repository structure, Python sources, notebooks, links and secrets
python Scripts/Automation/verify_repository_health.py

# Preview the MTSD supervised-detection matrix
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --dry-run

# Preview the complete attribute-classification ablation
python Scripts/MTSD-Scripts/AttributeClassification/run_all.py --plan --profile size_ablation_all

# Preview the fixed prompt-evaluation protocol on both datasets
python Scripts/Other-Scripts/PromptDetect/batch_evaluation/run_dissertation_protocol.py \
  --dataset both --split test --dry-run

# Inspect the dissertation evidence tools
python Scripts/FinalEvaluation/export_dissertation_tables.py --dry-run
```

Training and full inference require authorised access to the private imagery and locally downloaded pretrained weights. See the component READMEs for exact commands and hardware-specific settings.

## Repository guide

```text
Urban-Vision-Benchmark/
├── Datasets/       MDWD exports, MTSD source groups, annotations and prepared sets
├── Documents/      methodology notes, EDA, audits, final figures, tables and reports
├── Models/         local pretrained checkpoints (not distributed)
├── Requirements/   pip requirements and conda environment definitions
├── Results/        immutable training, evaluation and prediction artefacts
├── Scripts/
│   ├── Automation/       workflow runner and repository health checks
│   ├── FinalEvaluation/  integrity, uncertainty, robustness and export tools
│   ├── MDWD-Scripts/     waste-dataset analysis and supervised training
│   ├── MTSD-Scripts/     annotation QA, detection and attribute classification
│   └── Other-Scripts/    PromptDetect, GDPR redaction and inference benchmarks
└── launch_uvb.py   managed entry point for interactive tools
```

Useful starting points:

- [Final-evaluation documentation](Scripts/FinalEvaluation/README.md)
- [MTSD supervised-detection documentation](Scripts/MTSD-Scripts/MTSD-SupervisedDetection/README.md)
- [MTSD attribute-classification documentation](Scripts/MTSD-Scripts/AttributeClassification/README.md)
- [PromptDetect batch-evaluation protocol](Scripts/Other-Scripts/PromptDetect/batch_evaluation/README.md)
- [Completed inference benchmark](Documents/Final-Tables/20260910-inference-benchmark/all_model_inference_benchmark.md)
- [Dataset-integrity report](Documents/Final-Reports/dataset_integrity_report.md)
- [Automation and workflow registry](Scripts/Automation/README.md)

## Data availability, ethics and privacy

Raw street-level images are **not publicly distributed** because they may contain faces, vehicle registration plates, location metadata and other personal data. Access is restricted and governed by the dissertation's GDPR-aware handling procedure. Any imagery selected for publication must pass the repository's face, plate and QR-code redaction workflow, including manual preview and verification.

The public research artefact is therefore intended to provide the implementation, annotation schema, configurations, provenance records and numeric evidence without exposing the underlying personal data. GPS information is used only in aggregate reporting.

## Citation

If you use this repository or its methodology, please cite:

> A. F. Lucas, *A Comparative Study of Vision-Based Perception Paradigms for Urban Waste and Infrastructure Monitoring*. MSc dissertation, University of Malta, 2026. Repository: `AFLucas-UOM/Urban-Vision-Benchmark`.

No public licence is currently declared; contact the author before redistributing code, annotations or derived artefacts.
