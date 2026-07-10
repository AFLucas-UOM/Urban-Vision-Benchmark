# Annotation and operational effort report

Generated: 2026-07-10T12:16:41  |  Commit: `15853d305771`

Automatically derived values come from dataset folders, QA JSONs, audit reports, run summaries and manifests (every row in `annotation_effort_summary.csv` records its source file). Manual values come only from `Scripts/FinalEvaluation/config/annotation_effort_manual.yaml`; missing ones are *not recorded*, never estimated (see `missing_manual_values.md`).

## Dataset and annotation scale

| Dataset | Raw images | Annotated | Unannotated | Boxes | Classes | Aux. attributes | Disk (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MDWD (10x-augmented export) | 30225 | 30225 | 0 | 211512 | 5 | 0 | 5.52 |
| MTSD | 4683 | 2628 | 2055 | 7266 | 12 | 4 | 20.99 |

- MTSD QA-approved groups: GRP-1, GRP-2, GRP-3, GRP-5; excluded (unannotated) groups: GRP-4, GRP-6, GRP-7.
- MTSD attribute-labelled boxes: 7266.
- Latest annotation QA audit (audit-20260709-201340): missing_attribute_findings = 0, invalid_attribute_findings = 5, duplicate_pair_findings = 5, reference_problem_findings = 0
- Confirmed applied corrections: 3.
- MDWD integrity findings: {'out_of_range_box': 15, 'empty_annotation': 5, 'source_in_multiple_splits': 30} (leakage measured negligible - see MDWD-Leakage-Analysis).

## Training effort (completed experiments)

- MDWD detection: 15 completed runs (+1 historical sweep runs, 1 RF-DETR run(s) - RF-DETR nano has a checkpoint/log but no normalised summary - excluded from headline tables). Hardware: RTX 4090 (EUVIP suites) / DGX (YOLO26-DGX suite).
- MTSD attributes: 6 completed variants (+6 smoke tests), historical GRP-1..GRP-3 snapshot.
- MTSD detection: pending (no training runs).

## Prompt-based evaluation effort

- Batch-evaluation runs so far: 2 (all pilot scale), covering 15 images and 7 prompt evaluations.

| Model | Confidence outputs | Gated weights | Separate env | Prompt engineering | Retraining |
| --- | --- | --- | --- | --- | --- |
| SAM 3 | yes | yes | no | text prompt per concept | none |
| SAM 3.1 | yes | yes | no | text prompt per concept | none |
| Cosmos Reason2 2B | no | no | no | grounding prompt (JSON bbox reply) | none |
| Cosmos Reason2 8B | no | no | no | grounding prompt (JSON bbox reply) | none |
| Cosmos Reason2 32B | no | no | no | grounding prompt; opt-in via --allow-heavy | none |
| LocateAnything 3B | no | no | yes | text prompt; runs in mtsd-la worker env | none |

## Paradigm-level operational comparison

No composite score is computed: the dimensions are not commensurable and collapsing them would be arbitrary. Raw dimensions only:

| Paradigm | Ground-truth requirement | Human annotation burden | Training requirement | Prompt engineering | Model size | Latency | GPU memory | Maintenance burden | Main limitation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Supervised detection (MDWD, YOLO) | 211512 boxes over 30225 images (10x augmented export) | annotation_hours: not recorded; qa_hours: not recorded | 15 completed runs, 100 epochs each (per-run wall time in details JSON) | none | up to 50.54 MB (yolo26l) | ~16.867737590633762 ms/img (val-time measurement) | not recorded | retrain to add classes; dataset/label upkeep | needs full box annotation; augmented-export leakage audited (negligible effect) |
| Supervised detection (MTSD) | 7266 QA boxes over 2628 images (4 QA groups) | annotation_hours: not recorded; qa_hours: not recorded | pending (no MTSD detection training yet) | none | pending | pending | pending | as MDWD, plus ongoing group annotation | 2055 images in 3 groups still unannotated |
| Attribute classification (MTSD crops) | 7266 attribute-labelled boxes (4 attribute heads) | attribute labels captured during box QA; extra hours: not recorded | 6 completed variants (+6 smoke) - LoRA trains ~304909 params | none | 27M-305M backbone params (frozen for probe/LoRA variants) | not recorded | not recorded | manifest regeneration whenever QA annotations change | historical GRP-1..3 snapshot; weak `condition` head (class imbalance) |
| Prompt-based localisation (PromptDetect) | none for inference (GT only used for scoring) | prompt_design_hours: not recorded; manual_review_hours: not recorded | none (zero-shot) | 7 prompts evaluated so far; per-model prompt phrasing matters | 2B-32B VLM / SAM 3-scale weights | hundreds of ms/img (pilot measurements in predictions.csv) | not recorded | gated weight access (SAM 3/3.1), separate env for LocateAnything, prompt upkeep | only pilot-scale evaluations exist; Cosmos/LocateAnything provide no per-box confidence (AP not comparable) |

## Inference-speed benchmark status

- no inference-speed benchmark runs yet - run Scripts/Other-Scripts/Inference-Benchmark/inference_speed_benchmark.py; MDWD detection latency below comes from the training pipelines' own val timing instead

## Figures

- `Documents/Final-Figures/Annotation-Effort/20260710-121641/annotation_scale_per_dataset.png`
- `Documents/Final-Figures/Annotation-Effort/20260710-121641/mdwd_training_time_per_run.png`
- `Documents/Final-Figures/Annotation-Effort/20260710-121641/mdwd_latency_vs_model_size.png`
- `Documents/Final-Figures/Annotation-Effort/20260710-121641/attr_trainable_params_by_adaptation.png`
- `Documents/Final-Figures/Annotation-Effort/20260710-121641/evidence_status_by_paradigm.png`

*Read-only report - nothing outside the output folders was written; no value was estimated.*
