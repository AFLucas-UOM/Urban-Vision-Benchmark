# Robustness comparison across the MDWD and MTSD tracks

Generated 2026-09-14 from completed stored-result analyses. This is a comparison and writing guide, not a new model evaluation: no detector, prompt backend, or attribute classifier was run while producing it.

## Executive summary

Three distinct kinds of robustness evidence are available and should be reported as three tracks, **not collapsed into one score**.

| Track | MDWD | MTSD | What changes under the robustness test? | Primary comparison measure |
|---|---|---|---|---|
| Supervised object detection | Final stored-prediction slice audit | Completed stored-unified-prediction inventory | Image/object characteristics: brightness, contrast, sharpness, density, size, position, class frequency, and class | Aggregate and image-slice F1; object-slice recall |
| Zero-shot PromptDetect | Final bounded prompt-sensitivity evaluation | Final bounded prompt-sensitivity evaluation | Only wording within a fixed target class/family | Mean, worst-case, and range of F1 across four formulations; raw-output consistency |
| MTSD attribute classification | Not applicable: MDWD has no corresponding sign-attribute task | Historical GRP-1--GRP-3 snapshot | Attribute head and true attribute class, using ground-truth sign crops | Per-head macro-F1 and per-class F1 |

The strongest supervised-detector findings are consistent across the two datasets: small objects, peripheral objects, and high-clutter images are the main observed weak conditions. On MDWD, YOLO26-L reaches 90.94% aggregate F1 but only 41.75% recall for small objects; on the selected highest-scoring MTSD test snapshot, YOLO26-M at 1280 reaches 82.28% aggregate F1 but 53.37% small-object recall. These values are **within-dataset** results, not evidence that one dataset is intrinsically easier.

The zero-shot track answers a different question: whether a model's output is stable when a semantically equivalent request is rephrased. On MDWD, SAM 3.1 has the highest macro mean F1 (52.62%) but a 49.03% mean relative degradation from each family’s best to worst formulation. On MTSD, Cosmos Reason2 8B has the highest macro mean F1 (30.54%) and a smaller 30.59% mean relative degradation; SAM 3/3.1 achieve higher scores for some individual MTSD formulations but are markedly prompt-sensitive (about 61% mean relative degradation).

For MTSD attributes, VJEPA 2.1-L with LoRA is the strongest retained 1,890-crop configuration (mean of the four head macro-F1 values: 90.06%). Condition is the weak head (74.70% macro-F1), especially Weathered (67.00% F1) and Heavily Damaged (65.49% F1), while viewing angle, mounting, and sign shape are substantially stronger.

## What may and may not be compared

### Valid comparisons

- Compare a detector’s aggregate performance with its own data slices, using the same split, model run, confidence threshold, matcher, and taxonomy.
- Compare detector configurations only within the same dataset and matching split/protocol. For MTSD, retain the run identifier and resolution because the report is a completed-run inventory, not a statistically controlled model-selection study.
- Compare prompt formulations only within the same sensitivity family: every formulation in a family has identical target ground truth, and wording is the intended intervention.
- Compare PromptDetect models within a dataset using the macro mean and worst-prompt F1 across the fixed family set.
- Compare MTSD attribute representations on the 1,890-crop GRP-1--GRP-3 snapshot, using the mean of the four stated head macro-F1 values. Keep the older 503-crop configurations separate.

### Comparisons to avoid

- Do not average supervised-detector F1, zero-shot F1, and attribute macro-F1 into a universal “robustness score.” They use different tasks, target populations, and error definitions.
- Do not interpret the MDWD and MTSD detector F1 values as a direct dataset-difficulty contest. They differ in taxonomy, image corpus, annotation scope, object scale distribution, and experiment inventory.
- Do not turn object-level recall slices into F1. False positives cannot be uniquely attributed to one object-size or object-position group, so recall is the defensible object-level measure.
- Do not interpret observational brightness/contrast/sharpness slices as controlled corruption experiments. The slices are correlated with scene content, object composition, and acquisition conditions.
- Do not treat a high matched-box IoU as high semantic correctness. In PromptDetect, a detected non-target sign can be well-localised yet be a false positive for the queried class.
- Do not present MTSD attribute results as detector-plus-classifier pipeline performance: attribute evaluation uses **ground-truth crops**, so detector crop errors are outside this experiment.
- Do not claim confidence intervals or statistical significance. Both slice reports used `bootstrap_samples=0`; no interval estimates were produced.

## Common evaluation definitions

### Supervised detection

Stored predictions are matched to ground truth with class-aware greedy matching at IoU >= 0.50. The working detection confidence is 0.25. Aggregate and image-level groups report precision, recall, F1, and mean matched IoU. Object-level groups report recall and mean matched IoU only.

`precision = TP / (TP + FP)`, `recall = TP / (TP + FN)`, and `F1 = 2PR / (P + R)`. A slice requires support >= 15. Image-slice support is image count; object-slice support is object count. Rows below that threshold remain in machine-readable output but are marked `insufficient_support` and should not be highlighted.

### Image and object slicing rule

| Dimension | Grouping rule | Interpretation boundary |
|---|---|---|
| Brightness / contrast / sharpness | Empirical test-split terciles, computed separately for the relevant dataset/split/run annotation source | Observational image properties; sharpness is a blur proxy, not synthetic blur |
| Object density | 1 object = `single-object`; 2--4 = `low-clutter`; >=5 = `high-clutter` | An image-level group, so F1 is meaningful |
| Object size | COCO thresholds from box area: `<32^2` small, `<96^2` medium, otherwise large | MDWD areas are in resized 640x640 evaluation pixels; MTSD uses native stored-annotation coordinates |
| Object position | `central` if the box centre lies in the middle half of both image axes; otherwise `near-edge` | Object-level recall only |
| Class frequency | Class support >= median test-class support = `common`; otherwise `rare` | Composition-dependent descriptive group, not a causal rarity effect |
| Class | Ground-truth semantic class | A semantic diagnostic, with support shown |

MDWD test tercile thresholds are brightness 136.27/149.63, contrast 41.85/51.77, and sharpness 829.95/1620.70. The selected MTSD test run uses brightness 115.38/129.07, contrast 50.29/59.31, and sharpness 100.48/292.85. Thresholds are dataset/run-specific and must not be reused across datasets.

### Zero-shot PromptDetect sensitivity

Both datasets use the `prompt-sensitivity-v2` protocol, a confidence threshold of 0.30, class-aware IoU matching at 0.50, NMS IoU 0.50, and a test-set maximum-detections policy. Each target family contains four formulations: canonical, lexical paraphrase, descriptive paraphrase, and instruction. Every family is run on the entire respective test split, including target-negative images; any detection on a target-negative image is a false positive.

For a family, `relative F1 degradation = (max F1 - min F1) / max F1`. It measures worst-case sensitivity to formulation. `pairwise box F1` measures agreement between raw prediction sets of two prompt variants after greedy IoU matching; it is an output-stability measure and does **not** use ground truth. A pair that yields no boxes for both prompts on an image is treated as fully consistent for that image.

Several backends emit constant confidence. Their AP values collapse to one operating point, so AP is stored but should not be used as the sole cross-model ranking metric.

### MTSD attributes

The retained attribute analysis consumes aggregate test metrics and confusion matrices, not per-crop prediction files. It reports accuracy and macro-F1 by head, then class F1 within each head. The four heads are viewing angle, mounting, condition, and sign shape. Since only 1,890-crop aggregate/confusion-matrix outputs are retained, it cannot produce reliable attribute slices by object size, brightness, blur, or capture group.

## Coverage and provenance matrix

| Evidence item | Dataset / split | Status | Population | Main source |
|---|---|---|---|---|
| Supervised detection slices | MDWD test and valid | Final | Test: 369 images / 1,106 objects; valid: 369 / 1,072 | `Documents/Final-Reports/Robustness-Slices/20260908-103240/robustness_slice_results.csv` |
| Supervised detection slices | MTSD test and valid | Historical completed-run inventory | Test: 747 images / 1,975 non-excluded objects; valid: 749 / 1,993 | `Documents/Final-Reports/Robustness-Slices/20260914-123609/robustness_slice_results.csv` |
| Prompt sensitivity | MDWD test | Final bounded evaluation | 369 images; 4 target families; 4 formulations/family; 5 models | `Results/PromptDetect/BatchEvaluation/MDWD/20260821-170445-optimized-v2-bounded-sensitivity/` |
| Prompt sensitivity | MTSD test | Final bounded evaluation | 747 images; 5 target families; 4 formulations/family; 5 models | `Results/PromptDetect/BatchEvaluation/MTSD/20260822-102455-optimized-v2-bounded-sensitivity/` |
| Attribute classification | MTSD test | Historical GRP-1--GRP-3 snapshot | 1,890 ground-truth sign crops for current configurations | `Documents/Final-Reports/Robustness-Slices/20260914-123609/robustness_slice_results.csv` |

The MTSD supervised-detection inventory retained 47 completed exports from 13 model labels across test and valid. Five exports were skipped because they lacked either a completed run record or compatible COCO ground truth. The inventory uses a default exclusion of `Tourist Sign`; do not mix it with a protocol that scores that class or handles it as an ignore region.

## Track 1 — supervised object detection

### Headline detector comparison

This table names one clearly defined reference configuration per dataset to make the slice comparison readable. It is not a claim that these two models are directly ranked against one another across datasets.

| Dataset | Reference configuration | Split | Precision | Recall | F1 | Mean matched IoU | Images | Objects | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| MDWD | YOLO26-L, `YOLO26-EUVIP` | test | 94.54% | 87.61% | 90.94% | 92.15% | 369 | 1,106 | Final |
| MTSD | YOLO26-M, strong augmentation, 1280, `strongaug-wandb-followup-yolo26m-img1280-s42` | test | 82.85% | 81.72% | 82.28% | 93.37% | 747 | 1,975 | Historical completed-run inventory |

For MDWD, the other stored reference test runs are YOLO26-L `YOLO26-DGX` (F1 90.85%), YOLO12-M (88.88%), and YOLO11-L (87.84%). For MTSD, the leading stored test entries are YOLO26-S 1280 (81.72%), YOLO11-M 1280 (80.90%), YOLO26-S 960 (79.28%), YOLO11-S 1280 (78.42%), and RF-DETR-M 704 (77.46%). The full run-level inventory, including taxonomy/protocol caveats, is in the linked MTSD CSV.

### Data-slice results for the two reference configurations

| Dimension | MDWD YOLO26-L | MTSD YOLO26-M 1280 | Cross-track reading |
|---|---|---|---|
| Brightness | F1: low 90.53%, medium 90.86%, high 91.54% | F1: low 81.52%, medium 83.41%, high 81.99% | Limited separation in both references; do not infer an illumination causal effect. |
| Contrast | F1: low 94.72%, medium 89.99%, high 88.89% | F1: low 84.06%, medium 80.00%, high 83.07% | The worst tercile differs by dataset; it is an observational association. |
| Sharpness | F1: blurred 91.98%, intermediate 93.69%, sharp 87.78% | F1: blurred 83.14%, intermediate 81.54%, sharp 82.35% | No monotonic blur relationship is established. |
| Density | F1: single 94.81%, low clutter 93.63%, high clutter 87.72% | F1: single 87.68%, low clutter 83.01%, high clutter 79.12% | Higher clutter is the common adverse condition. |
| Size | Recall: small 41.75%, medium 87.78%, large 95.45% | Recall: small 53.37%, medium 71.91%, large 89.84% | Small-instance recovery is the largest detector weakness in each reference. |
| Position | Recall: near edge 70.18%, central 93.67% | Recall: near edge 71.99%, central 87.77% | Peripheral objects are harder to recover in both references. |
| Frequency | Recall: rare 89.84%, common 86.94% | Recall: rare 74.00%, common 84.00% | Frequency grouping is composition-dependent; only MTSD shows the expected lower rare-group recall. |

### MDWD reference: class diagnostics

| Class | Support (objects) | Precision | Recall | F1 | Mean matched IoU |
|---|---:|---:|---:|---:|---:|
| Mixed Waste | 334 | 95.82% | 89.22% | 92.40% | 91.83% |
| Orange CMD | 78 | 93.85% | 78.21% | 85.31% | 94.09% |
| Organic Waste | 178 | 97.13% | 94.94% | 96.02% | 93.31% |
| Other Waste | 278 | 91.63% | 78.78% | 84.72% | 89.88% |
| Recyclable Material | 238 | 94.07% | 93.28% | 93.67% | 93.40% |

`Orange CMD` and `Other Waste` are the lowest-F1 MDWD reference classes. The former is chiefly recall-limited; the latter has both lower precision and recall. This supports choosing examples from those classes for qualitative error inspection, but does not by itself establish error causes.

### MTSD reference: class diagnostics

| Class | Support (objects) | Precision | Recall | F1 | Mean matched IoU |
|---|---:|---:|---:|---:|---:|
| Auxiliary Sign | 120 | 65.00% | 65.00% | 65.00% | 87.46% |
| Back-Unknown | 661 | 84.75% | 79.88% | 82.24% | 91.92% |
| Blind-Spot Mirror | 160 | 84.00% | 91.87% | 87.76% | 95.39% |
| Directional Sign | 108 | 75.28% | 62.04% | 68.02% | 92.15% |
| No Entry (One Way) | 211 | 88.69% | 92.89% | 90.74% | 95.58% |
| No Through Road (T-Junction) | 61 | 90.62% | 95.08% | 92.80% | 96.75% |
| Other-Unknown | 220 | 73.09% | 74.09% | 73.59% | 92.69% |
| Pedestrian Crossing | 143 | 88.36% | 90.21% | 89.27% | 93.64% |
| Roundabout Ahead | 67 | 89.39% | 88.06% | 88.72% | 95.77% |
| Stop Sign | 130 | 90.08% | 90.77% | 90.42% | 96.28% |
| Street Sign | 94 | 78.89% | 75.53% | 77.17% | 92.95% |

The weakest MTSD reference classes are Auxiliary Sign, Directional Sign, Other-Unknown, and Street Sign. `Tourist Sign` is intentionally absent from this comparison because it is excluded by the stored inventory policy.

## Track 2 — zero-shot PromptDetect robustness

### Target families and exact formulations

All variants below share one fixed target ground-truth set inside the family. This is the controlled experimental unit for prompt robustness.

| Dataset | Family / target class | Target boxes | Positive / negative images | Canonical | Lexical paraphrase | Descriptive paraphrase | Instruction |
|---|---|---:|---:|---|---|---|---|
| MDWD | Mixed Waste | 334 | 126 / 243 | `black garbage bag` | `black refuse bag` | `black bag containing mixed household waste` | `find all black mixed-waste bags` |
| MDWD | Orange CMD | 78 | 37 / 332 | `orange garbage bag` | `orange refuse bag` | `orange bag containing domestic waste` | `find all orange waste bags` |
| MDWD | Organic Waste | 178 | 107 / 262 | `white organic-waste bag` | `white domestic organic-waste bag` | `white bag containing organic household waste` | `find all white organic-waste bags` |
| MDWD | Recyclable Material | 238 | 114 / 255 | `grey recycling bag` | `grey household recycling bag` | `grey bag containing recyclable household material` | `find all grey recycling bags` |
| MTSD | Blind-Spot Mirror | 160 | 145 / 602 | `blind-spot convex mirror` | `roadside convex mirror for blind spots` | `convex traffic mirror used to improve road visibility` | `find all roadside blind-spot mirrors` |
| MTSD | No Entry (One Way) | 211 | 181 / 566 | `no entry sign` | `road sign indicating no entry` | `traffic sign prohibiting vehicles from entering` | `find all no entry signs` |
| MTSD | Pedestrian Crossing | 143 | 115 / 632 | `pedestrian crossing sign` | `road sign for a pedestrian crossing` | `sign indicating a pedestrian crossing` | `find all pedestrian crossing signs` |
| MTSD | Roundabout Ahead | 67 | 59 / 688 | `roundabout ahead sign` | `warning sign for a roundabout ahead` | `traffic sign indicating an upcoming roundabout` | `find all roundabout ahead signs` |
| MTSD | Stop Sign | 130 | 124 / 623 | `stop sign` | `road sign instructing drivers to stop` | `traffic sign indicating that vehicles must stop` | `find all stop signs` |

### Model-level prompt sensitivity

`Mean F1` is the macro mean over all families and four formulations. `Worst-prompt F1` is the macro mean of the worst formulation in each family. The difference between them is a robustness-relevant worst-case penalty, not a new correctness metric.

#### MDWD

| Model | Mean F1 | Worst-prompt F1 | Canonical F1 | Instruction F1 | Mean F1 range | Mean relative degradation | Worst family | Mean inference (ms) |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Cosmos Reason2 2B | 21.11% | 12.21% | 19.94% | 32.58% | 0.2037 | 58.9% | Recyclable Material (80.2%) | 1,567.2 |
| Cosmos Reason2 8B | 36.33% | 32.65% | 37.07% | 35.76% | 0.0713 | 18.9% | Orange CMD (29.7%) | 2,561.2 |
| LocateAnything 3B | 30.30% | 24.04% | 33.94% | 24.69% | 0.1245 | 34.3% | Orange CMD (44.9%) | 1,984.7 |
| SAM 3 | 52.11% | 36.27% | 56.76% | 36.52% | 0.2927 | 49.5% | Recyclable Material (100.0%) | 112.0 |
| SAM 3.1 | 52.62% | 36.13% | 56.48% | 38.11% | 0.2942 | 49.0% | Recyclable Material (96.0%) | 111.8 |

#### MTSD

| Model | Mean F1 | Worst-prompt F1 | Canonical F1 | Instruction F1 | Mean F1 range | Mean relative degradation | Worst family | Mean inference (ms) |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Cosmos Reason2 2B | 19.26% | 15.93% | 19.45% | 21.60% | 0.0742 | 34.7% | Roundabout Ahead (54.1%) | 2,643.3 |
| Cosmos Reason2 8B | 30.54% | 25.61% | 27.05% | 30.49% | 0.0980 | 30.6% | Roundabout Ahead (66.5%) | 1,988.4 |
| LocateAnything 3B | 8.65% | 6.00% | 12.28% | 8.28% | 0.0628 | 47.1% | Pedestrian Crossing (71.4%) | 8,110.5 |
| SAM 3 | 24.18% | 13.55% | 39.85% | 23.59% | 0.2736 | 61.6% | Blind-Spot Mirror (78.2%) | 189.1 |
| SAM 3.1 | 23.67% | 13.21% | 39.44% | 22.74% | 0.2732 | 61.4% | Blind-Spot Mirror (79.7%) | 194.6 |

### Prediction-set consistency

This table is supplementary to correctness. It shows whether two prompt variants produce similar boxes; it cannot say whether those boxes are correct.

| Dataset | Model | Macro pairwise box F1 | Macro pairwise Jaccard | Matched-pair IoU | Least consistent family | Family pairwise F1 |
|---|---|---:|---:|---:|---|---:|
| MDWD | Cosmos Reason2 2B | 55.15% | 54.67% | 97.23% | Orange CMD | 54.30% |
| MDWD | Cosmos Reason2 8B | 81.55% | 80.13% | 97.46% | Orange CMD | 72.96% |
| MDWD | LocateAnything 3B | 69.06% | 64.00% | 94.39% | Organic Waste | 62.66% |
| MDWD | SAM 3 | 79.93% | 79.49% | 98.70% | Recyclable Material | 67.33% |
| MDWD | SAM 3.1 | 79.12% | 78.63% | 98.80% | Recyclable Material | 65.43% |
| MTSD | Cosmos Reason2 2B | 64.39% | 64.25% | 97.32% | Blind-Spot Mirror | 49.96% |
| MTSD | Cosmos Reason2 8B | 69.35% | 69.16% | 98.11% | Blind-Spot Mirror | 62.59% |
| MTSD | LocateAnything 3B | 55.15% | 50.18% | 93.11% | Roundabout Ahead | 53.17% |
| MTSD | SAM 3 | 61.64% | 57.72% | 97.59% | Blind-Spot Mirror | 43.30% |
| MTSD | SAM 3.1 | 61.17% | 57.09% | 97.62% | Blind-Spot Mirror | 41.68% |

### Full family-level zero-shot statistics

The following is the compact, complete table needed to reproduce the prompt-robustness comparison without opening 180 per-prompt rows. `Min` and `Max` are across the four formulations in a family.

#### MDWD

| Model | Family | Boxes | F1 mean | Min | Max | Range | Relative degradation | Best / worst formulation |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Cosmos Reason2 2B | Mixed Waste | 334 | 0.2995 | 0.1882 | 0.4251 | 0.2369 | 55.7% | instruction / descriptive |
| Cosmos Reason2 2B | Orange CMD | 78 | 0.1306 | 0.1106 | 0.1690 | 0.0584 | 34.6% | instruction / descriptive |
| Cosmos Reason2 2B | Organic Waste | 178 | 0.2013 | 0.1136 | 0.3249 | 0.2113 | 65.0% | instruction / descriptive |
| Cosmos Reason2 2B | Recyclable Material | 238 | 0.2130 | 0.0760 | 0.3843 | 0.3083 | 80.2% | instruction / descriptive |
| Cosmos Reason2 8B | Mixed Waste | 334 | 0.4915 | 0.4257 | 0.5222 | 0.0965 | 18.5% | canonical / instruction |
| Cosmos Reason2 8B | Orange CMD | 78 | 0.2101 | 0.1826 | 0.2598 | 0.0772 | 29.7% | instruction / descriptive |
| Cosmos Reason2 8B | Organic Waste | 178 | 0.3866 | 0.3499 | 0.4142 | 0.0643 | 15.5% | canonical / instruction |
| Cosmos Reason2 8B | Recyclable Material | 238 | 0.3649 | 0.3476 | 0.3950 | 0.0474 | 12.0% | instruction / canonical |
| LocateAnything 3B | Mixed Waste | 334 | 0.4822 | 0.3935 | 0.5922 | 0.1987 | 33.6% | canonical / instruction |
| LocateAnything 3B | Orange CMD | 78 | 0.2123 | 0.1590 | 0.2886 | 0.1296 | 44.9% | lexical / descriptive |
| LocateAnything 3B | Organic Waste | 178 | 0.2481 | 0.1884 | 0.2762 | 0.0878 | 31.8% | lexical / instruction |
| LocateAnything 3B | Recyclable Material | 238 | 0.2696 | 0.2209 | 0.3028 | 0.0819 | 27.1% | lexical / instruction |
| SAM 3 | Mixed Waste | 334 | 0.7739 | 0.6890 | 0.8560 | 0.1670 | 19.5% | lexical / descriptive |
| SAM 3 | Orange CMD | 78 | 0.6134 | 0.3226 | 0.8312 | 0.5086 | 61.2% | canonical / instruction |
| SAM 3 | Organic Waste | 178 | 0.4864 | 0.4393 | 0.5323 | 0.0930 | 17.5% | lexical / instruction |
| SAM 3 | Recyclable Material | 238 | 0.2106 | 0.0000 | 0.4021 | 0.4021 | 100.0% | lexical / instruction |
| SAM 3.1 | Mixed Waste | 334 | 0.7724 | 0.6645 | 0.8427 | 0.1782 | 21.1% | lexical / descriptive |
| SAM 3.1 | Orange CMD | 78 | 0.6218 | 0.3226 | 0.8312 | 0.5086 | 61.2% | canonical / instruction |
| SAM 3.1 | Organic Waste | 178 | 0.4804 | 0.4418 | 0.5372 | 0.0954 | 17.8% | lexical / descriptive |
| SAM 3.1 | Recyclable Material | 238 | 0.2303 | 0.0164 | 0.4108 | 0.3944 | 96.0% | descriptive / instruction |

#### MTSD

| Model | Family | Boxes | F1 mean | Min | Max | Range | Relative degradation | Best / worst formulation |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Cosmos Reason2 2B | Blind-Spot Mirror | 160 | 0.2204 | 0.1618 | 0.2491 | 0.0873 | 35.0% | canonical / instruction |
| Cosmos Reason2 2B | No Entry | 211 | 0.2743 | 0.2469 | 0.3545 | 0.1076 | 30.3% | instruction / lexical |
| Cosmos Reason2 2B | Pedestrian Crossing | 143 | 0.1909 | 0.1728 | 0.2234 | 0.0506 | 22.7% | instruction / lexical |
| Cosmos Reason2 2B | Roundabout Ahead | 67 | 0.0631 | 0.0368 | 0.0801 | 0.0433 | 54.1% | instruction / lexical |
| Cosmos Reason2 2B | Stop Sign | 130 | 0.2142 | 0.1782 | 0.2602 | 0.0820 | 31.5% | instruction / lexical |
| Cosmos Reason2 8B | Blind-Spot Mirror | 160 | 0.3519 | 0.3083 | 0.3870 | 0.0787 | 20.3% | descriptive / canonical |
| Cosmos Reason2 8B | No Entry | 211 | 0.3822 | 0.3662 | 0.4131 | 0.0469 | 11.3% | descriptive / lexical |
| Cosmos Reason2 8B | Pedestrian Crossing | 143 | 0.3021 | 0.2432 | 0.3529 | 0.1097 | 31.1% | descriptive / canonical |
| Cosmos Reason2 8B | Roundabout Ahead | 67 | 0.1622 | 0.0847 | 0.2531 | 0.1684 | 66.5% | descriptive / lexical |
| Cosmos Reason2 8B | Stop Sign | 130 | 0.3286 | 0.2781 | 0.3643 | 0.0862 | 23.7% | lexical / canonical |
| LocateAnything 3B | Blind-Spot Mirror | 160 | 0.1115 | 0.0805 | 0.1364 | 0.0559 | 41.0% | canonical / instruction |
| LocateAnything 3B | No Entry | 211 | 0.1077 | 0.0785 | 0.1419 | 0.0634 | 44.7% | canonical / descriptive |
| LocateAnything 3B | Pedestrian Crossing | 143 | 0.0954 | 0.0506 | 0.1767 | 0.1261 | 71.4% | canonical / lexical |
| LocateAnything 3B | Roundabout Ahead | 67 | 0.0281 | 0.0235 | 0.0348 | 0.0113 | 32.5% | canonical / descriptive |
| LocateAnything 3B | Stop Sign | 130 | 0.0898 | 0.0668 | 0.1241 | 0.0573 | 46.2% | canonical / lexical |
| SAM 3 | Blind-Spot Mirror | 160 | 0.2948 | 0.1196 | 0.5476 | 0.4280 | 78.2% | canonical / descriptive |
| SAM 3 | No Entry | 211 | 0.2566 | 0.1749 | 0.3449 | 0.1700 | 49.3% | instruction / descriptive |
| SAM 3 | Pedestrian Crossing | 143 | 0.3419 | 0.1818 | 0.6222 | 0.4404 | 70.8% | canonical / lexical |
| SAM 3 | Roundabout Ahead | 67 | 0.0982 | 0.0733 | 0.1248 | 0.0515 | 41.3% | canonical / descriptive |
| SAM 3 | Stop Sign | 130 | 0.2177 | 0.1277 | 0.4060 | 0.2783 | 68.5% | canonical / lexical |
| SAM 3.1 | Blind-Spot Mirror | 160 | 0.2835 | 0.1132 | 0.5581 | 0.4449 | 79.7% | canonical / descriptive |
| SAM 3.1 | No Entry | 211 | 0.2500 | 0.1663 | 0.3396 | 0.1733 | 51.0% | instruction / descriptive |
| SAM 3.1 | Pedestrian Crossing | 143 | 0.3414 | 0.1810 | 0.6074 | 0.4264 | 70.2% | canonical / lexical |
| SAM 3.1 | Roundabout Ahead | 67 | 0.0925 | 0.0712 | 0.1147 | 0.0435 | 37.9% | canonical / descriptive |
| SAM 3.1 | Stop Sign | 130 | 0.2163 | 0.1289 | 0.4067 | 0.2778 | 68.3% | canonical / descriptive |

## Track 3 — MTSD attribute-classification robustness

### Reference configuration and class slices

The selected reference is VJEPA 2.1-L with LoRA (`vjepa21_vitl_lora-20260805-141508`), evaluated on 1,890 ground-truth crops. Its reported overall figure is the unweighted mean of the four head macro-F1 values: **90.06%**.

| Attribute head | Accuracy | Macro-F1 | Classes / F1 / support |
|---|---:|---:|---|
| Viewing angle | 93.70% | 92.97% | Front 93.15% (699); Back 97.55% (738); Side 88.21% (453) |
| Mounting | 97.67% | 95.26% | Pole-Mounted 98.64% (1,629); Wall-Mounted 91.88% (261) |
| Condition | 84.87% | 74.70% | Good 91.62% (1,388); Weathered 67.00% (392); Heavily Damaged 65.49% (110) |
| Sign shape | 97.67% | 97.32% | Circular 98.33% (746); Quadrangle 97.57% (615); Triangular 97.30% (281); Octagonal 96.19% (212); Pentagon 97.22% (36) |

This supports a careful claim that condition severity is the principal retained attribute robustness limitation; it does **not** establish why individual signs are misclassified. Side views and wall-mounted signs are also weaker than their more frequent counterparts, but the drops are smaller.

### Current 1,890-crop configuration comparison

Rows are ordered by the mean of four per-head macro-F1 values. “Frozen”, “LoRA”, and “fine-tune” are experiment labels from the stored runs, not a controlled causal effect independent of architecture.

| Representation / adaptation | View | Mounting | Condition | Shape | Mean of heads |
|---|---:|---:|---:|---:|---:|
| VJEPA 2.1-L LoRA | 92.97% | 95.26% | 74.70% | 97.32% | 90.06% |
| DINOv3-L LoRA | 92.48% | 95.35% | 72.82% | 97.74% | 89.60% |
| LingBot-Vision-L LoRA | 92.97% | 95.69% | 70.55% | 97.65% | 89.21% |
| LingBot-Vision-B LoRA | 91.65% | 94.15% | 71.01% | 97.35% | 88.54% |
| VJEPA 2.1-B LoRA | 92.69% | 94.29% | 71.26% | 95.51% | 88.44% |
| DINOv3-B LoRA | 92.19% | 95.19% | 69.24% | 96.76% | 88.35% |
| ConvNeXt-B LoRA | 91.90% | 93.28% | 70.22% | 96.47% | 87.97% |
| ConvNeXt-L full fine-tune | 91.73% | 93.11% | 69.55% | 96.40% | 87.70% |
| ConvNeXt-L LoRA | 91.80% | 92.63% | 68.91% | 96.59% | 87.48% |
| ConvNeXt-B full fine-tune | 91.81% | 93.92% | 65.02% | 96.60% | 86.84% |
| DINOv3-L frozen | 85.91% | 91.27% | 58.35% | 90.95% | 81.62% |
| LingBot-Vision-L frozen | 87.44% | 89.99% | 55.78% | 91.51% | 81.18% |
| VJEPA 2.1-L frozen | 88.12% | 88.87% | 56.01% | 90.40% | 80.85% |
| DINOv3-B frozen | 86.86% | 89.85% | 57.67% | 88.34% | 80.68% |
| LingBot-Vision-B frozen | 87.90% | 89.61% | 53.06% | 90.01% | 80.14% |
| VJEPA 2.1-B frozen | 85.23% | 86.14% | 55.68% | 88.46% | 78.88% |
| ConvNeXt-L frozen | 83.70% | 88.19% | 55.38% | 87.65% | 78.73% |
| ConvNeXt-B frozen | 83.70% | 86.51% | 53.39% | 89.03% | 78.16% |

Six older 503-crop configurations are preserved in the machine-readable report but should not be pooled with this table because their evaluation population differs.

## Recommended dissertation comparison layout

Use three compact tables or panels, one per track:

1. **Supervised detection:** reference aggregate row plus the seven slice dimensions. Use separate panels for image-level F1 and object-level recall. The data-slice comparison table above supplies a concise across-dataset narrative.
2. **Zero-shot PromptDetect:** model-level table showing macro mean F1, worst-prompt F1, and mean relative degradation for each dataset. Add the exact family prompts in an appendix or a methods table. Plot mean versus worst-case F1, with colour/annotation for degradation; do not call it a supervised detector benchmark.
3. **MTSD attributes:** per-head macro-F1 for the best 1,890-crop representation, then a condition-class F1 panel. This makes the materially weaker condition head visible without implying that it is comparable to detector F1.

Suggested result wording:

> Across the two supervised detection audits, the largest observed degradation was associated with small objects, peripheral placement and high object density. In contrast, image-property terciles showed smaller and non-monotonic differences, and are interpreted as observational scene slices rather than controlled corruption tests. Prompt-based zero-shot localisation exhibited a separate robustness limitation: semantically related reformulations could change both correctness and raw prediction sets substantially. For MTSD attributes, sign condition was materially less reliable than view angle, mounting and sign shape on ground-truth crops.

## Source map and reproducibility checklist

Use these files as the authority for any number not reproduced above:

- MDWD supervised slices: `Documents/Final-Reports/Robustness-Slices/20260908-103240/robustness_slice_summary.md`, `robustness_slice_results.csv`, and `robustness_slice_config.json`.
- MTSD supervised and attribute slices: `Documents/Final-Reports/Robustness-Slices/20260914-123609/robustness_slice_summary.md`, `robustness_slice_results.csv`, `robustness_slice_config.json`, and `insufficient_or_missing_inputs.md`.
- MDWD PromptDetect: `Results/PromptDetect/BatchEvaluation/MDWD/20260821-170445-optimized-v2-bounded-sensitivity/prompt_sensitivity_summary.json`, `prompt_sensitivity_per_model.csv`, `prompt_sensitivity_per_prompt.csv`, and `protocol_snapshot.yaml`.
- MTSD PromptDetect: `Results/PromptDetect/BatchEvaluation/MTSD/20260822-102455-optimized-v2-bounded-sensitivity/prompt_sensitivity_summary.json`, `prompt_sensitivity_per_model.csv`, `prompt_sensitivity_per_prompt.csv`, and `protocol_snapshot.yaml`.
- Generator and matching/slicing implementation: `Scripts/FinalEvaluation/robustness_slice_analysis.py`.

Before publishing a table or figure, verify the following:

- The exact run, split, class/exclusion policy, and confidence threshold appear in the caption or note.
- Detection image-level F1 and object-level recall are not plotted on the same unlabelled axis.
- MTSD `Tourist Sign` handling is stated wherever MTSD detector results are used.
- Prompt-family score ranges are labelled as **formulation sensitivity**, not data-condition slicing.
- Attribute results state “ground-truth crops” and the 1,890-crop GRP-1--GRP-3 snapshot scope.
- No causal statement about blur, contrast, illumination, rarity, occlusion, or capture conditions is made without a separately controlled experiment or visual error inspection.
- No result is labelled “final” beyond its stored report status: MDWD detector and both bounded prompt-sensitivity evaluations are final; MTSD supervised detection and attributes remain historical snapshots/inventories.
