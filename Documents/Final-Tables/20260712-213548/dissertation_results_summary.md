# Dissertation results summary

Generated: 2026-07-12T21:35:48 - consolidated from existing result files only (no metrics recomputed or invented).


## MDWD supervised detection

| dataset | model_family | model_variant | parameters | model_size_mb | precision | recall | f1 | map50 | map50_95 |
|---|---|---|---|---|---|---|---|---|---|
| MDWD | YOLO11 | yolo11n | 2590815 | 5.22 | 0.9035638191227982 | 0.8036866029135148 | 0.8507037127455663 | 0.8636417638926028 | 0.6888369062440758 |
| MDWD | YOLO11 | yolo11s | 9429727 | 18.29 | 0.9345350765796423 | 0.8350317903928598 | 0.8819858833775887 | 0.8973637859791648 | 0.7346008341140627 |
| MDWD | YOLO11 | yolo11m | 20056863 | 38.64 | 0.9203475253528313 | 0.8200175360752494 | 0.8672905780509078 | 0.8866884997447217 | 0.7353568731041266 |
| MDWD | YOLO11 | yolo11l | 25314335 | 48.83 | 0.9430254248999848 | 0.8613766154054854 | 0.9003537244993949 | 0.8992977873215562 | 0.7614110989418542 |
| MDWD | YOLO12 | yolo12n | 2569023 | 5.26 | 0.9154546014676533 | 0.8216141219033026 | 0.8659996215550535 | 0.8834670451829236 | 0.7121870181049601 |
| MDWD | YOLO12 | yolo12s | 9255071 | 18.06 | 0.9268869459779859 | 0.8458354822242413 | 0.8845083183312759 | 0.8941019091806585 | 0.7413695245455391 |
| MDWD | YOLO12 | yolo12m | 20141343 | 38.88 | 0.9416166668810266 | 0.8759404715528101 | 0.9075919868142031 | 0.9146507932357523 | 0.7655754993119791 |
| MDWD | YOLO26 | yolo26n | 2505750 | 5.14 | 0.9222904546628795 | 0.803722135332503 | 0.8589337736179753 | 0.8697870789738549 | 0.6983701238316875 |
| MDWD | YOLO26 | yolo26s | 9951734 | 19.38 | 0.9469718641738665 | 0.8486864448127379 | 0.8951393265871109 | 0.8998867887087038 | 0.7559666179134011 |
| MDWD | YOLO26 | yolo26m | 21780598 | 42.0 | 0.9405646614668035 | 0.8614048960762952 | 0.8992460511580751 | 0.9073978148901288 | 0.762111243348453 |
| MDWD | YOLO26 | yolo26l | 26184054 | 50.54 | 0.9578149412414303 | 0.8732289459488067 | 0.9135681972513805 | 0.9152102000336587 | 0.7951067059184301 |
| MDWD | YOLO26-DGX | yolo26n | 2505750 | 5.14 | 0.9052559522558431 | 0.8332353816675917 | 0.8677538670065944 | 0.8819945517967597 | 0.7101449387196088 |
| MDWD | YOLO26-DGX | yolo26s | 9951734 | 19.38 | 0.9375666900010955 | 0.8561401190591977 | 0.8950051965560605 | 0.9091723737957915 | 0.7565106419877367 |
| MDWD | YOLO26-DGX | yolo26m | 21780598 | 41.99 | 0.9396810135352093 | 0.8592301594256876 | 0.8976566260803364 | 0.9087364468168133 | 0.7637484286319105 |
| MDWD | YOLO26-DGX | yolo26l | 26184054 | 50.54 | 0.9485779923671703 | 0.8795903108683845 | 0.9127824935072647 | 0.9099584651611433 | 0.7947859960634506 |

*Full table incl. source paths: `mdwd_detection_results.csv`*

## MTSD supervised detection

**PENDING** - MTSD detection: no results yet - pending (train via Scripts/MTSD-Scripts/MTSD-SupervisedNotebooks)

## MTSD attribute classification

| variant | backbone | adaptation | mean_macro_f1 | view_angle_f1 | mounting_f1 | condition_f1 | sign_shape_f1 | total_params | trainable_params |
|---|---|---|---|---|---|---|---|---|---|
| dinov3 | dinov3 | frozen | 0.7826 | 0.8649 | 0.8458 | 0.5534 | 0.8663 | 85670413 | 9997 |
| dinov3_lora | dinov3 | lora | 0.8734 | 0.9129 | 0.8959 | 0.7346 | 0.9503 | 85965325 | 304909 |
| vjepa | vjepa | frozen | 0.7699 | 0.8823 | 0.8672 | 0.5110 | 0.8189 | 304694285 | 13325 |
| vjepa_lora | vjepa | lora | 0.8400 | 0.9089 | 0.9054 | 0.5948 | 0.9507 | 305480717 | 799757 |
| convnext_frozen | convnext | frozen | 0.7629 | 0.8426 | 0.8026 | 0.5673 | 0.8389 | 27828589 | 9997 |
| convnext | convnext | finetune | 0.8641 | 0.9054 | 0.8734 | 0.7113 | 0.9663 | 27828589 | 27828589 |

> Best attribute model: dinov3_lora (mean macro-F1 0.8734); weakest head across all variants is 'condition'.

*Full table incl. source paths: `mtsd_attribute_results.csv`*

## PromptDetect (prompt-based detection)

| evaluation_protocol | dataset | split | model | prompt | precision | recall | f1 | ap50 | map50_95 |
|---|---|---|---|---|---|---|---|---|---|
| class-agnostic-exploratory | MDWD | test | SAM 3 | Black Garbage Bag | 1.0 | 0.2727 | 0.4286 | 0.2727 | 0.2454 |
| class-agnostic-exploratory | MDWD | test | SAM 3 | Grey Garbage Bag | 0.875 | 0.6364 | 0.7368 | 0.6136 | 0.5368 |
| class-agnostic-exploratory | MDWD | test | SAM 3 | White Garbage Bag | 0.875 | 0.6364 | 0.7368 | 0.6364 | 0.5864 |
| class-agnostic-exploratory | MDWD | test | SAM 3 | Orange Garbage Bag | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| class-agnostic-exploratory | MDWD | test | SAM 3 | Domestic Waste | 0.5 | 0.2727 | 0.3529 | 0.2273 | 0.1645 |
| class-agnostic-exploratory | MDWD | test | SAM 3 | Garbage | 0.7333 | 1.0 | 0.8462 | 0.9282 | 0.7515 |
| class-agnostic-exploratory | MDWD | test | SAM 3 | <all prompts combined> | 0.275 | 1.0 | 0.4314 | 0.7329 | 0.6493 |
| class-agnostic-exploratory | MDWD | test | Cosmos Reason2 2B | Black Garbage Bag | 1.0 | 0.1818 | 0.3077 | 0.1818 | 0.0546 |
| class-agnostic-exploratory | MDWD | test | Cosmos Reason2 2B | Grey Garbage Bag | 0.8 | 0.3636 | 0.5 | 0.3091 | 0.1182 |
| class-agnostic-exploratory | MDWD | test | Cosmos Reason2 2B | White Garbage Bag | 1.0 | 0.3636 | 0.5333 | 0.3636 | 0.2 |
| class-agnostic-exploratory | MDWD | test | Cosmos Reason2 2B | Orange Garbage Bag | 1.0 | 0.3636 | 0.5333 | 0.3636 | 0.1545 |
| class-agnostic-exploratory | MDWD | test | Cosmos Reason2 2B | Domestic Waste | 0.8571 | 0.5455 | 0.6667 | 0.4935 | 0.2725 |
| class-agnostic-exploratory | MDWD | test | Cosmos Reason2 2B | Garbage | 0.8333 | 0.4545 | 0.5882 | 0.4091 | 0.1758 |
| class-agnostic-exploratory | MDWD | test | Cosmos Reason2 2B | <all prompts combined> | 0.25 | 0.6364 | 0.359 | 0.2508 | 0.1167 |
| class-agnostic-exploratory | MDWD | test | SAM 3.1 | garbage bag | 0.9167 | 0.9565 | 0.9362 | 0.9565 | 0.8481 |
| class-agnostic-exploratory | MDWD | test | Cosmos Reason2 2B | garbage bag | 1.0 | 0.3478 | 0.5161 | 0.3478 | 0.2632 |

> Results\PromptDetect\BatchEvaluation\MDWD\20260709-234655 is exploratory class-agnostic evidence

> Results\PromptDetect\BatchEvaluation\MDWD\20260710-011505 is exploratory class-agnostic evidence

> Results\PromptDetect\BatchEvaluation\MTSD\20260712-213334-smoke-sensitivity is a prompt-sensitivity run; see prompt_sensitivity_results.csv

*Full table incl. source paths: `promptdetect_results.csv`*

## Model efficiency (inference speed)

**PENDING** - Model efficiency: no inference-speed benchmark runs yet - pending (run Scripts/Other-Scripts/Inference-Benchmark/inference_speed_benchmark.py)

## Prompt sensitivity and prediction consistency

Controlled paraphrase families only (identical target GT per family); synonym and broad-prompt effects are reported in the PromptDetect section, never here. Prediction consistency measures agreement between prompt variants, not accuracy.

| dataset | model | n_families | macro_mean_f1 | mean_within_family_f1_std | mean_f1_range | mean_relative_f1_degradation | worst_family | macro_canonical_f1 | macro_instruction_f1 | protocol_version |
|---|---|---|---|---|---|---|---|---|---|---|
| MTSD | SAM 3 | 5 | 0.2686 | 0.1283 | 0.3362 | 0.4278 | mtsd-blind-spot-mirror | 0.4 | 0.346 | prompt-sensitivity-v1 |

*Full tables: `prompt_sensitivity_results.csv` (per family), `prompt_sensitivity_model_summary.csv`, `prompt_consistency_results.csv` (per prompt pair); source run paths and protocol hashes are recorded per row.*

> Prompt-sensitivity tables aggregated from 1 run(s); prediction consistency measures agreement between prompt variants, not accuracy against GT.

---
*Source files are recorded per row for traceability. Pending sections list the command that produces them.*
