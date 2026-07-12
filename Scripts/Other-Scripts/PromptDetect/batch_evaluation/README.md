# PromptDetect — Batch Evaluation

Automated, ground-truth-scored evaluation of the PromptDetect prompt-based
models (SAM 3, SAM 3.1, Cosmos Reason2 2B/8B/32B, LocateAnything 3B) over
MDWD or MTSD. The existing manual PromptDetect UI ([../app.py](../app.py)) is
untouched — this is an additional mode that reuses the same
`backend.DetectionBackend` and model registry.

There are three intentionally distinct modes: `run_batch_eval.py` is the legacy
class-agnostic exploratory evaluator; `run_dissertation_protocol.py` loads a
fixed YAML protocol and performs target-class-aware dissertation evaluation;
the same runner with `prompt_protocols/prompt_sensitivity_protocol.yaml` (or
the thin `run_prompt_sensitivity.py` wrapper) runs the controlled
prompt-sensitivity experiment — 9 paraphrase families x 4 semantically
equivalent wordings with identical target GT per family, plus prompt-pair
prediction-consistency analysis. With `--dataset both` and no explicit
`--protocol`, `run_dissertation_protocol.py` runs the dissertation protocol
AND the prompt-sensitivity protocol as separate stages with separate run
directories (`...-sensitivity`) and protocol labels (`targeted-v1` /
`prompt-sensitivity-v1`).
See [the protocol document](../../../../Documents/PromptDetect-Dissertation-Protocol.md)
and [the prompt-sensitivity document](../../../../Documents/PromptDetect-Prompt-Sensitivity-Protocol.md).

The dissertation headline macro includes `class-targeted` and
`synonym-comparison` prompts only. Broad/optional prompts are reported
separately. Universal model comparison uses P/R/F1, matched IoU and runtime;
Cosmos/LocateAnything constant-score AP is marked `ap_meaningful=false` and is
never a headline ranking field. `--final` MTSD runs require the canonical
unaugmented prepared test folder, valid prep/split hashes, explicit approved
scope, QA-only data and a resolved QA gate.

## Current research status

`Results/PromptDetect/BatchEvaluation/MDWD/20260709-234655` is a completed
five-image MDWD pilot using SAM 3 and Cosmos Reason2 2B with six exploratory
prompts. It validates model loading and GT scoring only; do not use its metrics
as dissertation evidence. The next runs should use a fixed prompt protocol and
an adequately sampled MDWD test set, followed by MTSD after annotation QA and
detection-dataset preparation.

## What it does

1. Loads a dataset split into a common ground-truth format:
   - **MDWD** — YOLO layout at `Datasets/MDWD/MDWD-YOLO26` (`train`/`valid`/`test`);
   - **MTSD** — the prepared dataset at `Datasets/MTSD/Prepared/MTSD-Unaugmented/MTSD-YOLO`
     when it exists, otherwise the QA COCO annotations
     (`Datasets/MTSD/Annotations/GRP-*/Final-QA/`, groups discovered
     dynamically). Before the prepared dataset exists, `train`/`valid`/`test`
     reproduce the prep notebook's seeded per-group 80/10/10 split; `all`
     evaluates every QA image.
2. Runs each selected model sequentially (previous model is freed first),
   feeding every image each of the 1–15 prompts. Zero prompts are accepted
   for dry-run dataset/model sanity checks only.
3. Matches predictions to GT boxes (greedy IoU matching, threshold
   configurable) and reports: precision, recall, F1, accuracy
   (TP/(TP+FP+FN)), AP@50, mAP@50:95, mean matched IoU, FP/FN counts,
   duplicate detections, per-image / per-prompt / per-model breakdowns, and a
   **prompt vs ground-truth-class confusion matrix** (which classes each
   prompt actually matched — no manual prompt→class mapping needed).

## Safety defaults

- **No model runs unless explicitly selected** (CLI `--models` is required;
  the Gradio UI pre-selects nothing).
- **Heavy models are opt-in**: Cosmos Reason2 32B refuses to run without
  `--allow-heavy` (CLI) or the explicit *allow heavy models* toggle (UI).
- `--dry-run` loads the dataset and prints the execution plan without
  touching any model; `--max-images N` caps a run for smoke testing.
- Each run writes to its own timestamped folder under
  `Results/PromptDetect/BatchEvaluation/<DATASET>/<timestamp>/` — previous
  results are never overwritten.

## CLI

```bash
# Preview (no models loaded, nothing written):
python Scripts/Other-Scripts/PromptDetect/batch_evaluation/run_batch_eval.py \
    --dataset MTSD --split test \
    --prompts "traffic sign" "stop sign" "warning sign" \
    --models sam3 locateanything_3b \
    --max-images 25 --dry-run

# Small real run (SAM 3 only, all 10-image comparison sheets saved):
python Scripts/Other-Scripts/PromptDetect/batch_evaluation/run_batch_eval.py \
    --dataset MDWD --split test --prompts "garbage bag" \
    --models sam3 --max-images 10 --save-visualizations
```

Model slugs are derived from the backend registry, with convenience aliases
such as `sam3`, `sam3.1`, `cosmos8b`, and `locateanything`. Run with an invalid
`--models x` to see the current list (e.g. `sam_3`, `sam_3.1`,
`cosmos_reason2_2b`, `cosmos_reason2_8b`, `cosmos_reason2_32b`,
`locateanything_3b`).

## Gradio

```bash
python Scripts/Other-Scripts/PromptDetect/batch_evaluation/gradio_batch_eval.py
```

Dataset/split dropdowns (`all` means every MTSD QA image), prompt textbox
(0-15, one per line; zero prompts for dry-run only), model checkboxes (32B in a separate gated group),
max-images, thresholds, a **Save visual comparisons (GT vs predictions)** toggle,
dry-run toggle, metrics table and result-folder path. The visual toggle writes a
sheet for every evaluated image/model/prompt and displays the HTML-index path.

## Output files (per run)

| File | Contents |
| --- | --- |
| `run_config.json` | dataset, split, prompts, models, thresholds, counts |
| `ground_truth_index.csv` | one row per GT box |
| `predictions.csv` | every predicted box (model, prompt, bbox, score, timing) |
| `per_prompt_metrics.csv` | P/R/F1/accuracy, AP@50, mAP@50:95, FP/FN/dupes per model×prompt (+ all-prompts-combined row) |
| `per_image_metrics.csv` | matching outcome per image |
| `per_class_matches.csv` | matched detections per prompt × GT class |
| `metrics_bars.png`, `confusion_matrix.png` | plots |
| `visualizations/<model>/<prompt>/*.jpg` | optional side-by-side GT/prediction sheets, including labels, confidence (when available), TP/FP/FN status and matched IoU |
| `visualizations/index.html`, `visualization_index.csv` | browsable visual index and a machine-readable listing of all sheets |
| `evaluation_summary.json` | everything above, plus model load statuses and caveats |

## Caveats

- Cosmos and LocateAnything emit no per-box confidence (the backend assigns a
  constant 1.0), so their AP@50 / mAP@50:95 collapse to the single
  precision/recall operating point — compare them on P/R/F1 instead.
- Masks: SAM 3/3.1 produce masks, but neither dataset has mask ground truth,
  so evaluation is box-based (`has_mask` is recorded per prediction).
- Metrics are class-agnostic per prompt (each prompt scored against **all**
  GT boxes); use the confusion matrix and `per_class_matches.csv` to see
  class-level behaviour.
- Model weights are downloaded on first load (SAM 3/3.1 need gated HF access;
  see [Documents/PromptDetect.md](../../../../Documents/PromptDetect.md)).
  LocateAnything needs the `mtsd-la` conda env for its worker.
