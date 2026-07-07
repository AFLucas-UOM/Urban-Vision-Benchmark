# PromptDetect — Batch Evaluation

Automated, ground-truth-scored evaluation of the PromptDetect prompt-based
models (SAM 3, SAM 3.1, Cosmos Reason2 2B/8B/32B, LocateAnything 3B) over
MDWD or MTSD. The existing manual PromptDetect UI ([../app.py](../app.py)) is
untouched — this is an additional mode that reuses the same
`backend.DetectionBackend` and model registry.

## What it does

1. Loads a dataset split into a common ground-truth format:
   - **MDWD** — YOLO layout at `Datasets/MDWD/MDWD-YOLO26` (`train`/`valid`/`test`);
   - **MTSD** — the prepared dataset at `Datasets/MTSD/Prepared/MTSD-YOLO`
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

# Small real run (SAM 3 only, 25 images, with 10 visual overlays):
python Scripts/Other-Scripts/PromptDetect/batch_evaluation/run_batch_eval.py \
    --dataset MDWD --split test --prompts "garbage bag" \
    --models sam3 --max-images 25 --save-visuals 10
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
max-images, thresholds, dry-run toggle, metrics table and result-folder path.

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
| `samples/<model>/*.jpg` | optional GT (blue) vs prediction (red) overlays |
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
