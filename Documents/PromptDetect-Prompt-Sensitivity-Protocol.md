# PromptDetect Prompt-Sensitivity Protocol

`prompt-sensitivity-v2` is a separate, versioned evaluation protocol
(`Scripts/Other-Scripts/PromptDetect/batch_evaluation/prompt_protocols/prompt_sensitivity_protocol.yaml`)
that measures whether a prompt-based vision model produces stable results when
the *same request* is expressed with semantically equivalent wording. It reuses
the targeted-v2 evaluator unchanged (same thresholds, matching, dataset sources,
final-mode enforcement) and adds sensitivity-specific aggregation, prediction
consistency and reporting. The classic dissertation protocol
(`dissertation-v2`, see [PromptDetect-Dissertation-Protocol.md](PromptDetect-Dissertation-Protocol.md))
is untouched and remains the headline comparison.

## Research motivation

Prompt-based detectors expose natural language as their interface, so an
operational deployment inherits the user's wording. If `stop sign` and
`find all stop signs` return different detections, the model is *prompt
sensitive*: accurate results under one wording do not transfer to another. The
experiment quantifies that instability per model, per target family, so models
can be described on two axes — average performance and wording robustness.

## Three kinds of prompt variation (kept strictly separate)

1. **Controlled paraphrase sensitivity** (this protocol): the same target class
   and semantic scope, different wording only. The main prompt-sensitivity
   result uses *only* these families.
2. **Terminology/synonym sensitivity** (dissertation protocol,
   `synonym-comparison` group): alternative class names that may be genuinely
   ambiguous, e.g. `no entry sign` vs `one way sign`. Not a guaranteed
   paraphrase, therefore never mixed into paraphrase statistics.
3. **Prompt-specificity / granularity** (dissertation protocol, `broad`
   group): narrow class prompts vs superclass prompts such as `traffic sign`.
   These change the target scope, not just the wording.

## Controlled prompt families

Every family has exactly four prompts — one `canonical`, one
`lexical-paraphrase`, one `descriptive-paraphrase`, one `instruction` — with
identical `target_classes` and globally unique IDs. Protocol loading fails
immediately on: a family whose prompts have different target classes, duplicate
prompt text inside a family, zero or multiple canonical prompts, missing
sensitivity metadata, an unknown target class, or a family that does not have
exactly four prompts.

### MTSD (5 families, target class in brackets)

| family | canonical | lexical paraphrase | descriptive paraphrase | instruction |
|---|---|---|---|---|
| `mtsd-pedestrian-crossing` [Pedestrian Crossing] | pedestrian crossing sign | road sign for a pedestrian crossing | sign indicating a pedestrian crossing | find all pedestrian crossing signs |
| `mtsd-stop-sign` [Stop Sign] | stop sign | road sign instructing drivers to stop | traffic sign indicating that vehicles must stop | find all stop signs |
| `mtsd-no-entry` [No Entry (One Way)] | no entry sign | road sign indicating no entry | traffic sign prohibiting vehicles from entering | find all no entry signs |
| `mtsd-roundabout-ahead` [Roundabout Ahead] | roundabout ahead sign | warning sign for a roundabout ahead | traffic sign indicating an upcoming roundabout | find all roundabout ahead signs |
| `mtsd-blind-spot-mirror` [Blind-Spot Mirror (Convex Mirror)] | blind-spot convex mirror | roadside convex mirror for blind spots | convex traffic mirror used to improve road visibility | find all roadside blind-spot mirrors |

### MDWD (4 families)

| family | canonical | lexical paraphrase | descriptive paraphrase | instruction |
|---|---|---|---|---|
| `mdwd-mixed-waste` [Mixed Waste] | black garbage bag | black refuse bag | black bag containing mixed household waste | find all black mixed-waste bags |
| `mdwd-recyclable-material` [Recyclable Material] | grey recycling bag | grey household recycling bag | grey bag containing recyclable household material | find all grey recycling bags |
| `mdwd-organic-waste` [Organic Waste] | white organic-waste bag | white domestic organic-waste bag | white bag containing organic household waste | find all white organic-waste bags |
| `mdwd-orange-cmd` [Orange CMD] | orange garbage bag | orange refuse bag | orange bag containing domestic waste | find all orange waste bags |

Total: 9 families × 4 prompts = **36 prompts**. `Other Waste` is deliberately
excluded (heterogeneous residual category, no clean paraphrase family), and
`one way sign` stays in the synonym comparison only.

## Ground-truth construction

No new annotations are created. For each model × prompt combination, the
canonical test split is loaded and, per image, boxes whose `class_name` is in
the prompt's `target_classes` are positive GT; all other annotated classes are
non-target. Target-negative images are retained — any detection on them is a
false positive. Matching is the existing greedy scheme (score order, best IoU ≥
threshold); duplicates on an already-matched GT box stay false positives; a
false positive overlapping a non-target object at/above the IoU threshold has
that class recorded. The detection cap is derived from the busiest image in
each complete canonical test split (MDWD 28; MTSD 18), and class-agnostic NMS
at IoU 0.50 is applied before that cap. All confidence/IoU defaults, the canonical
MTSD prepared split, QA-only enforcement, manifest and split-hash validation,
the QA-resolved GRP-1…GRP-11 approved group scope, and `--final` mode behave exactly as in targeted-v2.

Model execution is isolated by subprocess. SAM uses one process per prompt;
LocateAnything and Cosmos use fresh 64-image workers. This bounds RAM/VRAM
growth during the much larger sensitivity matrix and makes partial chunks
resumable after interruption.

Because prompts in one family share `target_classes`, all four variants see
identical GT. The dry-run prints, per family, the target classes, number of
prompts, positive/negative image counts and target GT box count; the pipeline
**fails** if these are not identical across the family's four prompts.

## Metrics

Per prompt (unchanged targeted-v2 metrics): TP, FP, FN, duplicates, precision,
recall, F1, accuracy, mean matched IoU, AP50, mAP50–95, target GT box count,
positive/negative image counts, non-target overlap count, mean inference time,
confidence availability, `ap_meaningful`.

Per **dataset × model × family** (`prompt_sensitivity_per_family.csv`): mean,
population standard deviation (`statistics.pstdev`), min, max and range
(max − min) for precision, recall, F1 and mean matched IoU; best/worst prompt
(ID, text, variant type); absolute degradation `max_f1 − min_f1`; relative
degradation `(max_f1 − min_f1) / max_f1` (reported as `0.0` with
`all_variants_failed=true` when `max_f1 == 0`); canonical and instruction F1.

Per variant (`prompt_sensitivity_variant_differences.csv`):
`canonical_f1_difference = variant_f1 − canonical_f1`; negative means the
variant performed worse than the canonical prompt.

Per **dataset × model** (`prompt_sensitivity_per_model.csv`): number of
families, macro mean F1, mean and median within-family F1 std, mean and max F1
range, mean relative degradation, worst family and its degradation, macro
worst-prompt F1, macro canonical F1, macro instruction F1, mean matched-IoU
variability, mean inference time. There is deliberately **no single composite
sensitivity score**; all component statistics stay visible.

## Prediction-consistency analysis

Metric variance alone is insufficient: two prompts can score the same F1 while
detecting *different* objects. For every dataset × model × family × image ×
prompt pair, the two raw prediction sets are compared with greedy IoU matching
at the agreement threshold (default 0.5, `--consistency-iou`):

* `pairwise_box_f1 = 2M / (N_A + N_B)`
* `pairwise_box_jaccard = M / (N_A + N_B − M)`
* mean IoU of matched pairs, unmatched counts per side, `both_empty` flag.

A pair with zero boxes on both sides is fully consistent for that image
(`pairwise_box_f1 = pairwise_box_jaccard = 1.0`, `both_empty=true`). This is
**prediction consistency**, not detection accuracy — no ground truth is
involved, and high consistency can mean consistently right *or* consistently
wrong. Aggregation: per prompt pair, per family, per model.

## Commands

```powershell
cd Scripts/Other-Scripts/PromptDetect/batch_evaluation

# Plan + GT count validation, no models loaded
python run_dissertation_protocol.py `
  --protocol prompt_protocols/prompt_sensitivity_protocol.yaml `
  --dataset both --split test --dry-run

# 5-image smoke test with one model
python run_dissertation_protocol.py `
  --protocol prompt_protocols/prompt_sensitivity_protocol.yaml `
  --dataset MTSD --split test --smoke-test --models sam3

# Final runs (MTSD requires the canonical prepared dataset + QA gate)
python run_dissertation_protocol.py `
  --protocol prompt_protocols/prompt_sensitivity_protocol.yaml `
  --dataset MTSD --split test --final

# Regenerate all summaries/plots/reports from stored combination results
python run_dissertation_protocol.py --reports-only <run-directory>

# Convenience wrapper (same runner, sensitivity protocol pre-selected)
python run_prompt_sensitivity.py --dataset both --split test --dry-run
```

Running the standard evaluation over both datasets with no explicit
`--protocol` (`python run_dissertation_protocol.py --dataset both --split test`)
executes **two stages**: the classic dissertation protocol and then the
prompt-sensitivity protocol, each with its own run directory (sensitivity run
directories carry a `-sensitivity` suffix), outputs, reports and protocol
label (`targeted-v2` vs `prompt-sensitivity-v2`).

Resume behaviour: each model × prompt combination persists atomically and is
reused only when protocol hash, dataset manifest hash, split, thresholds,
model, prompt ID, **prompt text, target classes, sensitivity family and
variant type** all match; a stale prompt definition fails loudly instead of
being silently reused.

## Output files (per sensitivity run directory)

All targeted-v2 outputs, plus:

| File | Contents |
|---|---|
| `prompt_sensitivity_per_prompt.csv` | targeted metrics per prompt with family/variant metadata |
| `prompt_sensitivity_per_family.csv` | spread + degradation statistics per dataset × model × family |
| `prompt_sensitivity_variant_differences.csv` | canonical-vs-variant F1 differences |
| `prompt_sensitivity_per_model.csv` | macro sensitivity summary per model |
| `prompt_pair_consistency_per_image.csv` | pairwise agreement per image and prompt pair |
| `prompt_pair_consistency.csv` | pairwise agreement aggregated per prompt pair |
| `prompt_consistency_per_family.csv` / `prompt_consistency_per_model.csv` | consistency macro averages |
| `prompt_sensitivity_summary.json` | protocol version/hash, manifest hash, thresholds, family definitions, all statistics, methodological notes |
| `prompt_sensitivity_report.md` | human-readable interpretation |
| `sensitivity_*.png/svg`, `consistency_*.png/svg` | figures (F1 by variant per model, family mean±std, degradation by model, worst-family degradation, canonical vs instruction, consistency by model, model × family F1-range heatmap) |

`Scripts/FinalEvaluation/export_dissertation_tables.py` additionally exports
`prompt_sensitivity_results.csv`, `prompt_sensitivity_model_summary.csv` and
`prompt_consistency_results.csv` (with run paths and protocol versions) and a
dedicated "Prompt sensitivity and prediction consistency" section; sensitivity
runs are never merged into the exploratory or targeted rankings.

## Interpretation guidelines

* Judge a model on **mean and variation together**: high mean + large spread =
  accurate but prompt-sensitive; slightly lower mean + small spread = more
  operationally stable.
* Always read within-family variability next to the family mean and worst
  case: low variance with a near-zero mean means every prompt failed, not that
  the model is robust.
* Compare `macro_canonical_f1` with `macro_instruction_f1` (and the variant
  differences CSV) to see whether instruction-style prompts systematically
  differ from canonical naming.
* Similar aggregate F1 between two prompts does **not** imply identical
  detections — check the prediction-consistency tables.
* AP is unsuitable for universal cross-model ranking here because Cosmos and
  LocateAnything emit constant confidence (`ap_meaningful=false`).

## Limitations

* The prompt variants are a controlled sample, not an exhaustive language
  study.
* MDWD prompts retain bag colours because colour is part of the dataset's
  operational class representation (bags are colour-coded by waste stream).
* Low metric variance can occur when every prompt performs poorly; variability
  must always be interpreted alongside mean and worst-case performance.
* High prediction consistency can mean consistently correct or consistently
  incorrect outputs; interpret it together with the GT-based metrics.
* Some backends use constant confidence, making AP unsuitable as the sole
  universal ranking measure.
* Results apply to the tested models, prompts, datasets and prompt protocol
  version (`prompt-sensitivity-v2`).
