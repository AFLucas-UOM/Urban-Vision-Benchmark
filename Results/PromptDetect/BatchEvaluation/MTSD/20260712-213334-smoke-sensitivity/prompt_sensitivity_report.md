# Prompt-sensitivity report

Protocol: `prompt-sensitivity-v1` (hash `502faa200c0f...`), dataset **MTSD**, split `test`, conf ≥ 0.3, IoU ≥ 0.5, consistency IoU ≥ 0.5.

## What this experiment measures

Prompt sensitivity is the degree to which a prompt-based detector changes its output when the
*same request* is worded differently. Each sensitivity family contains four semantically
equivalent prompts (canonical, lexical paraphrase, descriptive paraphrase, instruction) with
identical `target_classes`, so the target ground truth is exactly the same for all four prompts;
wording is the only experimental variable. Synonym comparisons (e.g. “no entry” vs “one way”)
and broad superclass prompts (e.g. “traffic sign”) change the *meaning* or the *scope* of the
request, not only its wording — they remain in the main dissertation protocol and are deliberately
excluded from these statistics.

Ground truth reuses the canonical test-split annotations unchanged: boxes of the family's target
classes are positive GT; all other annotated objects are non-target. Images without any target box
are retained — every detection on them is a false positive — so precision reflects behaviour on
realistic target-free scenes.

## Family ground truth (identical for all four variants)

| family | target classes | positive images | negative images | target GT boxes |
|---|---|---|---|---|
| mtsd-blind-spot-mirror | Blind-Spot Mirror (Convex Mirror) | 2 | 3 | 2 |
| mtsd-no-entry | No Entry (One Way) | 0 | 5 | 0 |
| mtsd-pedestrian-crossing | Pedestrian Crossing | 1 | 4 | 1 |
| mtsd-roundabout-ahead | Roundabout Ahead | 0 | 5 | 0 |
| mtsd-stop-sign | Stop Sign | 2 | 3 | 2 |

## Model-level sensitivity summary

| model | families | macro mean F1 | mean within-family F1 std | mean F1 range | mean rel. degradation | worst family (rel. degr.) | macro canonical F1 | macro instruction F1 |
|---|---|---|---|---|---|---|---|---|
| SAM 3 | 5 | 0.2686 | 0.1283 | 0.3362 | 0.4278 | mtsd-blind-spot-mirror (0.7647) | 0.4000 | 0.3460 |

Most prompt-sensitive model (largest mean relative F1 degradation): **SAM 3** (0.4278); least sensitive: **SAM 3** (0.4278). A model with a high mean but a large spread is *accurate but prompt-sensitive*; a slightly weaker model with a small spread may be more operationally stable. Judge models on mean **and** variation, never on the mean alone.

## Largest best-to-worst degradations (dataset x model x family)

| model | family | best prompt (variant) | worst prompt (variant) | abs F1 degr. | rel. F1 degr. |
|---|---|---|---|---|---|
| SAM 3 | mtsd-blind-spot-mirror | find all roadside blind-spot mirrors (instruction) | convex traffic mirror used to improve road visibility (descriptive-paraphrase) | 0.7647 | 0.7647 |
| SAM 3 | mtsd-pedestrian-crossing | pedestrian crossing sign (canonical) | road sign for a pedestrian crossing (lexical-paraphrase) | 0.4849 | 0.7273 |
| SAM 3 | mtsd-stop-sign | stop sign (canonical) | road sign instructing drivers to stop (lexical-paraphrase) | 0.4314 | 0.6471 |
| SAM 3 | mtsd-no-entry | no entry sign (canonical) | no entry sign (canonical) | 0.0000 | 0.0000 |
| SAM 3 | mtsd-roundabout-ahead | roundabout ahead sign (canonical) | roundabout ahead sign (canonical) | 0.0000 | 0.0000 |

Families where **every** variant scored F1 = 0 (relative degradation reported as 0.0):

- SAM 3 / mtsd-no-entry
- SAM 3 / mtsd-roundabout-ahead

## Prediction consistency (agreement, not accuracy)

Pairwise box F1 compares the raw predictions of two prompt variants on the same image via
greedy IoU matching — no ground truth is involved. Two prompts can reach similar targeted F1
while detecting *different* objects, so this table must be read together with the GT-based
metrics above. High consistency can equally mean consistently correct or consistently wrong.

| model | families | macro pairwise box F1 | macro pairwise Jaccard | least consistent family |
|---|---|---|---|---|
| SAM 3 | 5 | 0.6716 | 0.6272 | mtsd-blind-spot-mirror (0.4833) |

## Reading guidance

- All prompts in a sensitivity family share identical target ground truth; wording is the only experimental variable.
- Target-negative images are retained; every detection on them counts as a false positive.
- Synonym-comparison and broad/superclass prompts belong to the separate dissertation protocol and are never mixed into these paraphrase-sensitivity statistics.
- Prediction consistency compares the raw prediction sets of two prompt variants (greedy IoU matching); it does not use ground truth and measures output stability, not correctness.
- A prompt pair producing zero boxes on an image is fully consistent for that image (pairwise F1 = Jaccard = 1.0, both_empty=true).
- Relative F1 degradation is (max_f1 - min_f1) / max_f1; when max_f1 == 0 it is reported as 0.0 with all_variants_failed=true.
- Backends with constant confidence make AP collapse to one operating point; AP is reported but never used as the sole cross-model ranking measure.
- Low metric variance can also mean every prompt performed poorly; interpret variability together with mean and worst-case performance.
- Instruction-style prompts are compared with canonical prompts in `prompt_sensitivity_variant_differences.csv` (negative `canonical_f1_difference` = variant worse).
- The prompt variants are a controlled sample of natural wordings, not an exhaustive language study; results apply to the tested models, prompts, datasets and protocol version.

