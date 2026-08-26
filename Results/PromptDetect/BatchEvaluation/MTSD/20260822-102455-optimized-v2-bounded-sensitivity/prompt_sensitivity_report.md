# Prompt-sensitivity report

Protocol: `prompt-sensitivity-v2` (hash `9559386e86fe...`), dataset **MTSD**, split `test`, conf ≥ 0.3, IoU ≥ 0.5, consistency IoU ≥ 0.5.

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
| mtsd-blind-spot-mirror | Blind-Spot Mirror (Convex Mirror) | 145 | 602 | 160 |
| mtsd-no-entry | No Entry (One Way) | 181 | 566 | 211 |
| mtsd-pedestrian-crossing | Pedestrian Crossing | 115 | 632 | 143 |
| mtsd-roundabout-ahead | Roundabout Ahead | 59 | 688 | 67 |
| mtsd-stop-sign | Stop Sign | 124 | 623 | 130 |

## Model-level sensitivity summary

| model | families | macro mean F1 | mean within-family F1 std | mean F1 range | mean rel. degradation | worst family (rel. degr.) | macro canonical F1 | macro instruction F1 |
|---|---|---|---|---|---|---|---|---|
| Cosmos Reason2 2B | 5 | 0.1926 | 0.0296 | 0.0742 | 0.3472 | mtsd-roundabout-ahead (0.5406) | 0.1945 | 0.2160 |
| Cosmos Reason2 8B | 5 | 0.3054 | 0.0367 | 0.0980 | 0.3059 | mtsd-roundabout-ahead (0.6653) | 0.2705 | 0.3049 |
| LocateAnything 3B | 5 | 0.0865 | 0.0244 | 0.0628 | 0.4713 | mtsd-pedestrian-crossing (0.7136) | 0.1228 | 0.0828 |
| SAM 3 | 5 | 0.2418 | 0.1045 | 0.2736 | 0.6161 | mtsd-blind-spot-mirror (0.7816) | 0.3985 | 0.2359 |
| SAM 3.1 | 5 | 0.2367 | 0.1049 | 0.2732 | 0.6144 | mtsd-blind-spot-mirror (0.7972) | 0.3944 | 0.2274 |

Most prompt-sensitive model (largest mean relative F1 degradation): **SAM 3** (0.6161); least sensitive: **Cosmos Reason2 8B** (0.3059). A model with a high mean but a large spread is *accurate but prompt-sensitive*; a slightly weaker model with a small spread may be more operationally stable. Judge models on mean **and** variation, never on the mean alone.

## Largest best-to-worst degradations (dataset x model x family)

| model | family | best prompt (variant) | worst prompt (variant) | abs F1 degr. | rel. F1 degr. |
|---|---|---|---|---|---|
| SAM 3.1 | mtsd-blind-spot-mirror | blind-spot convex mirror (canonical) | convex traffic mirror used to improve road visibility (descriptive-paraphrase) | 0.4449 | 0.7972 |
| SAM 3 | mtsd-blind-spot-mirror | blind-spot convex mirror (canonical) | convex traffic mirror used to improve road visibility (descriptive-paraphrase) | 0.4280 | 0.7816 |
| LocateAnything 3B | mtsd-pedestrian-crossing | pedestrian crossing sign (canonical) | road sign for a pedestrian crossing (lexical-paraphrase) | 0.1261 | 0.7136 |
| SAM 3 | mtsd-pedestrian-crossing | pedestrian crossing sign (canonical) | road sign for a pedestrian crossing (lexical-paraphrase) | 0.4404 | 0.7078 |
| SAM 3.1 | mtsd-pedestrian-crossing | pedestrian crossing sign (canonical) | road sign for a pedestrian crossing (lexical-paraphrase) | 0.4264 | 0.7020 |

## Prediction consistency (agreement, not accuracy)

Pairwise box F1 compares the raw predictions of two prompt variants on the same image via
greedy IoU matching — no ground truth is involved. Two prompts can reach similar targeted F1
while detecting *different* objects, so this table must be read together with the GT-based
metrics above. High consistency can equally mean consistently correct or consistently wrong.

| model | families | macro pairwise box F1 | macro pairwise Jaccard | least consistent family |
|---|---|---|---|---|
| Cosmos Reason2 2B | 5 | 0.6439 | 0.6425 | mtsd-blind-spot-mirror (0.4996) |
| Cosmos Reason2 8B | 5 | 0.6935 | 0.6916 | mtsd-blind-spot-mirror (0.6259) |
| LocateAnything 3B | 5 | 0.5515 | 0.5018 | mtsd-roundabout-ahead (0.5317) |
| SAM 3 | 5 | 0.6164 | 0.5772 | mtsd-blind-spot-mirror (0.4330) |
| SAM 3.1 | 5 | 0.6117 | 0.5709 | mtsd-blind-spot-mirror (0.4168) |

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

