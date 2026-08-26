# Prompt-sensitivity report

Protocol: `prompt-sensitivity-v2` (hash `9559386e86fe...`), dataset **MDWD**, split `test`, conf ≥ 0.3, IoU ≥ 0.5, consistency IoU ≥ 0.5.

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
| mdwd-mixed-waste | Mixed Waste | 126 | 243 | 334 |
| mdwd-orange-cmd | Orange CMD | 37 | 332 | 78 |
| mdwd-organic-waste | Organic Waste | 107 | 262 | 178 |
| mdwd-recyclable-material | Recyclable Material | 114 | 255 | 238 |

## Model-level sensitivity summary

| model | families | macro mean F1 | mean within-family F1 std | mean F1 range | mean rel. degradation | worst family (rel. degr.) | macro canonical F1 | macro instruction F1 |
|---|---|---|---|---|---|---|---|---|
| Cosmos Reason2 2B | 4 | 0.2111 | 0.0735 | 0.2037 | 0.5889 | mdwd-recyclable-material (0.8022) | 0.1994 | 0.3258 |
| Cosmos Reason2 8B | 4 | 0.3633 | 0.0274 | 0.0713 | 0.1893 | mdwd-orange-cmd (0.2972) | 0.3707 | 0.3576 |
| LocateAnything 3B | 4 | 0.3030 | 0.0510 | 0.1245 | 0.3432 | mdwd-orange-cmd (0.4491) | 0.3394 | 0.2469 |
| SAM 3 | 4 | 0.5211 | 0.1235 | 0.2927 | 0.4954 | mdwd-recyclable-material (1.0000) | 0.5676 | 0.3652 |
| SAM 3.1 | 4 | 0.5262 | 0.1204 | 0.2942 | 0.4903 | mdwd-recyclable-material (0.9601) | 0.5648 | 0.3811 |

Most prompt-sensitive model (largest mean relative F1 degradation): **Cosmos Reason2 2B** (0.5889); least sensitive: **Cosmos Reason2 8B** (0.1893). A model with a high mean but a large spread is *accurate but prompt-sensitive*; a slightly weaker model with a small spread may be more operationally stable. Judge models on mean **and** variation, never on the mean alone.

## Largest best-to-worst degradations (dataset x model x family)

| model | family | best prompt (variant) | worst prompt (variant) | abs F1 degr. | rel. F1 degr. |
|---|---|---|---|---|---|
| SAM 3 | mdwd-recyclable-material | grey household recycling bag (lexical-paraphrase) | find all grey recycling bags (instruction) | 0.4021 | 1.0000 |
| SAM 3.1 | mdwd-recyclable-material | grey bag containing recyclable household material (descriptive-paraphrase) | find all grey recycling bags (instruction) | 0.3944 | 0.9601 |
| Cosmos Reason2 2B | mdwd-recyclable-material | find all grey recycling bags (instruction) | grey bag containing recyclable household material (descriptive-paraphrase) | 0.3083 | 0.8022 |
| Cosmos Reason2 2B | mdwd-organic-waste | find all white organic-waste bags (instruction) | white bag containing organic household waste (descriptive-paraphrase) | 0.2113 | 0.6504 |
| SAM 3 | mdwd-orange-cmd | orange garbage bag (canonical) | find all orange waste bags (instruction) | 0.5086 | 0.6119 |

## Prediction consistency (agreement, not accuracy)

Pairwise box F1 compares the raw predictions of two prompt variants on the same image via
greedy IoU matching — no ground truth is involved. Two prompts can reach similar targeted F1
while detecting *different* objects, so this table must be read together with the GT-based
metrics above. High consistency can equally mean consistently correct or consistently wrong.

| model | families | macro pairwise box F1 | macro pairwise Jaccard | least consistent family |
|---|---|---|---|---|
| Cosmos Reason2 2B | 4 | 0.5515 | 0.5467 | mdwd-orange-cmd (0.5430) |
| Cosmos Reason2 8B | 4 | 0.8155 | 0.8013 | mdwd-orange-cmd (0.7296) |
| LocateAnything 3B | 4 | 0.6906 | 0.6400 | mdwd-organic-waste (0.6266) |
| SAM 3 | 4 | 0.7993 | 0.7949 | mdwd-recyclable-material (0.6733) |
| SAM 3.1 | 4 | 0.7912 | 0.7863 | mdwd-recyclable-material (0.6543) |

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

