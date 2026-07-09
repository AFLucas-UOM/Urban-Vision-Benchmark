# FinalEvaluation

Dissertation-preparation tooling: dataset integrity sign-off, split-leakage
sensitivity analysis, results consolidation, and qualitative error analysis.
Everything here is **read-only
with respect to datasets, runs and previous results** — outputs go to
`Documents/Final-Reports/`, `Documents/Final-Tables/` and
`Documents/Final-Figures/` (timestamped where regeneration matters).

| Script | Purpose | Safe by default? |
| --- | --- | --- |
| `final_dataset_integrity_check.py` | PASS/WARNING/FAIL integrity report over MDWD + MTSD | yes (read-only; `--dry-run` prints only) |
| `mdwd_leakage_sensitivity.py` | independently measures the effect of MDWD cross-split augmented-source leakage | yes for the audit; `--run-inference` explicitly re-evaluates existing checkpoints without changing datasets |
| `export_dissertation_tables.py` | consolidates existing result CSVs/JSONs into final tables | yes (reads results, writes timestamped tables; `--overwrite` opts into fixed names) |
| `failure_case_sampler.py` | annotated FP/FN/duplicate/success image samples | yes, **unless** you pass `--model`, which explicitly opts into running one checkpoint over a small sample |

## Dataset integrity

```bash
python Scripts/FinalEvaluation/final_dataset_integrity_check.py --dataset all
python Scripts/FinalEvaluation/final_dataset_integrity_check.py --dataset MDWD --max-images 500 --dry-run
```

Reuses `mdwd_eda` (MDWD + prepared MTSD detection set) and the
MTSD-AnnotationQA scanner (QA attributes/duplicates/references), so this
report always agrees with the working tools. Outputs:
`Documents/Final-Reports/dataset_integrity_report.{md,json}` +
`dataset_integrity_issues.csv`. FAIL = would corrupt training/evaluation;
WARNING = acknowledge in the dissertation (duplicates, leakage, drop-values).

## MDWD split-leakage sensitivity

The completed analysis independently re-derived the 30 leaked MDWD source
identities and found a negligible effect on the published YOLO metrics (maximum
absolute change: 0.48 percentage points mAP50-95). The consolidated YOLO tables
need no adjustment; retain the issue as a limitations-section caveat. RF-DETR is
outside this Ultralytics-based sensitivity analysis. Full results and
recommended wording: [Documents/MDWDLeakageSensitivity.md](../../Documents/MDWDLeakageSensitivity.md).

```bash
python Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py --self-test
python Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py
python Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py --run-inference --models best-per-family
```

The first two commands are audit-only. The inference option reads existing
checkpoints and writes reports under `Documents/Final-Reports/MDWD-Leakage-Analysis/`;
it does not alter dataset files, training runs, or headline result tables.

## Dissertation tables

```bash
python Scripts/FinalEvaluation/export_dissertation_tables.py              # -> Documents/Final-Tables/<timestamp>/
python Scripts/FinalEvaluation/export_dissertation_tables.py --overwrite  # -> fixed names in Documents/Final-Tables/
```

Collects: consolidated MDWD **YOLO** detection summaries, MTSD detection
(pending until trained), attribute-classification comparison, PromptDetect
batch-eval summaries, and inference-speed benchmarks. RF-DETR nano currently
has a run log/checkpoint but no normalised summary input, so it is deliberately
omitted rather than parsed heuristically. Missing sections are marked
**pending** with the command that produces them; every row records its source
file. No metric is recomputed or invented.

## Failure-case sampling

```bash
# From an existing PromptDetect batch-eval run (no inference):
python Scripts/FinalEvaluation/failure_case_sampler.py --dataset PromptDetect --latest --max-samples 25

# From stored YOLO predictions (no inference):
python Scripts/FinalEvaluation/failure_case_sampler.py --dataset MDWD --pred-labels <dir-of-txt> --max-samples 25

# EXPLICIT opt-in: run one trained checkpoint over a seeded sample (light):
python Scripts/FinalEvaluation/failure_case_sampler.py --dataset MDWD --model best --max-samples 25
python Scripts/FinalEvaluation/failure_case_sampler.py --dataset MTSD --model yolo26m --max-samples 25   # after MTSD training
```

Categories: false positives, false negatives (+ small-object subset ≤32²px),
duplicates, low-confidence TPs, clean successes. Outputs annotated JPEGs per
category plus `error_case_index.csv` under
`Documents/Final-Figures/ErrorAnalysis/<DATASET>/<timestamp>-<model>/`
(box colours: GT blue, TP green, low-conf TP violet, FP red, duplicate orange,
missed amber). Occlusion/clutter failures are *not* auto-identified (no such
labels exist); select them manually from the FN/FP samples.

If no prediction source exists the sampler says so and exits without writing.

## What may run heavy work

Nothing here trains. The only inference is `failure_case_sampler.py --model ...`
(one checkpoint, ≤ `--max-images` images, explicit) and the explicit
`mdwd_leakage_sensitivity.py --run-inference ...` re-evaluation of existing
checkpoints. Run the tools via the workflow runner too: targets
`Final-Dataset-Integrity`, `MDWD-Leakage-Sensitivity`,
`Export-Dissertation-Tables`, `Failure-Case-Sampler`, `Repo-Health-Check`
(see [Scripts/Automation/README.md](../Automation/README.md)).
