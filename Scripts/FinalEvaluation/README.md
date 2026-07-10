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
| `robustness_slice_analysis.py` | per-slice performance (object size, brightness, blur, density, class, attributes, prompts) from **stored** predictions | yes (never runs models; pending inputs are reported, not recomputed) |
| `annotation_effort_report.py` | dataset/annotation/QA/training/operational effort per paradigm | yes (reads existing artefacts; manual hours only from `config/annotation_effort_manual.yaml`, never estimated) |
| `bootstrap_uncertainty.py` | seeded percentile-bootstrap CIs + paired comparisons over stored sample-level outcomes | yes (never runs inference; missing sample-level inputs -> pending report) |
| `build_dissertation_dashboard.py` | static offline HTML evidence dashboard (viewer only) | yes (renders existing evidence; missing sections show PENDING) |
| `evidence_lib.py` | shared read-only helpers for the four tools above (loaders, matching, bootstrap core, figure export) | library, no CLI |

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

## Robustness slice analysis

```bash
python Scripts/FinalEvaluation/robustness_slice_analysis.py --self-test
python Scripts/FinalEvaluation/robustness_slice_analysis.py --dry-run
python Scripts/FinalEvaluation/robustness_slice_analysis.py                 # all available evidence
python Scripts/FinalEvaluation/robustness_slice_analysis.py --dataset MDWD --task detection
```

Slices performance by object size (COCO thresholds, 640x640 evaluation
pixels), brightness/contrast/sharpness (dataset-quantile terciles, recorded in
`robustness_slice_config.json`), object density, object position, class and
class frequency; MTSD attribute heads/classes (historical GRP-1..3 snapshot);
PromptDetect model x prompt (pilot runs labelled as such). Inputs: the stored
MDWD predictions from the leakage analysis, attribute `test_metrics.json`
files, and PromptDetect batch runs — **no inference is ever run**; anything
unavailable lands in `insufficient_or_missing_inputs.md` with the command that
produces it. Slices under `--min-support` stay in the CSV marked
`insufficient_support` and are never highlighted or plotted. Outputs:
`Documents/Final-Reports/Robustness-Slices/<stamp>/` +
`Documents/Final-Figures/Robustness-Slices/<stamp>/` (PNG/PDF/SVG);
`--overwrite` targets `latest/` instead.

## Annotation / operational effort

```bash
python Scripts/FinalEvaluation/annotation_effort_report.py --self-test
python Scripts/FinalEvaluation/annotation_effort_report.py --dry-run --skip-disk-size
python Scripts/FinalEvaluation/annotation_effort_report.py
```

Derives dataset/annotation/QA/training/prompt-evaluation facts from existing
artefacts (EDA summary, QA JSONs, audit reports, `pipeline_complete.json`
files, `comparison.csv`, run configs) and merges the hand-edited
`config/annotation_effort_manual.yaml` (annotation hours, costs). Missing
manual values are reported as *not recorded* — never estimated — and listed in
`missing_manual_values.md`. Produces the paradigm-level
`operational_comparison.csv` (raw dimensions only; deliberately **no composite
score**). Outputs under `Documents/Final-Reports/Annotation-Effort/<stamp>/` +
figures.

## Bootstrap uncertainty

```bash
python Scripts/FinalEvaluation/bootstrap_uncertainty.py --self-test
python Scripts/FinalEvaluation/bootstrap_uncertainty.py --dry-run
python Scripts/FinalEvaluation/bootstrap_uncertainty.py
python Scripts/FinalEvaluation/bootstrap_uncertainty.py --dataset MDWD `
    --model-a yolo26l@YOLO26-EUVIP --model-b yolo12m@YOLO12-EUVIP
```

Seeded percentile bootstrap (default 2,000 resamples, 95% intervals) over
stored sample-level outcomes: image-level for MDWD detection (stored
leakage-analysis predictions) and PromptDetect runs; test-crop level for
attribute heads (samples reconstructed exactly from the stored confusion
matrices — the documented iid-crop assumption). Paired comparisons resample
identical unit indices on both sides and report the CI of the difference plus
the fraction of resamples favouring A; nothing is ever labelled
"statistically significant". Cross-head attribute CIs, paired variant
comparisons and detection mAP intervals need inputs that do not exist yet and
are listed in `missing_inputs.md`. Outputs under
`Documents/Final-Reports/Statistical-Uncertainty/<stamp>/` + figures.

## Dissertation evidence dashboard

```bash
python Scripts/FinalEvaluation/build_dissertation_dashboard.py --self-test
python Scripts/FinalEvaluation/build_dissertation_dashboard.py               # timestamped build
python Scripts/FinalEvaluation/build_dissertation_dashboard.py --overwrite   # -> Dissertation-Dashboard/latest/
```

Static, fully offline HTML site (plain HTML/CSS/vanilla JS, no CDN, no
telemetry, no backend) indexing all dissertation evidence: research-question
matrix (`config/research_questions.yaml`), dataset/integrity/leakage status,
experiment cards with status badges (final / historical snapshot / pilot /
smoke test / pending / excluded) and headline-use permissions, final tables,
figures, robustness slices, bootstrap CIs, effort comparison, existing
failure-case galleries, auto-generated limitations, and a machine-readable
`data/evidence_index.json` with per-item provenance (source file, generating
script, commit SHA). Missing evidence renders as PENDING. All links are
repository-relative — open `index.html` directly or via a local static server.

## What may run heavy work

Nothing here trains. The only inference is `failure_case_sampler.py --model ...`
(one checkpoint, ≤ `--max-images` images, explicit) and the explicit
`mdwd_leakage_sensitivity.py --run-inference ...` re-evaluation of existing
checkpoints. The robustness/bootstrap/effort/dashboard tools never load a
model. Run the tools via the workflow runner too: targets
`Final-Dataset-Integrity`, `MDWD-Leakage-Sensitivity`,
`Export-Dissertation-Tables`, `Failure-Case-Sampler`, `Robustness-Slices`,
`Annotation-Effort`, `Bootstrap-Uncertainty`, `Dissertation-Dashboard`,
`Repo-Health-Check` (see [Scripts/Automation/README.md](../Automation/README.md)).
