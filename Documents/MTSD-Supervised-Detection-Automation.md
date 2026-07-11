# MTSD Supervised Detection Automation

The canonical entry point is `Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py` in the `MDWD` conda environment. It replaces duplicated notebook orchestration while leaving the notebooks available for read-only exploration.

## Annotation and preparation policy

Development may use `annotations.group_scope: auto`, but final preparation and
training require `group_scope: explicit`. The configured candidate lock is
GRP-1/2/3/5/6 (3,098 images, 8,372 source boxes). A new Final-QA group outside
that list fails under the default policy; a missing or invalid approved group
also fails. GRP-6 is explicitly listed rather than silently admitted by
discovery, but final methodological approval remains blocked because the latest
audit predates GRP-6.

The machine-readable gate is `config/qa_gate.yaml`. It records the audit path,
timestamp, finding counts, audited/approved scope, resolution status and
approval fields. Final operations cannot override an unresolved gate. A
development-only preparation or smoke run can proceed with
`--acknowledge-open-qa-gate`, and that override is recorded. Backup JSONs are
excluded. Raw XML is never automatic, never overrides QA, and requires both
fallback/confirmation flags; it is incompatible with `--final`.

### Refreshing the gate and approved scope

`config/default.yaml` (`annotations.approved_groups`) and `config/qa_gate.yaml`
must never be hand-edited into disagreement: `update_qa_gate.py` discovers the
current Final-QA groups and the newest annotation-QA audit and rewrites both
files consistently.

```powershell
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/update_qa_gate.py --dry-run
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/update_qa_gate.py --apply
```

Behaviour:

* a group is discovered only if it has exactly one active (non-backup)
  Final-QA JSON that parses, has `images`/`annotations`/`categories`, and uses
  the canonical 12-class vocabulary with no duplicate category IDs or names -
  a vocabulary mismatch or more than one active file for a group is a
  structural error (non-zero exit), not a silent exclusion;
* a QA file that is simply invalid (bad JSON, missing keys) is reported and
  excluded, and the run still succeeds;
* the newest audit is chosen by its own `generated_at` timestamp inside
  `audit_summary.json`, not by directory name or mtime; an audit summary that
  exists but cannot be parsed is a structural error, while no usable audit at
  all is reported without fabricating a path, timestamp, or finding counts;
* `resolution_status` is only `resolved` when a usable audit covers every
  approved group and reports zero unresolved invalid-attribute, duplicate,
  missing-attribute, and reference-problem findings - an unresolved gate is
  reported but never causes a non-zero exit by itself;
* `--apply` never creates backup files, never rewrites `group_scope` or
  `unexpected_group_policy`, and reloads both files afterward to confirm their
  scopes still agree before reporting success; `--dry-run` (the default when
  neither flag is given) writes nothing.

A newly QA'd group is added to `approved_groups`/`approved_scope`
automatically, but it stays blocked from `--final` preparation/training until
a refreshed audit covers it with zero unresolved findings - this script only
refreshes the two config files, it never runs the audit itself.

The v1 split exactly reproduces the notebook: sort within each group, use one seeded RNG (42), shuffle, then round 80/10/10. It is computed once for both variants. The augmented variant adds two deterministic, train-only photometric copies per original (brightness, contrast, colour, mild blur/noise/JPEG/gamma); it never applies geometry. Valid/test membership and hashes must match the unaugmented variant.

```powershell
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --dry-run
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --prepare-only --dataset-variant both --final
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --validate-prepared --strict
```

The final preparation command above is expected to refuse while the QA gate is
unresolved. For isolated development only, replace `--final` with
`--acknowledge-open-qa-gate`. Use `--rebuild` only when intentionally archiving
existing variants to `Prepared/_archive/`.

## Training and evaluation

The dissertation order is YOLO12 n/s/m, YOLO26 n/s/m, YOLO11 n/s/m, RF-DETR n/s/m, then YOLO26l. Defaults are 100 epochs, 640 px, effective batch 32, AdamW, seed 42, patience 10, validation-selected best checkpoint, and one final test evaluation. Ultralytics online augmentation is disabled for both dataset variants, so the clean offline-augmentation ablation is YOLO11m/YOLO12m/YOLO26m, with YOLO26l optional. RF-DETR 1.3.0 retains an internal augmentation pipeline that is not fully controllable through the public API; RF-DETR comparisons are descriptive only and excluded from the headline causal ablation. Never silently reduce resolution or effective batch after OOM.

```powershell
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --smoke-test --wandb-mode disabled --acknowledge-open-qa-gate
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --matrix dissertation --dataset-variant augmented --final
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --matrix aug_ablation --dataset-variant unaugmented
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --resume --skip-completed
python Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py --reports-only
```

W&B project: `MSc-MTSD-SupervisedDetection`. Resume refuses any dataset version,
manifest hash, source mode, variant, model order, or config fingerprint mismatch,
and never reopens a completed run. Native framework metrics remain usable if
unified COCOeval fails. Unified metrics record the evaluator, IoU thresholds and
maxDets; failure is recorded as `unified_eval_status: failed` unless
`--require-unified-eval` explicitly makes it fatal. Framework preprocessing
remains native and is a documented cross-family limitation.
There is no silent metric-library fallback: if `pycocotools` is unavailable,
native metrics are retained and unified evaluation is marked failed until the
same pycocotools protocol can be run in a compatible environment.

Strict validation treats leakage, empty valid/test partitions, structural or
manifest mismatch, stale annotations and configured minimum-support violations
as fatal. Rare classes missing from valid/test are warnings by default and do
not silently trigger a new split.

Final checklist: resolve QA findings; confirm a clean Git state; run dry-run; prepare both variants; pass strict validation; verify checkpoint inventory; run the two-framework smoke test; inspect its records/W&B group; then start the resumable matrix. Real training is deliberately not part of repository implementation.
