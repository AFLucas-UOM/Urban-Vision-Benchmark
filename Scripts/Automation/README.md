# Automation

One safe command-line interface for running the repository's workflows, plus
the repository health check. Targets live in
[workflow_targets.json](workflow_targets.json) — a single registry consumed by
both the runner and `verify_repository_health.py`.

## `run_urban_workflows.ps1`

```powershell
# What can I run?
powershell -ExecutionPolicy Bypass -File Scripts/Automation/run_urban_workflows.ps1 -List

# Preview anything (nothing executes)
powershell -ExecutionPolicy Bypass -File Scripts/Automation/run_urban_workflows.ps1 -Target MDWD-EDA -DryRun

# Run a safe workflow
powershell -ExecutionPolicy Bypass -File Scripts/Automation/run_urban_workflows.ps1 -Target MDWD-EDA

# Custom arguments (quoted; passed to the underlying script)
powershell -ExecutionPolicy Bypass -File Scripts/Automation/run_urban_workflows.ps1 `
    -Target PromptDetect-BatchEval `
    -Args "--dataset MDWD --split test --prompts 'garbage bag' --models sam3 --max-images 25 --dry-run"

# Training notebooks REFUSE to run without the explicit flag
powershell -ExecutionPolicy Bypass -File Scripts/Automation/run_urban_workflows.ps1 -Target MDWD-YOLO26-Notebook -AllowTraining
```

### Behaviour and safety

- **Logs**: every run streams to the console *and* to
  `Scripts/Automation/logs/<Target>-<timestamp>.log`.
- **Notebooks** are executed headlessly (papermill if installed in the target
  env, else `jupyter nbconvert --execute`) into
  `Scripts/Automation/executed/<Target>-<timestamp>.ipynb` — the **source
  notebook is never modified**. Per-cell timeout via `-TimeoutMinutes`
  (default 240).
- **Training gate**: targets flagged `"training": true` (supervised notebooks,
  the MTSD matrices and the attribute size-ablation profiles/smoke) refuse to run without `-AllowTraining`; `-DryRun`
  always works. Note the MTSD training notebooks additionally ship with
  `RUN_TRAINING=False` inside — executing them without flipping that cell runs
  setup only.
- **Environments**: each target names its conda env (resolved under
  `%USERPROFILE%\anaconda3\envs`); override with `-PythonExe <path>` or
  `-CondaEnvsRoot <dir>`.
- **User-site isolation**: targets flagged `"noUserSite": true` (all
  AttributeClassification targets) run as `python -s` with
  `PYTHONNOUSERSITE=1`, as the `mtsd-attrcls` environment requires; the flag
  is restored afterwards.
- **Secrets**: the runner never prints `.env` contents; workflows read
  `WANDB_API_KEY` themselves from the repo-root `.env`.
- Long-running local web applications (`PromptDetect-App`, `AttrCls-Compare-UI`,
  `MTSD-AnnotationQA-Review-UI`) run in the foreground —
  stop them with Ctrl+C in their console. For managed start/reopen/restart/stop
  with readiness polling, prefer the UVB launcher (`python launch_uvb.py`).
- **Intentionally excluded**: MDWD YOLO11 training (completed historical
  EUVIP runs are preserved under `Results/MDWD-Runs/YOLO11-EUVIP`; no runnable
  notebook remains) — MTSD YOLO11 is covered by the canonical 13-model matrix.
  Low-level helpers (annotation-path migration, reset scripts, one-off
  fixers) are deliberately not registry targets.

### Periodic execution

Both options are implemented:

```powershell
# Option A - internal loop in the current console (Ctrl+C stops it)
powershell -ExecutionPolicy Bypass -File Scripts/Automation/run_urban_workflows.ps1 -Target Repo-Health-Check -EveryMinutes 60

# Option B - Windows Task Scheduler (task name prefix: UrbanVisionBenchmark_)
powershell -ExecutionPolicy Bypass -File Scripts/Automation/run_urban_workflows.ps1 -CreateScheduledTask -Target MDWD-EDA -EveryMinutes 1440
Get-ScheduledTask UrbanVisionBenchmark_MDWD-EDA | Get-ScheduledTaskInfo   # inspect
powershell -ExecutionPolicy Bypass -File Scripts/Automation/run_urban_workflows.ps1 -RemoveScheduledTask -Target MDWD-EDA
```

Existing scheduled tasks are **never replaced silently** — re-registering an
existing task requires `-Force`. Don't schedule training targets.

### Adding a target

Edit `workflow_targets.json`: `type` (`python`/`notebook`), repo-relative
`path`, `env`, optional `defaultArgs`, `training` flag, optional `noUserSite`
flag, `description`. The health check verifies the path exists, and
`tests/test_workflow_registry.py` enforces registry consistency (required
fields, known types/envs, path/extension match, gating conventions, coverage
of the principal workflows):

```powershell
& "$env:USERPROFILE\anaconda3\envs\MDWD\python.exe" -m pytest Scripts/Automation/tests -q
```

## `verify_repository_health.py`

Read-only structure/path/code verification (expected folders, README presence,
notebook JSON validity, Python compilation, old-layout path patterns,
cross-dataset contamination in supervised notebooks, hardcoded W&B keys,
`.env` git-ignore status, broken Markdown links, runner-target existence).

```bash
python Scripts/Automation/verify_repository_health.py            # writes reports
python Scripts/Automation/verify_repository_health.py --dry-run  # console only
```

Reports: `Documents/Final-Reports/repository_health_check.md` / `.json`
(fixed names — a living "current health" snapshot with its timestamp inside).
Exit code 1 on FAIL-level findings, so it can gate scheduled jobs.

### Local pre-commit / pre-push check (why there is no CI workflow)

A GitHub Actions workflow for this check was considered and deliberately
**not** added: a fresh CI checkout contains only git-tracked files, and five
directories the health check requires exist only locally as git-ignored
data/weights — `Datasets/MTSD/GRP-1/`, `Models/`, `Results/MTSD-Results/`,
`Results/MTSD-Runs/` and
`Scripts/MTSD-Scripts/AttributeClassification/outputs/checkpoints/` — so the
`expected_folders` and `result_folders_preserved` checks would FAIL on every
CI run by design. Run the check locally before committing or pushing instead
(standard library only, any Python ≥ 3.9, no packages to install):

```bash
python Scripts/Automation/verify_repository_health.py --dry-run
```

Exit code 0 = PASS/WARN, 1 = FAIL; `--dry-run` prints to the console and
writes nothing.

Optionally enforce it as a git pre-push hook (hooks are local-only and never
committed). Create `.git/hooks/pre-push` containing:

```sh
#!/bin/sh
python "$(git rev-parse --show-toplevel)/Scripts/Automation/verify_repository_health.py" --dry-run || {
    echo "Repository health check FAILED - push aborted (bypass with: git push --no-verify)."
    exit 1
}
```

Git for Windows runs hooks through its bundled `sh`, so no `chmod` is needed.
If the check reports a FAIL the hook blocks the push; fix the finding, or
bypass deliberately with `git push --no-verify`. (As of 2026-07-10 the check
is fully green — about 2.5 s on the system Python.)
