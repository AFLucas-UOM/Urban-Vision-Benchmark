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
- **Training gate**: targets flagged `"training": true` (all six supervised
  notebooks) refuse to run without `-AllowTraining`; `-DryRun` always works.
  Note the MTSD training notebooks additionally ship with `RUN_TRAINING=False`
  inside — executing them without flipping that cell runs setup only.
- **Environments**: each target names its conda env (resolved under
  `%USERPROFILE%\anaconda3\envs`); override with `-PythonExe <path>` or
  `-CondaEnvsRoot <dir>`.
- **Secrets**: the runner never prints `.env` contents; workflows read
  `WANDB_API_KEY` themselves from the repo-root `.env`.
- `PromptDetect-App` starts a long-running Gradio server — run it in its own
  console and stop with Ctrl+C.

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
`path`, `env`, optional `defaultArgs`, `training` flag, `description`. The
health check will verify the path exists.

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
