<#
.SYNOPSIS
Safe command-line runner (and optional scheduler) for Urban-Vision-Benchmark workflows.

.DESCRIPTION
One interface for running the repository's main workflows - EDA, supervised
notebooks, PromptDetect, GDPR tooling, benchmarks and final-evaluation scripts.
Targets are defined in workflow_targets.json (single source of truth, also
consumed by verify_repository_health.py).

Safety model:
  * -List shows all targets; -DryRun prints exactly what would run, runs nothing;
  * targets flagged "training": true REFUSE to run without -AllowTraining;
  * notebooks are executed to a timestamped copy under Scripts/Automation/executed/
    (papermill if available, else jupyter nbconvert) - the source .ipynb is never
    modified;
  * all stdout/stderr is captured to timestamped logs under Scripts/Automation/logs/;
  * scheduled tasks use the UrbanVisionBenchmark_ prefix and are never overwritten
    silently (-Force required to replace).

.EXAMPLE
powershell -ExecutionPolicy Bypass -File Scripts/Automation/run_urban_workflows.ps1 -List

.EXAMPLE
powershell -ExecutionPolicy Bypass -File Scripts/Automation/run_urban_workflows.ps1 -Target MDWD-EDA -DryRun

.EXAMPLE
powershell -ExecutionPolicy Bypass -File Scripts/Automation/run_urban_workflows.ps1 -Target PromptDetect-BatchEval -Args "--dataset MDWD --split test --prompts 'garbage bag' --models sam3 --max-images 25 --dry-run"

.EXAMPLE
powershell -ExecutionPolicy Bypass -File Scripts/Automation/run_urban_workflows.ps1 -Target MDWD-YOLO26-Notebook -AllowTraining

.EXAMPLE
# every 60 minutes in this console (Ctrl+C to stop)
powershell -ExecutionPolicy Bypass -File Scripts/Automation/run_urban_workflows.ps1 -Target Repo-Health-Check -EveryMinutes 60

.EXAMPLE
# Windows Task Scheduler registration / removal
powershell -ExecutionPolicy Bypass -File Scripts/Automation/run_urban_workflows.ps1 -CreateScheduledTask -Target MDWD-EDA -EveryMinutes 1440
powershell -ExecutionPolicy Bypass -File Scripts/Automation/run_urban_workflows.ps1 -RemoveScheduledTask -Target MDWD-EDA
#>

[CmdletBinding()]
param(
    [string]$Target,
    [switch]$List,
    [switch]$DryRun,
    [switch]$AllowTraining,
    [Alias("Args")][string]$TargetArgs = "",
    [string]$PythonExe = "",            # overrides the target's conda env python
    [string]$CondaEnvsRoot = "$env:USERPROFILE\anaconda3\envs",
    [int]$TimeoutMinutes = 240,          # per-cell timeout for notebook execution
    [int]$EveryMinutes = 0,              # >0: internal repeat loop
    [switch]$CreateScheduledTask,
    [switch]$RemoveScheduledTask,
    [switch]$Force                       # required to replace an existing scheduled task
)

$ErrorActionPreference = "Stop"
$TaskPrefix = "UrbanVisionBenchmark_"

# --- locate repo root robustly (walk up until Datasets/ + Scripts/ exist) -----
function Find-RepoRoot([string]$Start) {
    $dir = Get-Item $Start
    while ($null -ne $dir) {
        if ((Test-Path (Join-Path $dir.FullName "Datasets")) -and
            (Test-Path (Join-Path $dir.FullName "Scripts"))) { return $dir.FullName }
        $dir = $dir.Parent
    }
    throw "Could not locate the repository root from $Start"
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Find-RepoRoot $ScriptDir
$RegistryPath = Join-Path $ScriptDir "workflow_targets.json"
$LogsDir = Join-Path $ScriptDir "logs"
$ExecutedDir = Join-Path $ScriptDir "executed"

if (-not (Test-Path $RegistryPath)) { throw "Target registry not found: $RegistryPath" }
$Registry = (Get-Content $RegistryPath -Raw | ConvertFrom-Json).targets

function Get-TargetNames { $Registry.PSObject.Properties.Name | Sort-Object }

function Show-Targets {
    Write-Host "`nAvailable workflow targets (registry: Scripts/Automation/workflow_targets.json):`n"
    foreach ($name in Get-TargetNames) {
        $t = $Registry.$name
        $flag = if ($t.training) { " [TRAINING - needs -AllowTraining]" } else { "" }
        $exists = if (Test-Path (Join-Path $RepoRoot $t.path)) { "" } else { " [MISSING FILE!]" }
        Write-Host ("  {0,-32} {1,-9} env={2,-12}{3}{4}" -f $name, $t.type, $t.env, $flag, $exists)
        Write-Host ("  {0,-32} {1}" -f "", $t.description) -ForegroundColor DarkGray
    }
    Write-Host ""
}

function Resolve-Python([object]$TargetSpec) {
    if ($PythonExe) { return $PythonExe }
    $candidate = Join-Path (Join-Path $CondaEnvsRoot $TargetSpec.env) "python.exe"
    if (Test-Path $candidate) { return $candidate }
    Write-Warning "Conda env '$($TargetSpec.env)' not found at $candidate - falling back to 'python' on PATH."
    return "python"
}

# Argument splitter that respects single/double quotes in -Args strings.
function Split-Arguments([string]$Text) {
    if (-not $Text.Trim()) { return @() }
    $pattern = '("[^"]*"|''[^'']*''|\S+)'
    ([regex]::Matches($Text, $pattern) | ForEach-Object { $_.Value.Trim('"').Trim("'") })
}

function Invoke-Logged([string]$Exe, [string[]]$Arguments, [string]$LogPath, [string]$WorkDir,
                       [string]$EnvName = "", [bool]$NoUserSite = $false) {
    Write-Host "  exe : $Exe"
    Write-Host "  args: $($Arguments -join ' ')"
    Write-Host "  cwd : $WorkDir"
    if ($EnvName)    { Write-Host "  env : $EnvName" }
    if ($NoUserSite) { Write-Host "  site: PYTHONNOUSERSITE=1 (python -s)" }
    Write-Host "  log : $LogPath"
    "=== $(Get-Date -Format o)" | Out-File $LogPath -Encoding utf8
    "exe : $Exe"                 | Out-File $LogPath -Append -Encoding utf8
    "args: $($Arguments -join ' ')" | Out-File $LogPath -Append -Encoding utf8
    "cwd : $WorkDir"             | Out-File $LogPath -Append -Encoding utf8
    "env : $EnvName"             | Out-File $LogPath -Append -Encoding utf8
    "PYTHONNOUSERSITE: $(if ($NoUserSite) { '1' } else { '(unset)' })" | Out-File $LogPath -Append -Encoding utf8
    Push-Location $WorkDir
    $previousNoUserSite = $env:PYTHONNOUSERSITE
    try {
        if ($NoUserSite) { $env:PYTHONNOUSERSITE = "1" }
        # 2>&1 merges stderr; Tee keeps live console output while logging.
        & $Exe @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
        $code = $LASTEXITCODE
    } finally {
        $env:PYTHONNOUSERSITE = $previousNoUserSite
        Pop-Location
    }
    "=== exit code: $code | $(Get-Date -Format o)" | Out-File $LogPath -Append -Encoding utf8
    if ($code -ne 0) { throw "Target exited with code $code (log: $LogPath)" }
    Write-Host "  OK (exit 0)" -ForegroundColor Green
}

function Invoke-Target([string]$Name) {
    if (-not $Registry.PSObject.Properties[$Name]) {
        Show-Targets
        throw "Unknown target '$Name'."
    }
    $t = $Registry.$Name
    $scriptPath = Join-Path $RepoRoot $t.path
    if (-not (Test-Path $scriptPath)) { throw "Target file missing on disk: $scriptPath" }

    # Training gate ----------------------------------------------------------
    if ($t.training -and -not $AllowTraining -and -not $DryRun) {
        throw ("Target '$Name' is a TRAINING workflow. Re-run with -AllowTraining to confirm " +
               "(or -DryRun to preview). Nothing was executed.")
    }
    if ($t.training -and $AllowTraining) {
        Write-Warning "'$Name' may launch full model training and can run for hours."
    }

    $python = Resolve-Python $t
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    New-Item -ItemType Directory -Force $LogsDir | Out-Null
    $logPath = Join-Path $LogsDir "$Name-$stamp.log"
    $effectiveArgs = if ($TargetArgs) { Split-Arguments $TargetArgs }
                     elseif ($t.PSObject.Properties['defaultArgs']) { @($t.defaultArgs) }
                     else { @() }

    Write-Host "`n=== $Name ($($t.type)) ===" -ForegroundColor Cyan
    if ($DryRun) {
        Write-Host "  DRY RUN - nothing executed." -ForegroundColor Yellow
        Write-Host "  file  : $scriptPath"
        Write-Host "  python: $python"
        if ($t.type -eq "notebook") {
            Write-Host "  would execute notebook to: $ExecutedDir\$Name-$stamp.ipynb (papermill or nbconvert, timeout $TimeoutMinutes min/cell)"
        } else {
            $siteFlag = if ([bool]$t.PSObject.Properties['noUserSite'] -and [bool]$t.noUserSite) { "-s " } else { "" }
            Write-Host "  would run: $python $siteFlag$scriptPath $($effectiveArgs -join ' ')"
            if ($siteFlag) { Write-Host "  with PYTHONNOUSERSITE=1 (user-site packages disabled)" }
        }
        if ($t.training) { Write-Host "  NOTE: training target - requires -AllowTraining for a real run." -ForegroundColor Yellow }
        return
    }

    if ($t.type -eq "python") {
        $noUserSite = [bool]$t.PSObject.Properties['noUserSite'] -and [bool]$t.noUserSite
        $pythonArgs = if ($noUserSite) { @("-s", $scriptPath) } else { @($scriptPath) }
        Invoke-Logged $python ($pythonArgs + $effectiveArgs) $logPath $RepoRoot $t.env $noUserSite
        return
    }

    # Notebook execution ------------------------------------------------------
    New-Item -ItemType Directory -Force $ExecutedDir | Out-Null
    $outNotebook = Join-Path $ExecutedDir "$Name-$stamp.ipynb"
    $timeoutSeconds = $TimeoutMinutes * 60
    & $python -m papermill --version *> $null
    if ($LASTEXITCODE -eq 0) {
        $nbArgs = @("-m", "papermill", $scriptPath, $outNotebook,
                    "--execution-timeout", "$timeoutSeconds", "--cwd", (Split-Path $scriptPath))
        Invoke-Logged $python $nbArgs $logPath $RepoRoot $t.env
    } else {
        & $python -m jupyter nbconvert --version *> $null
        if ($LASTEXITCODE -ne 0) {
            throw ("Neither papermill nor jupyter nbconvert is available in '$python'. " +
                   "Install one: pip install papermill  (or)  pip install nbconvert jupyter")
        }
        $nbArgs = @("-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", $scriptPath,
                    "--output", (Split-Path $outNotebook -Leaf), "--output-dir", $ExecutedDir,
                    "--ExecutePreprocessor.timeout=$timeoutSeconds")
        Invoke-Logged $python $nbArgs $logPath (Split-Path $scriptPath) $t.env
    }
    Write-Host "  executed copy: $outNotebook (source notebook untouched)"
}

# --- Windows Task Scheduler helpers -------------------------------------------
function New-WorkflowTask([string]$Name, [int]$Minutes) {
    if ($Minutes -le 0) { throw "-CreateScheduledTask needs -EveryMinutes > 0." }
    if (-not $Registry.PSObject.Properties[$Name]) { throw "Unknown target '$Name'." }
    $taskName = "$TaskPrefix$Name"
    $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($existing -and -not $Force) {
        throw ("Scheduled task '$taskName' already exists. Inspect with " +
               "'Get-ScheduledTask $taskName | Get-ScheduledTaskInfo'; re-run with -Force to replace, " +
               "or remove it with -RemoveScheduledTask -Target $Name.")
    }
    $scriptPath = Join-Path $ScriptDir "run_urban_workflows.ps1"
    $trainingFlag = if ($AllowTraining) { " -AllowTraining" } else { "" }
    $argString = if ($TargetArgs) { " -Args `"$TargetArgs`"" } else { "" }
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument `
        "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" -Target $Name$trainingFlag$argString"
    $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
        -RepetitionInterval (New-TimeSpan -Minutes $Minutes)
    if ($existing) { Unregister-ScheduledTask -TaskName $taskName -Confirm:$false }
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
        -Description "Urban-Vision-Benchmark workflow '$Name' every $Minutes min (created $(Get-Date -Format s))" | Out-Null
    Write-Host "Registered scheduled task '$taskName' (every $Minutes min)."
    Write-Host "Inspect: Get-ScheduledTask '$taskName' | Get-ScheduledTaskInfo"
    Write-Host "Remove : powershell -File `"$scriptPath`" -RemoveScheduledTask -Target $Name"
}

function Remove-WorkflowTask([string]$Name) {
    $taskName = "$TaskPrefix$Name"
    if (-not (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue)) {
        Write-Host "No scheduled task named '$taskName'."
        return
    }
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Removed scheduled task '$taskName'."
}

# --- entry point ----------------------------------------------------------------
if ($List -or (-not $Target -and -not $RemoveScheduledTask)) { Show-Targets; return }

if ($RemoveScheduledTask) { Remove-WorkflowTask $Target; return }
if ($CreateScheduledTask) { New-WorkflowTask $Target $EveryMinutes; return }

if ($EveryMinutes -gt 0) {
    Write-Host "Running '$Target' every $EveryMinutes minute(s) in this console. Ctrl+C to stop."
    while ($true) {
        try { Invoke-Target $Target }
        catch { Write-Warning "Run failed: $_" }
        Write-Host ("Next run at {0:HH:mm:ss}; sleeping..." -f (Get-Date).AddMinutes($EveryMinutes))
        Start-Sleep -Seconds ($EveryMinutes * 60)
    }
} else {
    Invoke-Target $Target
}
