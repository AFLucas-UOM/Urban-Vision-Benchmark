$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..\..')).Path
$Python = 'C:\Users\fridge\anaconda3\envs\mtsd-base\python.exe'
$Runner = Join-Path $RepoRoot 'Scripts\Other-Scripts\PromptDetect\batch_evaluation\run_dissertation_protocol.py'
$TargetProtocol = Join-Path $RepoRoot 'Scripts\Other-Scripts\PromptDetect\batch_evaluation\prompt_protocols\dissertation_protocol.yaml'
$SensitivityProtocol = Join-Path $RepoRoot 'Scripts\Other-Scripts\PromptDetect\batch_evaluation\prompt_protocols\prompt_sensitivity_protocol.yaml'
$TargetRun = $PSScriptRoot
$LogDir = Join-Path $PSScriptRoot 'launcher_logs\bounded-model-specific-v2'
$StatusPath = Join-Path $LogDir 'queue_status.json'
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

function Write-QueueStatus {
    param([string]$Status, [string]$Stage, [string]$Detail = '')
    [ordered]@{
        status = $Status
        stage = $Stage
        detail = $Detail
        updated_at = [DateTimeOffset]::UtcNow.ToString('o')
        process_id = $PID
        execution_revision = 'bounded-model-specific-2b2560-8b1536-chunk8-v2'
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding utf8
}

function Invoke-EvaluationStage {
    param([string]$Name, [string[]]$Arguments)
    $LogPath = Join-Path $LogDir ($Name + '.log')
    Write-QueueStatus -Status 'running' -Stage $Name
    "=== $Name started $([DateTimeOffset]::UtcNow.ToString('o')) ===" | Tee-Object -FilePath $LogPath
    & $Python $Runner @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "$Name exited with code $LASTEXITCODE"
    }
    "=== $Name completed $([DateTimeOffset]::UtcNow.ToString('o')) ===" | Tee-Object -FilePath $LogPath -Append
}

try {
    Set-Location -LiteralPath $RepoRoot
    Invoke-EvaluationStage -Name '01-resume-mtsd-targeted' -Arguments @(
        '--protocol', $TargetProtocol,
        '--dataset', 'MTSD',
        '--split', 'test',
        '--final',
        '--resume', $TargetRun,
        '--skip-completed',
        '--wandb-mode', 'online',
        '--save-visualizations', '10'
    )
    Invoke-EvaluationStage -Name '02-mdwd-prompt-sensitivity' -Arguments @(
        '--protocol', $SensitivityProtocol,
        '--dataset', 'MDWD',
        '--split', 'test',
        '--final',
        '--run-label', 'optimized-v2-bounded',
        '--wandb-mode', 'online',
        '--save-visualizations', '10'
    )
    Invoke-EvaluationStage -Name '03-mtsd-prompt-sensitivity' -Arguments @(
        '--protocol', $SensitivityProtocol,
        '--dataset', 'MTSD',
        '--split', 'test',
        '--final',
        '--run-label', 'optimized-v2-bounded',
        '--wandb-mode', 'online',
        '--save-visualizations', '10'
    )
    Write-QueueStatus -Status 'completed' -Stage 'all'
}
catch {
    Write-QueueStatus -Status 'failed' -Stage 'queue' -Detail $_.Exception.Message
    throw
}
