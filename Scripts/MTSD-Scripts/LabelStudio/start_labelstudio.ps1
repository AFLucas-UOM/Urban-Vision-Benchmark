param(
    [string]$Group = "GRP-1",
    [int]$Port = 8080
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$DatasetDir = Join-Path $RepoRoot "Datasets\MTSD\$Group"
$OutputDir = Join-Path $RepoRoot "Datasets\MTSD\$Group\labelstudio_output"
$DataDir = Join-Path $OutputDir "LabelStudioData"
$EnvPath = Join-Path $DataDir ".env"

if (-not (Test-Path $DatasetDir)) {
    throw "Could not find dataset directory for $Group at $DatasetDir"
}

if (-not (Test-Path $DataDir)) {
    New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
}

$HostUrl = "http://localhost:$Port"

$ListenerProcessIds = netstat -ano |
    Select-String ":$Port" |
    Select-String "LISTENING" |
    ForEach-Object { ($_ -split "\s+")[-1] } |
    Where-Object { $_ -match "^\d+$" -and [int]$_ -ne 0 } |
    Sort-Object -Unique

foreach ($ProcessId in $ListenerProcessIds) {
    $Process = Get-Process -Id ([int]$ProcessId) -ErrorAction SilentlyContinue
    if ($Process) {
        Write-Host "Stopping existing process on port ${Port}: PID $ProcessId ($($Process.ProcessName))"
        Stop-Process -Id ([int]$ProcessId) -Force
    }
}

if ($ListenerProcessIds.Count -gt 0) {
    Start-Sleep -Seconds 2
}

$env:DEBUG = "false"
$env:LOCAL_FILES_SERVING_ENABLED = "true"
$env:LOCAL_FILES_DOCUMENT_ROOT = $RepoRoot
$env:LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN = "true"
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = $RepoRoot

$ExistingEnv = @()
if (Test-Path $EnvPath) {
    $ExistingEnv = Get-Content $EnvPath | Where-Object {
        $_ -notmatch "^(LOCAL_FILES_SERVING_ENABLED|LOCAL_FILES_DOCUMENT_ROOT|LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED|LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT)="
    }
}
$ExistingEnv += "LOCAL_FILES_SERVING_ENABLED=true"
$ExistingEnv += "LOCAL_FILES_DOCUMENT_ROOT=$RepoRoot"
$ExistingEnv += "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true"
$ExistingEnv += "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$RepoRoot"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines($EnvPath, $ExistingEnv, $Utf8NoBom)

$LabelStudio = Get-Command label-studio -ErrorAction SilentlyContinue

Write-Host "Starting Label Studio for $Group..."
Write-Host "URL: $HostUrl"
Write-Host "Username: sample@example.com"
Write-Host "Password: SampleAnnotations123!"
Write-Host ""
Write-Host "Keep this window open while using Label Studio. Press Ctrl+C to stop it."
Write-Host ""

Push-Location $RepoRoot
try {
    if ($LabelStudio) {
        & label-studio start `
            --no-browser `
            --data-dir $DataDir `
            --host $HostUrl `
            --port $Port `
            --username sample@example.com `
            --password "SampleAnnotations123!" `
            --user-token sample-annotations-token `
            --enable-legacy-api-token
    }
    else {
        & uvx --python 3.12 label-studio start `
            --no-browser `
            --data-dir $DataDir `
            --host $HostUrl `
            --port $Port `
            --username sample@example.com `
            --password "SampleAnnotations123!" `
            --user-token sample-annotations-token `
            --enable-legacy-api-token
    }
}
finally {
    Pop-Location
}
