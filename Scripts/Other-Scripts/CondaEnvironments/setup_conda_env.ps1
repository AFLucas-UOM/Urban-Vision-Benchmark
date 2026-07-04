<#
.SYNOPSIS
Creates (or updates) the project's Conda environments on Windows PowerShell.

.DESCRIPTION
Reproducible environment setup for the Urban-Vision-Benchmark repository. For the chosen
environment the script:
  1. creates the env from its environment-<name>.yml (or updates + prunes it
     if it already exists);
  2. installs the platform-appropriate PyTorch build (CUDA by default,
     CPU wheels with -Cpu);
  3. for mtsd-base only, installs the Meta sam3 package (requires torch,
     hence this ordering).

.PARAMETER Name
Environment to set up: MDWD, mtsd-attrcls, mtsd-base, mtsd-la, or all.

.PARAMETER Cpu
Install CPU-only PyTorch wheels instead of CUDA builds (machines without an
NVIDIA GPU).

.EXAMPLE
.\setup_conda_env.ps1 -Name mtsd-base

.EXAMPLE
.\setup_conda_env.ps1 -Name all -Cpu
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('MDWD', 'mtsd-attrcls', 'mtsd-base', 'mtsd-la', 'all')]
    [string]$Name,

    [switch]$Cpu
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

# ---- Locate conda ----------------------------------------------------------
$conda = (Get-Command conda -ErrorAction SilentlyContinue).Source
if (-not $conda) {
    foreach ($candidate in @(
            "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
            "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
            "$env:ProgramData\anaconda3\Scripts\conda.exe",
            "$env:ProgramData\miniconda3\Scripts\conda.exe")) {
        if (Test-Path $candidate) { $conda = $candidate; break }
    }
}
if (-not $conda) { throw 'conda was not found on PATH or in the default Anaconda/Miniconda locations.' }

# ---- Per-environment definitions (tested versions, 2026-07) ----------------
# yaml       : environment file in this folder
# torch      : pinned torch/torchvision pair known to work with the env
# cudaIndex  : PyTorch wheel index used unless -Cpu is passed
$envs = [ordered]@{
    'MDWD'         = @{ yaml = 'environment-mdwd.yml';         torch = 'torch==2.11.0', 'torchvision==0.26.0'; cudaIndex = 'https://download.pytorch.org/whl/cu130' }
    'mtsd-attrcls' = @{ yaml = 'environment-mtsd-attrcls.yml'; torch = 'torch==2.11.0', 'torchvision==0.26.0'; cudaIndex = 'https://download.pytorch.org/whl/cu128' }
    'mtsd-base'    = @{ yaml = 'environment-mtsd-base.yml';    torch = 'torch==2.10.0', 'torchvision==0.25.0'; cudaIndex = 'https://download.pytorch.org/whl/cu128' }
    'mtsd-la'      = @{ yaml = 'environment-mtsd-la.yml';      torch = 'torch==2.11.0', 'torchvision==0.26.0'; cudaIndex = 'https://download.pytorch.org/whl/cu128' }
}

$targets = if ($Name -eq 'all') { @($envs.Keys) } else { @($Name) }
$existing = (& $conda env list) -join "`n"

foreach ($target in $targets) {
    $spec = $envs[$target]
    $yaml = Join-Path $here $spec.yaml
    Write-Host "`n=== [$target] ===" -ForegroundColor Cyan

    if ($existing -match "(?m)^\s*$([regex]::Escape($target))\s") {
        Write-Host "[$target] exists - updating from $($spec.yaml) (--prune)"
        & $conda env update -n $target -f $yaml --prune
    }
    else {
        Write-Host "[$target] creating from $($spec.yaml)"
        & $conda env create -f $yaml
    }
    if ($LASTEXITCODE -ne 0) { throw "[$target] conda env create/update failed." }

    $index = if ($Cpu) { 'https://download.pytorch.org/whl/cpu' } else { $spec.cudaIndex }
    Write-Host "[$target] installing PyTorch ($($spec.torch -join ' ')) from $index"
    & $conda run -n $target python -m pip install @($spec.torch) --index-url $index
    if ($LASTEXITCODE -ne 0) { throw "[$target] PyTorch installation failed." }

    if ($target -eq 'mtsd-base') {
        Write-Host '[mtsd-base] installing Meta sam3 package (SAM 3 / SAM 3.1)'
        & $conda run -n mtsd-base python -m pip install 'git+https://github.com/facebookresearch/sam3.git'
        if ($LASTEXITCODE -ne 0) { throw '[mtsd-base] sam3 installation failed.' }
    }

    Write-Host "[$target] done. Activate with:  conda activate $target" -ForegroundColor Green
}

Write-Host "`nNote: mtsd-base / mtsd-la / MDWD download gated Hugging Face weights on first use - run 'hf auth login' inside the env once (see Documents/PromptDetect.md)."
