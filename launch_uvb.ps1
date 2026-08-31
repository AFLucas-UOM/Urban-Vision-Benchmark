<#
.SYNOPSIS
Cross-platform-friendly PowerShell entry point for the Urban Vision Benchmark launcher.

.DESCRIPTION
Finds a usable Python interpreter, switches to the repository root, enables
UTF-8 output and forwards every remaining argument to launch_uvb.py. Use
-PythonExe to select a specific interpreter; otherwise an active environment,
python on PATH, or the Windows py launcher is used.

.EXAMPLE
.\launch_uvb.ps1
.\launch_uvb.ps1 --doctor
.\launch_uvb.ps1 --list
.\launch_uvb.ps1 workflow:Export-Dissertation-Tables --dry-run
#>

[CmdletBinding()]
param(
    [string]$PythonExe = "",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$LauncherArguments
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path $PSScriptRoot).Path
$Launcher = Join-Path $RepoRoot "launch_uvb.py"

if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) {
    throw "Launcher not found: $Launcher"
}

function Resolve-LauncherPython {
    param([string]$Requested)

    if ($Requested) {
        $resolved = Get-Command $Requested -ErrorAction SilentlyContinue
        if ($resolved) { return @{ Exe = $resolved.Source; Prefix = @() } }
        if (Test-Path -LiteralPath $Requested -PathType Leaf) {
            return @{ Exe = (Resolve-Path $Requested).Path; Prefix = @() }
        }
        throw "Requested Python interpreter was not found: $Requested"
    }

    if ($env:CONDA_PREFIX) {
        $activePython = if ($IsWindows -or $env:OS -eq "Windows_NT") {
            Join-Path $env:CONDA_PREFIX "python.exe"
        } else {
            Join-Path $env:CONDA_PREFIX "bin/python"
        }
        if (Test-Path -LiteralPath $activePython -PathType Leaf) {
            return @{ Exe = $activePython; Prefix = @() }
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) { return @{ Exe = $python.Source; Prefix = @() } }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) { return @{ Exe = $py.Source; Prefix = @("-3") } }

    throw "Python 3 was not found. Install Python or run with -PythonExe <path>."
}

$runtime = Resolve-LauncherPython $PythonExe
$previousPythonUtf8 = $env:PYTHONUTF8
$previousPythonIoEncoding = $env:PYTHONIOENCODING
$previousOutputEncoding = [Console]::OutputEncoding
$exitCode = 1

Push-Location $RepoRoot
try {
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

    & $runtime.Exe @($runtime.Prefix) -u $Launcher @LauncherArguments
    $exitCode = $LASTEXITCODE
}
catch {
    Write-Error "UVB launcher failed: $($_.Exception.Message)"
    $exitCode = 1
}
finally {
    $env:PYTHONUTF8 = $previousPythonUtf8
    $env:PYTHONIOENCODING = $previousPythonIoEncoding
    [Console]::OutputEncoding = $previousOutputEncoding
    Pop-Location
}

exit $exitCode
