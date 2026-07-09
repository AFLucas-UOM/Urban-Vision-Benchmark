param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path $PSScriptRoot).Path
Push-Location $RepoRoot
try {
    & python (Join-Path $RepoRoot "launch_uvb.py") @Arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
