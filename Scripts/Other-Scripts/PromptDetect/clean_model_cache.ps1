<#
.SYNOPSIS
    List or delete the PromptDetect model weights cached by Hugging Face, to
    reclaim disk space.

.DESCRIPTION
    SAM 3 / 3.1, Cosmos Reason2, and LocateAnything weights download into the
    Hugging Face hub cache (default: %USERPROFILE%\.cache\huggingface\hub, or
    $env:HF_HUB_CACHE / $env:HF_HOME\hub if set). Each repo lives in a
    "models--<org>--<name>" folder.

    This script touches ONLY the PromptDetect model repos — any other cached
    models (e.g. sentence-transformers, gemma) are left untouched.

.PARAMETER Delete
    One or more model short-names to delete (e.g. Cosmos-Reason2-32B), or "all"
    to remove every PromptDetect model. Omit to just list.

.PARAMETER All
    Shorthand for "-Delete all".

.PARAMETER Yes
    Skip the confirmation prompt (for non-interactive use).

.EXAMPLE
    ./clean_model_cache.ps1
    # Lists each PromptDetect model and its size on disk.

.EXAMPLE
    ./clean_model_cache.ps1 -Delete Cosmos-Reason2-32B,Cosmos-Reason2-8B
    # Deletes just the 8B and 32B Cosmos checkpoints.

.EXAMPLE
    ./clean_model_cache.ps1 -All -Yes
    # Deletes every PromptDetect model without prompting.
#>
[CmdletBinding()]
param(
    [string[]]$Delete,
    [switch]$All,
    [switch]$Yes
)

function Get-HfCacheDir {
    if ($env:HF_HUB_CACHE) { return $env:HF_HUB_CACHE }
    if ($env:HF_HOME)      { return (Join-Path $env:HF_HOME 'hub') }
    return (Join-Path $env:USERPROFILE '.cache\huggingface\hub')
}

function Get-FolderSize($path) {
    if (-not (Test-Path -LiteralPath $path)) { return [int64]0 }
    $sum = (Get-ChildItem -LiteralPath $path -Recurse -File -Force -ErrorAction SilentlyContinue |
            Measure-Object -Property Length -Sum).Sum
    if ($null -eq $sum) { return [int64]0 }
    return [int64]$sum
}

function Format-GB($bytes) { '{0,8:N2} GB' -f ($bytes / 1GB) }

# PromptDetect model repos (Hugging Face ids).
$repoIds = @(
    'facebook/sam3',
    'facebook/sam3.1',
    'nvidia/Cosmos-Reason2-2B',
    'nvidia/Cosmos-Reason2-8B',
    'nvidia/Cosmos-Reason2-32B',
    'nvidia/LocateAnything-3B'
)

$cache = Get-HfCacheDir
Write-Host "Hugging Face hub cache: $cache"
if (-not (Test-Path -LiteralPath $cache)) {
    Write-Warning "Cache directory not found — nothing to do."
    return
}

# Build inventory.
$items = foreach ($id in $repoIds) {
    $folder = 'models--' + ($id -replace '/', '--')
    $path   = Join-Path $cache $folder
    $exists = Test-Path -LiteralPath $path
    [pscustomobject]@{
        Repo   = $id
        Short  = ($id -split '/')[-1]
        Path   = $path
        Exists = $exists
        Bytes  = if ($exists) { Get-FolderSize $path } else { [int64]0 }
    }
}

Write-Host ""
Write-Host "PromptDetect models in cache:" -ForegroundColor Cyan
foreach ($it in $items) {
    $state = if ($it.Exists) { Format-GB $it.Bytes } else { '   (not cached)' }
    Write-Host ('  {0,-24} {1}' -f $it.Short, $state)
}
$total = ($items | Where-Object Exists | Measure-Object -Property Bytes -Sum).Sum
Write-Host ('  {0,-24} {1}' -f 'TOTAL', (Format-GB ([int64]($total ?? 0)))) -ForegroundColor Cyan

# Decide what (if anything) to delete.
if ($All) { $Delete = @('all') }
if (-not $Delete) {
    Write-Host ""
    Write-Host "Nothing deleted. Use -Delete <name,...> or -All to remove (add -Yes to skip the prompt)."
    return
}

if ($Delete -contains 'all') {
    $targets = $items | Where-Object Exists
} else {
    $targets = $items | Where-Object { $_.Exists -and ($Delete -contains $_.Short) }
    $unknown = $Delete | Where-Object { $_ -ne 'all' -and ($items.Short -notcontains $_) }
    if ($unknown) { Write-Warning ("Unknown / not-a-PromptDetect-model: {0}" -f ($unknown -join ', ')) }
}

if (-not $targets) { Write-Warning "Nothing matched to delete."; return }

$freeing = ($targets | Measure-Object -Property Bytes -Sum).Sum
Write-Host ""
Write-Host "Will delete:" -ForegroundColor Yellow
foreach ($t in $targets) { Write-Host ('  {0}  ({1})' -f $t.Short, (Format-GB $t.Bytes)) }
Write-Host ("Reclaims ~{0}" -f (Format-GB ([int64]$freeing))) -ForegroundColor Yellow

if (-not $Yes) {
    $answer = Read-Host "Proceed with deletion? (y/N)"
    if ($answer -notmatch '^(y|yes)$') { Write-Host "Aborted — nothing deleted."; return }
}

foreach ($t in $targets) {
    Write-Host ("Deleting {0} ..." -f $t.Short)
    Remove-Item -LiteralPath $t.Path -Recurse -Force -Confirm:$false -ErrorAction Continue
}
Write-Host ("Done. Reclaimed ~{0}" -f (Format-GB ([int64]$freeing))) -ForegroundColor Green
