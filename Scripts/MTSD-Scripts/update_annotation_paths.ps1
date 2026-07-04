<#
.SYNOPSIS
Migrates old "source_image" paths in MTSD annotation JSON files to the new
unified repository layout (Datasets/MTSD/GRP-<n>/Images/...).

.DESCRIPTION
The MTSD annotations were produced when the datasets lived at the old
MTSDataset repository root, so their "source_image" entries look like:

    "source_image": "Datasets\\GRP-2\\Images\\008ad2a7-20251223_145924.jpg"

After the move into Urban-Vision-Benchmark the images live under
Datasets/MTSD/GRP-<n>/Images/. This script rewrites exactly those
"source_image" values (old forms with either separator, with or without the
"Datasets" prefix) to the new repo-relative form with forward slashes:

    "source_image": "Datasets/MTSD/GRP-2/Images/008ad2a7-20251223_145924.jpg"

Nothing else in the files is touched: the replacement is a targeted regex on
the raw text, so formatting, key order and all other fields are preserved.
Already-migrated values are ignored, making the script safe to re-run.

Safety:
  * -DryRun previews every change without writing anything.
  * Before a file is modified, a timestamped backup is written next to it
    (<name>.pre-migration-<yyyyMMdd-HHmmss>.bak).
  * The rewritten text must still parse as JSON or the file is left untouched.
  * Each new path is checked against the filesystem; missing target images
    are reported (the rewrite still happens, since the mapping is correct).

.PARAMETER AnnotationsRoot
Folder scanned recursively for *.json files.
Default: <repo>/Datasets/MTSD/Annotations

.PARAMETER DryRun
Preview all changes; no files are written.

.PARAMETER NoBackup
Skip the .bak backups (not recommended).

.EXAMPLE
powershell -ExecutionPolicy Bypass -File Scripts/MTSD-Scripts/update_annotation_paths.ps1 -DryRun

.EXAMPLE
powershell -ExecutionPolicy Bypass -File Scripts/MTSD-Scripts/update_annotation_paths.ps1
#>

[CmdletBinding()]
param(
    [string]$AnnotationsRoot,
    [switch]$DryRun,
    [switch]$NoBackup
)

$ErrorActionPreference = "Stop"

# <repo>/Scripts/MTSD-Scripts/update_annotation_paths.ps1 -> repo root is two levels up.
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $AnnotationsRoot) {
    $AnnotationsRoot = Join-Path $RepoRoot "Datasets\MTSD\Annotations"
}
if (-not (Test-Path $AnnotationsRoot)) {
    throw "Annotations folder not found: $AnnotationsRoot"
}

# Matches the OLD source_image prefixes only, in raw JSON text (where a
# backslash is stored as \\). Four legacy forms are covered:
#   Datasets\\GRP-<n>\\Images\\   Datasets/GRP-<n>/Images/
#   GRP-<n>\\Images\\             GRP-<n>/Images/         (value starts at GRP)
# Values already starting with Datasets/MTSD/ (or Datasets\\MTSD\\) cannot
# match: the Datasets-prefixed alternative requires GRP directly after the
# separator, and the bare alternative requires GRP directly after the quote.
$Pattern = '(?<key>"source_image"\s*:\s*")(?<old>(?:Datasets(?:\\\\|/))?GRP-(?<grp>\d+)(?:\\\\|/)Images(?:\\\\|/))'

$filesScanned   = 0
$filesChanged   = 0
$filesSkipped   = 0
$entriesUpdated = 0
$missingImages  = New-Object System.Collections.Generic.List[string]
$errors         = New-Object System.Collections.Generic.List[string]
$timestamp      = Get-Date -Format "yyyyMMdd-HHmmss"

$jsonFiles = Get-ChildItem -Path $AnnotationsRoot -Recurse -File |
    Where-Object { $_.Extension -eq ".json" }

Write-Host "Repository root : $RepoRoot"
Write-Host "Annotations root: $AnnotationsRoot"
Write-Host "Mode            : $(if ($DryRun) { 'DRY RUN (no files will be written)' } else { 'LIVE' })"
Write-Host ""

foreach ($file in $jsonFiles) {
    $filesScanned++
    try {
        $text = [System.IO.File]::ReadAllText($file.FullName)
    } catch {
        $errors.Add("READ FAILED $($file.FullName): $($_.Exception.Message)")
        continue
    }

    $fileMatches = [regex]::Matches($text, $Pattern)
    if ($fileMatches.Count -eq 0) {
        $filesSkipped++
        continue
    }

    # Verify that each rewritten path points at a real image.
    $fileMissing = 0
    foreach ($m in $fileMatches) {
        $tailStart = $m.Index + $m.Length
        $tailEnd   = $text.IndexOf('"', $tailStart)
        if ($tailEnd -gt $tailStart) {
            $fileName  = $text.Substring($tailStart, $tailEnd - $tailStart) -replace '\\\\', '\'
            $newTarget = Join-Path $RepoRoot ("Datasets\MTSD\GRP-" + $m.Groups['grp'].Value + "\Images\" + $fileName)
            if (-not (Test-Path -LiteralPath $newTarget)) {
                $fileMissing++
                $missingImages.Add("GRP-" + $m.Groups['grp'].Value + "/Images/" + $fileName)
            }
        }
    }

    $newText = [regex]::Replace($text, $Pattern, {
        param($m)
        $m.Groups['key'].Value + "Datasets/MTSD/GRP-" + $m.Groups['grp'].Value + "/Images/"
    })

    # The result must still be valid JSON, otherwise leave the file alone.
    try {
        $null = $newText | ConvertFrom-Json
    } catch {
        $errors.Add("VALIDATION FAILED (file left untouched) $($file.FullName): $($_.Exception.Message)")
        continue
    }

    $relPath = $file.FullName.Substring($RepoRoot.Length).TrimStart('\')
    if ($DryRun) {
        Write-Host ("[DRY] {0}: {1} source_image entr{2} would be updated{3}" -f $relPath, $fileMatches.Count,
            $(if ($fileMatches.Count -eq 1) { 'y' } else { 'ies' }),
            $(if ($fileMissing) { " ($fileMissing target image(s) not found on disk)" } else { "" }))
        $sample = $fileMatches[0]
        Write-Host ("      e.g. '{0}...' -> 'Datasets/MTSD/GRP-{1}/Images/...'" -f $sample.Groups['old'].Value, $sample.Groups['grp'].Value)
    } else {
        if (-not $NoBackup) {
            $backupPath = "$($file.FullName).pre-migration-$timestamp.bak"
            Copy-Item -LiteralPath $file.FullName -Destination $backupPath -Force:$false
        }
        # Preserve the file's original encoding (BOM or not).
        $bytes  = [System.IO.File]::ReadAllBytes($file.FullName)
        $hasBom = $bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF
        $encoding = New-Object System.Text.UTF8Encoding($hasBom)
        [System.IO.File]::WriteAllText($file.FullName, $newText, $encoding)
        Write-Host ("[OK ] {0}: {1} source_image entr{2} updated{3}" -f $relPath, $fileMatches.Count,
            $(if ($fileMatches.Count -eq 1) { 'y' } else { 'ies' }),
            $(if ($fileMissing) { " ($fileMissing target image(s) not found on disk)" } else { "" }))
    }

    $filesChanged++
    $entriesUpdated += $fileMatches.Count
}

Write-Host ""
Write-Host "================ Summary ================"
Write-Host ("Files scanned          : {0}" -f $filesScanned)
Write-Host ("Files {0}          : {1}" -f $(if ($DryRun) { "to change" } else { "changed  " }), $filesChanged)
Write-Host ("source_image updates   : {0}" -f $entriesUpdated)
Write-Host ("Files skipped (no old paths): {0}" -f $filesSkipped)
Write-Host ("Missing target images  : {0}" -f $missingImages.Count)
Write-Host ("Errors                 : {0}" -f $errors.Count)

if ($missingImages.Count -gt 0) {
    Write-Host ""
    Write-Host "Paths were rewritten, but these images do not exist under Datasets/MTSD/ (first 20):" -ForegroundColor Yellow
    $missingImages | Select-Object -First 20 | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
}
if ($errors.Count -gt 0) {
    Write-Host ""
    $errors | ForEach-Object { Write-Host "ERROR: $_" -ForegroundColor Red }
    exit 1
}
