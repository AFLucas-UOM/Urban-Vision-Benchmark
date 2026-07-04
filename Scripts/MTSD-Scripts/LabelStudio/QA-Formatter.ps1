<#
.SYNOPSIS
Converts a Label Studio QA export JSON file into the COCO-style QA format used by QA-GRP1.json.

.DESCRIPTION
The script reads a raw Label Studio task export, converts rectangle labels and their per-box
choice attributes into COCO annotations, and overwrites the input file by default.

.PARAMETER Group
Dataset group to format, for example GRP-3. When Path is omitted, the script uses:
Datasets\Annotations\<Group>\Final-QA\QA-<GroupWithoutDash>.json

.PARAMETER Path
Optional path to a Label Studio QA export JSON file. This overrides Group-derived path lookup.

.PARAMETER OutputPath
Optional output path. When omitted, the input file is replaced.

.PARAMETER NoBackup
Do not create a .bak copy before overwriting the input file.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\Scripts\LabelStudio\QA-Formatter.ps1

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\Scripts\LabelStudio\QA-Formatter.ps1 -Group GRP-3

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\Scripts\LabelStudio\QA-Formatter.ps1 -Path .\Datasets\Annotations\GRP-3\Final-QA\QA-GRP3.json
#>

[CmdletBinding()]
param(
    [string]$Group,

    [string]$Path,

    [string]$OutputPath,

    [switch]$NoBackup
)

$ErrorActionPreference = "Stop"

$DefaultCategories = @(
    "Pedestrian Crossing",
    "Stop Sign",
    "No Entry (One Way)",
    "Roundabout Ahead",
    "No Through Road (T-Junction)",
    "Blind-Spot Mirror (Convex Mirror)",
    "Street Sign",
    "Directional Sign",
    "Tourist Sign",
    "Auxiliary Sign",
    "Back-Unknown",
    "Other-Unknown"
)

function Normalize-GroupName {
    param([string]$InputGroup)

    $trimmed = $InputGroup.Trim().ToUpperInvariant()
    if ($trimmed -match "^GRP-?\d+$") {
        $number = ($trimmed -replace "^GRP-?", "")
        return "GRP-$number"
    }

    throw "Invalid group '$InputGroup'. Use a value like GRP-3."
}

function Get-CompactGroupName {
    param([string]$InputGroup)

    return $InputGroup -replace "-", ""
}

function Get-QAPathForGroup {
    param([string]$InputGroup)

    $compactGroup = Get-CompactGroupName -InputGroup $InputGroup
    return Join-Path -Path "Datasets\Annotations\$InputGroup\Final-QA" -ChildPath "QA-$compactGroup.json"
}

function Read-GroupName {
    $annotationsRoot = "Datasets\Annotations"
    $groups = @()
    if (Test-Path -LiteralPath $annotationsRoot) {
        $groups = @(Get-ChildItem -LiteralPath $annotationsRoot -Directory | Sort-Object Name | Select-Object -ExpandProperty Name)
    }

    if ($groups.Count -gt 0) {
        Write-Host "Available groups: $($groups -join ', ')"
    }

    $answer = Read-Host "Enter group to format, for example GRP-3"
    if ([string]::IsNullOrWhiteSpace($answer)) {
        throw "No group entered."
    }

    return Normalize-GroupName -InputGroup $answer
}

function Resolve-ExistingPath {
    param([string]$InputPath)

    $resolved = Resolve-Path -LiteralPath $InputPath -ErrorAction Stop
    return $resolved.ProviderPath
}

function Get-FirstValue {
    param(
        [object]$ArrayValue,
        [string]$Default = ""
    )

    if ($null -eq $ArrayValue) {
        return $Default
    }

    if ($ArrayValue -is [array]) {
        if ($ArrayValue.Count -gt 0) {
            return [string]$ArrayValue[0]
        }
        return $Default
    }

    return [string]$ArrayValue
}

function Get-ImageFileName {
    param([object]$Task)

    if ($Task.data -and $Task.data.image_filename) {
        return [string]$Task.data.image_filename
    }

    if ($Task.meta -and $Task.meta.source_image) {
        return Split-Path -Leaf ([string]$Task.meta.source_image)
    }

    if ($Task.data -and $Task.data.image) {
        $image = [string]$Task.data.image
        $match = [regex]::Match($image, "[?&]d=([^&]+)")
        if ($match.Success) {
            $decoded = [System.Uri]::UnescapeDataString($match.Groups[1].Value)
            return Split-Path -Leaf ($decoded -replace "/", "\")
        }
    }

    return "task-$($Task.id)"
}

function Get-SourceImage {
    param([object]$Task)

    if ($Task.meta -and $Task.meta.source_image) {
        return [string]$Task.meta.source_image
    }

    if ($Task.data -and $Task.data.image) {
        $image = [string]$Task.data.image
        $match = [regex]::Match($image, "[?&]d=([^&]+)")
        if ($match.Success) {
            return ([System.Uri]::UnescapeDataString($match.Groups[1].Value)) -replace "/", "\"
        }
    }

    return $null
}

function Get-TaskResults {
    param([object]$Task)

    if (-not $Task.annotations -or $Task.annotations.Count -eq 0) {
        return @()
    }

    $annotation = $Task.annotations | Select-Object -First 1
    if ($annotation.result) {
        return @($annotation.result)
    }

    return @()
}

function Get-PrimaryAnnotation {
    param([object]$Task)

    if (-not $Task.annotations -or $Task.annotations.Count -eq 0) {
        return $null
    }

    return $Task.annotations | Select-Object -First 1
}

if ([string]::IsNullOrWhiteSpace($Path)) {
    if ([string]::IsNullOrWhiteSpace($Group)) {
        $Group = Read-GroupName
    } else {
        $Group = Normalize-GroupName -InputGroup $Group
    }

    $Path = Get-QAPathForGroup -InputGroup $Group
    Write-Host "Using QA file for ${Group}: $Path"
}

$inputPath = Resolve-ExistingPath -InputPath $Path
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $outputPathResolved = $inputPath
} else {
    $outputPathResolved = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputPath)
}

$rawJson = Get-Content -LiteralPath $inputPath -Raw -Encoding UTF8
$tasks = @($rawJson | ConvertFrom-Json)
if ($tasks.Count -eq 1 -and $tasks[0] -is [array]) {
    $tasks = @($tasks[0])
}

if ($tasks.Count -eq 1 -and $tasks[0].PSObject.Properties.Name -contains "images" -and $tasks[0].PSObject.Properties.Name -contains "annotations") {
    throw "Input already looks like COCO JSON. Refusing to reformat: $inputPath"
}

$categoryNames = [System.Collections.Generic.List[string]]::new()
foreach ($name in $DefaultCategories) {
    [void]$categoryNames.Add($name)
}

foreach ($task in $tasks) {
    foreach ($result in (Get-TaskResults -Task $task)) {
        if ($result.type -eq "rectanglelabels") {
            $label = Get-FirstValue -ArrayValue $result.value.rectanglelabels
            if ($label -and -not $categoryNames.Contains($label)) {
                [void]$categoryNames.Add($label)
            }
        }
    }
}

$categoryIdByName = @{}
$categories = [System.Collections.Generic.List[object]]::new()
for ($index = 0; $index -lt $categoryNames.Count; $index++) {
    $id = $index + 1
    $categoryIdByName[$categoryNames[$index]] = $id
    $categories.Add([ordered]@{
        id = $id
        name = $categoryNames[$index]
        supercategory = "traffic_sign"
    })
}

$images = [System.Collections.Generic.List[object]]::new()
$annotations = [System.Collections.Generic.List[object]]::new()
$annotationId = 1
$imageId = 1

foreach ($task in $tasks) {
    $results = Get-TaskResults -Task $task
    $rectangles = @($results | Where-Object { $_.type -eq "rectanglelabels" })
    $primaryAnnotation = Get-PrimaryAnnotation -Task $task

    $width = $null
    $height = $null
    $firstSizedResult = $results | Where-Object { $_.original_width -and $_.original_height } | Select-Object -First 1
    if ($firstSizedResult) {
        $width = [int][double]$firstSizedResult.original_width
        $height = [int][double]$firstSizedResult.original_height
    }

    $images.Add([ordered]@{
        id = $imageId
        file_name = Get-ImageFileName -Task $task
        width = $width
        height = $height
        license = 0
        date_captured = $null
        label_studio_task_id = $task.id
        label_studio_inner_id = $task.inner_id
        source_image = Get-SourceImage -Task $task
    })

    foreach ($rect in $rectangles) {
        if (-not $rect.value) {
            continue
        }

        $label = Get-FirstValue -ArrayValue $rect.value.rectanglelabels
        if (-not $label) {
            continue
        }

        $regionId = [string]$rect.id
        $rectWidth = [double]$rect.original_width
        $rectHeight = [double]$rect.original_height
        $x = [Math]::Round(([double]$rect.value.x / 100.0) * $rectWidth, 2)
        $y = [Math]::Round(([double]$rect.value.y / 100.0) * $rectHeight, 2)
        $w = [Math]::Round(([double]$rect.value.width / 100.0) * $rectWidth, 2)
        $h = [Math]::Round(([double]$rect.value.height / 100.0) * $rectHeight, 2)

        $attributes = [ordered]@{
            view_angle = $null
            mounting = $null
            condition = $null
            sign_shape = $null
            label_studio_region_id = $regionId
            label_studio_annotation_id = $regionId
            rotation = $rect.value.rotation
            origin = $rect.origin
        }

        foreach ($choice in ($results | Where-Object { $_.type -eq "choices" -and [string]$_.id -eq $regionId })) {
            $attributeName = [string]$choice.from_name
            if ($attributes.Contains($attributeName)) {
                $attributes[$attributeName] = Get-FirstValue -ArrayValue $choice.value.choices
            }
        }

        $annotations.Add([ordered]@{
            id = $annotationId
            image_id = $imageId
            category_id = $categoryIdByName[$label]
            bbox = @($x, $y, $w, $h)
            area = [Math]::Round($w * $h, 4)
            iscrowd = 0
            segmentation = @()
            attributes = $attributes
        })
        $annotationId++
    }

    $imageId++
}

$inputLeaf = Split-Path -Leaf $inputPath
$coco = [ordered]@{
    info = [ordered]@{
        description = "COCO export converted from Label Studio export with per-box auxiliary attributes"
        source_file = $inputPath
        exported_from = $inputLeaf
    }
    licenses = @()
    images = $images
    annotations = $annotations
    categories = $categories
}

if ((Test-Path -LiteralPath $outputPathResolved) -and $outputPathResolved -eq $inputPath -and -not $NoBackup) {
    $backupPath = "$inputPath.bak"
    Copy-Item -LiteralPath $inputPath -Destination $backupPath -Force
    Write-Host "Backup written: $backupPath"
}

$outputDirectory = Split-Path -Parent $outputPathResolved
if ($outputDirectory -and -not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory | Out-Null
}

$jsonOutput = $coco | ConvertTo-Json -Depth 100
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($outputPathResolved, $jsonOutput, $utf8NoBom)

Write-Host "Converted $($tasks.Count) Label Studio tasks."
Write-Host "Wrote $($images.Count) images, $($annotations.Count) annotations, and $($categories.Count) categories."
Write-Host "Output: $outputPathResolved"
