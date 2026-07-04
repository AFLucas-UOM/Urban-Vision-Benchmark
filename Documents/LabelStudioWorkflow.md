# Label Studio Group Workflow

This note explains how to prepare, start, import, and reload Label Studio projects for any `GRP-X` dataset group.

Run commands from the repository root:

```powershell
cd "E:\2. UM-Student\MSC Dissertation\MTSDataset"
```

## Quick Copy-Paste Commands

Use these commands first. Replace `GRP-3` with the group you are working on.

### 1. Start Label Studio

Open one terminal and keep it running:

```powershell
.\Scripts\LabelStudio\start_labelstudio.ps1 -Group GRP-3
```

Then open Label Studio:

```text
http://localhost:8080
```

Default login:

```text
Username: sample@example.com
Password: SampleAnnotations123!
API token: sample-annotations-token
```

### 2. Add A New Group From QA Annotations

Use this when the group already has a final QA file such as:

```text
Datasets/Annotations/GRP-3/Final-QA/QA-GRP3.json
```

Run this in a second terminal:

```powershell
python .\Scripts\LabelStudio\setup_labelstudio_project.py --group GRP-3 --no-start --reload --import-mode qa
```

This creates or reloads the `GRP-3` Label Studio project with the completed QA boxes and attributes already imported.

### 3. Add A New Group From RAW/XML Annotations

Use this when there is no final QA file yet, or when you want to start from the original XML/Fiverr annotations.

Expected XML folder:

```text
Datasets/Annotations/GRP-3/Fiverr-Annotations/
```

Run:

```powershell
python .\Scripts\LabelStudio\setup_labelstudio_project.py --group GRP-3 --no-start --reload --import-mode raw
```

This creates or reloads the `GRP-3` project from the raw annotation source.

### 4. Reload The Current Best Version

Use this after changing QA JSON, changing XML files, or fixing images/labels:

```powershell
python .\Scripts\LabelStudio\setup_labelstudio_project.py --group GRP-3 --no-start --reload
```

This auto-picks QA if `Final-QA/QA-GRP3.json` exists. Otherwise it falls back to RAW/XML.

### 5. Make A Test Project Without Deleting The Main One

Use a different title when you want to inspect an import without replacing the main group project:

```powershell
python .\Scripts\LabelStudio\setup_labelstudio_project.py --group GRP-3 --no-start --reload --title GRP-3-CHECK --keep-existing
```

## Folder Layout

Each group should use this structure:

```text
Datasets/
  Annotations/
    GRP-X/
      Final-QA/
        QA-GRPX.json
      Fiverr-Annotations/
        *.xml
  GRP-X/
    Images/
      *.jpg, *.jpeg, *.png, ...
    Export/
      ImageSets.txt
      LabelMap.txt
```

Examples:

```text
Datasets/Annotations/GRP-1/Final-QA/QA-GRP1.json
Datasets/Annotations/GRP-10/Final-QA/QA-GRP10.json
Datasets/Annotations/GRP-1/Fiverr-Annotations/*.xml
```

The preferred image folder is:

```text
Datasets/<GROUP>/Images/
```

The scripts also support these image folder fallbacks:

```text
Datasets/<GROUP>/<GROUP>/
Datasets/<GROUP>/merged_images/
```

## Import Priority

For each group, the preparation script chooses the annotation source in this order:

1. `Datasets/Annotations/<GROUP>/Final-QA/QA-<GROUP_WITHOUT_DASH>.json`
2. Other `QA-*.json` files in `Final-QA/`
3. XML files in `Datasets/Annotations/<GROUP>/Fiverr-Annotations/`

For `GRP-1`, the preferred QA file is:

```text
Datasets/Annotations/GRP-1/Final-QA/QA-GRP1.json
```

If that file exists, the import is a QA import. If it does not exist, the scripts fall back to the XML workflow.

## Start Label Studio

Start Label Studio for a group:

```powershell
.\Scripts\LabelStudio\start_labelstudio.ps1 -Group GRP-1
```

Or with the command wrapper:

```powershell
.\Scripts\LabelStudio\start_labelstudio.cmd -Group GRP-1
```

On Bash:

```bash
./Scripts/LabelStudio/start_labelstudio.sh GRP-1
```

Keep the Label Studio terminal open while using the app.

Default local credentials:

```text
URL: http://localhost:8080
Username: sample@example.com
Password: SampleAnnotations123!
API token: sample-annotations-token
```

## Create Or Reload A Project

The easiest refresh command is:

```powershell
python .\Scripts\LabelStudio\setup_labelstudio_project.py --group GRP-1 --no-start --reload
```

This command:

- detects the current best source for the group;
- regenerates the Label Studio task file and config;
- deletes any existing Label Studio project with the same title;
- waits for the old project deletion to settle;
- recreates the project with the latest annotations;
- imports tasks in chunks and retries transient Label Studio import failures;
- reconnects local image storage.

By default, the project title is the group name. For `GRP-1`, the project is titled `GRP-1`.

Use this same command whenever:

- `Final-QA/QA-GRPX.json` becomes available after an XML-only import;
- `QA-GRPX.json` is updated after QA work;
- XML files in `Fiverr-Annotations/` are changed before QA is complete.

## Prepare Only

To generate import files without touching Label Studio:

```powershell
python .\Scripts\LabelStudio\prepare_labelstudio_import.py --group GRP-1
```

Outputs are written to:

```text
Datasets/GRP-1/labelstudio_output/
```

Important files:

```text
GRP-1.tasks.json
GRP-1.qa.tasks.json
GRP-1.raw.tasks.json
config.xml
import_manifest.json
validation_report.md
validation_report.json
LabelStudioImportLog.json
```

Only the files for the selected workflow are created or refreshed. The compatibility file `<GROUP>.tasks.json` points at the most recently prepared workflow.

## Force A Workflow

Force QA mode:

```powershell
python .\Scripts\LabelStudio\setup_labelstudio_project.py --group GRP-1 --no-start --reload --import-mode qa
```

Force XML/RAW mode even when QA exists:

```powershell
python .\Scripts\LabelStudio\setup_labelstudio_project.py --group GRP-1 --no-start --reload --import-mode raw
```

Prepare only with a forced mode:

```powershell
python .\Scripts\LabelStudio\prepare_labelstudio_import.py --group GRP-1 --import-mode raw
```

## Keep Existing Projects

Normally reloads recreate the project with the same title. To create a second project without deleting the existing one, pass a different title and keep existing projects:

```powershell
python .\Scripts\LabelStudio\setup_labelstudio_project.py --group GRP-1 --no-start --reload --title GRP-1-CHECK --keep-existing
```

## Useful Checks

Check whether QA exists:

```powershell
Test-Path .\Datasets\Annotations\GRP-1\Final-QA\QA-GRP1.json
```

Count XML files:

```powershell
Get-ChildItem .\Datasets\Annotations\GRP-1\Fiverr-Annotations -Recurse -Filter *.xml | Measure-Object
```

Count images:

```powershell
Get-ChildItem .\Datasets\GRP-1\Images -File | Measure-Object
```

Review import warnings:

```text
Datasets/GRP-1/labelstudio_output/validation_report.md
```

Important fields:

```text
Import mode
Detected annotation format
Valid for import
Images without annotations
Annotations without images
Unknown labels
Unknown attribute values
Out-of-bounds boxes
```

## Common Problems

If images do not display, restart Label Studio:

```powershell
.\Scripts\LabelStudio\start_labelstudio.ps1 -Group GRP-1
```

The startup script enables local file serving and sets the repository root as the local file document root.

If setup says import files are missing, use reload:

```powershell
python .\Scripts\LabelStudio\setup_labelstudio_project.py --group GRP-1 --no-start --reload
```

If Label Studio returns a `500` during import with a message about an `atomic` transaction block, rerun the same reload command. The setup script now retries transient import failures and removes a half-created project if the import cannot complete.

If port `8080` is already in use, the startup script stops the existing listener before starting Label Studio again.
