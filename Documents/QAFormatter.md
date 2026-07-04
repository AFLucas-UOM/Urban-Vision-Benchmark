# QA Formatter

`Scripts\LabelStudio\QA-Formatter.ps1` converts a raw Label Studio QA export into the COCO-style QA JSON used by the final annotation files.

The script rewrites the selected QA file in place by default and creates a `.bak` backup beside it.

## Interactive Use

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\Scripts\LabelStudio\QA-Formatter.ps1
```

The script lists available groups and asks for one, for example:

```text
GRP-3
```

It then formats:

```text
Datasets\Annotations\GRP-3\Final-QA\QA-GRP3.json
```

## Direct Group Use

To skip the prompt:

```powershell
powershell -ExecutionPolicy Bypass -File .\Scripts\LabelStudio\QA-Formatter.ps1 -Group GRP-3
```

`GRP3` also works; it is normalized to `GRP-3`.

## Explicit File Use

To format a specific JSON file:

```powershell
powershell -ExecutionPolicy Bypass -File .\Scripts\LabelStudio\QA-Formatter.ps1 -Path .\Datasets\Annotations\GRP-3\Final-QA\QA-GRP3.json
```

## Optional Output File

To test conversion without replacing the source file:

```powershell
powershell -ExecutionPolicy Bypass -File .\Scripts\LabelStudio\QA-Formatter.ps1 -Group GRP-3 -OutputPath .\Datasets\Annotations\GRP-3\Final-QA\QA-GRP3.converted-test.json -NoBackup
```

## Reload Label Studio After Formatting

After formatting a group, reload it into Label Studio with the QA import mode:

```powershell
python .\Scripts\LabelStudio\setup_labelstudio_project.py --group GRP-3 --no-start --reload --import-mode qa
```

Remove `--no-start` if you want the setup script to start Label Studio itself.
