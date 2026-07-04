# MTSD-Scripts

Tooling for the **Maltese Traffic Sign Dataset (MTSD)** track of the dissertation.

| Folder / file | Purpose |
| --- | --- |
| [AttributeClassification/](AttributeClassification/README.md) | Multi-attribute sign classification (DINOv3 / V-JEPA / ConvNeXt, frozen · LoRA · fine-tune) with W&B logging |
| [MTSD-Analysis/](MTSD-Analysis/) | EDA: `mtsd_eda` package, `MTSD-EDA.ipynb`, dataset mapper and audit scripts (outputs go to `Documents/MTSD-EDA/`) |
| [LabelStudio/](LabelStudio/) | Label Studio launchers, import preparation, project setup and the QA formatter |
| `update_annotation_paths.ps1` | One-off migration of old `source_image` paths in annotation JSONs (see below) |

## Annotation path migration (`update_annotation_paths.ps1`)

The QA annotation JSONs under `Datasets/MTSD/Annotations/GRP-*/Final-QA/` were
exported when the images lived at the old repository layout, so they contain
entries such as:

```json
"source_image": "Datasets\\GRP-2\\Images\\008ad2a7-20251223_145924.jpg"
```

The images now live under `Datasets/MTSD/GRP-<n>/Images/`. The script rewrites
exactly those `source_image` values (all legacy separator/prefix variants) to
the new repo-relative form:

```json
"source_image": "Datasets/MTSD/GRP-2/Images/008ad2a7-20251223_145924.jpg"
```

Everything else in the files (formatting included) is left untouched, values
that are already migrated are skipped, and each modified file gets a
timestamped `.pre-migration-<stamp>.bak` backup next to it first.

**Preview the changes (writes nothing):**

```powershell
powershell -ExecutionPolicy Bypass -File Scripts/MTSD-Scripts/update_annotation_paths.ps1 -DryRun
```

**Apply the migration:**

```powershell
powershell -ExecutionPolicy Bypass -File Scripts/MTSD-Scripts/update_annotation_paths.ps1
```

The summary reports files scanned/changed, the number of `source_image`
entries updated, and any rewritten paths whose image is missing on disk
(annotated images that are not present under the group's `Images/` folder —
worth reviewing, but the rewrite itself is still correct).

After migrating, regenerate the attribute-classification manifest when you
next need it (`python -m mtsd_attr.data_manifest` from
`Scripts/MTSD-Scripts/AttributeClassification/`), since the stored QA SHA-256
hashes will have changed.
