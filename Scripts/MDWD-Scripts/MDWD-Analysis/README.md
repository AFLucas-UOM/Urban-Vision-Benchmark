# MDWD-Analysis

Exploratory data analysis for the **Maltese Domestic Waste Dataset (MDWD)**,
mirroring the structure and output conventions of the MTSD EDA
([Scripts/MTSD-Scripts/MTSD-Analysis/](../../MTSD-Scripts/MTSD-Analysis/)).

## Contents

| File | Purpose |
| --- | --- |
| `mdwd_eda/config.py` | Repository paths, variant discovery, format detection |
| `mdwd_eda/mapper.py` | Dataset mapper: indexes images, labels, splits, dimensions and boxes (YOLO **and** COCO) |
| `mdwd_eda/checks.py` | Integrity checks over the mapped dataset |
| `mdwd_eda/stats.py` | Statistics tables and the summary dict |
| `mdwd_eda/plotstyle.py` | Publication chart style (same validated palette as the MTSD EDA) |
| `run_eda.py` | CLI: map → check → tables → charts → `eda_summary.json` |
| `visualise_samples.py` | Renders annotated sample images with bounding boxes |
| `DatasetVisualisation.ipynb`, `DatasetVisualisation-OG.ipynb` | **Legacy** notebooks from the pre-reorganisation repo (they expect the defunct `Versions/AICOM-YOLOv26` layout); kept for provenance, superseded by `run_eda.py` |

## Expected dataset format

The tooling auto-detects the format of each variant under `Datasets/MDWD/`:

* **YOLO** (`MDWD-YOLO11`, `MDWD-YOLO12`, `MDWD-YOLO26`): `data.yaml` plus
  `train|valid|test/images` and `.../labels` with normalised
  `class cx cy w h` rows.
* **COCO** (`MDWD-RFDETR`): `train|valid|test/_annotations.coco.json`.

All four variants are exports of the same Roboflow project (v20, 5 classes,
640×640, ~10× augmentation on train), so `MDWD-YOLO26` is the default.

## Running the EDA

From the repository root (any environment with `pandas`, `matplotlib`,
`Pillow`, `PyYAML` — e.g. the `mdwd` conda env):

```bash
# Full run (default variant MDWD-YOLO26)
python Scripts/MDWD-Scripts/MDWD-Analysis/run_eda.py

# Quick sample run, outputs kept separate from the main EDA
python Scripts/MDWD-Scripts/MDWD-Analysis/run_eda.py --max-images 50 --out-tag sample

# Another variant
python Scripts/MDWD-Scripts/MDWD-Analysis/run_eda.py --variant MDWD-RFDETR

# Annotated sample images (8 per split by default)
python Scripts/MDWD-Scripts/MDWD-Analysis/visualise_samples.py --per-split 8
```

## Outputs

Everything is written under `Documents/MDWD-EDA/` (or `Documents/MDWD-EDA-<tag>/`
with `--out-tag`, which leaves existing outputs untouched):

```
Documents/MDWD-EDA/
├── eda_summary.json              headline numbers, per-split counts, integrity issue counts
├── GeneratedCSVs/
│   ├── image_inventory.csv       one row per image (split, dims, box count, source stem)
│   ├── box_inventory.csv         one row per bounding box (class, geometry)
│   ├── split_overview.csv        images/boxes/unique sources per split
│   ├── class_distribution.csv    instances per class, overall + per split
│   ├── bbox_geometry_summary.csv area/aspect quantiles
│   ├── coco_size_classes.csv     small/medium/large per class
│   ├── resolution_distribution.csv
│   ├── instances_per_image_histogram.csv
│   ├── class_cooccurrence.csv
│   ├── integrity_summary.csv     per-issue-type counts (zero rows included)
│   └── integrity_issues.csv      every individual finding
├── Figures/                      fig01..fig10, 300-dpi PNGs
└── SampleAnnotationImages/       annotated samples per split
```

Integrity checks cover: missing/unreadable images, missing labels, orphan
labels/annotations, malformed label rows, invalid class ids, out-of-range
boxes, empty annotations, duplicate filenames, and augmented copies of one
source image leaking across splits.
