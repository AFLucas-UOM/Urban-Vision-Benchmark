#!/usr/bin/env python3
"""Run the full MDWD EDA: map the dataset, check it, and generate outputs.

Outputs (mirroring the MTSD convention of Documents/MTSD-EDA):
    Documents/MDWD-EDA/GeneratedCSVs/   tables (inventory, stats, integrity)
    Documents/MDWD-EDA/Figures/         fig01..fig10 charts (300 dpi PNG)
    Documents/MDWD-EDA/eda_summary.json headline numbers + integrity counts

Usage (from the repository root or this folder):
    python Scripts/MDWD-Scripts/MDWD-Analysis/run_eda.py
    python Scripts/MDWD-Scripts/MDWD-Analysis/run_eda.py --variant MDWD-RFDETR
    python Scripts/MDWD-Scripts/MDWD-Analysis/run_eda.py --max-images 50 --out-tag sample
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mdwd_eda import checks, config, plotstyle, stats  # noqa: E402
from mdwd_eda.mapper import DatasetMap, build_map  # noqa: E402

SPLIT_LABELS = {"train": "Train", "valid": "Validation", "test": "Test"}


# --- Charts ------------------------------------------------------------------

def fig_dataset_composition(dataset_map: DatasetMap, figures_dir: Path) -> None:
    overview = stats.split_overview(dataset_map)
    body = overview[overview["split"] != "TOTAL"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, column, title in (
        (axes[0], "images", "Images per split"),
        (axes[1], "boxes", "Annotated instances per split"),
    ):
        labels = [SPLIT_LABELS.get(s, s) for s in body["split"]]
        bars = ax.barh(labels, body[column], color=plotstyle.ACCENT)
        ax.invert_yaxis()
        plotstyle.style_h_barh(ax)
        plotstyle.bar_value_labels(ax, bars)
        ax.set_title(title)
        ax.set_xlabel("Count")
        ax.xaxis.set_major_formatter(FuncFormatter(plotstyle.thousands))
        ax.margins(x=0.15)
    fig.suptitle(f"{dataset_map.variant}: dataset composition", fontweight="bold")
    fig.tight_layout()
    plotstyle.save_fig(fig, "fig01_dataset_composition", figures_dir)
    plt.close(fig)


def fig_class_distribution(dataset_map: DatasetMap, figures_dir: Path) -> None:
    table = stats.class_distribution(dataset_map)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(table["class_name"], table["instances"], color=plotstyle.ACCENT)
    ax.invert_yaxis()
    plotstyle.style_h_barh(ax)
    for bar, (_, row) in zip(bars, table.iterrows()):
        ax.text(
            bar.get_width(), bar.get_y() + bar.get_height() / 2,
            f" {row['instances']:,} ({row['share_pct']:.1f}%)",
            va="center", fontsize=9, color=plotstyle.INK_SECONDARY,
        )
    ax.set_title("Class distribution (all splits)")
    ax.set_xlabel("Annotated instances")
    ax.xaxis.set_major_formatter(FuncFormatter(plotstyle.thousands))
    ax.margins(x=0.22)
    plotstyle.save_fig(fig, "fig02_class_distribution", figures_dir)
    plt.close(fig)


def fig_class_by_split(dataset_map: DatasetMap, figures_dir: Path) -> None:
    table = stats.class_distribution(dataset_map)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    class_names = table["class_name"]
    y = np.arange(len(class_names))
    bar_height = 0.26
    for offset, split in enumerate(config.SPLITS):
        counts = table[split]
        share = counts / counts.sum() * 100 if counts.sum() else counts
        ax.barh(
            y + (offset - 1) * bar_height, share, height=bar_height,
            color=plotstyle.CATEGORICAL[offset], label=SPLIT_LABELS.get(split, split),
        )
    ax.set_yticks(y, class_names)
    ax.invert_yaxis()
    plotstyle.style_h_barh(ax)
    ax.set_xlabel("Share of the split's instances (%)")
    ax.set_title("Class balance per split")
    ax.legend(loc="lower right")
    plotstyle.save_fig(fig, "fig03_class_balance_per_split", figures_dir)
    plt.close(fig)


def fig_instances_per_image(dataset_map: DatasetMap, figures_dir: Path) -> None:
    per_image = dataset_map.images["n_boxes"]
    fig, ax = plt.subplots()
    upper = int(per_image.quantile(0.995)) + 1
    bins = np.arange(-0.5, upper + 1.5)
    ax.hist(per_image.clip(upper=upper), bins=bins, color=plotstyle.ACCENT, edgecolor=plotstyle.SURFACE)
    ax.set_xlabel(f"Instances per image (clipped at {upper})")
    ax.set_ylabel("Images")
    ax.yaxis.set_major_formatter(FuncFormatter(plotstyle.thousands))
    ax.set_title("Instances per image")
    ax.axvline(per_image.mean(), color=plotstyle.INK_SECONDARY, linestyle="--", linewidth=1.2)
    ax.text(
        per_image.mean(), ax.get_ylim()[1] * 0.95,
        f" mean {per_image.mean():.2f} / median {per_image.median():.0f}",
        fontsize=9, color=plotstyle.INK_SECONDARY, va="top",
    )
    plotstyle.save_fig(fig, "fig04_instances_per_image", figures_dir)
    plt.close(fig)


def fig_bbox_area(dataset_map: DatasetMap, figures_dir: Path) -> None:
    area = dataset_map.boxes["area_frac"].dropna()
    area = area[area > 0]
    fig, ax = plt.subplots()
    bins = np.logspace(np.log10(area.min()), np.log10(area.max()), 40)
    ax.hist(area, bins=bins, color=plotstyle.ACCENT, edgecolor=plotstyle.SURFACE)
    ax.set_xscale("log")
    ax.set_xlabel("Box area as a fraction of the image (log scale)")
    ax.set_ylabel("Boxes")
    ax.yaxis.set_major_formatter(FuncFormatter(plotstyle.thousands))
    ax.set_title("Bounding-box area distribution")
    median = area.median()
    ax.axvline(median, color=plotstyle.INK_SECONDARY, linestyle="--", linewidth=1.2)
    ax.text(median, ax.get_ylim()[1] * 0.95, f" median {median:.3%}",
            fontsize=9, color=plotstyle.INK_SECONDARY, va="top")
    plotstyle.save_fig(fig, "fig05_bbox_area", figures_dir)
    plt.close(fig)


def fig_bbox_aspect(dataset_map: DatasetMap, figures_dir: Path) -> None:
    aspect = dataset_map.boxes["aspect_ratio"].dropna()
    aspect = aspect[aspect > 0]
    fig, ax = plt.subplots()
    bins = np.logspace(np.log10(max(aspect.min(), 0.05)), np.log10(min(aspect.max(), 20)), 40)
    ax.hist(aspect.clip(0.05, 20), bins=bins, color=plotstyle.ACCENT, edgecolor=plotstyle.SURFACE)
    ax.set_xscale("log")
    ax.set_xlabel("Width / height (log scale; 1.0 = square)")
    ax.set_ylabel("Boxes")
    ax.yaxis.set_major_formatter(FuncFormatter(plotstyle.thousands))
    ax.axvline(1.0, color=plotstyle.BASELINE, linewidth=1.2)
    ax.set_title("Bounding-box aspect-ratio distribution")
    plotstyle.save_fig(fig, "fig06_bbox_aspect_ratio", figures_dir)
    plt.close(fig)


def fig_size_classes(dataset_map: DatasetMap, figures_dir: Path) -> None:
    table = stats.size_class_distribution(dataset_map)
    if table.empty:
        return
    totals = table[["small", "medium", "large"]].sum(axis=1)
    shares = table[["small", "medium", "large"]].div(totals, axis=0) * 100
    ramp = [plotstyle.SEQUENTIAL_BLUES[2], plotstyle.SEQUENTIAL_BLUES[6], plotstyle.SEQUENTIAL_BLUES[10]]
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    left = np.zeros(len(table))
    for column, colour in zip(("small", "medium", "large"), ramp):
        ax.barh(
            table["class_name"], shares[column], left=left, color=colour,
            edgecolor=plotstyle.SURFACE, linewidth=2, label=f"{column.capitalize()}",
        )
        left += shares[column].to_numpy()
    ax.invert_yaxis()
    plotstyle.style_h_barh(ax)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the class's boxes (%)")
    ax.set_title("COCO size classes per class (small ≤ 32² px, medium ≤ 96² px)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=3)
    plotstyle.save_fig(fig, "fig07_coco_size_classes", figures_dir)
    plt.close(fig)


def fig_resolution(dataset_map: DatasetMap, figures_dir: Path) -> None:
    table = stats.resolution_distribution(dataset_map)
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    top = table.head(8).copy()
    labels = [f"{int(w)} x {int(h)}" for w, h in zip(top["width"], top["height"])]
    bars = ax.barh(labels, top["images"], color=plotstyle.ACCENT)
    ax.invert_yaxis()
    plotstyle.style_h_barh(ax)
    plotstyle.bar_value_labels(ax, bars)
    ax.set_xlabel("Images")
    ax.xaxis.set_major_formatter(FuncFormatter(plotstyle.thousands))
    extra = len(table) - len(top)
    suffix = f" (top {len(top)} of {len(table)} resolutions)" if extra > 0 else ""
    ax.set_title(f"Image resolutions{suffix}")
    ax.margins(x=0.15)
    plotstyle.save_fig(fig, "fig08_image_resolutions", figures_dir)
    plt.close(fig)


def fig_centre_density(dataset_map: DatasetMap, figures_dir: Path) -> None:
    boxes = dataset_map.boxes.dropna(subset=["cx", "cy"])
    if boxes.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 6))
    counts, _, _ = np.histogram2d(boxes["cx"], boxes["cy"], bins=40, range=[[0, 1], [0, 1]])
    mesh = ax.imshow(
        counts.T, origin="upper", extent=(0, 1, 1, 0), cmap=plotstyle.BLUES_CMAP, aspect="equal",
    )
    ax.set_xlabel("Normalised x (left to right)")
    ax.set_ylabel("Normalised y (top to bottom)")
    ax.set_title("Bounding-box centre density")
    ax.grid(False)
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.85)
    cbar.set_label("Boxes", color=plotstyle.INK_SECONDARY)
    cbar.outline.set_visible(False)
    plotstyle.save_fig(fig, "fig09_box_centre_density", figures_dir)
    plt.close(fig)


def fig_class_cooccurrence(dataset_map: DatasetMap, figures_dir: Path) -> None:
    matrix = stats.class_cooccurrence(dataset_map)
    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    mesh = ax.imshow(matrix.to_numpy(), cmap=plotstyle.BLUES_CMAP)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title("Class co-occurrence (images containing both classes)")
    ax.grid(False)
    threshold = matrix.to_numpy().max() * 0.55
    for i, row_name in enumerate(matrix.index):
        for j, col_name in enumerate(matrix.columns):
            value = matrix.loc[row_name, col_name]
            ax.text(
                j, i, f"{value:,}", ha="center", va="center", fontsize=8.5,
                color="white" if value > threshold else plotstyle.INK_SECONDARY,
            )
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.8)
    cbar.set_label("Images", color=plotstyle.INK_SECONDARY)
    cbar.outline.set_visible(False)
    plotstyle.save_fig(fig, "fig10_class_cooccurrence", figures_dir)
    plt.close(fig)


CHARTS = (
    fig_dataset_composition,
    fig_class_distribution,
    fig_class_by_split,
    fig_instances_per_image,
    fig_bbox_area,
    fig_bbox_aspect,
    fig_size_classes,
    fig_resolution,
    fig_centre_density,
    fig_class_cooccurrence,
)


# --- Orchestration -----------------------------------------------------------

def write_tables(dataset_map: DatasetMap, csv_dir: Path) -> list[Path]:
    csv_dir.mkdir(parents=True, exist_ok=True)
    written = []
    tables = {
        "image_inventory.csv": dataset_map.images,
        "box_inventory.csv": dataset_map.boxes,
        "split_overview.csv": stats.split_overview(dataset_map),
        "class_distribution.csv": stats.class_distribution(dataset_map),
        "bbox_geometry_summary.csv": stats.bbox_geometry(dataset_map),
        "coco_size_classes.csv": stats.size_class_distribution(dataset_map),
        "resolution_distribution.csv": stats.resolution_distribution(dataset_map),
        "instances_per_image_histogram.csv": stats.instances_per_image_histogram(dataset_map),
        "class_cooccurrence.csv": stats.class_cooccurrence(dataset_map),
        "integrity_summary.csv": checks.integrity_summary(dataset_map),
        "integrity_issues.csv": checks.issues_table(dataset_map),
    }
    for name, frame in tables.items():
        target = csv_dir / name
        frame.to_csv(target, index=name == "class_cooccurrence.csv")
        written.append(target)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MDWD exploratory data analysis.")
    parser.add_argument(
        "--variant", default=config.DEFAULT_VARIANT,
        help=f"Dataset variant under Datasets/MDWD (default: {config.DEFAULT_VARIANT}).",
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Per-split image cap for a quick sample run (skips cross-split checks).",
    )
    parser.add_argument("--skip-charts", action="store_true", help="Tables and summary only.")
    parser.add_argument(
        "--out-tag", default=None,
        help="Write outputs to Documents/MDWD-EDA-<tag>/ instead of Documents/MDWD-EDA/ "
             "(use to keep a versioned run without overwriting the main outputs).",
    )
    args = parser.parse_args()

    eda_dir = config.EDA_DIR if not args.out_tag else config.EDA_DIR.with_name(f"MDWD-EDA-{args.out_tag}")
    csv_dir = eda_dir / "GeneratedCSVs"
    figures_dir = eda_dir / "Figures"

    print(f"Mapping {args.variant} (max_images={args.max_images or 'all'})...")
    dataset_map = build_map(args.variant, max_images=args.max_images)
    print(
        f"  {len(dataset_map.images):,} images / {len(dataset_map.boxes):,} boxes / "
        f"{len(dataset_map.class_names)} classes ({dataset_map.format.upper()} format)"
    )
    if dataset_map.issue_counts:
        print(f"  integrity issues: {dataset_map.issue_counts}")
    else:
        print("  integrity issues: none")

    written = write_tables(dataset_map, csv_dir)
    print(f"Wrote {len(written)} tables to {csv_dir}")

    if not args.skip_charts:
        plotstyle.apply()
        for chart in CHARTS:
            chart(dataset_map, figures_dir)
        print(f"Wrote {len(CHARTS)} figures to {figures_dir}")

    summary = stats.to_native(stats.summary_dict(dataset_map))
    summary["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary["max_images"] = args.max_images
    summary_path = eda_dir / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
