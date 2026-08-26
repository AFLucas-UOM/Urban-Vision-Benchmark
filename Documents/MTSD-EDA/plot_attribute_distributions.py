"""Plot final MTSD attribute distributions from the generated CSV outputs.

The script intentionally uses only the four attribute_distribution_*.csv files.
Run from the repository root:

    python Documents/MTSD-EDA/plot_attribute_distributions.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_DIR = REPO_ROOT / "Documents" / "MTSD-EDA" / "GeneratedCSVs"
OUTPUT_DIR = REPO_ROOT / "Documents" / "MTSD-EDA" / "Figures"
EXPECTED_FINAL_INSTANCES = 20_509

PANELS = [
    ("Viewing angle", "view_angle", "attribute_distribution_view_angle.csv"),
    ("Mounting type", "mounting", "attribute_distribution_mounting.csv"),
    ("Physical condition", "condition", "attribute_distribution_condition.csv"),
    ("Geometric shape", "sign_shape", "attribute_distribution_sign_shape.csv"),
]

INK = "#20242B"
GRID = "#D9DEE5"
BAR = "#0072B2"
ACCENT = "#E69F00"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": "#7A8491",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_attribute_csv(filename: str, key: str) -> pd.DataFrame:
    frame = pd.read_csv(CSV_DIR / filename)
    frame = frame[frame["value_kind"].isin(["regular", "fallback"])].copy()
    frame["annotations"] = pd.to_numeric(frame["annotations"], errors="raise")
    frame["pct_of_annotations"] = pd.to_numeric(
        frame["pct_of_annotations"], errors="raise"
    )
    if frame.empty or frame["annotations"].sum() <= 0:
        raise ValueError(f"No usable rows found in {filename}")
    frame["attribute"] = key
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    configure_style()
    frames = [load_attribute_csv(filename, key) for _, key, filename in PANELS]
    totals = {int(frame["annotations"].sum()) for frame in frames}
    if len(totals) != 1:
        raise ValueError(f"Attribute CSV totals disagree: {sorted(totals)}")
    csv_total = totals.pop()
    if csv_total != EXPECTED_FINAL_INSTANCES:
        raise ValueError(
            f"CSV snapshot contains {csv_total:,} instances, not the canonical "
            f"{EXPECTED_FINAL_INSTANCES:,}. Regenerate the attribute CSVs from "
            "the canonical Final-QA annotations."
        )

    shown_total = f"{csv_total:,} instances"

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=False)
    axes = axes.ravel()
    for ax, (title, _, _), frame in zip(axes, PANELS, frames):
        frame = frame.sort_values("annotations", ascending=True)
        bars = ax.barh(
            frame["value"],
            frame["annotations"],
            color=BAR,
            edgecolor=INK,
            linewidth=0.45,
            height=0.66,
        )
        xmax = frame["annotations"].max()
        ax.set_xlim(0, xmax * 1.27)
        ax.set_title(title, loc="left", fontweight="bold", pad=9)
        ax.set_xlabel("Instances")
        ax.xaxis.grid(True, color=GRID, linewidth=0.65)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#7A8491")
        ax.spines["bottom"].set_color("#7A8491")
        ax.tick_params(axis="y", length=0, pad=5)
        ax.tick_params(axis="x", length=3, color="#7A8491")
        labels = [
            f"{int(count):,}  ({pct:.2f}%)"
            for count, pct in zip(frame["annotations"], frame["pct_of_annotations"])
        ]
        ax.bar_label(bars, labels=labels, padding=5, fontsize=9.3, color=INK)

    fig.suptitle(
        "MTSD attribute distributions",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.952,
        f"Final-QA attribute annotations; {shown_total}",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#56616F",
    )
    fig.text(
        0.055,
        0.018,
        "Bars show instance count; labels show count and percentage of the attribute-labelled CSV snapshot.",
        ha="left",
        fontsize=8.7,
        color="#56616F",
    )
    fig.subplots_adjust(left=0.19, right=0.95, top=0.89, bottom=0.09, wspace=0.38, hspace=0.48)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "mtsd_attribute_distributions"
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote {stem.with_suffix('.png')}")
    print(f"Wrote {stem.with_suffix('.pdf')}")
    print(f"CSV instance total: {csv_total:,}; canonical expected: {EXPECTED_FINAL_INSTANCES:,}")


if __name__ == "__main__":
    main()
