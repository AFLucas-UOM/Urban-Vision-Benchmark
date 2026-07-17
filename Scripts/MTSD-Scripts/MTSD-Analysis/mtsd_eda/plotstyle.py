"""Publication-quality matplotlib styling for the MTSD EDA.

Implements a validated colour system: an eight-slot categorical palette
(assigned in fixed order, never cycled), a single-hue sequential blue ramp
for magnitude encodings, and recessive chart chrome.  Categorical hues were
verified for colour-vision-deficiency separation (worst adjacent pair
Machado-2009 dE 24.2) and the ramp for lightness monotonicity.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import config

# Ink and chrome -------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots, fixed order - assign in sequence, never cycle.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#1baf7a",  # 2 aqua
    "#eda100",  # 3 yellow
    "#008300",  # 4 green
    "#4a3aa7",  # 5 violet
    "#e34948",  # 6 red
    "#e87ba4",  # 7 magenta
    "#eb6834",  # 8 orange
]
ACCENT = CATEGORICAL[0]

# Single-hue sequential ramp (blue steps 100 -> 700) for heatmaps/choropleths.
SEQUENTIAL_BLUES = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

BLUES_CMAP = LinearSegmentedColormap.from_list("mtsd_blues", SEQUENTIAL_BLUES)
FIG_DPI = 300


def apply() -> None:
    """Install the EDA style into matplotlib's rcParams."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "figure.figsize": (9.0, 5.0),
            "figure.dpi": 110,
            "savefig.dpi": FIG_DPI,
            "savefig.bbox": "tight",
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "text.color": INK,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        }
    )


def save_fig(fig: plt.Figure, name: str, figures_dir: Path = config.FIGURES_DIR) -> Path:
    """Persist a figure as a 300-dpi PNG under Documents/MTSD-EDA/Figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    target = figures_dir / f"{name}.png"
    fig.savefig(target)
    return target


def thousands(value: float, _pos=None) -> str:
    """Axis tick formatter: 12 345 -> '12,345'."""
    return f"{value:,.0f}"


def bar_value_labels(ax: plt.Axes, bars, fmt: str = "{:,.0f}", padding: int = 3) -> None:
    """Direct value labels on bars, in text ink (never the series colour)."""
    ax.bar_label(bars, fmt=fmt, padding=padding, fontsize=9, color=INK_SECONDARY)


def style_h_barh(ax: plt.Axes) -> None:
    """Horizontal bar convention: no vertical grid noise, baseline on x."""
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)


def heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    fmt: str = "{:,.0f}",
    cbar_label: str = "",
    annot: bool = True,
    hide_zeros: bool = True,
    annot_size: float = 8.0,
    cmap=None,
    vmin: float | None = None,
    vmax: float | None = None,
    row_flags: pd.Series | None = None,
):
    """Annotated matrix heatmap in the EDA style.

    ``matrix`` rows/columns become y/x tick labels. ``row_flags`` (optional,
    aligned with the rows) appends a marker to flagged row labels - used to
    mark classes below the minimum-sample threshold. Returns the AxesImage
    so callers can add further decoration.
    """
    data = matrix.to_numpy(dtype=float)
    mesh = ax.imshow(data, cmap=cmap or BLUES_CMAP, aspect="auto", vmin=vmin, vmax=vmax)

    row_labels = [str(label) for label in matrix.index]
    if row_flags is not None:
        row_labels = [
            f"{label} *" if flagged else label
            for label, flagged in zip(row_labels, row_flags)
        ]
    ax.set_xticks(range(matrix.shape[1]), [str(c) for c in matrix.columns],
                  rotation=45, ha="right")
    ax.set_yticks(range(matrix.shape[0]), row_labels)
    ax.grid(False)

    if annot:
        finite = data[np.isfinite(data)]
        threshold = finite.max() * 0.55 if finite.size and finite.max() > 0 else np.inf
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                if not np.isfinite(value) or (hide_zeros and value == 0):
                    continue
                ax.text(j, i, fmt.format(value), ha="center", va="center",
                        fontsize=annot_size,
                        color=SURFACE if value > threshold else INK_SECONDARY)

    cbar = ax.figure.colorbar(mesh, ax=ax, shrink=0.9, pad=0.015)
    cbar.set_label(cbar_label, color=INK_SECONDARY)
    cbar.outline.set_visible(False)
    return mesh
