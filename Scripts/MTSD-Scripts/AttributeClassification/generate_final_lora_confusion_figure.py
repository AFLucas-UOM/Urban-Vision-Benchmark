"""Create the final 1,890-crop LoRA confusion-matrix comparison figure.

The source JSON files are the final, non-smoke test metrics for V-JEPA 2.1-L
and DINOv3-L.  Each panel uses row-normalised counts and annotates the raw
count followed by the row percentage.  Rows are ground truth and columns are
predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
METRICS_DIR = ROOT / "Scripts" / "MTSD-Scripts" / "AttributeClassification" / "outputs" / "metrics"
OUTPUT_DIR = ROOT / "Final-Dissertation" / "figures"

MODELS = (
    ("vjepa21_vitl_lora", "V-JEPA 2.1-L + LoRA"),
    ("dinov3_vitl_lora", "DINOv3-L + LoRA"),
)
HEADS = (
    ("view_angle", "Viewing angle"),
    ("mounting", "Mounting type"),
    ("condition", "Physical condition"),
    ("sign_shape", "Geometric shape"),
)
CMAP = "Blues"


def read_metrics(variant: str) -> dict:
    path = METRICS_DIR / variant / "test_metrics.json"
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("split") != "test" or data.get("smoke_test"):
        raise ValueError(f"Expected final non-smoke test metrics: {path}")
    if data.get("evaluation", {}).get("n_images") != 1890:
        raise ValueError(f"Expected 1,890 test crops: {path}")
    return data


def display_class(label: str) -> str:
    return {"Pole-Mounted": "Pole-\nMounted", "Wall-Mounted": "Wall-\nMounted",
            "Heavily Damaged": "Heavily\nDamaged", "Quadrangle": "Quad-\nrangle",
            "Triangular": "Tri-\nangular", "Octagonal": "Octa-\ngonal"}.get(label, label)


def annotation(count: int, proportion: float) -> str:
    return f"{count:d}\n{proportion * 100:.1f}%"


def main() -> None:
    records = [(label, read_metrics(variant)) for variant, label in MODELS]
    fig, axes = plt.subplots(len(HEADS), len(records), figsize=(13.2, 17.2),
                             constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=3 / 72, h_pad=4 / 72,
                                    wspace=0.045, hspace=0.10)

    for row, (head_key, head_title) in enumerate(HEADS):
        for col, (model_label, record) in enumerate(records):
            ax = axes[row, col]
            head = record["attributes"][head_key]
            classes = head["classes"]
            counts = np.asarray(head["confusion_matrix"], dtype=int)
            support = counts.sum(axis=1)
            normalized = counts / support[:, None]
            image = ax.imshow(normalized, cmap=CMAP, vmin=0, vmax=1,
                              interpolation="nearest", aspect="equal")

            shown_classes = [display_class(item) for item in classes]
            ax.set_xticks(range(len(classes)), shown_classes, fontsize=9)
            ax.set_yticks(range(len(classes)),
                          [f"{display_class(item)}\n(n={n:,})"
                           for item, n in zip(classes, support)], fontsize=9)
            ax.tick_params(axis="x", length=0, pad=6)
            ax.tick_params(axis="y", length=0, pad=6)
            ax.set_xlabel("Predicted class", fontsize=10, labelpad=8)
            ax.set_ylabel("True class (support)", fontsize=10, labelpad=8)
            ax.set_title(f"{head_title} — {model_label}", fontsize=12,
                         fontweight="bold", pad=12)

            for i in range(len(classes)):
                for j in range(len(classes)):
                    value = normalized[i, j]
                    color = "white" if value >= 0.55 else "#17202A"
                    ax.text(j, i, annotation(counts[i, j], value), ha="center",
                            va="center", fontsize=9, fontweight="medium",
                            color=color, linespacing=1.25)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("#607080")

        # One shared, row-level scale makes the two models directly comparable.
        colorbar = fig.colorbar(image, ax=axes[row, :], fraction=0.024, pad=0.018)
        colorbar.set_label("Row-normalised proportion", fontsize=10)
        colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        colorbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

    fig.suptitle("Final test-set confusion matrices (1,890 crops)", fontsize=15,
                 fontweight="bold")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUTPUT_DIR / "attribute_lora_confusion_matrices_final_test"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight",
                facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
