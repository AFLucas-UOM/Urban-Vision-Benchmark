"""Render candidate contact sheets for selecting qualitative MDWD panels."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from make_mdwd_rfdetrm_qualitative_figure import (
    ANNOTATIONS, CLASS_COLORS, CONFIDENCE, ERROR_COLOR, MATCH_IOU,
    OUTPUT_DIR, PREDICTIONS, TEST_DIR, draw_box, match_image,
)

# C options prioritise a visible false-positive event; D options favour
# challenging, non-trivial scenes.  IDs map directly to the selection labels.
CANDIDATES = {
    "C": [36, 40, 81, 145, 171, 181, 235, 249, 310, 337],
    "D": [7, 81, 127, 134, 175, 181, 187, 264, 280, 357],
    "FN": [48, 59, 83, 127, 161, 162, 164, 206, 317, 357],
    "FP": [7, 134, 171, 187, 235, 264, 280, 320, 145, 181],
}


def records() -> dict[int, dict]:
    truth = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    predictions = [row for row in json.loads(PREDICTIONS.read_text(encoding="utf-8")) if row["score"] >= CONFIDENCE]
    categories = {row["id"]: row["name"] for row in truth["categories"]}
    images = {row["id"]: row for row in truth["images"]}
    gt, pred = defaultdict(list), defaultdict(list)
    for row in truth["annotations"]:
        gt[row["image_id"]].append({**row, "category_name": categories[row["category_id"]]})
    for row in predictions:
        pred[row["image_id"]].append({**row, "category_name": categories[row["category_id"]]})
    result = {}
    for image_id, image in images.items():
        tp, fp, fn = match_image(pred[image_id], gt[image_id])
        result[image_id] = {"image": image, "tp": tp, "fp": fp, "fn": fn}
    return result


def render(kind: str, ids: list[int], data: dict[int, dict]) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(20, 8.4), constrained_layout=True)
    notes = [f"RF-DETR-M qualitative candidate options ({kind}): confidence ≥ {CONFIDENCE:.2f}, class-aware IoU ≥ {MATCH_IOU:.2f}", ""]
    for index, (axis, image_id) in enumerate(zip(axes.flat, ids), start=1):
        row = data[image_id]
        image = ImageOps.exif_transpose(Image.open(TEST_DIR / row["image"]["file_name"])).convert("RGB")
        axis.imshow(image)
        for prediction in row["tp"]:
            draw_box(axis, prediction["bbox"], CLASS_COLORS[prediction["category_name"]], linewidth=2.0,
                     label=f"{prediction['score']:.2f}")
        for error in row["fp"]:
            draw_box(axis, error["bbox"], ERROR_COLOR, linestyle=":", linewidth=2.5, label="FP")
        for error in row["fn"]:
            draw_box(axis, error["bbox"], ERROR_COLOR, linestyle=":", linewidth=2.5, label="FN")
        axis.set_title(f"{kind}{index}  |  TP {len(row['tp'])}, FP {len(row['fp'])}, FN {len(row['fn'])}",
                       fontsize=10.5, loc="left", fontweight="semibold", pad=5)
        axis.set_axis_off()
        errors = ", ".join(sorted({item["category_name"] for item in row["fp"] + row["fn"]})) or "None"
        notes.append(f"{kind}{index}: ID {image_id}; {row['image']['file_name']}; TP={len(row['tp'])}, FP={len(row['fp'])}, FN={len(row['fn'])}; error class(es): {errors}")
    figure.legend(
        handles=[patches.Patch(facecolor="none", edgecolor=ERROR_COLOR, linestyle=":", label="False Positive (FP) / False Negative (FN)"),
                 *[patches.Patch(facecolor=color, edgecolor="none", label=name) for name, color in CLASS_COLORS.items()]],
        title="Taxonomy", loc="lower center", bbox_to_anchor=(0.5, -0.038), ncol=6, frameon=False,
        fontsize=9.5, title_fontsize=9.5,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / f"MDWD-RFDETRM-Options-{kind}.png", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    (OUTPUT_DIR / f"MDWD-RFDETRM-Options-{kind}.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")


def main() -> None:
    data = records()
    for kind, ids in CANDIDATES.items():
        render(kind, ids, data)
    print("Created C1–C10 and D1–D10 candidate sheets.")


if __name__ == "__main__":
    main()
