"""Evaluation metrics, confusion-matrix plots, and the consolidated report.

Per attribute head: accuracy, macro-F1, per-class F1, and a confusion matrix,
saved as machine-readable JSON/CSV plus a plotted PNG. The consolidated report
compares all trained variants across attributes with macro-F1 as the primary
metric (class imbalance makes accuracy misleading, especially for condition).

Plot styling follows the dataviz reference palette: a single-hue blue
sequential ramp for confusion matrices (magnitude), fixed-order categorical
slots for model variants (identity), recessive grid and axis chrome, and
direct value labels on bars.
"""

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .dataset import MISSING_LABEL

log = logging.getLogger("mtsd_attr")

SEQUENTIAL_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                   "#256abf", "#1c5cab", "#104281", "#0d366b"]
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
               "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

CONFUSION_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)


def evaluate_model(model, loader, attributes, device, amp=False):
    """Run inference over a loader and compute per-attribute metrics.

    Crops with a missing label on a head are excluded from that head's metrics.

    Returns:
        dict attribute -> {"n", "accuracy", "macro_f1", "classes",
        "per_class_f1", "support", "confusion_matrix"}.
    """
    model.eval()
    names = list(attributes)
    preds = {a: [] for a in names}
    trues = {a: [] for a in names}
    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=amp):
                logits = model(images)
            for i, attr in enumerate(names):
                t = targets[:, i]
                mask = t != MISSING_LABEL
                if mask.any():
                    p = logits[attr].argmax(dim=1).cpu()
                    preds[attr].extend(p[mask].tolist())
                    trues[attr].extend(t[mask].tolist())
    metrics = {}
    for attr in names:
        classes = attributes[attr]["classes"]
        labels = list(range(len(classes)))
        y_true, y_pred = trues[attr], preds[attr]
        if not y_true:
            log.warning("Attribute %s: no labelled samples in this split", attr)
            metrics[attr] = {"n": 0, "accuracy": None, "macro_f1": None,
                             "classes": classes, "per_class_f1": {},
                             "support": {}, "confusion_matrix": []}
            continue
        per_class = f1_score(y_true, y_pred, labels=labels, average=None,
                             zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        metrics[attr] = {
            "n": len(y_true),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                       average="macro", zero_division=0)),
            "classes": classes,
            "per_class_f1": {c: float(f) for c, f in zip(classes, per_class)},
            "support": {c: int((np.array(y_true) == i).sum())
                        for i, c in enumerate(classes)},
            "confusion_matrix": cm.tolist(),
        }
    return metrics


def mean_macro_f1(metrics):
    """Mean of the per-attribute macro-F1 scores (heads with no data excluded)."""
    values = [m["macro_f1"] for m in metrics.values() if m["macro_f1"] is not None]
    return float(np.mean(values)) if values else 0.0


def plot_confusion_matrix(cm, classes, title, path):
    """Save a row-normalised confusion matrix heatmap with count annotations."""
    cm = np.asarray(cm, dtype=float)
    rows = cm.sum(axis=1, keepdims=True)
    normalised = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    side = max(3.2, 0.85 * len(classes) + 1.6)
    fig, ax = plt.subplots(figsize=(side, side), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.imshow(normalised, cmap=CONFUSION_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right",
                  fontsize=8, color=INK_SECONDARY)
    ax.set_yticks(range(len(classes)), classes, fontsize=8, color=INK_SECONDARY)
    ax.set_xlabel("Predicted", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("True", fontsize=9, color=INK_SECONDARY)
    ax.set_title(title, fontsize=10, color=INK_PRIMARY)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)
    ax.tick_params(color=BASELINE)
    for i in range(len(classes)):
        for j in range(len(classes)):
            colour = "#ffffff" if normalised[i, j] > 0.55 else INK_PRIMARY
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    fontsize=8, color=colour)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def save_metrics_bundle(metrics_dir, variant, split, metrics, run_id,
                        smoke_test=False, run_info=None):
    """Write one split's metrics as JSON + per-class-F1 CSV + confusion PNGs.

    Smoke-test outputs go to a "<variant>-smoke" folder so they never mix with
    real results. run_info (best/stopping epoch, stop reason, adaptation mode,
    parameter breakdown) is embedded in the JSON for the consolidated report.
    """
    folder = Path(metrics_dir) / (f"{variant}-smoke" if smoke_test else variant)
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "variant": variant,
        "split": split,
        "smoke_test": smoke_test,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mean_macro_f1": mean_macro_f1(metrics),
        "run_info": run_info or {},
        "attributes": metrics,
    }
    with open(folder / f"{split}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    with open(folder / f"{split}_per_class_f1.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "split", "attribute", "class", "f1", "support"])
        for attr, m in metrics.items():
            for cls in m["classes"]:
                writer.writerow([variant, split, attr,
                                 cls, f"{m['per_class_f1'].get(cls, 0.0):.4f}",
                                 m["support"].get(cls, 0)])
    for attr, m in metrics.items():
        if m["confusion_matrix"]:
            plot_confusion_matrix(
                m["confusion_matrix"], m["classes"],
                f"{variant} · {attr} ({split})",
                folder / f"confusion_{split}_{attr}.png")
    log.info("Metrics for %s/%s saved to %s", variant, split, folder)


def _plot_comparison(rows, variants, attributes, path):
    """Grouped bar chart of test macro-F1 per attribute (plus mean) per variant."""
    groups = attributes + ["mean"]
    x = np.arange(len(groups))
    width = min(0.22, 0.7 / max(1, len(variants)))
    fig, ax = plt.subplots(figsize=(1.9 * len(groups) + 1.5, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for k, variant in enumerate(variants):
        values = [rows[variant].get(g) for g in groups]
        offset = (k - (len(variants) - 1) / 2) * (width + 0.02)
        bars = ax.bar(x + offset, [v or 0 for v in values], width,
                      label=variant, color=CATEGORICAL[k % len(CATEGORICAL)],
                      linewidth=0)
        for bar, value in zip(bars, values):
            if value is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012,
                        f"{value:.2f}", ha="center", va="bottom", fontsize=7.5,
                        color=INK_PRIMARY)
    ax.set_xticks(x, groups, fontsize=9, color=INK_SECONDARY)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro-F1 (test)", fontsize=9, color=INK_SECONDARY)
    ax.set_title("Attribute classification: model comparison",
                 fontsize=11, color=INK_PRIMARY, loc="left")
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED)
    ax.legend(frameon=False, fontsize=8, ncols=min(3, len(variants)),
              loc="lower right", bbox_to_anchor=(1.0, 1.0),
              labelcolor=INK_SECONDARY)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def write_comparison_report(cfg, smoke_test=False):
    """Consolidate saved test metrics for every completed variant into one report.

    Every variant defined under cfg["models"] that has saved test metrics is
    included, in config order — the report never assumes a fixed variant count.
    Produces reports/comparison.csv, comparison.md, and comparison_macro_f1.png
    (suffixed -smoke when consolidating smoke-test outputs).
    """
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    attributes = list(cfg["attributes"])
    suffix = "-smoke" if smoke_test else ""

    rows, details, missing = {}, {}, []
    for variant in cfg["models"]:
        path = metrics_dir / f"{variant}{suffix}" / "test_metrics.json"
        if not path.is_file():
            missing.append(variant)
            continue
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        rows[variant] = {a: payload["attributes"][a]["macro_f1"]
                         for a in attributes if a in payload["attributes"]}
        rows[variant]["mean"] = payload["mean_macro_f1"]
        details[variant] = payload

    present = list(rows)
    if not present:
        log.warning("No test metrics found for any variant; nothing to report")
        return None

    def info(variant, key, default="n/a"):
        return details[variant].get("run_info", {}).get(key, default)

    def pinfo(variant, key):
        parameters = info(variant, "parameters", {})
        return parameters.get(key, "n/a") if isinstance(parameters, dict) else "n/a"

    csv_path = reports_dir / f"comparison{suffix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["variant", "run_id", "adaptation"] + attributes
            + ["mean_macro_f1", "val_mean_macro_f1", "best_epoch",
               "stopped_epoch", "stop_reason", "total_params",
               "trainable_params", "trainable_pct"])
        for variant in present:
            val = info(variant, "val_mean_macro_f1")
            writer.writerow(
                [variant, details[variant]["run_id"], info(variant, "adaptation")]
                + [f"{rows[variant].get(a, float('nan')):.4f}" for a in attributes]
                + [f"{rows[variant]['mean']:.4f}",
                   f"{val:.4f}" if isinstance(val, float) else val,
                   info(variant, "best_epoch"), info(variant, "stopped_epoch"),
                   info(variant, "stop_reason"), pinfo(variant, "total"),
                   pinfo(variant, "trainable"), pinfo(variant, "trainable_pct")])

    md_lines = [
        "# Attribute classification: consolidated comparison",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        "Primary metric: **macro-F1** on the test split (robust to class "
        "imbalance; accuracy shown for reference).",
        "",
        "## Macro-F1 by attribute",
        "",
        "| Variant | " + " | ".join(attributes) + " | Mean |",
        "|" + "---|" * (len(attributes) + 2),
    ]
    for variant in present:
        cells = [f"{rows[variant].get(a, float('nan')):.4f}" for a in attributes]
        md_lines.append(f"| {variant} | " + " | ".join(cells)
                        + f" | **{rows[variant]['mean']:.4f}** |")
    md_lines += ["", "## Accuracy by attribute", "",
                 "| Variant | " + " | ".join(attributes) + " |",
                 "|" + "---|" * (len(attributes) + 1)]
    for variant in present:
        cells = [f"{details[variant]['attributes'][a]['accuracy']:.4f}"
                 if a in details[variant]["attributes"] else "n/a"
                 for a in attributes]
        md_lines.append(f"| {variant} | " + " | ".join(cells) + " |")

    md_lines += [
        "", "## Run details", "",
        "| Variant | Adaptation | Best epoch | Stopped at | Stop reason | "
        "Val mean macro-F1 | Total params | Trainable | Trainable % |",
        "|" + "---|" * 9,
    ]
    def fmt(value, spec):
        return format(value, spec) if isinstance(value, (int, float)) else str(value)

    for variant in present:
        cells = [variant, info(variant, "adaptation"),
                 str(info(variant, "best_epoch")),
                 str(info(variant, "stopped_epoch")),
                 str(info(variant, "stop_reason")),
                 fmt(info(variant, "val_mean_macro_f1"), ".4f"),
                 fmt(pinfo(variant, "total"), ","),
                 fmt(pinfo(variant, "trainable"), ","),
                 str(pinfo(variant, "trainable_pct"))]
        md_lines.append("| " + " | ".join(cells) + " |")

    for attr in attributes:
        classes = cfg["attributes"][attr]["classes"]
        md_lines += ["", f"## Per-class F1: {attr}", "",
                     "| Class | Support | " + " | ".join(present) + " |",
                     "|" + "---|" * (len(present) + 2)]
        for cls in classes:
            support = next(
                (details[v]["attributes"][attr]["support"].get(cls, 0)
                 for v in present if attr in details[v]["attributes"]), 0)
            cells = []
            for variant in present:
                m = details[variant]["attributes"].get(attr, {})
                f1 = m.get("per_class_f1", {}).get(cls)
                cells.append(f"{f1:.4f}" if f1 is not None else "n/a")
            md_lines.append(f"| {cls} | {support} | " + " | ".join(cells) + " |")

    md_lines += ["", "## Confusion matrices", ""]
    for variant in present:
        links = ", ".join(
            f"[{attr}](../metrics/{variant}{suffix}/confusion_test_{attr}.png)"
            for attr in attributes)
        md_lines.append(f"- {variant}: {links}")

    if missing:
        md_lines += ["", f"Variants without saved test metrics: "
                         f"{', '.join(missing)} (skipped or not yet trained)."]
    md_lines.append("")
    md_path = reports_dir / f"comparison{suffix}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    _plot_comparison(rows, present, attributes,
                     reports_dir / f"comparison_macro_f1{suffix}.png")
    log.info("Consolidated report written for %s: %s, %s",
             present, csv_path, md_path)
    return md_path


def reevaluate_from_checkpoint(cfg, variant, split="test", checkpoint=None):
    """Rebuild a variant from its saved checkpoint metadata and re-evaluate it.

    The checkpoint's stored model_cfg (including adaptation mode and LoRA
    hyperparameters) drives reconstruction, so old frozen/fine-tuned
    checkpoints and new LoRA checkpoints are both evaluable. Metrics are saved
    exactly like the post-training evaluation.
    """
    from .backbones import build_backbone
    from .config import adaptation_of
    from .dataset import AttributeCropDataset, build_transforms
    from .multihead_model import MultiHeadClassifier, parameter_breakdown
    from torch.utils.data import DataLoader

    ckpt_path = Path(checkpoint) if checkpoint else (
        cfg["paths"]["checkpoints_dir"] / variant / "best.pt")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}; train the "
                                f"variant first")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["model_cfg"]
    adaptation = ckpt.get("adaptation") or adaptation_of(model_cfg)
    attributes = ckpt.get("attributes", cfg["attributes"])

    backbone = build_backbone(model_cfg)
    model = MultiHeadClassifier(backbone, attributes, ckpt["probe"], adaptation)
    if adaptation == "finetune":
        model.load_state_dict(ckpt["model_state"])
    else:
        result = model.load_state_dict(ckpt["model_state"], strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"Checkpoint has unexpected keys: "
                               f"{result.unexpected_keys[:5]}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    manifest_path = cfg["paths"]["manifest"]
    with open(manifest_path, encoding="utf-8-sig") as f:
        manifest = json.load(f)
    records = [r for r in manifest["records"] if r["split"] == split]
    dataset = AttributeCropDataset(
        records, attributes, cfg["paths"]["subproject_root"],
        build_transforms(backbone.image_size, False,
                         cfg["training"]["augmentation"]))
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"],
                        num_workers=0, pin_memory=device.type == "cuda")
    log.info("Re-evaluating %s (%s, epoch %s) on %d %s crops",
             variant, adaptation, ckpt.get("epoch"), len(dataset), split)
    metrics = evaluate_model(model, loader, attributes, device)
    for attr, m in metrics.items():
        log.info("  %-12s acc=%s macro_f1=%s", attr,
                 f"{m['accuracy']:.4f}" if m["accuracy"] is not None else "n/a",
                 f"{m['macro_f1']:.4f}" if m["macro_f1"] is not None else "n/a")
    run_info = {
        "adaptation": adaptation,
        "best_epoch": ckpt.get("epoch"),
        "stopped_epoch": ckpt.get("stopped_epoch", "n/a"),
        "stop_reason": ckpt.get("stop_reason", "n/a"),
        "max_epochs": ckpt.get("max_epochs", "n/a"),
        "val_mean_macro_f1": ckpt.get("val_mean_macro_f1", "n/a"),
        "parameters": ckpt.get("parameters") or parameter_breakdown(model),
    }
    save_metrics_bundle(cfg["paths"]["metrics_dir"], variant, split, metrics,
                        ckpt.get("run_id", "reeval"), run_info=run_info)
    return metrics


def main():
    """CLI: consolidate the comparison report, or re-evaluate one checkpoint.

    Without --variant, writes the consolidated comparison across all variants
    with saved test metrics. With --variant, rebuilds that variant from its
    best checkpoint's metadata and re-evaluates it on --split.
    """
    from .config import load_config, setup_logging

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Consolidate smoke-test metrics instead")
    parser.add_argument("--variant", default=None,
                        help="Re-evaluate this variant from its checkpoint")
    parser.add_argument("--split", default="test", choices=["val", "test"],
                        help="Split to re-evaluate (with --variant)")
    parser.add_argument("--checkpoint", default=None,
                        help="Explicit checkpoint path (with --variant)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging()
    if args.variant:
        reevaluate_from_checkpoint(cfg, args.variant, args.split,
                                   args.checkpoint)
    else:
        write_comparison_report(cfg, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
