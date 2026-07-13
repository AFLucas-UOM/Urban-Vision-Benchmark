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
import copy
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

    def vmeta(variant):
        """Variant metadata: prefer the metrics payload's stored metadata;
        fall back to the current config entry (covers historical runs that
        predate metadata recording)."""
        meta = details[variant].get("run_info", {}).get("variant_meta")
        if isinstance(meta, dict) and meta.get("family"):
            return meta
        from .variants import variant_metadata
        return variant_metadata(cfg["models"][variant], variant)

    def bmeta(variant, key, default="n/a"):
        meta = details[variant].get("run_info", {}).get("backbone_meta")
        if isinstance(meta, dict):
            return meta.get(key, default)
        return default

    csv_path = reports_dir / f"comparison{suffix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["variant", "run_id", "family", "architecture", "model_size",
             "adaptation", "resolution"] + attributes
            + ["mean_macro_f1", "val_mean_macro_f1", "best_epoch",
               "stopped_epoch", "stop_reason", "total_params",
               "trainable_params", "trainable_pct", "train_duration_s",
               "loaded_backend", "loaded_version"])
        for variant in present:
            val = info(variant, "val_mean_macro_f1")
            meta = vmeta(variant)
            writer.writerow(
                [variant, details[variant]["run_id"], meta["family"],
                 meta["architecture"], meta["model_size"],
                 info(variant, "adaptation", meta["adaptation"]),
                 meta["resolution"]]
                + [f"{rows[variant].get(a, float('nan')):.4f}" for a in attributes]
                + [f"{rows[variant]['mean']:.4f}",
                   f"{val:.4f}" if isinstance(val, float) else val,
                   info(variant, "best_epoch"), info(variant, "stopped_epoch"),
                   info(variant, "stop_reason"), pinfo(variant, "total"),
                   pinfo(variant, "trainable"), pinfo(variant, "trainable_pct"),
                   info(variant, "train_duration_s"),
                   bmeta(variant, "loaded_backend"),
                   bmeta(variant, "loaded_version")])

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
        "| Variant | Family | Architecture | Size | Adaptation | Res | "
        "Best epoch | Stopped at | Stop reason | Val mean macro-F1 | "
        "Total params | Trainable | Trainable % | Train time (s) | Backend |",
        "|" + "---|" * 15,
    ]
    def fmt(value, spec):
        return format(value, spec) if isinstance(value, (int, float)) else str(value)

    for variant in present:
        meta = vmeta(variant)
        backend = bmeta(variant, "loaded_backend")
        version = bmeta(variant, "loaded_version")
        backend_cell = (f"{backend} ({version})"
                        if version not in (None, "n/a") else str(backend))
        cells = [variant, meta["family"], meta["architecture"],
                 meta["model_size"],
                 str(info(variant, "adaptation", meta["adaptation"])),
                 str(meta["resolution"]),
                 str(info(variant, "best_epoch")),
                 str(info(variant, "stopped_epoch")),
                 str(info(variant, "stop_reason")),
                 fmt(info(variant, "val_mean_macro_f1"), ".4f"),
                 fmt(pinfo(variant, "total"), ","),
                 fmt(pinfo(variant, "trainable"), ","),
                 str(pinfo(variant, "trainable_pct")),
                 str(info(variant, "train_duration_s")),
                 backend_cell]
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


def _load_variant_row(cfg, variant, metrics_dir, suffix, attributes):
    """Load one variant's test metrics into a flat size-ablation row, or None."""
    path = metrics_dir / f"{variant}{suffix}" / "test_metrics.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    run_info = payload.get("run_info", {})
    meta = run_info.get("variant_meta")
    if not (isinstance(meta, dict) and meta.get("family")):
        from .variants import variant_metadata
        meta = variant_metadata(cfg["models"][variant], variant)
    parameters = run_info.get("parameters", {})
    backbone_meta = run_info.get("backbone_meta", {})
    if not isinstance(backbone_meta, dict):
        backbone_meta = {}
    row = {
        "variant": variant,
        "run_id": payload.get("run_id", "n/a"),
        "family": meta["family"],
        "model_size": meta["model_size"],
        "architecture": meta["architecture"],
        "adaptation": run_info.get("adaptation", meta["adaptation"]),
        "resolution": meta["resolution"],
        "total_params": parameters.get("total", "n/a"),
        "trainable_params": parameters.get("trainable", "n/a"),
        "best_epoch": run_info.get("best_epoch", "n/a"),
        "val_mean_macro_f1": run_info.get("val_mean_macro_f1", "n/a"),
        "test_mean_macro_f1": payload.get("mean_macro_f1"),
        "train_duration_s": run_info.get("train_duration_s", "n/a"),
        "test_images_per_s": run_info.get("test_images_per_s", "n/a"),
        "loaded_backend": backbone_meta.get("loaded_backend", "n/a"),
        "loaded_version": backbone_meta.get("loaded_version", "n/a"),
    }
    for attr in attributes:
        m = payload.get("attributes", {}).get(attr, {})
        row[f"{attr}_macro_f1"] = m.get("macro_f1")
        row[f"{attr}_accuracy"] = m.get("accuracy")
    return row


def _ablation_md_table(rows, attributes):
    """Markdown table for a group of size-ablation rows."""
    def num(value, spec=".4f"):
        return format(value, spec) if isinstance(value, (int, float)) else \
            str(value)

    header = (["Variant", "Family", "Architecture", "Size", "Adaptation",
               "Res", "Params", "Trainable", "Best epoch",
               "Val mean macro-F1", "Test mean macro-F1"]
              + [f"{a} F1" for a in attributes]
              + [f"{a} acc" for a in attributes]
              + ["Train time (s)", "Test im/s", "Backend"])
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for row in rows:
        backend = row["loaded_backend"]
        if row["loaded_version"] not in (None, "n/a"):
            backend = f"{backend} ({row['loaded_version']})"
        cells = ([row["variant"], row["family"], row["architecture"],
                  row["model_size"], row["adaptation"], str(row["resolution"]),
                  num(row["total_params"], ","),
                  num(row["trainable_params"], ","),
                  str(row["best_epoch"]),
                  num(row["val_mean_macro_f1"]),
                  num(row["test_mean_macro_f1"])]
                 + [num(row.get(f"{a}_macro_f1")) if row.get(f"{a}_macro_f1")
                    is not None else "n/a" for a in attributes]
                 + [num(row.get(f"{a}_accuracy")) if row.get(f"{a}_accuracy")
                    is not None else "n/a" for a in attributes]
                 + [str(row["train_duration_s"]),
                    str(row["test_images_per_s"]), backend])
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return lines


def write_size_ablation_report(cfg, smoke_test=False):
    """Consolidate the model-size ablation into its own report files.

    Covers the variants of the size_ablation_all profile (never the legacy
    variants — they duplicate equivalent new runs) and writes
    reports/size_ablation{suffix}.csv and .md, distinct from the historical
    comparison files, with sections grouped by (1) family + frozen,
    (2) family + LoRA, (3) ConvNeXt full fine-tuning, and (4) an all-frozen
    cross-family ranking. Adaptation modes are never mixed in one table
    without the table saying so.
    """
    from .variants import available_profiles

    profiles = available_profiles(cfg)
    members = profiles.get("size_ablation_all")
    if not members:
        log.warning("No size_ablation_all profile in the config; skipping "
                    "the size-ablation report")
        return None
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    attributes = list(cfg["attributes"])
    suffix = "-smoke" if smoke_test else ""

    rows, missing = [], []
    for variant in members:
        row = _load_variant_row(cfg, variant, metrics_dir, suffix, attributes)
        if row is None:
            missing.append(variant)
        else:
            rows.append(row)
    if not rows:
        log.info("No size-ablation variants have test metrics yet; skipping "
                 "the size-ablation report")
        return None

    csv_path = reports_dir / f"size_ablation{suffix}.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def sort_key(row):
        score = row["test_mean_macro_f1"]
        return -(score if isinstance(score, (int, float)) else -1.0)

    md_lines = [
        "# Model-size ablation: consolidated report",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        "Primary metric: **macro-F1** on the test split. Every table is "
        "labelled with its adaptation mode; frozen probes, LoRA adaptation, "
        "and full fine-tuning are never mixed in one table.",
        "",
        "## Full matrix (all completed size-ablation variants)",
        "",
    ]
    md_lines += _ablation_md_table(sorted(rows, key=sort_key), attributes)

    families = list(dict.fromkeys(r["family"] for r in rows))

    md_lines += ["", "## 1. Frozen linear probes, by family "
                     "(adaptation: frozen)", ""]
    for family in families:
        group = [r for r in rows
                 if r["family"] == family and r["adaptation"] == "frozen"]
        if group:
            md_lines += [f"### {family} (frozen)", ""]
            md_lines += _ablation_md_table(sorted(group, key=sort_key),
                                           attributes)
            md_lines.append("")

    md_lines += ["## 2. LoRA adaptation, by family (adaptation: lora)", ""]
    for family in families:
        group = [r for r in rows
                 if r["family"] == family and r["adaptation"] == "lora"]
        if group:
            md_lines += [f"### {family} (LoRA)", ""]
            md_lines += _ablation_md_table(sorted(group, key=sort_key),
                                           attributes)
            md_lines.append("")

    md_lines += ["## 3. ConvNeXt full fine-tuning (adaptation: finetune)", ""]
    finetune = [r for r in rows if r["family"] == "ConvNeXt"
                and r["adaptation"] == "finetune"]
    if finetune:
        md_lines += _ablation_md_table(sorted(finetune, key=sort_key),
                                       attributes)
    else:
        md_lines.append("_No completed runs yet._")
    md_lines.append("")

    md_lines += ["## 4. All frozen representation models, cross-family "
                 "ranking (adaptation: frozen)", ""]
    frozen = [r for r in rows if r["adaptation"] == "frozen"]
    if frozen:
        md_lines += _ablation_md_table(sorted(frozen, key=sort_key),
                                       attributes)
    else:
        md_lines.append("_No completed runs yet._")
    md_lines.append("")

    if missing:
        md_lines += [f"Size-ablation variants without saved test metrics: "
                     f"{', '.join(missing)} (skipped or not yet trained).", ""]

    md_path = reports_dir / f"size_ablation{suffix}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    log.info("Size-ablation report written (%d/%d variants): %s, %s",
             len(rows), len(members), csv_path, md_path)
    return md_path


def reevaluate_from_checkpoint(cfg, variant, split="test", checkpoint=None):
    """Rebuild a variant from its saved checkpoint metadata and re-evaluate it.

    The checkpoint's stored model_cfg (including adaptation mode and LoRA
    hyperparameters) drives reconstruction, so old frozen/fine-tuned
    checkpoints and new LoRA checkpoints are both evaluable. Metrics are saved
    exactly like the post-training evaluation.
    """
    from .backbones import build_backbone
    from .train_common import _checkpoint_adaptation, _load_checkpoint_state
    from .dataset import AttributeCropDataset, build_transforms
    from .multihead_model import MultiHeadClassifier, parameter_breakdown
    from torch.utils.data import DataLoader

    ckpt_path = Path(checkpoint) if checkpoint else (
        cfg["paths"]["checkpoints_dir"] / variant / "best.pt")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}; train the "
                                f"variant first")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = copy.deepcopy(ckpt["model_cfg"])
    adaptation = _checkpoint_adaptation(ckpt, model_cfg)
    model_cfg["adaptation"] = adaptation
    attributes = ckpt.get("attributes", cfg["attributes"])

    backbone = build_backbone(model_cfg)
    model = MultiHeadClassifier(backbone, attributes, ckpt["probe"], adaptation)
    _load_checkpoint_state(model, ckpt["model_state"], adaptation)
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
        # New checkpoints carry explicit metadata; old ones fall back to the
        # freshly built backbone's metadata and the config at report time.
        "variant_meta": ckpt.get("variant_meta"),
        "backbone_meta": (ckpt.get("backbone_meta")
                          or getattr(backbone, "backbone_meta", None)),
        "training_meta": ckpt.get("training_meta"),
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
        write_size_ablation_report(cfg, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
