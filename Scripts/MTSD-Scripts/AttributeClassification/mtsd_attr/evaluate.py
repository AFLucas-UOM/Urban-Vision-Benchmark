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
from statistics import median
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

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


def _report_value(value, default="n/a"):
    """Render missing report values consistently without literal ``None``."""
    return default if value is None else value


def _macro_f1_from_confusion(cm):
    """Compute macro-F1 from a fixed-label confusion matrix."""
    cm = np.asarray(cm, dtype=float)
    true_positive = np.diag(cm)
    precision = np.divide(
        true_positive, cm.sum(axis=0), out=np.zeros_like(true_positive),
        where=cm.sum(axis=0) > 0)
    recall = np.divide(
        true_positive, cm.sum(axis=1), out=np.zeros_like(true_positive),
        where=cm.sum(axis=1) > 0)
    f1 = np.divide(
        2.0 * precision * recall, precision + recall,
        out=np.zeros_like(precision), where=(precision + recall) > 0)
    return float(np.mean(f1))


def _bootstrap_ci(samples, seed=None):
    """Return a JSON-safe percentile 95% confidence-interval summary."""
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return None
    result = {
        "level": 0.95,
        "method": "percentile",
        "lower": float(np.percentile(samples, 2.5)),
        "upper": float(np.percentile(samples, 97.5)),
        "n_bootstrap": int(len(samples)),
    }
    if seed is not None:
        result["seed"] = int(seed)
    return result


def bootstrap_macro_f1_shared(y_true_by_attr, y_pred_by_attr,
                              labels_by_attr, n_bootstrap=2000, seed=42):
    """Bootstrap aligned multi-head macro-F1 scores with shared crop indices.

    Inputs are aligned by crop index and may contain ``MISSING_LABEL`` in the
    true-label arrays. One crop-index resample is reused across all heads;
    missing labels are masked per head before computing macro-F1. The mean
    samples average only heads with valid labels for that iteration, matching
    :func:`mean_macro_f1`'s exclusion rule.
    """
    names = list(y_true_by_attr)
    if set(names) != set(y_pred_by_attr) or set(names) != set(labels_by_attr):
        raise ValueError("Bootstrap head mappings must have identical keys")
    n_samples = len(next(iter(y_true_by_attr.values()), []))
    if any(len(y_true_by_attr[attr]) != n_samples or
           len(y_pred_by_attr[attr]) != n_samples for attr in names):
        raise ValueError("Bootstrap head arrays must be aligned by crop")
    n_bootstrap = int(n_bootstrap)
    per_attr = {attr: np.full(n_bootstrap, np.nan, dtype=float)
                for attr in names}
    mean_samples = np.full(n_bootstrap, np.nan, dtype=float)
    if n_bootstrap <= 0 or n_samples == 0:
        return per_attr, mean_samples

    true_arrays = {attr: np.asarray(y_true_by_attr[attr], dtype=np.int64)
                   for attr in names}
    pred_arrays = {attr: np.asarray(y_pred_by_attr[attr], dtype=np.int64)
                   for attr in names}
    rng = np.random.default_rng(int(seed))
    for iteration in range(n_bootstrap):
        # One crop resample is deliberately reused across every head.
        indices = rng.integers(0, n_samples, size=n_samples)
        scores = []
        for attr in names:
            y_true = true_arrays[attr][indices]
            y_pred = pred_arrays[attr][indices]
            valid = y_true != MISSING_LABEL
            if not np.any(valid):
                continue
            cm = confusion_matrix(
                y_true[valid], y_pred[valid],
                labels=list(labels_by_attr[attr]))
            score = _macro_f1_from_confusion(cm)
            per_attr[attr][iteration] = score
            scores.append(score)
        if scores:
            mean_samples[iteration] = float(np.mean(scores))
    return per_attr, mean_samples


def bootstrap_macro_f1(y_true, y_pred, labels, n_bootstrap=2000, seed=42):
    """Return a reproducible percentile-bootstrap 95% CI for one head."""
    per_attr, _ = bootstrap_macro_f1_shared(
        {"head": y_true}, {"head": y_pred}, {"head": labels},
        n_bootstrap=n_bootstrap, seed=seed)
    return _bootstrap_ci(per_attr["head"], seed=seed)


def _mean_metric(metrics, key):
    values = [m.get(key) for m in metrics.values() if m.get(key) is not None]
    return float(np.mean(values)) if values else 0.0


def mean_macro_precision(metrics):
    """Mean of the per-attribute macro-precision scores."""
    return _mean_metric(metrics, "macro_precision")


def mean_macro_recall(metrics):
    """Mean of the per-attribute macro-recall scores."""
    return _mean_metric(metrics, "macro_recall")


def _cuda_synchronize(device):
    """Synchronize CUDA only when the evaluation device is CUDA."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _forward(model, images, device, amp):
    """Run one model forward pass using the configured autocast mode."""
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=amp):
        return model(images)


def _timing_warmup(model, loader, attributes, device, amp, warmup_batches):
    """Run untimed forward batches before repeated timing passes."""
    if warmup_batches <= 0:
        return
    with torch.no_grad():
        for batch_index, (images, _, _) in enumerate(loader):
            if batch_index >= warmup_batches:
                break
            images = images.to(device, non_blocking=True)
            logits = _forward(model, images, device, amp)
            for attr in attributes:
                logits[attr].argmax(dim=1)
    _cuda_synchronize(device)


def _time_end_to_end(model, loader, attributes, device, amp):
    """Time loader iteration, transfer, forward, and prediction extraction."""
    _cuda_synchronize(device)
    started = time.perf_counter()
    n_images = 0
    with torch.no_grad():
        for images, _, _ in loader:
            n_images += len(images)
            images = images.to(device, non_blocking=True)
            logits = _forward(model, images, device, amp)
            for attr in attributes:
                # CPU transfer is part of extracting predictions for metrics.
                logits[attr].argmax(dim=1).cpu()
    _cuda_synchronize(device)
    return n_images, time.perf_counter() - started


def _time_model_forward(model, loader, device, amp):
    """Time only forward passes after each batch is prepared on the device."""
    total_duration = 0.0
    n_images = 0
    with torch.no_grad():
        for images, _, _ in loader:
            n_images += len(images)
            device_images = images.to(device, non_blocking=True)
            _cuda_synchronize(device)
            started = time.perf_counter()
            _forward(model, device_images, device, amp)
            _cuda_synchronize(device)
            total_duration += time.perf_counter() - started
    return n_images, total_duration


def _measure_timing(model, loader, attributes, device, amp,
                    timing_warmup_batches, timing_repeats, n_images):
    """Return median repeated end-to-end and forward-only timings."""
    timing_repeats = max(0, int(timing_repeats))
    warmup_batches = max(0, int(timing_warmup_batches))
    result = {
        "n_images": int(n_images),
        "timing_repeats": timing_repeats,
        "timing_warmup_batches": warmup_batches,
        "end_to_end_duration_s": None,
        "end_to_end_images_per_s": None,
        "model_forward_duration_s": None,
        "model_forward_ms_per_image": None,
        # Historical aliases; populated when timing is measured.
        "duration_s": None,
        "images_per_s": None,
        "ms_per_image": None,
    }
    if timing_repeats == 0 or n_images == 0:
        return result
    _timing_warmup(model, loader, attributes, device, amp, warmup_batches)
    end_to_end_durations = []
    forward_durations = []
    for _ in range(timing_repeats):
        measured_images, duration = _time_end_to_end(
            model, loader, attributes, device, amp)
        _, forward_duration = _time_model_forward(
            model, loader, device, amp)
        if measured_images != n_images:
            raise RuntimeError("Timing loader size changed between evaluation passes")
        end_to_end_durations.append(duration)
        forward_durations.append(forward_duration)
    end_to_end = float(median(end_to_end_durations))
    model_forward = float(median(forward_durations))
    result.update({
        "end_to_end_duration_s": round(end_to_end, 4),
        "end_to_end_images_per_s": round(n_images / end_to_end, 2)
        if end_to_end else None,
        "model_forward_duration_s": round(model_forward, 4),
        "model_forward_ms_per_image": round(model_forward * 1000.0 / n_images, 4)
        if n_images else None,
    })
    # Legacy aliases retained for historical callers and payload readers.
    result.update({
        "duration_s": result["end_to_end_duration_s"],
        "images_per_s": result["end_to_end_images_per_s"],
        "ms_per_image": round(end_to_end * 1000.0 / n_images, 4)
        if n_images else None,
    })
    return result


def evaluate_model(model, loader, attributes, device, amp=False,
                   bootstrap_samples=0, bootstrap_seed=42,
                   return_timing=False, timing_warmup_batches=0,
                   timing_repeats=1):
    """Run inference over a loader and compute per-attribute metrics.

    Crops with a missing label on a head are excluded from that head's metrics.

    Returns:
        dict attribute -> {"n", "accuracy", "macro_precision",
        "macro_recall", "macro_f1", "macro_f1_ci", "classes",
        "per_class_precision", "per_class_recall", "per_class_f1",
        "support", "confusion_matrix"}. If return_timing is true, returns
        ``(metrics, timing)`` where timing separates repeated end-to-end
        evaluation-loop and model-forward measurements.
    """
    model.eval()
    names = list(attributes)
    # Each head keeps one entry per crop. Positions are the original crop
    # indices, so shared bootstrap resampling preserves cross-head alignment.
    preds = {a: [] for a in names}
    trues = {a: [] for a in names}
    sample_indices = []
    n_images = 0
    with torch.no_grad():
        for images, targets, _ in loader:
            batch_indices = list(range(n_images, n_images + len(images)))
            sample_indices.extend(batch_indices)
            n_images += len(images)
            images = images.to(device, non_blocking=True)
            logits = _forward(model, images, device, amp)
            for i, attr in enumerate(names):
                t = targets[:, i]
                p = logits[attr].argmax(dim=1).cpu()
                preds[attr].extend(p.tolist())
                trues[attr].extend(t.tolist())
    if len(sample_indices) != n_images:
        raise RuntimeError("Evaluation crop indices are not contiguous")
    bootstrap_per_attr, bootstrap_mean = bootstrap_macro_f1_shared(
        trues, preds,
        {attr: list(range(len(attributes[attr]["classes"]))) for attr in names},
        n_bootstrap=int(bootstrap_samples), seed=int(bootstrap_seed))
    metrics = {}
    for attr in names:
        classes = attributes[attr]["classes"]
        labels = list(range(len(classes)))
        all_true = np.asarray(trues[attr], dtype=np.int64)
        all_pred = np.asarray(preds[attr], dtype=np.int64)
        valid = all_true != MISSING_LABEL
        y_true, y_pred = all_true[valid].tolist(), all_pred[valid].tolist()
        if not y_true:
            log.warning("Attribute %s: no labelled samples in this split", attr)
            metrics[attr] = {"n": 0, "accuracy": None,
                             "macro_precision": None, "macro_recall": None,
                             "macro_f1": None, "macro_f1_ci": None,
                             "classes": classes, "per_class_precision": {},
                             "per_class_recall": {}, "per_class_f1": {},
                             "support": {}, "confusion_matrix": []}
            continue
        per_class_precision = precision_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0)
        per_class_recall = recall_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0)
        per_class_f1 = f1_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        metrics[attr] = {
            "n": len(y_true),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision_score(
                y_true, y_pred, labels=labels, average="macro",
                zero_division=0)),
            "macro_recall": float(recall_score(
                y_true, y_pred, labels=labels, average="macro",
                zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                       average="macro", zero_division=0)),
            "macro_f1_ci": _bootstrap_ci(
                bootstrap_per_attr[attr], seed=int(bootstrap_seed)),
            "classes": classes,
            "per_class_precision": {
                c: float(value) for c, value in zip(classes, per_class_precision)},
            "per_class_recall": {
                c: float(value) for c, value in zip(classes, per_class_recall)},
            "per_class_f1": {
                c: float(value) for c, value in zip(classes, per_class_f1)},
            "support": {c: int((np.array(y_true) == i).sum())
                        for i, c in enumerate(classes)},
            "confusion_matrix": cm.tolist(),
        }
    if return_timing:
        timing = _measure_timing(
            model, loader, names, device, amp, timing_warmup_batches,
            timing_repeats, n_images)
    else:
        timing = {"n_images": int(n_images), "timing_repeats": 0,
                  "timing_warmup_batches": 0,
                  "end_to_end_duration_s": None,
                  "end_to_end_images_per_s": None,
                  "model_forward_duration_s": None,
                  "model_forward_ms_per_image": None,
                  "duration_s": None, "images_per_s": None,
                  "ms_per_image": None}
    timing["mean_macro_f1_ci"] = _bootstrap_ci(
        bootstrap_mean, seed=int(bootstrap_seed))
    return (metrics, timing) if return_timing else metrics


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
                        smoke_test=False, run_info=None, evaluation_meta=None):
    """Write one split's metrics as JSON + per-class CSV + confusion PNGs.

    Smoke-test outputs go to a "<variant>-smoke" folder so they never mix with
    real results. run_info (best/stopping epoch, stop reason, adaptation mode,
    parameter breakdown) is embedded in the JSON for the consolidated report.
    """
    folder = Path(metrics_dir) / (f"{variant}-smoke" if smoke_test else variant)
    folder.mkdir(parents=True, exist_ok=True)
    evaluation_meta = evaluation_meta or {}
    payload = {
        "run_id": run_id,
        "variant": variant,
        "split": split,
        "smoke_test": smoke_test,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mean_macro_precision": mean_macro_precision(metrics),
        "mean_macro_recall": mean_macro_recall(metrics),
        "mean_macro_f1": mean_macro_f1(metrics),
        "mean_macro_f1_ci": evaluation_meta.get("mean_macro_f1_ci"),
        "evaluation_split": split,
        "evaluation": {
            key: evaluation_meta.get(key)
            for key in ("n_images", "timing_repeats",
                        "timing_warmup_batches", "end_to_end_duration_s",
                        "end_to_end_images_per_s",
                        "model_forward_duration_s",
                        "model_forward_ms_per_image")
            if key in evaluation_meta
        },
        "run_info": run_info or {},
        "attributes": metrics,
    }
    with open(folder / f"{split}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    with open(folder / f"{split}_per_class_f1.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "split", "attribute", "class",
                         "precision", "recall", "f1", "support"])
        for attr, m in metrics.items():
            for cls in m["classes"]:
                writer.writerow([variant, split, attr,
                                 cls,
                                 f"{m.get('per_class_precision', {}).get(cls, 0.0):.4f}",
                                 f"{m.get('per_class_recall', {}).get(cls, 0.0):.4f}",
                                 f"{m.get('per_class_f1', {}).get(cls, 0.0):.4f}",
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
        attributes_payload = payload.get("attributes", {})
        rows[variant] = {a: attributes_payload[a].get("macro_f1")
                         for a in attributes if a in attributes_payload}
        f1_values = [m.get("macro_f1") for m in attributes_payload.values()
                     if m.get("macro_f1") is not None]
        rows[variant]["mean"] = _report_value(
            payload.get("mean_macro_f1",
                        float(np.mean(f1_values)) if f1_values else None))
        details[variant] = payload

    present = list(rows)
    if not present:
        log.warning("No test metrics found for any variant; nothing to report")
        return None

    def info(variant, key, default="n/a"):
        run_info = details[variant].get("run_info") or {}
        return _report_value(
            run_info.get(key, default), default)

    def pinfo(variant, key):
        parameters = info(variant, "parameters", {})
        value = parameters.get(key, "n/a") if isinstance(parameters, dict) else "n/a"
        return _report_value(value)

    def vmeta(variant):
        """Variant metadata: prefer the metrics payload's stored metadata;
        fall back to the current config entry (covers historical runs that
        predate metadata recording)."""
        meta = (details[variant].get("run_info") or {}).get("variant_meta")
        if isinstance(meta, dict) and meta.get("family"):
            return meta
        from .variants import variant_metadata
        return variant_metadata(cfg["models"][variant], variant)

    def bmeta(variant, key, default="n/a"):
        meta = (details[variant].get("run_info") or {}).get("backbone_meta")
        if isinstance(meta, dict):
            return _report_value(meta.get(key, default), default)
        return default

    def csv_num(value):
        value = _report_value(value)
        return f"{value:.4f}" if isinstance(value, (int, float)) else value

    def eval_info(variant, key, legacy_key=None):
        """Read neutral test timing, then new payload, then legacy aliases."""
        payload = details[variant]
        run_info = payload.get("run_info") or {}
        evaluation = payload.get("evaluation") or {}
        value = run_info.get(key)
        if value is None:
            value = evaluation.get(key.removeprefix("eval_"))
        if value is None and legacy_key:
            value = run_info.get(legacy_key)
        return _report_value(value)

    def ci_bounds(metric):
        ci = metric.get("macro_f1_ci") or {}
        return ci.get("lower", "n/a"), ci.get("upper", "n/a")

    csv_path = reports_dir / f"comparison{suffix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["variant", "run_id", "family", "architecture", "model_size",
             "adaptation", "resolution"]
            + [f"{a}_macro_precision" for a in attributes]
            + [f"{a}_macro_recall" for a in attributes]
            + [f"{a}_macro_f1" for a in attributes]
            + [f"{a}_macro_f1_ci_lower" for a in attributes]
            + [f"{a}_macro_f1_ci_upper" for a in attributes]
            + ["mean_macro_precision", "mean_macro_recall", "mean_macro_f1",
               "mean_macro_f1_ci_lower", "mean_macro_f1_ci_upper",
               "val_mean_macro_f1", "best_epoch",
               "stopped_epoch", "stop_reason", "total_params",
               "trainable_params", "trainable_pct", "train_duration_s",
               "test_end_to_end_duration_s", "test_end_to_end_images_per_s",
               "test_model_forward_duration_s",
               "test_model_forward_ms_per_image", "loaded_backend",
               "loaded_version"])
        for variant in present:
            val = info(variant, "val_mean_macro_f1")
            meta = vmeta(variant)
            payload = details[variant]
            metric_by_attr = payload.get("attributes", {})
            per_attr_precision = [csv_num(metric_by_attr.get(a, {}).get(
                "macro_precision", "n/a")) for a in attributes]
            per_attr_recall = [csv_num(metric_by_attr.get(a, {}).get(
                "macro_recall", "n/a")) for a in attributes]
            per_attr_f1 = [csv_num(metric_by_attr.get(a, {}).get(
                "macro_f1", "n/a")) for a in attributes]
            ci_lower = [csv_num(ci_bounds(metric_by_attr.get(a, {}))[0])
                        for a in attributes]
            ci_upper = [csv_num(ci_bounds(metric_by_attr.get(a, {}))[1])
                        for a in attributes]
            mean_ci = payload.get("mean_macro_f1_ci") or {}
            writer.writerow(
                [variant, details[variant]["run_id"], meta["family"],
                 meta["architecture"], meta["model_size"],
                 info(variant, "adaptation", meta["adaptation"]),
                 meta["resolution"]]
                + per_attr_precision + per_attr_recall + per_attr_f1
                + ci_lower + ci_upper
                + [csv_num(payload.get("mean_macro_precision", "n/a")),
                   csv_num(payload.get("mean_macro_recall", "n/a")),
                   csv_num(payload.get("mean_macro_f1", "n/a")),
                   csv_num(mean_ci.get("lower", "n/a")),
                   csv_num(mean_ci.get("upper", "n/a")),
                   csv_num(val),
                   info(variant, "best_epoch"), info(variant, "stopped_epoch"),
                   info(variant, "stop_reason"), pinfo(variant, "total"),
                   pinfo(variant, "trainable"), pinfo(variant, "trainable_pct"),
                   info(variant, "train_duration_s"),
                   eval_info(variant, "eval_end_to_end_duration_s",
                             "test_eval_duration_s"),
                   eval_info(variant, "eval_end_to_end_images_per_s",
                             "test_images_per_s"),
                   eval_info(variant, "eval_model_forward_duration_s"),
                   eval_info(variant, "eval_model_forward_ms_per_image"),
                   bmeta(variant, "loaded_backend"),
                   bmeta(variant, "loaded_version")])

    md_lines = [
        "# Attribute classification: consolidated comparison",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        "Primary metric: **macro-F1** on the test split (robust to class "
        "imbalance; accuracy shown for reference).",
        "",
        "## Macro metrics by attribute",
        "",
        "| Variant | "
        + " | ".join(f"{a} P" for a in attributes)
        + " | " + " | ".join(f"{a} R" for a in attributes)
        + " | " + " | ".join(f"{a} F1" for a in attributes)
        + " | " + " | ".join(f"{a} F1 95% CI" for a in attributes)
        + " | Mean P | Mean R | Mean F1 | Mean F1 95% CI |",
        "|" + "---|" * (len(attributes) * 4 + 5),
    ]
    for variant in present:
        payload = details[variant]
        attr_metrics = payload.get("attributes", {})
        def fmt_metric(value):
            return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"
        def fmt_ci(metric):
            return (f"[{metric.get('lower'):.4f}, {metric.get('upper'):.4f}]"
                    if isinstance(metric.get("lower"), (int, float))
                    and isinstance(metric.get("upper"), (int, float))
                    else "n/a")
        cells = [fmt_metric(attr_metrics.get(a, {}).get("macro_precision"))
                 for a in attributes]
        cells += [fmt_metric(attr_metrics.get(a, {}).get("macro_recall"))
                  for a in attributes]
        cells += [fmt_metric(attr_metrics.get(a, {}).get("macro_f1"))
                  for a in attributes]
        cells += [fmt_ci(attr_metrics.get(a, {}).get("macro_f1_ci") or {})
                  for a in attributes]
        mean_ci = payload.get("mean_macro_f1_ci") or {}
        cells += [fmt_metric(payload.get("mean_macro_precision")),
                  fmt_metric(payload.get("mean_macro_recall")),
                  fmt_metric(payload.get("mean_macro_f1")),
                  fmt_ci(mean_ci)]
        md_lines.append(f"| {variant} | " + " | ".join(cells)
                        + " |")
    md_lines += ["", "## Accuracy by attribute", "",
                 "| Variant | " + " | ".join(attributes) + " |",
                 "|" + "---|" * (len(attributes) + 1)]
    for variant in present:
        cells = [
            (f"{details[variant]['attributes'][a]['accuracy']:.4f}"
             if isinstance(details[variant]["attributes"].get(a, {}).get(
                 "accuracy"), (int, float)) else "n/a")
            for a in attributes]
        md_lines.append(f"| {variant} | " + " | ".join(cells) + " |")

    md_lines += [
        "", "## Run details", "",
        "| Variant | Family | Architecture | Size | Adaptation | Res | "
        "Best epoch | Stopped at | Stop reason | Val mean macro-F1 | "
        "Total params | Trainable | Trainable % | Train time (s) | "
        "End-to-end time (s) | End-to-end im/s | Forward time (s) | "
        "Forward ms/image | Backend |",
        "|" + "---|" * 18,
    ]
    def fmt(value, spec):
        value = _report_value(value)
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
                 str(eval_info(variant, "eval_end_to_end_duration_s",
                               "test_eval_duration_s")),
                 str(eval_info(variant, "eval_end_to_end_images_per_s",
                               "test_images_per_s")),
                 str(eval_info(variant, "eval_model_forward_duration_s")),
                 str(eval_info(variant, "eval_model_forward_ms_per_image")),
                 backend_cell]
        md_lines.append("| " + " | ".join(cells) + " |")

    for attr in attributes:
        classes = cfg["attributes"][attr]["classes"]
        md_lines += ["", f"## Per-class precision, recall and F1: {attr}", "",
                     "| Class | Support | "
                     + " | ".join(f"{v} precision | {v} recall | {v} F1"
                                    for v in present) + " |",
                     "|" + "---|" * (len(present) * 3 + 2)]
        for cls in classes:
            support = next(
                (details[v]["attributes"][attr]["support"].get(cls, 0)
                 for v in present if attr in details[v]["attributes"]), 0)
            support = _report_value(support)
            cells = []
            for variant in present:
                m = details[variant]["attributes"].get(attr, {})
                precision = m.get("per_class_precision", {}).get(cls)
                recall = m.get("per_class_recall", {}).get(cls)
                f1 = m.get("per_class_f1", {}).get(cls)
                cells += [f"{value:.4f}" if value is not None else "n/a"
                          for value in (precision, recall, f1)]
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
    run_info = payload.get("run_info") or {}
    meta = run_info.get("variant_meta")
    if not (isinstance(meta, dict) and meta.get("family")):
        from .variants import variant_metadata
        meta = variant_metadata(cfg["models"][variant], variant)
    parameters = run_info.get("parameters") or {}
    backbone_meta = run_info.get("backbone_meta") or {}
    evaluation = payload.get("evaluation", {})
    if not isinstance(backbone_meta, dict):
        backbone_meta = {}
    attributes_payload = payload.get("attributes") or {}
    f1_values = [m.get("macro_f1") for m in attributes_payload.values()
                 if m.get("macro_f1") is not None]
    mean_f1 = payload.get("mean_macro_f1")
    if mean_f1 is None and f1_values:
        mean_f1 = float(np.mean(f1_values))

    def stored(default="n/a", *keys):
        for key in keys:
            value = run_info.get(key)
            if value is None:
                value = evaluation.get(key.removeprefix("eval_"))
            if value is None:
                continue
            return value
        return default

    row = {
        "variant": variant,
        "run_id": payload.get("run_id", "n/a"),
        "family": meta["family"],
        "model_size": meta["model_size"],
        "architecture": meta["architecture"],
        "adaptation": run_info.get("adaptation", meta["adaptation"]),
        "resolution": meta["resolution"],
        "total_params": _report_value(parameters.get("total", "n/a")),
        "trainable_params": _report_value(parameters.get("trainable", "n/a")),
        "best_epoch": run_info.get("best_epoch", "n/a"),
        "val_mean_macro_f1": run_info.get("val_mean_macro_f1", "n/a"),
        "test_mean_macro_precision": _report_value(
            payload.get("mean_macro_precision")),
        "test_mean_macro_recall": _report_value(
            payload.get("mean_macro_recall")),
        "test_mean_macro_f1": _report_value(mean_f1),
        "test_mean_macro_f1_ci_lower": _report_value(
            (payload.get("mean_macro_f1_ci") or {}).get("lower")),
        "test_mean_macro_f1_ci_upper": _report_value(
            (payload.get("mean_macro_f1_ci") or {}).get("upper")),
        "train_duration_s": _report_value(
            run_info.get("train_duration_s", "n/a")),
        "test_end_to_end_duration_s": stored(
            "n/a", "eval_end_to_end_duration_s", "test_eval_duration_s"),
        "test_end_to_end_images_per_s": stored(
            "n/a", "eval_end_to_end_images_per_s", "test_images_per_s"),
        "test_model_forward_duration_s": stored(
            "n/a", "eval_model_forward_duration_s"),
        "test_model_forward_ms_per_image": stored(
            "n/a", "eval_model_forward_ms_per_image"),
        "loaded_backend": backbone_meta.get("loaded_backend", "n/a"),
        "loaded_version": backbone_meta.get("loaded_version", "n/a"),
    }
    for attr in attributes:
        m = payload.get("attributes", {}).get(attr, {})
        row[f"{attr}_macro_precision"] = _report_value(m.get("macro_precision"))
        row[f"{attr}_macro_recall"] = _report_value(m.get("macro_recall"))
        row[f"{attr}_macro_f1"] = _report_value(m.get("macro_f1"))
        row[f"{attr}_macro_f1_ci_lower"] = _report_value(
            (m.get("macro_f1_ci") or {}).get("lower"))
        row[f"{attr}_macro_f1_ci_upper"] = _report_value(
            (m.get("macro_f1_ci") or {}).get("upper"))
        row[f"{attr}_accuracy"] = _report_value(m.get("accuracy"))
    return row


def _ablation_md_table(rows, attributes):
    """Markdown table for a group of size-ablation rows."""
    def num(value, spec=".4f"):
        value = _report_value(value)
        return format(value, spec) if isinstance(value, (int, float)) else \
            str(value)

    header = (["Variant", "Family", "Architecture", "Size", "Adaptation",
               "Res", "Params", "Trainable", "Best epoch",
               "Val mean macro-F1", "Test mean macro-Precision",
               "Test mean macro-Recall", "Test mean macro-F1",
               "Test mean macro-F1 95% CI"]
              + [f"{a} macro-P" for a in attributes]
              + [f"{a} macro-R" for a in attributes]
              + [f"{a} macro-F1" for a in attributes]
              + [f"{a} F1 95% CI" for a in attributes]
              + [f"{a} acc" for a in attributes]
              + ["Train time (s)", "End-to-end time (s)",
                 "End-to-end im/s", "Forward time (s)",
                 "Forward ms/image", "Backend"])
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for row in rows:
        backend = _report_value(row["loaded_backend"])
        version = _report_value(row["loaded_version"])
        if version != "n/a":
            backend = f"{backend} ({version})"
        def ci_text(prefix):
            lower = row.get(f"{prefix}_lower")
            upper = row.get(f"{prefix}_upper")
            if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                return f"[{lower:.4f}, {upper:.4f}]"
            return "n/a"

        cells = ([row["variant"], row["family"], row["architecture"],
                  row["model_size"], row["adaptation"], str(row["resolution"]),
                  num(row["total_params"], ","),
                  num(row["trainable_params"], ","),
                  str(row["best_epoch"]),
                  num(row["val_mean_macro_f1"]),
                  num(row["test_mean_macro_precision"]),
                  num(row["test_mean_macro_recall"]),
                  num(row["test_mean_macro_f1"]),
                  ci_text("test_mean_macro_f1_ci")]
                 + [num(row.get(f"{a}_macro_precision"))
                    if row.get(f"{a}_macro_precision") is not None else "n/a"
                    for a in attributes]
                 + [num(row.get(f"{a}_macro_recall"))
                    if row.get(f"{a}_macro_recall") is not None else "n/a"
                    for a in attributes]
                 + [num(row.get(f"{a}_macro_f1")) if row.get(f"{a}_macro_f1")
                    is not None else "n/a" for a in attributes]
                 + [ci_text(f"{a}_macro_f1_ci") for a in attributes]
                 + [num(row.get(f"{a}_accuracy")) if row.get(f"{a}_accuracy")
                    is not None else "n/a" for a in attributes]
                 + [str(_report_value(row["train_duration_s"])),
                    str(_report_value(row["test_end_to_end_duration_s"])),
                    str(_report_value(row["test_end_to_end_images_per_s"])),
                    str(_report_value(row["test_model_forward_duration_s"])),
                    str(_report_value(row["test_model_forward_ms_per_image"])),
                    backend])
        cells = [_report_value(c) for c in cells]
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

    md_lines += ["## 5. Inference efficiency by adaptation", "",
                 "Inference time covers the held-out test DataLoader and "
                 "model forward pass; metric aggregation/bootstrap is "
                 "excluded. Throughput is images per second.", "",
                 "| Variant | Family | Adaptation | Params | End-to-end time (s) | "
                 "End-to-end im/s | Forward time (s) | Forward ms/image |",
                 "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in sorted(rows, key=lambda item: (item["adaptation"],
                                                item["variant"])):
        md_lines.append(
            f"| {row['variant']} | {row['family']} | {row['adaptation']} | "
            f"{_report_value(row['total_params'])} | "
            f"{_report_value(row['test_end_to_end_duration_s'])} | "
            f"{_report_value(row['test_end_to_end_images_per_s'])} | "
            f"{_report_value(row['test_model_forward_duration_s'])} | "
            f"{_report_value(row['test_model_forward_ms_per_image'])} |")
    md_lines.append("")

    if missing:
        md_lines += [f"Size-ablation variants without saved test metrics: "
                     f"{', '.join(missing)} (skipped or not yet trained).", ""]

    md_path = reports_dir / f"size_ablation{suffix}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    log.info("Size-ablation report written (%d/%d variants): %s, %s",
             len(rows), len(members), csv_path, md_path)
    return md_path


def _bootstrap_samples_for_split(cfg, split):
    """Return bootstrap count, keeping validation bootstrap opt-in."""
    evaluation = cfg.get("evaluation", {})
    if split == "test":
        return int(evaluation.get(
            "test_bootstrap_samples",
            evaluation.get("bootstrap_samples", 2000)))
    return int(evaluation.get(f"{split}_bootstrap_samples", 0))


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
    evaluation_cfg = cfg.get("evaluation", {})
    metrics, evaluation_meta = evaluate_model(
        model, loader, attributes, device,
        bootstrap_samples=_bootstrap_samples_for_split(cfg, split),
        bootstrap_seed=int(evaluation_cfg.get("bootstrap_seed", cfg["seed"])),
        timing_warmup_batches=int(
            evaluation_cfg.get("timing_warmup_batches", 5)),
        timing_repeats=int(evaluation_cfg.get("timing_repeats", 3)),
        return_timing=True)
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
        "evaluation_split": split,
        "eval_end_to_end_duration_s": evaluation_meta.get(
            "end_to_end_duration_s"),
        "eval_end_to_end_images_per_s": evaluation_meta.get(
            "end_to_end_images_per_s"),
        "eval_model_forward_duration_s": evaluation_meta.get(
            "model_forward_duration_s"),
        "eval_model_forward_ms_per_image": evaluation_meta.get(
            "model_forward_ms_per_image"),
        "eval_timing_repeats": evaluation_meta.get("timing_repeats"),
        "eval_timing_warmup_batches": evaluation_meta.get(
            "timing_warmup_batches"),
    }
    if split == "test":
        # These aliases are intentionally absent from validation re-evaluation.
        duration = evaluation_meta.get("end_to_end_duration_s")
        n_images = evaluation_meta.get("n_images")
        run_info.update({
            "test_eval_duration_s": duration,
            "test_images_per_s": evaluation_meta.get(
                "end_to_end_images_per_s"),
            "test_ms_per_image": (duration * 1000.0 / n_images
                                   if duration and n_images else None),
            "test_mean_macro_f1_ci": evaluation_meta.get(
                "mean_macro_f1_ci"),
        })
    save_metrics_bundle(cfg["paths"]["metrics_dir"], variant, split, metrics,
                        ckpt.get("run_id", "reeval"), run_info=run_info,
                        evaluation_meta=evaluation_meta)
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
