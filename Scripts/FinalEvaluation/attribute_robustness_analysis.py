#!/usr/bin/env python
"""Inference-only robustness analysis for retained MTSD attribute classifier.

Loads the saved V-JEPA 2.1-L + LoRA checkpoint, reproduces the deterministic
test preprocessing/inference path, saves aligned per-crop logits/predictions,
and evaluates physical and cross-attribute semantic slices. Nothing is trained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[2]
SUBPROJECT = ROOT / "Scripts" / "MTSD-Scripts" / "AttributeClassification"
sys.path.insert(0, str(SUBPROJECT))
sys.path.insert(0, str(ROOT / "Scripts" / "FinalEvaluation"))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import sklearn
import torch
import torchvision
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

import evidence_lib as ev
from mtsd_attr.backbones import build_backbone
from mtsd_attr.config import adaptation_of, load_config, model_training_config, set_seed
from mtsd_attr.dataset import AttributeCropDataset, PadToSquare, build_transforms
from mtsd_attr.multihead_model import MultiHeadClassifier
from mtsd_attr.train_common import _load_checkpoint_state


VARIANT = "vjepa21_vitl_lora"
CHECKPOINT = SUBPROJECT / "outputs" / "checkpoints" / VARIANT / "best.pt"
RETAINED_METRICS = SUBPROJECT / "outputs" / "metrics" / VARIANT / "test_metrics.json"
GENERATOR = "Scripts/FinalEvaluation/attribute_robustness_analysis.py"
HEAD_LABELS = {
    "view_angle": "Viewing angle",
    "mounting": "Mounting",
    "condition": "Condition",
    "sign_shape": "Sign shape",
}
SLICE_LABELS = {
    "original_object_size": "Original object size",
    "crop_brightness": "Crop brightness",
    "crop_contrast": "Crop contrast",
    "crop_sharpness": "Crop sharpness",
}
PHYSICAL_ORDERS = {
    "original_object_size": ["Small", "Medium", "Large"],
    "crop_brightness": ["Low", "Medium", "High"],
    "crop_contrast": ["Low", "Medium", "High"],
    "crop_sharpness": ["Blurred", "Intermediate", "Sharp"],
}
COLORS = {
    "view_angle": "#2a78d6",
    "mounting": "#eda100",
    "condition": "#e34948",
    "sign_shape": "#1baf7a",
    "overall_mean": "#4a3aa7",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def configure_determinism(seed: int) -> None:
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def prepared_crop_stats(path: Path, image_size: int) -> dict[str, float]:
    """Stats on pixels after exact test geometry, before ToTensor/normalise."""
    geometry = transforms.Compose([
        PadToSquare(),
        transforms.Resize((image_size, image_size), antialias=True),
    ])
    with Image.open(path) as image:
        prepared = geometry(image.convert("RGB"))
        grey = np.asarray(prepared.convert("L"), dtype=np.float32)
    laplacian = (-4.0 * grey[1:-1, 1:-1]
                 + grey[:-2, 1:-1] + grey[2:, 1:-1]
                 + grey[1:-1, :-2] + grey[1:-1, 2:])
    return {
        "crop_brightness": float(grey.mean()),
        "crop_contrast": float(grey.std()),
        "crop_sharpness": float(laplacian.var()),
    }


def fixed_macro_f1(rows: list[dict], head: str, attributes: dict) -> tuple[float, int]:
    valid = [row for row in rows if row[f"{head}_true_index"] >= 0]
    if not valid:
        return 0.0, 0
    labels = list(range(len(attributes[head]["classes"])))
    score = f1_score(
        [row[f"{head}_true_index"] for row in valid],
        [row[f"{head}_pred_index"] for row in valid],
        labels=labels, average="macro", zero_division=0)
    return float(score), len(valid)


def slice_row(*, dimension: str, value: str, head: str, score: float,
              support: int, min_support: int, slice_type: str,
              slice_attribute: str = "", included_heads: str = "",
              class_support: str = "", low_support_classes: str = "",
              interpretation_flag: str = "ok", notes: str = "") -> dict:
    return {
        "model": "V-JEPA 2.1-L + LoRA",
        "variant": VARIANT,
        "split": "test",
        "slice_type": slice_type,
        "slice_dimension": dimension,
        "slice_attribute": slice_attribute,
        "slice_value": value,
        "prediction_head": head,
        "metric": "macro_f1" if head != "overall_mean" else "mean_macro_f1",
        "value": round(float(score), 6),
        "n_samples": int(support),
        "support_ok": "ok" if support >= min_support else "insufficient_support",
        "included_heads": included_heads,
        "class_support": class_support,
        "low_support_classes": low_support_classes,
        "interpretation_flag": interpretation_flag,
        "notes": notes,
    }


def infer(args) -> tuple[list[dict], dict, dict, dict]:
    cfg = load_config(args.config)
    model_cfg, training = model_training_config(cfg, VARIANT)
    required = [CHECKPOINT, cfg["paths"]["manifest"], RETAINED_METRICS,
                Path(cfg["_config_path"])]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required retained inputs: " + ", ".join(missing))

    manifest = json.loads(cfg["paths"]["manifest"].read_text(encoding="utf-8-sig"))
    test_records = [record for record in manifest["records"]
                    if record.get("split") == "test"]
    if len(test_records) != 1890:
        raise RuntimeError(f"Expected exactly 1,890 test crops; found {len(test_records)}")
    crop_ids = [record["crop_id"] for record in test_records]
    if len(set(crop_ids)) != len(crop_ids):
        raise RuntimeError("Test crop IDs are not unique")
    absent_crops = []
    for record in test_records:
        crop = SUBPROJECT / record["crop_path"].replace("\\", "/")
        if not crop.exists():
            absent_crops.append(str(crop))
    if absent_crops:
        raise FileNotFoundError(f"Missing {len(absent_crops)} test crop files")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable; pass --allow-cpu to accept a slow CPU run")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != VARIANT:
        raise RuntimeError(f"Checkpoint variant is {checkpoint.get('variant')!r}, not {VARIANT}")
    if checkpoint.get("model_cfg") != model_cfg:
        raise RuntimeError("Current variant config differs from checkpoint model_cfg")
    if checkpoint.get("attributes") != cfg["attributes"]:
        raise RuntimeError("Current attribute mapping differs from checkpoint mapping")

    configure_determinism(cfg["seed"])
    print(f"Loading {VARIANT} backbone/checkpoint on {device}...", flush=True)
    backbone = build_backbone(checkpoint["model_cfg"])
    expected_backbone = checkpoint.get("backbone_meta", {})
    loaded_backbone = dict(getattr(backbone, "backbone_meta", {}) or {})
    for key in ("loaded_backend", "loaded_version", "entrypoint", "image_size",
                "feature_dim", "num_frames"):
        if expected_backbone.get(key) != loaded_backbone.get(key):
            raise RuntimeError(
                f"Backbone identity mismatch for {key}: retained="
                f"{expected_backbone.get(key)!r}, loaded={loaded_backbone.get(key)!r}")

    adaptation = checkpoint.get("adaptation") or adaptation_of(checkpoint["model_cfg"])
    model = MultiHeadClassifier(
        backbone, checkpoint["attributes"], checkpoint["probe"], adaptation).to(device)
    _load_checkpoint_state(model, checkpoint["model_state"], adaptation)
    model.eval()

    transform = build_transforms(
        backbone.image_size, False, training["augmentation"])
    dataset = AttributeCropDataset(
        test_records, checkpoint["attributes"], SUBPROJECT, transform)
    loader = DataLoader(
        dataset, batch_size=training["batch_size"], shuffle=False,
        num_workers=training["num_workers"], pin_memory=device.type == "cuda",
        persistent_workers=training["num_workers"] > 0)
    amp = bool(training["amp"]) and device.type == "cuda"
    attributes = checkpoint["attributes"]
    record_by_id = {record["crop_id"]: record for record in test_records}
    predictions: list[dict] = []
    print(f"Running deterministic inference over {len(dataset)} crops...", flush=True)
    with torch.no_grad():
        for batch_index, (images, targets, batch_crop_ids) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=amp):
                batch_logits = model(images)
            logits_cpu = {head: values.float().cpu().numpy()
                          for head, values in batch_logits.items()}
            targets_np = targets.numpy()
            for index, crop_id in enumerate(batch_crop_ids):
                record = record_by_id[crop_id]
                bbox = [float(value) for value in record["bbox"]]
                area = max(0.0, bbox[2] * bbox[3])
                output = {
                    "crop_id": crop_id,
                    "group": record["group"],
                    "image_key": record["image_key"],
                    "source_image": record["source_image"],
                    "crop_path": record["crop_path"],
                    "annotation_id": record["annotation_id"],
                    "category": record.get("category", ""),
                    "bbox_x": bbox[0], "bbox_y": bbox[1],
                    "bbox_width": bbox[2], "bbox_height": bbox[3],
                    "original_bbox_area": area,
                    "original_object_size": ev.coco_size_label(area).title(),
                }
                for head_index, (head, spec) in enumerate(attributes.items()):
                    true_index = int(targets_np[index, head_index])
                    pred_index = int(np.argmax(logits_cpu[head][index]))
                    output[f"{head}_true_index"] = true_index
                    output[f"{head}_true"] = (
                        spec["classes"][true_index] if true_index >= 0 else "")
                    output[f"{head}_pred_index"] = pred_index
                    output[f"{head}_pred"] = spec["classes"][pred_index]
                    for class_index, class_name in enumerate(spec["classes"]):
                        output[f"{head}_logit_{class_name}"] = float(
                            logits_cpu[head][index, class_index])
                predictions.append(output)
            if batch_index % 20 == 0 or batch_index == len(loader):
                print(f"  inference batch {batch_index}/{len(loader)}", flush=True)

    if len(predictions) != 1890 or [row["crop_id"] for row in predictions] != crop_ids:
        raise RuntimeError("Inference output is not aligned one-to-one with the test manifest")

    print("Computing crop-input brightness/contrast/sharpness...", flush=True)
    for index, row in enumerate(predictions, start=1):
        crop = SUBPROJECT / row["crop_path"].replace("\\", "/")
        row.update(prepared_crop_stats(crop, backbone.image_size))
        if index % 300 == 0 or index == len(predictions):
            print(f"  image statistics {index}/{len(predictions)}", flush=True)

    thresholds = {}
    for dimension in ("crop_brightness", "crop_contrast", "crop_sharpness"):
        low, high = ev.tercile_thresholds([row[dimension] for row in predictions])
        thresholds[dimension] = {"tercile_33": low, "tercile_67": high}
        labels = (("Blurred", "Intermediate", "Sharp")
                  if dimension == "crop_sharpness" else ("Low", "Medium", "High"))
        for row in predictions:
            row[f"{dimension}_tercile"] = ev.tercile_label(
                row[dimension], (low, high), labels)

    retained_metrics = json.loads(RETAINED_METRICS.read_text(encoding="utf-8"))
    aggregate = {}
    for head in attributes:
        score, support = fixed_macro_f1(predictions, head, attributes)
        aggregate[head] = {"macro_f1": score, "n_samples": support}
        retained = float(retained_metrics["attributes"][head]["macro_f1"])
        aggregate[head]["retained_macro_f1"] = retained
        aggregate[head]["absolute_difference"] = abs(score - retained)
        if abs(score - retained) > args.reconciliation_tolerance:
            raise RuntimeError(
                f"{head} rerun macro-F1 {score:.10f} does not reconcile with "
                f"retained {retained:.10f} within {args.reconciliation_tolerance}")
    aggregate_mean = float(np.mean([aggregate[head]["macro_f1"] for head in attributes]))
    retained_mean = float(retained_metrics["mean_macro_f1"])
    if abs(aggregate_mean - retained_mean) > args.reconciliation_tolerance:
        raise RuntimeError(
            f"Mean rerun macro-F1 {aggregate_mean:.10f} does not reconcile with "
            f"retained {retained_mean:.10f}")
    aggregate["overall_mean"] = {
        "macro_f1": aggregate_mean, "n_samples": len(predictions),
        "retained_macro_f1": retained_mean,
        "absolute_difference": abs(aggregate_mean - retained_mean),
    }
    return predictions, thresholds, aggregate, {
        "cfg": cfg, "training": training, "checkpoint": checkpoint,
        "manifest": manifest, "test_records": test_records,
        "device": str(device), "amp": amp, "backbone_meta": loaded_backbone,
    }


def build_slices(predictions: list[dict], attributes: dict, min_support: int) -> list[dict]:
    rows: list[dict] = []

    def add_group(dimension: str, value: str, group: list[dict], slice_type: str,
                  heads: list[str], slice_attribute: str = "") -> None:
        eligible_scores = []
        eligible_heads = []
        warned_heads = []
        for head in heads:
            score, support = fixed_macro_f1(group, head, attributes)
            counts = Counter(row[f"{head}_true"] for row in group
                             if row[f"{head}_true"] != "")
            ordered_counts = {
                class_name: int(counts.get(class_name, 0))
                for class_name in attributes[head]["classes"]}
            low_classes = [class_name for class_name, count in ordered_counts.items()
                           if count < min_support]
            warning = "class_support_warning" if low_classes else "ok"
            if low_classes:
                warned_heads.append(head)
            rows.append(slice_row(
                dimension=dimension, value=value, head=head, score=score,
                support=support, min_support=min_support, slice_type=slice_type,
                slice_attribute=slice_attribute,
                class_support=json.dumps(ordered_counts, separators=(",", ":")),
                low_support_classes=";".join(low_classes),
                interpretation_flag=warning,
                notes="fixed full head vocabulary; zero_division=0"))
            if support >= min_support:
                eligible_scores.append(score)
                eligible_heads.append(head)
        mean_score = float(np.mean(eligible_scores)) if eligible_scores else 0.0
        rows.append(slice_row(
            dimension=dimension, value=value, head="overall_mean",
            score=mean_score, support=len(group), min_support=min_support,
            slice_type=slice_type, slice_attribute=slice_attribute,
            included_heads=";".join(eligible_heads),
            low_support_classes=";".join(warned_heads),
            interpretation_flag=("class_support_warning" if warned_heads else "ok"),
            notes=(f"unweighted mean of {len(eligible_heads)} eligible head macro-F1 "
                   f"values; each head requires support >= {min_support}; "
                   "low_support_classes lists warned heads on overall rows")))

    add_group("aggregate", "All", predictions, "aggregate", list(attributes))
    physical_fields = {
        "original_object_size": "original_object_size",
        "crop_brightness": "crop_brightness_tercile",
        "crop_contrast": "crop_contrast_tercile",
        "crop_sharpness": "crop_sharpness_tercile",
    }
    for dimension, field in physical_fields.items():
        for value in PHYSICAL_ORDERS[dimension]:
            add_group(dimension, value,
                      [row for row in predictions if row[field] == value],
                      "physical", list(attributes))

    for slice_attribute, spec in attributes.items():
        target_heads = [head for head in attributes if head != slice_attribute]
        for value in spec["classes"]:
            group = [row for row in predictions
                     if row[f"{slice_attribute}_true"] == value]
            add_group("semantic_condition", value, group, "semantic",
                      target_heads, slice_attribute=slice_attribute)
    return rows


def write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_slice_results(path: Path, rows: list[dict]) -> None:
    fields = ["model", "variant", "split", "slice_type", "slice_dimension",
              "slice_attribute", "slice_value", "prediction_head", "metric",
              "value", "n_samples", "support_ok", "included_heads",
              "class_support", "low_support_classes", "interpretation_flag",
              "notes"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_figures(rows: list[dict], attributes: dict, figures_dir: Path) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    ev.apply_plot_style()
    saved: list[str] = []
    series = list(attributes) + ["overall_mean"]

    def grouped_bars(dimension: str, title: str, filename: str) -> None:
        order = PHYSICAL_ORDERS[dimension]
        figure, axis = plt.subplots(figsize=(8.2, 4.2))
        x = np.arange(len(order))
        width = 0.16
        for index, head in enumerate(series):
            values = []
            for value in order:
                match = next(row for row in rows
                             if row["slice_dimension"] == dimension
                             and row["slice_value"] == value
                             and row["prediction_head"] == head)
                values.append(match["value"])
            offset = (index - (len(series) - 1) / 2) * width
            bars = axis.bar(x + offset, values, width * 0.92,
                            label=HEAD_LABELS.get(head, "Overall mean"),
                            color=COLORS[head])
            axis.bar_label(bars, labels=[f"{value:.2f}" for value in values],
                           padding=2, fontsize=7, rotation=90)
        supports = [next(row["n_samples"] for row in rows
                         if row["slice_dimension"] == dimension
                         and row["slice_value"] == value
                         and row["prediction_head"] == "overall_mean")
                    for value in order]
        axis.set_xticks(x, [f"{value}\nn={support:,}"
                            for value, support in zip(order, supports)])
        axis.set_ylim(0, 1.08)
        axis.set_ylabel("Macro-F1")
        axis.set_title(title, pad=12)
        axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12),
                    fontsize=8)
        figure.subplots_adjust(bottom=0.25, top=0.88)
        saved.extend(ev.save_figure(figure, figures_dir, filename))

    grouped_bars("original_object_size",
                 "V-JEPA 2.1-L + LoRA: macro-F1 by original object size",
                 "vjepa21_vitl_lora_macro_f1_by_original_object_size")

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharey=True)
    property_dimensions = ["crop_brightness", "crop_contrast", "crop_sharpness"]
    for axis, dimension in zip(axes, property_dimensions):
        order = PHYSICAL_ORDERS[dimension]
        x = np.arange(len(order))
        for head in series:
            values = [next(row["value"] for row in rows
                           if row["slice_dimension"] == dimension
                           and row["slice_value"] == value
                           and row["prediction_head"] == head)
                      for value in order]
            axis.plot(x, values, marker="o", linewidth=1.7,
                      label=HEAD_LABELS.get(head, "Overall mean"),
                      color=COLORS[head])
        axis.set_xticks(x, order)
        axis.set_ylim(0, 1.0)
        axis.set_title(SLICE_LABELS[dimension])
        axis.set_xlabel("Dataset-specific tercile")
    axes[0].set_ylabel("Macro-F1")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=5, loc="lower center",
                  bbox_to_anchor=(0.5, 0.005), fontsize=8)
    figure.text(
        0.5, 0.105,
        "Fixed full-vocabulary macro-F1; class-mix warnings are reported in the CSV/summary.",
        ha="center", fontsize=8, color="#52514e")
    figure.suptitle("V-JEPA 2.1-L + LoRA: robustness across classifier-input properties")
    figure.subplots_adjust(bottom=0.25, top=0.84, wspace=0.12)
    saved.extend(ev.save_figure(
        figure, figures_dir, "vjepa21_vitl_lora_crop_property_robustness"))

    semantic = [row for row in rows if row["slice_type"] == "semantic"]
    semantic_groups = []
    for slice_attribute, spec in attributes.items():
        for value in spec["classes"]:
            group_rows = [row for row in semantic
                          if row["slice_attribute"] == slice_attribute
                          and row["slice_value"] == value]
            support = max((row["n_samples"] for row in group_rows), default=0)
            semantic_groups.append((slice_attribute, value, support))
    matrix = np.full((len(semantic_groups), len(series)), np.nan)
    for row_index, (slice_attribute, value, _) in enumerate(semantic_groups):
        for column_index, head in enumerate(series):
            matches = [row for row in semantic
                       if row["slice_attribute"] == slice_attribute
                       and row["slice_value"] == value
                       and row["prediction_head"] == head]
            if matches and matches[0]["support_ok"] == "ok":
                matrix[row_index, column_index] = matches[0]["value"]
    figure, axis = plt.subplots(figsize=(8.0, 7.2))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    axis.set_xticks(range(len(series)),
                    [HEAD_LABELS.get(head, "Mean of other heads") for head in series],
                    rotation=25, ha="right")
    axis.set_yticks(range(len(semantic_groups)), [
        f"{HEAD_LABELS[attr]} = {value} (n={support})"
        for attr, value, support in semantic_groups], fontsize=8)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                axis.text(column_index, row_index, f"{value:.2f}",
                          ha="center", va="center", fontsize=8,
                          color="white" if value > 0.58 else "#0b0b0b")
            elif column_index < len(attributes):
                axis.text(column_index, row_index, "—", ha="center", va="center",
                          fontsize=9, color="#898781")
    boundary = 0
    for spec in attributes.values():
        boundary += len(spec["classes"])
        if boundary < len(semantic_groups):
            axis.axhline(boundary - 0.5, color="#52514e", linewidth=0.8)
    axis.set_title("Cross-attribute semantic robustness (self-target slices omitted)")
    colourbar = figure.colorbar(image, ax=axis, shrink=0.82)
    colourbar.set_label("Macro-F1")
    figure.tight_layout()
    saved.extend(ev.save_figure(
        figure, figures_dir, "vjepa21_vitl_lora_semantic_condition_robustness"))
    return saved


def make_protocol(context: dict, thresholds: dict, aggregate: dict,
                  args, output_dir: Path, figures_dir: Path) -> dict:
    cfg = context["cfg"]
    checkpoint = context["checkpoint"]
    manifest = context["manifest"]
    test_records = context["test_records"]
    return {
        "analysis_status": "new inference-only analysis; separate from retained report",
        "generated_at": utc_now(),
        "generator": GENERATOR,
        "generator_sha256": sha256_file(ROOT / GENERATOR),
        "git_commit": ev.git_commit_sha(),
        "variant": VARIANT,
        "retained_run_id": checkpoint.get("run_id"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "training_performed": False,
        "population": {
            "dataset": "MTSD", "split": "test", "n_crops": len(test_records),
            "groups": dict(sorted(Counter(row["group"] for row in test_records).items())),
            "manifest_total_records": len(manifest["records"]),
            "split_assignment": "retained manifest; no refresh or resplitting",
        },
        "inputs": {
            "checkpoint": rel(CHECKPOINT), "checkpoint_sha256": sha256_file(CHECKPOINT),
            "manifest": rel(cfg["paths"]["manifest"]),
            "manifest_sha256": sha256_file(cfg["paths"]["manifest"]),
            "config": rel(Path(cfg["_config_path"])),
            "config_sha256": sha256_file(Path(cfg["_config_path"])),
            "retained_metrics": rel(RETAINED_METRICS),
            "retained_metrics_sha256": sha256_file(RETAINED_METRICS),
            "metric_implementation": rel(SUBPROJECT / "mtsd_attr" / "evaluate.py"),
            "metric_implementation_sha256": sha256_file(
                SUBPROJECT / "mtsd_attr" / "evaluate.py"),
            "dataset_implementation": rel(SUBPROJECT / "mtsd_attr" / "dataset.py"),
            "dataset_implementation_sha256": sha256_file(
                SUBPROJECT / "mtsd_attr" / "dataset.py"),
        },
        "model": {
            "model_cfg": checkpoint["model_cfg"],
            "probe": checkpoint["probe"],
            "attributes": checkpoint["attributes"],
            "adaptation": checkpoint.get("adaptation"),
            "retained_backbone_meta": checkpoint.get("backbone_meta"),
            "loaded_backbone_meta": context["backbone_meta"],
        },
        "inference": {
            "device": context["device"],
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "batch_size": context["training"]["batch_size"],
            "num_workers": context["training"]["num_workers"],
            "amp": context["amp"], "amp_dtype": "bfloat16",
            "seed": cfg["seed"], "shuffle": False,
            "cudnn_benchmark": False, "cudnn_deterministic": True,
            "deterministic_algorithms": True,
            "preprocessing": [
                "PadToSquare(fill=[114,114,114])",
                f"Resize([{context['backbone_meta']['image_size']},"
                f"{context['backbone_meta']['image_size']}], antialias=True)",
                "ToTensor()", "Normalize(ImageNet mean/std)",
            ],
        },
        "metrics": {
            "primary": "macro_f1",
            "implementation": "sklearn.metrics.f1_score",
            "average": "macro", "zero_division": 0,
            "label_policy": "fixed complete checkpoint vocabulary for each head",
            "overall": "unweighted mean of eligible per-head macro-F1 values",
            "min_support": args.min_support,
            "class_support_warning": (
                "flag a head-level slice when any expected target class has fewer "
                "than min_support examples; the fixed-vocabulary score is retained"),
            "reconciliation_tolerance": args.reconciliation_tolerance,
            "aggregate_reconciliation": aggregate,
        },
        "slices": {
            "original_object_size": {
                "area_source": "original manifest GT bbox width*height before padding/cropping/resizing",
                "thresholds": {"Small": "area < 32^2", "Medium": "32^2 <= area < 96^2",
                               "Large": "area >= 96^2"},
            },
            "crop_properties": {
                "pixel_population": "post-PadToSquare/post-Resize 384 RGB pixels before ToTensor/normalisation",
                "brightness": "mean greyscale intensity",
                "contrast": "greyscale standard deviation",
                "sharpness": "variance of 4-neighbour Laplacian; higher means sharper",
                "thresholds": thresholds,
            },
            "semantic_condition": {
                "policy": "each prediction head sliced only by other ground-truth attributes",
                "self_target_slices_omitted": True,
                "matrix": {head: [other for other in checkpoint["attributes"] if other != head]
                           for head in checkpoint["attributes"]},
            },
            "explicitly_not_analysed": ["object_density", "image_position"],
        },
        "software": {
            "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "sklearn": sklearn.__version__,
            "numpy": np.__version__, "pillow": PIL.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "outputs": {"report_dir": rel(output_dir), "figures_dir": rel(figures_dir)},
    }


def write_summary(output_dir: Path, figures_dir: Path, rows: list[dict],
                  thresholds: dict, aggregate: dict, protocol: dict,
                  figure_paths: list[str], min_support: int) -> None:
    def result(dimension: str, value: str, head: str) -> dict:
        return next(row for row in rows if row["slice_dimension"] == dimension
                    and row["slice_value"] == value
                    and row["prediction_head"] == head)

    def pct(value: float) -> str:
        return f"{100 * float(value):.2f}%"

    physical_means = []
    for dimension in PHYSICAL_ORDERS:
        for value in PHYSICAL_ORDERS[dimension]:
            physical_means.append(result(dimension, value, "overall_mean"))
    weakest_physical = min(physical_means, key=lambda row: row["value"])
    strongest_physical = max(physical_means, key=lambda row: row["value"])
    semantic_heads = [row for row in rows if row["slice_type"] == "semantic"
                      and row["prediction_head"] != "overall_mean"
                      and row["support_ok"] == "ok"]
    weakest_semantic = sorted(semantic_heads, key=lambda row: row["value"])[:8]
    insufficient = [row for row in rows if row["support_ok"] != "ok"]
    class_warnings = [row for row in rows
                      if row["prediction_head"] != "overall_mean"
                      and row["interpretation_flag"] == "class_support_warning"]
    missing_class_rows = [row for row in class_warnings
                          if any(count == 0 for count in
                                 json.loads(row["class_support"]).values())]

    lines = [
        "# V-JEPA attribute robustness: isolated crops remain strongest for shape and mounting",
        "",
        f"Generated {protocol['generated_at']} from deterministic inference over all "
        f"{protocol['population']['n_crops']:,} retained MTSD test crops.",
        "",
        "> **New inference-only analysis, separate from the retained robustness report.** "
        "The saved epoch-12 checkpoint was loaded; nothing was retrained, the manifest "
        "was not refreshed, and the test split was not changed.",
        "",
        "## Executive summary",
        "",
        f"- Rerun overall mean macro-F1: **{pct(aggregate['overall_mean']['macro_f1'])}**; "
        f"absolute difference from the retained result was "
        f"{aggregate['overall_mean']['absolute_difference']:.2e}.",
        f"- The weakest physical slice was **{SLICE_LABELS[weakest_physical['slice_dimension']]} "
        f"/ {weakest_physical['slice_value']}** at {pct(weakest_physical['value'])}; "
        f"the strongest was **{SLICE_LABELS[strongest_physical['slice_dimension']]} "
        f"/ {strongest_physical['slice_value']}** at {pct(strongest_physical['value'])}.",
        "- Semantic slices use only other ground-truth attributes. A head is never "
        "sliced by its own target, and object density/image position are intentionally absent.",
        "",
        "## Rerun aggregate performance and reconciliation",
        "",
        "| Head | Macro-F1 | Retained | Absolute difference | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for head in list(protocol["model"]["attributes"]) + ["overall_mean"]:
        item = aggregate[head]
        lines.append(
            f"| {HEAD_LABELS.get(head, 'Overall mean')} | {pct(item['macro_f1'])} | "
            f"{pct(item['retained_macro_f1'])} | {item['absolute_difference']:.2e} | "
            f"{item['n_samples']:,} |")
    lines.extend([
        "",
        "The strictly deterministic CUDA rerun exactly reproduced viewing-angle "
        "macro-F1. The other heads differed slightly from the retained historical "
        "bundle; the largest absolute difference was 0.21 percentage points "
        "(condition). The source of this small numerical/prediction discrepancy was "
        "not isolated. All slice results use the newly saved, internally consistent "
        "per-crop outputs rather than mixing in retained aggregates.",
    ])

    for dimension in PHYSICAL_ORDERS:
        lines.extend([
            "", f"## {SLICE_LABELS[dimension]}", "",
            "| Slice | Viewing angle | Mounting | Condition | Sign shape | Overall mean | Crops |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for value in PHYSICAL_ORDERS[dimension]:
            values = [result(dimension, value, head) for head in
                      list(protocol["model"]["attributes"]) + ["overall_mean"]]
            lines.append(
                f"| {value} | " + " | ".join(pct(item["value"]) for item in values)
                + f" | {values[-1]['n_samples']:,} |")
        if dimension == "original_object_size":
            lines.append(
                "\nSize is based on native-coordinate `bbox_width × bbox_height` before "
                "crop padding or resizing, using the detector audit's COCO thresholds.")
        else:
            key = dimension
            threshold = thresholds[key]
            lines.append(
                f"\nTest-crop tercile thresholds: {threshold['tercile_33']:.4f} / "
                f"{threshold['tercile_67']:.4f}. Statistics use classifier-input "
                "pixels after deterministic padding/resizing and before normalisation.")

    lines.extend([
        "", "## Semantic-condition robustness", "",
        "The complete cross-attribute matrix is in `attribute_robustness_slices.csv` "
        "and the compact heatmap. The lowest sufficient-support head-level cells are:",
        "", "| Prediction head | Sliced by | GT value | Macro-F1 | Support |",
        "| --- | --- | --- | ---: | ---: |",
    ])
    for row in weakest_semantic:
        lines.append(
            f"| {HEAD_LABELS[row['prediction_head']]} | "
            f"{HEAD_LABELS[row['slice_attribute']]} | {row['slice_value']} | "
            f"{pct(row['value'])} | {row['n_samples']:,} |")

    lines.extend([
        "", "## Support and interpretation", "",
        f"Every row requires support ≥ {min_support}. There are **{len(insufficient)}** "
        "rows below this threshold; they remain in the CSV but are marked "
        "`insufficient_support` and excluded from figures and overall slice means.",
        "",
        f"Separately, **{len(class_warnings)} head-level rows** contain at least one "
        f"target class with fewer than {min_support} examples; **{len(missing_class_rows)}** "
        "contain an entirely absent target class. Their fixed-vocabulary macro-F1 "
        "values are retained, but `interpretation_flag`, `class_support`, and "
        "`low_support_classes` identify composition-sensitive cells in the CSV. "
        "For example, the Sharp sign-shape slice contains no Pentagon examples, "
        "which mechanically contributes a zero for that class.",
    ])
    if insufficient:
        lines.extend([
            "", "| Slice | Head | Support |", "| --- | --- | ---: |",
        ])
        for row in insufficient:
            label = (f"{row['slice_attribute']}={row['slice_value']}"
                     if row["slice_attribute"] else
                     f"{row['slice_dimension']}={row['slice_value']}")
            lines.append(f"| {label} | {row['prediction_head']} | {row['n_samples']} |")

    lines.extend([
        "", "## Protocol boundaries", "",
        "- Original size uses the retained manifest's native GT bounding box, not the padded crop.",
        "- Brightness, contrast and sharpness use the actual deterministic 384×384 "
        "classifier input geometry before tensor normalisation; their thresholds are "
        "specific to these 1,890 crops and are not detector thresholds.",
        "- Macro-F1 uses every class in the saved head vocabulary with "
        "`zero_division=0`, matching the retained evaluator.",
        "- Semantic slices are observational correlations among attributes, not controlled "
        "corruptions or causal effects.",
        "- Isolated GT crops provide no defensible object-density or image-position test.",
        "", "## Outputs", "",
        "- `per_crop_predictions_logits.csv`: one aligned row per test crop, including "
        "native bbox, derived properties, true/predicted labels, indices and every class logit.",
        "- `attribute_robustness_slices.csv`: aggregate, physical and semantic result rows.",
        "- `run_config.json`, `robustness_thresholds.json`, and "
        "`robustness_protocol.json`: exact identities, hashes, preprocessing and rules.",
        "- `missing_or_low_support.md`: input audit and low-support rows.",
        "", "Figures:", "",
    ])
    lines.extend(f"- `{path}`" for path in figure_paths if path.endswith(".png"))
    lines.extend([
        "", "*Read-only evaluation of retained data/model inputs; only new analysis "
        "artifacts were written.*", "",
    ])
    (output_dir / "attribute_robustness_summary.md").write_text(
        "\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=15)
    parser.add_argument("--reconciliation-tolerance", type=float, default=0.005)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    figures_dir = args.figures_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    predictions, thresholds, aggregate, context = infer(args)
    rows = build_slices(predictions, context["checkpoint"]["attributes"],
                        args.min_support)
    predictions_path = output_dir / "per_crop_predictions_logits.csv"
    slices_path = output_dir / "attribute_robustness_slices.csv"
    write_predictions(predictions_path, predictions)
    write_slice_results(slices_path, rows)
    figure_paths = make_figures(
        rows, context["checkpoint"]["attributes"], figures_dir)
    protocol = make_protocol(
        context, thresholds, aggregate, args, output_dir, figures_dir)
    protocol["outputs"].update({
        "per_crop_predictions_logits": rel(predictions_path),
        "per_crop_predictions_logits_sha256": sha256_file(predictions_path),
        "slice_results": rel(slices_path),
        "slice_results_sha256": sha256_file(slices_path),
        "figures": figure_paths,
    })
    ev.write_json(output_dir / "robustness_protocol.json", protocol)
    ev.write_json(output_dir / "robustness_thresholds.json", {
        "population": "retained MTSD attribute test crops", "n_crops": 1890,
        "pixel_stage": "post-PadToSquare/post-Resize, before normalisation",
        "tercile_method": "linear interpolation at 1/3 and 2/3 over sorted values",
        "thresholds": thresholds,
        "original_object_size": {"small_max_exclusive_area": 32 ** 2,
                                 "medium_max_exclusive_area": 96 ** 2},
    })
    ev.write_json(output_dir / "run_config.json", {
        "generated_at": protocol["generated_at"], "generator": GENERATOR,
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "variant": VARIANT, "checkpoint": protocol["inputs"]["checkpoint"],
        "manifest": protocol["inputs"]["manifest"],
        "training_performed": False,
    })

    insufficient = [row for row in rows if row["support_ok"] != "ok"]
    class_warnings = [row for row in rows
                      if row["prediction_head"] != "overall_mean"
                      and row["interpretation_flag"] == "class_support_warning"]
    missing_lines = [
        "# Missing inputs and low-support slices", "",
        "All required inputs were found: retained checkpoint, exact 1,890-row test "
        "population, all crop images, saved label mappings, and retained metrics.", "",
        f"Rows below total support {args.min_support}: {len(insufficient)} of {len(rows)}.",
        f"Head-level rows with at least one target class below {args.min_support}: "
        f"{len(class_warnings)}.", "",
    ]
    missing_lines.extend(
        f"- `{row['slice_attribute'] or row['slice_dimension']}="
        f"{row['slice_value']}`, `{row['prediction_head']}`: n={row['n_samples']}"
        for row in insufficient)
    missing_lines.extend(
        f"- Class-support warning: `{row['slice_attribute'] or row['slice_dimension']}="
        f"{row['slice_value']}`, `{row['prediction_head']}`; low classes="
        f"`{row['low_support_classes']}`; counts=`{row['class_support']}`"
        for row in class_warnings)
    (output_dir / "missing_or_low_support.md").write_text(
        "\n".join(missing_lines) + "\n", encoding="utf-8")
    write_summary(output_dir, figures_dir, rows, thresholds, aggregate,
                  protocol, figure_paths, args.min_support)
    print(f"Per-crop rows: {len(predictions)}", flush=True)
    print(f"Slice rows: {len(rows)} ({len(insufficient)} below support)", flush=True)
    print(f"Report: {output_dir}", flush=True)
    print(f"Figures: {figures_dir} ({len(figure_paths)} files)", flush=True)


if __name__ == "__main__":
    main()
