#!/usr/bin/env python3
"""Inference-speed / deployment benchmark for the Urban-Vision-Benchmark models.

Measures practical deployment metrics - latency (mean/median/p95), throughput
(FPS), cold-start load time, peak GPU memory, model file size and parameter
count - for every trained model family in the repository:

  detection  : MDWD YOLO11/12/26 (n-l) and RF-DETR from Results/MDWD-Runs;
               MTSD detection models are discovered the same way from
               Results/MTSD-Runs and reported as PENDING until trained.
  attribute  : MTSD attribute classifiers (DINOv3 / V-JEPA / ConvNeXt in
               frozen / LoRA / fine-tuned variants) from
               Scripts/MTSD-Scripts/AttributeClassification/outputs/checkpoints.
  prompt     : PromptDetect models (SAM 3 / SAM 3.1 / Cosmos Reason2 /
               LocateAnything) via the existing DetectionBackend; heavy Cosmos
               32B is opt-in via --allow-heavy.

Benchmarking rules implemented:
  * torch.cuda.synchronize() before/after every timed call on CUDA;
  * explicit warmup iterations, excluded from measurements;
  * fixed --seed drives the image sample, so reruns use the same images
    (the exact file list is saved to benchmark_images_used.csv);
  * model loading is timed separately as cold-start, never as inference;
  * batch size 1 by default; --batch-sizes 1 4 8 opts into larger batches
    (engines without batch support fall back to 1 with a note);
  * every run writes to a fresh timestamped folder under
    Results/Inference-Benchmark/InferenceSpeed/ - nothing is overwritten.

Usage:
  python inference_speed_benchmark.py --list-models
  python inference_speed_benchmark.py --dataset MDWD --task detection \\
      --models yolo26 rf-detr --split test --max-images 50 --dry-run
  python inference_speed_benchmark.py --dataset MTSD --task attribute \\
      --models dinov3_lora convnext_finetuned --max-images 50 --dry-run
  python inference_speed_benchmark.py --dataset MTSD --task prompt \\
      --models sam3 locateanything --max-images 10 --prompt "traffic sign" --dry-run

Environments: run detection benchmarks from the `MDWD` conda env
(ultralytics / rfdetr), attribute benchmarks from `mtsd-attrcls`, and prompt
benchmarks from `mtsd-base`. Heavy imports are lazy, so --list-models and
--dry-run work from any Python.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Timing primitives are shared with the Jetson edge benchmark so both suites
# use one implementation of warmup / CUDA synchronisation / percentiles.
from uvb_bench_core import (  # noqa: E402
    clear_metric_lists,
    file_size_mb,
    find_project_root as _find_project_root,
    make_batches,
    percentile as _percentile,
    timed_loop,
    write_csv,
)


def find_project_root(start: Path = SCRIPT_DIR) -> Path:
    return _find_project_root(start)


PROJECT_ROOT = find_project_root()
RESULTS_ROOT = PROJECT_ROOT / "Results" / "Inference-Benchmark" / "InferenceSpeed"
MDWD_RUNS = PROJECT_ROOT / "Results" / "MDWD-Runs"
MTSD_RUNS = PROJECT_ROOT / "Results" / "MTSD-Runs"
ATTRCLS_DIR = PROJECT_ROOT / "Scripts" / "MTSD-Scripts" / "AttributeClassification"
ATTR_CHECKPOINTS = ATTRCLS_DIR / "outputs" / "checkpoints"
ATTR_CROPS = ATTRCLS_DIR / "outputs" / "crops"
PROMPTDETECT_DIR = PROJECT_ROOT / "Scripts" / "Other-Scripts" / "PromptDetect"
MDWD_YOLO_DATASET = PROJECT_ROOT / "Datasets" / "MDWD" / "MDWD-YOLO26"
MTSD_PREPARED_YOLO = PROJECT_ROOT / "Datasets" / "MTSD" / "Prepared" / "MTSD-YOLO"
MTSD_GROUPS_ROOT = PROJECT_ROOT / "Datasets" / "MTSD"

HEAVY_PROMPT_MODELS = {"Cosmos Reason2 32B"}
RFDETR_CHECKPOINT_PREFERENCE = ("checkpoint_best_total.pth", "checkpoint_best_ema.pth",
                                "checkpoint_best_regular.pth", "checkpoint.pth")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Selector aliases (user-facing convenience -> canonical id/prefix).
SELECTOR_ALIASES = {
    "convnext_finetuned": "convnext",
    "cosmos2b": "cosmos_reason2_2b",
    "cosmos8b": "cosmos_reason2_8b",
    "cosmos32b": "cosmos_reason2_32b",
    "rf-detr": "rfdetr",
    "rf_detr": "rfdetr",
}


@dataclass
class ModelEntry:
    id: str                    # user-facing selector, e.g. yolo26n, dinov3_lora, sam3
    task: str                  # detection | attribute | prompt
    dataset: str               # MDWD | MTSD | any
    family: str
    status: str                # ready | pending | heavy-opt-in
    checkpoint: str = ""       # path (empty for on-demand prompt models)
    suite: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def norm_id(self) -> str:
        return re.sub(r"[^a-z0-9@]", "", self.id.lower())

    @property
    def norm_family(self) -> str:
        return re.sub(r"[^a-z0-9]", "", self.family.lower())


# ---------------------------------------------------------------------------
# Model inventory
# ---------------------------------------------------------------------------

def _variant_from_run_name(run_name: str) -> str:
    """'E003_yolo26m_rfv20_img640_...' -> 'yolo26m'."""
    parts = run_name.split("_")
    return parts[1] if len(parts) > 1 else run_name


def _detection_entries(runs_root: Path, dataset: str) -> list[ModelEntry]:
    entries: list[ModelEntry] = []
    if not runs_root.is_dir():
        return entries
    suites = sorted(
        p for p in runs_root.iterdir()
        if p.is_dir() and not p.name.startswith("[OLD]")
    )
    # Prefer -EUVIP/-MTSD suites for the unsuffixed id; others get @<suite tag>.
    seen_variants: dict[str, str] = {}
    preferred = [s for s in suites if s.name.endswith(("-EUVIP", "-MTSD"))] + \
                [s for s in suites if not s.name.endswith(("-EUVIP", "-MTSD"))]
    for suite in preferred:
        for run_dir in sorted(p for p in suite.iterdir() if p.is_dir()):
            variant = _variant_from_run_name(run_dir.name)
            if variant.startswith("rfdetr"):
                checkpoint = next(
                    (run_dir / name for name in RFDETR_CHECKPOINT_PREFERENCE
                     if (run_dir / name).exists()), None)
                family = "rfdetr"
            else:
                checkpoint = run_dir / "weights" / "best.pt"
                checkpoint = checkpoint if checkpoint.exists() else None
                family = re.sub(r"[nsml]$", "", variant)
            if checkpoint is None:
                continue
            if variant in seen_variants:
                tag = suite.name.rsplit("-", 1)[-1].lower()
                entry_id = f"{variant}@{tag}"
            else:
                entry_id = variant
                seen_variants[variant] = suite.name
            entries.append(ModelEntry(
                id=entry_id, task="detection", dataset=dataset, family=family,
                status="ready", checkpoint=str(checkpoint), suite=suite.name,
                notes=f"trained run {run_dir.name}",
            ))
    return entries


def _pending_detection_entries(dataset: str) -> list[ModelEntry]:
    return [
        ModelEntry(id=family, task="detection", dataset=dataset, family=family,
                   status="pending",
                   notes="no trained checkpoint found - train via "
                         "Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py first")
        for family in ("yolo12", "yolo26", "rfdetr")
    ]


def _attribute_entries(include_smoke: bool) -> list[ModelEntry]:
    entries = []
    if not ATTR_CHECKPOINTS.is_dir():
        return entries
    display = {"convnext": "convnext (fine-tuned; alias: convnext_finetuned)"}
    for variant_dir in sorted(ATTR_CHECKPOINTS.iterdir()):
        if not variant_dir.is_dir():
            continue
        if variant_dir.name.endswith("-smoke") and not include_smoke:
            continue
        best = variant_dir / "best.pt"
        if not best.exists():
            continue
        family = variant_dir.name.split("_")[0].replace("-smoke", "")
        entries.append(ModelEntry(
            id=variant_dir.name, task="attribute", dataset="MTSD", family=family,
            status="ready", checkpoint=str(best),
            notes=display.get(variant_dir.name, ""),
        ))
    return entries


def _prompt_entries() -> list[ModelEntry]:
    entries = []
    try:
        if str(PROMPTDETECT_DIR / "batch_evaluation") not in sys.path:
            sys.path.insert(0, str(PROMPTDETECT_DIR / "batch_evaluation"))
        import config as pd_config  # PromptDetect batch_evaluation config

        for slug, label in sorted(pd_config.available_models().items()):
            heavy = label in HEAVY_PROMPT_MODELS
            entries.append(ModelEntry(
                id=slug, task="prompt", dataset="any", family=label.split()[0].lower(),
                status="heavy-opt-in" if heavy else "ready",
                notes=f"{label}; weights load on demand (HF-gated for SAM)"
                      + ("; requires --allow-heavy" if heavy else ""),
                extra={"label": label},
            ))
    except Exception as exc:
        entries.append(ModelEntry(
            id="promptdetect", task="prompt", dataset="any", family="promptdetect",
            status="pending",
            notes=f"PromptDetect backend not importable here ({type(exc).__name__}: {exc}); "
                  f"run from the mtsd-base env",
        ))
    return entries


def build_inventory(include_smoke: bool = False) -> list[ModelEntry]:
    entries = _detection_entries(MDWD_RUNS, "MDWD")
    mtsd_detection = _detection_entries(MTSD_RUNS, "MTSD")
    entries += mtsd_detection if mtsd_detection else _pending_detection_entries("MTSD")
    entries += _attribute_entries(include_smoke)
    entries += _prompt_entries()
    return entries


def select_models(entries: list[ModelEntry], selectors: list[str],
                  task: str, dataset: str) -> tuple[list[ModelEntry], list[str]]:
    """Match user selectors against inventory entries for one task/dataset.

    Precedence per selector: exact id match, else family match (all sizes /
    suites of that family), else id-prefix fallback. This keeps 'sam3' from
    matching SAM 3.1 and 'convnext_finetuned' from matching convnext_frozen,
    while 'yolo26' still selects every yolo26 size.
    """
    pool = [e for e in entries if e.task == task
            and (e.dataset in (dataset, "any"))]
    chosen: list[ModelEntry] = []
    unmatched: list[str] = []
    for raw in selectors:
        norm = SELECTOR_ALIASES.get(raw.lower(), raw.lower())
        norm_key = re.sub(r"[^a-z0-9@]", "", norm)
        hits = [e for e in pool if e.norm_id == norm_key]
        if not hits:
            hits = [e for e in pool if e.norm_family == norm_key]
        if not hits:
            hits = [e for e in pool if e.norm_id.startswith(norm_key)] if norm_key else []
        if not hits:
            unmatched.append(raw)
        for hit in hits:
            if hit not in chosen:
                chosen.append(hit)
    return chosen, unmatched


# ---------------------------------------------------------------------------
# Image sampling (seeded, recorded)
# ---------------------------------------------------------------------------

def sample_images(task: str, dataset: str, split: str, max_images: int, seed: int) -> list[Path]:
    if task == "attribute":
        pool = sorted(ATTR_CROPS.rglob("*.jpg"))
        source = ATTR_CROPS
    elif task == "detection" or task == "prompt":
        if dataset == "MDWD":
            source = MDWD_YOLO_DATASET / split / "images"
            pool = sorted(p for p in source.iterdir()
                          if p.suffix.lower() in IMAGE_EXTENSIONS) if source.is_dir() else []
        else:  # MTSD
            source = MTSD_PREPARED_YOLO / split / "images"
            if source.is_dir():
                pool = sorted(p for p in source.iterdir()
                              if p.suffix.lower() in IMAGE_EXTENSIONS)
            else:
                # Fall back to raw group images (prompt task / un-prepared MTSD).
                source = MTSD_GROUPS_ROOT
                pool = sorted(MTSD_GROUPS_ROOT.glob("GRP-*/Images/*"))
                pool = [p for p in pool if p.suffix.lower() in IMAGE_EXTENSIONS]
    else:
        raise ValueError(f"unknown task {task}")
    if not pool:
        raise FileNotFoundError(f"No benchmark images found under {source}")
    rng = random.Random(seed)
    if max_images and max_images < len(pool):
        return sorted(rng.sample(pool, max_images))
    return pool


# ---------------------------------------------------------------------------
# Timing core
# ---------------------------------------------------------------------------

def hardware_info(device: str) -> dict:
    import torch

    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


# ---------------------------------------------------------------------------
# Engines (heavy imports stay inside each function)
# ---------------------------------------------------------------------------

def benchmark_yolo(entry: ModelEntry, images: list[Path], args, device: str) -> dict:
    import torch
    from ultralytics import YOLO

    t0 = time.perf_counter()
    model = YOLO(entry.checkpoint)
    model.to(device)
    cold_start_s = time.perf_counter() - t0
    params = int(sum(p.numel() for p in model.model.parameters()))

    speeds = {"preprocess": [], "inference": [], "postprocess": []}

    def run_batch(batch):
        results = model.predict([str(p) for p in batch], imgsz=args.imgsz,
                                device=device, verbose=False)
        for result in results:
            for key in speeds:
                value = (result.speed or {}).get(key)
                if value is not None:
                    speeds[key].append(value)

    stats = timed_loop(
        run_batch, make_batches(images, args.batch_size), device, args.warmup,
        reset_after_warmup=lambda: clear_metric_lists(speeds),
    )
    return {
        **stats,
        "cold_start_s": round(cold_start_s, 2),
        "parameters": params,
        "preprocess_ms": round(statistics.fmean(speeds["preprocess"]), 2) if speeds["preprocess"] else None,
        "inference_ms": round(statistics.fmean(speeds["inference"]), 2) if speeds["inference"] else None,
        "postprocess_ms": round(statistics.fmean(speeds["postprocess"]), 2) if speeds["postprocess"] else None,
        "notes": entry.notes,
    }


def benchmark_rfdetr(entry: ModelEntry, images: list[Path], args, device: str) -> dict:
    from PIL import Image

    # Historical MTSD run folders use both ``rfdetr-m`` and ``rfdetrm``.
    # Parse the scale rather than assuming one particular run-name spelling.
    size_match = re.search(r"rfdetr[-_]?([nsm])", entry.id.lower())
    size_token = size_match.group(1) if size_match else None
    try:
        from rfdetr import RFDETRMedium, RFDETRNano, RFDETRSmall
    except ImportError as exc:
        raise RuntimeError(f"rfdetr package not installed in this env: {exc}") from exc
    model_class = {"n": RFDETRNano, "s": RFDETRSmall, "m": RFDETRMedium}.get(size_token)
    if model_class is None:
        raise RuntimeError(f"Unknown RF-DETR size '{size_token}' for {entry.id}")

    t0 = time.perf_counter()
    model = model_class(pretrain_weights=entry.checkpoint)
    cold_start_s = time.perf_counter() - t0
    try:
        params = int(sum(p.numel() for p in model.model.model.parameters()))
    except Exception:
        params = None

    loaded = [Image.open(p).convert("RGB") for p in images]

    def run_batch(batch):
        for image in batch:
            model.predict(image, threshold=args.conf_threshold)

    # rfdetr's high-level predict is single-image; batch>1 falls back to 1.
    stats = timed_loop(run_batch, make_batches(loaded, 1), device, args.warmup)
    note = "rfdetr .predict() is single-image; batch forced to 1"
    return {**stats, "cold_start_s": round(cold_start_s, 2), "parameters": params,
            "preprocess_ms": None, "inference_ms": None, "postprocess_ms": None,
            "notes": (entry.notes + "; " + note).strip("; ")}


def benchmark_attribute(entry: ModelEntry, images: list[Path], args, device: str) -> dict:
    import torch
    from PIL import Image

    if str(ATTRCLS_DIR) not in sys.path:
        sys.path.insert(0, str(ATTRCLS_DIR))
    from mtsd_attr.backbones import build_backbone
    from mtsd_attr.config import adaptation_of, load_config
    from mtsd_attr.dataset import build_transforms
    from mtsd_attr.multihead_model import MultiHeadClassifier
    from mtsd_attr.train_common import _load_checkpoint_state

    t0 = time.perf_counter()
    ckpt = torch.load(entry.checkpoint, map_location="cpu", weights_only=False)
    model_cfg = ckpt["model_cfg"]
    adaptation = ckpt.get("adaptation") or adaptation_of(model_cfg)
    backbone = build_backbone(model_cfg)
    model = MultiHeadClassifier(backbone, ckpt["attributes"], ckpt["probe"], adaptation)
    _load_checkpoint_state(model, ckpt["model_state"], adaptation)
    model = model.to(device).eval()
    transform = build_transforms(backbone.image_size, False,
                                 load_config()["training"]["augmentation"])
    cold_start_s = time.perf_counter() - t0
    params = int(sum(p.numel() for p in model.parameters()))

    # Preprocess (transform) timed separately from the forward pass.
    t_pre = time.perf_counter()
    tensors = [transform(Image.open(p).convert("RGB")) for p in images]
    preprocess_ms = (time.perf_counter() - t_pre) * 1000 / len(tensors)

    def run_batch(batch):
        stacked = torch.stack(batch).to(device)
        with torch.inference_mode():
            model(stacked)

    stats = timed_loop(run_batch, make_batches(tensors, args.batch_size), device, args.warmup)
    return {**stats, "cold_start_s": round(cold_start_s, 2), "parameters": params,
            "preprocess_ms": round(preprocess_ms, 2), "inference_ms": None,
            "postprocess_ms": None, "notes": entry.notes}


def benchmark_prompt(entry: ModelEntry, images: list[Path], args, device: str) -> dict:
    import numpy as np
    from PIL import Image

    if str(PROMPTDETECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROMPTDETECT_DIR))
    from backend import DetectionBackend

    label = entry.extra.get("label", entry.id)
    backend = DetectionBackend(device=device)
    t0 = time.perf_counter()
    status = backend.load(label)
    cold_start_s = time.perf_counter() - t0
    if not status.get("ok"):
        raise RuntimeError(f"model load failed: {status.get('error')}")

    loaded = [np.array(Image.open(p).convert("RGB")) for p in images]
    breakdown = {"preprocess": [], "inference": [], "postprocess": []}

    def run_batch(batch):
        for image in batch:
            result = backend.predict(image=image, text_prompt=args.prompt,
                                     conf_threshold=args.conf_threshold)
            for key in breakdown:
                breakdown[key].append(result["runtime"][key] * 1000)

    try:
        stats = timed_loop(
            run_batch, make_batches(loaded, 1), device, args.warmup,
            reset_after_warmup=lambda: clear_metric_lists(breakdown),
        )
    finally:
        try:
            backend._loaded.engine.close()  # noqa: SLF001 - no public close API
        except Exception:
            pass
    note = f"prompt='{args.prompt}'; single-image API, batch forced to 1"
    return {
        **stats, "cold_start_s": round(cold_start_s, 2), "parameters": None,
        "preprocess_ms": round(statistics.fmean(breakdown["preprocess"]), 2),
        "inference_ms": round(statistics.fmean(breakdown["inference"]), 2),
        "postprocess_ms": round(statistics.fmean(breakdown["postprocess"]), 2),
        "notes": (entry.notes + "; " + note).strip("; "),
    }


ENGINES = {
    ("detection", "yolo"): benchmark_yolo,
    ("detection", "rfdetr"): benchmark_rfdetr,
    ("attribute", None): benchmark_attribute,
    ("prompt", None): benchmark_prompt,
}


def engine_for(entry: ModelEntry):
    if entry.task == "detection":
        return benchmark_rfdetr if entry.family == "rfdetr" else benchmark_yolo
    if entry.task == "attribute":
        return benchmark_attribute
    return benchmark_prompt


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_summary_md(path: Path, rows: list[dict], config: dict) -> None:
    lines = [
        "# Inference speed benchmark",
        f"\nGenerated: {config['started_at']}  ",
        f"Dataset: **{config['dataset']}** | task: **{config['task']}** | "
        f"split: {config.get('split', '-')} | images: {config['n_images']} | "
        f"batch size: {config['batch_size']} | warmup: {config['warmup']} | "
        f"seed: {config['seed']}  ",
        f"Hardware: {config['hardware'].get('gpu_name') or config['hardware']['processor']} "
        f"({config['hardware']['device']}), torch {config['hardware']['torch']}",
        "\n| Model | Status | Mean ms | Median ms | p95 ms | FPS | Peak GPU GB | "
        "Params (M) | File MB | Cold start s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        params_m = f"{row['parameters'] / 1e6:.1f}" if row.get("parameters") else "-"
        lines.append(
            f"| {row['model']} | {row['status']} | {row.get('mean_latency_ms', '-')} | "
            f"{row.get('median_latency_ms', '-')} | {row.get('p95_latency_ms', '-')} | "
            f"{row.get('fps', '-')} | {row.get('peak_gpu_mem_gb') if row.get('peak_gpu_mem_gb') is not None else '-'} | "
            f"{params_m} | {row.get('model_file_mb') if row.get('model_file_mb') is not None else '-'} | "
            f"{row.get('cold_start_s', '-')} |"
        )
    failed = [row for row in rows if row["status"] != "ok"]
    if failed:
        lines.append("\n## Not benchmarked\n")
        for row in failed:
            lines.append(f"- **{row['model']}** - {row['status']}: {row.get('error', row.get('notes', ''))}")
    lines.append("\nLatency statistics are per image, warmup excluded, CUDA-synchronised. "
                 "Cold start = model construction + weight loading, reported separately.\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_plots(run_dir: Path, rows: list[dict]) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    names = [row["model"] for row in ok_rows]
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 1.4 * len(names)), 4.5))
    axes[0].bar(names, [row["mean_latency_ms"] for row in ok_rows], color="#2a78d6")
    axes[0].errorbar(names, [row["mean_latency_ms"] for row in ok_rows],
                     yerr=[[0] * len(ok_rows),
                           [row["p95_latency_ms"] - row["mean_latency_ms"] for row in ok_rows]],
                     fmt="none", ecolor="#52514e", capsize=4)
    axes[0].set_title("Mean latency per image (whisker to p95)")
    axes[0].set_ylabel("ms")
    axes[1].bar(names, [row["fps"] for row in ok_rows], color="#1baf7a")
    axes[1].set_title("Throughput")
    axes[1].set_ylabel("FPS")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(run_dir / "latency_fps.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_inventory(entries: list[ModelEntry]) -> None:
    print(f"{'ID':<22} {'TASK':<10} {'DATASET':<8} {'STATUS':<13} {'SUITE':<14} CHECKPOINT / NOTES")
    print("-" * 110)
    for entry in entries:
        location = entry.checkpoint or entry.notes
        if entry.checkpoint:
            try:
                location = str(Path(entry.checkpoint).relative_to(PROJECT_ROOT))
            except ValueError:
                pass
        print(f"{entry.id:<22} {entry.task:<10} {entry.dataset:<8} {entry.status:<13} "
              f"{entry.suite:<14} {location}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deployment / inference-speed benchmark (see module docstring).")
    parser.add_argument("--list-models", action="store_true",
                        help="Print the model inventory and exit.")
    parser.add_argument("--include-smoke", action="store_true",
                        help="Also list/benchmark AttrCls smoke checkpoints.")
    parser.add_argument("--dataset", choices=["MDWD", "MTSD", "PromptDetect"],
                        help="Image source; 'PromptDetect' is an alias for MTSD images.")
    parser.add_argument("--task", choices=["detection", "attribute", "prompt"])
    parser.add_argument("--models", nargs="+", default=[],
                        help="Model ids/families from --list-models "
                             "(e.g. yolo26, yolo26n, rf-detr, dinov3_lora, sam3).")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None,
                        help="Optional list (e.g. 1 4 8) - one summary row per size.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-threshold", type=float, default=0.30)
    parser.add_argument("--prompt", default="traffic sign",
                        help="Text prompt for the prompt task.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--allow-heavy", action="store_true",
                        help="Required for Cosmos Reason2 32B.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve models + images and print the plan; load nothing, write nothing.")
    args = parser.parse_args()

    inventory = build_inventory(args.include_smoke)
    if args.list_models or not (args.dataset and args.task):
        print_inventory(inventory)
        if not args.list_models and not (args.dataset and args.task):
            print("\nSpecify --dataset and --task (and --models) to benchmark; see --help.")
        return 0

    dataset = "MTSD" if args.dataset == "PromptDetect" else args.dataset
    if not args.models:
        raise SystemExit("Select models explicitly with --models (see --list-models).")
    chosen, unmatched = select_models(inventory, args.models, args.task, dataset)
    if unmatched:
        raise SystemExit(f"No inventory match for: {unmatched}. Run --list-models.")
    heavy = [e for e in chosen if e.status == "heavy-opt-in"]
    if heavy and not args.allow_heavy:
        raise SystemExit(f"Heavy model(s) need --allow-heavy: {[e.id for e in heavy]}")
    pending = [e for e in chosen if e.status == "pending"]
    runnable = [e for e in chosen if e.status in ("ready", "heavy-opt-in")]

    images = sample_images(args.task, dataset, args.split, args.max_images, args.seed)
    batch_sizes = args.batch_sizes or [args.batch_size]

    print(f"Task     : {args.task} on {dataset} (split={args.split})")
    print(f"Images   : {len(images)} (seed {args.seed})")
    print(f"Batching : {batch_sizes} | warmup {args.warmup}")
    print(f"Models   : {[e.id for e in runnable]}")
    if pending:
        print(f"Pending  : {[e.id for e in pending]} (no checkpoints - skipped)")

    if args.dry_run:
        print("\nDRY RUN - no models loaded, nothing written.")
        for entry in runnable:
            print(f"  would benchmark {entry.id:<20} [{entry.suite or entry.task}] "
                  f"ckpt={entry.checkpoint or '(on demand)'}")
        return 0

    import torch

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available in this env.")
    hw = hardware_info(device)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS_ROOT / stamp
    run_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset, "task": args.task, "split": args.split,
        "models_requested": args.models, "models_resolved": [e.id for e in chosen],
        "n_images": len(images), "batch_size": args.batch_size,
        "batch_sizes": batch_sizes, "warmup": args.warmup, "seed": args.seed,
        "imgsz": args.imgsz, "conf_threshold": args.conf_threshold,
        "prompt": args.prompt if args.task == "prompt" else None,
        "hardware": hw,
    }
    (run_dir / "benchmark_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    write_csv(run_dir / "benchmark_images_used.csv",
              [{"index": i, "image": str(p)} for i, p in enumerate(images)])
    write_csv(run_dir / "model_inventory.csv", [asdict(e) for e in inventory])

    summary_rows, raw_rows = [], []
    for entry in runnable:
        for batch_size in batch_sizes:
            args.batch_size = batch_size
            base = {
                "model": entry.id, "task": entry.task, "dataset": dataset,
                "suite": entry.suite, "checkpoint": entry.checkpoint,
                "input_size": args.imgsz if entry.task != "attribute" else "model-native",
                "batch_size": batch_size, "device": device,
                "gpu_name": hw["gpu_name"], "warmup_iterations": args.warmup,
                "model_file_mb": file_size_mb(entry.checkpoint),
            }
            print(f"\n=== {entry.id} (batch={batch_size}) ===")
            try:
                result = engine_for(entry)(entry, images, args, device)
                per_image = result.pop("per_image_latency_ms")
                raw_rows.extend(
                    {"model": entry.id, "batch_size": batch_size,
                     "iteration": i, "latency_ms": round(v, 3)}
                    for i, v in enumerate(per_image))
                summary_rows.append({**base, "status": "ok", **result})
                print(f"    mean {result['mean_latency_ms']} ms | p95 "
                      f"{result['p95_latency_ms']} ms | {result['fps']} FPS")
            except Exception as exc:
                summary_rows.append({**base, "status": "failed",
                                     "error": f"{type(exc).__name__}: {exc}",
                                     "notes": entry.notes})
                print(f"    FAILED: {type(exc).__name__}: {exc}")
    for entry in pending:
        summary_rows.append({"model": entry.id, "task": entry.task, "dataset": dataset,
                             "status": "pending", "notes": entry.notes})

    write_csv(run_dir / "inference_speed_summary.csv", summary_rows)
    write_csv(run_dir / "inference_speed_raw_timings.csv", raw_rows)
    write_summary_md(run_dir / "inference_speed_summary.md", summary_rows, config)
    save_plots(run_dir, summary_rows)
    print(f"\nResults written to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
