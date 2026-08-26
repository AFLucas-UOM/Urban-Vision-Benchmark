"""Configuration for PromptDetect batch evaluation.

Paths derive from the repository root (found by walking up), model metadata
comes from the existing PromptDetect backend registry, and every run writes
into its own timestamped folder under Results/PromptDetect/BatchEvaluation/.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROMPTDETECT_DIR = PACKAGE_DIR.parent
if str(PROMPTDETECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROMPTDETECT_DIR))


def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "Datasets").exists() and (parent / "Scripts").exists():
            return parent
    raise RuntimeError("Could not locate project root (needs Datasets/ and Scripts/).")


PROJECT_ROOT = find_project_root()
DATASETS_DIR = PROJECT_ROOT / "Datasets"
RESULTS_ROOT = PROJECT_ROOT / "Results" / "PromptDetect" / "BatchEvaluation"

# Dataset locations -------------------------------------------------------------
MDWD_YOLO_DIR = DATASETS_DIR / "MDWD" / "MDWD-YOLO26"
MTSD_PREPARED_VARIANT_DIR = DATASETS_DIR / "MTSD" / "Prepared" / "MTSD-Unaugmented"
MTSD_PREPARED_YOLO_DIR = MTSD_PREPARED_VARIANT_DIR / "MTSD-YOLO"
MTSD_LEGACY_PREPARED_YOLO_DIR = DATASETS_DIR / "MTSD" / "Prepared" / "MTSD-YOLO"
MTSD_ANNOTATIONS_ROOT = DATASETS_DIR / "MTSD" / "Annotations"
MTSD_QA_SPLIT_RATIOS = {"train": 0.80, "valid": 0.10, "test": 0.10}
MTSD_QA_SPLIT_SEED = 42

SPLIT_ALIASES = {"val": "valid", "validation": "valid"}
VALID_SPLITS = ("train", "valid", "test", "all")

# Prompts -----------------------------------------------------------------------
# Zero prompts are allowed only for dry-run / dataset sanity checks. Real
# evaluation still needs at least one prompt to produce metrics.
MIN_PROMPTS, MAX_PROMPTS = 0, 15

# Models ------------------------------------------------------------------------
# Heavy checkpoints are OPT-IN: they are never selected by default and the CLI
# refuses them without --allow-heavy (the Gradio UI has an equivalent gate).
HEAVY_MODEL_LABELS = {"Cosmos Reason2 32B"}
MODEL_ALIASES = {
    "sam3": "sam_3",
    "sam31": "sam_3.1",
    "sam3.1": "sam_3.1",
    "cosmos2b": "cosmos_reason2_2b",
    "cosmos8b": "cosmos_reason2_8b",
    "cosmos32b": "cosmos_reason2_32b",
    "locateanything": "locateanything_3b",
}


def model_slug(label: str) -> str:
    """CLI-friendly alias for a backend model label ('SAM 3.1' -> 'sam3.1')."""
    return re.sub(r"[^a-z0-9.]+", "_", label.lower()).strip("_")


def available_models() -> dict[str, str]:
    """slug -> backend label, built from the live PromptDetect registry."""
    from backend import MODELS  # noqa: PLC0415 - resolved via PROMPTDETECT_DIR

    return {model_slug(label): label for label in MODELS}


# Matching / metrics defaults -----------------------------------------------------
CONF_THRESHOLD = 0.30
MAX_DETECTIONS = 28  # fallback only; final protocols derive this per test set
IOU_MATCH_THRESHOLD = 0.5          # TP threshold for P/R/F1
MAP_IOU_RANGE = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95

# Generative VLMs are reloaded in bounded image chunks. The load overhead is
# small compared with generation time and guarantees that Windows can reclaim
# model/checkpoint memory regularly during multi-day evaluations.
VLM_WORKER_CHUNK_SIZE = 64

# Cosmos must not receive native 3K/4K MTSD frames. Qwen3-VL's visual token
# count grows with image area; unbounded frames took 25 hours per prompt and a
# single 64-image worker reached ~98 GB private RAM / 23.7 GB VRAM. Coordinates
# are emitted on a normalised 0..1000 grid, so inference uses a bounded copy
# while boxes are mapped back to the original dimensions. The two checkpoints
# need different caps: the 2B model remains fast at 2560 and was more accurate
# than full resolution on two target-dense validation slices; 8B needs 1536 to
# retain safe headroom on a 24 GB RTX 4090.
COSMOS_MAX_SIDE_DEFAULT = 1536
COSMOS_MAX_SIDE_BY_MODEL = {
    "Cosmos Reason2 2B": 2560,
    "Cosmos Reason2 8B": 1536,
    "Cosmos Reason2 32B": 1536,
}
COSMOS_WORKER_CHUNK_SIZE = 8
COSMOS_EXECUTION_REVISION = "bounded-model-specific-2b2560-8b1536-chunk8-v2"
WANDB_RUN_REVISION = "bounded-v2"


def cosmos_max_side(model: str) -> int:
    """Return the audited inference-image cap for one Cosmos checkpoint."""
    return int(COSMOS_MAX_SIDE_BY_MODEL.get(model, COSMOS_MAX_SIDE_DEFAULT))


def new_run_dir(dataset: str, run_label: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"-{re.sub(r'[^A-Za-z0-9_.-]+', '-', run_label).strip('-')}" if run_label else ""
    run_dir = RESULTS_ROOT / dataset.upper() / f"{stamp}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
