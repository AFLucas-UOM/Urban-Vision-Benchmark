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
MTSD_PREPARED_YOLO_DIR = DATASETS_DIR / "MTSD" / "Prepared" / "MTSD-YOLO"
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
MAX_DETECTIONS = 100
IOU_MATCH_THRESHOLD = 0.5          # TP threshold for P/R/F1
MAP_IOU_RANGE = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95


def new_run_dir(dataset: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS_ROOT / dataset.upper() / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
