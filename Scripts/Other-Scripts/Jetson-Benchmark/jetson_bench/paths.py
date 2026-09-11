"""Repository-relative path resolution and cache/environment policy.

Nothing here hardcodes a mount point. The repository root is discovered by
walking up from this file until ``Datasets/`` and ``Scripts/`` are both
present, exactly as the existing workstation benchmark does, so the same
external SSD works whether it mounts at ``/media/user/SSD``, ``/mnt/ssd`` or
anywhere else.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
JETSON_BENCH_DIR = PACKAGE_DIR.parent
INFERENCE_BENCH_DIR = JETSON_BENCH_DIR.parent / "Inference-Benchmark"

if str(INFERENCE_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_BENCH_DIR))

from uvb_bench_core import find_project_root  # noqa: E402

PROJECT_ROOT = find_project_root(PACKAGE_DIR)

# --- Inputs (read-only) ----------------------------------------------------
DATASETS_DIR = PROJECT_ROOT / "Datasets"
MODELS_DIR = PROJECT_ROOT / "Models"
MDWD_RUNS = PROJECT_ROOT / "Results" / "MDWD-Runs"
MTSD_RUNS = PROJECT_ROOT / "Results" / "MTSD-Runs"
ATTRCLS_DIR = PROJECT_ROOT / "Scripts" / "MTSD-Scripts" / "AttributeClassification"
ATTR_CHECKPOINTS = ATTRCLS_DIR / "outputs" / "checkpoints"
ATTR_CROPS = ATTRCLS_DIR / "outputs" / "crops"
ATTR_REPORTS = ATTRCLS_DIR / "outputs" / "reports"
PROMPTDETECT_DIR = PROJECT_ROOT / "Scripts" / "Other-Scripts" / "PromptDetect"
PROMPT_PROTOCOLS = PROMPTDETECT_DIR / "batch_evaluation" / "prompt_protocols"

MDWD_YOLO_DATASET = DATASETS_DIR / "MDWD" / "MDWD-YOLO26"
MTSD_STRONG_YOLO = DATASETS_DIR / "MTSD" / "Prepared" / "MTSD-Augmented-Strong" / "MTSD-YOLO"
MTSD_UNAUG_YOLO = DATASETS_DIR / "MTSD" / "Prepared" / "MTSD-Unaugmented" / "MTSD-YOLO"
MTSD_GROUPS_ROOT = DATASETS_DIR / "MTSD"

# Existing workstation benchmark evidence (the RTX 4090 comparison baseline).
WORKSTATION_RUNS = PROJECT_ROOT / "Results" / "Inference-Benchmark" / "InferenceSpeed"
WORKSTATION_FINAL_TABLE = (PROJECT_ROOT / "Documents" / "Final-Tables"
                           / "20260910-inference-benchmark"
                           / "all_model_inference_benchmark.csv")

# Canonical predictive-accuracy sources.
MDWD_DETECTION_RESULTS = (PROJECT_ROOT / "Documents" / "Final-Tables" / "20260712-213548"
                          / "mdwd_detection_results.csv")
MTSD_EXPERIMENT_MATRIX = (PROJECT_ROOT / "Documents" / "Final-Reports"
                          / "MTSD-SupervisedDetection" / "mtsd_complete_experiment_matrix.csv")
MTSD_EXPERIMENT_AUDIT = PROJECT_ROOT / "MTSD-SupervisedOD_ExperimentAudit.csv"
ATTR_SIZE_ABLATION = ATTR_REPORTS / "size_ablation.csv"
PROMPT_SENSITIVITY_ROOT = PROJECT_ROOT / "Results" / "PromptDetect" / "BatchEvaluation"

# --- Outputs (always new, never overwritten) -------------------------------
RESULTS_ROOT = PROJECT_ROOT / "Results" / "Jetson-Benchmark"

# --- Jetson-local, git-ignored working areas -------------------------------
CACHE_ROOT = PROJECT_ROOT / ".cache" / "jetson"
VENV_ROOT = PROJECT_ROOT
VENV_NAMES = {
    "detection": ".venv-jetson-detection",
    "attribute": ".venv-jetson-attribute",
    "prompt": ".venv-jetson-prompt",
}
SETUP_STATE = CACHE_ROOT / "setup_state.json"

CONFIG_DIR = JETSON_BENCH_DIR / "config"
DEFAULT_CONFIG = CONFIG_DIR / "jetson_benchmark.yaml"


def venv_dir(track_env: str) -> Path:
    """Absolute path of one Jetson virtual environment on the SSD."""
    return VENV_ROOT / VENV_NAMES[track_env]


def venv_python(track_env: str) -> Path:
    """Interpreter inside one Jetson virtual environment (POSIX layout)."""
    return venv_dir(track_env) / "bin" / "python"


def repo_relative(path: str | Path) -> str:
    """Best-effort repository-relative POSIX path.

    Historical artefacts in this repository store absolute Windows paths. Every
    path this package writes is normalised through here so no personal path of
    any machine ends up in a benchmark artefact.
    """
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except (ValueError, OSError):
        return Path(str(path)).as_posix()


def resolve_recorded_path(recorded: str) -> Path | None:
    """Map a path recorded by an earlier run onto this checkout.

    Accepts Windows (``E:\\...\\Urban-Vision-Benchmark\\Datasets\\...``) and
    POSIX absolute paths as well as repository-relative ones, and rebuilds them
    against the current ``PROJECT_ROOT``. Returns ``None`` when the identity
    cannot be recovered.
    """
    if not recorded:
        return None
    text = str(recorded).strip().replace("\\", "/")
    marker = "Urban-Vision-Benchmark/"
    if marker in text:
        text = text.split(marker, 1)[1]
    elif Path(text).is_absolute():
        # An absolute path from another machine with a differently named root:
        # anchor on the first known top-level repository directory.
        parts = Path(text).parts
        for anchor in ("Datasets", "Scripts", "Results", "Models", "Documents"):
            if anchor in parts:
                text = "/".join(parts[parts.index(anchor):])
                break
        else:
            return None
    candidate = PROJECT_ROOT / text
    return candidate if candidate.exists() else None


CACHE_ENVIRONMENT = {
    # Keep every large download and compiled asset on the SSD rather than the
    # Jetson's small internal eMMC/NVMe root filesystem.
    "HF_HOME": CACHE_ROOT / "huggingface",
    "HUGGINGFACE_HUB_CACHE": CACHE_ROOT / "huggingface" / "hub",
    "TORCH_HOME": CACHE_ROOT / "torch",
    "XDG_CACHE_HOME": CACHE_ROOT / "xdg",
    "PIP_CACHE_DIR": CACHE_ROOT / "pip",
    "YOLO_CONFIG_DIR": CACHE_ROOT / "ultralytics",
    "ULTRALYTICS_CONFIG_DIR": CACHE_ROOT / "ultralytics",
    "MODEL_EXPORT_DIR": CACHE_ROOT / "model_exports",
}

CACHE_FLAGS = {
    # Street-level imagery and telemetry stay on the device: this benchmark is
    # locally authoritative and never uploads anything.
    "WANDB_MODE": "disabled",
    "WANDB_DISABLED": "true",
    "MPLBACKEND": "Agg",
    # Keep a Jetson's four/six/eight cores from being oversubscribed by BLAS
    # threads during a batch-1 latency measurement.
    "TOKENIZERS_PARALLELISM": "false",
}


def benchmark_environment(allow_downloads: bool = False) -> dict[str, str]:
    """Environment variables every model process is started with."""
    env: dict[str, str] = {key: str(value) for key, value in CACHE_ENVIRONMENT.items()}
    env.update(CACHE_FLAGS)
    # Offline unless downloads were explicitly permitted. This is what keeps a
    # required benchmark honest: it must run from the SSD's local files.
    env["HF_HUB_OFFLINE"] = "0" if allow_downloads else "1"
    env["TRANSFORMERS_OFFLINE"] = "0" if allow_downloads else "1"
    env["UVB_ALLOW_MODEL_DOWNLOADS"] = "1" if allow_downloads else "0"
    return env


def ensure_cache_dirs() -> None:
    for value in CACHE_ENVIRONMENT.values():
        Path(value).mkdir(parents=True, exist_ok=True)


def apply_benchmark_environment(allow_downloads: bool = False) -> dict[str, str]:
    """Create the cache directories and export the policy into ``os.environ``."""
    ensure_cache_dirs()
    env = benchmark_environment(allow_downloads)
    os.environ.update(env)
    return env
