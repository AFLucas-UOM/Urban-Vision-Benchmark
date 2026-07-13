"""Configuration loading, path resolution, seeding, and logging setup.

The single source of truth is a YAML file (config/default.yaml by default).
All relative paths in the config are resolved against the repository root,
which is derived from this file's location and therefore independent of the
current working directory.
"""

import copy
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml

# <repo>/Scripts/MTSD-Scripts/AttributeClassification/mtsd_attr/config.py
REPO_ROOT = Path(__file__).resolve().parents[4]
SUBPROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = SUBPROJECT_ROOT / "config" / "default.yaml"

log = logging.getLogger("mtsd_attr")


def load_config(config_path=None):
    """Load the YAML config and resolve every entry under paths: to an absolute Path.

    Args:
        config_path: Optional path to a YAML file; defaults to config/default.yaml.

    Returns:
        dict with the parsed config; cfg["paths"] values are absolute Path objects.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    resolved = {}
    for key, value in cfg.get("paths", {}).items():
        p = Path(value)
        resolved[key] = p if p.is_absolute() else REPO_ROOT / p
    cfg["paths"] = resolved
    return cfg


ADAPTATION_MODES = ("frozen", "lora", "finetune")


def adaptation_of(model_cfg):
    """Return a variant's adaptation mode: "frozen", "lora", or "finetune".

    Falls back to the legacy boolean "frozen" key (used by checkpoints and
    configs written before adaptation modes existed).
    """
    mode = model_cfg.get("adaptation")
    if mode is None:
        mode = "frozen" if model_cfg.get("frozen", True) else "finetune"
    if mode not in ADAPTATION_MODES:
        raise ValueError(f"Unknown adaptation mode {mode!r}; "
                         f"expected one of {ADAPTATION_MODES}")
    return mode


def _normalise_training(training):
    """Normalise epoch-budget and early-stopping keys, with legacy fallbacks.

    max_epochs is a maximum budget (early stopping may end a run sooner). The
    legacy keys "epochs" and "early_stopping_patience" are still honoured when
    the new ones are absent.
    """
    training["max_epochs"] = int(
        training.get("max_epochs", training.get("epochs", 30)))
    training["batch_size"] = int(training["batch_size"])
    accum = int(training.get("gradient_accumulation_steps", 1))
    if accum < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1, "
                         f"got {accum}")
    training["gradient_accumulation_steps"] = accum
    training["effective_batch_size"] = training["batch_size"] * accum
    es = training.get("early_stopping")
    if es is None:
        legacy = int(training.get("early_stopping_patience", 0) or 0)
        es = {"enabled": legacy > 0, "patience": legacy or 10}
    es = {
        "enabled": bool(es.get("enabled", True)),
        "monitor": es.get("monitor", "mean_macro_f1"),
        "mode": es.get("mode", "max"),
        "patience": int(es.get("patience", 10)),
        "min_delta": float(es.get("min_delta", 0.0)),
    }
    if es["monitor"] != "mean_macro_f1" or es["mode"] != "max":
        raise ValueError(
            "early_stopping supports only monitor=mean_macro_f1 with mode=max; "
            f"got monitor={es['monitor']!r} mode={es['mode']!r}")
    training["early_stopping"] = es
    return training


def model_training_config(cfg, variant):
    """Return the training config for a model variant, with per-variant overrides applied.

    Args:
        cfg: Full config dict from load_config.
        variant: Key under cfg["models"], e.g. "convnext".

    Returns:
        (model_cfg, training_cfg) where training_cfg is the global training block
        deep-copied, updated with the variant's training_overrides (if any), and
        normalised (max_epochs, early_stopping).
    """
    model_cfg = cfg["models"][variant]
    training = copy.deepcopy(cfg["training"])
    training.update(model_cfg.get("training_overrides", {}))
    return model_cfg, _normalise_training(training)


def apply_smoke_test(cfg, training_cfg):
    """Overlay smoke-test settings onto a training config and return the sample cap.

    Args:
        cfg: Full config dict.
        training_cfg: Per-variant training dict to mutate.

    Returns:
        max_samples_per_split (int) to subsample each split with.
    """
    smoke = cfg["smoke_test"]
    training_cfg["max_epochs"] = int(
        smoke.get("max_epochs", smoke.get("epochs", 1)))
    training_cfg["batch_size"] = int(smoke["batch_size"])
    training_cfg["num_workers"] = smoke["num_workers"]
    # Accumulation stays as configured so smoke runs exercise the same
    # optimiser-step cadence; only the effective size shrinks with the batch.
    training_cfg["effective_batch_size"] = (
        training_cfg["batch_size"]
        * training_cfg.get("gradient_accumulation_steps", 1))
    if smoke.get("disable_wandb", True):
        os.environ["WANDB_MODE"] = "disabled"
    return smoke["max_samples_per_split"]


def set_seed(seed):
    """Seed Python, NumPy, and torch (CPU and CUDA) for reproducibility.

    cuDNN autotuning is left enabled for speed; residual non-determinism from
    cuDNN kernel selection and floating-point reduction order is documented in
    the README rather than eliminated.
    """
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_file=None, level=logging.INFO):
    """Configure the mtsd_attr logger with console output and an optional file.

    Args:
        log_file: Optional path; if given, log records are also appended there.
        level: Logging level for both handlers.

    Returns:
        The configured logger instance.
    """
    logger = logging.getLogger("mtsd_attr")
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    return logger


def git_commit_hash():
    """Return the repository's current git commit hash, or None if unavailable."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None
