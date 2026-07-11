from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .utils import find_repo_root, fingerprint


def _resolve_paths(value: Any, root: Path) -> Any:
    data = deepcopy(value)
    for key in ("annotations_root", "prepared_root"):
        raw = data.get("dataset", {}).get(key)
        if raw:
            path = Path(raw)
            data["dataset"][key] = str(path if path.is_absolute() else root / path)
    for key in ("runs_root", "results_root", "state_root"):
        raw = data.get("outputs", {}).get(key)
        if raw:
            path = Path(raw)
            data["outputs"][key] = str(path if path.is_absolute() else root / path)
    return data


def load_config(path: Path | str, repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or find_repo_root()
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    required = ("dataset", "annotations", "split", "augmentation", "training", "matrix", "validation", "wandb", "outputs")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Configuration missing sections: {missing}")
    ratios = cfg["split"]["ratios"]
    if abs(sum(float(ratios[k]) for k in ("train", "valid", "test")) - 1.0) > 1e-9:
        raise ValueError("split.ratios must sum to 1")
    if cfg["split"].get("algorithm_version") != "v1":
        raise ValueError("Only split algorithm v1 is implemented")
    scope = cfg["annotations"].get("group_scope", "auto")
    if scope not in {"auto", "explicit"}:
        raise ValueError("annotations.group_scope must be auto or explicit")
    if scope == "explicit" and not cfg["annotations"].get("approved_groups"):
        raise ValueError("annotations.approved_groups is required for explicit scope")
    cfg = _resolve_paths(cfg, root)
    cfg["repo_root"] = str(root)
    cfg["config_fingerprint"] = fingerprint({k: v for k, v in cfg.items() if k != "config_fingerprint"})
    return cfg
