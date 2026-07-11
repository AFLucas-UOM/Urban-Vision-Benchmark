from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def compatibility_diff(existing: dict[str, Any], expected: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    keys = ("dataset_version", "prep_manifest_sha256", "split_manifest_sha256",
            "annotation_source_mode", "dataset_variant", "config_fingerprint", "requested_matrix")
    return {key: (existing.get(key), expected.get(key)) for key in keys if existing.get(key) != expected.get(key)}


def load_compatible(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    diff = compatibility_diff(state, expected)
    if diff: raise ValueError(f"Resume state is incompatible: {diff}")
    return state

