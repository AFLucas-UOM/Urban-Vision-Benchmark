from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    from prompt_sensitivity import validate_sensitivity_families

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not payload.get("protocol_version") or not isinstance(payload.get("datasets"), dict):
        raise ValueError(f"Invalid protocol file: {path}")
    ids = []
    for dataset, spec in payload["datasets"].items():
        prompts = spec.get("prompts", [])
        for prompt in prompts:
            for key in ("id", "prompt", "group", "target_classes"):
                if key not in prompt: raise ValueError(f"{dataset} prompt missing {key}: {prompt}")
            ids.append(prompt["id"])
        validate_sensitivity_families(dataset, spec)
    if len(ids) != len(set(ids)): raise ValueError("Protocol prompt IDs must be globally unique")
    payload["protocol_hash"] = canonical_hash(payload)
    payload["protocol_path"] = str(path)
    return payload


def validate_vocabulary(dataset: str, spec: dict[str, Any], actual: list[str]) -> None:
    expected = spec.get("class_vocabulary")
    if expected and list(expected) != list(actual):
        raise ValueError(f"{dataset} vocabulary mismatch. Loaded={actual!r}; protocol={expected!r}")
    known = set(actual)
    for prompt in spec.get("prompts", []):
        unknown = set(prompt["target_classes"]) - known
        if unknown: raise ValueError(f"Prompt {prompt['id']} has unknown target classes {sorted(unknown)}; vocabulary={actual}")


def select_prompts(spec: dict[str, Any], ids: list[str] | None = None, group: str | None = None,
                   include_optional: bool = False) -> list[dict[str, Any]]:
    prompts = []
    for row in spec["prompts"]:
        if ids and row["id"] not in ids: continue
        if group and row["group"] != group: continue
        if not include_optional and row.get("include_in_primary", True) is False: continue
        prompts.append(row)
    if ids:
        missing = set(ids) - {row["id"] for row in prompts}
        if missing: raise ValueError(f"Unknown or excluded prompt IDs: {sorted(missing)}")
    return prompts
