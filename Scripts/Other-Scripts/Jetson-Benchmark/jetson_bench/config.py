"""Benchmark configuration: YAML load, profile application, CLI overrides.

Every tunable is explicit and documented in
``Scripts/Other-Scripts/Jetson-Benchmark/config/jetson_benchmark.yaml`` - there
are no hidden magic numbers in the runner.

PyYAML is used when available. A deliberately small pure-standard-library
reader covers the subset the shipped configuration uses (nested mappings,
block sequences, inline flow sequences, scalars) so that ``--dry-run``, the
unit tests and the preflight all work on a bare Python installation before the
Jetson environments have been built.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

TRACKS = ("mdwd", "mtsd", "attributes", "zero-shot")
TIMING_BOUNDARIES = ("model_forward", "in_memory_pipeline", "end_to_end_file")


class ConfigError(ValueError):
    """Raised for a configuration that cannot be used as written."""


# ---------------------------------------------------------------------------
# Minimal YAML subset reader (fallback only)
# ---------------------------------------------------------------------------

_SCALARS = {"true": True, "false": False, "null": None, "~": None,
            "yes": True, "no": False}


def _scalar(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    if text[0] in "'\"" and text[-1] == text[0] and len(text) >= 2:
        return text[1:-1]
    lowered = text.lower()
    if lowered in _SCALARS:
        return _SCALARS[lowered]
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", text):
        return float(text)
    return text


def _flow_sequence(text: str) -> list:
    body = text.strip()[1:-1].strip()
    if not body:
        return []
    items, depth, current = [], 0, ""
    for char in body:
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        if char == "," and depth == 0:
            items.append(current)
            current = ""
        else:
            current += char
    items.append(current)
    return [_value(item.strip()) for item in items]


def _value(text: str) -> Any:
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return _flow_sequence(text)
    if text.startswith("{") and text.endswith("}"):
        body = text[1:-1].strip()
        result: dict[str, Any] = {}
        if body:
            for pair in _flow_sequence("[" + body + "]"):
                key, _, val = str(pair).partition(":")
                result[key.strip()] = _value(val)
        return result
    return _scalar(text)


def _strip_comment(line: str) -> str:
    out, quote = [], None
    for char in line:
        if quote:
            out.append(char)
            if char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
            out.append(char)
            continue
        if char == "#":
            break
        out.append(char)
    return "".join(out).rstrip()


def _parse_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines):
        return None, index
    depth, text = lines[index]
    if text.startswith("- "):
        items = []
        while index < len(lines) and lines[index][0] == indent and lines[index][1].startswith("- "):
            body = lines[index][1][2:].strip()
            if ":" in body and not body.startswith(("[", "{")) and _looks_like_mapping(body):
                entry: dict[str, Any] = {}
                key, _, rest = body.partition(":")
                if rest.strip():
                    entry[key.strip()] = _value(rest)
                    index += 1
                else:
                    index += 1
                    nested, index = _parse_block(lines, index, indent + 2)
                    entry[key.strip()] = nested
                while index < len(lines) and lines[index][0] > indent:
                    sub_key, _, sub_rest = lines[index][1].partition(":")
                    if sub_rest.strip():
                        entry[sub_key.strip()] = _value(sub_rest)
                        index += 1
                    else:
                        child_indent = lines[index][0]
                        index += 1
                        nested, index = _parse_block(lines, index, child_indent + 2)
                        entry[sub_key.strip()] = nested
                items.append(entry)
            else:
                items.append(_value(body))
                index += 1
        return items, index
    mapping: dict[str, Any] = {}
    while index < len(lines) and lines[index][0] == indent:
        depth, text = lines[index]
        key, _, rest = text.partition(":")
        key = key.strip()
        if rest.strip():
            mapping[key] = _value(rest)
            index += 1
            continue
        index += 1
        if index < len(lines) and lines[index][0] > indent:
            nested, index = _parse_block(lines, index, lines[index][0])
            mapping[key] = nested
        else:
            mapping[key] = None
    return mapping, index


def _looks_like_mapping(body: str) -> bool:
    head = body.split(":", 1)[0]
    return bool(re.fullmatch(r"[A-Za-z0-9_.\-]+", head.strip()))


def simple_yaml_load(text: str) -> Any:
    """Parse the YAML subset used by this package's configuration files."""
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        stripped = _strip_comment(raw)
        if not stripped.strip() or stripped.strip() == "---":
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append((indent, stripped.strip()))
    parsed, _ = _parse_block(lines, 0, lines[0][0] if lines else 0)
    return parsed or {}


def load_yaml(path: Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # noqa: PLC0415 - optional dependency
    except ImportError:
        return simple_yaml_load(text)
    return yaml.safe_load(text) or {}


# ---------------------------------------------------------------------------
# Configuration handling
# ---------------------------------------------------------------------------

def deep_merge(base: dict, override: dict) -> dict:
    """Recursive mapping merge; sequences and scalars are replaced wholesale."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_profile(config: dict, profile: str) -> dict:
    profiles = config.get("profiles") or {}
    if profile not in profiles:
        raise ConfigError(
            f"Unknown profile {profile!r}; available: {sorted(profiles)}")
    merged = deep_merge(config, profiles[profile])
    merged["profile"] = profile
    merged.pop("profiles", None)
    merged["available_profiles"] = sorted(profiles)
    return merged


def apply_overrides(config: dict, overrides: dict[str, Any]) -> dict:
    """Apply ``a.b.c=value`` style overrides (used for CLI ``--set``)."""
    result = copy.deepcopy(config)
    for dotted, value in overrides.items():
        if value is None:
            continue
        node = result
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"Cannot override inside scalar at {dotted!r}")
        node[parts[-1]] = value
    return result


def validate(config: dict) -> list[str]:
    """Return a list of human-readable configuration problems (empty = valid)."""
    problems: list[str] = []
    if int(config.get("batch_size", 1)) != 1:
        problems.append(
            "batch_size must be 1: the workstation baseline and every Jetson "
            "deployment figure in this study are single-image.")
    comparative = config.get("comparative_latency") or {}
    sustained = config.get("sustained_telemetry") or {}
    for name, section in (("comparative_latency", comparative),
                          ("sustained_telemetry", sustained)):
        if int(section.get("repeats", 0)) < 1:
            problems.append(f"{name}.repeats must be >= 1")
        if int(section.get("warmup", -1)) < 0:
            problems.append(f"{name}.warmup must be >= 0")
    if float(sustained.get("minimum_duration_seconds", 0)) <= 0:
        problems.append("sustained_telemetry.minimum_duration_seconds must be > 0")
    if float(sustained.get("maximum_duration_seconds", 0)) < \
            float(sustained.get("minimum_duration_seconds", 0)):
        problems.append("sustained_telemetry.maximum_duration_seconds must be >= minimum")
    cooldown = config.get("cooldown") or {}
    if float(cooldown.get("max_seconds", 0)) < float(cooldown.get("min_seconds", 0)):
        problems.append("cooldown.max_seconds must be >= cooldown.min_seconds")
    interval = int((config.get("telemetry") or {}).get("tegrastats_interval_ms", 0))
    if not 10 <= interval <= 5000:
        problems.append("telemetry.tegrastats_interval_ms must be between 10 and 5000")
    zero_shot_mode = ((config.get("zero_shot") or {}).get("mode") or "auto").lower()
    if zero_shot_mode not in ("auto", "off", "required"):
        problems.append("zero_shot.mode must be one of auto|off|required")
    for boundary in (config.get("timing") or {}).get("boundaries", []):
        if boundary not in TIMING_BOUNDARIES:
            problems.append(f"unknown timing boundary {boundary!r}")
    strategy = ((config.get("power") or {}).get("rail_strategy") or "auto").lower()
    if strategy not in ("auto", "total_rail_only", "explicit"):
        problems.append("power.rail_strategy must be auto|total_rail_only|explicit")
    return problems


def load_config(path: Path, profile: str | None = None,
                overrides: dict[str, Any] | None = None) -> dict:
    config = load_yaml(path)
    config["config_path"] = str(path)
    if profile:
        config = apply_profile(config, profile)
    else:
        config.setdefault("profile", "dissertation")
        config.pop("profiles", None)
    if overrides:
        config = apply_overrides(config, overrides)
    problems = validate(config)
    if problems:
        raise ConfigError("Invalid benchmark configuration:\n  - " + "\n  - ".join(problems))
    return config
