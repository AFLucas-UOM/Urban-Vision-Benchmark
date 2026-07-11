from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "_", value.lower()).strip("_")


def combination_dir(run_dir: Path, model: str, prompt_id: str) -> Path:
    return run_dir / "combinations" / f"{slug(model)}__{prompt_id}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields: path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def compatible_complete(path: Path, expected: dict[str, Any]) -> bool:
    status_path = path / "status.json"
    if not status_path.is_file(): return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    keys = ("protocol_hash", "dataset_manifest_hash", "model", "prompt_id", "conf_threshold", "iou_threshold", "split")
    differences = {key: (status.get(key), expected.get(key)) for key in keys if status.get(key) != expected.get(key)}
    if differences: raise ValueError(f"Incompatible completed combination {path}: {differences}")
    return status.get("status") == "completed"

