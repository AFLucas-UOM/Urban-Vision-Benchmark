from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable

from .utils import fingerprint, sha256_file


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or list(rows[0]) if rows else (fieldnames or [])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_manifest(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("manifest_fingerprint", None)
    result["manifest_fingerprint"] = fingerprint(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)
    return result


__all__ = ["fingerprint", "sha256_file", "write_csv", "write_manifest"]
