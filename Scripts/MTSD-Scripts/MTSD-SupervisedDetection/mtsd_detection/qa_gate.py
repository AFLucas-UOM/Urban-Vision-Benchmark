from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_qa_gate(path: Path, repo_root: Path) -> dict[str, Any]:
    if not path.is_absolute():
        path = repo_root / path
    if not path.is_file():
        raise FileNotFoundError(f"QA gate file is missing: {path}")
    gate = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = {"gate_version", "audit_path", "audit_timestamp", "finding_counts",
                "resolution_status", "approved_scope", "audited_groups"}
    missing = sorted(required - set(gate))
    if missing:
        raise ValueError(f"QA gate {path} is missing fields: {missing}")
    gate["gate_file"] = str(path)
    gate["resolved"] = gate["resolution_status"] == "resolved"
    return gate


def enforce_qa_gate(gate: dict[str, Any], *, final: bool, acknowledged_override: bool) -> None:
    if gate.get("resolved"):
        return
    if final:
        raise PermissionError(
            f"Final operation refused: QA gate is {gate.get('resolution_status')} "
            f"({gate.get('audit_path')}). Resolve and approve the audit first."
        )
    if not acknowledged_override:
        raise PermissionError(
            "QA gate is unresolved. Development preparation/training requires "
            "--acknowledge-open-qa-gate; final operations cannot bypass it."
        )
