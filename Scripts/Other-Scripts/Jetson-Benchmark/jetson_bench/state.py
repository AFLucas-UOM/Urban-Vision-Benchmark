"""Incremental run state, atomic writes and resumability.

State is written after every model and every repeat, so a benchmark that loses
power, is interrupted with Ctrl+C, or is killed by the OOM reaper leaves a
usable partial run. ``--resume <run-directory>`` then skips exactly the
model/pass/repeat combinations that already completed.

All manifests are written atomically (write to a sibling temporary file, then
``os.replace``), so a partially written CSV can never be mistaken for a
complete one.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import tempfile
from pathlib import Path

STATE_FILENAME = "run_state.json"
STATE_VERSION = 1


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding=encoding, newline="", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, default=str) + "\n")


def atomic_write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None,
                     compress: bool = False) -> None:
    """Write a CSV (optionally gzip-compressed) atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if compress:
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                handle.write("")
        else:
            atomic_write_text(path, "")
        return
    fields = fieldnames or list(dict.fromkeys(k for row in rows for k in row))
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    if not compress:
        atomic_write_text(path, buffer.getvalue())
        return
    temporary = path.parent / f".{path.name}.tmp"
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
            handle.write(buffer.getvalue())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def combination_key(model_id: str, pass_name: str, repeat: int) -> str:
    return f"{model_id}|{pass_name}|{repeat}"


class RunState:
    """Durable, resumable record of what a run has already completed."""

    def __init__(self, run_dir: Path, run_id: str, config_digest: str = ""):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / STATE_FILENAME
        self.data = {
            "state_version": STATE_VERSION,
            "run_id": run_id,
            "config_digest": config_digest,
            "completed": {},
            "model_status": {},
            "events": [],
        }

    # -- persistence ---------------------------------------------------------

    @classmethod
    def load(cls, run_dir: Path) -> "RunState":
        path = Path(run_dir) / STATE_FILENAME
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} does not exist; --resume needs a run directory produced by "
                "a previous Jetson benchmark")
        payload = json.loads(path.read_text(encoding="utf-8"))
        state = cls(run_dir, payload.get("run_id", Path(run_dir).name),
                    payload.get("config_digest", ""))
        state.data.update(payload)
        state.data.setdefault("completed", {})
        state.data.setdefault("model_status", {})
        state.data.setdefault("events", [])
        return state

    def save(self) -> None:
        atomic_write_json(self.path, self.data)

    # -- queries -------------------------------------------------------------

    def is_complete(self, model_id: str, pass_name: str, repeat: int) -> bool:
        return combination_key(model_id, pass_name, repeat) in self.data["completed"]

    def completed_payload(self, model_id: str, pass_name: str, repeat: int):
        return self.data["completed"].get(combination_key(model_id, pass_name, repeat))

    def model_status(self, model_id: str) -> str | None:
        return self.data["model_status"].get(model_id)

    def model_is_terminal(self, model_id: str) -> bool:
        """A model whose recorded status means no further attempt is useful."""
        status = self.model_status(model_id)
        return status in {"ok", "skipped", "missing_checkpoint", "missing_dataset",
                          "ambiguous_checkpoint", "dependency_error"}

    # -- updates -------------------------------------------------------------

    def record_pass(self, model_id: str, pass_name: str, repeat: int, payload: dict) -> None:
        self.data["completed"][combination_key(model_id, pass_name, repeat)] = payload
        self.save()

    def record_model(self, model_id: str, status: str, detail: dict | None = None) -> None:
        self.data["model_status"][model_id] = status
        if detail:
            self.data.setdefault("model_detail", {})[model_id] = detail
        self.save()

    def record_event(self, kind: str, message: str, **fields) -> None:
        import datetime

        self.data["events"].append({
            "at": datetime.datetime.now().isoformat(timespec="seconds"),
            "kind": kind, "message": message, **fields})
        self.save()
