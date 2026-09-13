"""Atomic writes, incremental state and resume handling."""

from __future__ import annotations

import csv
import gzip
import json

import pytest

from jetson_bench.state import (
    RunState,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    combination_key,
)


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------

def test_atomic_text_write_leaves_no_temporary_file(tmp_path):
    target = tmp_path / "nested" / "manifest.csv"
    atomic_write_text(target, "a,b\n1,2\n")
    assert target.read_text(encoding="utf-8") == "a,b\n1,2\n"
    assert [p.name for p in tmp_path.rglob("*") if p.is_file()] == ["manifest.csv"]


def test_a_failed_write_does_not_corrupt_the_previous_file(tmp_path, monkeypatch):
    target = tmp_path / "manifest.json"
    atomic_write_json(target, {"good": True})

    import os
    def explode(*_args, **_kwargs):
        raise OSError("simulated disk failure")
    monkeypatch.setattr(os, "replace", explode)

    with pytest.raises(OSError):
        atomic_write_json(target, {"good": False})
    assert json.loads(target.read_text(encoding="utf-8")) == {"good": True}
    assert not list(tmp_path.glob(".*tmp"))


def test_csv_written_with_the_union_of_all_keys(tmp_path):
    target = tmp_path / "rows.csv"
    atomic_write_csv(target, [{"a": 1}, {"b": 2}])
    rows = list(csv.DictReader(target.open(newline="", encoding="utf-8")))
    assert rows == [{"a": "1", "b": ""}, {"a": "", "b": "2"}]


def test_empty_rows_produce_an_empty_file(tmp_path):
    target = tmp_path / "rows.csv"
    atomic_write_csv(target, [])
    assert target.read_text(encoding="utf-8") == ""


def test_raw_timings_can_be_gzipped(tmp_path):
    target = tmp_path / "raw_timings.csv.gz"
    atomic_write_csv(target, [{"model": "m", "latency_ms": 1.5}], compress=True)
    with gzip.open(target, "rt", encoding="utf-8") as handle:
        assert "latency_ms" in handle.read()


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------

def test_new_state_starts_empty_and_persists(tmp_path):
    state = RunState(tmp_path, "20260911-101112", "digest")
    state.save()
    assert (tmp_path / "run_state.json").is_file()
    assert state.is_complete("m", "comparative_latency", 0) is False


def test_completed_combinations_are_remembered_across_a_reload(tmp_path):
    state = RunState(tmp_path, "run-1", "digest")
    state.record_pass("m", "comparative_latency", 0, {"model_id": "m", "summary": {}})
    state.record_model("m", "ok")

    reloaded = RunState.load(tmp_path)
    assert reloaded.is_complete("m", "comparative_latency", 0) is True
    assert reloaded.is_complete("m", "comparative_latency", 1) is False
    assert reloaded.model_status("m") == "ok"


def test_resume_requires_a_real_run_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="--resume"):
        RunState.load(tmp_path / "never-existed")


def test_terminal_statuses_are_not_retried(tmp_path):
    state = RunState(tmp_path, "run-1")
    for status in ("ok", "missing_checkpoint", "missing_dataset", "skipped",
                   "dependency_error", "ambiguous_checkpoint"):
        state.record_model(f"m-{status}", status)
        assert state.model_is_terminal(f"m-{status}") is True


def test_transient_failures_remain_retryable(tmp_path):
    state = RunState(tmp_path, "run-1")
    for status in ("cuda_oom", "runtime_error", "thermal_abort", "load_error"):
        state.record_model(f"m-{status}", status)
        assert state.model_is_terminal(f"m-{status}") is False


def test_events_are_appended_with_a_timestamp(tmp_path):
    state = RunState(tmp_path, "run-1")
    state.record_event("interrupt", "operator interrupt")
    reloaded = RunState.load(tmp_path)
    assert reloaded.data["events"][0]["kind"] == "interrupt"
    assert reloaded.data["events"][0]["at"]


def test_state_survives_a_partial_run_and_keeps_completed_payloads(tmp_path):
    state = RunState(tmp_path, "run-1")
    payload = {"model_id": "m", "summary": {"latency_mean_ms": 123.4}}
    state.record_pass("m", "sustained_telemetry", 2, payload)
    reloaded = RunState.load(tmp_path)
    recovered = reloaded.completed_payload("m", "sustained_telemetry", 2)
    assert recovered["summary"]["latency_mean_ms"] == 123.4


def test_combination_key_separates_models_passes_and_repeats():
    keys = {combination_key("a", "p", 0), combination_key("a", "p", 1),
            combination_key("a", "q", 0), combination_key("b", "p", 0)}
    assert len(keys) == 4
