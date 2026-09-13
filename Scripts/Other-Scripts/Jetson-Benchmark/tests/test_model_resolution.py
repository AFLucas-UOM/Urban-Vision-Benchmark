"""Exact run selection, accuracy provenance and comparability decisions.

These tests assert the *rules*, using synthetic evidence where the repository's
private artefacts may be absent, plus a small number of repository-evidence
tests that are skipped when the corresponding tables are not checked out.
"""

from __future__ import annotations

import pytest

from jetson_bench import models as models_module
from jetson_bench.config import load_config
from jetson_bench.models import (
    RFDETR_DEFAULT_RESOLUTION,
    BenchmarkSpec,
    annotate_workstation_comparability,
    is_excluded_run,
    resolve_attribute,
    resolve_matrix,
    resolve_mtsd,
    run_imgsz,
)
from jetson_bench.paths import MTSD_EXPERIMENT_MATRIX


# ---------------------------------------------------------------------------
# Run exclusion rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "[OLD] YOLO26-Sweep",
    "PROBE_yolo12m_mtsd_strong_img640_pb32_eb64_e2_adamw_s42",
    "dinov3_vitb_lora-smoke",
    "yolo26s-smoke-run",
])
def test_obsolete_probe_and_smoke_runs_are_excluded(name):
    assert is_excluded_run(name) is True


@pytest.mark.parametrize("name", [
    "FINAL_rfdetrm_mtsd_strong_img576_pb32_eb32_e100_adamw_s42",
    "strongaug-wandb-cuda-yolo26s-img1280-s42",
    "E002_yolo26s_rfv20_img640_b32_e100_adamw_lr0p001",
])
def test_retained_runs_are_not_excluded(name):
    assert is_excluded_run(name) is False


# ---------------------------------------------------------------------------
# Input size comes from run metadata, never a hardcoded assumption
# ---------------------------------------------------------------------------

def test_input_size_is_read_from_the_run_args(tmp_path):
    run = tmp_path / "E002_yolo26s_rfv20_img640_b32_e100_adamw_lr0p001"
    run.mkdir()
    (run / "args.yaml").write_text("model: yolo26s.pt\nimgsz: 640\nbatch: 32\n",
                                   encoding="utf-8")
    assert run_imgsz(run) == 640


def test_input_size_is_read_from_a_run_record_when_args_are_absent(tmp_path):
    run = tmp_path / "FINAL_rfdetrm_mtsd_strong_img576"
    run.mkdir()
    (run / "run_record.json").write_text(
        '{"config": {"model_args": {"resolution": 576}}}', encoding="utf-8")
    assert run_imgsz(run) == 576


def test_missing_metadata_yields_none_not_a_guess(tmp_path):
    run = tmp_path / "mystery_run"
    run.mkdir()
    assert run_imgsz(run) is None


# ---------------------------------------------------------------------------
# MTSD tie-breaking
# ---------------------------------------------------------------------------

def _matrix_row(run_path: str, architecture: str, scale: str, regime: str,
                size: str, test_metric: str = "0.7", status: str = "completed") -> dict:
    return {"run_path": run_path, "architecture": architecture, "model_scale": scale,
            "augmentation_regime": regime, "image_size": size,
            "test_map50_95": test_metric, "val_map50_95": test_metric,
            "completion_status": status, "evaluation_split": "test"}


def test_partial_runs_are_reported_missing_not_substituted(monkeypatch):
    """A configuration that only ever produced partial runs must be reported.

    This is the YOLO12-S / strong / 1280 situation in this repository: weights
    exist on disk but no completed 1280 result does. Silently dropping to the
    640 or 960 checkpoint would attach the wrong accuracy to the benchmark.
    """
    rows = [
        _matrix_row(r"Results\MTSD-Runs\YOLO12-MTSD\FINAL_yolo12s_mtsd_strong_img1280",
                    "YOLO12", "s", "Unknown", "1280", "", "partial"),
        _matrix_row(r"Results\MTSD-Runs\YOLO12-MTSD\FINAL_yolo12s_mtsd_strong_img640",
                    "YOLO12", "s", "Strong", "640", "0.6556"),
    ]
    monkeypatch.setattr(models_module, "mtsd_experiment_rows", lambda: rows)
    monkeypatch.setattr(models_module, "_mtsd_disk_candidates", lambda *a, **k: [])

    spec = resolve_mtsd({"family": "yolo12", "scale": "s",
                         "augmentation": "strong", "input_size": 1280})
    assert spec.status == "missing_checkpoint"
    assert spec.checkpoint is None
    assert "partial" in spec.reason
    assert "640" in spec.reason          # explicitly says it did not substitute


def test_final_named_run_wins_over_a_resolution_followup(monkeypatch, tmp_path):
    final = tmp_path / "FINAL_rfdetrm_mtsd_strong_img576_pb32_eb32_e100_adamw_s42"
    followup = tmp_path / "strongaug-wandb-followup-rfdetr-m-img704-s42"
    for run in (final, followup):
        run.mkdir()
        (run / "checkpoint_best_total.pth").write_bytes(b"x" * 200_000)
    rows = [_matrix_row(str(final), "RF-DETR", "m", "Strong", "576", "0.7114"),
            _matrix_row(str(followup), "RF-DETR", "m", "Strong", "704", "0.7148")]
    monkeypatch.setattr(models_module, "mtsd_experiment_rows", lambda: rows)
    monkeypatch.setattr(
        models_module, "resolve_recorded_path",
        lambda recorded: tmp_path / str(recorded).replace("\\", "/").split("/")[-1])

    spec = resolve_mtsd({"family": "rfdetr", "scale": "m",
                         "augmentation": "strong", "input_size": "trained"})
    assert spec.status == "preflight_pending"
    assert "FINAL_rfdetrm" in spec.run_directory
    assert spec.input_resolution == 576
    assert any("FINAL_*" in note for note in spec.resolution_evidence)


def test_a_genuine_tie_is_reported_ambiguous_rather_than_guessed(monkeypatch, tmp_path):
    first = tmp_path / "strongaug-alpha-yolo26s-img1280-s42"
    second = tmp_path / "strongaug-beta-yolo26s-img1280-s42"
    for run in (first, second):
        (run / "weights").mkdir(parents=True)
        (run / "weights" / "best.pt").write_bytes(b"x" * 200_000)
    rows = [_matrix_row(str(first), "YOLO26", "s", "Strong", "1280", "0.74"),
            _matrix_row(str(second), "YOLO26", "s", "Strong", "1280", "0.75")]
    monkeypatch.setattr(models_module, "mtsd_experiment_rows", lambda: rows)
    monkeypatch.setattr(
        models_module, "resolve_recorded_path",
        lambda recorded: tmp_path / str(recorded).replace("\\", "/").split("/")[-1])
    monkeypatch.setattr(models_module, "workstation_rows", lambda: [])

    spec = resolve_mtsd({"family": "yolo26", "scale": "s",
                         "augmentation": "strong", "input_size": 1280})
    assert spec.status == "ambiguous_checkpoint"
    assert spec.checkpoint is None
    assert "refusing to guess" in spec.reason


def test_the_higher_accuracy_run_is_not_automatically_chosen(monkeypatch, tmp_path):
    """Selection is by documented provenance, not by whichever scored best."""
    final = tmp_path / "FINAL_rfdetrs_mtsd_strong_img512_pb32_eb32_e100_adamw_s42"
    followup = tmp_path / "strongaug-wandb-followup-rfdetr-s-img640-s42"
    for run in (final, followup):
        run.mkdir()
        (run / "checkpoint_best_total.pth").write_bytes(b"x" * 200_000)
    rows = [_matrix_row(str(final), "RF-DETR", "s", "Strong", "512", "0.6889"),
            _matrix_row(str(followup), "RF-DETR", "s", "Strong", "640", "0.9999")]
    monkeypatch.setattr(models_module, "mtsd_experiment_rows", lambda: rows)
    monkeypatch.setattr(
        models_module, "resolve_recorded_path",
        lambda recorded: tmp_path / str(recorded).replace("\\", "/").split("/")[-1])

    spec = resolve_mtsd({"family": "rfdetr", "scale": "s",
                         "augmentation": "strong", "input_size": "trained"})
    assert "FINAL_rfdetrs" in spec.run_directory
    assert spec.predictive_metric_value == pytest.approx(0.6889)


def test_no_run_at_all_is_missing_checkpoint(monkeypatch):
    monkeypatch.setattr(models_module, "mtsd_experiment_rows", lambda: [])
    monkeypatch.setattr(models_module, "_mtsd_disk_candidates", lambda *a, **k: [])
    spec = resolve_mtsd({"family": "yolo26", "scale": "x",
                         "augmentation": "strong", "input_size": 1280})
    assert spec.status == "missing_checkpoint"


# ---------------------------------------------------------------------------
# Attribute variants
# ---------------------------------------------------------------------------

def test_missing_attribute_checkpoint_is_never_retrained(monkeypatch, tmp_path):
    monkeypatch.setattr(models_module, "ATTR_CHECKPOINTS", tmp_path)
    spec = resolve_attribute("dinov3_vitl_lora")
    assert spec.status == "missing_checkpoint"
    assert "never retrains" in spec.reason


def test_attribute_spec_records_lora_adaptation(monkeypatch, tmp_path):
    variant = tmp_path / "vjepa21_vitb_lora"
    variant.mkdir()
    (variant / "best.pt").write_bytes(b"x" * 2_000_000)
    monkeypatch.setattr(models_module, "ATTR_CHECKPOINTS", tmp_path)
    spec = resolve_attribute("vjepa21_vitb_lora")
    assert spec.adaptation == "lora"
    assert spec.scale == "base"
    assert spec.engine == "attribute"
    assert spec.status == "preflight_pending"


# ---------------------------------------------------------------------------
# Workstation comparability
# ---------------------------------------------------------------------------

def _spec(**kwargs) -> BenchmarkSpec:
    defaults = dict(model_id="m", display_name="M", track="mtsd", dataset="MTSD",
                    task="detection", family="yolo26", scale="s", engine="yolo",
                    status="preflight_pending", input_resolution=1280,
                    run_directory="/repo/Results/MTSD-Runs/YOLO26-MTSD/run-a")
    defaults.update(kwargs)
    return BenchmarkSpec(**defaults)


def test_a_run_the_workstation_never_measured_is_not_comparable(monkeypatch):
    monkeypatch.setattr(models_module, "workstation_rows", lambda: [])
    spec = annotate_workstation_comparability(_spec())
    assert spec.workstation_comparable is False
    assert "not_protocol_comparable" in spec.workstation_reason


def test_yolo_at_a_different_input_size_is_not_comparable(monkeypatch):
    monkeypatch.setattr(models_module, "workstation_rows", lambda: [
        {"run_identifier": "run-a", "dataset": "MTSD", "input_size": "640",
         "mean_latency_ms": "167.8"}])
    spec = annotate_workstation_comparability(_spec(input_resolution=1280))
    assert spec.workstation_comparable is False
    assert "imgsz 640" in spec.workstation_reason


def test_yolo_at_the_same_input_size_is_comparable(monkeypatch):
    monkeypatch.setattr(models_module, "workstation_rows", lambda: [
        {"run_identifier": "run-a", "dataset": "MDWD", "input_size": "640",
         "mean_latency_ms": "14.55"}])
    spec = annotate_workstation_comparability(
        _spec(dataset="MDWD", track="mdwd", input_resolution=640))
    assert spec.workstation_comparable is True
    assert spec.extra["workstation_row"]["mean_latency_ms"] == "14.55"


def test_rfdetr_is_comparable_only_at_the_package_default_resolution(monkeypatch):
    monkeypatch.setattr(models_module, "workstation_rows", lambda: [
        {"run_identifier": "run-a", "dataset": "MTSD", "input_size": "640",
         "mean_latency_ms": "90.78"}])
    at_default = annotate_workstation_comparability(
        _spec(family="rfdetr", engine="rfdetr", scale="m",
              input_resolution=RFDETR_DEFAULT_RESOLUTION["m"]))
    assert at_default.workstation_comparable is True

    off_default = annotate_workstation_comparability(
        _spec(family="rfdetr", engine="rfdetr", scale="m", input_resolution=704))
    assert off_default.workstation_comparable is False
    assert "package default 576" in off_default.workstation_reason


def test_unresolved_checkpoints_are_never_compared():
    spec = annotate_workstation_comparability(_spec(status="missing_checkpoint"))
    assert spec.workstation_comparable is False
    assert spec.workstation_reason == "no_resolved_checkpoint"


# ---------------------------------------------------------------------------
# Repository-evidence checks (skipped when the tables are not checked out)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MTSD_EXPERIMENT_MATRIX.is_file(),
                    reason="MTSD experiment matrix not present in this checkout")
def test_repository_matrix_resolves_every_requested_configuration(default_config_path):
    config = load_config(default_config_path)
    specs = resolve_matrix(config, ["mdwd", "mtsd", "attributes"])
    assert len(specs) == 16
    for spec in specs:
        assert spec.status in ("preflight_pending", "missing_checkpoint",
                               "ambiguous_checkpoint")
        if spec.status == "preflight_pending":
            assert spec.checkpoint, spec.model_id
            assert spec.input_resolution, spec.model_id
        else:
            assert spec.reason, spec.model_id


@pytest.mark.skipif(not MTSD_EXPERIMENT_MATRIX.is_file(),
                    reason="MTSD experiment matrix not present in this checkout")
def test_resolved_mtsd_detectors_use_their_trained_resolution(default_config_path):
    config = load_config(default_config_path)
    for spec in resolve_matrix(config, ["mtsd"]):
        if spec.status != "preflight_pending":
            continue
        if spec.family == "rfdetr":
            # Never forced to 1280; uses the architecture's trained resolution.
            assert spec.input_resolution != 1280
        else:
            assert spec.input_resolution == 1280
