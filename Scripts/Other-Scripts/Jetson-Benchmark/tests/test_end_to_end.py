"""Full orchestration on a non-Jetson machine, with a stubbed model worker.

The worker subprocess is replaced by a deterministic stub, so the whole
pipeline - preflight, smoke, Pass A, Pass B, telemetry alignment, state,
consolidation, comparison, Pareto, figures and report - is exercised without a
Jetson, without CUDA and without loading a single model.

One model in the stub matrix fails on purpose, which is how failure isolation
and the ``failures.csv`` path get covered.
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import pytest

import run_jetson_benchmark as orchestrator
from jetson_bench import models as models_module
from jetson_bench import preflight as preflight_module
from jetson_bench import runner as runner_module
from jetson_bench.models import BenchmarkSpec
from jetson_bench.preflight import DatasetReport, EnvironmentReport


# ---------------------------------------------------------------------------
# Fixtures: a synthetic repository slice
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_matrix(tmp_path, monkeypatch):
    """Three detection configurations and one attribute configuration."""
    images = tmp_path / "images"
    images.mkdir()
    for index in range(6):
        (images / f"img{index:03d}.jpg").write_bytes(b"\xff\xd8\xff" + b"0" * 512)

    checkpoints = tmp_path / "ckpt"
    checkpoints.mkdir()

    def checkpoint(name: str) -> str:
        path = checkpoints / name
        path.write_bytes(b"\x80\x02" + b"y" * 3_000_000)
        return str(path)

    specs = [
        BenchmarkSpec(model_id="mdwd_yolo26_s", display_name="MDWD YOLO26-S",
                      track="mdwd", dataset="MDWD", task="detection",
                      family="yolo26", scale="s", engine="yolo",
                      input_resolution=640, checkpoint=checkpoint("yolo26s.pt"),
                      run_directory=str(tmp_path / "run-a"), suite="YOLO26-EUVIP",
                      status="preflight_pending",
                      predictive_metric_name="test_mAP50_95",
                      predictive_metric_value=0.7527,
                      predictive_metric_source="Documents/.../mdwd.csv",
                      workstation_comparable=True,
                      workstation_reason="same checkpoint, sample and boundary",
                      extra={"workstation_row": {
                          "mean_latency_ms": "14.55", "p95_latency_ms": "15.58",
                          "fps": "68.7", "peak_gpu_mem_gb": "0.31",
                          "source_run": "20260910-163921"}}),
        BenchmarkSpec(model_id="mdwd_yolo26_l", display_name="MDWD YOLO26-L",
                      track="mdwd", dataset="MDWD", task="detection",
                      family="yolo26", scale="l", engine="yolo",
                      input_resolution=640, checkpoint=checkpoint("yolo26l.pt"),
                      run_directory=str(tmp_path / "run-b"), suite="YOLO26-EUVIP",
                      status="preflight_pending",
                      predictive_metric_name="test_mAP50_95",
                      predictive_metric_value=0.7824,
                      predictive_metric_source="Documents/.../mdwd.csv",
                      workstation_comparable=False,
                      workstation_reason="not_protocol_comparable: imgsz differs"),
        BenchmarkSpec(model_id="mdwd_rfdetr_m", display_name="MDWD RF-DETR-M",
                      track="mdwd", dataset="MDWD", task="detection",
                      family="rfdetr", scale="m", engine="rfdetr",
                      status="missing_checkpoint",
                      reason="not trained for MDWD in this repository"),
        BenchmarkSpec(model_id="attr_dinov3_vitl_lora",
                      display_name="DINOv3 ViT-L/16 + LoRA",
                      track="attributes", dataset="MTSD", task="attribute",
                      family="dinov3", scale="large", engine="attribute",
                      adaptation="lora", input_resolution=224,
                      checkpoint=checkpoint("attr_best.pt"),
                      run_directory=str(tmp_path / "run-c"),
                      status="preflight_pending",
                      predictive_metric_name="test_mean_macro_f1",
                      predictive_metric_value=0.8959,
                      predictive_metric_source="Scripts/.../size_ablation.csv",
                      workstation_comparable=True,
                      workstation_reason="same checkpoint, crops and boundary",
                      extra={"workstation_row": {
                          "mean_latency_ms": "2.48", "p95_latency_ms": "2.9",
                          "fps": "403.2", "peak_gpu_mem_gb": "1.2",
                          "source_run": "20260910-164007"}}),
    ]

    monkeypatch.setattr(orchestrator, "resolve_matrix",
                        lambda config, tracks: list(specs))
    monkeypatch.setattr(orchestrator, "annotate_workstation_comparability",
                        lambda spec: spec)
    monkeypatch.setattr(orchestrator, "dataset_reports", lambda: {
        "MDWD/detection": DatasetReport("MDWD/detection", "Datasets/MDWD", True, 6),
        "MTSD/attribute": DatasetReport("MTSD/attribute", "Scripts/.../crops", True, 6),
    })

    def environments(specs_, use_venvs=True, timeout=180.0):
        reports = {}
        for name in ("detection", "attribute"):
            report = EnvironmentReport(name, f".venv-jetson-{name}", True,
                                       sys.executable)
            report.modules = {"torch": True}
            report.cuda_available = True
            report.torch_version = "2.5.0-stub"
            reports[name] = report
        return reports

    monkeypatch.setattr(orchestrator, "environment_reports", environments)
    monkeypatch.setattr(orchestrator, "freeze_environments",
                        lambda *a, **k: None)

    def fake_sample(task, dataset, **kwargs):
        from jetson_bench.samples import SampleSet
        sample = SampleSet(task=task, dataset=dataset,
                           items=sorted(images.glob("*.jpg")),
                           source="workstation_manifest",
                           workstation_run="20260910-163921",
                           workstation_manifest="Results/.../benchmark_images_used.csv",
                           seed=42, requested=6)
        return sample

    monkeypatch.setattr(orchestrator, "resolve_sample", fake_sample)
    monkeypatch.setattr(orchestrator, "resolve_prompt",
                        lambda config: {"prompt": "traffic sign",
                                        "prompt_id": "mtsd-p12",
                                        "protocol_version": "dissertation-v2",
                                        "conf_threshold": 0.3})
    return specs


@pytest.fixture()
def stub_worker(monkeypatch):
    """Deterministic worker results; ``mdwd_yolo26_l`` fails on purpose."""
    latency = {"mdwd_yolo26_s": 145.0, "mdwd_yolo26_l": 0.0,
               "attr_dinov3_vitl_lora": 31.0}

    def fake_run_worker(spec, job, run_dir, tag, timeout, use_venvs,
                        allow_downloads, log_dir):
        import time

        if spec.model_id == "mdwd_yolo26_l":
            return {"model_id": spec.model_id, "status": "cuda_oom",
                    "error": "CUDA out of memory (simulated)",
                    "log": "logs/stub.log"}
        base = {
            "model_id": spec.model_id, "engine": spec.engine, "status": "ok",
            "cold_start_s": 3.2, "parameters_total": 9_951_734,
            "parameters_trainable": 799_757, "notes": ["stub worker"],
            "extra": {"adapter_checkpoint_mb": 3.08},
            "memory_after_load": {"system_ram_model_delta_mb": 620.0,
                                  "torch_peak_allocated_mb": 410.0,
                                  "torch_peak_reserved_mb": 512.0},
            "comparable_boundary": "end_to_end_file" if spec.engine == "yolo"
                                   else "model_forward",
            "passes": [],
        }
        if tag == "smoke":
            base["status"] = "smoke_ok"
            base["smoke"] = {"status": "smoke_ok", "seconds": 0.4,
                             "detail": {"detections": 3}}
            return base

        mean = latency[spec.model_id]
        for pass_spec in job["passes"]:
            boundary = pass_spec["boundary"]
            if boundary == "comparable":
                boundary = base["comparable_boundary"]
            count = len(job["spec"]["items"])
            now = time.monotonic()
            base["passes"].append({
                "status": "ok", "pass_name": pass_spec["pass_name"],
                "repeat": pass_spec.get("repeat", 0), "timing_boundary": boundary,
                "is_comparable_boundary": boundary == base["comparable_boundary"],
                "latencies_ms": [mean + index * 0.5 for index in range(count)],
                "sample_indices": list(range(count)),
                "timed_items": count,
                "timed_seconds": round(count * mean / 1000.0, 4),
                "throughput_items_s": round(1000.0 / mean, 3),
                "start_monotonic": now, "end_monotonic": now + 0.05,
                "warmup": pass_spec.get("warmup", 0),
                "torch_peak_allocated_mb": 410.0,
                "torch_peak_reserved_mb": 512.0,
            })
        base["smoke"] = {"status": "smoke_ok", "seconds": 0.4, "detail": {}}
        return base

    monkeypatch.setattr(orchestrator, "run_worker", fake_run_worker)


@pytest.fixture()
def run_directory(tmp_path, fake_matrix, stub_worker, default_config_path):
    output = tmp_path / "Results" / "Jetson-Benchmark"
    status = orchestrator.main([
        "--config", str(default_config_path), "--profile", "smoke",
        "--no-venvs", "--no-telemetry", "--device", "cpu",
        "--output-root", str(output),
        "--set", "outputs.figures=true",
        "--set", "cooldown.max_seconds=0",
    ])
    assert status == 0
    runs = sorted(p for p in output.iterdir() if p.is_dir())
    assert len(runs) == 1
    return runs[0]


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def test_every_required_artefact_is_written(run_directory):
    for name in ("benchmark_config.json", "hardware_snapshot.json",
                 "software_snapshot.json", "repository_snapshot.json",
                 "requested_coverage.csv", "model_manifest.csv",
                 "sample_manifest.csv", "latency_summary.csv",
                 "telemetry_summary.csv", "deployment_summary.csv",
                 "failures.csv", "raw_timings.csv.gz",
                 "jetson_vs_workstation.csv", "benchmark_report.md",
                 "run_state.json"):
        assert (run_directory / name).exists(), name
    for directory in ("logs", "telemetry", "figures", "environment_freeze"):
        assert (run_directory / directory).is_dir(), directory


def test_the_run_directory_is_new_and_timestamped(run_directory):
    assert run_directory.name.count("-") == 1
    stamp, time_part = run_directory.name.split("-")
    assert len(stamp) == 8 and len(time_part) == 6


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def test_coverage_lists_every_requested_configuration(run_directory):
    rows = _rows(run_directory / "requested_coverage.csv")
    assert {row["model_id"] for row in rows} == {
        "mdwd_yolo26_s", "mdwd_yolo26_l", "mdwd_rfdetr_m", "attr_dinov3_vitl_lora"}


def test_a_missing_checkpoint_is_reported_not_substituted(run_directory):
    rows = {row["model_id"]: row for row in _rows(run_directory / "requested_coverage.csv")}
    missing = rows["mdwd_rfdetr_m"]
    assert missing["status"] == "missing_checkpoint"
    assert missing["selected"] == "False"
    assert "not trained" in missing["reason"]


def test_one_model_failing_does_not_stop_the_others(run_directory):
    failures = _rows(run_directory / "failures.csv")
    statuses = {row["model_id"]: row["status"] for row in failures}
    assert statuses["mdwd_yolo26_l"] == "cuda_oom"
    assert statuses["mdwd_rfdetr_m"] == "missing_checkpoint"

    summary = _rows(run_directory / "latency_summary.csv")
    measured = {row["model_id"] for row in summary}
    assert "mdwd_yolo26_s" in measured
    assert "attr_dinov3_vitl_lora" in measured
    assert "mdwd_yolo26_l" not in measured


def test_latency_rows_carry_the_full_provenance(run_directory):
    row = _rows(run_directory / "latency_summary.csv")[0]
    for field in ("run_id", "timestamp", "git_commit", "git_dirty", "jetson_model",
                  "l4t_version", "jetpack_version", "cuda_version", "cudnn_version",
                  "tensorrt_version", "torch_version", "python_version",
                  "dataset", "task", "model_id", "model_family", "model_scale",
                  "adaptation", "augmentation", "input_resolution",
                  "checkpoint_relative_path", "checkpoint_sha256", "checkpoint_mb",
                  "parameters_total", "parameters_trainable", "runtime_backend",
                  "precision", "batch_size", "seed", "sample_manifest", "warmup",
                  "repeat", "timed_items", "timed_seconds", "cold_start_s",
                  "latency_mean_ms", "latency_median_ms", "latency_p95_ms",
                  "throughput_items_s", "torch_peak_allocated_mb",
                  "torch_peak_reserved_mb", "system_ram_peak_mb",
                  "system_ram_model_delta_mb", "nvpmodel_mode",
                  "jetson_clocks_enabled", "predictive_metric_name",
                  "predictive_metric_value", "predictive_metric_source",
                  "status", "notes", "timing_boundary", "item_unit"):
        assert field in row, field


def test_batch_size_and_seed_are_recorded_as_configured(run_directory):
    for row in _rows(run_directory / "latency_summary.csv"):
        assert row["batch_size"] == "1"
        assert row["seed"] == "42"


def test_timing_boundaries_are_kept_in_separate_rows(run_directory):
    rows = [r for r in _rows(run_directory / "latency_summary.csv")
            if r["model_id"] == "mdwd_yolo26_s" and r["repeat"] != "all"]
    boundaries = {row["timing_boundary"] for row in rows}
    assert {"model_forward", "in_memory_pipeline", "end_to_end_file"} & boundaries
    for row in rows:
        assert row["timing_boundary"], "every row states its boundary"


def test_raw_per_item_observations_are_retained(run_directory):
    with gzip.open(run_directory / "raw_timings.csv.gz", "rt", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for field in ("model", "repeat", "sample", "sample_index", "latency_ms",
                  "timing_boundary", "timestamp"):
        assert field in rows[0], field
    # Individual observations, not just averages.
    assert len({row["latency_ms"] for row in rows}) > 1


def test_sample_manifest_records_identity_not_imagery(run_directory):
    rows = _rows(run_directory / "sample_manifest.csv")
    assert rows
    assert "relative_path" in rows[0]
    assert not any(key.lower() in ("image", "thumbnail", "data") for key in rows[0])


def test_comparison_marks_the_incomparable_configuration(run_directory):
    rows = {row["model"]: row for row in _rows(run_directory / "jetson_vs_workstation.csv")}
    comparable = rows["mdwd_yolo26_s"]
    assert comparable["comparison_valid"] == "True"
    assert float(comparable["slowdown_x"]) > 1
    assert comparable["memory_ratio"] == ""


def test_report_is_written_and_states_its_limitations(run_directory):
    report = (run_directory / "benchmark_report.md").read_text(encoding="utf-8")
    for heading in ("# NVIDIA Jetson Edge-Device Benchmark",
                    "## Experimental Platform", "## Protocol",
                    "## Requested Model Coverage",
                    "## Supervised Detection — MDWD",
                    "## Attribute Classification",
                    "## Memory Requirements", "## Thermal Behaviour",
                    "## Workstation vs Jetson",
                    "## Pareto-Efficient Configurations",
                    "## Models That Could Not Be Executed",
                    "## Deployment Interpretation",
                    "## Measurement Limitations",
                    "## Reproducibility Manifest"):
        assert heading in report, heading
    assert "only one Jetson" in report.lower() or "one Jetson device was tested" in report
    assert "ground-truth crops" in report
    assert "image-prompt pair" in report


def test_report_names_the_configurations_that_could_not_run(run_directory):
    report = (run_directory / "benchmark_report.md").read_text(encoding="utf-8")
    assert "MDWD RF-DETR-M" in report
    assert "cuda_oom" in report


def test_non_jetson_execution_is_declared_in_the_report(run_directory):
    report = (run_directory / "benchmark_report.md").read_text(encoding="utf-8")
    assert "was not executed on a Jetson" in report


def test_state_allows_a_resume(run_directory):
    from jetson_bench.state import RunState

    state = RunState.load(run_directory)
    assert state.model_status("mdwd_yolo26_s") == "ok"
    assert state.model_status("mdwd_yolo26_l") == "cuda_oom"
    assert state.is_complete("mdwd_yolo26_s", "comparative_latency", 0)


def test_resume_does_not_rerun_completed_models(run_directory, default_config_path,
                                                monkeypatch):
    calls = []
    original = orchestrator.run_worker

    def counting_worker(spec, *args, **kwargs):
        calls.append(spec.model_id)
        return original(spec, *args, **kwargs)

    monkeypatch.setattr(orchestrator, "run_worker", counting_worker)
    status = orchestrator.main([
        "--config", str(default_config_path), "--profile", "smoke",
        "--no-venvs", "--no-telemetry", "--device", "cpu",
        "--resume", str(run_directory),
        "--set", "cooldown.max_seconds=0",
    ])
    assert status == 0
    assert "mdwd_yolo26_s" not in calls, "a completed model was re-benchmarked"


def test_no_dataset_or_checkpoint_was_modified(run_directory, fake_matrix):
    for spec in fake_matrix:
        if not spec.checkpoint:
            continue
        assert Path(spec.checkpoint).stat().st_size == 3_000_002


def test_dry_run_creates_no_run_directory(tmp_path, fake_matrix, default_config_path):
    output = tmp_path / "dry-run-output"
    status = orchestrator.main([
        "--config", str(default_config_path), "--profile", "smoke",
        "--no-venvs", "--no-telemetry", "--device", "cpu", "--dry-run",
        "--output-root", str(output)])
    assert status == 0
    assert not output.exists() or not list(output.iterdir())
