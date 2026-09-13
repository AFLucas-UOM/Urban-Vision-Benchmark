"""Benchmark orchestration: passes, telemetry alignment, failure isolation.

Execution order per model is deterministic and driven by the configuration:

    smoke test -> PASS A (protocol-compatible latency) -> cooldown
               -> PASS B (sustained telemetry)         -> cooldown -> release

Each model runs in its own worker process (see ``jetson_worker.py``). The
orchestrator owns ``tegrastats`` and aligns its samples to the monotonic window
each worker reports, so power, thermal, memory and utilisation figures belong to
exactly the interval that was timed.

Nothing here writes outside the run directory. Datasets, annotations, trained
checkpoints and existing experiment directories are read-only throughout.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from uvb_bench_core import summarise_latencies

from . import power as power_module
from .engines import COMPARABLE_BOUNDARY, ITEM_UNIT
from .models import BenchmarkSpec
from .paths import (
    ATTRCLS_DIR,
    CACHE_ROOT,
    JETSON_BENCH_DIR,
    PROMPTDETECT_DIR,
    benchmark_environment,
    repo_relative,
    venv_python,
)
from .platform_info import thermal_zones
from .preflight import ENGINE_ENVIRONMENT
from .state import RunState, atomic_write_json
from .tegrastats import TegrastatsMonitor

STATUSES = (
    "pending", "preflight_ok", "smoke_ok", "benchmarking", "ok", "skipped",
    "missing_checkpoint", "missing_dataset", "dependency_error", "load_error",
    "cuda_oom", "runtime_error", "thermal_abort", "telemetry_unavailable",
    "ambiguous_checkpoint",
)

PASS_A = "comparative_latency"
PASS_B = "sustained_telemetry"


class BenchmarkInterrupted(Exception):
    """Raised when the operator interrupts the run; completed results are kept."""


@dataclass
class ModelOutcome:
    spec: BenchmarkSpec
    status: str
    reason: str | None = None
    summaries: list[dict] = field(default_factory=list)
    raw_rows: list[dict] = field(default_factory=list)
    telemetry_rows: list[dict] = field(default_factory=list)
    worker_results: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Worker invocation
# ---------------------------------------------------------------------------

def build_job(spec: BenchmarkSpec, items: list[Path], passes: list[dict],
              config: dict, device: str, prompt: dict | None,
              smoke_only: bool = False, software: dict | None = None) -> dict:
    payload = {
        "device": device,
        "smoke_only": smoke_only,
        "passes": passes,
        "spec": {
            "model_id": spec.model_id,
            "engine": spec.engine,
            "scale": spec.scale,
            "checkpoint": spec.checkpoint,
            "checkpoint_sha256": (spec.extra or {}).get("checkpoint_sha256"),
            "input_resolution": spec.input_resolution,
            "items": [str(p) for p in items],
            "conf_threshold": float((config.get("detection") or {}).get(
                "conf_threshold", 0.30)),
            "attrcls_dir": str(ATTRCLS_DIR),
            "promptdetect_dir": str(PROMPTDETECT_DIR),
        },
    }
    if spec.engine == "prompt":
        payload["spec"]["backend_label"] = (spec.extra or {}).get("backend_label", spec.display_name)
        payload["spec"]["prompt"] = (prompt or {}).get("prompt")
        payload["spec"]["conf_threshold"] = float((prompt or {}).get("conf_threshold", 0.30))
    if spec.engine == "yolo_tensorrt":
        tensorrt = (config.get("runtime") or {}).get("tensorrt") or {}
        payload["spec"].update({
            "export_cache_dir": str(CACHE_ROOT / "model_exports"),
            "precision": tensorrt.get("precision", "fp16"),
            "parity_check": bool(tensorrt.get("parity_check", True)),
            "parity_iou_threshold": float(tensorrt.get("parity_iou_threshold", 0.5)),
            "tensorrt_version": (software or {}).get("tensorrt_version"),
            "cuda_version": (software or {}).get("cuda_version"),
            "parameters_total": (spec.extra or {}).get("parameters_total"),
        })
    return payload


def worker_interpreter(spec: BenchmarkSpec, use_venvs: bool) -> str:
    if not use_venvs:
        return sys.executable
    candidate = venv_python(ENGINE_ENVIRONMENT.get(spec.engine, "detection"))
    return str(candidate) if candidate.is_file() else sys.executable


def run_worker(spec: BenchmarkSpec, job: dict, run_dir: Path, tag: str,
               timeout: float, use_venvs: bool, allow_downloads: bool,
               log_dir: Path) -> dict:
    """Run one worker process and return its parsed result.

    The process is always reaped: on timeout it is terminated then killed, and
    no monitoring child is left behind.
    """
    import os

    work = run_dir / "worker"
    work.mkdir(parents=True, exist_ok=True)
    job_path = work / f"{spec.model_id}.{tag}.job.json"
    result_path = work / f"{spec.model_id}.{tag}.result.json"
    log_path = log_dir / f"{spec.model_id}.{tag}.log"
    atomic_write_json(job_path, job)

    environment = dict(os.environ)
    environment.update(benchmark_environment(allow_downloads))
    environment["PYTHONNOUSERSITE"] = "1"
    environment.setdefault("PYTHONUNBUFFERED", "1")

    command = [worker_interpreter(spec, use_venvs), "-s",
               str(JETSON_BENCH_DIR / "jetson_worker.py"),
               "--job", str(job_path), "--result", str(result_path)]

    log_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"$ {' '.join(command)}\n\n")
        log.flush()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, cwd=str(JETSON_BENCH_DIR),
                                   start_new_session=(os.name != "nt"))
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate(process)
            return {"model_id": spec.model_id, "status": "runtime_error",
                    "error": f"worker exceeded the {timeout:.0f}s timeout and was terminated",
                    "log": repo_relative(log_path)}
        except KeyboardInterrupt:
            _terminate(process)
            raise BenchmarkInterrupted("interrupted while a model worker was running")
    duration = time.perf_counter() - start

    if not result_path.is_file():
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        status = "cuda_oom" if "out of memory" in tail.lower() else "runtime_error"
        return {"model_id": spec.model_id, "status": status,
                "error": f"worker exited with code {returncode} and wrote no result",
                "log": repo_relative(log_path), "log_tail": tail}
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["worker_seconds"] = round(duration, 3)
    result["worker_returncode"] = returncode
    result["log"] = repo_relative(log_path)
    return result


def _terminate(process: subprocess.Popen) -> None:
    """Terminate a worker and every process it spawned."""
    import os

    for action in ("terminate", "kill"):
        if process.poll() is not None:
            return
        try:
            if os.name != "nt" and hasattr(os, "killpg"):
                os.killpg(os.getpgid(process.pid),
                          signal.SIGTERM if action == "terminate" else signal.SIGKILL)
            else:
                getattr(process, action)()
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired, ProcessLookupError):
            continue


# ---------------------------------------------------------------------------
# Thermal management
# ---------------------------------------------------------------------------

def current_max_temperature(monitor: TegrastatsMonitor | None) -> float | None:
    if monitor is not None:
        sample = monitor.snapshot()
        if sample and "temperature_max_c" in sample.parsed:
            return sample.parsed["temperature_max_c"]
    zones = thermal_zones()
    return max(zones.values()) if zones else None


def cooldown(config: dict, monitor: TegrastatsMonitor | None,
             baseline_temperature: float | None, log) -> dict:
    """Wait between models, bounded so the benchmark can never hang."""
    settings = config.get("cooldown") or {}
    minimum = float(settings.get("min_seconds", 0))
    maximum = float(settings.get("max_seconds", 0))
    target_delta = float(settings.get("target_delta_c", 0))
    started = time.monotonic()
    record = {"cooldown_min_seconds": minimum, "cooldown_max_seconds": maximum,
              "cooldown_target_delta_c": target_delta,
              "cooldown_baseline_c": baseline_temperature}
    if maximum <= 0:
        record.update({"cooldown_seconds": 0.0, "cooldown_outcome": "disabled"})
        return record
    outcome = "reached_target"
    while True:
        elapsed = time.monotonic() - started
        if elapsed >= maximum:
            outcome = "max_seconds_reached"
            break
        temperature = current_max_temperature(monitor)
        if elapsed >= minimum:
            if baseline_temperature is None or temperature is None:
                outcome = "min_seconds_reached_no_thermal_reading"
                break
            if temperature <= baseline_temperature + target_delta:
                break
        time.sleep(0.5)
    record.update({
        "cooldown_seconds": round(time.monotonic() - started, 2),
        "cooldown_outcome": outcome,
        "cooldown_end_temperature_c": current_max_temperature(monitor),
    })
    log(f"    cooldown {record['cooldown_seconds']:.1f}s ({outcome})")
    return record


# ---------------------------------------------------------------------------
# Pass construction
# ---------------------------------------------------------------------------

def pass_a_specs(config: dict, spec: BenchmarkSpec) -> list[dict]:
    section = config.get(PASS_A) or {}
    if not section.get("enabled", True):
        return []
    boundaries = list((config.get("timing") or {}).get("boundaries", []))
    comparable = COMPARABLE_BOUNDARY.get(spec.engine)
    if comparable and comparable not in boundaries:
        boundaries.append(comparable)
    passes = []
    for repeat in range(int(section.get("repeats", 1))):
        for boundary in boundaries:
            passes.append({
                "pass_name": PASS_A, "repeat": repeat, "boundary": boundary,
                "warmup": int(section.get("warmup", 3)),
            })
    return passes


def pass_b_specs(config: dict, spec: BenchmarkSpec) -> list[dict]:
    section = config.get(PASS_B) or {}
    if not section.get("enabled", True):
        return []
    boundary = section.get("boundary", "comparable")
    return [{
        "pass_name": PASS_B, "repeat": repeat, "boundary": boundary,
        "warmup": int(section.get("warmup", 20)),
        "min_seconds": float(section.get("minimum_duration_seconds", 20)),
        "max_seconds": float(section.get("maximum_duration_seconds", 180)),
        "max_items": int(section.get("maximum_items", 20000)),
    } for repeat in range(int(section.get("repeats", 1)))]


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------

def summarise_pass(spec: BenchmarkSpec, measurement: dict, worker: dict,
                   context: dict, telemetry: dict) -> dict:
    """Build one ``latency_summary.csv`` row for a single measured pass."""
    latencies = measurement.get("latencies_ms") or []
    distribution = summarise_latencies(latencies)
    memory_after_load = worker.get("memory_after_load") or {}
    torch_allocated = measurement.get("torch_peak_allocated_mb")
    system_peak = telemetry.get("system_ram_peak_mb")

    row = {
        # -- provenance ---------------------------------------------------
        "run_id": context["run_id"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": context["git_commit"],
        "git_dirty": context["git_dirty"],
        "jetson_model": context["jetson_model"],
        "l4t_version": context["l4t_version"],
        "jetpack_version": context["jetpack_version"],
        "cuda_version": context["cuda_version"],
        "cudnn_version": context["cudnn_version"],
        "tensorrt_version": context["tensorrt_version"],
        "torch_version": worker.get("extra", {}).get("torch_version") or context["torch_version"],
        "python_version": context["python_version"],
        # -- identity -----------------------------------------------------
        "model_id": spec.model_id,
        "display_name": spec.display_name,
        "dataset": spec.dataset,
        "task": spec.task,
        "track": spec.track,
        "model_family": spec.family,
        "model_scale": spec.scale,
        "adaptation": spec.adaptation,
        "augmentation": spec.augmentation,
        "input_resolution": spec.input_resolution,
        "checkpoint_relative_path": repo_relative(spec.checkpoint) if spec.checkpoint else None,
        "checkpoint_sha256": (spec.extra or {}).get("checkpoint_sha256"),
        "checkpoint_mb": (spec.extra or {}).get("checkpoint_mb"),
        "run_directory": repo_relative(spec.run_directory) if spec.run_directory else None,
        "parameters_total": worker.get("parameters_total"),
        "parameters_trainable": worker.get("parameters_trainable"),
        # -- protocol -----------------------------------------------------
        # Per row, never run-wide: one run may contain both runtimes and they
        # must never be mixed into a single comparison.
        "runtime_backend": ("tensorrt" if spec.engine == "yolo_tensorrt"
                            else context.get("runtime_backend", "native_pytorch")),
        "tensorrt_engine": (worker.get("extra") or {}).get("tensorrt_engine"),
        "tensorrt_precision": (worker.get("extra") or {}).get("tensorrt_precision"),
        "tensorrt_export_seconds": (worker.get("extra") or {}).get("tensorrt_export_seconds"),
        "tensorrt_engine_reused": (worker.get("extra") or {}).get("tensorrt_engine_reused"),
        "tensorrt_version_used": (worker.get("extra") or {}).get("tensorrt_version"),
        "precision": ((worker.get("extra") or {}).get("tensorrt_precision")
                      if spec.engine == "yolo_tensorrt"
                      else context.get("precision", "fp32")),
        "batch_size": 1,
        "seed": context["seed"],
        "sample_manifest": context["sample_manifest"].get(f"{spec.dataset}/{spec.task}"),
        "sample_source": context["sample_source"].get(f"{spec.dataset}/{spec.task}"),
        "warmup": measurement.get("warmup"),
        "pass_name": measurement.get("pass_name"),
        "repeat": measurement.get("repeat"),
        "timing_boundary": measurement.get("timing_boundary"),
        "is_comparable_boundary": measurement.get("is_comparable_boundary"),
        "item_unit": ITEM_UNIT.get(spec.engine, "item"),
        "timed_items": measurement.get("timed_items"),
        "timed_seconds": measurement.get("timed_seconds"),
        "cold_start_s": worker.get("cold_start_s"),
        # -- latency ------------------------------------------------------
        "latency_mean_ms": distribution.get("mean_ms"),
        "latency_median_ms": distribution.get("median_ms"),
        "latency_p50_ms": distribution.get("p50_ms"),
        "latency_p90_ms": distribution.get("p90_ms"),
        "latency_p95_ms": distribution.get("p95_ms"),
        "latency_p99_ms": distribution.get("p99_ms"),
        "latency_min_ms": distribution.get("min_ms"),
        "latency_max_ms": distribution.get("max_ms"),
        "latency_std_ms": distribution.get("std_ms"),
        "latency_cv": distribution.get("cv"),
        "throughput_items_s": measurement.get("throughput_items_s"),
        # -- memory -------------------------------------------------------
        "torch_peak_allocated_mb": torch_allocated,
        "torch_peak_reserved_mb": measurement.get("torch_peak_reserved_mb"),
        "system_ram_peak_mb": system_peak,
        "system_ram_model_delta_mb": memory_after_load.get("system_ram_model_delta_mb"),
        "process_rss_model_delta_mb": memory_after_load.get("process_rss_model_delta_mb"),
        "peak_memory_mb": system_peak if system_peak is not None else torch_allocated,
        "peak_memory_basis": ("tegrastats unified system RAM" if system_peak is not None
                              else "torch CUDA allocator peak (not total Jetson memory)"),
        # -- accuracy provenance -------------------------------------------
        "predictive_metric_name": spec.predictive_metric_name,
        "predictive_metric_value": spec.predictive_metric_value,
        "predictive_metric_source": spec.predictive_metric_source,
        "predictive_protocol": spec.predictive_protocol,
        # -- platform policy ------------------------------------------------
        "nvpmodel_mode": context.get("nvpmodel_mode"),
        "nvpmodel_mode_name": context.get("nvpmodel_mode_name"),
        "jetson_clocks_enabled": context.get("jetson_clocks_enabled"),
        "status": "ok",
        "notes": "; ".join(worker.get("notes", [])),
    }
    for key, value in measurement.items():
        if key.startswith(("ultralytics_", "backend_")):
            row[key] = value
    row.update({k: v for k, v in telemetry.items()
                if k not in ("power_rail_mean_w", "power_rail_peak_w",
                             "temperature_zone_mean_c", "temperature_zone_peak_c")})
    row["mean_power_w"] = telemetry.get("power_w_mean")
    row["median_power_w"] = telemetry.get("power_w_median")
    row["p95_power_w"] = telemetry.get("power_w_p95")
    row["peak_power_w"] = telemetry.get("power_w_peak")
    row["gpu_util_mean"] = telemetry.get("gpu_util_mean")
    row["gpu_util_peak"] = telemetry.get("gpu_util_peak")
    return row


def aggregate_repeats(rows: list[dict],
                      observations: dict[tuple, list[float]]) -> list[dict]:
    """Add pooled ``repeat="all"`` rows per (model, pass, boundary).

    Individual repeats are never merged away: they remain in
    ``latency_summary.csv`` exactly as measured. These extra rows carry the
    distribution pooled over every repeat's individual per-item observations
    (not an average of averages), plus the list of per-repeat means so a reader
    can see the between-repeat spread directly.
    """
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("repeat") == "all":
            continue
        key = (row["model_id"], row.get("pass_name"), row.get("timing_boundary"))
        grouped.setdefault(key, []).append(row)

    pooled = []
    for key, group in grouped.items():
        if len(group) < 2:
            continue
        latencies = observations.get(key, [])
        combined = dict(group[0])
        combined["repeat"] = "all"
        combined["repeat_count"] = len(group)
        combined["repeat_means_ms"] = "; ".join(
            f"{r['latency_mean_ms']:.3f}" for r in group
            if r.get("latency_mean_ms") is not None)
        combined["timed_items"] = sum(r.get("timed_items") or 0 for r in group)
        combined["timed_seconds"] = round(sum(r.get("timed_seconds") or 0 for r in group), 4)
        if latencies:
            distribution = summarise_latencies(latencies)
            combined.update({
                "latency_mean_ms": distribution["mean_ms"],
                "latency_median_ms": distribution["median_ms"],
                "latency_p50_ms": distribution["p50_ms"],
                "latency_p90_ms": distribution["p90_ms"],
                "latency_p95_ms": distribution["p95_ms"],
                "latency_p99_ms": distribution["p99_ms"],
                "latency_min_ms": distribution["min_ms"],
                "latency_max_ms": distribution["max_ms"],
                "latency_std_ms": distribution["std_ms"],
                "latency_cv": distribution["cv"],
            })
        throughputs = [r["throughput_items_s"] for r in group
                       if r.get("throughput_items_s") is not None]
        if throughputs:
            combined["throughput_items_s"] = round(sum(throughputs) / len(throughputs), 3)
        for field_name in ("mean_power_w", "peak_power_w", "energy_j_per_item",
                           "dynamic_energy_j_per_item", "temperature_peak_c",
                           "system_ram_peak_mb", "peak_memory_mb", "gpu_util_mean"):
            values = [r[field_name] for r in group if r.get(field_name) is not None]
            if values:
                combined[field_name] = (round(max(values), 4)
                                        if "peak" in field_name
                                        else round(sum(values) / len(values), 5))
        pooled.append(combined)
    return pooled
