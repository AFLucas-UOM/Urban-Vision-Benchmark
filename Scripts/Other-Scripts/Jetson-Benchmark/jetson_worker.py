#!/usr/bin/env python3
"""Single-model benchmark worker.

The orchestrator spawns one of these per model, using the Python interpreter of
the virtual environment that owns that model's dependencies. Running each model
in its own short-lived process gives three things the dissertation needs:

* **failure isolation** - a missing aarch64 wheel, a CUDA OOM or a hard crash
  ends one model, never the experiment;
* **environment isolation** - detection, attribute and prompt stacks never have
  to be dependency-compatible with each other;
* **clean memory accounting** - every model starts from a fresh process, so
  ``system_ram_model_delta`` is meaningful.

Protocol: read a JSON job from ``--job``, write a JSON result to ``--result``.
Measured intervals are reported as ``time.monotonic()`` bounds; on Linux that
clock is shared between processes, so the orchestrator can align its
``tegrastats`` samples to exactly the interval this worker timed.

The worker never trains, never writes into a dataset or an existing run
directory, and never uploads anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from jetson_bench import engines  # noqa: E402


def classify_exception(exc: BaseException) -> str:
    """Map an exception onto one of the benchmark's structured statuses."""
    name = type(exc).__name__
    text = f"{name}: {exc}".lower()
    if "out of memory" in text or name == "OutOfMemoryError":
        return "cuda_oom"
    if isinstance(exc, (ImportError, ModuleNotFoundError)) or "no module named" in text:
        return "dependency_error"
    if isinstance(exc, FileNotFoundError) or "no such file" in text:
        return "missing_checkpoint"
    if "load" in text and ("state_dict" in text or "checkpoint" in text):
        return "load_error"
    return "runtime_error"


def memory_probe() -> dict:
    return {
        "system_ram_used_mb": engines.system_ram_used_mb(),
        "process_rss_mb": engines.process_rss_mb(),
    }


def run_job(job: dict) -> dict:
    device = job.get("device", "cuda")
    spec = job["spec"]
    result: dict = {
        "model_id": spec["model_id"],
        "engine": spec["engine"],
        "device": device,
        "status": "pending",
        "notes": [],
        "passes": [],
        "baseline_memory": memory_probe(),
    }

    if device.startswith("cuda") and not engines.cuda_available(device):
        result["status"] = "dependency_error"
        result["error"] = ("torch.cuda.is_available() is False in this environment; "
                           "a Jetson GPU benchmark cannot proceed on CPU")
        return result

    engine = None
    try:
        engine = engines.build_engine(spec, device)
        engines.reset_memory_stats(device)
        ram_before_load = engines.system_ram_used_mb()
        rss_before_load = engines.process_rss_mb()

        engine.load()
        result["cold_start_s"] = engine.cold_start_s
        result["parameters_total"] = engine.parameters_total
        result["parameters_trainable"] = engine.parameters_trainable
        result["notes"] += engine.notes
        result["extra"] = dict(engine.extra)
        ram_after_load = engines.system_ram_used_mb()
        result["memory_after_load"] = {
            "system_ram_used_mb": ram_after_load,
            "process_rss_mb": engines.process_rss_mb(),
            "system_ram_model_delta_mb": (
                round(ram_after_load - ram_before_load, 2)
                if None not in (ram_after_load, ram_before_load) else None),
            "process_rss_model_delta_mb": (
                round(engines.process_rss_mb() - rss_before_load, 2)
                if None not in (engines.process_rss_mb(), rss_before_load) else None),
            **engines.torch_memory(device),
        }

        # --- smoke test -----------------------------------------------------
        smoke_started = __import__("time").perf_counter()
        smoke = engine.smoke()
        engines.synchronise(device)
        result["smoke"] = {
            "status": "smoke_ok",
            "seconds": round(__import__("time").perf_counter() - smoke_started, 4),
            "detail": smoke,
            **engines.torch_memory(device),
        }
        result["status"] = "smoke_ok"
        if job.get("smoke_only"):
            return result

        # --- measured passes -------------------------------------------------
        available = engine.boundaries()
        comparable = engines.COMPARABLE_BOUNDARY.get(spec["engine"])
        result["comparable_boundary"] = comparable
        result["item_unit"] = engines.ITEM_UNIT.get(spec["engine"], "item")

        for pass_spec in job["passes"]:
            boundary = pass_spec["boundary"]
            if boundary == "comparable":
                boundary = comparable
            if boundary not in available:
                result["notes"].append(
                    f"boundary {boundary!r} is not measurable for this engine "
                    f"(available: {sorted(available)})")
                continue
            items, call = available[boundary]
            if hasattr(engine, "reset_internal"):
                engine.reset_internal()
            measurement = engines.measure(
                call, items, device,
                warmup=int(pass_spec.get("warmup", 0)),
                boundary=boundary,
                min_seconds=float(pass_spec.get("min_seconds", 0.0)),
                max_seconds=pass_spec.get("max_seconds"),
                max_items=pass_spec.get("max_items"),
            )
            measurement["pass_name"] = pass_spec["pass_name"]
            measurement["repeat"] = int(pass_spec.get("repeat", 0))
            measurement["is_comparable_boundary"] = boundary == comparable
            if hasattr(engine, "internal_breakdown"):
                measurement.update(engine.internal_breakdown())
            measurement["memory"] = memory_probe()
            result["passes"].append(measurement)

        result["status"] = "ok" if result["passes"] else "runtime_error"
        if not result["passes"]:
            result.setdefault("error", "no measured pass produced results")
        return result

    except BaseException as exc:  # noqa: BLE001 - the point is to classify everything
        result["status"] = classify_exception(exc)
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-4000:]
        return result
    finally:
        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass
        del engine
        engines.release(device)
        result["memory_after_release"] = memory_probe()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Path to the JSON job file.")
    parser.add_argument("--result", required=True, help="Path to write the JSON result.")
    args = parser.parse_args()

    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    try:
        result = run_job(job)
    except BaseException as exc:  # noqa: BLE001
        result = {"model_id": job.get("spec", {}).get("model_id"),
                  "status": "runtime_error",
                  "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()[-4000:]}
    Path(args.result).write_text(json.dumps(result, indent=2, default=str) + "\n",
                                 encoding="utf-8")
    return 0 if result.get("status") in ("ok", "smoke_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
