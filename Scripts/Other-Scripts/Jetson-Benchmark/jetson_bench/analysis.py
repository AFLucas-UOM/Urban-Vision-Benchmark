"""Pareto frontiers, workstation comparison and deployment summaries.

Two rules govern everything here:

* frontiers and ratios are computed **only inside protocol-compatible groups**
  (same task, same metric definition, same timing boundary, same runtime
  backend); there is no global frontier across detection, attribute and prompt
  tasks, because their accuracy axes mean different things;
* a comparison that is not valid is labelled ``not_protocol_comparable`` and
  left empty rather than being invented.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------

def pareto_flags(points: list[tuple[float, float]]) -> list[bool]:
    """Pareto efficiency for (maximise, minimise) pairs.

    A point is efficient when no other point is at least as good on both axes
    with at least one strict improvement. Duplicated points are all reported as
    efficient - none of them dominates the other.
    """
    flags = []
    for index, (benefit, cost) in enumerate(points):
        dominated = False
        for other_index, (other_benefit, other_cost) in enumerate(points):
            if other_index == index:
                continue
            at_least_as_good = other_benefit >= benefit and other_cost <= cost
            strictly_better = other_benefit > benefit or other_cost < cost
            if at_least_as_good and strictly_better:
                dominated = True
                break
        flags.append(not dominated)
    return flags


def pareto_frontier(rows: list[dict], benefit_key: str, cost_key: str,
                    flag_key: str) -> list[dict]:
    """Annotate rows with ``flag_key`` in place; rows missing either axis get None."""
    usable = [row for row in rows
              if row.get(benefit_key) is not None and row.get(cost_key) is not None]
    for row in rows:
        row[flag_key] = None
    if not usable:
        return rows
    points = [(float(row[benefit_key]), float(row[cost_key])) for row in usable]
    for row, flag in zip(usable, pareto_flags(points)):
        row[flag_key] = flag
    return rows


PARETO_AXES = (
    ("pareto_latency", "latency_mean_ms"),
    ("pareto_energy", "energy_j_per_item"),
    ("pareto_memory", "peak_memory_mb"),
)


def group_key(row: dict) -> tuple:
    """Configurations may only be compared inside one of these groups."""
    return (row.get("dataset"), row.get("task"), row.get("predictive_metric_name"),
            row.get("timing_boundary"), row.get("runtime_backend"))


def annotate_pareto(rows: list[dict]) -> list[dict]:
    """Compute per-group frontiers for accuracy vs latency / energy / memory."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row.get("status") != "ok":
            for flag, _ in PARETO_AXES:
                row[flag] = None
            continue
        groups.setdefault(group_key(row), []).append(row)
    for key, grouped in groups.items():
        row_group = "::".join(str(part) for part in key)
        for row in grouped:
            row["pareto_group"] = row_group
            row["pareto_group_size"] = len(grouped)
        for flag, cost in PARETO_AXES:
            pareto_frontier(grouped, "predictive_metric_value", cost, flag)
    return rows


# ---------------------------------------------------------------------------
# Workstation comparison
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComparisonRule:
    """Why a Jetson row may or may not be compared with the RTX 4090 baseline."""

    valid: bool
    note: str


def compare_row(jetson: dict, baseline: dict | None, comparable: bool,
                reason: str) -> dict:
    """Build one ``jetson_vs_workstation.csv`` row.

    Memory is deliberately *not* expressed as a ratio: the Jetson figure is
    unified-system memory and the RTX 4090 figure is discrete-VRAM allocator
    peak. The two are reported side by side with an explicit note instead.
    """
    row = {
        "model": jetson.get("model_id"),
        "display_name": jetson.get("display_name"),
        "task": jetson.get("task"),
        "dataset": jetson.get("dataset"),
        "checkpoint_relative_path": jetson.get("checkpoint_relative_path"),
        "input_resolution": jetson.get("input_resolution"),
        "timing_boundary": jetson.get("timing_boundary"),
        "batch_size": jetson.get("batch_size"),
        "sample_manifest": jetson.get("sample_manifest"),
        "jetson_mean_ms": jetson.get("latency_mean_ms"),
        "jetson_p95_ms": jetson.get("latency_p95_ms"),
        "jetson_throughput": jetson.get("throughput_items_s"),
        "jetson_peak_memory_mb": jetson.get("peak_memory_mb"),
        "jetson_memory_semantics": "unified system memory (Jetson)",
        "rtx4090_mean_ms": None,
        "rtx4090_p95_ms": None,
        "rtx4090_throughput": None,
        "rtx4090_peak_gpu_memory_mb": None,
        "rtx4090_memory_semantics": "discrete GPU allocator peak (torch.cuda)",
        "slowdown_x": None,
        "throughput_ratio": None,
        "memory_ratio": None,
        "comparison_valid": False,
        "comparison_notes": reason,
    }
    if not comparable or not baseline:
        row["comparison_valid"] = False
        row["comparison_notes"] = reason or "not_protocol_comparable"
        return row

    def number(value):
        try:
            text = str(value).strip()
            return float(text) if text else None
        except (TypeError, ValueError):
            return None

    row["rtx4090_mean_ms"] = number(baseline.get("mean_latency_ms"))
    row["rtx4090_p95_ms"] = number(baseline.get("p95_latency_ms"))
    row["rtx4090_throughput"] = number(baseline.get("fps"))
    gpu_gb = number(baseline.get("peak_gpu_mem_gb"))
    row["rtx4090_peak_gpu_memory_mb"] = round(gpu_gb * 1024, 2) if gpu_gb is not None else None
    row["rtx4090_source_run"] = baseline.get("source_run")

    if row["jetson_mean_ms"] and row["rtx4090_mean_ms"]:
        row["slowdown_x"] = round(row["jetson_mean_ms"] / row["rtx4090_mean_ms"], 3)
    if row["jetson_throughput"] and row["rtx4090_throughput"]:
        row["throughput_ratio"] = round(row["jetson_throughput"] / row["rtx4090_throughput"], 4)
    row["comparison_valid"] = True
    row["comparison_notes"] = (
        reason + " | memory is reported side by side without a ratio: Jetson unified "
        "memory and RTX 4090 VRAM allocation are not the same quantity")
    return row


def build_comparison(summary_rows: list[dict], specs_by_id: dict,
                     pass_name: str = "comparative_latency") -> list[dict]:
    """One comparison row per model, from the protocol-compatible pass only.

    The sustained telemetry pass measures the same boundary but over a cycled
    workload, so it is deliberately excluded: only Pass A reproduces the
    workstation protocol. Where several repeats exist, the pooled
    ``repeat="all"`` row is preferred over any individual repeat.
    """
    candidates: dict[str, dict] = {}
    for summary in summary_rows:
        if not summary.get("is_comparable_boundary"):
            continue
        if summary.get("pass_name") != pass_name:
            continue
        model_id = summary.get("model_id")
        existing = candidates.get(model_id)
        if existing is None or summary.get("repeat") == "all":
            candidates[model_id] = summary

    rows = []
    for model_id, summary in candidates.items():
        spec = specs_by_id.get(model_id)
        if spec is None:
            continue
        baseline = (spec.extra or {}).get("workstation_row")
        rows.append(compare_row(summary, baseline, spec.workstation_comparable,
                                spec.workstation_reason))
    return rows


# ---------------------------------------------------------------------------
# Deployment summary
# ---------------------------------------------------------------------------

def deployment_rows(summary_rows: list[dict]) -> list[dict]:
    """One compact per-configuration row for the deployment discussion."""
    rows = []
    for summary in summary_rows:
        if not summary.get("is_comparable_boundary"):
            continue
        rows.append({
            "model": summary.get("model_id"),
            "display_name": summary.get("display_name"),
            "dataset": summary.get("dataset"),
            "task": summary.get("task"),
            "runtime_backend": summary.get("runtime_backend"),
            "input_resolution": summary.get("input_resolution"),
            "timing_boundary": summary.get("timing_boundary"),
            "item_unit": summary.get("item_unit"),
            "status": summary.get("status"),
            "predictive_metric_name": summary.get("predictive_metric_name"),
            "predictive_metric_value": summary.get("predictive_metric_value"),
            "latency_mean_ms": summary.get("latency_mean_ms"),
            "latency_p95_ms": summary.get("latency_p95_ms"),
            "throughput_items_s": summary.get("throughput_items_s"),
            "peak_memory_mb": summary.get("peak_memory_mb"),
            "peak_memory_basis": summary.get("peak_memory_basis"),
            "mean_power_w": summary.get("mean_power_w"),
            "energy_j_per_item": summary.get("energy_j_per_item"),
            "dynamic_energy_j_per_item": summary.get("dynamic_energy_j_per_item"),
            "temperature_peak_c": summary.get("temperature_peak_c"),
            "checkpoint_mb": summary.get("checkpoint_mb"),
            "parameters_total": summary.get("parameters_total"),
            "parameters_trainable": summary.get("parameters_trainable"),
            "pareto_group": summary.get("pareto_group"),
            "pareto_latency": summary.get("pareto_latency"),
            "pareto_energy": summary.get("pareto_energy"),
            "pareto_memory": summary.get("pareto_memory"),
        })
    return rows
