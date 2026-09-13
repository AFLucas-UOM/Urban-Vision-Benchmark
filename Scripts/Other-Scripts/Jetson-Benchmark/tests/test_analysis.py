"""Pareto frontiers, workstation joining and result aggregation."""

from __future__ import annotations

import pytest

from uvb_bench_core import summarise_latencies

from jetson_bench.analysis import (
    annotate_pareto,
    build_comparison,
    compare_row,
    deployment_rows,
    group_key,
    pareto_flags,
    pareto_frontier,
)
from jetson_bench.models import BenchmarkSpec
from jetson_bench.runner import aggregate_repeats


# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------

def test_pareto_keeps_the_accurate_and_the_fast_and_drops_the_dominated():
    #            (accuracy, latency)
    points = [(0.90, 100.0),   # most accurate
              (0.70, 20.0),    # fastest
              (0.80, 60.0),    # on the frontier
              (0.60, 80.0)]    # dominated by both (0.70, 20) and (0.80, 60)
    assert pareto_flags(points) == [True, True, True, False]


def test_a_single_point_is_trivially_efficient():
    assert pareto_flags([(0.5, 10.0)]) == [True]


def test_duplicate_points_are_both_efficient():
    assert pareto_flags([(0.8, 50.0), (0.8, 50.0)]) == [True, True]


def test_a_strictly_worse_point_is_dominated():
    assert pareto_flags([(0.9, 10.0), (0.8, 20.0)]) == [True, False]


def test_equal_accuracy_but_slower_is_dominated():
    assert pareto_flags([(0.8, 10.0), (0.8, 20.0)]) == [True, False]


def test_rows_missing_an_axis_get_none_not_false():
    rows = [{"acc": 0.9, "lat": 10.0}, {"acc": None, "lat": 5.0},
            {"acc": 0.7, "lat": None}]
    pareto_frontier(rows, "acc", "lat", "flag")
    assert rows[0]["flag"] is True
    assert rows[1]["flag"] is None
    assert rows[2]["flag"] is None


def _row(model, dataset, task, metric_name, accuracy, latency, energy=None,
         memory=None, boundary="end_to_end_file", backend="native_pytorch"):
    return {"model": model, "display_name": model, "dataset": dataset, "task": task,
            "status": "ok", "predictive_metric_name": metric_name,
            "predictive_metric_value": accuracy, "latency_mean_ms": latency,
            "energy_j_per_item": energy, "peak_memory_mb": memory,
            "timing_boundary": boundary, "runtime_backend": backend}


def test_frontiers_are_never_computed_across_tasks():
    rows = [
        _row("yolo26s", "MDWD", "detection", "test_mAP50_95", 0.75, 15.0),
        _row("yolo26l", "MDWD", "detection", "test_mAP50_95", 0.78, 40.0),
        _row("dinov3_l", "MTSD", "attribute", "test_mean_macro_f1", 0.90, 8.0),
    ]
    annotate_pareto(rows)
    groups = {row["pareto_group"] for row in rows}
    assert len(groups) == 2
    assert rows[2]["pareto_group_size"] == 1      # attribute row is alone


def test_different_timing_boundaries_are_not_compared():
    rows = [
        _row("a", "MDWD", "detection", "test_mAP50_95", 0.75, 15.0,
             boundary="end_to_end_file"),
        _row("b", "MDWD", "detection", "test_mAP50_95", 0.70, 5.0,
             boundary="model_forward"),
    ]
    annotate_pareto(rows)
    assert rows[0]["pareto_group"] != rows[1]["pareto_group"]
    assert rows[0]["pareto_latency"] is True
    assert rows[1]["pareto_latency"] is True      # each is alone in its group


def test_tensorrt_rows_form_their_own_group():
    rows = [
        _row("a", "MDWD", "detection", "test_mAP50_95", 0.75, 15.0,
             backend="native_pytorch"),
        _row("a_trt", "MDWD", "detection", "test_mAP50_95", 0.75, 6.0,
             backend="tensorrt"),
    ]
    annotate_pareto(rows)
    assert group_key(rows[0]) != group_key(rows[1])


def test_separate_flags_are_produced_for_latency_energy_and_memory():
    rows = [
        _row("small", "MDWD", "detection", "test_mAP50_95", 0.70, 10.0, 0.5, 900),
        _row("large", "MDWD", "detection", "test_mAP50_95", 0.78, 40.0, 3.0, 2400),
        _row("bad", "MDWD", "detection", "test_mAP50_95", 0.60, 45.0, 3.5, 2600),
    ]
    annotate_pareto(rows)
    assert [r["pareto_latency"] for r in rows] == [True, True, False]
    assert [r["pareto_energy"] for r in rows] == [True, True, False]
    assert [r["pareto_memory"] for r in rows] == [True, True, False]


def test_failed_rows_are_excluded_from_every_frontier():
    rows = [_row("ok", "MDWD", "detection", "test_mAP50_95", 0.75, 15.0)]
    rows.append({**_row("bad", "MDWD", "detection", "test_mAP50_95", 0.99, 1.0),
                 "status": "cuda_oom"})
    annotate_pareto(rows)
    assert rows[0]["pareto_latency"] is True
    assert rows[1]["pareto_latency"] is None


# ---------------------------------------------------------------------------
# Workstation comparison
# ---------------------------------------------------------------------------

JETSON = {"model_id": "mdwd_yolo26_s", "display_name": "MDWD YOLO26-S",
          "task": "detection", "dataset": "MDWD", "latency_mean_ms": 145.6,
          "latency_p95_ms": 158.2, "throughput_items_s": 6.85,
          "peak_memory_mb": 2100.0, "input_resolution": 640,
          "timing_boundary": "end_to_end_file", "batch_size": 1}

BASELINE = {"mean_latency_ms": "14.55", "p95_latency_ms": "15.58", "fps": "68.7",
            "peak_gpu_mem_gb": "0.31", "source_run": "20260910-163921"}


def test_slowdown_and_throughput_ratio_are_computed_when_valid():
    row = compare_row(JETSON, BASELINE, True, "same checkpoint and sample")
    assert row["comparison_valid"] is True
    assert row["slowdown_x"] == pytest.approx(145.6 / 14.55, rel=1e-3)
    assert row["throughput_ratio"] == pytest.approx(6.85 / 68.7, rel=1e-3)
    assert row["rtx4090_peak_gpu_memory_mb"] == pytest.approx(317.44)


def test_memory_is_never_expressed_as_a_ratio():
    row = compare_row(JETSON, BASELINE, True, "same checkpoint and sample")
    assert row["memory_ratio"] is None
    assert "unified" in row["jetson_memory_semantics"]
    assert "discrete" in row["rtx4090_memory_semantics"]
    assert "not the same quantity" in row["comparison_notes"]


def test_an_incomparable_row_gets_no_numbers_at_all():
    row = compare_row(JETSON, BASELINE, False, "not_protocol_comparable: imgsz differs")
    assert row["comparison_valid"] is False
    assert row["slowdown_x"] is None
    assert row["rtx4090_mean_ms"] is None
    assert "not_protocol_comparable" in row["comparison_notes"]


def test_a_missing_baseline_never_invents_one():
    row = compare_row(JETSON, None, True, "would be comparable but no baseline row")
    assert row["comparison_valid"] is False
    assert row["rtx4090_mean_ms"] is None


def _spec_for_comparison():
    return {"m": BenchmarkSpec(
        model_id="m", display_name="M", track="mdwd", dataset="MDWD",
        task="detection", family="yolo26", scale="s", engine="yolo",
        workstation_comparable=True, workstation_reason="ok",
        extra={"workstation_row": BASELINE})}


def test_build_comparison_only_uses_the_comparable_boundary():
    summaries = [
        {**JETSON, "model_id": "m", "pass_name": "comparative_latency",
         "repeat": 0, "is_comparable_boundary": True},
        {**JETSON, "model_id": "m", "pass_name": "comparative_latency",
         "repeat": 0, "is_comparable_boundary": False,
         "timing_boundary": "model_forward"},
    ]
    rows = build_comparison(summaries, _spec_for_comparison())
    assert len(rows) == 1
    assert rows[0]["timing_boundary"] == "end_to_end_file"


def test_the_sustained_pass_is_never_compared_to_the_workstation():
    """Pass B cycles the sample; only Pass A reproduces the workstation protocol."""
    summaries = [
        {**JETSON, "model_id": "m", "pass_name": "sustained_telemetry",
         "repeat": 0, "is_comparable_boundary": True, "latency_mean_ms": 999.0},
    ]
    assert build_comparison(summaries, _spec_for_comparison()) == []


def test_the_pooled_repeat_row_is_preferred_over_an_individual_repeat():
    summaries = [
        {**JETSON, "model_id": "m", "pass_name": "comparative_latency",
         "repeat": 0, "is_comparable_boundary": True, "latency_mean_ms": 140.0},
        {**JETSON, "model_id": "m", "pass_name": "comparative_latency",
         "repeat": 1, "is_comparable_boundary": True, "latency_mean_ms": 150.0},
        {**JETSON, "model_id": "m", "pass_name": "comparative_latency",
         "repeat": "all", "is_comparable_boundary": True, "latency_mean_ms": 145.0},
    ]
    rows = build_comparison(summaries, _spec_for_comparison())
    assert len(rows) == 1
    assert rows[0]["jetson_mean_ms"] == pytest.approx(145.0)


# ---------------------------------------------------------------------------
# Aggregation across repeats
# ---------------------------------------------------------------------------

def _summary(model, repeat, boundary, mean, items=50, pass_name="comparative_latency"):
    return {"model_id": model, "display_name": model, "repeat": repeat,
            "pass_name": pass_name, "timing_boundary": boundary, "status": "ok",
            "latency_mean_ms": mean, "timed_items": items, "timed_seconds": 5.0,
            "throughput_items_s": 1000.0 / mean}


def test_repeats_are_pooled_without_being_merged_away():
    rows = [_summary("m", 0, "end_to_end_file", 100.0),
            _summary("m", 1, "end_to_end_file", 110.0),
            _summary("m", 2, "end_to_end_file", 120.0)]
    observations = {("m", "comparative_latency", "end_to_end_file"):
                    [100.0] * 50 + [110.0] * 50 + [120.0] * 50}
    pooled = aggregate_repeats(rows, observations)

    assert len(rows) == 3                      # originals untouched
    assert len(pooled) == 1
    combined = pooled[0]
    assert combined["repeat"] == "all"
    assert combined["repeat_count"] == 3
    assert combined["repeat_means_ms"] == "100.000; 110.000; 120.000"
    assert combined["timed_items"] == 150
    # Pooled from the individual observations, not an average of averages.
    assert combined["latency_mean_ms"] == pytest.approx(110.0)
    assert combined["latency_std_ms"] > 0


def test_a_single_repeat_produces_no_pooled_row():
    rows = [_summary("m", 0, "end_to_end_file", 100.0)]
    assert aggregate_repeats(rows, {}) == []


def test_boundaries_are_pooled_separately():
    rows = [_summary("m", 0, "end_to_end_file", 100.0),
            _summary("m", 1, "end_to_end_file", 110.0),
            _summary("m", 0, "model_forward", 40.0),
            _summary("m", 1, "model_forward", 42.0)]
    pooled = aggregate_repeats(rows, {})
    assert {row["timing_boundary"] for row in pooled} == {"end_to_end_file",
                                                          "model_forward"}


def test_failed_repeats_are_not_pooled():
    rows = [_summary("m", 0, "end_to_end_file", 100.0),
            {**_summary("m", 1, "end_to_end_file", 105.0), "status": "cuda_oom"}]
    assert aggregate_repeats(rows, {}) == []


# ---------------------------------------------------------------------------
# Distribution summary
# ---------------------------------------------------------------------------

def test_p99_is_null_when_the_sample_cannot_support_it():
    assert summarise_latencies([1.0] * 50)["p99_ms"] is None
    assert summarise_latencies([1.0] * 150)["p99_ms"] is not None


def test_distribution_reports_spread_and_coefficient_of_variation():
    summary = summarise_latencies([10.0, 12.0, 14.0, 16.0])
    assert summary["mean_ms"] == pytest.approx(13.0)
    assert summary["median_ms"] == pytest.approx(13.0)
    assert summary["min_ms"] == 10.0
    assert summary["max_ms"] == 16.0
    assert summary["cv"] == pytest.approx(summary["std_ms"] / 13.0, rel=1e-3)


def test_empty_observations_produce_an_empty_summary():
    assert summarise_latencies([]) == {}


# ---------------------------------------------------------------------------
# Deployment rows
# ---------------------------------------------------------------------------

def test_deployment_rows_only_use_the_comparable_boundary():
    summaries = [{**JETSON, "model_id": "m", "is_comparable_boundary": True,
                  "status": "ok"},
                 {**JETSON, "model_id": "m", "is_comparable_boundary": False,
                  "status": "ok"}]
    assert len(deployment_rows(summaries)) == 1
