"""Power-rail resolution, energy integration and telemetry summarisation."""

from __future__ import annotations

import pytest

from jetson_bench.power import (
    UNRESOLVED,
    integrate_energy_j,
    idle_baseline,
    resolve_power_source,
    sample_power_w,
    summarise_window,
)
from jetson_bench.tegrastats import TelemetrySample, parse_tegrastats_line


# ---------------------------------------------------------------------------
# Rail resolution
# ---------------------------------------------------------------------------

def test_true_total_rail_is_preferred_and_labelled_board_input():
    source = resolve_power_source({"VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC"})
    assert source.role == "board_input_rail"
    assert source.rails == ("VDD_IN",)
    assert source.is_board_input is True


def test_pom_5v_in_is_recognised_as_a_total_rail():
    source = resolve_power_source({"POM_5V_IN", "POM_5V_GPU", "POM_5V_CPU"})
    assert source.role == "board_input_rail"
    assert source.label == "POM_5V_IN"


def test_agx_orin_rails_use_a_documented_disjoint_sum_not_board_input():
    source = resolve_power_source({"VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0", "NC"})
    assert source.role == "documented_disjoint_rail_sum"
    assert source.is_board_input is False
    assert "NC" not in source.rails      # unpopulated rails never enter a sum
    assert set(source.rails) == {"VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0"}


def test_unknown_rail_set_is_never_blindly_summed():
    source = resolve_power_source({"VDD_MYSTERY_A", "VDD_MYSTERY_B", "VDD_MYSTERY_C"})
    assert source.role == "unresolved"
    assert source.is_board_input is False
    assert "not summed" in source.reason


def test_total_rail_only_strategy_refuses_a_disjoint_sum():
    source = resolve_power_source({"VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0"},
                                  strategy="total_rail_only")
    assert source.role == "unresolved"
    assert "total_rail_only" in source.reason


def test_explicit_strategy_uses_the_configured_rails():
    source = resolve_power_source({"VDD_GPU_SOC", "VDD_CPU_CV"}, strategy="explicit",
                                  explicit_rails=["VDD_GPU_SOC"])
    assert source.role == "explicit_rail_selection"
    assert source.rails == ("VDD_GPU_SOC",)


def test_explicit_strategy_rejects_rails_the_device_does_not_report():
    source = resolve_power_source({"VDD_GPU_SOC"}, strategy="explicit",
                                  explicit_rails=["VDD_IN"])
    assert source.role == "unresolved"
    assert "not reported" in source.reason


def test_no_rails_at_all_resolves_to_unresolved():
    assert resolve_power_source(set()).role == "unresolved"


def test_resolution_is_recorded_verbatim_in_every_row():
    payload = resolve_power_source({"VDD_IN"}).as_dict()
    assert payload["power_source_rail"] == "VDD_IN"
    assert payload["power_is_board_input"] is True
    assert payload["power_source_reason"]


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

def _samples(watts: list[float], step: float = 1.0) -> list[TelemetrySample]:
    return [TelemetrySample(index * step, 0.0,
                            {"power_rails_mw": {"VDD_IN": value * 1000}})
            for index, value in enumerate(watts)]


def test_energy_is_a_trapezoidal_integral_of_measured_power():
    # 10 W held for 2 seconds = 20 J.
    assert integrate_energy_j(_samples([10.0, 10.0, 10.0]), resolve_power_source({"VDD_IN"})) \
        == pytest.approx(20.0)


def test_energy_handles_a_ramp():
    # 0 W -> 10 W over 1 s then 10 W -> 20 W over 1 s = 5 + 15 = 20 J.
    assert integrate_energy_j(_samples([0.0, 10.0, 20.0]),
                              resolve_power_source({"VDD_IN"})) == pytest.approx(20.0)


def test_a_single_sample_yields_no_energy_rather_than_an_extrapolation():
    assert integrate_energy_j(_samples([12.0]), resolve_power_source({"VDD_IN"})) is None


def test_unresolved_power_source_yields_no_energy():
    assert integrate_energy_j(_samples([12.0, 12.0]), UNRESOLVED) is None
    assert sample_power_w(_samples([12.0])[0], UNRESOLVED) is None


def test_disjoint_rail_sum_adds_only_the_named_rails():
    source = resolve_power_source({"VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0"})
    sample = TelemetrySample(0.0, 0.0, {"power_rails_mw": {
        "VDD_GPU_SOC": 5000, "VDD_CPU_CV": 2000, "VIN_SYS_5V0": 3000, "NC": 9999}})
    assert sample_power_w(sample, source) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Window summaries
# ---------------------------------------------------------------------------

def test_gross_energy_per_item_and_dynamic_energy_are_both_reported():
    source = resolve_power_source({"VDD_IN"})
    summary = summarise_window(_samples([10.0] * 11, step=0.1), source,
                               items_processed=10, idle_power_w=4.0)
    assert summary["energy_j"] == pytest.approx(10.0)       # 10 W for 1 s
    assert summary["energy_j_per_item"] == pytest.approx(1.0)
    # Dynamic subtracts the idle baseline over the same window: (10-4)*1 s.
    assert summary["dynamic_energy_j"] == pytest.approx(6.0)
    assert summary["dynamic_energy_j_per_item"] == pytest.approx(0.6)
    assert "never instead of" in summary["dynamic_energy_method"]


def test_dynamic_energy_is_absent_without_an_idle_baseline():
    summary = summarise_window(_samples([10.0, 10.0]), resolve_power_source({"VDD_IN"}),
                               items_processed=2)
    assert summary["energy_j_per_item"] is not None
    assert summary["dynamic_energy_j_per_item"] is None


def test_unresolved_power_leaves_total_fields_null_but_keeps_per_rail_telemetry():
    sample = TelemetrySample(0.0, 0.0, {"power_rails_mw": {"VDD_MYSTERY": 5000}})
    other = TelemetrySample(1.0, 0.0, {"power_rails_mw": {"VDD_MYSTERY": 5000}})
    source = resolve_power_source({"VDD_MYSTERY"})
    summary = summarise_window([sample, other], source, items_processed=2)
    assert summary["energy_j"] is None
    assert summary["energy_j_per_item"] is None
    assert "power_w_mean" not in summary
    assert summary["power_rail_mean_w"] == {"VDD_MYSTERY": 5.0}


def test_window_summary_covers_thermal_memory_and_utilisation(make_samples,
                                                              tegrastats_lines):
    samples = make_samples([tegrastats_lines["orin_nano"]] * 5)
    source = resolve_power_source({"VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC"})
    summary = summarise_window(samples, source, items_processed=5)
    assert summary["system_ram_peak_mb"] == 3140
    assert summary["system_ram_total_mb"] == 7620
    assert summary["gpu_util_mean"] == pytest.approx(61.0)
    assert summary["temperature_peak_c"] == pytest.approx(48.906, abs=0.01)
    assert "GPU" in summary["temperature_zone_mean_c"]
    assert summary["power_w_mean"] == pytest.approx(8.942)


def test_empty_window_is_reported_as_no_samples():
    summary = summarise_window([], resolve_power_source({"VDD_IN"}), items_processed=5)
    assert summary["telemetry_status"] == "no_samples"


def test_idle_baseline_extracts_only_the_fields_the_runner_needs(make_samples,
                                                                 tegrastats_lines):
    samples = make_samples([tegrastats_lines["orin_nano"]] * 4)
    baseline = idle_baseline(samples, resolve_power_source({"VDD_IN"}))
    assert baseline["idle_samples"] == 4
    assert baseline["idle_power_w_mean"] == pytest.approx(8.942)
    assert baseline["idle_temperature_peak_c"] == pytest.approx(48.906, abs=0.01)
