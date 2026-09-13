"""tegrastats parsing across Jetson generations and truncated output."""

from __future__ import annotations

import pytest

from jetson_bench.tegrastats import parse_tegrastats_line


def test_agx_orin_line_is_fully_parsed(tegrastats_lines):
    sample = parse_tegrastats_line(tegrastats_lines["agx_orin"])
    assert sample["ram_used_mb"] == 6421
    assert sample["ram_total_mb"] == 30536
    assert sample["swap_used_mb"] == 0
    assert sample["lfb_blocks"] == 3
    assert sample["cpu_online_cores"] == 12
    assert sample["cpu_freq_max_mhz"] == 2201
    assert sample["gpu_util_pct"] == 78
    assert sample["gpu_freq_mhz"] == 1300
    assert sample["emc_util_pct"] == 31
    assert sample["emc_freq_mhz"] == 3199
    assert sample["device_timestamp"] == "11-29-2025 10:14:02"


def test_agx_orin_power_rails_and_suffixes(tegrastats_lines):
    sample = parse_tegrastats_line(tegrastats_lines["agx_orin"])
    rails = sample["power_rails_mw"]
    assert rails["VDD_GPU_SOC"] == 9431
    assert rails["VDD_CPU_CV"] == 3122
    assert rails["VIN_SYS_5V0"] == 4835
    assert rails["NC"] == 0
    assert sample["power_rails_avg_mw"]["VDD_GPU_SOC"] == 8102
    # RAM / SWAP / EMC have the same `NAME a/b` shape but are not rails.
    assert not {"RAM", "SWAP", "EMC_FREQ", "GR3D_FREQ"} & set(rails)


def test_disabled_sensors_are_dropped_not_reported_as_minus_256(tegrastats_lines):
    sample = parse_tegrastats_line(tegrastats_lines["agx_orin"])
    temperatures = sample["temperatures_c"]
    assert "CV0" not in temperatures and "CV1" not in temperatures
    assert temperatures["CPU"] == pytest.approx(57.593)
    assert temperatures["GPU"] == pytest.approx(56.125)
    assert sample["temperature_max_c"] == pytest.approx(57.593)


def test_orin_nano_exposes_a_total_input_rail(tegrastats_lines):
    sample = parse_tegrastats_line(tegrastats_lines["orin_nano"])
    assert "VDD_IN" in sample["power_rails_mw"]
    assert sample["power_rails_mw"]["VDD_IN"] == 8942
    # Two cores are offline on this board.
    assert sample["cpu_online_cores"] == 4
    assert sample["cpu_offline_cores"] == 2


def test_xavier_nx_without_mw_suffixes(tegrastats_lines):
    sample = parse_tegrastats_line(tegrastats_lines["xavier_nx"])
    rails = sample["power_rails_mw"]
    assert rails["VDD_IN"] == 3902
    assert rails["VDD_CPU_GPU_CV"] == 1331
    assert rails["VDD_SOC"] == 997
    # This generation reports GR3D_FREQ without a frequency.
    assert sample["gpu_util_pct"] == 33
    assert "gpu_freq_mhz" not in sample
    assert sample["temperatures_c"]["PMIC"] == 100


def test_nano_iram_field_is_not_mistaken_for_a_power_rail(tegrastats_lines):
    sample = parse_tegrastats_line(tegrastats_lines["nano"])
    rails = sample["power_rails_mw"]
    assert set(rails) == {"POM_5V_IN", "POM_5V_GPU", "POM_5V_CPU"}
    assert "IRAM" not in rails
    assert sample["ram_used_mb"] == 1276


@pytest.mark.parametrize("key", ["minimal", "thermal_only", "empty"])
def test_missing_fields_never_raise(tegrastats_lines, key):
    sample = parse_tegrastats_line(tegrastats_lines[key])
    assert isinstance(sample, dict)
    assert "raw" in sample


def test_minimal_line_still_yields_what_is_present(tegrastats_lines):
    sample = parse_tegrastats_line(tegrastats_lines["minimal"])
    assert sample["ram_used_mb"] == 900
    assert sample["gpu_util_pct"] == 0
    assert "power_rails_mw" not in sample
    assert "temperatures_c" not in sample


def test_thermal_only_line_has_no_power_or_memory(tegrastats_lines):
    sample = parse_tegrastats_line(tegrastats_lines["thermal_only"])
    assert sample["temperature_max_c"] == pytest.approx(41.5)
    assert "ram_used_mb" not in sample
    assert "power_rails_mw" not in sample


def test_raw_line_is_always_retained(tegrastats_lines):
    for line in tegrastats_lines.values():
        assert parse_tegrastats_line(line)["raw"] == line.rstrip("\n")


def test_garbage_input_is_tolerated():
    for line in ("not tegrastats output at all",
                 "RAM RAM RAM", "CPU [", "VDD_IN", "\x00\x01\x02"):
        assert isinstance(parse_tegrastats_line(line), dict)
