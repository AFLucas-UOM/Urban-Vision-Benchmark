"""Figure generation and report assembly from synthetic measurements."""

from __future__ import annotations

import pytest

from jetson_bench.analysis import annotate_pareto
from jetson_bench.figures import generate
from jetson_bench.report import build_report, derive_interpretation

matplotlib = pytest.importorskip("matplotlib")


def _detection_rows() -> list[dict]:
    rows = [
        {"model": "mdwd_yolo11_s", "display_name": "MDWD YOLO11-S", "dataset": "MDWD",
         "task": "detection", "status": "ok", "runtime_backend": "native_pytorch",
         "timing_boundary": "end_to_end_file", "item_unit": "image",
         "input_resolution": 640, "predictive_metric_name": "test_mAP50_95",
         "predictive_metric_value": 0.7334, "latency_mean_ms": 96.0,
         "latency_p95_ms": 104.0, "throughput_items_s": 10.4,
         "energy_j_per_item": 1.05, "peak_memory_mb": 1820.0,
         "mean_power_w": 10.9, "temperature_peak_c": 62.4, "checkpoint_mb": 18.3,
         "parameters_total": 9_429_727},
        {"model": "mdwd_yolo26_s", "display_name": "MDWD YOLO26-S", "dataset": "MDWD",
         "task": "detection", "status": "ok", "runtime_backend": "native_pytorch",
         "timing_boundary": "end_to_end_file", "item_unit": "image",
         "input_resolution": 640, "predictive_metric_name": "test_mAP50_95",
         "predictive_metric_value": 0.7527, "latency_mean_ms": 101.0,
         "latency_p95_ms": 110.0, "throughput_items_s": 9.9,
         "energy_j_per_item": 1.14, "peak_memory_mb": 1880.0,
         "mean_power_w": 11.3, "temperature_peak_c": 63.8, "checkpoint_mb": 19.4,
         "parameters_total": 9_951_734},
        {"model": "mdwd_yolo26_l", "display_name": "MDWD YOLO26-L", "dataset": "MDWD",
         "task": "detection", "status": "ok", "runtime_backend": "native_pytorch",
         "timing_boundary": "end_to_end_file", "item_unit": "image",
         "input_resolution": 640, "predictive_metric_name": "test_mAP50_95",
         "predictive_metric_value": 0.7824, "latency_mean_ms": 233.0,
         "latency_p95_ms": 254.0, "throughput_items_s": 4.3,
         "energy_j_per_item": 2.71, "peak_memory_mb": 2440.0,
         "mean_power_w": 11.6, "temperature_peak_c": 69.1, "checkpoint_mb": 50.5,
         "parameters_total": 26_184_054},
        {"model": "mdwd_yolo12_s", "display_name": "MDWD YOLO12-S", "dataset": "MDWD",
         "task": "detection", "status": "ok", "runtime_backend": "native_pytorch",
         "timing_boundary": "end_to_end_file", "item_unit": "image",
         "input_resolution": 640, "predictive_metric_name": "test_mAP50_95",
         "predictive_metric_value": 0.7441, "latency_mean_ms": 148.0,
         "latency_p95_ms": 161.0, "throughput_items_s": 6.8,
         "energy_j_per_item": 1.70, "peak_memory_mb": 1910.0,
         "mean_power_w": 11.5, "temperature_peak_c": 65.0, "checkpoint_mb": 18.1,
         "parameters_total": 9_255_071},
    ]
    annotate_pareto(rows)
    return rows


def _comparison_rows() -> list[dict]:
    return [
        {"model": "mdwd_yolo26_s", "display_name": "MDWD YOLO26-S", "task": "detection",
         "jetson_mean_ms": 101.0, "rtx4090_mean_ms": 14.55, "slowdown_x": 6.94,
         "jetson_p95_ms": 110.0, "rtx4090_p95_ms": 15.58, "jetson_throughput": 9.9,
         "rtx4090_throughput": 68.7, "comparison_valid": True,
         "comparison_notes": "same checkpoint and sample"},
        {"model": "mtsd_yolo26_s", "display_name": "MTSD YOLO26-S @1280",
         "task": "detection", "comparison_valid": False,
         "comparison_notes": "not_protocol_comparable: workstation ran this "
                             "checkpoint at imgsz 640"},
    ]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def test_tradeoff_figures_are_generated_when_the_data_supports_them(tmp_path):
    manifest = generate(_detection_rows(), _comparison_rows(), tmp_path)
    by_name = {entry["figure"]: entry for entry in manifest}
    for name in ("MDWD_accuracy_vs_latency.png", "MDWD_accuracy_vs_energy.png",
                 "MDWD_accuracy_vs_memory.png", "workstation_vs_jetson_latency.png",
                 "jetson_power_by_model.png", "jetson_memory_by_model.png",
                 "jetson_peak_temperature_by_model.png",
                 "jetson_energy_by_model.png"):
        assert by_name[name]["generated"] is True, name
        assert (tmp_path / name).stat().st_size > 1000


def test_a_figure_without_data_is_skipped_with_a_recorded_reason(tmp_path):
    manifest = generate(_detection_rows(), _comparison_rows(), tmp_path)
    skipped = [e for e in manifest if not e["generated"]]
    assert skipped, "the MTSD and zero-shot figures have no data here"
    for entry in skipped:
        assert entry["reason"]
        assert not (tmp_path / entry["figure"]).exists()


def test_tasks_with_different_accuracy_semantics_get_separate_figures(tmp_path):
    rows = _detection_rows()
    rows.append({
        "model": "attr_dinov3_vitl_lora", "display_name": "DINOv3 ViT-L + LoRA",
        "dataset": "MTSD", "task": "attribute", "status": "ok",
        "runtime_backend": "native_pytorch", "timing_boundary": "model_forward",
        "item_unit": "crop", "predictive_metric_name": "test_mean_macro_f1",
        "predictive_metric_value": 0.8959, "latency_mean_ms": 28.0,
        "energy_j_per_item": 0.31, "peak_memory_mb": 2300.0})
    annotate_pareto(rows)
    manifest = {entry["figure"]: entry for entry in generate(rows, [], tmp_path)}
    assert manifest["MDWD_accuracy_vs_latency.png"]["generated"] is True
    # One attribute point is not enough for its own frontier figure, and it is
    # never merged into the MDWD mAP axis.
    assert manifest["Attribute_accuracy_vs_latency.png"]["generated"] is False
    assert manifest["MDWD_accuracy_vs_latency.png"]["points"] == 4


def test_no_figure_is_written_for_a_failed_configuration(tmp_path):
    rows = _detection_rows()
    rows.append({**rows[0], "model": "broken", "display_name": "broken",
                 "status": "cuda_oom"})
    manifest = {entry["figure"]: entry for entry in generate(rows, [], tmp_path)}
    assert manifest["MDWD_accuracy_vs_latency.png"]["points"] == 4


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _context(deployment, comparison, coverage=None, failures=None) -> dict:
    return {
        "run_id": "20260911-120000",
        "config": {"seed": 42, "batch_size": 1,
                   "comparative_latency": {"repeats": 3, "warmup": 3,
                                           "reuse_workstation_sample": True},
                   "sustained_telemetry": {"repeats": 3,
                                           "minimum_duration_seconds": 20,
                                           "maximum_duration_seconds": 180,
                                           "warmup": 20},
                   "telemetry": {"tegrastats_interval_ms": 100,
                                 "idle_baseline_seconds": 10},
                   "cooldown": {"min_seconds": 10, "target_delta_c": 3,
                                "max_seconds": 120},
                   "runtime": {"tensorrt_yolo": False}},
        "hardware": {"jetson_model": "NVIDIA Jetson Orin Nano Developer Kit",
                     "architecture": "aarch64", "kernel_release": "5.15.148-tegra",
                     "l4t_version": "36.4.3", "jetpack_version": "6.2",
                     "is_jetson": True, "system_ram_total_mb": 7620.0,
                     "system_ram_available_mb": 5100.0,
                     "nvpmodel": {"nvpmodel_mode": 0, "nvpmodel_mode_name": "MAXN"}},
        "software": {"cuda_version": "12.6", "cudnn_version": "9.3.0",
                     "tensorrt_version": "10.3.0", "torch_version": "2.5.0",
                     "torchvision_version": "0.20.0", "python_version": "3.10.12",
                     "ultralytics_version": "8.4.60", "rfdetr_version": "1.3.0",
                     "torch_cuda_available": True, "gpu_name": "Orin"},
        "repository": {"git_commit_short": "353d511c", "git_branch": "main",
                       "git_dirty": False, "storage": {"filesystem": "ext4",
                                                       "mount_point": "/media/uvb/SSD",
                                                       "free_gb": 412.0},
                       "storage_warnings": []},
        "policy": {"jetson_clocks_enabled": False},
        "samples": {"MDWD/detection": {"sample_count": 50,
                                       "sample_source": "workstation_manifest",
                                       "workstation_run": "20260910-163921",
                                       "unresolved_entries": 0}},
        "idle": {"idle_power_w_mean": 4.2, "idle_temperature_peak_c": 41.0},
        "power_source": {"power_source_role": "board_input_rail",
                         "power_source_rail": "VDD_IN",
                         "power_is_board_input": True,
                         "power_source_reason": "platform exposes a total rail"},
        "coverage": coverage or [],
        "deployment": deployment,
        "comparison": comparison,
        "failures": failures or [],
        "figures": [{"figure": "MDWD_accuracy_vs_latency.png", "generated": True}],
        "memory_rows": [], "thermal_rows": [],
        "interpretation": derive_interpretation(deployment, comparison),
    }


def test_report_contains_every_required_section():
    report = build_report(_context(_detection_rows(), _comparison_rows()))
    for heading in ("## Experimental Platform", "## Protocol",
                    "## Requested Model Coverage", "## Supervised Detection — MDWD",
                    "## Supervised Detection — MTSD", "## Attribute Classification",
                    "## Optional Zero-Shot Localisation",
                    "## Memory Requirements", "## Thermal Behaviour",
                    "## Workstation vs Jetson", "## Pareto-Efficient Configurations",
                    "## Models That Could Not Be Executed",
                    "## Deployment Interpretation", "## Measurement Limitations",
                    "## Reproducibility Manifest"):
        assert heading in report, heading


def test_report_records_the_power_measurement_basis():
    report = build_report(_context(_detection_rows(), _comparison_rows()))
    assert "VDD_IN" in report
    assert "board_input_rail" in report


def test_report_warns_when_power_is_a_module_estimate_not_board_input():
    context = _context(_detection_rows(), _comparison_rows())
    context["power_source"] = {"power_source_role": "documented_disjoint_rail_sum",
                               "power_source_rail": "VDD_CPU_CV+VDD_GPU_SOC",
                               "power_is_board_input": False,
                               "power_source_reason": "sum of AGX Orin module rails"}
    report = build_report(context)
    assert "module power" in report
    assert "not board input power" in report


def test_report_says_so_when_no_power_rail_could_be_resolved():
    context = _context(_detection_rows(), _comparison_rows())
    context["power_source"] = {"power_source_role": "unresolved",
                               "power_source_rail": "",
                               "power_is_board_input": False,
                               "power_source_reason": "no usable rail"}
    report = build_report(context)
    assert "Board power is not estimated" in report


def test_incomparable_configurations_are_listed_separately():
    report = build_report(_context(_detection_rows(), _comparison_rows()))
    assert "### Not protocol-comparable" in report
    assert "MTSD YOLO26-S @1280" in report


def test_interpretation_states_only_measured_facts():
    statements = derive_interpretation(_detection_rows(), _comparison_rows())
    joined = " ".join(statements)
    assert "MDWD detection" in joined
    assert "ms/image" in joined
    assert "slower than the RTX 4090" in joined
    assert "Pareto-efficient" in joined


def test_interpretation_declines_to_speak_without_energy_measurements():
    rows = [{**row, "energy_j_per_item": None} for row in _detection_rows()]
    annotate_pareto(rows)
    statements = derive_interpretation(rows, [])
    assert any("No energy figure could be derived" in s for s in statements)
    assert any("no workstation slowdown factor is reported" in s for s in statements)


def test_interpretation_handles_a_run_with_no_results():
    statements = derive_interpretation([], [])
    assert statements == ["No configuration completed, so no deployment "
                          "conclusion can be drawn."]


def test_accuracy_cost_trade_off_is_quantified():
    statements = " ".join(derive_interpretation(_detection_rows(), _comparison_rows()))
    # YOLO26-L is the most accurate and YOLO11-S the fastest in this fixture.
    assert "MDWD YOLO26-L" in statements
    assert "MDWD YOLO11-S" in statements
    assert "x its Jetson latency" in statements
