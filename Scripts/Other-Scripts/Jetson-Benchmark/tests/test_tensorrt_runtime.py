"""The optional TensorRT runtime stays a separate experiment."""

from __future__ import annotations

import pytest

import run_jetson_benchmark as orchestrator
from jetson_bench.analysis import group_key
from jetson_bench.config import load_config
from jetson_bench.engines import COMPARABLE_BOUNDARY, ENGINES, ITEM_UNIT
from jetson_bench.models import BenchmarkSpec
from jetson_bench.preflight import ENGINE_ENVIRONMENT, ENGINE_REQUIRED_MODULES
from jetson_bench.runner import build_job


def _specs() -> list[BenchmarkSpec]:
    return [
        BenchmarkSpec(model_id="mdwd_yolo26_s", display_name="MDWD YOLO26-S",
                      track="mdwd", dataset="MDWD", task="detection",
                      family="yolo26", scale="s", engine="yolo",
                      input_resolution=640, checkpoint="/repo/best.pt",
                      status="preflight_pending", workstation_comparable=True,
                      workstation_reason="same checkpoint and sample"),
        BenchmarkSpec(model_id="mtsd_rfdetr_m", display_name="MTSD RF-DETR-M",
                      track="mtsd", dataset="MTSD", task="detection",
                      family="rfdetr", scale="m", engine="rfdetr",
                      input_resolution=576, checkpoint="/repo/ckpt.pth",
                      status="preflight_pending"),
        BenchmarkSpec(model_id="attr_dinov3_vitl_lora", display_name="DINOv3-L LoRA",
                      track="attributes", dataset="MTSD", task="attribute",
                      family="dinov3", scale="large", engine="attribute",
                      checkpoint="/repo/best.pt", status="preflight_pending"),
        BenchmarkSpec(model_id="mdwd_rfdetr_m", display_name="MDWD RF-DETR-M",
                      track="mdwd", dataset="MDWD", task="detection",
                      family="rfdetr", scale="m", engine="rfdetr",
                      status="missing_checkpoint", reason="never trained"),
    ]


def test_tensorrt_is_not_expanded_unless_enabled(default_config_path):
    config = load_config(default_config_path)
    expanded = orchestrator.expand_tensorrt(_specs(), config)
    assert [spec.model_id for spec in expanded] == [spec.model_id for spec in _specs()]
    assert all(spec.engine != "yolo_tensorrt" for spec in expanded)


def test_enabling_tensorrt_adds_rows_rather_than_replacing_them(default_config_path):
    config = load_config(default_config_path, profile="dissertation_tensorrt")
    expanded = orchestrator.expand_tensorrt(_specs(), config)
    native = [s for s in expanded if s.engine != "yolo_tensorrt"]
    tensorrt = [s for s in expanded if s.engine == "yolo_tensorrt"]
    assert len(native) == len(_specs()), "native configurations are preserved"
    assert len(tensorrt) == 1, "only YOLO models are exported"
    assert tensorrt[0].model_id == "mdwd_yolo26_s_tensorrt"
    assert tensorrt[0].input_resolution == 640, "model/input-size identity is preserved"
    assert tensorrt[0].checkpoint == "/repo/best.pt"


def test_only_yolo_models_are_exported(default_config_path):
    config = load_config(default_config_path, profile="dissertation_tensorrt")
    expanded = orchestrator.expand_tensorrt(_specs(), config)
    exported_from = {s.model_id.removesuffix("_tensorrt")
                     for s in expanded if s.engine == "yolo_tensorrt"}
    assert exported_from == {"mdwd_yolo26_s"}


def test_an_unresolved_checkpoint_is_not_exported(default_config_path):
    config = load_config(default_config_path, profile="dissertation_tensorrt")
    expanded = orchestrator.expand_tensorrt(_specs(), config)
    assert "mdwd_rfdetr_m_tensorrt" not in {s.model_id for s in expanded}


def test_tensorrt_rows_are_never_compared_to_the_workstation(default_config_path):
    config = load_config(default_config_path, profile="dissertation_tensorrt")
    variant = [s for s in orchestrator.expand_tensorrt(_specs(), config)
               if s.engine == "yolo_tensorrt"][0]
    assert variant.workstation_comparable is False
    assert "separate optimised runtime" in variant.workstation_reason


def test_tensorrt_and_native_rows_fall_into_different_pareto_groups():
    native = {"dataset": "MDWD", "task": "detection",
              "predictive_metric_name": "test_mAP50_95",
              "timing_boundary": "end_to_end_file",
              "runtime_backend": "native_pytorch"}
    assert group_key(native) != group_key({**native, "runtime_backend": "tensorrt"})


def test_the_tensorrt_job_carries_export_and_parity_settings(default_config_path,
                                                             tmp_path):
    config = load_config(default_config_path, profile="dissertation_tensorrt")
    variant = [s for s in orchestrator.expand_tensorrt(_specs(), config)
               if s.engine == "yolo_tensorrt"][0]
    variant.extra["checkpoint_sha256"] = "a" * 64
    job = build_job(variant, [tmp_path / "a.jpg"], [], config, "cuda", None,
                    software={"tensorrt_version": "10.3.0", "cuda_version": "12.6"})
    spec = job["spec"]
    assert spec["engine"] == "yolo_tensorrt"
    assert spec["precision"] == "fp16"
    assert spec["parity_check"] is True
    assert spec["tensorrt_version"] == "10.3.0"
    assert spec["export_cache_dir"].endswith("model_exports")
    assert spec["checkpoint_sha256"] == "a" * 64


def test_int8_is_rejected_because_no_calibration_protocol_exists(tmp_path):
    engine = ENGINES["yolo_tensorrt"]({
        "model_id": "m", "engine": "yolo_tensorrt", "checkpoint": str(tmp_path / "m.pt"),
        "input_resolution": 640, "precision": "int8",
        "export_cache_dir": str(tmp_path), "items": []}, "cuda")
    with pytest.raises(RuntimeError, match="INT8 is out of scope"):
        engine.load()


def test_the_tensorrt_engine_is_registered_consistently():
    assert "yolo_tensorrt" in ENGINES
    assert ENGINE_ENVIRONMENT["yolo_tensorrt"] == "detection"
    assert "tensorrt" in ENGINE_REQUIRED_MODULES["yolo_tensorrt"]
    assert COMPARABLE_BOUNDARY["yolo_tensorrt"] == COMPARABLE_BOUNDARY["yolo"]
    assert ITEM_UNIT["yolo_tensorrt"] == "image"
