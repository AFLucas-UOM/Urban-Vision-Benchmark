"""Checkpoint validation, LFS-pointer detection and the coverage table."""

from __future__ import annotations

import pytest

from uvb_bench_core import lfs_pointer_reason, sha256_file

from jetson_bench.models import BenchmarkSpec
from jetson_bench.preflight import (
    CheckpointReport,
    DatasetReport,
    EnvironmentReport,
    build_coverage,
    inspect_checkpoint,
)


# ---------------------------------------------------------------------------
# Git-LFS pointers and other files that "exist" but are not checkpoints
# ---------------------------------------------------------------------------

LFS_POINTER = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1258daaa5e2ca24d17e2393\n"
    "size 12345\n"
)


def test_git_lfs_pointer_is_not_accepted_as_a_checkpoint(tmp_path):
    pointer = tmp_path / "best.pt"
    pointer.write_text(LFS_POINTER, encoding="utf-8")
    assert lfs_pointer_reason(pointer) == "git_lfs_pointer"
    report = inspect_checkpoint(pointer)
    assert report.exists is True
    assert report.real is False
    assert report.reason == "git_lfs_pointer"


def test_zero_byte_placeholder_is_rejected(tmp_path):
    empty = tmp_path / "best.pt"
    empty.write_bytes(b"")
    assert lfs_pointer_reason(empty) == "empty_file"
    assert inspect_checkpoint(empty).real is False


def test_suspiciously_small_file_is_rejected(tmp_path):
    tiny = tmp_path / "best.pt"
    tiny.write_bytes(b"x" * 4096)
    assert "suspiciously_small" in lfs_pointer_reason(tiny)
    assert inspect_checkpoint(tiny).real is False


def test_missing_file_is_rejected(tmp_path):
    report = inspect_checkpoint(tmp_path / "absent.pt")
    assert report.exists is False
    assert report.real is False
    assert report.reason == "missing"


def test_a_plausible_checkpoint_is_accepted_with_size_and_hash(tmp_path):
    real = tmp_path / "best.pt"
    real.write_bytes(b"\x80\x02" + b"y" * 2_000_000)
    report = inspect_checkpoint(real, hash_it=True)
    assert report.real is True
    assert report.reason is None
    assert report.size_mb == pytest.approx(1.907, abs=0.01)
    assert report.sha256 == sha256_file(real)
    assert len(report.sha256) == 64


def test_hashing_can_be_skipped_for_speed(tmp_path):
    real = tmp_path / "best.pt"
    real.write_bytes(b"y" * 2_000_000)
    assert inspect_checkpoint(real, hash_it=False).sha256 is None


def test_no_checkpoint_path_reports_a_reason():
    report = inspect_checkpoint(None)
    assert report.real is False
    assert report.reason == "no checkpoint resolved"


def test_checkpoint_report_serialises_the_fields_the_row_needs(tmp_path):
    real = tmp_path / "best.pt"
    real.write_bytes(b"y" * 2_000_000)
    payload = inspect_checkpoint(real).as_dict()
    assert set(payload) == {"checkpoint_relative_path", "checkpoint_exists",
                            "checkpoint_real", "checkpoint_issue",
                            "checkpoint_mb", "checkpoint_sha256"}


# ---------------------------------------------------------------------------
# Coverage table
# ---------------------------------------------------------------------------

def _spec(model_id="m", **kwargs) -> BenchmarkSpec:
    defaults = dict(model_id=model_id, display_name=model_id, track="mdwd",
                    dataset="MDWD", task="detection", family="yolo26", scale="s",
                    engine="yolo", status="preflight_pending", input_resolution=640)
    defaults.update(kwargs)
    return BenchmarkSpec(**defaults)


def _datasets(available=True):
    return {"MDWD/detection": DatasetReport("MDWD/detection", "Datasets/MDWD",
                                            available, 369,
                                            "" if available else "split absent")}


def _environment(usable=True, cuda=True, missing=()):
    report = EnvironmentReport("detection", ".venv-jetson-detection", True,
                               "/repo/.venv-jetson-detection/bin/python")
    report.modules = {"torch": usable, "ultralytics": usable}
    for name in missing:
        report.modules[name] = False
    report.cuda_available = cuda
    report.torch_version = "2.5.0"
    return {"detection": report}


def test_a_ready_configuration_is_selected(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"y" * 2_000_000)
    spec = _spec(checkpoint=str(checkpoint))
    rows = build_coverage([spec], _datasets(), _environment(), hash_checkpoints=False)
    assert spec.status == "preflight_ok"
    assert rows[0]["selected"] is True


def test_an_lfs_pointer_blocks_the_configuration(tmp_path):
    pointer = tmp_path / "best.pt"
    pointer.write_text(LFS_POINTER, encoding="utf-8")
    spec = _spec(checkpoint=str(pointer))
    rows = build_coverage([spec], _datasets(), _environment(), hash_checkpoints=False)
    assert spec.status == "missing_checkpoint"
    assert "git_lfs_pointer" in spec.reason
    assert rows[0]["selected"] is False


def test_a_missing_dataset_blocks_the_configuration(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"y" * 2_000_000)
    spec = _spec(checkpoint=str(checkpoint))
    build_coverage([spec], _datasets(available=False), _environment(),
                   hash_checkpoints=False)
    assert spec.status == "missing_dataset"


def test_a_missing_dependency_blocks_the_configuration(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"y" * 2_000_000)
    spec = _spec(checkpoint=str(checkpoint))
    build_coverage([spec], _datasets(), _environment(missing=("ultralytics",)),
                   hash_checkpoints=False)
    assert spec.status == "dependency_error"
    assert "ultralytics" in spec.reason


def test_cpu_only_pytorch_is_refused_for_a_gpu_benchmark(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"y" * 2_000_000)
    spec = _spec(checkpoint=str(checkpoint))
    build_coverage([spec], _datasets(), _environment(cuda=False), hash_checkpoints=False)
    assert spec.status == "dependency_error"
    assert "torch.cuda.is_available() == False" in spec.reason
    assert "debugging" in spec.reason


def test_an_unresolved_request_keeps_its_original_reason():
    spec = _spec(status="missing_checkpoint", reason="never trained for MDWD")
    rows = build_coverage([spec], _datasets(), _environment(), hash_checkpoints=False)
    assert spec.status == "missing_checkpoint"
    assert rows[0]["reason"] == "never trained for MDWD"
    assert rows[0]["selected"] is False


def test_coverage_row_carries_the_whole_requested_story(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"y" * 2_000_000)
    spec = _spec(checkpoint=str(checkpoint), predictive_metric_name="test_mAP50_95",
                 predictive_metric_value=0.7527,
                 predictive_metric_source="Documents/.../mdwd_detection_results.csv")
    row = build_coverage([spec], _datasets(), _environment(),
                         hash_checkpoints=False)[0]
    for field in ("requested_model", "checkpoint_relative_path", "checkpoint_exists",
                  "checkpoint_real", "checkpoint_mb", "dataset_available",
                  "environment_usable", "smoke_status", "benchmark_status",
                  "status", "reason", "predictive_metric_value",
                  "workstation_comparable", "input_resolution"):
        assert field in row, field


def test_prompt_configurations_need_no_checkpoint():
    spec = _spec(model_id="zs_sam_3", track="zero-shot", task="prompt",
                 engine="prompt", dataset="MTSD")
    rows = build_coverage([spec], {}, {}, hash_checkpoints=False)
    assert spec.status == "preflight_ok"
    assert rows[0]["selected"] is True
