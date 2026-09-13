"""Hardware-detection fallbacks, path resolution and sample provenance."""

from __future__ import annotations

from pathlib import Path

import pytest

from jetson_bench import platform_info
from jetson_bench.paths import (
    CACHE_ENVIRONMENT,
    PROJECT_ROOT,
    benchmark_environment,
    repo_relative,
    resolve_recorded_path,
    venv_python,
)
from jetson_bench.samples import (
    WORKSTATION_SAMPLE_RUNS,
    read_workstation_manifest,
    resolve_prompt,
    resolve_sample,
    workstation_manifest_path,
)


# ---------------------------------------------------------------------------
# Hardware detection on a non-Jetson machine
# ---------------------------------------------------------------------------

def test_jetson_detection_degrades_instead_of_raising():
    info = platform_info.detect_jetson()
    assert set(info) >= {"is_jetson", "jetson_model", "architecture",
                         "l4t_version", "jetpack_version", "detection_notes"}
    if not info["is_jetson"]:
        assert info["detection_notes"], "a negative detection must explain itself"


def test_jetpack_is_never_guessed_from_an_unknown_l4t(monkeypatch):
    monkeypatch.setattr(platform_info, "_read_text",
                        lambda path: "# R99 (release), REVISION: 9.9"
                        if str(path).endswith("nv_tegra_release") else None)
    monkeypatch.setattr(platform_info, "_run", lambda *a, **k: None)
    info = platform_info.detect_jetson()
    assert info["l4t_version"] == "99.9.9"
    assert info["jetpack_version"] is None
    assert any("not determinable" in note for note in info["detection_notes"])


def test_known_l4t_maps_to_its_jetpack(monkeypatch):
    monkeypatch.setattr(platform_info, "_read_text",
                        lambda path: "# R36 (release), REVISION: 4.3"
                        if str(path).endswith("nv_tegra_release") else None)
    monkeypatch.setattr(platform_info, "_run", lambda *a, **k: None)
    assert platform_info.detect_jetson()["jetpack_version"] == "6.2"


def test_power_mode_probe_reports_absence_rather_than_failing():
    info = platform_info.detect_power_mode()
    assert set(info) >= {"nvpmodel_available", "nvpmodel_mode",
                         "jetson_clocks_available"}
    if not info["nvpmodel_available"]:
        assert info["notes"]


def test_set_power_mode_validates_the_requested_id(monkeypatch):
    monkeypatch.setattr(platform_info, "detect_power_mode", lambda: {
        "nvpmodel_available": True, "nvpmodel_mode": 0,
        "nvpmodel_available_modes": [{"id": 0, "name": "MAXN"},
                                     {"id": 1, "name": "15W"}]})
    result = platform_info.set_power_mode(7)
    assert result["applied"] is False
    assert "not offered by this Jetson" in result["error"]


def test_set_power_mode_refuses_when_nvpmodel_is_absent(monkeypatch):
    monkeypatch.setattr(platform_info, "detect_power_mode",
                        lambda: {"nvpmodel_available": False})
    result = platform_info.set_power_mode(0)
    assert result["applied"] is False
    assert "not available" in result["error"]


def test_hardware_snapshot_records_no_identifying_information():
    snapshot = platform_info.hardware_snapshot()
    forbidden = {"serial", "serial_number", "mac", "mac_address", "hostname",
                 "node", "username", "user"}
    assert not forbidden & set(snapshot)
    text = str(snapshot).lower()
    import getpass
    try:
        user = getpass.getuser()
    except Exception:
        user = None
    if user and len(user) > 2:
        assert user.lower() not in text


def test_software_snapshot_lists_the_versions_the_report_needs():
    snapshot = platform_info.software_snapshot()
    for field in ("python_version", "cuda_version", "cudnn_version",
                  "tensorrt_version", "torch_version", "torchvision_version",
                  "ultralytics_version", "rfdetr_version"):
        assert field in snapshot


def test_repository_snapshot_reports_writability_and_storage():
    snapshot = platform_info.repository_snapshot()
    assert snapshot["repo_path"] == str(PROJECT_ROOT)
    assert "writable" in snapshot and "ok" in snapshot["writable"]
    assert "storage" in snapshot
    assert isinstance(snapshot["storage_warnings"], list)


def test_non_linux_filesystem_produces_a_warning_not_an_abort(monkeypatch):
    monkeypatch.setattr(platform_info, "filesystem_for", lambda path: {
        "filesystem": "exfat", "mount_options": "rw", "free_gb": 100.0})
    snapshot = platform_info.repository_snapshot()
    assert any("exfat" in warning for warning in snapshot["storage_warnings"])


# ---------------------------------------------------------------------------
# Path handling - no personal Windows path may leak into an artefact
# ---------------------------------------------------------------------------

def test_windows_absolute_path_from_an_older_run_is_re_anchored():
    recorded = (r"E:\2. UM-Student\MSC Dissertation\Urban-Vision-Benchmark"
                r"\Datasets\MDWD\MDWD-YOLO26\test\images\example.jpg")
    resolved = resolve_recorded_path(recorded)
    # The file itself may or may not exist in this checkout, but the mapping
    # must never return the literal recorded path.
    assert resolved is None or str(resolved).startswith(str(PROJECT_ROOT))


def test_posix_absolute_path_from_another_machine_is_re_anchored():
    recorded = "/mnt/external_ssd/Urban-Vision-Benchmark/Datasets/MDWD/data.yaml"
    resolved = resolve_recorded_path(recorded)
    assert resolved is None or str(resolved).startswith(str(PROJECT_ROOT))


def test_unrecognisable_path_resolves_to_none_not_a_guess():
    assert resolve_recorded_path("/somewhere/else/entirely/file.jpg") is None
    assert resolve_recorded_path("") is None


def test_repo_relative_never_emits_a_drive_letter_or_backslash():
    relative = repo_relative(PROJECT_ROOT / "Datasets" / "MDWD" / "data.yaml")
    assert relative == "Datasets/MDWD/data.yaml"
    assert ":" not in relative and "\\" not in relative


def test_cache_locations_are_inside_the_repository():
    for value in CACHE_ENVIRONMENT.values():
        assert str(value).startswith(str(PROJECT_ROOT))


def test_venvs_live_on_the_repository_volume():
    for name in ("detection", "attribute", "prompt"):
        assert str(venv_python(name)).startswith(str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Network and privacy policy
# ---------------------------------------------------------------------------

def test_wandb_is_disabled_and_downloads_are_off_by_default():
    environment = benchmark_environment(allow_downloads=False)
    assert environment["WANDB_MODE"] == "disabled"
    assert environment["MPLBACKEND"] == "Agg"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["UVB_ALLOW_MODEL_DOWNLOADS"] == "0"


def test_downloads_are_only_enabled_when_explicitly_permitted():
    environment = benchmark_environment(allow_downloads=True)
    assert environment["HF_HUB_OFFLINE"] == "0"
    assert environment["UVB_ALLOW_MODEL_DOWNLOADS"] == "1"
    # W&B stays disabled regardless: telemetry is never uploaded.
    assert environment["WANDB_MODE"] == "disabled"


# ---------------------------------------------------------------------------
# Sample provenance
# ---------------------------------------------------------------------------

def test_every_task_maps_to_a_workstation_source_run():
    assert set(WORKSTATION_SAMPLE_RUNS) == {
        ("MDWD", "detection"), ("MTSD", "detection"),
        ("MTSD", "attribute"), ("MTSD", "prompt")}


def test_manifest_reader_orders_by_recorded_index(tmp_path):
    manifest = tmp_path / "benchmark_images_used.csv"
    manifest.write_text("index,image\n2,c.jpg\n0,a.jpg\n1,b.jpg\n", encoding="utf-8")
    assert read_workstation_manifest(manifest) == ["a.jpg", "b.jpg", "c.jpg"]


def test_manifest_reader_tolerates_an_unknown_schema(tmp_path):
    manifest = tmp_path / "benchmark_images_used.csv"
    manifest.write_text("something,else\n1,2\n", encoding="utf-8")
    assert read_workstation_manifest(manifest) == []


def test_fallback_sample_is_labelled_as_not_comparable(monkeypatch, tmp_path):
    pool = tmp_path / "images"
    pool.mkdir()
    for index in range(10):
        (pool / f"{index:03d}.jpg").write_bytes(b"\xff\xd8\xff")
    monkeypatch.setattr("jetson_bench.samples.pool_for",
                        lambda task, dataset, split: (sorted(pool.glob("*.jpg")), pool))
    sample = resolve_sample("detection", "MDWD", seed=42, max_items=4,
                            reuse_workstation=False)
    assert sample.source == "seeded_sample"
    assert sample.count == 4
    assert any("not protocol-comparable" in note for note in sample.notes)


def test_the_seeded_fallback_is_deterministic(monkeypatch, tmp_path):
    pool = tmp_path / "images"
    pool.mkdir()
    for index in range(20):
        (pool / f"{index:03d}.jpg").write_bytes(b"\xff\xd8\xff")
    monkeypatch.setattr("jetson_bench.samples.pool_for",
                        lambda task, dataset, split: (sorted(pool.glob("*.jpg")), pool))
    first = resolve_sample("detection", "MDWD", seed=42, max_items=5,
                           reuse_workstation=False)
    second = resolve_sample("detection", "MDWD", seed=42, max_items=5,
                            reuse_workstation=False)
    assert first.items == second.items


def test_manifest_rows_contain_identifiers_not_imagery(tmp_path, monkeypatch):
    pool = tmp_path / "images"
    pool.mkdir()
    (pool / "a.jpg").write_bytes(b"\xff\xd8\xff" + b"0" * 100)
    monkeypatch.setattr("jetson_bench.samples.pool_for",
                        lambda task, dataset, split: (sorted(pool.glob("*.jpg")), pool))
    sample = resolve_sample("detection", "MDWD", reuse_workstation=False)
    row = sample.manifest_rows(with_hashes=True)[0]
    assert set(row) == {"index", "task", "dataset", "relative_path", "exists",
                        "bytes", "sample_source", "workstation_run", "sha256"}
    assert "image_data" not in row and "thumbnail" not in row


def test_prompt_definition_is_reused_from_the_existing_protocol(default_config_path):
    from jetson_bench.config import load_config

    resolved = resolve_prompt(load_config(default_config_path))
    if resolved.get("error"):
        pytest.skip(f"prompt protocol unavailable: {resolved['error']}")
    assert resolved["prompt"]
    assert resolved["prompt_source"].endswith("dissertation_protocol.yaml")
    assert resolved["protocol_version"]


@pytest.mark.skipif(
    not (PROJECT_ROOT / "Results" / "Inference-Benchmark" / "InferenceSpeed").is_dir(),
    reason="workstation benchmark runs not present in this checkout")
def test_workstation_manifests_resolve_against_this_checkout():
    for (dataset, task) in WORKSTATION_SAMPLE_RUNS:
        manifest, run = workstation_manifest_path(dataset, task)
        if manifest is None:
            continue
        sample = resolve_sample(task, dataset, reuse_workstation=True)
        assert sample.source == "workstation_manifest", f"{dataset}/{task}"
        assert sample.workstation_run == run
        assert sample.unresolved == [], f"{dataset}/{task} has unresolved entries"
        for item in sample.items[:3]:
            assert Path(item).is_absolute()
            assert str(item).startswith(str(PROJECT_ROOT))
