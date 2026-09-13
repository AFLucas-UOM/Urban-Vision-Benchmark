"""Configuration parsing, profiles, overrides and validation."""

from __future__ import annotations

import pytest

from jetson_bench.config import (
    ConfigError,
    apply_overrides,
    apply_profile,
    deep_merge,
    load_config,
    simple_yaml_load,
    validate,
)


# ---------------------------------------------------------------------------
# The shipped configuration
# ---------------------------------------------------------------------------

def test_default_config_loads_and_validates(default_config_path):
    config = load_config(default_config_path)
    assert config["seed"] == 42
    assert config["batch_size"] == 1
    assert validate(config) == []


def test_default_config_encodes_the_required_model_matrix(default_config_path):
    models = load_config(default_config_path)["models"]

    mdwd = [(m["family"], m["scale"]) for m in models["mdwd_detection"]]
    assert mdwd == [("yolo26", "s"), ("yolo26", "l"), ("rfdetr", "m"),
                    ("rfdetr", "s"), ("yolo12", "s"), ("yolo11", "s")]

    mtsd = [(m["family"], m["scale"], m["augmentation"], m["input_size"])
            for m in models["mtsd_detection"]]
    assert mtsd == [
        ("yolo26", "s", "strong", 1280),
        ("yolo26", "m", "strong", 1280),
        ("rfdetr", "m", "strong", "trained"),
        ("rfdetr", "s", "strong", "trained"),
        ("yolo12", "s", "strong", 1280),
        ("yolo11", "s", "strong", 1280),
    ]

    assert models["attribute"] == ["dinov3_vitl_lora", "vjepa21_vitl_lora",
                                   "dinov3_vitb_lora", "vjepa21_vitb_lora"]


def test_rfdetr_is_not_forced_to_1280(default_config_path):
    models = load_config(default_config_path)["models"]["mtsd_detection"]
    for entry in models:
        if entry["family"] == "rfdetr":
            assert entry["input_size"] == "trained"


def test_cosmos_32b_is_excluded_from_the_default_matrix(default_config_path):
    zero_shot = load_config(default_config_path)["zero_shot"]
    assert "cosmos_reason2_32b" not in zero_shot["candidates"]
    assert "cosmos_reason2_32b" in zero_shot["excluded"]


def test_downloads_are_off_by_default(default_config_path):
    assert load_config(default_config_path)["zero_shot"]["allow_downloads"] is False


def test_power_and_clock_changes_are_off_by_default(default_config_path):
    power = load_config(default_config_path)["power"]
    assert power["change_nvpmodel"] is False
    assert power["lock_clocks"] is False


def test_tensorrt_is_off_by_default(default_config_path):
    assert load_config(default_config_path)["runtime"]["tensorrt_yolo"] is False


def test_prompt_definition_comes_from_the_existing_protocol(default_config_path):
    zero_shot = load_config(default_config_path)["zero_shot"]
    assert zero_shot["prompt_source"] == "dissertation_protocol"
    assert zero_shot["prompt_id"].startswith("mtsd-")


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

def test_smoke_profile_shrinks_the_experiment(default_config_path):
    smoke = load_config(default_config_path, profile="smoke")
    dissertation = load_config(default_config_path, profile="dissertation")
    assert smoke["comparative_latency"]["repeats"] == 1
    assert smoke["sustained_telemetry"]["repeats"] == 1
    assert (smoke["sustained_telemetry"]["minimum_duration_seconds"]
            < dissertation["sustained_telemetry"]["minimum_duration_seconds"])
    assert smoke["zero_shot"]["mode"] == "off"
    assert smoke["outputs"]["figures"] is False


def test_smoke_profile_preserves_the_model_matrix(default_config_path):
    smoke = load_config(default_config_path, profile="smoke")
    dissertation = load_config(default_config_path, profile="dissertation")
    assert smoke["models"] == dissertation["models"]


def test_tensorrt_profile_enables_only_the_optional_runtime(default_config_path):
    config = load_config(default_config_path, profile="dissertation_tensorrt")
    assert config["runtime"]["tensorrt_yolo"] is True
    assert config["runtime"]["native"] is True


def test_unknown_profile_is_rejected(default_config_path):
    with pytest.raises(ConfigError, match="Unknown profile"):
        load_config(default_config_path, profile="does-not-exist")


def test_profile_application_does_not_mutate_the_source():
    config = {"a": {"b": 1}, "profiles": {"p": {"a": {"b": 2}}}}
    merged = apply_profile(config, "p")
    assert merged["a"]["b"] == 2
    assert config["a"]["b"] == 1
    assert "profiles" not in merged


# ---------------------------------------------------------------------------
# Overrides and validation
# ---------------------------------------------------------------------------

def test_dotted_overrides_reach_nested_keys():
    config = apply_overrides({"telemetry": {"tegrastats_interval_ms": 100}},
                             {"telemetry.tegrastats_interval_ms": 250})
    assert config["telemetry"]["tegrastats_interval_ms"] == 250


def test_overrides_create_missing_sections():
    config = apply_overrides({}, {"a.b.c": 5})
    assert config["a"]["b"]["c"] == 5


def test_cli_overrides_apply_to_the_shipped_config(default_config_path):
    config = load_config(default_config_path, "dissertation",
                         {"sustained_telemetry.repeats": 1,
                          "telemetry.tegrastats_interval_ms": 250})
    assert config["sustained_telemetry"]["repeats"] == 1
    assert config["telemetry"]["tegrastats_interval_ms"] == 250


def test_batch_size_other_than_one_is_rejected(default_config_path):
    with pytest.raises(ConfigError, match="batch_size must be 1"):
        load_config(default_config_path, "dissertation", {"batch_size": 4})


def test_impossible_cooldown_window_is_rejected(default_config_path):
    with pytest.raises(ConfigError, match="cooldown.max_seconds"):
        load_config(default_config_path, "dissertation",
                    {"cooldown.min_seconds": 200, "cooldown.max_seconds": 10})


def test_sustained_duration_bounds_are_checked(default_config_path):
    with pytest.raises(ConfigError, match="maximum_duration_seconds"):
        load_config(default_config_path, "dissertation",
                    {"sustained_telemetry.maximum_duration_seconds": 1})


def test_absurd_telemetry_interval_is_rejected(default_config_path):
    with pytest.raises(ConfigError, match="tegrastats_interval_ms"):
        load_config(default_config_path, "dissertation",
                    {"telemetry.tegrastats_interval_ms": 0})


def test_unknown_timing_boundary_is_rejected(default_config_path):
    with pytest.raises(ConfigError, match="unknown timing boundary"):
        load_config(default_config_path, "dissertation",
                    {"timing.boundaries": ["wall_clock_guess"]})


def test_unknown_power_strategy_is_rejected(default_config_path):
    with pytest.raises(ConfigError, match="rail_strategy"):
        load_config(default_config_path, "dissertation",
                    {"power.rail_strategy": "sum_everything"})


def test_unknown_zero_shot_mode_is_rejected(default_config_path):
    with pytest.raises(ConfigError, match="zero_shot.mode"):
        load_config(default_config_path, "dissertation", {"zero_shot.mode": "maybe"})


# ---------------------------------------------------------------------------
# The standard-library YAML fallback
# ---------------------------------------------------------------------------

def test_deep_merge_replaces_sequences_and_merges_mappings():
    merged = deep_merge({"a": {"x": 1, "y": 2}, "l": [1, 2]},
                        {"a": {"y": 3}, "l": [9]})
    assert merged == {"a": {"x": 1, "y": 3}, "l": [9]}


def test_fallback_parser_handles_nested_mappings_and_flow_sequences():
    parsed = simple_yaml_load(
        "top:\n"
        "  number: 42\n"
        "  ratio: 0.5\n"
        "  flag: true\n"
        "  nothing: null\n"
        "  items: [a, b, c]\n"
        "  nested:\n"
        "    deeper: hello\n")
    assert parsed["top"] == {"number": 42, "ratio": 0.5, "flag": True,
                             "nothing": None, "items": ["a", "b", "c"],
                             "nested": {"deeper": "hello"}}


def test_fallback_parser_handles_block_sequences_of_mappings():
    parsed = simple_yaml_load(
        "models:\n"
        "  - {family: yolo26, scale: s, input_size: 1280}\n"
        "  - {family: rfdetr, scale: m, input_size: trained}\n")
    assert parsed["models"][0]["input_size"] == 1280
    assert parsed["models"][1]["input_size"] == "trained"


def test_fallback_parser_handles_scalar_sequences():
    parsed = simple_yaml_load("names:\n  - alpha\n  - beta\n")
    assert parsed["names"] == ["alpha", "beta"]


def test_fallback_parser_strips_comments_but_not_quoted_hashes():
    parsed = simple_yaml_load('a: 1  # trailing comment\nb: "has # inside"\n')
    assert parsed == {"a": 1, "b": "has # inside"}


def test_fallback_parser_matches_pyyaml_on_the_shipped_config(default_config_path):
    yaml = pytest.importorskip("yaml")
    text = default_config_path.read_text(encoding="utf-8")
    assert simple_yaml_load(text) == yaml.safe_load(text)
