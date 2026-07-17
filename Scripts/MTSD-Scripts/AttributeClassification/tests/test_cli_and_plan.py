"""CLI routing, plan/list read-only guarantees, and launcher integration."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SUBPROJECT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUBPROJECT.parents[2]
MANIFEST = SUBPROJECT / "outputs" / "manifests" / "manifest.json"


def _run(args, cwd=SUBPROJECT):
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    return subprocess.run([sys.executable, "-s", *args], cwd=str(cwd),
                          env=env, capture_output=True, text=True,
                          timeout=300)


def _snapshot(path):
    if not path.exists():
        return None
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns)


def test_plan_is_read_only_and_lists_all_16_variants():
    manifest_before = _snapshot(MANIFEST)
    ckpt_root = SUBPROJECT / "outputs" / "checkpoints"
    ckpt_dirs_before = {p.name for p in ckpt_root.iterdir()} if ckpt_root.exists() else set()

    result = _run(["run_all.py", "--plan", "--profile", "size_ablation_all"])
    assert result.returncode == 0, result.stderr
    for variant in ("dinov3_vitb_frozen", "dinov3_vitl_lora",
                    "vjepa21_vitb_frozen", "vjepa21_vitl_lora",
                    "convnext_base_frozen", "convnext_large_finetune",
                    "lingbot_vitb_frozen", "lingbot_vitl_lora"):
        assert variant in result.stdout
    assert "16 variant(s)" in result.stdout
    assert "vjepa2_1_vit_base_384" in result.stdout
    assert "robbyant/lingbot-vision-vit-large" in result.stdout

    # No manifest refresh, no new checkpoint dirs, no W&B init.
    assert _snapshot(MANIFEST) == manifest_before
    ckpt_dirs_after = {p.name for p in ckpt_root.iterdir()} if ckpt_root.exists() else set()
    assert ckpt_dirs_after == ckpt_dirs_before
    assert "wandb" not in result.stdout.lower()


def test_plan_default_selection_is_legacy_set():
    result = _run(["run_all.py", "--plan"])
    assert result.returncode == 0, result.stderr
    assert "4 variant(s)" in result.stdout
    for variant in ("dinov3", "vjepa", "convnext_frozen", "convnext"):
        assert variant in result.stdout


def test_list_profiles_and_variants():
    result = _run(["run_all.py", "--list-profiles"])
    assert result.returncode == 0, result.stderr
    assert "size_ablation_frozen" in result.stdout
    assert "size_ablation_adapted" in result.stdout
    assert "size_ablation_all" in result.stdout
    assert "legacy_default" in result.stdout

    result = _run(["run_all.py", "--list-variants"])
    assert result.returncode == 0, result.stderr
    assert "22 variant(s)" in result.stdout  # 6 legacy + 16 ablation


def test_profile_and_variants_are_mutually_exclusive():
    result = _run(["run_all.py", "--plan", "--profile", "size_ablation_all",
                   "--variants", "dinov3"])
    assert result.returncode != 0
    assert "mutually exclusive" in (result.stdout + result.stderr)


def test_unknown_profile_and_variant_fail_clearly():
    result = _run(["run_all.py", "--plan", "--profile", "nope"])
    assert result.returncode != 0
    assert "Unknown profile" in (result.stdout + result.stderr)
    result = _run(["run_all.py", "--plan", "--variants", "nope"])
    assert result.returncode != 0
    assert "Unknown variant" in (result.stdout + result.stderr)


def test_train_variant_cli():
    result = _run(["train_variant.py", "--help"])
    assert result.returncode == 0
    assert "--variant" in result.stdout
    result = _run(["train_variant.py", "--list-variants"])
    assert result.returncode == 0, result.stderr
    assert "lingbot_vitl_lora" in result.stdout
    result = _run(["train_variant.py", "--variant", "nope"])
    assert result.returncode != 0
    assert "Unknown variant" in (result.stdout + result.stderr)


def test_train_variant_routes_to_run_training(monkeypatch):
    import train_variant

    calls = {}

    def fake_run_training(variant, config_path, smoke):
        calls["args"] = (variant, config_path, smoke)

    monkeypatch.setattr(train_variant, "run_training", fake_run_training)
    monkeypatch.setattr(sys, "argv",
                        ["train_variant.py", "--variant",
                         "dinov3_vitb_frozen", "--smoke-test"])
    train_variant.main()
    assert calls["args"] == ("dinov3_vitb_frozen", None, True)


def test_launcher_has_readonly_size_ablation_plan_item():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import launch_uvb
    finally:
        sys.path.remove(str(REPO_ROOT))
    tools = {tool.id: tool for tool in launch_uvb.TOOLS}
    plan = tools["attr-size-plan"]
    assert plan.args == ("--plan", "--profile", "size_ablation_all")
    assert plan.read_only is True
    assert plan.env == "mtsd-attrcls"
    assert (REPO_ROOT / plan.script).is_file()
    assert "size" in plan.description.lower()
    attr_ui = tools["attr-ui"]
    assert "LingBot" in attr_ui.description
    assert "size" in attr_ui.description.lower()
