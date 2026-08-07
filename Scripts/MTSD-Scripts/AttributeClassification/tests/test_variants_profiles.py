"""Variant registry, metadata, profiles, and selection precedence."""

import pytest

from mtsd_attr.config import adaptation_of, load_config, model_training_config
from mtsd_attr.variants import (available_profiles, build_plan,
                                resolve_profile, select_variants,
                                variant_metadata)

LEGACY = ["dinov3", "dinov3_lora", "vjepa", "vjepa_lora", "convnext_frozen",
          "convnext"]
FROZEN_8 = ["dinov3_vitb_frozen", "dinov3_vitl_frozen", "vjepa21_vitb_frozen",
            "vjepa21_vitl_frozen", "convnext_base_frozen",
            "convnext_large_frozen", "lingbot_vitb_frozen",
            "lingbot_vitl_frozen"]
ADAPTED_8 = ["dinov3_vitb_lora", "dinov3_vitl_lora", "vjepa21_vitb_lora",
             "vjepa21_vitl_lora", "convnext_base_finetune",
             "convnext_large_finetune", "lingbot_vitb_lora",
             "lingbot_vitl_lora", "convnext_base_lora",
             "convnext_large_lora"]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_every_new_variant_and_legacy_variant_resolves(cfg):
    for variant in LEGACY + FROZEN_8 + ADAPTED_8:
        assert variant in cfg["models"], variant
        meta = variant_metadata(cfg["models"][variant], variant)
        assert meta["family"] in ("DINOv3", "V-JEPA", "ConvNeXt",
                                  "LingBot-Vision")


def test_profile_membership_and_counts(cfg):
    profiles = available_profiles(cfg)
    assert profiles["legacy_default"] == ["dinov3", "vjepa", "convnext_frozen",
                                          "convnext"]
    assert resolve_profile(cfg, "size_ablation_frozen") == FROZEN_8
    assert resolve_profile(cfg, "size_ablation_adapted") == ADAPTED_8
    assert resolve_profile(cfg, "size_ablation_all") == FROZEN_8 + ADAPTED_8
    assert len(FROZEN_8) == 8 and len(ADAPTED_8) == 10
    assert len(resolve_profile(cfg, "size_ablation_all")) == 18


def test_legacy_variants_excluded_from_ablation_profiles(cfg):
    members = set(resolve_profile(cfg, "size_ablation_all"))
    assert not members.intersection(LEGACY)


def test_new_variant_metadata_is_explicit(cfg):
    m = variant_metadata(cfg["models"]["dinov3_vitl_frozen"],
                         "dinov3_vitl_frozen")
    assert m["architecture"] == "ViT-L/16" and m["model_size"] == "large"
    assert m["model_id"] == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert m["resolution"] == 224 and m["adaptation"] == "frozen"

    m = variant_metadata(cfg["models"]["vjepa21_vitb_lora"],
                         "vjepa21_vitb_lora")
    assert m["model_id"] == "vjepa2_1_vit_base_384"
    assert m["resolution"] == 384 and m["adaptation"] == "lora"
    assert cfg["models"]["vjepa21_vitb_lora"]["torch_hub_checkpoint_url"] \
        .endswith("vjepa2_1_vitb_dist_vitG_384.pt")
    assert cfg["models"]["vjepa21_vitb_lora"]["allow_backend_fallback"] is False

    m = variant_metadata(cfg["models"]["convnext_large_finetune"],
                         "convnext_large_finetune")
    assert m["model_size"] == "large" and m["adaptation"] == "finetune"
    assert "convnext_large" in m["model_id"]

    m = variant_metadata(cfg["models"]["lingbot_vitl_frozen"],
                         "lingbot_vitl_frozen")
    assert m["family"] == "LingBot-Vision"
    assert m["model_id"] == "robbyant/lingbot-vision-vit-large"


def test_legacy_metadata_fallback_for_old_checkpoint_cfgs():
    # Old checkpoints store model_cfg without family/architecture/model_size.
    m = variant_metadata({"backbone": "dinov3", "adaptation": "frozen",
                          "hf_model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                          "image_size": 224})
    assert (m["family"], m["architecture"], m["model_size"]) == \
        ("DINOv3", "ViT-B/16", "base")
    m = variant_metadata({"backbone": "vjepa", "adaptation": "frozen",
                          "backend": "torch_hub",
                          "image_size_torch_hub": 384,
                          "image_size_transformers": 256})
    assert (m["family"], m["architecture"], m["resolution"]) == \
        ("V-JEPA", "ViT-L/16", 384)
    m = variant_metadata({"backbone": "convnext", "adaptation": "finetune",
                          "image_size": 224})
    assert (m["family"], m["model_size"]) == ("ConvNeXt", "tiny")


def test_unknown_values_fail_clearly(cfg):
    with pytest.raises(ValueError, match="adaptation"):
        adaptation_of({"adaptation": "weird"})
    with pytest.raises(ValueError, match="Unknown profile"):
        resolve_profile(cfg, "nope")
    with pytest.raises(ValueError, match="Unknown variant"):
        select_variants(cfg, variants=["no_such_variant"])
    from mtsd_attr.backbones import build_backbone
    with pytest.raises(ValueError, match="Unknown backbone"):
        build_backbone({"backbone": "nope"})


def test_selection_precedence(cfg):
    default = select_variants(cfg)
    assert default == ["dinov3", "vjepa", "convnext_frozen", "convnext"]
    assert select_variants(cfg, profile="size_ablation_frozen") == FROZEN_8
    assert select_variants(cfg, variants=["dinov3"]) == ["dinov3"]
    with pytest.raises(ValueError, match="mutually exclusive"):
        select_variants(cfg, profile="size_ablation_all", variants=["dinov3"])
    with_include = select_variants(cfg, profile="size_ablation_frozen",
                                   include=["dinov3_vitb_lora"])
    assert with_include == FROZEN_8 + ["dinov3_vitb_lora"]
    # --include deduplicates against the base set.
    assert select_variants(cfg, include=["dinov3"]) == default


def test_plan_rows_and_effective_batch_consistency(cfg):
    rows = build_plan(cfg, resolve_profile(cfg, "size_ablation_all"))
    assert len(rows) == 18
    for row in rows:
        assert row["effective_batch_size"] == \
            row["batch_size"] * row["gradient_accumulation_steps"]
        # Fairness: initial settings target effective batch 32 everywhere.
        assert row["effective_batch_size"] == 32
        assert row["output_dir"].endswith(row["variant"])
        for key in ("family", "architecture", "model_size", "adaptation",
                    "model_id", "resolution"):
            assert row[key]


def test_gradient_accumulation_normalisation(cfg):
    _, training = model_training_config(cfg, "vjepa21_vitl_lora")
    assert training["batch_size"] == 16
    assert training["gradient_accumulation_steps"] == 2
    assert training["effective_batch_size"] == 32
    _, training = model_training_config(cfg, "dinov3")
    assert training["gradient_accumulation_steps"] == 1
    assert training["effective_batch_size"] == training["batch_size"]
