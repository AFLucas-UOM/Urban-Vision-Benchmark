"""Gradio comparison UI: family/size metadata formatting (no app launch)."""

from pathlib import Path

import pytest

gradio_compare = pytest.importorskip("inference.gradio_compare")


def _info(**kwargs):
    defaults = dict(path=Path("outputs/checkpoints/x/best.pt"),
                    family="DINO", label="l", variant="v", run_id="r")
    defaults.update(kwargs)
    return gradio_compare.CheckpointInfo(**defaults)


def test_family_order_includes_lingbot():
    assert "LingBot" in gradio_compare.FAMILY_ORDER
    assert gradio_compare.FAMILY_DISPLAY_NAMES["LingBot"] == "LingBot-Vision"
    assert gradio_compare.UI_FAMILY["LingBot-Vision"] == "LingBot"


def test_family_from_text_detects_lingbot():
    assert gradio_compare._family_from_text("lingbot_vitb_frozen") == "LingBot"
    assert gradio_compare._family_from_text("robbyant checkpoint") == "LingBot"
    assert gradio_compare._family_from_text("convnext_base") == "ConvNeXt"


def test_metadata_preferred_over_name_parsing():
    # variant_meta wins even when the name would parse differently.
    ckpt = {"variant": "convnext_weird_name",
            "variant_meta": {"family": "LingBot-Vision",
                             "architecture": "ViT-L/16",
                             "model_size": "large", "adaptation": "frozen",
                             "resolution": 224}}
    family = gradio_compare._family_from_metadata(
        ckpt, Path("outputs/checkpoints/convnext_weird_name/best.pt"))
    assert family == "LingBot"


def test_old_checkpoint_falls_back_to_model_cfg_then_text():
    # model_cfg reconstruction (legacy defaults) is preferred over raw text.
    ckpt = {"variant": "dinov3",
            "model_cfg": {"backbone": "dinov3", "adaptation": "frozen",
                          "image_size": 224}}
    meta = gradio_compare._checkpoint_variant_meta(ckpt)
    assert meta["architecture"] == "ViT-B/16"
    assert gradio_compare._family_from_metadata(
        ckpt, Path("x/best.pt")) == "DINO"
    # No metadata at all -> text parsing.
    assert gradio_compare._family_from_metadata(
        {}, Path("outputs/checkpoints/vjepa_lora/best.pt")) == "V-JEPA"


def test_comparison_display_names_include_size():
    ckpt_b = {"adaptation": "frozen",
              "variant_meta": {"family": "DINOv3", "architecture": "ViT-B/16",
                               "model_size": "base", "adaptation": "frozen",
                               "resolution": 224}}
    ckpt_l = {"adaptation": "frozen",
              "variant_meta": {"family": "DINOv3", "architecture": "ViT-L/16",
                               "model_size": "large", "adaptation": "frozen",
                               "resolution": 224}}
    name_b = gradio_compare._comparison_display_name(
        _info(architecture="ViT-B/16"), ckpt_b)
    name_l = gradio_compare._comparison_display_name(
        _info(architecture="ViT-L/16"), ckpt_l)
    assert name_b == "DINOv3 ViT-B/16 frozen"
    assert name_l == "DINOv3 ViT-L/16 frozen"
    assert name_b != name_l

    lingbot = {"adaptation": "lora",
               "variant_meta": {"family": "LingBot-Vision",
                                "architecture": "ViT-L/16",
                                "model_size": "large", "adaptation": "lora",
                                "resolution": 224}}
    assert gradio_compare._comparison_display_name(
        _info(family="LingBot", architecture="ViT-L/16"), lingbot) \
        == "LingBot-Vision ViT-L/16 LoRA"


def test_indistinguishable_names_are_deduped_with_run_ids():
    names = ["DINOv3 ViT-B/16 frozen", "DINOv3 ViT-B/16 frozen",
             "DINOv3 ViT-L/16 frozen"]
    run_ids = ["run-a", "run-b", "run-c"]
    deduped = gradio_compare._dedupe_display_names(names, run_ids)
    assert deduped == ["DINOv3 ViT-B/16 frozen [run-a]",
                       "DINOv3 ViT-B/16 frozen [run-b]",
                       "DINOv3 ViT-L/16 frozen"]


def test_checkpoint_details_line():
    info = _info(architecture="ViT-L/16", adaptation="lora", resolution="224",
                 run_id="dinov3_vitl_lora-20260713-101010")
    details = gradio_compare._checkpoint_details(info)
    assert "ViT-L/16" in details and "LoRA" in details and "224px" in details
    assert "dinov3_vitl_lora-20260713-101010" in details
