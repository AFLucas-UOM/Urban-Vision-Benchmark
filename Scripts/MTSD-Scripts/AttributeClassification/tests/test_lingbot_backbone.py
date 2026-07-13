"""LingBot-Vision wrapper: loader args, tuple handling, pooling, LoRA targets."""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mtsd_attr.backbones import BackboneUnavailableError
from mtsd_attr.backbones.lingbot_backbone import LingBotBackbone


class FakeAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)


class FakeBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = FakeAttention(dim)


class FakeLingBotViT(nn.Module):
    """Mimics LingBotVisionTransformer: token dict with is_training=True."""

    def __init__(self, dim, n_patches=4):
        super().__init__()
        self.dim = dim
        self.n_patches = n_patches
        self.patch_size = 16
        self.blocks = nn.ModuleList([FakeBlock(dim) for _ in range(2)])

    def forward(self, x, is_training=False):
        b = x.shape[0]
        patches = torch.arange(
            b * self.n_patches * self.dim,
            dtype=torch.float32).reshape(b, self.n_patches, self.dim)
        out = {
            "x_norm_clstoken": torch.full((b, self.dim), 7.0),
            "x_storage_tokens": torch.zeros(b, 4, self.dim),
            "x_norm_patchtokens": patches,
        }
        if is_training:
            return out
        return out["x_norm_clstoken"]


def _install_fake_lingbot(monkeypatch, dim=768, capture=None,
                          return_value=None):
    def load_pretrained_backbone(repo_id_or_path=None, variant="auto", *,
                                 device=None, dtype="auto", verbose=True,
                                 **kwargs):
        if capture is not None:
            capture.update(repo_id_or_path=repo_id_or_path, variant=variant,
                           device=device, dtype=dtype, verbose=verbose)
        if return_value is not None:
            return return_value
        model = FakeLingBotViT(dim)
        for p in model.parameters():
            p.requires_grad_(False)
        return model, dim

    monkeypatch.setitem(sys.modules, "lingbot_vision", SimpleNamespace(
        load_pretrained_backbone=load_pretrained_backbone))


def _cfg(adaptation="frozen", variant="base", pooling="mean_patch"):
    cfg = {
        "backbone": "lingbot",
        "lingbot_variant": variant,
        "hf_model_id": f"robbyant/lingbot-vision-vit-{variant}",
        "pooling": pooling,
        "adaptation": adaptation,
        "image_size": 224,
    }
    if adaptation == "lora":
        cfg["lora"] = {"r": 4, "alpha": 8, "dropout": 0.0,
                       "target_modules": ["qkv"]}
    return cfg


def test_loader_arguments_and_mean_patch_pooling(monkeypatch):
    captured = {}
    _install_fake_lingbot(monkeypatch, dim=768, capture=captured)
    backbone = LingBotBackbone(_cfg())
    assert captured["repo_id_or_path"] == "robbyant/lingbot-vision-vit-base"
    assert captured["variant"] == "base"
    assert captured["dtype"] == "fp32" and captured["device"] == "cpu"
    assert backbone.feature_dim == 768
    x = torch.zeros(2, 3, 224, 224)
    expected = backbone.model(x, is_training=True)["x_norm_patchtokens"] \
        .mean(dim=1)
    assert torch.equal(backbone.forward(x), expected)
    assert backbone.backbone_meta["pooling"] == "mean_patch"
    assert backbone.backbone_meta["loaded_backend"] == "lingbot_vision"


def test_large_variant_routing_and_cls_pooling(monkeypatch):
    captured = {}
    _install_fake_lingbot(monkeypatch, dim=1024, capture=captured)
    backbone = LingBotBackbone(_cfg(variant="large", pooling="cls"))
    assert captured["repo_id_or_path"] == "robbyant/lingbot-vision-vit-large"
    assert captured["variant"] == "large"
    assert backbone.feature_dim == 1024
    out = backbone.forward(torch.zeros(1, 3, 224, 224))
    assert torch.equal(out, torch.full((1, 1024), 7.0))


def test_lora_targets_fused_qkv(monkeypatch):
    _install_fake_lingbot(monkeypatch, dim=768)
    backbone = LingBotBackbone(_cfg(adaptation="lora"))
    trainable = [n for n, p in backbone.model.named_parameters()
                 if p.requires_grad]
    assert trainable, "LoRA produced no trainable parameters"
    assert all("lora_" in n for n in trainable)
    assert any(".qkv." in n for n in trainable)
    # proj must not be adapted.
    assert not any(".proj." in n and "lora_" in n for n in trainable)


def test_unexpected_return_type_fails(monkeypatch):
    _install_fake_lingbot(monkeypatch, return_value=FakeLingBotViT(768))
    with pytest.raises(BackboneUnavailableError, match="tuple"):
        LingBotBackbone(_cfg())


def test_missing_dependency_raises_backbone_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "lingbot_vision", None)
    monkeypatch.setitem(sys.modules, "lbot_vision_infer", None)
    with pytest.raises(BackboneUnavailableError, match="not installed"):
        LingBotBackbone(_cfg())


def test_invalid_variant_pooling_and_finetune_fail(monkeypatch):
    _install_fake_lingbot(monkeypatch)
    with pytest.raises(ValueError, match="lingbot_variant"):
        LingBotBackbone(_cfg(variant="giant"))
    with pytest.raises(ValueError, match="pooling"):
        LingBotBackbone(_cfg(pooling="max"))
    with pytest.raises(ValueError, match="fine-tuning"):
        LingBotBackbone(_cfg(adaptation="finetune"))
