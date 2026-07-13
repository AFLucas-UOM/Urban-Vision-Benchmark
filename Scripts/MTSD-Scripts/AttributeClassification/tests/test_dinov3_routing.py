"""DINOv3 Base/Large config routing and gated-access error handling."""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mtsd_attr.backbones import BackboneUnavailableError
from mtsd_attr.backbones.dinov3_backbone import DINOv3Backbone
from mtsd_attr.config import load_config


class FakeHFViT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.dummy = nn.Linear(1, 1)

    def forward(self, pixel_values):
        b = pixel_values.shape[0]
        return SimpleNamespace(pooler_output=torch.zeros(b, self.dim),
                               last_hidden_state=torch.zeros(b, 5, self.dim))


def _install_fake_transformers(monkeypatch, dims, captured):
    def from_pretrained(model_id):
        captured.append(model_id)
        if model_id not in dims:
            raise OSError(f"403 Client Error: gated repo {model_id}")
        return FakeHFViT(dims[model_id])

    monkeypatch.setitem(
        sys.modules, "transformers",
        SimpleNamespace(AutoModel=SimpleNamespace(
            from_pretrained=from_pretrained)))


def test_base_and_large_route_to_their_checkpoints(monkeypatch):
    cfg = load_config()
    captured = []
    dims = {"facebook/dinov3-vitb16-pretrain-lvd1689m": 768,
            "facebook/dinov3-vitl16-pretrain-lvd1689m": 1024}
    _install_fake_transformers(monkeypatch, dims, captured)

    b = DINOv3Backbone(dict(cfg["models"]["dinov3_vitb_frozen"]))
    assert captured[-1] == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert b.feature_dim == 768
    l = DINOv3Backbone(dict(cfg["models"]["dinov3_vitl_frozen"]))
    assert captured[-1] == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert l.feature_dim == 1024
    assert l.backbone_meta["model_id"].endswith("vitl16-pretrain-lvd1689m")


def test_gated_access_raises_backbone_unavailable(monkeypatch):
    cfg = load_config()
    _install_fake_transformers(monkeypatch, dims={}, captured=[])
    with pytest.raises(BackboneUnavailableError, match="gated"):
        DINOv3Backbone(dict(cfg["models"]["dinov3_vitl_frozen"]))


def test_lora_config_targets_are_identical_across_sizes():
    cfg = load_config()
    vitb = cfg["models"]["dinov3_vitb_lora"]["lora"]["target_modules"]
    vitl = cfg["models"]["dinov3_vitl_lora"]["lora"]["target_modules"]
    assert vitb == vitl == ["q_proj", "v_proj"]
