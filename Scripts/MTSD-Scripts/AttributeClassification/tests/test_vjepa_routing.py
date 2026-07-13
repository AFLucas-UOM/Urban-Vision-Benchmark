"""V-JEPA 2.1 Base/Large routing, version strictness, and legacy fallback."""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mtsd_attr.backbones import BackboneUnavailableError, vjepa_backbone
from mtsd_attr.backbones.vjepa_backbone import VJEPABackbone
from mtsd_attr.config import load_config


class FakeEncoder(nn.Module):
    """Mimics Meta's hub encoder: (B, C, T, H, W) clip -> (B, N, D) tokens."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(1, dim)

    def forward(self, clip):
        b = clip.shape[0]
        return torch.zeros(b, 7, self.dim)


def _hub_cfg(variant):
    return dict(load_config()["models"][variant])


def test_vjepa21_base_and_large_entrypoint_routing(monkeypatch):
    calls = []

    def fake_hub_load(repo, entrypoint, trust_repo=True):
        calls.append((repo, entrypoint))
        dim = 768 if "base" in entrypoint else 1024
        return FakeEncoder(dim), object()  # (encoder, predictor) tuple

    monkeypatch.setattr(torch.hub, "load", fake_hub_load)
    monkeypatch.setattr(vjepa_backbone, "_ensure_hub_checkpoint",
                        lambda url: None)

    for variant, entrypoint, dim in (
            ("vjepa21_vitb_frozen", "vjepa2_1_vit_base_384", 768),
            ("vjepa21_vitl_frozen", "vjepa2_1_vit_large_384", 1024)):
        backbone = VJEPABackbone(_hub_cfg(variant))
        assert calls[-1] == ("facebookresearch/vjepa2", entrypoint)
        assert backbone.backend == "torch_hub"
        assert backbone.image_size == 384
        assert backbone.feature_dim == dim
        meta = backbone.backbone_meta
        assert meta["requested_version"] == "2.1"
        assert meta["loaded_version"] == "2.1"
        assert meta["entrypoint"] == entrypoint
        assert meta["allow_backend_fallback"] is False


def test_vjepa21_fallback_prohibited(monkeypatch):
    def failing_hub_load(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(torch.hub, "load", failing_hub_load)
    monkeypatch.setattr(vjepa_backbone, "_ensure_hub_checkpoint",
                        lambda url: None)

    # transformers must never be touched on the strict path.
    def _explode(*args, **kwargs):
        raise AssertionError("transformers fallback was used")

    monkeypatch.setitem(
        sys.modules, "transformers",
        SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=_explode)))

    with pytest.raises(BackboneUnavailableError,
                       match="allow_backend_fallback is false"):
        VJEPABackbone(_hub_cfg("vjepa21_vitb_frozen"))
    with pytest.raises(BackboneUnavailableError):
        VJEPABackbone(_hub_cfg("vjepa21_vitl_lora"))


def test_legacy_vjepa_still_falls_back_to_transformers(monkeypatch):
    monkeypatch.setattr(torch.hub, "load",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    monkeypatch.setattr(vjepa_backbone, "_ensure_hub_checkpoint",
                        lambda url: None)

    class FakeHFVideoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Linear(1, 1)

        def forward(self, pixel_values_videos):
            b = pixel_values_videos.shape[0]
            return SimpleNamespace(last_hidden_state=torch.zeros(b, 5, 1024))

        def eval(self):
            return self

    captured = {}

    def fake_from_pretrained(model_id):
        captured["model_id"] = model_id
        return FakeHFVideoModel()

    monkeypatch.setitem(
        sys.modules, "transformers",
        SimpleNamespace(AutoModel=SimpleNamespace(
            from_pretrained=fake_from_pretrained)))

    # Legacy config: no allow_backend_fallback key -> fallback allowed.
    backbone = VJEPABackbone(_hub_cfg("vjepa"))
    assert backbone.backend == "transformers"
    assert backbone.image_size == 256
    assert captured["model_id"] == "facebook/vjepa2-vitl-fpc64-256"
    meta = backbone.backbone_meta
    assert meta["loaded_version"] == "2.0"
    assert meta["requested_version"] == "2.1"
    assert meta["allow_backend_fallback"] is True


def test_legacy_vjepa_lora_never_adapts_the_fallback(monkeypatch):
    monkeypatch.setattr(torch.hub, "load",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    monkeypatch.setattr(vjepa_backbone, "_ensure_hub_checkpoint",
                        lambda url: None)
    with pytest.raises(BackboneUnavailableError, match="LoRA"):
        VJEPABackbone(_hub_cfg("vjepa_lora"))
