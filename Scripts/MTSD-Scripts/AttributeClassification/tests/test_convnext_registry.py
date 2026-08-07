"""ConvNeXt size registry: constructor/dim routing without downloading weights."""

import pytest
from torch import nn

from mtsd_attr.backbones import convnext_backbone
from mtsd_attr.backbones.convnext_backbone import (CONVNEXT_REGISTRY,
                                                   ConvNeXtBackbone)


def test_registry_maps_sizes_to_official_constructors():
    assert set(CONVNEXT_REGISTRY) == {"tiny", "base", "large"}
    for size, expected_dim in (("tiny", 768), ("base", 1024), ("large", 1536)):
        constructor, weights, dim = CONVNEXT_REGISTRY[size]
        assert constructor.__name__ == f"convnext_{size}"
        assert "IMAGENET1K_V1" in str(weights)
        assert dim == expected_dim


def _fake_convnext(feature_dim):
    class FakeConvNeXt(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Conv2d(3, feature_dim, 1)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Flatten(), nn.Linear(feature_dim, 10))

    def constructor(weights=None):
        return FakeConvNeXt()

    return constructor


@pytest.mark.parametrize("size,dim", [("tiny", 768), ("base", 1024),
                                      ("large", 1536)])
def test_backbone_selects_size_and_derives_feature_dim(monkeypatch, size, dim):
    monkeypatch.setitem(convnext_backbone.CONVNEXT_REGISTRY, size,
                        (_fake_convnext(dim), None, dim))
    backbone = ConvNeXtBackbone({"backbone": "convnext", "model_size": size,
                                 "adaptation": "frozen", "image_size": 224})
    assert backbone.feature_dim == dim
    assert backbone.model_size == size
    assert backbone.backbone_meta["model_id"] == f"torchvision convnext_{size}"


def test_old_checkpoint_cfg_without_model_size_defaults_to_tiny(monkeypatch):
    monkeypatch.setitem(convnext_backbone.CONVNEXT_REGISTRY, "tiny",
                        (_fake_convnext(768), None, 768))
    backbone = ConvNeXtBackbone({"backbone": "convnext",
                                 "adaptation": "finetune", "image_size": 224})
    assert backbone.model_size == "tiny" and backbone.feature_dim == 768


def test_registry_dim_mismatch_fails(monkeypatch):
    monkeypatch.setitem(convnext_backbone.CONVNEXT_REGISTRY, "base",
                        (_fake_convnext(999), None, 1024))
    with pytest.raises(RuntimeError, match="does not match"):
        ConvNeXtBackbone({"backbone": "convnext", "model_size": "base",
                          "adaptation": "frozen", "image_size": 224})


def test_unknown_size_fails_clearly():
    with pytest.raises(ValueError, match="model_size"):
        ConvNeXtBackbone({"backbone": "convnext", "model_size": "small",
                          "adaptation": "frozen", "image_size": 224})
