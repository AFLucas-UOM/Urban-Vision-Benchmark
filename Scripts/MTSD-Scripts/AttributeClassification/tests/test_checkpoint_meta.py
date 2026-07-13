"""Checkpoint state round-trips and old-checkpoint metadata compatibility."""

import torch
from torch import nn

from mtsd_attr.multihead_model import MultiHeadClassifier
from mtsd_attr.train_common import (_checkpoint_adaptation, _checkpoint_state,
                                    _load_checkpoint_state)
from mtsd_attr.variants import variant_metadata

ATTRIBUTES = {
    "view_angle": {"classes": ["Front", "Back", "Side"]},
    "mounting": {"classes": ["Pole-Mounted", "Wall-Mounted"]},
}
PROBE = {"type": "linear", "mlp_hidden_dim": 8, "mlp_dropout": 0.0}


class FakeBackbone(nn.Module):
    def __init__(self, dim=16, lora=False):
        super().__init__()
        self.feature_dim = dim
        self.image_size = 32
        self.adaptation = "lora" if lora else "frozen"
        self.base = nn.Linear(dim, dim)
        if lora:
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.lora_A = nn.Linear(dim, 2, bias=False)
            self.lora_B = nn.Linear(2, dim, bias=False)

    def forward(self, x):
        return x


def test_frozen_checkpoint_roundtrip():
    model = MultiHeadClassifier(FakeBackbone(), ATTRIBUTES, PROBE, "frozen")
    state = _checkpoint_state(model)
    assert all(not k.startswith("backbone.") for k in state)
    assert any(k.startswith("heads.") for k in state)
    fresh = MultiHeadClassifier(FakeBackbone(), ATTRIBUTES, PROBE, "frozen")
    _load_checkpoint_state(fresh, state, "frozen")
    for key, value in state.items():
        assert torch.equal(fresh.state_dict()[key], value)


def test_lora_checkpoint_roundtrip_keeps_adapters_only():
    model = MultiHeadClassifier(FakeBackbone(lora=True), ATTRIBUTES, PROBE,
                                "lora")
    state = _checkpoint_state(model)
    backbone_keys = [k for k in state if k.startswith("backbone.")]
    assert backbone_keys and all("lora_" in k for k in backbone_keys)
    fresh = MultiHeadClassifier(FakeBackbone(lora=True), ATTRIBUTES, PROBE,
                                "lora")
    _load_checkpoint_state(fresh, state, "lora")
    for key, value in state.items():
        assert torch.equal(fresh.state_dict()[key], value)


def test_old_checkpoint_metadata_compatibility():
    # Old checkpoints: no variant_meta/backbone_meta, legacy model_cfg without
    # family/architecture/model_size keys, legacy boolean "frozen".
    old_ckpt = {
        "variant": "convnext",
        "run_id": "convnext-20260703-120000",
        "frozen": False,
        "model_cfg": {"backbone": "convnext", "adaptation": "finetune",
                      "image_size": 224},
        "model_state": {},
    }
    assert _checkpoint_adaptation(old_ckpt, old_ckpt["model_cfg"]) == "finetune"
    meta = variant_metadata(old_ckpt["model_cfg"], old_ckpt["variant"])
    assert meta["family"] == "ConvNeXt"
    assert meta["model_size"] == "tiny"  # historical ConvNeXt was Tiny
    assert meta["legacy"] is False or isinstance(meta["legacy"], bool)

    pre_adaptation_cfg = {"backbone": "dinov3", "frozen": True,
                          "hf_model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                          "image_size": 224}
    assert _checkpoint_adaptation({"model_state": {}}, pre_adaptation_cfg) \
        == "frozen"
    meta = variant_metadata(pre_adaptation_cfg)
    assert (meta["architecture"], meta["model_size"]) == ("ViT-B/16", "base")


def test_new_checkpoint_metadata_fields_roundtrip(tmp_path):
    model = MultiHeadClassifier(FakeBackbone(), ATTRIBUTES, PROBE, "frozen")
    ckpt = {
        "variant": "lingbot_vitb_frozen",
        "adaptation": "frozen",
        "model_cfg": {"backbone": "lingbot", "family": "LingBot-Vision",
                      "architecture": "ViT-B/16", "model_size": "base",
                      "lingbot_variant": "base", "pooling": "mean_patch",
                      "adaptation": "frozen", "image_size": 224,
                      "hf_model_id": "robbyant/lingbot-vision-vit-base"},
        "variant_meta": {"family": "LingBot-Vision", "architecture": "ViT-B/16",
                         "model_size": "base", "adaptation": "frozen",
                         "resolution": 224},
        "backbone_meta": {"loaded_backend": "lingbot_vision",
                          "feature_dim": 768},
        "training_meta": {"physical_batch_size": 32,
                          "gradient_accumulation_steps": 1,
                          "effective_batch_size": 32},
        "model_state": _checkpoint_state(model),
    }
    path = tmp_path / "best.pt"
    torch.save(ckpt, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded["variant_meta"]["architecture"] == "ViT-B/16"
    assert loaded["training_meta"]["effective_batch_size"] == 32
    meta = variant_metadata(loaded["model_cfg"], loaded["variant"])
    assert meta["family"] == "LingBot-Vision"
    fresh = MultiHeadClassifier(FakeBackbone(), ATTRIBUTES, PROBE, "frozen")
    _load_checkpoint_state(fresh, loaded["model_state"], "frozen")
