"""ConvNeXt backbones (torchvision, ImageNet-1K pretrained), size-configurable.

A validated registry maps the configured model_size to the official
torchvision constructor and its IMAGENET1K_V1 weights. Tiny serves only the
legacy convnext / convnext_frozen variants (the historical six-model round);
Base and Large are the size-ablation variants. The feature dimension is
derived from the built model's classifier head rather than hard-coded, and
cross-checked against the registry's expected value.

Adaptation modes: "frozen" (linear probe), "lora" (pointwise MLP adapters),
or "finetune" (end to end).
"""

import logging

from torch import nn
from torchvision.models import (ConvNeXt_Base_Weights, ConvNeXt_Large_Weights,
                                ConvNeXt_Tiny_Weights, convnext_base,
                                convnext_large, convnext_tiny)

log = logging.getLogger("mtsd_attr")

# model_size -> (constructor, pretrained weights, expected feature dim).
# "tiny" exists only for the legacy variants; the size ablation uses base and
# large. The expected dim is a sanity check — the actual value is derived
# from the instantiated architecture below.
CONVNEXT_REGISTRY = {
    "tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 768),
    "base": (convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1, 1024),
    "large": (convnext_large, ConvNeXt_Large_Weights.IMAGENET1K_V1, 1536),
}


class ConvNeXtBackbone(nn.Module):
    """Wraps a torchvision ConvNeXt into the shared backbone interface."""

    def __init__(self, model_cfg):
        super().__init__()
        from ..config import adaptation_of

        self.adaptation = adaptation_of(model_cfg)
        # Old checkpoints (and the legacy variants) predate the model_size
        # key; they were all ConvNeXt-Tiny.
        size = str(model_cfg.get("model_size", "tiny")).lower()
        if size not in CONVNEXT_REGISTRY:
            raise ValueError(f"Unknown ConvNeXt model_size {size!r}; "
                             f"expected one of {sorted(CONVNEXT_REGISTRY)}")
        constructor, weights, expected_dim = CONVNEXT_REGISTRY[size]
        model = constructor(weights=weights)
        self.features = model.features
        self.avgpool = model.avgpool
        if self.adaptation == "lora":
            from .lora import apply_lora
            self.features = apply_lora(
                self.features, model_cfg.get("lora", {}), "ConvNeXt")
        self.image_size = model_cfg["image_size"]
        # classifier = [LayerNorm2d, Flatten, Linear]; the Linear's input size
        # is the pooled feature dimension of this architecture.
        self.feature_dim = model.classifier[2].in_features
        if self.feature_dim != expected_dim:
            raise RuntimeError(
                f"ConvNeXt-{size} feature dim {self.feature_dim} does not "
                f"match the registry's expected {expected_dim}; torchvision "
                f"architecture changed?")
        self.model_size = size
        self.backbone_meta = {
            "requested_backend": "torchvision",
            "loaded_backend": "torchvision",
            "model_id": f"torchvision convnext_{size}",
            "weights": str(weights),
            "image_size": self.image_size,
            "feature_dim": self.feature_dim,
        }
        log.info("ConvNeXt-%s backbone loaded (adaptation=%s, feature_dim=%d)",
                 size.capitalize(), self.adaptation, self.feature_dim)

    def forward(self, x):
        """Map (B, 3, H, W) images to (B, feature_dim) pooled features."""
        return self.avgpool(self.features(x)).flatten(1)
