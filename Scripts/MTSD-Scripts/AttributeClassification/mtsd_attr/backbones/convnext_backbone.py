"""ConvNeXt-Tiny backbone (ImageNet-pretrained).

Used in two variants: fully fine-tuned (adaptation: finetune, the supervised
task-adapted baseline) and frozen linear probe (adaptation: frozen, the
controlled frozen-representation comparison against DINOv3 and V-JEPA). Both
share the same weights, feature extraction, and 768-d pooled representation.
"""

import logging

from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

log = logging.getLogger("mtsd_attr")


class ConvNeXtBackbone(nn.Module):
    """Wraps torchvision's ConvNeXt-Tiny into the shared backbone interface."""

    def __init__(self, model_cfg):
        super().__init__()
        from ..config import adaptation_of

        self.adaptation = adaptation_of(model_cfg)
        if self.adaptation == "lora":
            raise ValueError("LoRA is not implemented for ConvNeXt; use "
                             "adaptation: frozen or finetune")
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.features = model.features
        self.avgpool = model.avgpool
        self.image_size = model_cfg["image_size"]
        self.feature_dim = 768
        log.info("ConvNeXt-Tiny backbone loaded (adaptation=%s, feature_dim=%d)",
                 self.adaptation, self.feature_dim)

    def forward(self, x):
        """Map (B, 3, H, W) images to (B, 768) pooled features."""
        return self.avgpool(self.features(x)).flatten(1)
