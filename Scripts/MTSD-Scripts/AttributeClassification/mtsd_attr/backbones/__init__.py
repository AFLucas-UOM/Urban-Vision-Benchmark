"""Swappable visual backbones sharing a single interface.

Every backbone is an nn.Module exposing:
    feature_dim: int, dimensionality of the pooled feature vector
    image_size: int, expected square input side in pixels
    forward(x): (B, 3, H, W) float tensor -> (B, feature_dim) features
"""


class BackboneUnavailableError(RuntimeError):
    """Raised when a backbone cannot be loaded in this environment.

    Carries user-facing instructions (e.g. how to request access to a gated
    Hugging Face checkpoint) so orchestration can skip the variant cleanly.
    """


def build_backbone(model_cfg):
    """Instantiate the backbone named in a model variant's config block.

    Args:
        model_cfg: One entry of the config's models: mapping.

    Returns:
        A backbone module implementing the interface above.

    Raises:
        BackboneUnavailableError: If the checkpoint is inaccessible (e.g. gated).
    """
    kind = model_cfg["backbone"]
    if kind == "dinov3":
        from .dinov3_backbone import DINOv3Backbone
        return DINOv3Backbone(model_cfg)
    if kind == "vjepa":
        from .vjepa_backbone import VJEPABackbone
        return VJEPABackbone(model_cfg)
    if kind == "convnext":
        from .convnext_backbone import ConvNeXtBackbone
        return ConvNeXtBackbone(model_cfg)
    raise ValueError(f"Unknown backbone {kind!r}")
