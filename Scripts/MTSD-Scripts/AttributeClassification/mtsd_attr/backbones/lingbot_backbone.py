"""LingBot-Vision backbone (Robbyant / Ant Group), frozen or LoRA.

Loaded through the official inference package (pip: lingbot-vision, import
module lingbot_vision, pinned to a reviewed upstream commit in
Requirements/requirements-attribute-classification.txt). LingBot is NOT a
standard transformers AutoModel checkpoint: ``load_pretrained_backbone``
returns a ``(model, embed_dim)`` tuple — the model already frozen in eval
mode — and the model's forward with ``is_training=True`` returns the token
dictionary (``x_norm_clstoken`` / ``x_storage_tokens`` /
``x_norm_patchtokens``), not a tensor.

Pooling: the shared interface needs one vector per image. The default
``pooling: mean_patch`` mean-pools the normalised patch tokens
(``x_norm_patchtokens``), the dense representation the model card recommends
for frozen readouts; ``pooling: cls`` uses the normalised CLS token instead.

Adaptation modes: "frozen" or "lora" (adapters on the fused per-block ``qkv``
projection, a ``LinearKMaskedBias`` subclass of nn.Linear — verified against
lingbot_vision/layers.py at the pinned commit). Full fine-tuning is
deliberately not implemented for LingBot in this experiment.

The ViT uses RoPE positional embeddings, so inputs need not match the native
512 px training resolution; the configured image_size (default 224) is used.
"""

import logging

import torch
from torch import nn

from . import BackboneUnavailableError
from .lora import apply_lora

log = logging.getLogger("mtsd_attr")

POOLING_MODES = ("mean_patch", "cls")


def _import_lingbot():
    """Import the official LingBot-Vision inference package."""
    try:
        import lingbot_vision
        return lingbot_vision
    except ImportError:
        pass
    try:  # Older README drafts referenced this alias; accept it if present.
        import lbot_vision_infer
        return lbot_vision_infer
    except ImportError as exc:
        raise BackboneUnavailableError(
            "The LingBot-Vision inference package is not installed in this "
            "environment. Install the pinned dependency with:\n"
            "  pip install -r Requirements/requirements-attribute-"
            "classification.txt\n"
            "(package lingbot-vision, pinned to a specific commit of "
            "https://github.com/robbyant/lingbot-vision). "
            f"Original error: {exc}"
        ) from exc


class LingBotBackbone(nn.Module):
    """Wraps a LingBot-Vision ViT into the shared backbone interface."""

    def __init__(self, model_cfg):
        super().__init__()
        from ..config import adaptation_of

        self.adaptation = adaptation_of(model_cfg)
        if self.adaptation == "finetune":
            raise ValueError("LingBot-Vision full fine-tuning is deliberately "
                             "not implemented in this experiment; use "
                             "adaptation: frozen or lora")
        variant = str(model_cfg.get("lingbot_variant", "base")).lower()
        if variant not in ("base", "large"):
            raise ValueError(f"Unsupported lingbot_variant {variant!r}; this "
                             f"experiment supports 'base' and 'large'")
        self.pooling = str(model_cfg.get("pooling", "mean_patch"))
        if self.pooling not in POOLING_MODES:
            raise ValueError(f"Unknown LingBot pooling {self.pooling!r}; "
                             f"expected one of {POOLING_MODES}")
        repo_id = model_cfg.get("hf_model_id")
        self.image_size = int(model_cfg["image_size"])

        lbv = _import_lingbot()
        try:
            # Weights stay fp32; bf16 compute comes from the training loop's
            # autocast, exactly like the other backbones.
            loaded = lbv.load_pretrained_backbone(
                repo_id_or_path=repo_id,
                variant=variant,
                device="cpu",
                dtype="fp32",
                verbose=False,
            )
        except BackboneUnavailableError:
            raise
        except Exception as exc:
            raise BackboneUnavailableError(
                f"LingBot-Vision checkpoint {repo_id or variant!r} could not "
                f"be loaded (download/access/config failure). Check network "
                f"access to Hugging Face and that the repo id is correct "
                f"(robbyant/lingbot-vision-vit-base / -vit-large). "
                f"Original error: {exc}"
            ) from exc
        # The official loader returns a (model, embed_dim) tuple; verify the
        # exact upstream return type instead of assuming it.
        if (not isinstance(loaded, tuple) or len(loaded) != 2
                or not isinstance(loaded[0], nn.Module)):
            raise BackboneUnavailableError(
                "lingbot_vision.load_pretrained_backbone returned "
                f"{type(loaded).__name__!r} instead of the documented "
                "(model, embed_dim) tuple; the installed lingbot-vision "
                "version does not match the pinned interface.")
        model, embed_dim = loaded
        self.model = model
        self.feature_dim = int(embed_dim)
        if self.adaptation == "lora":
            # The loader ships the model frozen; apply_lora re-freezes the
            # base weights and injects trainable adapters on the fused qkv.
            self.model = apply_lora(self.model, model_cfg["lora"],
                                    "LingBot-Vision")
        self.model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            probed = self.forward(dummy).shape[-1]
        if probed != self.feature_dim:
            raise RuntimeError(
                f"LingBot-Vision pooled feature dim {probed} does not match "
                f"the loader-reported embed_dim {self.feature_dim}")
        self.backbone_meta = {
            "requested_backend": "lingbot_vision",
            "loaded_backend": "lingbot_vision",
            "model_id": repo_id or f"lingbot-vision {variant}",
            "lingbot_variant": variant,
            "pooling": self.pooling,
            "image_size": self.image_size,
            "feature_dim": self.feature_dim,
        }
        log.info("LingBot-Vision %s backbone loaded (adaptation=%s, "
                 "pooling=%s, feature_dim=%d)",
                 variant, self.adaptation, self.pooling, self.feature_dim)

    def _pool(self, outputs):
        """Reduce the upstream token dictionary to one vector per image."""
        if not isinstance(outputs, dict) or "x_norm_patchtokens" not in outputs:
            raise RuntimeError(
                "LingBot-Vision forward(is_training=True) did not return the "
                "documented token dictionary (missing x_norm_patchtokens); "
                f"got {type(outputs).__name__}. The installed lingbot-vision "
                "version does not match the pinned interface.")
        if self.pooling == "cls":
            return outputs["x_norm_clstoken"]
        return outputs["x_norm_patchtokens"].mean(dim=1)

    def forward(self, x):
        """Map (B, 3, H, W) images to (B, feature_dim) pooled features."""
        return self._pool(self.model(x, is_training=True))
