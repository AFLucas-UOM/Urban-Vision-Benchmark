"""DINOv3 backbone loaded from Hugging Face transformers (frozen or LoRA).

DINOv3 checkpoints (facebook/dinov3-*-pretrain-lvd1689m) are gated: access must
be requested and granted on Hugging Face first. If the checkpoint cannot be
downloaded, a BackboneUnavailableError with instructions is raised so callers
can skip this variant instead of crashing the whole run.

Adaptation modes: "frozen" (default, probe use) or "lora", which freezes the
base weights and injects LoRA adapters into the per-block query and value
projections (DINOv3ViTModel exposes separate q_proj/k_proj/v_proj/o_proj
Linear layers per attention block).
"""

import logging

import torch
from torch import nn

from . import BackboneUnavailableError
from .lora import apply_lora

log = logging.getLogger("mtsd_attr")


class DINOv3Backbone(nn.Module):
    """Wraps a Hugging Face DINOv3 model into the shared backbone interface."""

    def __init__(self, model_cfg):
        super().__init__()
        from ..config import adaptation_of
        from transformers import AutoModel

        model_id = model_cfg["hf_model_id"]
        self.image_size = model_cfg["image_size"]
        self.adaptation = adaptation_of(model_cfg)
        if self.adaptation == "finetune":
            raise ValueError("DINOv3 full fine-tuning is not part of this "
                             "experiment; use adaptation: frozen or lora")
        try:
            self.model = AutoModel.from_pretrained(model_id)
        except Exception as exc:
            text = str(exc).lower()
            if any(k in text for k in ("gated", "403", "restricted", "awaiting",
                                       "access to model", "not authorized")):
                raise BackboneUnavailableError(
                    f"DINOv3 checkpoint {model_id!r} is gated and your Hugging "
                    f"Face account does not have access (yet). Request access at "
                    f"https://huggingface.co/{model_id} and ensure you are "
                    f"logged in (hf auth login). Weights are also distributed "
                    f"via https://github.com/facebookresearch/dinov3. "
                    f"Original error: {exc}"
                ) from exc
            raise
        if self.adaptation == "lora":
            self.model = apply_lora(self.model, model_cfg["lora"], "DINOv3")
        self.model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            self.feature_dim = self._pool(self.model(pixel_values=dummy)).shape[-1]
        self.backbone_meta = {
            "requested_backend": "transformers",
            "loaded_backend": "transformers",
            "model_id": model_id,
            "image_size": self.image_size,
            "feature_dim": self.feature_dim,
        }
        log.info("DINOv3 backbone %s loaded (adaptation=%s, feature_dim=%d)",
                 model_id, self.adaptation, self.feature_dim)

    @staticmethod
    def _pool(outputs):
        """Pool model outputs to one vector per image (CLS via pooler when present)."""
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            return pooled
        return outputs.last_hidden_state[:, 0]

    def forward(self, x):
        """Map (B, 3, H, W) images to (B, feature_dim) features."""
        return self._pool(self.model(pixel_values=x))
