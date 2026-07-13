"""V-JEPA backbone: V-JEPA 2.1 via torch.hub or V-JEPA 2.0 via transformers.

V-JEPA is a video model, so a single image is repeated along the temporal axis
to form a pseudo-clip (the pattern documented on the model card). As of
2026-07, V-JEPA 2.1 checkpoints are only distributed through Meta's GitHub
repository via torch.hub (official entry points: vjepa2_1_vit_base_384,
vjepa2_1_vit_large_384, ...; each returns an (encoder, predictor) tuple).

Version policy:
- New size-ablation variants set allow_backend_fallback: false — if the
  requested 2.1 hub load fails, a BackboneUnavailableError is raised. A silent
  substitution of the V-JEPA 2.0 transformers checkpoint would be a
  cross-version swap that invalidates the size ablation.
- LEGACY BEHAVIOUR: the historical vjepa / vjepa_lora variants (and old
  checkpoints, whose stored model_cfg has no allow_backend_fallback key) keep
  the original fallback-with-warning to V-JEPA 2.0 via transformers, so they
  remain loadable exactly as before.
The requested and actually loaded backend/version are recorded in
backbone_meta for checkpoints, metrics, reports, and W&B.

Note: Meta's vjepa2 repo contains a top-level package named "src", which is why
this subproject's own package is named mtsd_attr; a second "src" package in the
same process would shadow it and break the torch.hub import.
"""

import logging
from pathlib import Path

import torch
from torch import nn

from . import BackboneUnavailableError
from .lora import apply_lora

log = logging.getLogger("mtsd_attr")


def _ensure_hub_checkpoint(url):
    """Pre-cache a checkpoint file under torch.hub's checkpoints directory.

    Meta's vjepa2 hubconf currently ships with a localhost placeholder base URL
    (the real https://dl.fbaipublicfiles.com/vjepa2 is commented out upstream).
    torch.hub.load_state_dict_from_url reuses an already-cached file with the
    same basename, so downloading the real file first makes the hub load
    succeed without patching Meta's code.
    """
    target = Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[-1]
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading V-JEPA 2.1 checkpoint from %s (several GB)", url)
    torch.hub.download_url_to_file(url, str(target), progress=True)


class VJEPABackbone(nn.Module):
    """Wraps V-JEPA 2.0/2.1 into the shared backbone interface via mean token pooling.

    Adaptation modes: "frozen" (default) or "lora", which freezes the base
    weights and injects adapters into the fused qkv attention projections
    (Meta's VisionTransformer has no separate q/v Linears, so the fused qkv is
    the architecture-equivalent target; q, k and v are adapted jointly).
    """

    def __init__(self, model_cfg):
        super().__init__()
        from ..config import adaptation_of

        requested = model_cfg.get("backend", "torch_hub")
        # LEGACY BEHAVIOUR: fallback defaults to allowed when the key is
        # absent so old checkpoints/configs keep loading; the new vjepa21_*
        # variants set allow_backend_fallback: false explicitly.
        allow_fallback = bool(model_cfg.get("allow_backend_fallback", True))
        requested_version = str(model_cfg.get(
            "vjepa_version", "2.1" if requested == "torch_hub" else "2.0"))
        frames = int(model_cfg.get("num_frames", 2))
        self.num_frames = max(2, frames - frames % 2)
        self.adaptation = adaptation_of(model_cfg)
        if self.adaptation == "finetune":
            raise ValueError("V-JEPA full fine-tuning is not part of this "
                             "experiment; use adaptation: frozen or lora")
        self.backend = None
        loaded_version = None
        entrypoint = model_cfg.get("torch_hub_entrypoint")
        if requested == "torch_hub":
            try:
                if model_cfg.get("torch_hub_checkpoint_url"):
                    _ensure_hub_checkpoint(model_cfg["torch_hub_checkpoint_url"])
                loaded = torch.hub.load(
                    model_cfg["torch_hub_repo"],
                    entrypoint,
                    trust_repo=True,
                )
                # V-JEPA 2.1 entry points return (encoder, predictor).
                self.model = loaded[0] if isinstance(loaded, tuple) else loaded
                self.backend = "torch_hub"
                loaded_version = requested_version
                self.image_size = model_cfg["image_size_torch_hub"]
                log.info("V-JEPA %s loaded via torch.hub (%s)",
                         loaded_version, entrypoint)
            except Exception as exc:
                if not allow_fallback:
                    raise BackboneUnavailableError(
                        f"V-JEPA {requested_version} could not be loaded via "
                        f"torch.hub ({model_cfg.get('torch_hub_repo')}:"
                        f"{entrypoint}) and allow_backend_fallback is false "
                        f"for this variant — substituting the V-JEPA 2.0 "
                        f"transformers checkpoint would silently change the "
                        f"model version and invalidate the size ablation. Fix "
                        f"the hub load (network/cache/dependencies) or change "
                        f"the config explicitly. Original error: {exc}"
                    ) from exc
                log.warning(
                    "torch.hub load of V-JEPA 2.1 failed (%s); LEGACY "
                    "fallback to V-JEPA 2.0 via transformers (%s)",
                    exc, model_cfg.get("hf_model_id"),
                )
        if self.backend is None:
            if self.adaptation == "lora" and requested == "torch_hub":
                raise BackboneUnavailableError(
                    "V-JEPA LoRA is configured against the torch.hub 2.1 "
                    "architecture (fused qkv target modules); the hub load "
                    "failed and the transformers 2.0 fallback has different "
                    "module names, so LoRA cannot proceed. Fix the hub load "
                    "or reconfigure the variant explicitly for the "
                    "transformers backend."
                )
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(model_cfg["hf_model_id"])
            self.backend = "transformers"
            loaded_version = "2.0"
            self.image_size = model_cfg["image_size_transformers"]
            log.info("V-JEPA 2.0 loaded via transformers (%s)",
                     model_cfg["hf_model_id"])
        if self.adaptation == "lora":
            self.model = apply_lora(self.model, model_cfg["lora"],
                                    f"V-JEPA ({self.backend})")
        self.model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            self.feature_dim = self.forward(dummy).shape[-1]
        self.backbone_meta = {
            "requested_backend": requested,
            "loaded_backend": self.backend,
            "requested_version": requested_version,
            "loaded_version": loaded_version,
            "allow_backend_fallback": allow_fallback,
            "entrypoint": entrypoint if self.backend == "torch_hub" else None,
            "checkpoint": (model_cfg.get("torch_hub_checkpoint_url")
                           if self.backend == "torch_hub"
                           else model_cfg.get("hf_model_id")),
            "model_id": (entrypoint if self.backend == "torch_hub"
                         else model_cfg.get("hf_model_id")),
            "image_size": self.image_size,
            "feature_dim": self.feature_dim,
            "num_frames": self.num_frames,
        }
        log.info("V-JEPA backbone ready (backend=%s, version=%s, "
                 "num_frames=%d, feature_dim=%d)",
                 self.backend, loaded_version, self.num_frames,
                 self.feature_dim)

    def forward(self, x):
        """Map (B, 3, H, W) images to (B, feature_dim) via a repeated pseudo-clip."""
        if self.backend == "torch_hub":
            clip = x.unsqueeze(2).repeat(1, 1, self.num_frames, 1, 1)
            tokens = self.model(clip)
        else:
            clip = x.unsqueeze(1).repeat(1, self.num_frames, 1, 1, 1)
            tokens = self.model(pixel_values_videos=clip).last_hidden_state
        return tokens.mean(dim=1)
