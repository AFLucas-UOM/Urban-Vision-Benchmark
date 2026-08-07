"""LoRA adapter injection shared by the DINOv3 and V-JEPA backbones.

Uses peft's inject_adapter_in_model, which operates on any nn.Module tree, so
the same helper serves both Hugging Face models and Meta's torch.hub
VisionTransformer. Base weights are frozen first; after injection only the
lora_A/lora_B adapter parameters require gradients. The helper verifies that
every requested target module actually matched, and fails loudly otherwise —
a LoRA variant must never silently degrade to frozen or full fine-tuning.
"""

import logging
import warnings

from torch import nn

log = logging.getLogger("mtsd_attr")


def apply_lora(model, lora_cfg, context):
    """Freeze a model's base parameters and inject LoRA adapters in place.

    Args:
        model: The backbone module to adapt (modified in place).
        lora_cfg: dict with r, alpha, dropout, and target_modules (a list of
            module-name suffixes to adapt, e.g. ["q_proj", "v_proj"]).
        context: Short human-readable label for error messages (e.g. "DINOv3").

    Returns:
        The adapted model.

    Raises:
        RuntimeError: If any requested target matches no module, or if the
            injection produced no trainable adapter parameters.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Importing from .* is deprecated, please import via timm\.layers",
            category=FutureWarning,
        )
        from peft import LoraConfig, inject_adapter_in_model

    targets = list(lora_cfg.get("target_modules") or [])
    if not targets:
        raise RuntimeError(f"{context}: lora.target_modules must be a "
                           f"non-empty list of module-name suffixes")
    for p in model.parameters():
        p.requires_grad = False

    config = LoraConfig(
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        target_modules=targets,
        bias="none",
    )
    model = inject_adapter_in_model(config, model)

    matched = {t: 0 for t in targets}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            suffix = name.split(".")[-1]
            for target in targets:
                if suffix == target or name.endswith(target):
                    matched[target] += 1
    unmatched = [t for t, n in matched.items() if n == 0]
    if unmatched:
        available = sorted({
            name.split(".")[-1] for name, m in model.named_modules()
            if isinstance(m, nn.Linear)
        })
        raise RuntimeError(
            f"{context}: LoRA target modules {unmatched} matched nothing in "
            f"the loaded architecture. Available Linear module suffixes: "
            f"{available}. Fix lora.target_modules in the config."
        )

    lora_params = sum(p.numel() for n, p in model.named_parameters()
                      if "lora_" in n and p.requires_grad)
    if lora_params == 0:
        raise RuntimeError(f"{context}: LoRA injection produced no trainable "
                           f"adapter parameters")
    log.info("%s: LoRA injected (r=%d, alpha=%d, dropout=%.3f) on %s -> "
             "%d adapted layers, %.3fM adapter parameters",
             context, config.r, config.lora_alpha, config.lora_dropout,
             targets, sum(matched.values()), lora_params / 1e6)
    return model
