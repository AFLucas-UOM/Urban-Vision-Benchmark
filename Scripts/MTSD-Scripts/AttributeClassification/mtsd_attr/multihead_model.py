"""Shared multi-head classifier and masked multi-task loss.

One backbone produces a pooled feature vector; one independent classification
head per attribute maps it to that attribute's class logits. Frozen backbones
keep requires_grad=False and stay in eval() mode even while the heads train.
"""

import torch
from torch import nn

from .dataset import MISSING_LABEL


def build_head(feature_dim, n_classes, probe_cfg):
    """Build one classification head according to the probe config.

    Args:
        feature_dim: Backbone feature dimensionality.
        n_classes: Size of this attribute's class vocabulary.
        probe_cfg: The probe: block of the config (type, mlp_hidden_dim, mlp_dropout).

    Returns:
        nn.Module mapping (B, feature_dim) -> (B, n_classes).
    """
    if probe_cfg["type"] == "linear":
        return nn.Linear(feature_dim, n_classes)
    if probe_cfg["type"] == "mlp":
        return nn.Sequential(
            nn.Linear(feature_dim, probe_cfg["mlp_hidden_dim"]),
            nn.GELU(),
            nn.Dropout(probe_cfg["mlp_dropout"]),
            nn.Linear(probe_cfg["mlp_hidden_dim"], n_classes),
        )
    raise ValueError(f"Unknown probe type {probe_cfg['type']!r}")


class MultiHeadClassifier(nn.Module):
    """Backbone -> pooled features -> N independent per-attribute heads.

    Args:
        backbone: Module with .feature_dim, mapping (B, 3, H, W) -> (B, D).
        attributes: Config mapping attribute name -> {"classes": [...]}.
        probe_cfg: Head architecture config.
        adaptation: "frozen" (backbone excluded from gradients and pinned to
            eval()), "lora" (base weights frozen by the backbone wrapper, LoRA
            adapters and heads trainable), or "finetune" (everything trains).
    """

    def __init__(self, backbone, attributes, probe_cfg, adaptation):
        super().__init__()
        self.backbone = backbone
        self.adaptation = adaptation
        self.frozen = adaptation == "frozen"
        self.attribute_names = list(attributes)
        self.heads = nn.ModuleDict({
            attr: build_head(backbone.feature_dim, len(spec["classes"]), probe_cfg)
            for attr, spec in attributes.items()
        })
        if self.frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        elif adaptation == "lora":
            trainable = [n for n, p in self.backbone.named_parameters()
                         if p.requires_grad]
            if not trainable or any("lora_" not in n for n in trainable):
                raise RuntimeError(
                    "LoRA adaptation expects exactly the adapter parameters to "
                    f"be trainable in the backbone; found {trainable[:5]}...")

    def train(self, mode=True):
        """Standard train/eval switching, except a frozen backbone stays in eval()."""
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, x):
        """Return a dict attribute name -> (B, n_classes) logits."""
        if self.frozen:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        return {attr: self.heads[attr](features) for attr in self.attribute_names}

    def trainable_parameters(self):
        """Yield (name, parameter) pairs that require gradients."""
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]


def parameter_breakdown(model):
    """Count parameters by role, making the adaptation regimes explicit.

    Returns:
        dict with total, trainable, trainable_pct, backbone_total,
        backbone_trainable, head_trainable, and lora_trainable counts.
    """
    total = trainable = 0
    backbone_total = backbone_trainable = 0
    head_trainable = lora_trainable = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if name.startswith("backbone."):
            backbone_total += n
        if p.requires_grad:
            trainable += n
            if name.startswith("backbone."):
                backbone_trainable += n
            if name.startswith("heads."):
                head_trainable += n
            if "lora_" in name:
                lora_trainable += n
    return {
        "total": total,
        "trainable": trainable,
        "trainable_pct": round(100.0 * trainable / max(1, total), 4),
        "backbone_total": backbone_total,
        "backbone_trainable": backbone_trainable,
        "head_trainable": head_trainable,
        "lora_trainable": lora_trainable,
    }


class MaskedMultiTaskLoss(nn.Module):
    """Weighted sum of per-head cross-entropies with missing labels masked out.

    Targets use MISSING_LABEL (-1) for crops without a label on a head; those
    entries contribute nothing to that head's loss. A head whose entire batch
    is missing is skipped for that step.

    Args:
        attributes: Config mapping attribute name -> {"classes": [...]}.
        head_weights: Mapping attribute name -> loss weight (defaults to 1.0).
        class_weights: Optional mapping attribute name -> per-class weight tensor.
        label_smoothing: Passed through to cross-entropy.
    """

    def __init__(self, attributes, head_weights=None, class_weights=None,
                 label_smoothing=0.0):
        super().__init__()
        self.attribute_names = list(attributes)
        head_weights = head_weights or {}
        self.head_weights = {a: float(head_weights.get(a, 1.0))
                             for a in self.attribute_names}
        self.criteria = nn.ModuleDict()
        for attr in self.attribute_names:
            weight = None
            if class_weights and class_weights.get(attr) is not None:
                weight = class_weights[attr]
            self.criteria[attr] = nn.CrossEntropyLoss(
                weight=weight,
                ignore_index=MISSING_LABEL,
                label_smoothing=label_smoothing,
            )

    def forward(self, logits, targets):
        """Compute the joint loss.

        Args:
            logits: dict attribute name -> (B, n_classes) tensor.
            targets: (B, n_attributes) long tensor with MISSING_LABEL for gaps.

        Returns:
            (total_loss tensor, per_head dict of detached float losses)
        """
        total = None
        per_head = {}
        for i, attr in enumerate(self.attribute_names):
            t = targets[:, i]
            if (t != MISSING_LABEL).sum() == 0:
                continue
            loss = self.criteria[attr](logits[attr], t)
            per_head[attr] = loss.detach().item()
            weighted = self.head_weights[attr] * loss
            total = weighted if total is None else total + weighted
        if total is None:
            total = next(iter(logits.values())).sum() * 0.0
        return total, per_head
