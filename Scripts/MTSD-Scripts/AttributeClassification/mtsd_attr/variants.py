"""Variant metadata, experiment profiles, and experiment-plan resolution.

Model family, architecture, size, adaptation, and display information come
from explicit metadata keys on each ``models:`` entry in the config
(``family``, ``architecture``, ``model_size``), never from parsing variant
names or filesystem paths. Checkpoints written before those keys existed
(the historical six-variant round: dinov3, dinov3_lora, vjepa, vjepa_lora,
convnext_frozen, convnext) fall back to per-backbone defaults that match
what those variants actually were.
"""

from .config import adaptation_of, model_training_config

FAMILY_FOR_BACKBONE = {
    "dinov3": "DINOv3",
    "vjepa": "V-JEPA",
    "convnext": "ConvNeXt",
    "lingbot": "LingBot-Vision",
}

# What the historical (pre-metadata) variants actually were; used only as a
# fallback when a model_cfg carries no explicit architecture/model_size keys
# (old checkpoints and old config snapshots).
LEGACY_ARCHITECTURE = {
    "dinov3": ("ViT-B/16", "base"),
    "vjepa": ("ViT-L/16", "large"),
    "convnext": ("ConvNeXt-Tiny", "tiny"),
}

ADAPTATION_LABEL = {"frozen": "frozen", "lora": "LoRA", "finetune": "fine-tuned"}


def variant_resolution(model_cfg):
    """Input resolution for a variant, honouring V-JEPA's per-backend sizes."""
    if "image_size" in model_cfg:
        return int(model_cfg["image_size"])
    backend = model_cfg.get("backend", "torch_hub")
    key = ("image_size_torch_hub" if backend == "torch_hub"
           else "image_size_transformers")
    if key in model_cfg:
        return int(model_cfg[key])
    raise ValueError("Model config has no image_size (or per-backend "
                     f"image_size_*) entry: {sorted(model_cfg)}")


def variant_model_id(model_cfg):
    """The pretrained model identifier: HF repo id, hub entry point, or
    torchvision constructor name."""
    backbone = model_cfg.get("backbone")
    if backbone == "vjepa":
        if model_cfg.get("backend", "torch_hub") == "torch_hub":
            return model_cfg.get("torch_hub_entrypoint", "?")
        return model_cfg.get("hf_model_id", "?")
    if backbone == "convnext":
        size = model_cfg.get("model_size", "tiny")
        return f"torchvision convnext_{size} (IMAGENET1K_V1)"
    return model_cfg.get("hf_model_id", "?")


def variant_metadata(model_cfg, variant=None):
    """Normalised metadata dict for one variant's model config.

    Works both for entries of the current config (which carry explicit
    family/architecture/model_size keys) and for model_cfg dicts stored in
    old checkpoints (which do not); the fallbacks reproduce exactly what the
    historical variants were.
    """
    backbone = model_cfg.get("backbone", "unknown")
    adaptation = adaptation_of(model_cfg)
    family = model_cfg.get("family") or FAMILY_FOR_BACKBONE.get(backbone,
                                                                "Unknown")
    architecture = model_cfg.get("architecture")
    model_size = model_cfg.get("model_size")
    if architecture is None or model_size is None:
        default_arch, default_size = LEGACY_ARCHITECTURE.get(
            backbone, ("unknown", "unknown"))
        architecture = architecture or default_arch
        model_size = model_size or default_size
    display = model_cfg.get("display_name") or (
        f"{family} {architecture} {ADAPTATION_LABEL.get(adaptation, adaptation)}")
    return {
        "variant": variant,
        "family": family,
        "backbone": backbone,
        "architecture": architecture,
        "model_size": model_size,
        "adaptation": adaptation,
        "model_id": variant_model_id(model_cfg),
        "resolution": variant_resolution(model_cfg),
        "display_name": display,
        "legacy": bool(model_cfg.get("legacy", False)),
    }


def available_profiles(cfg):
    """Mapping profile name -> variant list from run_all.profiles."""
    return dict(cfg.get("run_all", {}).get("profiles", {}) or {})


def resolve_profile(cfg, name):
    """Return the validated variant list for a named profile."""
    profiles = available_profiles(cfg)
    if name not in profiles:
        raise ValueError(f"Unknown profile {name!r}; available: "
                         f"{sorted(profiles)}")
    return validate_variants(cfg, list(profiles[name]),
                             context=f"profile {name!r}")


def validate_variants(cfg, names, context="selection"):
    """Fail clearly if any variant name is not defined under models:."""
    available = list(cfg["models"])
    unknown = [v for v in names if v not in available]
    if unknown:
        raise ValueError(f"Unknown variant(s) {unknown} in {context}; "
                         f"available: {available}")
    deduped = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def select_variants(cfg, profile=None, variants=None, include=None):
    """Resolve the variant list for a run.

    Deterministic precedence (documented in the README):
      1. --variants replaces the base set entirely and cannot be combined
         with --profile (ambiguous; raises).
      2. --profile selects a named profile as the base set.
      3. Otherwise the base set is run_all.variants (the legacy default).
      --include appends extra variants (deduplicated) to whichever base set
      was selected in steps 1-3.
    """
    if variants and profile:
        raise ValueError("--variants and --profile are mutually exclusive; "
                         "--variants already replaces the set entirely")
    if variants:
        base = list(variants)
    elif profile:
        base = resolve_profile(cfg, profile)
    else:
        base = list(cfg["run_all"]["variants"])
    base += list(include or [])
    return validate_variants(cfg, base)


def build_plan(cfg, variant_names):
    """Resolve the experiment matrix for the given variants without loading
    any model, refreshing the manifest, or touching W&B.

    Returns a list of row dicts (one per variant) with the metadata, batch
    configuration, and output directory each run would use.
    """
    rows = []
    for variant in variant_names:
        model_cfg, training = model_training_config(cfg, variant)
        meta = variant_metadata(model_cfg, variant)
        rows.append({
            **meta,
            "batch_size": training["batch_size"],
            "gradient_accumulation_steps": training["gradient_accumulation_steps"],
            "effective_batch_size": training["effective_batch_size"],
            "output_dir": str(cfg["paths"]["checkpoints_dir"] / variant),
        })
    return rows


def format_plan(rows, title="Experiment plan"):
    """Plain-text table for --plan / --list-variants output."""
    columns = [
        ("variant", "Variant"),
        ("family", "Family"),
        ("architecture", "Architecture"),
        ("model_size", "Size"),
        ("adaptation", "Adaptation"),
        ("model_id", "Model ID / entry point"),
        ("resolution", "Res"),
        ("batch_size", "BS"),
        ("gradient_accumulation_steps", "Accum"),
        ("effective_batch_size", "EffBS"),
        ("output_dir", "Output dir"),
    ]
    present = [(key, header) for key, header in columns
               if any(key in row for row in rows)]
    widths = {key: max(len(header), *(len(str(row.get(key, ""))) for row in rows))
              for key, header in present}
    lines = [title, ""]
    lines.append("  ".join(header.ljust(widths[key]) for key, header in present))
    lines.append("  ".join("-" * widths[key] for key, _ in present))
    for row in rows:
        lines.append("  ".join(
            str(row.get(key, "")).ljust(widths[key]) for key, _ in present))
    lines.append("")
    lines.append(f"{len(rows)} variant(s).")
    return "\n".join(lines)
