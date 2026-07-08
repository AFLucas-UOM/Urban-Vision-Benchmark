"""Shared multi-task training loop used by all three model variants.

Every run: loads the config, refreshes the manifest (automatic group discovery),
builds datasets and the multi-head model, trains with a masked joint loss,
selects the best checkpoint by mean macro-F1 across heads on the validation
split, evaluates that checkpoint on the test split, saves metrics and plots,
logs to Weights & Biases, and appends a line to the experiment log.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .backbones import BackboneUnavailableError, build_backbone
from .config import (REPO_ROOT, adaptation_of, apply_smoke_test,
                     git_commit_hash, load_config, model_training_config,
                     set_seed, setup_logging)
from .data_manifest import update_manifest
from .dataset import (AttributeCropDataset, build_transforms,
                      compute_class_weights, subsample_records)
from .evaluate import evaluate_model, mean_macro_f1, save_metrics_bundle
from .multihead_model import (MaskedMultiTaskLoss, MultiHeadClassifier,
                              parameter_breakdown)

log = logging.getLogger("mtsd_attr")


def _build_loaders(cfg, training, backbone_image_size, records_by_split, smoke_max):
    """Construct train/val/test DataLoaders for the given records.

    Returns:
        dict split -> DataLoader (train shuffled and augmented, others not).
    """
    attributes = cfg["attributes"]
    root = cfg["paths"]["subproject_root"]
    aug = training["augmentation"]
    loaders = {}
    for split in ("train", "val", "test"):
        records = records_by_split.get(split, [])
        if smoke_max is not None:
            records = subsample_records(records, smoke_max, cfg["seed"])
        transform = build_transforms(backbone_image_size, split == "train", aug)
        dataset = AttributeCropDataset(records, attributes, root, transform)
        loaders[split] = DataLoader(
            dataset,
            batch_size=training["batch_size"],
            shuffle=(split == "train"),
            num_workers=training["num_workers"],
            pin_memory=torch.cuda.is_available(),
            persistent_workers=training["num_workers"] > 0,
        )
        log.info("Split %-5s: %d crops", split, len(dataset))
    return loaders


def _build_optimizer(model, training):
    """AdamW with separate parameter groups for heads and backbone-resident
    trainable parameters (the full backbone when fine-tuning, or just the LoRA
    adapters), each with its own learning rate."""
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("backbone.")]
    groups = [{"params": head_params, "lr": training["lr_heads"]}]
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and n.startswith("backbone.")]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": training["lr_backbone"]})
    return torch.optim.AdamW(groups, weight_decay=training["weight_decay"])


def _build_scheduler(optimizer, training, steps_per_epoch):
    """Per-step linear warmup followed by cosine decay to zero.

    The cosine horizon is max_epochs * steps_per_epoch; early stopping simply
    truncates the schedule. Warmup lasts warmup_epochs * steps_per_epoch.
    """
    total = max(1, training["max_epochs"] * steps_per_epoch)
    warmup = int(training["warmup_epochs"] * steps_per_epoch)

    def factor(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _train_one_epoch(model, loader, loss_fn, optimizer, scheduler, device,
                     amp, grad_clip):
    """Run one training epoch and return (mean_total_loss, mean_per_head_losses)."""
    model.train()
    totals, head_totals, batches = 0.0, {}, 0
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=amp):
            logits = model(images)
            loss, per_head = loss_fn(logits, targets)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), grad_clip)
        optimizer.step()
        scheduler.step()
        totals += loss.item()
        batches += 1
        for attr, value in per_head.items():
            head_totals[attr] = head_totals.get(attr, 0.0) + value
    mean_heads = {a: v / batches for a, v in head_totals.items()}
    return totals / max(1, batches), mean_heads


def _checkpoint_state(model):
    """Model state dict pruned to what the adaptation mode actually trains.

    Frozen: heads only. LoRA: heads plus the backbone's lora_* adapter
    tensors. Finetune: everything. The pretrained base weights are always
    reconstructable from the model identifiers stored alongside.
    """
    state = model.state_dict()
    if model.adaptation == "frozen":
        return {k: v for k, v in state.items() if not k.startswith("backbone.")}
    if model.adaptation == "lora":
        return {k: v for k, v in state.items()
                if not k.startswith("backbone.") or "lora_" in k}
    return state


def _load_checkpoint_state(model, state, adaptation):
    """Load a pruned checkpoint back into a freshly built model, verifying fit.

    For frozen/LoRA checkpoints the backbone base weights are expected to be
    missing (they come from the pretrained load); anything else missing, or
    any unexpected key, is an error.
    """
    if adaptation == "finetune":
        model.load_state_dict(state)
        return
    model_keys = set(model.state_dict())
    state = _normalise_checkpoint_keys(state, model_keys)
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Checkpoint has unexpected keys: "
                           f"{result.unexpected_keys[:5]}...")
    bad = [k for k in result.missing_keys
           if not k.startswith("backbone.") or "lora_" in k]
    if bad:
        raise RuntimeError(f"Checkpoint is missing trained parameters: "
                           f"{bad[:5]}...")


def _normalise_checkpoint_keys(state, model_keys):
    """Map compatible wrapper-depth differences between library versions."""
    normalised = {}
    for key, value in state.items():
        target = key
        if target not in model_keys:
            candidates = []
            if key.startswith("backbone.model.model."):
                candidates.append("backbone.model." + key[len("backbone.model.model."):])
            if key.startswith("backbone.model."):
                candidates.append("backbone.model.model." + key[len("backbone.model."):])
            for candidate in candidates:
                if candidate in model_keys:
                    target = candidate
                    break
        normalised[target] = value
    return normalised


def _checkpoint_adaptation(ckpt, model_cfg):
    """Infer the adaptation mode from metadata, with LoRA-key compatibility.

    Some older/manual checkpoint paths can carry LoRA adapter tensors even when
    the caller supplies the base variant name. The saved tensors are the source
    of truth in that case: rebuilding a frozen backbone would reject them as
    unexpected keys.
    """
    has_lora = any("lora_" in key for key in ckpt.get("model_state", {}))
    adaptation = ckpt.get("adaptation") or adaptation_of(model_cfg)
    return "lora" if has_lora else adaptation


def _append_experiment_log(path, entry):
    """Append one JSON line to the experiment log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def run_training(variant, config_path=None, smoke_test=False):
    """Train one model variant end to end and return its summary dict.

    Raises:
        BackboneUnavailableError: If the variant's checkpoint is inaccessible
            (e.g. gated DINOv3 weights); callers may skip the variant.
    """
    cfg = load_config(config_path)
    model_cfg, training = model_training_config(cfg, variant)
    run_id = f"{variant}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if smoke_test:
        run_id += "-smoke"
    setup_logging(cfg["paths"]["subproject_root"] / "outputs" / "logs" / f"{run_id}.log")
    log.info("Run %s starting (variant=%s, smoke_test=%s)", run_id, variant, smoke_test)
    log.info("Config: %s | git commit: %s", cfg["_config_path"], git_commit_hash())

    from dotenv import load_dotenv
    load_dotenv(cfg["paths"]["env_file"])

    smoke_max = apply_smoke_test(cfg, training) if smoke_test else None
    set_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True

    manifest = update_manifest(cfg)
    records_by_split = {}
    for rec in manifest["records"]:
        records_by_split.setdefault(rec["split"], []).append(rec)

    adaptation = adaptation_of(model_cfg)
    backbone = build_backbone(model_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        log.warning("CUDA not available; training on CPU will be very slow")
    amp = bool(training["amp"]) and device.type == "cuda"

    loaders = _build_loaders(cfg, training, backbone.image_size,
                             records_by_split, smoke_max)
    class_weights = None
    if training["class_weighted_loss"]:
        class_weights = compute_class_weights(
            loaders["train"].dataset.records, cfg["attributes"])
        for attr, w in class_weights.items():
            log.info("Class weights %s: %s", attr,
                     [round(float(x), 3) for x in w])

    model = MultiHeadClassifier(backbone, cfg["attributes"], cfg["probe"],
                                adaptation).to(device)
    params = parameter_breakdown(model)
    log.info(
        "Parameters (%s): %.2fM total, %.2fM trainable (%.3f%%) | backbone "
        "trainable %.2fM of %.2fM, heads %.3fM, LoRA %.3fM",
        adaptation, params["total"] / 1e6, params["trainable"] / 1e6,
        params["trainable_pct"], params["backbone_trainable"] / 1e6,
        params["backbone_total"] / 1e6, params["head_trainable"] / 1e6,
        params["lora_trainable"] / 1e6)

    loss_fn = MaskedMultiTaskLoss(
        cfg["attributes"],
        head_weights=training["head_loss_weights"],
        class_weights=class_weights,
        label_smoothing=training["label_smoothing"],
    ).to(device)
    optimizer = _build_optimizer(model, training)
    scheduler = _build_scheduler(optimizer, training, len(loaders["train"]))

    import wandb
    wandb_cfg = cfg["wandb"]
    wandb_mode = None if wandb_cfg["enabled"] else "disabled"
    run_prefix = wandb_cfg.get("run_name_prefix") or ""
    wandb_run = wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg["entity"],
        group=wandb_cfg.get("group") or None,
        name=f"{run_prefix}{run_id}",
        mode=wandb_mode or os.environ.get("WANDB_MODE"),
        config={
            "variant": variant,
            "adaptation": adaptation,
            "lora": model_cfg.get("lora"),
            "model": {k: str(v) for k, v in model_cfg.items()},
            "training": training,
            "parameters": params,
            "seed": cfg["seed"],
            "groups": {g: info["n_crops"] for g, info in manifest["groups"].items()},
            "smoke_test": smoke_test,
        },
    )

    ckpt_dir = cfg["paths"]["checkpoints_dir"] / (
        f"{variant}-smoke" if smoke_test else variant)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
    best_score, best_epoch, best_val_metrics = -1.0, -1, None
    es = training["early_stopping"]
    es_best = float("-inf")
    epochs_without_improvement = 0
    max_epochs = training["max_epochs"]
    stop_reason = "max_epochs"
    ckpt_meta = {
        "variant": variant, "run_id": run_id,
        "adaptation": adaptation, "lora": model_cfg.get("lora"),
        "frozen": adaptation == "frozen", "model_cfg": model_cfg,
        "probe": cfg["probe"], "attributes": cfg["attributes"],
        "max_epochs": max_epochs, "parameters": params,
    }

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss, head_losses = _train_one_epoch(
            model, loaders["train"], loss_fn, optimizer, scheduler, device,
            amp, training["grad_clip"])
        val_metrics = evaluate_model(model, loaders["val"], cfg["attributes"],
                                     device, amp)
        score = mean_macro_f1(val_metrics)
        log.info("Epoch %d/%d: train_loss=%.4f val_mean_macro_f1=%.4f (%.1fs)",
                 epoch, max_epochs, train_loss, score, time.time() - t0)
        wandb_log = {"epoch": epoch, "train/loss": train_loss,
                     "val/mean_macro_f1": score,
                     "lr": optimizer.param_groups[0]["lr"]}
        for attr, value in head_losses.items():
            wandb_log[f"train/loss_{attr}"] = value
        for attr, m in val_metrics.items():
            wandb_log[f"val/{attr}/macro_f1"] = m["macro_f1"]
            wandb_log[f"val/{attr}/accuracy"] = m["accuracy"]

        if score > best_score:
            best_score, best_epoch, best_val_metrics = score, epoch, val_metrics
            torch.save({**ckpt_meta, "epoch": epoch,
                        "val_mean_macro_f1": score,
                        "model_state": _checkpoint_state(model)},
                       ckpt_dir / "best.pt")
            log.info("Epoch %d is the new best (val mean macro-F1 %.4f); "
                     "checkpoint saved", epoch, score)

        if score > es_best + es["min_delta"]:
            es_best = score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        wandb_log["early_stopping/epochs_without_improvement"] = (
            epochs_without_improvement)
        wandb_run.log(wandb_log)
        if es["enabled"] and epochs_without_improvement >= es["patience"]:
            stop_reason = "early_stopping"
            log.info(
                "Early stopping at epoch %d: no val mean macro-F1 improvement "
                "of at least %.4f for %d consecutive epochs (best %.4f at "
                "epoch %d)", epoch, es["min_delta"], es["patience"],
                best_score, best_epoch)
            break

    stopped_epoch = epoch
    log.info("Training ended at epoch %d/%d (reason: %s); best epoch %d "
             "(val mean macro-F1 %.4f)", stopped_epoch, max_epochs,
             stop_reason, best_epoch, best_score)
    torch.save({**ckpt_meta, "epoch": stopped_epoch,
                "stopped_epoch": stopped_epoch, "stop_reason": stop_reason,
                "model_state": _checkpoint_state(model)},
               ckpt_dir / "last.pt")

    best = torch.load(ckpt_dir / "best.pt", map_location=device,
                      weights_only=False)
    _load_checkpoint_state(model, best["model_state"], adaptation)
    test_metrics = evaluate_model(model, loaders["test"], cfg["attributes"],
                                  device, amp)
    test_score = mean_macro_f1(test_metrics)
    log.info("Test (best epoch %d): mean macro-F1 %.4f", best_epoch, test_score)
    for attr, m in test_metrics.items():
        log.info("  %-12s acc=%.4f macro_f1=%.4f", attr, m["accuracy"],
                 m["macro_f1"])

    run_info = {
        "adaptation": adaptation,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "stop_reason": stop_reason,
        "max_epochs": max_epochs,
        "val_mean_macro_f1": best_score,
        "parameters": params,
    }
    metrics_dir = cfg["paths"]["metrics_dir"]
    save_metrics_bundle(metrics_dir, variant, "val", best_val_metrics, run_id,
                        smoke_test, run_info=run_info)
    save_metrics_bundle(metrics_dir, variant, "test", test_metrics, run_id,
                        smoke_test, run_info=run_info)
    wandb_run.summary["test/mean_macro_f1"] = test_score
    wandb_run.summary["best_epoch"] = best_epoch
    wandb_run.summary["stopped_epoch"] = stopped_epoch
    wandb_run.summary["stop_reason"] = stop_reason
    for attr, m in test_metrics.items():
        wandb_run.summary[f"test/{attr}/macro_f1"] = m["macro_f1"]
        wandb_run.summary[f"test/{attr}/accuracy"] = m["accuracy"]
    wandb_run.finish()

    summary = {
        "run_id": run_id,
        "variant": variant,
        "adaptation": adaptation,
        "smoke_test": smoke_test,
        "start_time": start_time,
        "end_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit_hash(),
        "seed": cfg["seed"],
        "groups": {g: info["n_crops"] for g, info in manifest["groups"].items()},
        "split_sizes": {s: len(loaders[s].dataset) for s in loaders},
        "backbone_backend": getattr(backbone, "backend", None),
        "parameters": params,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "stop_reason": stop_reason,
        "max_epochs": max_epochs,
        "val_mean_macro_f1": best_score,
        "test_mean_macro_f1": test_score,
        "test_macro_f1": {a: m["macro_f1"] for a, m in test_metrics.items()},
        "checkpoint": str(ckpt_dir / "best.pt"),
    }
    _append_experiment_log(cfg["paths"]["experiment_log"], summary)
    log.info("Run %s finished", run_id)
    return summary


def main_cli(variant):
    """Shared argparse CLI used by the three thin entry scripts."""
    parser = argparse.ArgumentParser(
        description=f"Train the {variant} variant of the MTSD attribute classifier")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Tiny data subset and epoch count to verify the "
                             "pipeline end to end")
    args = parser.parse_args()
    try:
        run_training(variant, args.config, args.smoke_test)
    except BackboneUnavailableError as exc:
        log.error("%s", exc)
        sys.exit(2)
