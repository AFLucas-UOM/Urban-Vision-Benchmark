"""Orchestrator: refresh the manifest, train selected variants, consolidate results.

Runs group discovery/manifest update once, then trains each selected variant
(sequentially by default; run_all.parallel launches the thin entry scripts as
subprocesses instead), and finally writes the consolidated comparison report
over every variant that has completed test metrics.

Variant selection: run_all.variants in the YAML is the default experiment set
(the four non-LoRA variants). `--include dinov3_lora vjepa_lora` adds the
optional LoRA variants on top; `--variants ...` replaces the set entirely. A
variant that fails (unavailable backbone, bad LoRA target, crash) is reported
and skipped so the remaining variants still run.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mtsd_attr.backbones import BackboneUnavailableError
from mtsd_attr.config import load_config, setup_logging
from mtsd_attr.data_manifest import update_manifest
from mtsd_attr.evaluate import write_comparison_report
from mtsd_attr.train_common import run_training

import logging

log = logging.getLogger("mtsd_attr")


# Variant keys whose entry-script filename does not follow train_<variant>.py.
# The ConvNeXt scripts were renamed (frozen -> train_convnext.py, fine-tuned ->
# train_convnext_finetuned.py) while the variant keys keep their historical
# names so existing checkpoints/metrics/W&B runs stay valid.
SCRIPT_FOR_VARIANT = {
    "convnext_frozen": "train_convnext.py",
    "convnext": "train_convnext_finetuned.py",
}


def _run_parallel(variants, config_arg, smoke):
    """Launch each variant's entry script as a subprocess and wait for all."""
    procs = {}
    for variant in variants:
        script = SCRIPT_FOR_VARIANT.get(variant, f"train_{variant}.py")
        cmd = [sys.executable, str(Path(__file__).parent / script)]
        if config_arg:
            cmd += ["--config", config_arg]
        if smoke:
            cmd.append("--smoke-test")
        log.info("Launching %s", " ".join(cmd))
        procs[variant] = subprocess.Popen(cmd)
    failed = []
    for variant, proc in procs.items():
        if proc.wait() != 0:
            failed.append(variant)
    return failed


def _select_variants(cfg, args):
    """Resolve the variant list from config defaults and CLI selection flags."""
    available = list(cfg["models"])
    if args.variants:
        selected = args.variants
    else:
        selected = list(cfg["run_all"]["variants"]) + (args.include or [])
    unknown = [v for v in selected if v not in available]
    if unknown:
        raise SystemExit(f"Unknown variant(s) {unknown}; available: {available}")
    seen = []
    for variant in selected:
        if variant not in seen:
            seen.append(variant)
    return seen


def main():
    """CLI entry point for the full experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run every variant in smoke-test mode")
    parser.add_argument("--include", nargs="+", default=None, metavar="VARIANT",
                        help="Additional variants to run on top of the "
                             "configured default set (e.g. dinov3_lora)")
    parser.add_argument("--variants", nargs="+", default=None, metavar="VARIANT",
                        help="Replace the variant set entirely")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg["paths"]["subproject_root"] / "outputs" / "logs" / "run_all.log")

    update_manifest(cfg)
    variants = _select_variants(cfg, args)
    log.info("Selected variants: %s", variants)
    skipped = []

    if cfg["run_all"].get("parallel"):
        log.info("Running %s in parallel subprocesses", variants)
        skipped = _run_parallel(variants, args.config, args.smoke_test)
    else:
        for variant in variants:
            log.info("=== Variant %s ===", variant)
            try:
                run_training(variant, args.config, args.smoke_test)
            except BackboneUnavailableError as exc:
                log.error("Skipping %s: %s", variant, exc)
                skipped.append(variant)
            except Exception:
                log.exception("Variant %s failed; continuing with the rest",
                              variant)
                skipped.append(variant)

    if skipped:
        log.warning("Variants skipped or failed: %s", skipped)
    write_comparison_report(cfg, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
