"""Orchestrator: refresh the manifest, train selected variants, consolidate results.

Runs group discovery/manifest update once, then trains each selected variant
(sequentially by default; run_all.parallel launches train_variant.py
subprocesses instead), and finally writes the consolidated comparison report
plus the size-ablation report over every variant that has completed test
metrics.

Variant selection (deterministic precedence, see mtsd_attr.variants):
  --variants ...   replaces the set entirely (mutually exclusive with --profile)
  --profile NAME   selects a named profile from run_all.profiles
                   (legacy_default, size_ablation_frozen,
                   size_ablation_adapted, size_ablation_all)
  (neither)        run_all.variants — the legacy default set
  --include ...    appends extra variants to whichever base set was selected

Read-only commands (no manifest refresh, no model loading or downloading, no
checkpoint creation, no W&B):
  --list-variants  print every configured variant with its metadata
  --list-profiles  print the named profiles and their member counts
  --plan           print the resolved experiment matrix for the selection

A variant that fails (unavailable backbone, bad LoRA target, crash) is
reported and skipped so the remaining variants still run. There is no
automatic OOM batch-size fallback: adjust the per-variant
batch_size/gradient_accumulation_steps overrides in the config explicitly.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mtsd_attr.backbones import BackboneUnavailableError
from mtsd_attr.config import load_config, setup_logging
from mtsd_attr.data_manifest import update_manifest
from mtsd_attr.evaluate import write_comparison_report, write_size_ablation_report
from mtsd_attr.train_common import run_training
from mtsd_attr.variants import (available_profiles, build_plan, format_plan,
                                select_variants)

import logging

log = logging.getLogger("mtsd_attr")


def _run_parallel(variants, config_arg, smoke):
    """Launch train_variant.py per variant as a subprocess and wait for all."""
    procs = {}
    entry = Path(__file__).parent / "train_variant.py"
    for variant in variants:
        cmd = [sys.executable, str(entry), "--variant", variant]
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


def main():
    """CLI entry point for the full experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run every variant in smoke-test mode")
    parser.add_argument("--profile", default=None, metavar="PROFILE",
                        help="Named experiment profile from run_all.profiles "
                             "(e.g. size_ablation_all)")
    parser.add_argument("--include", nargs="+", default=None, metavar="VARIANT",
                        help="Additional variants appended to the selected "
                             "base set (default set or --profile)")
    parser.add_argument("--variants", nargs="+", default=None, metavar="VARIANT",
                        help="Replace the variant set entirely (mutually "
                             "exclusive with --profile)")
    parser.add_argument("--list-variants", action="store_true",
                        help="Print every configured variant with metadata "
                             "and exit (read-only)")
    parser.add_argument("--list-profiles", action="store_true",
                        help="Print the named profiles and exit (read-only)")
    parser.add_argument("--plan", action="store_true",
                        help="Print the resolved experiment matrix for the "
                             "current selection and exit — no manifest "
                             "refresh, no model loading/downloading, no "
                             "checkpoints, no W&B, no training")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.list_profiles:
        profiles = available_profiles(cfg)
        print("Available profiles:")
        for name, members in profiles.items():
            print(f"  {name:<24} {len(members):>2} variant(s): "
                  f"{', '.join(members)}")
        return
    if args.list_variants:
        print(format_plan(build_plan(cfg, list(cfg["models"])),
                          title="Configured variants"))
        return

    try:
        variants = select_variants(cfg, profile=args.profile,
                                   variants=args.variants,
                                   include=args.include)
    except ValueError as exc:
        raise SystemExit(str(exc))

    if args.plan:
        source = (f"profile {args.profile!r}" if args.profile
                  else "--variants" if args.variants
                  else "run_all.variants (legacy default)")
        title = f"Experiment plan - {source}"
        if args.include:
            title += f" + --include {' '.join(args.include)}"
        print(format_plan(build_plan(cfg, variants), title=title))
        return

    setup_logging(cfg["paths"]["subproject_root"] / "outputs" / "logs" / "run_all.log")
    update_manifest(cfg)
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
    write_size_ablation_report(cfg, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
