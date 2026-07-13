"""Generic training entry point: train any configured variant by name.

    python train_variant.py --variant dinov3_vitl_frozen [--smoke-test]
    python train_variant.py --list-variants

This replaces the need for one thin script per variant; the historical thin
scripts (train_dinov3.py, train_vjepa.py, ...) remain as compatibility
wrappers around the same shared training core.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mtsd_attr.backbones import BackboneUnavailableError
from mtsd_attr.config import load_config
from mtsd_attr.train_common import log, run_training
from mtsd_attr.variants import build_plan, format_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default=None,
                        help="Variant key from config models: "
                             "(e.g. dinov3_vitb_frozen)")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Tiny data subset and epoch count to verify the "
                             "pipeline end to end")
    parser.add_argument("--list-variants", action="store_true",
                        help="Print every configured variant with its "
                             "metadata and exit (no training)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.list_variants:
        rows = build_plan(cfg, list(cfg["models"]))
        print(format_plan(rows, title="Configured variants"))
        return
    if not args.variant:
        parser.error("--variant is required (or use --list-variants)")
    if args.variant not in cfg["models"]:
        raise SystemExit(f"Unknown variant {args.variant!r}; available: "
                         f"{list(cfg['models'])}")
    try:
        run_training(args.variant, args.config, args.smoke_test)
    except BackboneUnavailableError as exc:
        log.error("%s", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()
