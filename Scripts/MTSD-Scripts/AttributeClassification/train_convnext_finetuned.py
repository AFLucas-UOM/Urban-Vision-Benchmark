"""Train the ConvNeXt-Tiny (fully fine-tuned) baseline variant.

Thin wrapper around the shared training core; all behaviour is configured in
config/default.yaml.

Note: for historical continuity this script drives the config variant key
``convnext`` (existing checkpoints/metrics/W&B runs keep that name); only
the script filename changed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mtsd_attr.train_common import main_cli

if __name__ == "__main__":
    main_cli("convnext")
