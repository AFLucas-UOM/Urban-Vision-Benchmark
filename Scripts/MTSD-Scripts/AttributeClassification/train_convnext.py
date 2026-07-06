"""Train the ConvNeXt-Tiny frozen linear-probe variant.

Thin wrapper around the shared training core. Same ImageNet-pretrained
weights and 768-d features as the fine-tuned ConvNeXt baseline, but the
backbone is frozen and pinned to eval(); only the heads train. This gives the
controlled frozen-representation comparison against DINOv3 and V-JEPA.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mtsd_attr.train_common import main_cli

if __name__ == "__main__":
    main_cli("convnext_frozen")
