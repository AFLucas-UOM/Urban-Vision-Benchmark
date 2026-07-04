"""Train the DINOv3 (frozen backbone, probed heads) variant.

Thin wrapper around the shared training core; all behaviour is configured in
config/default.yaml. Note that DINOv3 checkpoints are gated on Hugging Face:
if access has not been granted yet this exits with instructions (code 2).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mtsd_attr.train_common import main_cli

if __name__ == "__main__":
    main_cli("dinov3")
