"""Train the V-JEPA (frozen backbone, probed heads) variant.

Thin wrapper around the shared training core. The backend (V-JEPA 2.1 via
torch.hub vs V-JEPA 2.0 via transformers) is selected in config/default.yaml;
a failed torch.hub load falls back to transformers with a warning.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mtsd_attr.train_common import main_cli

if __name__ == "__main__":
    main_cli("vjepa")
