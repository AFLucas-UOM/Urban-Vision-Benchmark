"""Train the DINOv3 LoRA variant (frozen base + adapters on q_proj/v_proj).

Thin wrapper around the shared training core; LoRA hyperparameters live in
config/default.yaml under models.dinov3_lora.lora. Only the LoRA adapter
parameters and the classification heads are trainable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mtsd_attr.train_common import main_cli

if __name__ == "__main__":
    main_cli("dinov3_lora")
