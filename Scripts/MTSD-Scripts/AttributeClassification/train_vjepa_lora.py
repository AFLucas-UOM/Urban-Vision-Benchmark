"""Train the V-JEPA 2.1 LoRA variant (frozen base + adapters on fused qkv).

Thin wrapper around the shared training core; LoRA hyperparameters live in
config/default.yaml under models.vjepa_lora.lora. Only the LoRA adapter
parameters and the classification heads are trainable. Requires the torch.hub
backend: the transformers fallback has different module names, so this variant
fails explicitly rather than adapting the wrong layers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mtsd_attr.train_common import main_cli

if __name__ == "__main__":
    main_cli("vjepa_lora")
