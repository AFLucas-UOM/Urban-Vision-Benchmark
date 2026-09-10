# Inference speed benchmark

Generated: 2026-09-10T22:23:07  
Dataset: **MTSD** | task: **attribute** | split: test | images: 50 | batch size: 1 | warmup: 3 | seed: 42  
Hardware: NVIDIA GeForce RTX 4090 (cuda), torch 2.11.0+cu128

| Model | Status | Mean ms | Median ms | p95 ms | FPS | Peak GPU GB | Params (M) | File MB | Cold start s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vjepa21_vitb_frozen | ok | 29.27 | 26.9 | 39.72 | 34.16 | 0.365 | 86.8 | 0.04 | 2.9 |
| vjepa21_vitb_lora | ok | 31.17 | 29.58 | 46.22 | 32.07 | 0.367 | 87.1 | 1.17 | 10.41 |
| vjepa21_vitl_frozen | ok | 56.53 | 56.15 | 62.99 | 17.69 | 1.183 | 304.7 | 0.06 | 4.94 |
| vjepa21_vitl_lora | ok | 61.03 | 59.39 | 73.51 | 16.38 | 1.185 | 305.5 | 3.07 | 4.3 |

Latency statistics are per image, warmup excluded, CUDA-synchronised. Cold start = model construction + weight loading, reported separately.
