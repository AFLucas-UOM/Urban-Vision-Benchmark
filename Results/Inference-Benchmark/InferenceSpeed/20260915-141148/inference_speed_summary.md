# Inference speed benchmark

Generated: 2026-09-15T14:11:48  
Dataset: **MTSD** | task: **prompt** | split: test | images: 10 | batch size: 1 | warmup: 3 | seed: 42  
Hardware: NVIDIA GeForce RTX 4090 (cuda), torch 2.10.0+cu128

| Model | Status | Mean ms | Median ms | p95 ms | FPS | Peak GPU GB | Params (M) | File MB | Cold start s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sam_3 | ok | 110.06 | 108.69 | 119.36 | 9.09 | 4.969 | - | - | 8.9 |
| sam_3.1 | ok | 108.18 | 107.84 | 110.76 | 9.24 | 4.97 | - | - | 6.22 |
| locateanything_3b | ok | 2326.66 | 2974.08 | 4208.03 | 0.43 | 8.006 | - | - | 17.28 |
| cosmos_reason2_2b | ok | 1012.42 | 855.56 | 1517.02 | 0.99 | 4.096 | - | - | 7.05 |
| cosmos_reason2_8b | ok | 2057.88 | 1780.4 | 4567.18 | 0.49 | 16.532 | - | - | 16.48 |
| cosmos_reason2_32b | ok | 425871.8 | 164916.05 | 2678703.75 | 0.0 | 19.766 | - | - | 30.27 |

Latency statistics are per image, warmup excluded, CUDA-synchronised. Cold start = model construction + weight loading, reported separately.
