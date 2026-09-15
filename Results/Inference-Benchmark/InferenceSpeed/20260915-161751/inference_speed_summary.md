# Inference speed benchmark

Generated: 2026-09-15T16:17:51  
Dataset: **MDWD** | task: **prompt** | split: test | images: 50 | batch size: 1 | warmup: 3 | seed: 42  
Hardware: NVIDIA GeForce RTX 4090 (cuda), torch 2.10.0+cu128

| Model | Status | Mean ms | Median ms | p95 ms | FPS | Peak GPU GB | Params (M) | File MB | Cold start s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sam_3 | ok | 105.53 | 105.24 | 107.28 | 9.48 | 4.969 | - | - | 8.45 |
| sam_3.1 | ok | 105.62 | 105.44 | 107.0 | 9.47 | 4.97 | - | - | 5.93 |
| locateanything_3b | ok | 241.68 | 239.93 | 254.75 | 4.14 | 8.006 | - | - | 10.86 |
| cosmos_reason2_2b | ok | 1054.23 | 864.26 | 1377.42 | 0.95 | 4.096 | - | - | 5.35 |
| cosmos_reason2_8b | ok | 1008.6 | 373.25 | 1767.12 | 0.99 | 16.532 | - | - | 9.64 |

Latency statistics are per image, warmup excluded, CUDA-synchronised. Cold start = model construction + weight loading, reported separately.
