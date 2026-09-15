# Inference speed benchmark

Generated: 2026-09-15T15:56:12  
Dataset: **MTSD** | task: **prompt** | split: test | images: 50 | batch size: 1 | warmup: 3 | seed: 42  
Hardware: NVIDIA GeForce RTX 4090 (cuda), torch 2.10.0+cu128

| Model | Status | Mean ms | Median ms | p95 ms | FPS | Peak GPU GB | Params (M) | File MB | Cold start s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sam_3 | ok | 107.02 | 106.73 | 109.73 | 9.34 | 4.969 | - | - | 9.21 |
| sam_3.1 | ok | 108.26 | 107.56 | 111.52 | 9.24 | 4.97 | - | - | 6.1 |
| locateanything_3b | ok | 2566.71 | 3032.54 | 4221.7 | 0.39 | 8.006 | - | - | 18.71 |
| cosmos_reason2_2b | ok | 1140.41 | 866.92 | 2462.34 | 0.88 | 4.096 | - | - | 7.05 |
| cosmos_reason2_8b | ok | 2772.45 | 1803.93 | 4715.11 | 0.36 | 16.589 | - | - | 16.8 |

Latency statistics are per image, warmup excluded, CUDA-synchronised. Cold start = model construction + weight loading, reported separately.
