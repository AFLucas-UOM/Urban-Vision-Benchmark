# Inference speed benchmark

Generated: 2026-09-10T22:19:23  
Dataset: **MTSD** | task: **detection** | split: test | images: 50 | batch size: 1 | warmup: 3 | seed: 42  
Hardware: NVIDIA GeForce RTX 4090 (cuda), torch 2.10.0.dev20251013+cu130

| Model | Status | Mean ms | Median ms | p95 ms | FPS | Peak GPU GB | Params (M) | File MB | Cold start s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rfdetr-m | ok | 101.97 | 104.29 | 209.76 | 9.81 | 0.681 | 33.4 | 127.62 | 0.49 |
| rfdetrm | ok | 112.32 | 107.53 | 207.7 | 8.9 | 0.681 | 33.4 | 127.62 | 0.44 |
| rfdetr-s | ok | 110.25 | 103.25 | 219.09 | 9.07 | 0.675 | 31.8 | 121.6 | 0.41 |
| rfdetrs | ok | 109.91 | 102.73 | 209.37 | 9.1 | 0.675 | 31.8 | 121.6 | 0.43 |

Latency statistics are per image, warmup excluded, CUDA-synchronised. Cold start = model construction + weight loading, reported separately.
