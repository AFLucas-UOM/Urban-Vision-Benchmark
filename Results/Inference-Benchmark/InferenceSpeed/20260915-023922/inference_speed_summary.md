# Inference speed benchmark

Generated: 2026-09-15T02:39:22  
Dataset: **MTSD** | task: **detection** | split: test | images: 50 | batch size: 1 | warmup: 3 | seed: 42  
Hardware: NVIDIA GeForce RTX 4090 (cuda), torch 2.10.0+cu128

| Model | Status | Mean ms | Median ms | p95 ms | FPS | Peak GPU GB | Params (M) | File MB | Cold start s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rfdetr-m@mtsd | ok | 86.92 | 83.94 | 158.25 | 11.5 | 0.681 | 33.4 | 127.62 | 0.52 |
| rfdetr-n@mtsd | ok | 88.65 | 79.88 | 158.58 | 11.28 | 0.669 | 30.2 | 115.32 | 0.46 |
| rfdetr-s@mtsd | ok | 88.8 | 77.53 | 157.16 | 11.26 | 0.675 | 31.8 | 121.6 | 0.41 |

Latency statistics are per image, warmup excluded, CUDA-synchronised. Cold start = model construction + weight loading, reported separately.
