# Inference speed benchmark

Generated: 2026-09-15T02:39:03  
Dataset: **MDWD** | task: **detection** | split: test | images: 50 | batch size: 1 | warmup: 3 | seed: 42  
Hardware: NVIDIA GeForce RTX 4090 (cuda), torch 2.10.0.dev20251013+cu130

| Model | Status | Mean ms | Median ms | p95 ms | FPS | Peak GPU GB | Params (M) | File MB | Cold start s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rfdetr-s@mdwd-results | ok | 17.64 | 17.62 | 18.37 | 56.67 | 0.173 | 31.8 | 121.52 | 0.47 |
| rfdetr-m@mdwd-results | ok | 19.1 | 19.04 | 19.74 | 52.34 | 0.193 | 33.4 | 127.54 | 0.43 |

Latency statistics are per image, warmup excluded, CUDA-synchronised. Cold start = model construction + weight loading, reported separately.
