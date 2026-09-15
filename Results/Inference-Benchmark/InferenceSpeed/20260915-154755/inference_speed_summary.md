# Inference speed benchmark

Generated: 2026-09-15T15:47:55  
Dataset: **MTSD** | task: **detection** | split: test | images: 50 | batch size: 1 | warmup: 3 | seed: 42  
Hardware: NVIDIA GeForce RTX 4090 (cuda), torch 2.10.0.dev20251013+cu130

| Model | Status | Mean ms | Median ms | p95 ms | FPS | Peak GPU GB | Params (M) | File MB | Cold start s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rfdetr-m@mtsd | ok | 18.25 | 16.49 | 32.5 | 54.77 | 0.192 | 33.4 | 127.62 | 0.47 |
| rfdetr-s@mtsd | ok | 19.0 | 18.34 | 26.14 | 52.6 | 0.177 | 31.8 | 121.6 | 0.38 |

Latency statistics are per image, warmup excluded, CUDA-synchronised. Cold start = model construction + weight loading, reported separately.
