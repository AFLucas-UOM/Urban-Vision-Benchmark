# Inference speed benchmark

Generated: 2026-09-15T16:06:44  
Dataset: **MTSD** | task: **detection** | split: test | images: 50 | batch size: 1 | warmup: 3 | seed: 42  
Hardware: NVIDIA GeForce RTX 4090 (cuda), torch 2.10.0.dev20251013+cu130

| Model | Status | Mean ms | Median ms | p95 ms | FPS | Peak GPU GB | Params (M) | File MB | Cold start s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo26s | ok | 11.94 | 11.64 | 12.76 | 83.71 | 0.114 | 10.0 | 19.42 | 0.17 |
| yolo26n@mtsd | ok | 11.79 | 11.53 | 13.93 | 84.81 | 0.097 | 2.5 | 5.19 | 0.05 |

Latency statistics are per image, warmup excluded, CUDA-synchronised. Cold start = model construction + weight loading, reported separately.
