# Inference speed benchmark

Generated: 2026-09-10T22:40:07  
Dataset: **MTSD** | task: **detection** | split: test | images: 50 | batch size: 1 | warmup: 3 | seed: 42  
Hardware: NVIDIA GeForce RTX 4090 (cuda), torch 2.10.0.dev20251013+cu130

| Model | Status | Mean ms | Median ms | p95 ms | FPS | Peak GPU GB | Params (M) | File MB | Cold start s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rfdetrm@mtsd | ok | 90.78 | 82.11 | 163.87 | 11.01 | 0.681 | 33.4 | 127.62 | 0.45 |

Latency statistics are per image, warmup excluded, CUDA-synchronised. Cold start = model construction + weight loading, reported separately.
