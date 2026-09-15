# Inference speed benchmark

Generated: 2026-09-15T16:01:57  
Dataset: **MTSD** | task: **detection** | split: test | images: 50 | batch size: 1 | warmup: 3 | seed: 42  
Hardware: NVIDIA GeForce RTX 4090 (cuda), torch 2.10.0.dev20251013+cu130

| Model | Status | Mean ms | Median ms | p95 ms | FPS | Peak GPU GB | Params (M) | File MB | Cold start s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo26n@mtsd | ok | 15.93 | 15.86 | 17.27 | 62.76 | 0.063 | 2.5 | 5.19 | 0.31 |
| yolo12n@mtsd | ok | 18.39 | 18.23 | 20.26 | 54.34 | 0.072 | 2.6 | 5.31 | 0.24 |

Latency statistics are per image, warmup excluded, CUDA-synchronised. Cold start = model construction + weight loading, reported separately.
