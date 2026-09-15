# Inference speed benchmark

Generated: 2026-09-15T13:32:29  
Dataset: **MTSD** | task: **detection** | split: test | images: 50 | batch size: 1 | warmup: 3 | seed: 42  
Hardware: NVIDIA GeForce RTX 4090 (cuda), torch 2.10.0.dev20251013+cu130

| Model | Status | Mean ms | Median ms | p95 ms | FPS | Peak GPU GB | Params (M) | File MB | Cold start s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo26s | ok | 11.07 | 10.96 | 11.61 | 90.31 | 0.114 | 10.0 | 19.42 | 0.19 |
| yolo26m@mtsd | ok | 12.5 | 11.55 | 20.14 | 79.94 | 0.221 | 21.8 | 42.05 | 0.08 |
| yolo11s@mtsd | ok | 11.94 | 11.12 | 13.61 | 83.75 | 0.112 | 9.4 | 18.33 | 0.14 |
| rfdetr-m@mtsd | ok | 19.93 | 18.75 | 25.98 | 50.15 | 0.215 | 33.4 | 127.62 | 0.37 |
| rfdetr-n@mtsd | ok | 18.45 | 16.94 | 28.01 | 54.19 | 0.171 | 30.2 | 115.32 | 0.27 |
| rfdetr-s@mtsd | ok | 20.39 | 18.13 | 31.18 | 49.03 | 0.198 | 31.8 | 121.6 | 0.33 |

Latency statistics are per image, warmup excluded, CUDA-synchronised. Cold start = model construction + weight loading, reported separately.
