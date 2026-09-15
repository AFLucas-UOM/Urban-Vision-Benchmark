# Inference speed benchmark

Generated: 2026-09-15T15:46:46  
Dataset: **MTSD** | task: **detection** | split: test | images: 50 | batch size: 1 | warmup: 3 | seed: 42  
Hardware: NVIDIA GeForce RTX 4090 (cuda), torch 2.10.0.dev20251013+cu130

| Model | Status | Mean ms | Median ms | p95 ms | FPS | Peak GPU GB | Params (M) | File MB | Cold start s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo26m@mtsd | ok | 13.45 | 11.82 | 19.92 | 74.33 | 0.188 | 21.8 | 41.99 | 0.19 |
| yolo26n@mtsd | ok | 11.63 | 11.44 | 12.78 | 85.94 | 0.136 | 2.5 | 5.12 | 0.06 |
| yolo26s@mtsd | ok | 12.37 | 11.83 | 15.52 | 80.78 | 0.113 | 10.0 | 19.36 | 0.17 |
| yolo11s@mtsd | ok | 12.38 | 11.9 | 15.45 | 80.73 | 0.147 | 9.4 | 18.27 | 0.06 |
| rfdetr-n@mtsd | ok | 17.56 | 16.95 | 20.16 | 56.91 | 0.171 | 30.2 | 115.32 | 0.41 |

Latency statistics are per image, warmup excluded, CUDA-synchronised. Cold start = model construction + weight loading, reported separately.
