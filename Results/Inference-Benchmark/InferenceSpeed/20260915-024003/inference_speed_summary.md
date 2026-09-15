# Inference speed benchmark

Generated: 2026-09-15T02:40:03  
Dataset: **MTSD** | task: **detection** | split: test | images: 50 | batch size: 1 | warmup: 3 | seed: 42  
Hardware: NVIDIA GeForce RTX 4090 (cuda), torch 2.10.0+cu128

| Model | Status | Mean ms | Median ms | p95 ms | FPS | Peak GPU GB | Params (M) | File MB | Cold start s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo11m | ok | 129.64 | 116.6 | 246.66 | 7.71 | 0.348 | 20.1 | 38.69 | 0.18 |
| yolo11s@mtsd | ok | 142.01 | 135.45 | 267.93 | 7.04 | 0.271 | 9.4 | 18.33 | 0.06 |
| yolo11n@mtsd | ok | 128.98 | 121.29 | 239.54 | 7.75 | 0.2 | 2.6 | 5.26 | 0.05 |
| yolo11s@mtsd | ok | 140.2 | 132.47 | 251.47 | 7.13 | 0.284 | 9.4 | 18.34 | 0.07 |
| yolo11m@mtsd | ok | 163.42 | 151.37 | 294.48 | 6.12 | 0.437 | 20.1 | 38.69 | 0.12 |
| yolo12n@mtsd | ok | 144.45 | 136.16 | 269.84 | 6.92 | 0.325 | 2.6 | 5.31 | 0.06 |
| yolo12s@mtsd | ok | 133.83 | 121.52 | 277.07 | 7.47 | 0.468 | 9.3 | 71.4 | 0.14 |
| yolo12s@mtsd | ok | 167.37 | 155.03 | 318.71 | 5.97 | 0.502 | 9.3 | 71.4 | 0.13 |
| yolo26s | ok | 146.22 | 137.78 | 268.9 | 6.84 | 0.432 | 10.0 | 19.42 | 0.08 |
| yolo26n@mtsd | ok | 131.47 | 122.53 | 247.26 | 7.61 | 0.372 | 2.5 | 5.19 | 0.06 |
| yolo26s@mtsd | ok | 149.44 | 141.69 | 270.81 | 6.69 | 0.197 | 10.0 | 19.43 | 0.2 |
| yolo26s@mtsd | ok | 147.0 | 139.04 | 264.44 | 6.8 | 0.23 | 10.0 | 19.42 | 0.09 |
| yolo26m@mtsd | ok | 136.11 | 127.63 | 265.74 | 7.35 | 0.42 | 21.8 | 42.05 | 0.13 |

Latency statistics are per image, warmup excluded, CUDA-synchronised. Cold start = model construction + weight loading, reported separately.
