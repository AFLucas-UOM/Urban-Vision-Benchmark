# Inference speed benchmark

Generated: 2026-09-10T16:39:21  
Dataset: **MDWD** | task: **detection** | split: test | images: 50 | batch size: 1 | warmup: 3 | seed: 42  
Hardware: NVIDIA GeForce RTX 4090 (cuda), torch 2.10.0.dev20251013+cu130

| Model | Status | Mean ms | Median ms | p95 ms | FPS | Peak GPU GB | Params (M) | File MB | Cold start s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo11n | ok | 17.13 | 17.21 | 18.72 | 58.35 | 0.064 | 2.6 | 5.22 | 0.19 |
| yolo11s | ok | 13.97 | 13.64 | 16.73 | 71.57 | 0.118 | 9.4 | 18.29 | 0.06 |
| yolo11m | ok | 15.37 | 15.23 | 16.26 | 65.05 | 0.21 | 20.1 | 38.64 | 0.08 |
| yolo11l | ok | 19.01 | 18.92 | 19.59 | 52.57 | 0.319 | 25.3 | 48.83 | 0.12 |
| yolo12n | ok | 16.0 | 15.92 | 17.0 | 62.49 | 0.254 | 2.6 | 5.26 | 0.06 |
| yolo12s | ok | 16.81 | 16.49 | 20.25 | 59.46 | 0.108 | 9.3 | 18.06 | 0.17 |
| yolo12m | ok | 17.83 | 17.05 | 20.35 | 56.06 | 0.221 | 20.1 | 38.88 | 0.1 |
| yolo26n | ok | 14.22 | 14.12 | 14.74 | 70.31 | 0.14 | 2.5 | 5.14 | 0.05 |
| yolo26s | ok | 14.55 | 14.37 | 15.58 | 68.7 | 0.199 | 10.0 | 19.38 | 0.07 |
| yolo26m | ok | 14.91 | 14.81 | 15.75 | 67.03 | 0.297 | 21.8 | 42.0 | 0.09 |
| yolo26l | ok | 18.22 | 18.1 | 19.44 | 54.87 | 0.397 | 26.2 | 50.54 | 0.12 |
| yolo26n@dgx | ok | 14.21 | 14.06 | 15.06 | 70.32 | 0.331 | 2.5 | 5.14 | 0.05 |
| yolo26s@dgx | ok | 14.62 | 14.52 | 15.42 | 68.38 | 0.39 | 10.0 | 19.38 | 0.07 |
| yolo26m@dgx | ok | 16.96 | 17.43 | 19.58 | 58.94 | 0.188 | 21.8 | 41.99 | 0.22 |
| yolo26l@dgx | ok | 18.67 | 18.55 | 20.14 | 53.55 | 0.296 | 26.2 | 50.54 | 0.12 |
| rfdetr-n | ok | 19.89 | 18.88 | 31.63 | 50.25 | 0.171 | 30.2 | 345.77 | 0.79 |

Latency statistics are per image, warmup excluded, CUDA-synchronised. Cold start = model construction + weight loading, reported separately.
