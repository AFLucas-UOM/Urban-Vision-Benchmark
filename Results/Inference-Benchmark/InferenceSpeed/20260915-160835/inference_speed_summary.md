# Inference speed benchmark

Generated: 2026-09-15T16:08:35  
Dataset: **MTSD** | task: **detection** | split: test | images: 50 | batch size: 1 | warmup: 3 | seed: 42  
Hardware: NVIDIA GeForce RTX 4090 (cuda), torch 2.10.0.dev20251013+cu130

| Model | Status | Mean ms | Median ms | p95 ms | FPS | Peak GPU GB | Params (M) | File MB | Cold start s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo26s | ok | 11.48 | 11.3 | 12.09 | 87.09 | 0.114 | 10.0 | 19.42 | 0.16 |
| yolo26m@mtsd | ok | 13.63 | 11.78 | 27.62 | 73.32 | 0.221 | 21.8 | 41.99 | 0.07 |
| yolo26n@mtsd | ok | 11.63 | 11.46 | 13.01 | 85.92 | 0.063 | 2.5 | 5.19 | 0.14 |
| yolo26n@mtsd | ok | 11.76 | 11.51 | 14.65 | 84.99 | 0.071 | 2.5 | 5.12 | 0.05 |
| yolo26s@mtsd | ok | 13.27 | 12.75 | 16.28 | 75.33 | 0.131 | 10.0 | 19.36 | 0.07 |
| yolo26m@mtsd | ok | 13.64 | 12.05 | 19.28 | 73.27 | 0.24 | 21.8 | 42.05 | 0.12 |
| yolo11s@mtsd | ok | 12.48 | 11.61 | 16.32 | 80.1 | 0.241 | 9.4 | 18.33 | 0.07 |
| yolo11s@mtsd | ok | 12.73 | 11.36 | 16.97 | 78.51 | 0.275 | 9.4 | 18.27 | 0.05 |
| yolo12n@mtsd | ok | 14.41 | 14.22 | 15.97 | 69.38 | 0.262 | 2.6 | 5.31 | 0.05 |
| rfdetr-m@mtsd | ok | 19.77 | 18.47 | 28.49 | 50.57 | 0.215 | 33.4 | 127.62 | 0.38 |
| rfdetr-n@mtsd | ok | 17.52 | 16.34 | 21.58 | 57.04 | 0.171 | 30.2 | 115.32 | 0.29 |
| rfdetr-s@mtsd | ok | 18.63 | 17.32 | 23.94 | 53.67 | 0.198 | 31.8 | 121.6 | 0.34 |

Latency statistics are per image, warmup excluded, CUDA-synchronised. Cold start = model construction + weight loading, reported separately.
