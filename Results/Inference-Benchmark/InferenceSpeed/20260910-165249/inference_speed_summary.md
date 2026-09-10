# Inference speed benchmark

Generated: 2026-09-10T16:52:49  
Dataset: **MTSD** | task: **prompt** | split: test | images: 10 | batch size: 1 | warmup: 3 | seed: 42  
Hardware: NVIDIA GeForce RTX 4090 (cuda), torch 2.10.0+cu128

| Model | Status | Mean ms | Median ms | p95 ms | FPS | Peak GPU GB | Params (M) | File MB | Cold start s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sam_3 | ok | 234.88 | 223.38 | 377.24 | 4.26 | 7.399 | - | - | 9.01 |
| sam_3.1 | ok | 257.11 | 227.89 | 548.78 | 3.89 | 8.314 | - | - | 5.8 |
| locateanything_3b | ok | 3794.8 | 4343.76 | 5242.92 | 0.26 | 19.352 | - | - | 16.79 |
| cosmos_reason2_2b | ok | 1574.21 | 1556.03 | 1674.61 | 0.64 | 4.774 | - | - | 7.14 |
| cosmos_reason2_8b | ok | 2435.88 | 2163.51 | 6286.78 | 0.41 | 17.732 | - | - | 16.88 |
| cosmos_reason2_32b | ok | 190793.05 | 152337.04 | 618314.79 | 0.01 | 20.595 | - | - | 526.51 |

Latency statistics are per image, warmup excluded, CUDA-synchronised. Cold start = model construction + weight loading, reported separately.
