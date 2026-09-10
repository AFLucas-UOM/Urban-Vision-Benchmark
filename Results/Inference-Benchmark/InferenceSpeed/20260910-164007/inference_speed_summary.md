# Inference speed benchmark

Generated: 2026-09-10T16:40:07  
Dataset: **MTSD** | task: **attribute** | split: test | images: 50 | batch size: 1 | warmup: 3 | seed: 42  
Hardware: NVIDIA GeForce RTX 4090 (cuda), torch 2.11.0+cu128

| Model | Status | Mean ms | Median ms | p95 ms | FPS | Peak GPU GB | Params (M) | File MB | Cold start s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| convnext_base_finetune | ok | 8.3 | 7.78 | 10.13 | 120.35 | 0.353 | 87.6 | 334.19 | 1.16 |
| convnext_base_frozen | ok | 9.96 | 7.85 | 15.92 | 100.35 | 0.353 | 87.6 | 0.06 | 0.68 |
| convnext_base_lora | ok | 16.35 | 16.17 | 18.93 | 61.12 | 0.365 | 89.0 | 5.6 | 10.12 |
| convnext_large_finetune | ok | 11.45 | 7.76 | 20.47 | 87.25 | 0.778 | 196.2 | 748.73 | 2.17 |
| convnext_large_frozen | ok | 7.97 | 7.69 | 9.05 | 125.42 | 0.778 | 196.2 | 0.08 | 1.57 |
| convnext_large_lora | ok | 16.59 | 15.1 | 29.71 | 60.26 | 0.786 | 198.4 | 8.38 | 1.52 |
| dinov3_vitb_frozen | ok | 8.42 | 7.08 | 16.09 | 118.64 | 0.337 | 85.7 | 0.04 | 1.08 |
| dinov3_vitb_lora | ok | 10.88 | 10.0 | 14.08 | 91.81 | 0.338 | 86.0 | 1.18 | 0.61 |
| dinov3_vitl_frozen | ok | 14.88 | 13.52 | 19.73 | 67.19 | 1.149 | 303.1 | 0.06 | 1.29 |
| dinov3_vitl_lora | ok | 19.25 | 18.4 | 22.87 | 51.92 | 1.152 | 303.9 | 3.08 | 1.09 |
| lingbot_vitb_frozen | ok | 9.19 | 7.14 | 15.45 | 108.67 | 0.34 | 85.7 | 0.04 | 0.97 |
| lingbot_vitb_lora | ok | 9.25 | 8.65 | 15.93 | 108.04 | 0.342 | 86.0 | 1.17 | 0.87 |
| lingbot_vitl_frozen | ok | 13.59 | 12.48 | 19.22 | 73.53 | 1.15 | 303.2 | 0.06 | 3.55 |
| lingbot_vitl_lora | ok | 21.39 | 18.41 | 33.82 | 46.74 | 1.154 | 304.0 | 3.07 | 2.62 |
| vjepa21_vitb_frozen | failed | - | - | - | - | - | - | 0.04 | - |
| vjepa21_vitb_lora | failed | - | - | - | - | - | - | 1.17 | - |
| vjepa21_vitl_frozen | failed | - | - | - | - | - | - | 0.06 | - |
| vjepa21_vitl_lora | failed | - | - | - | - | - | - | 3.07 | - |

## Not benchmarked

- **vjepa21_vitb_frozen** - failed: BackboneUnavailableError: V-JEPA 2.1 could not be loaded via torch.hub (facebookresearch/vjepa2:vjepa2_1_vit_base_384) and allow_backend_fallback is false for this variant — substituting the V-JEPA 2.0 transformers checkpoint would silently change the model version and invalidate the size ablation. Fix the hub load (network/cache/dependencies) or change the config explicitly. Original error: No module named 'app.vjepa_2_1'; 'app' is not a package
- **vjepa21_vitb_lora** - failed: BackboneUnavailableError: V-JEPA 2.1 could not be loaded via torch.hub (facebookresearch/vjepa2:vjepa2_1_vit_base_384) and allow_backend_fallback is false for this variant — substituting the V-JEPA 2.0 transformers checkpoint would silently change the model version and invalidate the size ablation. Fix the hub load (network/cache/dependencies) or change the config explicitly. Original error: No module named 'app.vjepa_2_1'; 'app' is not a package
- **vjepa21_vitl_frozen** - failed: BackboneUnavailableError: V-JEPA 2.1 could not be loaded via torch.hub (facebookresearch/vjepa2:vjepa2_1_vit_large_384) and allow_backend_fallback is false for this variant — substituting the V-JEPA 2.0 transformers checkpoint would silently change the model version and invalidate the size ablation. Fix the hub load (network/cache/dependencies) or change the config explicitly. Original error: No module named 'app.vjepa_2_1'; 'app' is not a package
- **vjepa21_vitl_lora** - failed: BackboneUnavailableError: V-JEPA 2.1 could not be loaded via torch.hub (facebookresearch/vjepa2:vjepa2_1_vit_large_384) and allow_backend_fallback is false for this variant — substituting the V-JEPA 2.0 transformers checkpoint would silently change the model version and invalidate the size ablation. Fix the hub load (network/cache/dependencies) or change the config explicitly. Original error: No module named 'app.vjepa_2_1'; 'app' is not a package

Latency statistics are per image, warmup excluded, CUDA-synchronised. Cold start = model construction + weight loading, reported separately.
