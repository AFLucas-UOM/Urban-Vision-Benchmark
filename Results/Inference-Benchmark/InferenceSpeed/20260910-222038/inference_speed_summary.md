# Inference speed benchmark

Generated: 2026-09-10T22:20:39  
Dataset: **MTSD** | task: **attribute** | split: test | images: 50 | batch size: 1 | warmup: 3 | seed: 42  
Hardware: NVIDIA GeForce RTX 4090 (cuda), torch 2.11.0+cu128

| Model | Status | Mean ms | Median ms | p95 ms | FPS | Peak GPU GB | Params (M) | File MB | Cold start s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
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
