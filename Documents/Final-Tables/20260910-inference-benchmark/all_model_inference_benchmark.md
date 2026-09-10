# Completed inference benchmark — all models

**Coverage:** 89 unique evaluated model artifacts: 16 MDWD detectors, 49 MTSD detectors, 18 MTSD attribute classifiers, and 6 PromptDetect models.

Each row is one successfully completed benchmark result. Targeted repair runs supersede earlier failed rows for the same checkpoint. Timing is only comparable within the same task/protocol; prompt models use a 10-image sample, all other groups use 50 images. All runs use seed 42, batch size 1, 3 warm-up iterations, CUDA on an NVIDIA GeForce RTX 4090.

`checkpoint_artifact_mb` is the saved checkpoint artifact, not necessarily the full deployable model footprint. In particular, frozen and LoRA attribute-classifier checkpoints store only learned heads/adapters; use `parameters` to compare their instantiated model scale.

## Coverage and provenance

| Dataset | Task | Final models | Protocol |
|---|---:|---:|---|
| MDWD | detection | 16 | 50 images; seed 42; batch 1; warm-up 3 |
| MTSD | attribute | 18 | 50 images; seed 42; batch 1; warm-up 3 |
| MTSD | detection | 49 | 50 images; seed 42; batch 1; warm-up 3 |
| MTSD | prompt | 6 | 10 images; seed 42; batch 1; warm-up 3 |

| Source run | Purpose | Successful rows in source |
|---|---|---:|
| `20260910-163921` | MDWD object detectors | 16 |
| `20260910-164007` | MTSD attribute classifiers (initial) | 14 |
| `20260910-164504` | MTSD object detectors (initial) | 46 |
| `20260910-165249` | PromptDetect models | 6 |
| `20260910-221923` | MTSD RF-DETR compact-name repair | 4 |
| `20260910-222307` | V-JEPA import-isolation repair | 4 |
| `20260910-224006` | MTSD RF-DETR-M strong repair | 1 |

## MDWD — detection (16 models)

| Model | Suite / trained run | Artifact MB | Parameters | Mean ms | p95 ms | FPS | Cold start s | Peak GPU GB | Source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| rfdetr-n | RF-DETR-EUVIP / E001_rfdetr-n_rfv20_coco-mmdet_img640_b32_e100_adamw_lr0p001 | 345.77 | 30,157,870 | 19.89 | 31.63 | 50.25 | 0.79 | 0.17 | `20260910-163921` |
| yolo11l | YOLO11-EUVIP / E004_yolo11l_rfv20_img640_b32_e100_adamw_lr0p001 | 48.83 | 25,314,335 | 19.01 | 19.59 | 52.57 | 0.12 | 0.32 | `20260910-163921` |
| yolo11m | YOLO11-EUVIP / E003_yolo11m_rfv20_img640_b32_e100_adamw_lr0p001 | 38.64 | 20,056,863 | 15.37 | 16.26 | 65.05 | 0.08 | 0.21 | `20260910-163921` |
| yolo11n | YOLO11-EUVIP / E001_yolo11n_rfv20_img640_b32_e100_adamw_lr0p001 | 5.22 | 2,590,815 | 17.13 | 18.72 | 58.35 | 0.19 | 0.06 | `20260910-163921` |
| yolo11s | YOLO11-EUVIP / E002_yolo11s_rfv20_img640_b32_e100_adamw_lr0p001 | 18.29 | 9,429,727 | 13.97 | 16.73 | 71.57 | 0.06 | 0.12 | `20260910-163921` |
| yolo12m | YOLO12-EUVIP / E003_yolo12m_rfv20_img640_b32_e100_adamw_lr0p001 | 38.88 | 20,141,343 | 17.83 | 20.35 | 56.06 | 0.10 | 0.22 | `20260910-163921` |
| yolo12n | YOLO12-EUVIP / E001_yolo12n_rfv20_img640_b32_e100_adamw_lr0p001 | 5.26 | 2,569,023 | 16.00 | 17.00 | 62.49 | 0.06 | 0.25 | `20260910-163921` |
| yolo12s | YOLO12-EUVIP / E002_yolo12s_rfv20_img640_b32_e100_adamw_lr0p001 | 18.06 | 9,255,071 | 16.81 | 20.25 | 59.46 | 0.17 | 0.11 | `20260910-163921` |
| yolo26l@dgx | YOLO26-DGX / E004_yolo26l_rfv20_img640_b32_e100_adamw_lr0p001 | 50.54 | 26,184,054 | 18.67 | 20.14 | 53.55 | 0.12 | 0.30 | `20260910-163921` |
| yolo26m@dgx | YOLO26-DGX / E003_yolo26m_rfv20_img640_b32_e100_adamw_lr0p001 | 41.99 | 21,780,598 | 16.96 | 19.58 | 58.94 | 0.22 | 0.19 | `20260910-163921` |
| yolo26n@dgx | YOLO26-DGX / E001_yolo26n_rfv20_img640_b32_e100_adamw_lr0p001 | 5.14 | 2,505,750 | 14.21 | 15.06 | 70.32 | 0.05 | 0.33 | `20260910-163921` |
| yolo26s@dgx | YOLO26-DGX / E002_yolo26s_rfv20_img640_b32_e100_adamw_lr0p001 | 19.38 | 9,951,734 | 14.62 | 15.42 | 68.38 | 0.07 | 0.39 | `20260910-163921` |
| yolo26l | YOLO26-EUVIP / E004_yolo26l_rfv20_img640_b32_e100_adamw_lr0p001 | 50.54 | 26,184,054 | 18.22 | 19.44 | 54.87 | 0.12 | 0.40 | `20260910-163921` |
| yolo26m | YOLO26-EUVIP / E003_yolo26m_rfv20_img640_b32_e100_adamw_lr0p001 | 42.00 | 21,780,598 | 14.91 | 15.75 | 67.03 | 0.09 | 0.30 | `20260910-163921` |
| yolo26n | YOLO26-EUVIP / E001_yolo26n_rfv20_img640_b32_e100_adamw_lr0p001 | 5.14 | 2,505,750 | 14.22 | 14.74 | 70.31 | 0.05 | 0.14 | `20260910-163921` |
| yolo26s | YOLO26-EUVIP / E002_yolo26s_rfv20_img640_b32_e100_adamw_lr0p001 | 19.38 | 9,951,734 | 14.55 | 15.58 | 68.70 | 0.07 | 0.20 | `20260910-163921` |

## MTSD — attribute (18 models)

| Model | Suite / trained run | Artifact MB | Parameters | Mean ms | p95 ms | FPS | Cold start s | Peak GPU GB | Source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| convnext_base_finetune | — | 334.19 | 87,577,741 | 8.30 | 10.13 | 120.35 | 1.16 | 0.35 | `20260910-164007` |
| convnext_base_frozen | — | 0.06 | 87,577,741 | 9.96 | 15.92 | 100.35 | 0.68 | 0.35 | `20260910-164007` |
| convnext_base_lora | — | 5.60 | 89,021,581 | 16.35 | 18.93 | 61.12 | 10.12 | 0.36 | `20260910-164007` |
| convnext_large_finetune | — | 748.73 | 196,247,245 | 11.45 | 20.47 | 87.25 | 2.17 | 0.78 | `20260910-164007` |
| convnext_large_frozen | — | 0.08 | 196,247,245 | 7.97 | 9.05 | 125.42 | 1.57 | 0.78 | `20260910-164007` |
| convnext_large_lora | — | 8.38 | 198,413,005 | 16.59 | 29.71 | 60.26 | 1.52 | 0.79 | `20260910-164007` |
| dinov3_vitb_frozen | — | 0.04 | 85,670,413 | 8.42 | 16.09 | 118.64 | 1.08 | 0.34 | `20260910-164007` |
| dinov3_vitb_lora | — | 1.18 | 85,965,325 | 10.88 | 14.08 | 91.81 | 0.61 | 0.34 | `20260910-164007` |
| dinov3_vitl_frozen | — | 0.06 | 303,142,925 | 14.88 | 19.73 | 67.19 | 1.29 | 1.15 | `20260910-164007` |
| dinov3_vitl_lora | — | 3.08 | 303,929,357 | 19.25 | 22.87 | 51.92 | 1.09 | 1.15 | `20260910-164007` |
| lingbot_vitb_frozen | — | 0.04 | 85,679,629 | 9.19 | 15.45 | 108.67 | 0.97 | 0.34 | `20260910-164007` |
| lingbot_vitb_lora | — | 1.17 | 85,974,541 | 9.25 | 15.93 | 108.04 | 0.87 | 0.34 | `20260910-164007` |
| lingbot_vitl_frozen | — | 0.06 | 303,167,501 | 13.59 | 19.22 | 73.53 | 3.55 | 1.15 | `20260910-164007` |
| lingbot_vitl_lora | — | 3.07 | 303,953,933 | 21.39 | 33.82 | 46.74 | 2.62 | 1.15 | `20260910-164007` |
| vjepa21_vitb_frozen | — | 0.04 | 86,843,149 | 29.27 | 39.72 | 34.16 | 2.90 | 0.36 | `20260910-222307` |
| vjepa21_vitb_lora | — | 1.17 | 87,138,061 | 31.17 | 46.22 | 32.07 | 10.41 | 0.37 | `20260910-222307` |
| vjepa21_vitl_frozen | — | 0.06 | 304,694,285 | 56.53 | 62.99 | 17.69 | 4.94 | 1.18 | `20260910-222307` |
| vjepa21_vitl_lora | — | 3.07 | 305,480,717 | 61.03 | 73.51 | 16.38 | 4.30 | 1.19 | `20260910-222307` |

## MTSD — detection (49 models)

| Model | Suite / trained run | Artifact MB | Parameters | Mean ms | p95 ms | FPS | Cold start s | Peak GPU GB | Source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| rfdetr-m | RF-DETR-MTSD / E12_rfdetr-m_mtsdqa1aug_img576_eb32_e100_adamw_s42 | 127.62 | 33,403,216 | 101.97 | 209.76 | 9.81 | 0.49 | 0.68 | `20260910-221923` |
| rfdetr-n | RF-DETR-MTSD / E10_rfdetr-n_mtsdqa1aug_img384_eb32_e100_adamw_s42 | 115.32 | 30,183,056 | 118.67 | 218.48 | 8.43 | 0.40 | 0.69 | `20260910-164504` |
| rfdetr-s | RF-DETR-MTSD / E11_rfdetr-s_mtsdqa1aug_img512_eb32_e100_adamw_s42 | 121.60 | 31,826,928 | 110.25 | 219.09 | 9.07 | 0.41 | 0.68 | `20260910-221923` |
| rfdetrm | RF-DETR-MTSD / FINAL_rfdetrm_mtsd_noaug_img576_pb32_eb32_e100_adamw_s42 | 127.62 | 33,403,216 | 112.32 | 207.70 | 8.90 | 0.44 | 0.68 | `20260910-221923` |
| rfdetrm@mtsd | RF-DETR-MTSD / FINAL_rfdetrm_mtsd_strong_img576_pb32_eb32_e100_adamw_s42 | 127.62 | 33,403,216 | 90.78 | 163.87 | 11.01 | 0.45 | 0.68 | `20260910-224006` |
| rfdetrs | RF-DETR-MTSD / FINAL_rfdetrs_mtsd_strong_img512_pb32_eb32_e100_adamw_s42 | 121.60 | 31,826,928 | 109.91 | 209.37 | 9.10 | 0.43 | 0.68 | `20260910-221923` |
| yolo11m | YOLO11-MTSD / E01_yolo11m_mtsdqa1aug_img1280_eb32_e100_adamw_s42 | 38.69 | 20,062,260 | 180.16 | 345.15 | 5.55 | 0.20 | 0.17 | `20260910-164504` |
| yolo11m@mtsd | YOLO11-MTSD / E01_yolo11m_mtsdqa1noaug_img640_eb32_e100_adamw_s42 | 38.63 | 20,062,260 | 178.79 | 340.56 | 5.59 | 0.12 | 0.24 | `20260910-164504` |
| yolo11m@mtsd | YOLO11-MTSD / E02_yolo11m_mtsdqa1aug_img960_eb32_e100_adamw_s42 | 38.65 | 20,062,260 | 179.23 | 344.27 | 5.58 | 0.21 | 0.17 | `20260910-164504` |
| yolo11m@mtsd | YOLO11-MTSD / E09_yolo11m_mtsdqa1aug_img640_eb32_e100_adamw_s42 | 38.63 | 20,062,260 | 174.96 | 336.85 | 5.72 | 0.11 | 0.23 | `20260910-164504` |
| yolo11m@mtsd | YOLO11-MTSD / FINAL_yolo11m_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | 38.63 | 20,062,260 | 174.38 | 335.96 | 5.73 | 0.12 | 0.30 | `20260910-164504` |
| yolo11m@mtsd | YOLO11-MTSD / FINAL_yolo11m_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | 38.65 | 20,062,260 | 175.94 | 343.41 | 5.68 | 0.14 | 0.37 | `20260910-164504` |
| yolo11n | YOLO11-MTSD / E07_yolo11n_mtsdqa1aug_img640_eb32_e100_adamw_s42 | 5.20 | 2,592,180 | 166.23 | 325.99 | 6.02 | 0.05 | 0.11 | `20260910-164504` |
| yolo11n@mtsd | YOLO11-MTSD / FINAL_yolo11n_mtsd_strong_img1280_pb8_eb64_e100_adamw_s42 | 5.26 | 2,592,180 | 164.88 | 330.13 | 6.06 | 0.05 | 0.30 | `20260910-164504` |
| yolo11n@mtsd | YOLO11-MTSD / FINAL_yolo11n_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | 5.20 | 2,592,180 | 164.15 | 327.41 | 6.09 | 0.05 | 0.31 | `20260910-164504` |
| yolo11n@mtsd | YOLO11-MTSD / FINAL_yolo11n_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | 5.22 | 2,592,180 | 164.17 | 326.69 | 6.09 | 0.05 | 0.31 | `20260910-164504` |
| yolo11s | YOLO11-MTSD / E01_yolo11s_mtsdqa1aug_img960_eb32_e100_adamw_s42 | 18.29 | 9,432,436 | 170.98 | 332.53 | 5.85 | 0.07 | 0.25 | `20260910-164504` |
| yolo11s@mtsd | YOLO11-MTSD / E03_yolo11s_mtsdqa1aug_img1280_eb32_e100_adamw_s42 | 18.33 | 9,432,436 | 166.76 | 330.87 | 6.00 | 0.07 | 0.15 | `20260910-164504` |
| yolo11s@mtsd | YOLO11-MTSD / E08_yolo11s_mtsdqa1aug_img640_eb32_e100_adamw_s42 | 18.27 | 9,432,436 | 166.40 | 324.89 | 6.01 | 0.07 | 0.16 | `20260910-164504` |
| yolo11s@mtsd | YOLO11-MTSD / FINAL_yolo11s_mtsd_noaug_img640_pb32_eb64_e100_adamw_s42 | 18.27 | 9,432,436 | 165.69 | 330.72 | 6.04 | 0.08 | 0.36 | `20260910-164504` |
| yolo11s@mtsd | YOLO11-MTSD / FINAL_yolo11s_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | 18.27 | 9,432,436 | 165.96 | 323.88 | 6.03 | 0.07 | 0.38 | `20260910-164504` |
| yolo12m | YOLO12-MTSD / E02_yolo12m_mtsdqa1noaug_img640_eb32_e100_adamw_s42 | 38.87 | 20,146,740 | 176.43 | 337.27 | 5.67 | 0.13 | 0.46 | `20260910-164504` |
| yolo12m@mtsd | YOLO12-MTSD / E03_yolo12m_mtsdqa1aug_img640_eb32_e100_adamw_s42 | 38.87 | 20,146,740 | 178.57 | 346.24 | 5.60 | 0.13 | 0.21 | `20260910-164504` |
| yolo12m@mtsd | YOLO12-MTSD / E03_yolo12m_mtsdqa1aug_img960_eb32_e100_adamw_s42 | 154.53 | 20,146,740 | 179.12 | 336.47 | 5.58 | 0.21 | 0.25 | `20260910-164504` |
| yolo12m@mtsd | YOLO12-MTSD / FINAL_yolo12m_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | 38.87 | 20,146,740 | 175.52 | 329.43 | 5.70 | 0.13 | 0.33 | `20260910-164504` |
| yolo12m@mtsd | YOLO12-MTSD / PROBE_yolo12m_mtsd_strong_img640_pb32_eb64_e2_adamw_s42 | 38.86 | 20,146,740 | 177.32 | 336.65 | 5.64 | 0.13 | 0.20 | `20260910-164504` |
| yolo12n | YOLO12-MTSD / E01_yolo12n_mtsdqa1aug_img640_eb32_e100_adamw_s42 | 5.24 | 2,570,388 | 167.45 | 318.95 | 5.97 | 0.06 | 0.34 | `20260910-164504` |
| yolo12n@mtsd | YOLO12-MTSD / FINAL_yolo12n_mtsd_strong_img1280_pb8_eb64_e100_adamw_s42 | 5.31 | 2,570,388 | 167.73 | 325.56 | 5.96 | 0.06 | 0.25 | `20260910-164504` |
| yolo12n@mtsd | YOLO12-MTSD / FINAL_yolo12n_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | 5.24 | 2,570,388 | 168.07 | 328.24 | 5.95 | 0.06 | 0.26 | `20260910-164504` |
| yolo12n@mtsd | YOLO12-MTSD / FINAL_yolo12n_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | 5.26 | 2,570,388 | 168.56 | 322.62 | 5.93 | 0.06 | 0.27 | `20260910-164504` |
| yolo12s | YOLO12-MTSD / E02_yolo12s_mtsdqa1aug_img640_eb32_e100_adamw_s42 | 18.04 | 9,257,780 | 168.96 | 331.33 | 5.92 | 0.20 | 0.10 | `20260910-164504` |
| yolo12s@mtsd | YOLO12-MTSD / FINAL_yolo12s_mtsd_noaug_img640_pb32_eb64_e100_adamw_s42 | 18.04 | 9,257,780 | 168.44 | 318.91 | 5.94 | 0.08 | 0.32 | `20260910-164504` |
| yolo12s@mtsd | YOLO12-MTSD / FINAL_yolo12s_mtsd_strong_img1280_pb8_eb64_e100_adamw_s42 | 71.40 | 9,257,780 | 167.81 | 328.06 | 5.96 | 0.14 | 0.34 | `20260910-164504` |
| yolo12s@mtsd | YOLO12-MTSD / FINAL_yolo12s_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | 18.04 | 9,257,780 | 169.06 | 323.12 | 5.91 | 0.19 | 0.10 | `20260910-164504` |
| yolo26l | YOLO26-MTSD / E13_yolo26l_mtsdqa1aug_img640_eb32_e100_adamw_s42 | 50.53 | 26,194,848 | 182.30 | 346.85 | 5.49 | 0.27 | 0.20 | `20260910-164504` |
| yolo26m | YOLO26-MTSD / E03_yolo26m_mtsdqa1noaug_img640_eb32_e100_adamw_s42 | 41.99 | 21,791,392 | 178.59 | 341.18 | 5.60 | 0.13 | 0.31 | `20260910-164504` |
| yolo26m@mtsd | YOLO26-MTSD / E06_yolo26m_mtsdqa1aug_img640_eb32_e100_adamw_s42 | 41.99 | 21,791,392 | 174.85 | 337.22 | 5.72 | 0.13 | 0.42 | `20260910-164504` |
| yolo26m@mtsd | YOLO26-MTSD / FINAL_yolo26m_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | 41.99 | 21,791,392 | 176.81 | 340.16 | 5.66 | 0.13 | 0.26 | `20260910-164504` |
| yolo26m@mtsd | YOLO26-MTSD / FINAL_yolo26m_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | 42.01 | 21,791,392 | 175.90 | 339.99 | 5.68 | 0.14 | 0.33 | `20260910-164504` |
| yolo26n | YOLO26-MTSD / E04_yolo26n_mtsdqa1aug_img640_eb32_e100_adamw_s42 | 5.12 | 2,508,480 | 167.42 | 328.81 | 5.97 | 0.06 | 0.28 | `20260910-164504` |
| yolo26n@mtsd | YOLO26-MTSD / FINAL_yolo26n_mtsd_strong_img1280_pb8_eb64_e100_adamw_s42 | 5.19 | 2,508,480 | 165.61 | 325.65 | 6.04 | 0.06 | 0.27 | `20260910-164504` |
| yolo26n@mtsd | YOLO26-MTSD / FINAL_yolo26n_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | 5.12 | 2,508,480 | 166.95 | 332.49 | 5.99 | 0.06 | 0.27 | `20260910-164504` |
| yolo26n@mtsd | YOLO26-MTSD / FINAL_yolo26n_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | 5.14 | 2,508,480 | 165.57 | 327.55 | 6.04 | 0.06 | 0.28 | `20260910-164504` |
| yolo26s | YOLO26-MTSD / E02_yolo26s_mtsdqa1aug_img1280_eb32_e100_adamw_s42 | 19.42 | 9,957,152 | 169.64 | 331.84 | 5.89 | 0.09 | 0.21 | `20260910-164504` |
| yolo26s@mtsd | YOLO26-MTSD / E05_yolo26s_mtsdqa1aug_img640_eb32_e100_adamw_s42 | 19.36 | 9,957,152 | 166.56 | 330.36 | 6.00 | 0.09 | 0.33 | `20260910-164504` |
| yolo26s@mtsd | YOLO26-MTSD / FINAL_yolo26s_mtsd_noaug_img1280_pb8_eb64_e100_adamw_s42 | 19.43 | 9,957,152 | 166.16 | 323.06 | 6.02 | 0.09 | 0.34 | `20260910-164504` |
| yolo26s@mtsd | YOLO26-MTSD / FINAL_yolo26s_mtsd_noaug_img640_pb32_eb64_e100_adamw_s42 | 19.36 | 9,957,152 | 166.67 | 326.36 | 6.00 | 0.09 | 0.37 | `20260910-164504` |
| yolo26s@mtsd | YOLO26-MTSD / FINAL_yolo26s_mtsd_noaug_img960_pb16_eb64_e100_adamw_s42 | 19.39 | 9,957,152 | 166.23 | 329.95 | 6.02 | 0.09 | 0.39 | `20260910-164504` |
| yolo26s@mtsd | YOLO26-MTSD / FINAL_yolo26s_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | 19.36 | 9,957,152 | 166.91 | 322.99 | 5.99 | 0.09 | 0.42 | `20260910-164504` |

## MTSD — prompt (6 models)

| Model | Suite / trained run | Artifact MB | Parameters | Mean ms | p95 ms | FPS | Cold start s | Peak GPU GB | Source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| cosmos_reason2_2b | — | — | — | 1574.21 | 1674.61 | 0.64 | 7.14 | 4.77 | `20260910-165249` |
| cosmos_reason2_32b | — | — | — | 190793.05 | 618314.79 | 0.01 | 526.51 | 20.59 | `20260910-165249` |
| cosmos_reason2_8b | — | — | — | 2435.88 | 6286.78 | 0.41 | 16.88 | 17.73 | `20260910-165249` |
| locateanything_3b | — | — | — | 3794.80 | 5242.92 | 0.26 | 16.79 | 19.35 | `20260910-165249` |
| sam_3 | — | — | — | 234.88 | 377.24 | 4.26 | 9.01 | 7.40 | `20260910-165249` |
| sam_3.1 | — | — | — | 257.11 | 548.78 | 3.89 | 5.80 | 8.31 | `20260910-165249` |
