# Robustness slice analysis

Generated: 2026-09-14T12:36:10  |  Commit: `2a158e051662`

Performance across data slices, computed **only from stored predictions and ground truth** (no model was run). Full machine-readable results: `robustness_slice_results.csv/.json`; thresholds and matching rules: `robustness_slice_config.json`; gaps: `insufficient_or_missing_inputs.md`.

## MTSD-attributes - status: **historical snapshot**

> Historical GRP-1..GRP-3 snapshot; a final-scope round is still pending.


Slices deviating >= 0.03 from their aggregate (sufficient support only):

| Model | Split | Slice | Metric | Value | Aggregate | Delta | Support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vjepa_lora | test | condition_class=Heavily Damaged | f1 | 0.3182 | 0.5948 | -0.2767 | 30 |
| convnext_base_finetune | test | condition_class=Heavily Damaged | f1 | 0.4094 | 0.6502 | -0.2409 | 110 |
| vjepa | test | condition_class=Heavily Damaged | f1 | 0.2887 | 0.511 | -0.2224 | 30 |
| dinov3 | test | condition_class=Heavily Damaged | f1 | 0.3462 | 0.5534 | -0.2073 | 30 |
| lingbot_vitl_frozen | test | condition_class=Heavily Damaged | f1 | 0.3519 | 0.5578 | -0.206 | 110 |
| vjepa21_vitl_frozen | test | condition_class=Heavily Damaged | f1 | 0.3623 | 0.5601 | -0.1978 | 110 |
| lingbot_vitb_frozen | test | condition_class=Heavily Damaged | f1 | 0.3353 | 0.5306 | -0.1953 | 110 |
| dinov3_vitb_frozen | test | condition_class=Heavily Damaged | f1 | 0.388 | 0.5767 | -0.1888 | 110 |
| convnext_large_lora | test | condition_class=Heavily Damaged | f1 | 0.5048 | 0.6891 | -0.1844 | 110 |
| vjepa21_vitb_frozen | test | condition_class=Heavily Damaged | f1 | 0.3755 | 0.5568 | -0.1813 | 110 |
| convnext_large_frozen | test | condition_class=Weathered | f1 | 0.3844 | 0.5538 | -0.1694 | 392 |
| convnext_base_frozen | test | condition_class=Heavily Damaged | f1 | 0.3647 | 0.5339 | -0.1692 | 110 |
| convnext_frozen | test | condition_class=Heavily Damaged | f1 | 0.4043 | 0.5673 | -0.1631 | 30 |
| convnext_base_lora | test | condition_class=Heavily Damaged | f1 | 0.5436 | 0.7022 | -0.1586 | 110 |
| dinov3_vitl_frozen | test | condition_class=Heavily Damaged | f1 | 0.4264 | 0.5835 | -0.1572 | 110 |
| dinov3_vitb_lora | test | condition_class=Heavily Damaged | f1 | 0.5463 | 0.6924 | -0.1461 | 110 |
| vjepa | test | condition_class=Weathered | f1 | 0.3681 | 0.511 | -0.1429 | 85 |
| convnext_frozen | test | mounting_class=Wall-Mounted | f1 | 0.6621 | 0.8026 | -0.1405 | 59 |
| dinov3_vitl_lora | test | condition_class=Heavily Damaged | f1 | 0.5951 | 0.7282 | -0.133 | 110 |
| convnext_large_frozen | test | condition_class=Heavily Damaged | f1 | 0.4225 | 0.5538 | -0.1312 | 110 |

> 6 slice rows have support below 15 and are marked `insufficient_support`.

## MTSD-detection - status: **historical snapshot**

> Completed stored-run inventory; compare only runs with matching split, taxonomy and protocol.

| Model | Run | Split | Metric | Value | Images | Objects |
| --- | --- | --- | --- | --- | --- | --- |
| RF-DETR m | Results/MTSD-Runs/RF-DETR-MTSD/FINAL_rfdetrm_mtsd_noaug_img576_pb32_eb32_e100_adamw_s42 | valid | precision | 0.7402 | 749 | 1993 |
| RF-DETR m | Results/MTSD-Runs/RF-DETR-MTSD/FINAL_rfdetrm_mtsd_noaug_img576_pb32_eb32_e100_adamw_s42 | valid | recall | 0.7762 | 749 | 1993 |
| RF-DETR m | Results/MTSD-Runs/RF-DETR-MTSD/FINAL_rfdetrm_mtsd_noaug_img576_pb32_eb32_e100_adamw_s42 | valid | f1 | 0.7578 | 749 | 1993 |
| RF-DETR m | Results/MTSD-Runs/RF-DETR-MTSD/FINAL_rfdetrm_mtsd_noaug_img576_pb32_eb32_e100_adamw_s42 | valid | mean_matched_iou | 0.9132 | 749 | 1993 |
| RF-DETR m | Results/MTSD-Runs/RF-DETR-MTSD/FINAL_rfdetrm_mtsd_strong_img576_pb32_eb32_e100_adamw_s42 | valid | precision | 0.7934 | 749 | 1993 |
| RF-DETR m | Results/MTSD-Runs/RF-DETR-MTSD/FINAL_rfdetrm_mtsd_strong_img576_pb32_eb32_e100_adamw_s42 | valid | recall | 0.7747 | 749 | 1993 |
| RF-DETR m | Results/MTSD-Runs/RF-DETR-MTSD/FINAL_rfdetrm_mtsd_strong_img576_pb32_eb32_e100_adamw_s42 | valid | f1 | 0.784 | 749 | 1993 |
| RF-DETR m | Results/MTSD-Runs/RF-DETR-MTSD/FINAL_rfdetrm_mtsd_strong_img576_pb32_eb32_e100_adamw_s42 | valid | mean_matched_iou | 0.9196 | 749 | 1993 |
| RF-DETR s | Results/MTSD-Runs/RF-DETR-MTSD/FINAL_rfdetrs_mtsd_strong_img512_pb32_eb32_e100_adamw_s42 | valid | precision | 0.7435 | 749 | 1993 |
| RF-DETR s | Results/MTSD-Runs/RF-DETR-MTSD/FINAL_rfdetrs_mtsd_strong_img512_pb32_eb32_e100_adamw_s42 | valid | recall | 0.7722 | 749 | 1993 |
| RF-DETR s | Results/MTSD-Runs/RF-DETR-MTSD/FINAL_rfdetrs_mtsd_strong_img512_pb32_eb32_e100_adamw_s42 | valid | f1 | 0.7576 | 749 | 1993 |
| RF-DETR s | Results/MTSD-Runs/RF-DETR-MTSD/FINAL_rfdetrs_mtsd_strong_img512_pb32_eb32_e100_adamw_s42 | valid | mean_matched_iou | 0.9105 | 749 | 1993 |
| RF-DETR m | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-cuda-rfdetr-m-img640-fixed-b16ga2-s42 | valid | precision | 0.725 | 749 | 1993 |
| RF-DETR m | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-cuda-rfdetr-m-img640-fixed-b16ga2-s42 | valid | recall | 0.8003 | 749 | 1993 |
| RF-DETR m | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-cuda-rfdetr-m-img640-fixed-b16ga2-s42 | valid | f1 | 0.7608 | 749 | 1993 |
| RF-DETR m | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-cuda-rfdetr-m-img640-fixed-b16ga2-s42 | valid | mean_matched_iou | 0.9145 | 749 | 1993 |
| RF-DETR n | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-cuda-rfdetr-n-img384-s42 | test | precision | 0.7668 | 747 | 1975 |
| RF-DETR n | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-cuda-rfdetr-n-img384-s42 | test | recall | 0.6992 | 747 | 1975 |
| RF-DETR n | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-cuda-rfdetr-n-img384-s42 | test | f1 | 0.7315 | 747 | 1975 |
| RF-DETR n | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-cuda-rfdetr-n-img384-s42 | test | mean_matched_iou | 0.9125 | 747 | 1975 |
| RF-DETR n | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-cuda-rfdetr-n-img640-fixed-b16ga2-s42-resume3 | valid | precision | 0.7851 | 749 | 1993 |
| RF-DETR n | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-cuda-rfdetr-n-img640-fixed-b16ga2-s42-resume3 | valid | recall | 0.7792 | 749 | 1993 |
| RF-DETR n | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-cuda-rfdetr-n-img640-fixed-b16ga2-s42-resume3 | valid | f1 | 0.7822 | 749 | 1993 |
| RF-DETR n | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-cuda-rfdetr-n-img640-fixed-b16ga2-s42-resume3 | valid | mean_matched_iou | 0.9157 | 749 | 1993 |
| RF-DETR s | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-cuda-rfdetr-s-img640-fixed-b16ga2-s42 | valid | precision | 0.8066 | 749 | 1993 |
| RF-DETR s | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-cuda-rfdetr-s-img640-fixed-b16ga2-s42 | valid | recall | 0.7913 | 749 | 1993 |
| RF-DETR s | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-cuda-rfdetr-s-img640-fixed-b16ga2-s42 | valid | f1 | 0.7989 | 749 | 1993 |
| RF-DETR s | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-cuda-rfdetr-s-img640-fixed-b16ga2-s42 | valid | mean_matched_iou | 0.9192 | 749 | 1993 |
| RF-DETR m | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-followup-rfdetr-m-img704-s42 | test | precision | 0.7386 | 747 | 1975 |
| RF-DETR m | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-followup-rfdetr-m-img704-s42 | test | recall | 0.8142 | 747 | 1975 |
| RF-DETR m | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-followup-rfdetr-m-img704-s42 | test | f1 | 0.7746 | 747 | 1975 |
| RF-DETR m | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-followup-rfdetr-m-img704-s42 | test | mean_matched_iou | 0.9159 | 747 | 1975 |
| RF-DETR s | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-followup-rfdetr-s-img640-s42 | test | precision | 0.7459 | 747 | 1975 |
| RF-DETR s | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-followup-rfdetr-s-img640-s42 | test | recall | 0.8056 | 747 | 1975 |
| RF-DETR s | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-followup-rfdetr-s-img640-s42 | test | f1 | 0.7746 | 747 | 1975 |
| RF-DETR s | Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-followup-rfdetr-s-img640-s42 | test | mean_matched_iou | 0.9114 | 747 | 1975 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/E01_yolo11m_mtsdqa1aug_img1280_eb32_e100_adamw_s42 | test | precision | 0.8018 | 747 | 1975 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/E01_yolo11m_mtsdqa1aug_img1280_eb32_e100_adamw_s42 | test | recall | 0.7499 | 747 | 1975 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/E01_yolo11m_mtsdqa1aug_img1280_eb32_e100_adamw_s42 | test | f1 | 0.775 | 747 | 1975 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/E01_yolo11m_mtsdqa1aug_img1280_eb32_e100_adamw_s42 | test | mean_matched_iou | 0.9322 | 747 | 1975 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/E01_yolo11m_mtsdqa1noaug_img640_eb32_e100_adamw_s42 | test | precision | 0.7679 | 747 | 1975 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/E01_yolo11m_mtsdqa1noaug_img640_eb32_e100_adamw_s42 | test | recall | 0.6182 | 747 | 1975 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/E01_yolo11m_mtsdqa1noaug_img640_eb32_e100_adamw_s42 | test | f1 | 0.685 | 747 | 1975 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/E01_yolo11m_mtsdqa1noaug_img640_eb32_e100_adamw_s42 | test | mean_matched_iou | 0.9377 | 747 | 1975 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/E01_yolo11s_mtsdqa1aug_img960_eb32_e100_adamw_s42 | test | precision | 0.7711 | 747 | 1975 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/E01_yolo11s_mtsdqa1aug_img960_eb32_e100_adamw_s42 | test | recall | 0.6906 | 747 | 1975 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/E01_yolo11s_mtsdqa1aug_img960_eb32_e100_adamw_s42 | test | f1 | 0.7286 | 747 | 1975 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/E01_yolo11s_mtsdqa1aug_img960_eb32_e100_adamw_s42 | test | mean_matched_iou | 0.9313 | 747 | 1975 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/E02_yolo11m_mtsdqa1aug_img960_eb32_e100_adamw_s42 | test | precision | 0.805 | 747 | 1975 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/E02_yolo11m_mtsdqa1aug_img960_eb32_e100_adamw_s42 | test | recall | 0.6962 | 747 | 1975 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/E02_yolo11m_mtsdqa1aug_img960_eb32_e100_adamw_s42 | test | f1 | 0.7467 | 747 | 1975 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/E02_yolo11m_mtsdqa1aug_img960_eb32_e100_adamw_s42 | test | mean_matched_iou | 0.9359 | 747 | 1975 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/E03_yolo11s_mtsdqa1aug_img1280_eb32_e100_adamw_s42 | test | precision | 0.7892 | 747 | 1975 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/E03_yolo11s_mtsdqa1aug_img1280_eb32_e100_adamw_s42 | test | recall | 0.7296 | 747 | 1975 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/E03_yolo11s_mtsdqa1aug_img1280_eb32_e100_adamw_s42 | test | f1 | 0.7582 | 747 | 1975 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/E03_yolo11s_mtsdqa1aug_img1280_eb32_e100_adamw_s42 | test | mean_matched_iou | 0.9292 | 747 | 1975 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11m_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | precision | 0.7949 | 749 | 1993 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11m_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | recall | 0.7391 | 749 | 1993 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11m_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | f1 | 0.766 | 749 | 1993 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11m_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | mean_matched_iou | 0.9266 | 749 | 1993 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11m_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | valid | precision | 0.8255 | 749 | 1993 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11m_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | valid | recall | 0.7973 | 749 | 1993 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11m_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | valid | f1 | 0.8111 | 749 | 1993 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11m_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | valid | mean_matched_iou | 0.9285 | 749 | 1993 |
| YOLO11 n | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11n_mtsd_strong_img1280_pb8_eb64_e100_adamw_s42 | valid | precision | 0.8041 | 749 | 1993 |
| YOLO11 n | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11n_mtsd_strong_img1280_pb8_eb64_e100_adamw_s42 | valid | recall | 0.7682 | 749 | 1993 |
| YOLO11 n | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11n_mtsd_strong_img1280_pb8_eb64_e100_adamw_s42 | valid | f1 | 0.7857 | 749 | 1993 |
| YOLO11 n | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11n_mtsd_strong_img1280_pb8_eb64_e100_adamw_s42 | valid | mean_matched_iou | 0.928 | 749 | 1993 |
| YOLO11 n | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11n_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | precision | 0.7651 | 749 | 1993 |
| YOLO11 n | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11n_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | recall | 0.6538 | 749 | 1993 |
| YOLO11 n | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11n_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | f1 | 0.7051 | 749 | 1993 |
| YOLO11 n | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11n_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | mean_matched_iou | 0.9282 | 749 | 1993 |
| YOLO11 n | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11n_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | valid | precision | 0.7854 | 749 | 1993 |
| YOLO11 n | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11n_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | valid | recall | 0.7105 | 749 | 1993 |
| YOLO11 n | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11n_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | valid | f1 | 0.746 | 749 | 1993 |
| YOLO11 n | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11n_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | valid | mean_matched_iou | 0.9267 | 749 | 1993 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11s_mtsd_noaug_img640_pb32_eb64_e100_adamw_s42 | valid | precision | 0.7406 | 749 | 1993 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11s_mtsd_noaug_img640_pb32_eb64_e100_adamw_s42 | valid | recall | 0.6247 | 749 | 1993 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11s_mtsd_noaug_img640_pb32_eb64_e100_adamw_s42 | valid | f1 | 0.6777 | 749 | 1993 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11s_mtsd_noaug_img640_pb32_eb64_e100_adamw_s42 | valid | mean_matched_iou | 0.9277 | 749 | 1993 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11s_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | precision | 0.777 | 749 | 1993 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11s_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | recall | 0.7045 | 749 | 1993 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11s_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | f1 | 0.7389 | 749 | 1993 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/FINAL_yolo11s_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | mean_matched_iou | 0.9252 | 749 | 1993 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/strongaug-wandb-cuda-yolo11s-img1280-s42 | test | precision | 0.7653 | 747 | 1975 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/strongaug-wandb-cuda-yolo11s-img1280-s42 | test | recall | 0.8041 | 747 | 1975 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/strongaug-wandb-cuda-yolo11s-img1280-s42 | test | f1 | 0.7842 | 747 | 1975 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/strongaug-wandb-cuda-yolo11s-img1280-s42 | test | mean_matched_iou | 0.9278 | 747 | 1975 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/strongaug-wandb-cuda-yolo11s-img960-s42 | test | precision | 0.7905 | 747 | 1975 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/strongaug-wandb-cuda-yolo11s-img960-s42 | test | recall | 0.7585 | 747 | 1975 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/strongaug-wandb-cuda-yolo11s-img960-s42 | test | f1 | 0.7742 | 747 | 1975 |
| YOLO11 s | Results/MTSD-Runs/YOLO11-MTSD/strongaug-wandb-cuda-yolo11s-img960-s42 | test | mean_matched_iou | 0.9272 | 747 | 1975 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/strongaug-wandb-followup-yolo11m-img1280-s42 | test | precision | 0.7951 | 747 | 1975 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/strongaug-wandb-followup-yolo11m-img1280-s42 | test | recall | 0.8233 | 747 | 1975 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/strongaug-wandb-followup-yolo11m-img1280-s42 | test | f1 | 0.809 | 747 | 1975 |
| YOLO11 m | Results/MTSD-Runs/YOLO11-MTSD/strongaug-wandb-followup-yolo11m-img1280-s42 | test | mean_matched_iou | 0.933 | 747 | 1975 |
| YOLO12 m | Results/MTSD-Runs/YOLO12-MTSD/E02_yolo12m_mtsdqa1noaug_img640_eb32_e100_adamw_s42 | test | precision | 0.7037 | 747 | 1975 |
| YOLO12 m | Results/MTSD-Runs/YOLO12-MTSD/E02_yolo12m_mtsdqa1noaug_img640_eb32_e100_adamw_s42 | test | recall | 0.6289 | 747 | 1975 |
| YOLO12 m | Results/MTSD-Runs/YOLO12-MTSD/E02_yolo12m_mtsdqa1noaug_img640_eb32_e100_adamw_s42 | test | f1 | 0.6642 | 747 | 1975 |
| YOLO12 m | Results/MTSD-Runs/YOLO12-MTSD/E02_yolo12m_mtsdqa1noaug_img640_eb32_e100_adamw_s42 | test | mean_matched_iou | 0.9235 | 747 | 1975 |
| YOLO12 m | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12m_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | precision | 0.7891 | 749 | 1993 |
| YOLO12 m | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12m_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | recall | 0.7245 | 749 | 1993 |
| YOLO12 m | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12m_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | f1 | 0.7554 | 749 | 1993 |
| YOLO12 m | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12m_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | mean_matched_iou | 0.9276 | 749 | 1993 |
| YOLO12 n | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12n_mtsd_strong_img1280_pb8_eb64_e100_adamw_s42 | valid | precision | 0.7595 | 749 | 1993 |
| YOLO12 n | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12n_mtsd_strong_img1280_pb8_eb64_e100_adamw_s42 | valid | recall | 0.7842 | 749 | 1993 |
| YOLO12 n | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12n_mtsd_strong_img1280_pb8_eb64_e100_adamw_s42 | valid | f1 | 0.7717 | 749 | 1993 |
| YOLO12 n | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12n_mtsd_strong_img1280_pb8_eb64_e100_adamw_s42 | valid | mean_matched_iou | 0.9288 | 749 | 1993 |
| YOLO12 n | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12n_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | precision | 0.7323 | 749 | 1993 |
| YOLO12 n | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12n_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | recall | 0.6438 | 749 | 1993 |
| YOLO12 n | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12n_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | f1 | 0.6852 | 749 | 1993 |
| YOLO12 n | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12n_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | mean_matched_iou | 0.9285 | 749 | 1993 |
| YOLO12 n | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12n_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | valid | precision | 0.7704 | 749 | 1993 |
| YOLO12 n | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12n_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | valid | recall | 0.7391 | 749 | 1993 |
| YOLO12 n | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12n_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | valid | f1 | 0.7544 | 749 | 1993 |
| YOLO12 n | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12n_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | valid | mean_matched_iou | 0.9245 | 749 | 1993 |
| YOLO12 s | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12s_mtsd_noaug_img640_pb32_eb64_e100_adamw_s42 | valid | precision | 0.8004 | 749 | 1993 |
| YOLO12 s | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12s_mtsd_noaug_img640_pb32_eb64_e100_adamw_s42 | valid | recall | 0.6217 | 749 | 1993 |
| YOLO12 s | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12s_mtsd_noaug_img640_pb32_eb64_e100_adamw_s42 | valid | f1 | 0.6998 | 749 | 1993 |
| YOLO12 s | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12s_mtsd_noaug_img640_pb32_eb64_e100_adamw_s42 | valid | mean_matched_iou | 0.9291 | 749 | 1993 |
| YOLO12 s | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12s_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | precision | 0.7733 | 749 | 1993 |
| YOLO12 s | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12s_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | recall | 0.707 | 749 | 1993 |
| YOLO12 s | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12s_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | f1 | 0.7387 | 749 | 1993 |
| YOLO12 s | Results/MTSD-Runs/YOLO12-MTSD/FINAL_yolo12s_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | mean_matched_iou | 0.9241 | 749 | 1993 |
| YOLO12 s | Results/MTSD-Runs/YOLO12-MTSD/strongaug-wandb-cuda-yolo12s-img960-s42 | test | precision | 0.7643 | 747 | 1975 |
| YOLO12 s | Results/MTSD-Runs/YOLO12-MTSD/strongaug-wandb-cuda-yolo12s-img960-s42 | test | recall | 0.7489 | 747 | 1975 |
| YOLO12 s | Results/MTSD-Runs/YOLO12-MTSD/strongaug-wandb-cuda-yolo12s-img960-s42 | test | f1 | 0.7565 | 747 | 1975 |
| YOLO12 s | Results/MTSD-Runs/YOLO12-MTSD/strongaug-wandb-cuda-yolo12s-img960-s42 | test | mean_matched_iou | 0.9225 | 747 | 1975 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/E02_yolo26s_mtsdqa1aug_img1280_eb32_e100_adamw_s42 | test | precision | 0.8107 | 747 | 1975 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/E02_yolo26s_mtsdqa1aug_img1280_eb32_e100_adamw_s42 | test | recall | 0.722 | 747 | 1975 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/E02_yolo26s_mtsdqa1aug_img1280_eb32_e100_adamw_s42 | test | f1 | 0.7638 | 747 | 1975 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/E02_yolo26s_mtsdqa1aug_img1280_eb32_e100_adamw_s42 | test | mean_matched_iou | 0.9329 | 747 | 1975 |
| YOLO26 m | Results/MTSD-Runs/YOLO26-MTSD/E03_yolo26m_mtsdqa1noaug_img640_eb32_e100_adamw_s42 | test | precision | 0.8345 | 747 | 1975 |
| YOLO26 m | Results/MTSD-Runs/YOLO26-MTSD/E03_yolo26m_mtsdqa1noaug_img640_eb32_e100_adamw_s42 | test | recall | 0.6203 | 747 | 1975 |
| YOLO26 m | Results/MTSD-Runs/YOLO26-MTSD/E03_yolo26m_mtsdqa1noaug_img640_eb32_e100_adamw_s42 | test | f1 | 0.7116 | 747 | 1975 |
| YOLO26 m | Results/MTSD-Runs/YOLO26-MTSD/E03_yolo26m_mtsdqa1noaug_img640_eb32_e100_adamw_s42 | test | mean_matched_iou | 0.9408 | 747 | 1975 |
| YOLO26 m | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26m_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | precision | 0.8382 | 749 | 1993 |
| YOLO26 m | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26m_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | recall | 0.7486 | 749 | 1993 |
| YOLO26 m | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26m_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | f1 | 0.7909 | 749 | 1993 |
| YOLO26 m | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26m_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | mean_matched_iou | 0.9304 | 749 | 1993 |
| YOLO26 m | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26m_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | valid | precision | 0.8522 | 749 | 1993 |
| YOLO26 m | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26m_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | valid | recall | 0.7958 | 749 | 1993 |
| YOLO26 m | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26m_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | valid | f1 | 0.823 | 749 | 1993 |
| YOLO26 m | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26m_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | valid | mean_matched_iou | 0.9338 | 749 | 1993 |
| YOLO26 n | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26n_mtsd_strong_img1280_pb8_eb64_e100_adamw_s42 | valid | precision | 0.8293 | 749 | 1993 |
| YOLO26 n | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26n_mtsd_strong_img1280_pb8_eb64_e100_adamw_s42 | valid | recall | 0.7657 | 749 | 1993 |
| YOLO26 n | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26n_mtsd_strong_img1280_pb8_eb64_e100_adamw_s42 | valid | f1 | 0.7962 | 749 | 1993 |
| YOLO26 n | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26n_mtsd_strong_img1280_pb8_eb64_e100_adamw_s42 | valid | mean_matched_iou | 0.9326 | 749 | 1993 |
| YOLO26 n | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26n_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | precision | 0.8092 | 749 | 1993 |
| YOLO26 n | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26n_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | recall | 0.6322 | 749 | 1993 |
| YOLO26 n | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26n_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | f1 | 0.7099 | 749 | 1993 |
| YOLO26 n | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26n_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | mean_matched_iou | 0.9317 | 749 | 1993 |
| YOLO26 n | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26n_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | valid | precision | 0.8035 | 749 | 1993 |
| YOLO26 n | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26n_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | valid | recall | 0.712 | 749 | 1993 |
| YOLO26 n | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26n_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | valid | f1 | 0.755 | 749 | 1993 |
| YOLO26 n | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26n_mtsd_strong_img960_pb16_eb64_e100_adamw_s42 | valid | mean_matched_iou | 0.9311 | 749 | 1993 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26s_mtsd_noaug_img1280_pb8_eb64_e100_adamw_s42 | valid | precision | 0.8347 | 749 | 1993 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26s_mtsd_noaug_img1280_pb8_eb64_e100_adamw_s42 | valid | recall | 0.7245 | 749 | 1993 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26s_mtsd_noaug_img1280_pb8_eb64_e100_adamw_s42 | valid | f1 | 0.7757 | 749 | 1993 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26s_mtsd_noaug_img1280_pb8_eb64_e100_adamw_s42 | valid | mean_matched_iou | 0.9341 | 749 | 1993 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26s_mtsd_noaug_img640_pb32_eb64_e100_adamw_s42 | valid | precision | 0.848 | 749 | 1993 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26s_mtsd_noaug_img640_pb32_eb64_e100_adamw_s42 | valid | recall | 0.6157 | 749 | 1993 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26s_mtsd_noaug_img640_pb32_eb64_e100_adamw_s42 | valid | f1 | 0.7134 | 749 | 1993 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26s_mtsd_noaug_img640_pb32_eb64_e100_adamw_s42 | valid | mean_matched_iou | 0.9353 | 749 | 1993 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26s_mtsd_noaug_img960_pb16_eb64_e100_adamw_s42 | valid | precision | 0.8244 | 749 | 1993 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26s_mtsd_noaug_img960_pb16_eb64_e100_adamw_s42 | valid | recall | 0.6879 | 749 | 1993 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26s_mtsd_noaug_img960_pb16_eb64_e100_adamw_s42 | valid | f1 | 0.75 | 749 | 1993 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26s_mtsd_noaug_img960_pb16_eb64_e100_adamw_s42 | valid | mean_matched_iou | 0.9328 | 749 | 1993 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26s_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | precision | 0.8138 | 749 | 1993 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26s_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | recall | 0.704 | 749 | 1993 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26s_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | f1 | 0.7549 | 749 | 1993 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/FINAL_yolo26s_mtsd_strong_img640_pb32_eb64_e100_adamw_s42 | valid | mean_matched_iou | 0.9322 | 749 | 1993 |
| YOLO26 l | Results/MTSD-Runs/YOLO26-MTSD/strongaug-wandb-cuda-yolo26l-img640-b16ga2-s42 | valid | precision | 0.8106 | 749 | 1993 |
| YOLO26 l | Results/MTSD-Runs/YOLO26-MTSD/strongaug-wandb-cuda-yolo26l-img640-b16ga2-s42 | valid | recall | 0.7536 | 749 | 1993 |
| YOLO26 l | Results/MTSD-Runs/YOLO26-MTSD/strongaug-wandb-cuda-yolo26l-img640-b16ga2-s42 | valid | f1 | 0.7811 | 749 | 1993 |
| YOLO26 l | Results/MTSD-Runs/YOLO26-MTSD/strongaug-wandb-cuda-yolo26l-img640-b16ga2-s42 | valid | mean_matched_iou | 0.9308 | 749 | 1993 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/strongaug-wandb-cuda-yolo26s-img1280-s42 | test | precision | 0.8401 | 747 | 1975 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/strongaug-wandb-cuda-yolo26s-img1280-s42 | test | recall | 0.7954 | 747 | 1975 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/strongaug-wandb-cuda-yolo26s-img1280-s42 | test | f1 | 0.8172 | 747 | 1975 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/strongaug-wandb-cuda-yolo26s-img1280-s42 | test | mean_matched_iou | 0.9345 | 747 | 1975 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/strongaug-wandb-cuda-yolo26s-img960-s42 | test | precision | 0.8398 | 747 | 1975 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/strongaug-wandb-cuda-yolo26s-img960-s42 | test | recall | 0.7509 | 747 | 1975 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/strongaug-wandb-cuda-yolo26s-img960-s42 | test | f1 | 0.7928 | 747 | 1975 |
| YOLO26 s | Results/MTSD-Runs/YOLO26-MTSD/strongaug-wandb-cuda-yolo26s-img960-s42 | test | mean_matched_iou | 0.9358 | 747 | 1975 |
| YOLO26 m | Results/MTSD-Runs/YOLO26-MTSD/strongaug-wandb-followup-yolo26m-img1280-s42 | test | precision | 0.8285 | 747 | 1975 |
| YOLO26 m | Results/MTSD-Runs/YOLO26-MTSD/strongaug-wandb-followup-yolo26m-img1280-s42 | test | recall | 0.8172 | 747 | 1975 |
| YOLO26 m | Results/MTSD-Runs/YOLO26-MTSD/strongaug-wandb-followup-yolo26m-img1280-s42 | test | f1 | 0.8228 | 747 | 1975 |
| YOLO26 m | Results/MTSD-Runs/YOLO26-MTSD/strongaug-wandb-followup-yolo26m-img1280-s42 | test | mean_matched_iou | 0.9337 | 747 | 1975 |

Slices deviating >= 0.03 from their aggregate (sufficient support only):

| Model | Split | Slice | Metric | Value | Aggregate | Delta | Support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RF-DETR m | valid | object_size=small | recall | 0.1283 | 0.7762 | -0.6479 | 187 |
| RF-DETR s | valid | object_size=small | recall | 0.1283 | 0.7722 | -0.6439 | 187 |
| RF-DETR m | valid | object_size=small | recall | 0.139 | 0.7747 | -0.6357 | 187 |
| YOLO12 n | valid | object_size=small | recall | 0.0107 | 0.6438 | -0.6331 | 187 |
| RF-DETR n | test | object_size=small | recall | 0.0674 | 0.6992 | -0.6318 | 193 |
| YOLO11 n | valid | object_size=small | recall | 0.0321 | 0.6538 | -0.6217 | 187 |
| YOLO12 s | valid | object_size=small | recall | 0.0856 | 0.707 | -0.6214 | 187 |
| RF-DETR m | valid | object_size=small | recall | 0.1818 | 0.8003 | -0.6185 | 187 |
| YOLO26 n | valid | object_size=small | recall | 0.0321 | 0.6322 | -0.6001 | 187 |
| RF-DETR s | valid | object_size=small | recall | 0.1925 | 0.7913 | -0.5988 | 187 |
| RF-DETR n | valid | object_size=small | recall | 0.1818 | 0.7792 | -0.5974 | 187 |
| YOLO12 m | valid | object_size=small | recall | 0.1283 | 0.7245 | -0.5962 | 187 |
| YOLO11 n | valid | object_size=small | recall | 0.123 | 0.7105 | -0.5875 | 187 |
| YOLO12 s | valid | object_size=small | recall | 0.0374 | 0.6217 | -0.5843 | 187 |
| YOLO12 n | valid | object_size=small | recall | 0.1551 | 0.7391 | -0.584 | 187 |
| YOLO26 n | valid | object_size=small | recall | 0.1283 | 0.712 | -0.5837 | 187 |
| YOLO11 s | valid | object_size=small | recall | 0.0428 | 0.6247 | -0.5819 | 187 |
| YOLO11 m | valid | object_size=small | recall | 0.1658 | 0.7391 | -0.5733 | 187 |
| YOLO11 s | valid | object_size=small | recall | 0.1337 | 0.7045 | -0.5708 | 187 |
| YOLO26 s | valid | object_size=small | recall | 0.0535 | 0.6157 | -0.5622 | 187 |

## Figures

- `Documents/Final-Figures/Robustness-Slices/20260914-123609/mtsd_attr_macro_f1_by_head.png`
- `Documents/Final-Figures/Robustness-Slices/20260914-123609/mtsd_attr_condition_per_class_f1.png`
- `Documents/Final-Figures/Robustness-Slices/20260914-123609/mtsd_attr_view_angle_per_class_f1.png`

*Read-only analysis - no datasets, checkpoints or previous results were modified.*
