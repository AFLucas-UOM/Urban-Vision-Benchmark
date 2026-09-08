# Evaluation chapter evidence audit

Methodology is authoritative. The chapter is `evaluation.tex`; the existing `evalutation.tex` entry point includes it, so `main.tex` needs no change. No figures were created or inserted. Figure requests remain LaTeX comments.

## Decisions and limitations

- MDWD: use explicit `test_` columns from the family summaries, not the unprefixed validation values used by the July consolidated table. Exclude YOLO11-L and YOLO26-DGX from the principal Methodology matrix. RF-DETR-N training artifacts exist, but completed RF-DETR-N/S/M test exports were not located. No paper numbers or architecture claims were invented.
- MTSD: preserve the complete July mild baseline separately from later strong runs. The August generated main-scale report mixes validation, probe and test rows; it is not used as a numerical authority. Original metrics JSONs are used. Missing strong test rows remain missing, even where validation or completed training exists.
- The MTSD unified evaluator computes micro P/R/F1 at the best confidence on the evaluated split (`_common_confidence_f1`). Test values are explicitly labelled descriptive test-sweep maxima, not validation-selected operating points.
- Methodology requires Tourist Sign exclusion from training and evaluation, with retained ignore regions. Existing run records can retain the training taxonomy; `_filtered_evaluation_payload` deletes excluded categories/annotations and class-labelled predictions rather than ignoring predictions spatially overlapping excluded boxes. The prompt targeted matcher also has no explicit overlap-ignore step. Chapter reports existing exports with a prominent limitation and does not claim these are fully protocol-compliant final numbers. No correction was estimated or inference rerun.
- Main MDWD prompt scores and timing use the four canonical rows of the completed bounded sensitivity run. All four targeted concepts are retained; their exported per-query metrics are averaged. This avoids older Cosmos execution settings. MDWD broad queries from older execution records are not represented as final bounded-protocol results.
- Main MTSD prompting uses the completed optimized-v2 export, whose final run configuration and rows identify model-specific Cosmos max sides and chunk size 8. Eleven class-targeted/synonym queries are averaged; two broad queries are separate. Final bounded sensitivity results use the later MDWD/MTSD sensitivity exports.
- Prompt matcher actually chooses the highest-IoU target, then marks an already-used target as a duplicate; it does not try another unmatched GT. The chapter describes this implementation rather than relying on the module's more general docstring.
- Attribute results use all 18 non-smoke test_metrics.json files with 1,890 crops, not historical six-model/503-crop results. Stored 2,000-resample percentile intervals preserve shared crop resampling across heads but do not cluster source images. No unperformed paired test is claimed.
- Existing MDWD bootstrap/slices at confidence 0.25 are labelled separately from native detector summaries. Historical attribute and prompt-pilot intervals/slices are excluded from current conclusions. No compatible final full-split prompt CI or MTSD image-condition slice report was located.
- Cross-paradigm observations explicitly distinguish class/query scope, macro/micro averaging, thresholds and post-processing. No new harmonised benchmark or end-to-end deployment result is claimed.
- Runtime boundaries are distinct: native MDWD inference; MTSD prediction loop including loading/conversion; prompt per-image-query pipeline; batched attribute forward and end-to-end timings. No edge hardware claim, total VRAM estimate or unmeasured model file size is inferred.
- The optional foreign-sign subsection records the absence of a supported annotated benchmark. Tiling and Ox Alpha are excluded from principal results because they are outside Methodology's retained model/protocol scope.

## Table source groups

- `mdwd-benchmark`: Results/MDWD-Results/{YOLO11,YOLO12,YOLO26}/Model-Size-Comparison/*_summary.csv; explicit test_ columns only.
- `mdwd-runtime`: Results/MDWD-Results/{YOLO11,YOLO12,YOLO26}/Model-Size-Comparison/*_summary.csv; test_inference_time_ms and total_training_time_seconds.
- `mdwd-classes`: Documents/Final-Reports/Robustness-Slices/20260710-120353/robustness_slice_results.csv; YOLO26-EUVIP, test, class slices.
- `robustness`: Documents/Final-Reports/Robustness-Slices/20260710-120353/robustness_slice_results.csv and robustness_slice_config.json; yolo26l@YOLO26-EUVIP test only.
- `mdwd-ci`: Documents/Final-Reports/Statistical-Uncertainty/20260710-120457/bootstrap_results.csv and bootstrap_config.json; yolo26l@YOLO26-EUVIP test only.
- `mtsd-benchmark`: Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/*/{metrics_summary,evaluation_record}.json.
- `mtsd-strong`: Results/MTSD-Results/Unified-Evaluation-NoTourist/20260818-084730-pending-yolo-test/*/metrics_summary.json; Strong-640 entries only.
- `augmentation`: Results/MTSD-Results/Unified-Evaluation-NoTourist/{20260722-all13,20260818-084730-pending-yolo-test}/*/metrics_summary.json; Results/MTSD-Runs/*/*noaug_img640*/unified_evaluation/metrics_summary.json.
- `resolution-mild`: Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/*/metrics_summary.json; Results/MTSD-Runs/*/*mtsdqa1aug_img*/unified_evaluation/metrics_summary.json.
- `resolution`: Results/MTSD-Results/Unified-Evaluation-NoTourist/20260818-084730-pending-yolo-test/*/metrics_summary.json; Results/MTSD-Runs/*/strongaug-*/unified_evaluation/metrics_summary.json.
- `size-ap`: Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/rfdetr-m/metrics_summary.json; 20260818-084730-pending-yolo-test/YOLO26M-Strong-{640,960}/metrics_summary.json; Results/MTSD-Runs/YOLO26-MTSD/strongaug-wandb-followup-yolo26m-img1280-s42/unified_evaluation/metrics_summary.json.
- `mtsd-classes`: Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/rfdetr-m/metrics_summary.json; Results/MTSD-Runs/YOLO26-MTSD/strongaug-wandb-followup-yolo26m-img1280-s42/unified_evaluation/metrics_summary.json; per_class fields.
- `mtsd-runtime`: Results/MTSD-Results/Unified-Evaluation-NoTourist/20260818-084730-pending-yolo-test/*/metrics_summary.json; Results/MTSD-Runs/*/strongaug-*/unified_evaluation/metrics_summary.json; inference_latency_ms_per_image and inference_throughput_images_per_second.
- `attributes`: Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/*/test_metrics.json; 18 non-smoke configurations, n_images=1890. Numerical confusion matrices used directly.
- `attribute-heads`: Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/*/test_metrics.json; 18 non-smoke configurations, n_images=1890. Numerical confusion matrices used directly.
- `attribute-accuracy`: Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/*/test_metrics.json; 18 non-smoke configurations, n_images=1890. Numerical confusion matrices used directly.
- `attribute-classes`: Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/*/test_metrics.json; 18 non-smoke configurations, n_images=1890. Numerical confusion matrices used directly.
- `condition-vjepa`: Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/*/test_metrics.json; 18 non-smoke configurations, n_images=1890. Numerical confusion matrices used directly.
- `condition-dino`: Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/*/test_metrics.json; 18 non-smoke configurations, n_images=1890. Numerical confusion matrices used directly.
- `attribute-runtime`: Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/*/test_metrics.json; 18 non-smoke configurations, n_images=1890. Numerical confusion matrices used directly.
- `mdwd-sensitivity`: Results/PromptDetect/BatchEvaluation/MDWD/20260821-170445-optimized-v2-bounded-sensitivity/{prompt_sensitivity_per_model,prompt_consistency_per_model}.csv.
- `mdwd-prompt-macro`: Results/PromptDetect/BatchEvaluation/MDWD/20260821-170445-optimized-v2-bounded-sensitivity/per_prompt_metrics.csv; canonical rows only; arithmetic mean of four exported prompt metrics.
- `mdwd-prompt-detail`: Results/PromptDetect/BatchEvaluation/MDWD/20260821-170445-optimized-v2-bounded-sensitivity/per_prompt_metrics.csv; canonical rows only; arithmetic mean of four exported prompt metrics.
- `mdwd-broad`: Results/PromptDetect/BatchEvaluation/MDWD/20260821-170445-optimized-v2-bounded-sensitivity/per_prompt_metrics.csv; canonical rows only; arithmetic mean of four exported prompt metrics.
- `mtsd-sensitivity`: Results/PromptDetect/BatchEvaluation/MTSD/20260822-102455-optimized-v2-bounded-sensitivity/{prompt_sensitivity_per_model,prompt_consistency_per_model}.csv.
- `mtsd-prompt-macro`: Results/PromptDetect/BatchEvaluation/MTSD/20260818-134723-optimized-v2/per_prompt_metrics.csv; class-targeted and synonym-comparison macro; broad group separately.
- `mtsd-prompt-detail`: Results/PromptDetect/BatchEvaluation/MTSD/20260818-134723-optimized-v2/per_prompt_metrics.csv; class-targeted and synonym-comparison macro; broad group separately.
- `mtsd-broad`: Results/PromptDetect/BatchEvaluation/MTSD/20260818-134723-optimized-v2/per_prompt_metrics.csv; class-targeted and synonym-comparison macro; broad group separately.

## Exact numerical files read

- `Documents/Final-Reports/Robustness-Slices/20260710-120353/robustness_slice_results.csv`
- `Documents/Final-Reports/Statistical-Uncertainty/20260710-120457/bootstrap_results.csv`
- `Results/MDWD-Results/YOLO11/Model-Size-Comparison/yolo11_multi_model_template_summary.csv`
- `Results/MDWD-Results/YOLO12/Model-Size-Comparison/yolo12_multi_model_template_summary.csv`
- `Results/MDWD-Results/YOLO26/Model-Size-Comparison/yolo26_multi_model_template_summary.csv`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/rfdetr-m/evaluation_record.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/rfdetr-m/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/rfdetr-n/evaluation_record.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/rfdetr-n/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/rfdetr-s/evaluation_record.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/rfdetr-s/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/yolo11m/evaluation_record.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/yolo11m/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/yolo11n/evaluation_record.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/yolo11n/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/yolo11s/evaluation_record.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/yolo11s/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/yolo12m/evaluation_record.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/yolo12m/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/yolo12n/evaluation_record.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/yolo12n/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/yolo12s/evaluation_record.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/yolo12s/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/yolo26l/evaluation_record.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/yolo26l/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/yolo26m/evaluation_record.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/yolo26m/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/yolo26n/evaluation_record.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/yolo26n/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/yolo26s/evaluation_record.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260722-all13/yolo26s/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260818-084730-pending-yolo-test/YOLO11M-Strong-640/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260818-084730-pending-yolo-test/YOLO11M-Strong-960/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260818-084730-pending-yolo-test/YOLO11N-Strong-640/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260818-084730-pending-yolo-test/YOLO11N-Strong-960/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260818-084730-pending-yolo-test/YOLO11S-NoAug-640/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260818-084730-pending-yolo-test/YOLO11S-Strong-640/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260818-084730-pending-yolo-test/YOLO12N-Strong-1280/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260818-084730-pending-yolo-test/YOLO12N-Strong-960/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260818-084730-pending-yolo-test/YOLO26M-Strong-640/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260818-084730-pending-yolo-test/YOLO26M-Strong-960/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260818-084730-pending-yolo-test/YOLO26N-Strong-640/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260818-084730-pending-yolo-test/YOLO26N-Strong-960/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260818-084730-pending-yolo-test/YOLO26S-NoAug-640/metrics_summary.json`
- `Results/MTSD-Results/Unified-Evaluation-NoTourist/20260818-084730-pending-yolo-test/YOLO26S-Strong-640/metrics_summary.json`
- `Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-cuda-rfdetr-n-img384-s42/unified_evaluation/metrics_summary.json`
- `Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-followup-rfdetr-m-img704-s42/unified_evaluation/metrics_summary.json`
- `Results/MTSD-Runs/RF-DETR-MTSD/strongaug-wandb-followup-rfdetr-s-img640-s42/unified_evaluation/metrics_summary.json`
- `Results/MTSD-Runs/YOLO11-MTSD/E01_yolo11m_mtsdqa1aug_img1280_eb32_e100_adamw_s42/unified_evaluation/metrics_summary.json`
- `Results/MTSD-Runs/YOLO11-MTSD/E01_yolo11m_mtsdqa1noaug_img640_eb32_e100_adamw_s42/unified_evaluation/metrics_summary.json`
- `Results/MTSD-Runs/YOLO11-MTSD/E01_yolo11s_mtsdqa1aug_img960_eb32_e100_adamw_s42/unified_evaluation/metrics_summary.json`
- `Results/MTSD-Runs/YOLO11-MTSD/E02_yolo11m_mtsdqa1aug_img960_eb32_e100_adamw_s42/unified_evaluation/metrics_summary.json`
- `Results/MTSD-Runs/YOLO11-MTSD/E03_yolo11s_mtsdqa1aug_img1280_eb32_e100_adamw_s42/unified_evaluation/metrics_summary.json`
- `Results/MTSD-Runs/YOLO11-MTSD/strongaug-wandb-cuda-yolo11s-img1280-s42/unified_evaluation/metrics_summary.json`
- `Results/MTSD-Runs/YOLO11-MTSD/strongaug-wandb-cuda-yolo11s-img960-s42/unified_evaluation/metrics_summary.json`
- `Results/MTSD-Runs/YOLO11-MTSD/strongaug-wandb-followup-yolo11m-img1280-s42/unified_evaluation/metrics_summary.json`
- `Results/MTSD-Runs/YOLO12-MTSD/E02_yolo12m_mtsdqa1noaug_img640_eb32_e100_adamw_s42/unified_evaluation/metrics_summary.json`
- `Results/MTSD-Runs/YOLO12-MTSD/strongaug-wandb-cuda-yolo12s-img960-s42/unified_evaluation/metrics_summary.json`
- `Results/MTSD-Runs/YOLO26-MTSD/E02_yolo26s_mtsdqa1aug_img1280_eb32_e100_adamw_s42/unified_evaluation/metrics_summary.json`
- `Results/MTSD-Runs/YOLO26-MTSD/E03_yolo26m_mtsdqa1noaug_img640_eb32_e100_adamw_s42/unified_evaluation/metrics_summary.json`
- `Results/MTSD-Runs/YOLO26-MTSD/strongaug-wandb-cuda-yolo26s-img1280-s42/unified_evaluation/metrics_summary.json`
- `Results/MTSD-Runs/YOLO26-MTSD/strongaug-wandb-cuda-yolo26s-img960-s42/unified_evaluation/metrics_summary.json`
- `Results/MTSD-Runs/YOLO26-MTSD/strongaug-wandb-followup-yolo26m-img1280-s42/unified_evaluation/metrics_summary.json`
- `Results/PromptDetect/BatchEvaluation/MDWD/20260821-170445-optimized-v2-bounded-sensitivity/per_prompt_metrics.csv`
- `Results/PromptDetect/BatchEvaluation/MDWD/20260821-170445-optimized-v2-bounded-sensitivity/prompt_consistency_per_model.csv`
- `Results/PromptDetect/BatchEvaluation/MDWD/20260821-170445-optimized-v2-bounded-sensitivity/prompt_sensitivity_per_model.csv`
- `Results/PromptDetect/BatchEvaluation/MTSD/20260818-134723-optimized-v2/per_prompt_metrics.csv`
- `Results/PromptDetect/BatchEvaluation/MTSD/20260822-102455-optimized-v2-bounded-sensitivity/prompt_consistency_per_model.csv`
- `Results/PromptDetect/BatchEvaluation/MTSD/20260822-102455-optimized-v2-bounded-sensitivity/prompt_sensitivity_per_model.csv`
- `Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/convnext_base_finetune/test_metrics.json`
- `Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/convnext_base_frozen/test_metrics.json`
- `Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/convnext_base_lora/test_metrics.json`
- `Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/convnext_large_finetune/test_metrics.json`
- `Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/convnext_large_frozen/test_metrics.json`
- `Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/convnext_large_lora/test_metrics.json`
- `Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/dinov3_vitb_frozen/test_metrics.json`
- `Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/dinov3_vitb_lora/test_metrics.json`
- `Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/dinov3_vitl_frozen/test_metrics.json`
- `Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/dinov3_vitl_lora/test_metrics.json`
- `Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/lingbot_vitb_frozen/test_metrics.json`
- `Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/lingbot_vitb_lora/test_metrics.json`
- `Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/lingbot_vitl_frozen/test_metrics.json`
- `Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/lingbot_vitl_lora/test_metrics.json`
- `Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/vjepa21_vitb_frozen/test_metrics.json`
- `Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/vjepa21_vitb_lora/test_metrics.json`
- `Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/vjepa21_vitl_frozen/test_metrics.json`
- `Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/vjepa21_vitl_lora/test_metrics.json`

## Validation

Table values were generated from the listed CSV/JSON fields (percent scaling and rounding only, plus explicitly described macro means and reciprocal FPS). Structural checks cover sections, table column counts, LaTeX environment balance, unique labels, resolved internal references and acronym keys. A TeX compiler is not available in the current environment, so PDF compilation and page-layout inspection remain unverified.

Completed checks: 8 sections, 30 subsections, 27 tables; internal references/acronyms and table column counts pass. Current MTSD test exports share 747 images and 1,975 evaluated targets. Both controlled prompt studies have four variants per model/family with invariant target/positive/negative counts. All 18 current attribute accuracy and macro-F1 results independently match their saved confusion matrices. No figure/image commands occur in the chapter.
