# Repository health check

Generated: 2026-07-07T17:06:51  |  Overall: **FAIL**

| Check | Status | Findings |
| --- | --- | --- |
| expected_folders | PASS | 0 |
| key_readmes | PASS | 0 |
| result_folders_preserved | PASS | 0 |
| notebooks_valid_json | PASS | 0 |
| python_compiles | PASS | 0 |
| pattern_old_grp_path | FAIL | 60 |
| pattern_bare_grp_images | WARN | 1 |
| pattern_old_repo_name | WARN | 34 |
| pattern_abs_users_path | WARN | 11 |
| pattern_wandb_key | PASS | 0 |
| cross_MTSD-SupervisedNotebooks | PASS | 0 |
| cross_MDWD-SupervisedNotebooks | PASS | 0 |
| markdown_links | WARN | 1 |
| env_gitignored | PASS | 0 |
| automation_targets | PASS | 0 |

## pattern_old_grp_path (FAIL)

*pre-migration dataset path (Datasets/GRP-n instead of Datasets/MTSD/GRP-n)*

- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:87: "source_image": "Datasets/GRP-1/Images/0062fb9d-IMG_4749.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:109: "source_image": "Datasets/GRP-1/Images/0062fb9d-IMG_4749.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:131: "source_image": "Datasets/GRP-1/Images/0062fb9d-IMG_4749.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:153: "source_image": "Datasets/GRP-1/Images/0062fb9d-IMG_4749.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:175: "source_image": "Datasets/GRP-1/Images/0062fb9d-IMG_4749.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:197: "source_image": "Datasets/GRP-1/Images/0062fb9d-IMG_4749.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:219: "source_image": "Datasets/GRP-1/Images/0062fb9d-IMG_4749.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:241: "source_image": "Datasets/GRP-1/Images/00801c04-IMG_4632.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:263: "source_image": "Datasets/GRP-1/Images/00801c04-IMG_4632.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:285: "source_image": "Datasets/GRP-1/Images/0091344f-RV_001_51.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:307: "source_image": "Datasets/GRP-1/Images/0091344f-RV_001_51.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:329: "source_image": "Datasets/GRP-1/Images/0091344f-RV_001_51.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:351: "source_image": "Datasets/GRP-1/Images/009f06df-IMG20260105121707.jpg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:373: "source_image": "Datasets/GRP-1/Images/009f06df-IMG20260105121707.jpg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:395: "source_image": "Datasets/GRP-1/Images/009f06df-IMG20260105121707.jpg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:417: "source_image": "Datasets/GRP-1/Images/009f06df-IMG20260105121707.jpg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:439: "source_image": "Datasets/GRP-1/Images/009f06df-IMG20260105121707.jpg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:461: "source_image": "Datasets/GRP-1/Images/009f06df-IMG20260105121707.jpg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:483: "source_image": "Datasets/GRP-1/Images/009f06df-IMG20260105121707.jpg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:505: "source_image": "Datasets/GRP-1/Images/028195a4-IMG20251123165145.jpg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:527: "source_image": "Datasets/GRP-1/Images/02995950-IMG20260105125533.jpg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:549: "source_image": "Datasets/GRP-1/Images/02ac92f5-IMG20251123165127.jpg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:571: "source_image": "Datasets/GRP-1/Images/02ac92f5-IMG20251123165127.jpg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:593: "source_image": "Datasets/GRP-1/Images/02b98ac6-IMG20251124134652.jpg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:615: "source_image": "Datasets/GRP-1/Images/02b98ac6-IMG20251124134652.jpg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:637: "source_image": "Datasets/GRP-1/Images/02dcf721-IMG_5638.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:659: "source_image": "Datasets/GRP-1/Images/02fdc633-RV_001_161.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:681: "source_image": "Datasets/GRP-1/Images/02fdc633-RV_001_161.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:703: "source_image": "Datasets/GRP-1/Images/02fdc633-RV_001_161.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:725: "source_image": "Datasets/GRP-1/Images/02fdc633-RV_001_161.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:747: "source_image": "Datasets/GRP-1/Images/02fdc633-RV_001_161.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:769: "source_image": "Datasets/GRP-1/Images/02fdc633-RV_001_161.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:791: "source_image": "Datasets/GRP-1/Images/02fdc633-RV_001_161.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:813: "source_image": "Datasets/GRP-1/Images/02fdc633-RV_001_161.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:835: "source_image": "Datasets/GRP-1/Images/031abd7e-RV_001_85.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:857: "source_image": "Datasets/GRP-1/Images/031abd7e-RV_001_85.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:879: "source_image": "Datasets/GRP-1/Images/031abd7e-RV_001_85.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:901: "source_image": "Datasets/GRP-1/Images/031abd7e-RV_001_85.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:923: "source_image": "Datasets/GRP-1/Images/031abd7e-RV_001_85.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:945: "source_image": "Datasets/GRP-1/Images/031abd7e-RV_001_85.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:967: "source_image": "Datasets/GRP-1/Images/031abd7e-RV_001_85.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:989: "source_image": "Datasets/GRP-1/Images/031abd7e-RV_001_85.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:1011: "source_image": "Datasets/GRP-1/Images/031abd7e-RV_001_85.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:1033: "source_image": "Datasets/GRP-1/Images/0342ceaf-IMG_4649.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:1055: "source_image": "Datasets/GRP-1/Images/035a9593-IMG_5818.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:1077: "source_image": "Datasets/GRP-1/Images/035a9593-IMG_5818.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:1099: "source_image": "Datasets/GRP-1/Images/035a9593-IMG_5818.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:1121: "source_image": "Datasets/GRP-1/Images/0396b469-IMG_20260108_180701.jpg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:1143: "source_image": "Datasets/GRP-1/Images/0396b469-IMG_20260108_180701.jpg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:1165: "source_image": "Datasets/GRP-1/Images/0396b469-IMG_20260108_180701.jpg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:1187: "source_image": "Datasets/GRP-1/Images/0396b469-IMG_20260108_180701.jpg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:1209: "source_image": "Datasets/GRP-1/Images/0396b469-IMG_20260108_180701.jpg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:1231: "source_image": "Datasets/GRP-1/Images/03b2f434-IMG_5771.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:1253: "source_image": "Datasets/GRP-1/Images/03b9fcf6-RV_001_140.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:1275: "source_image": "Datasets/GRP-1/Images/03b9fcf6-RV_001_140.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:1297: "source_image": "Datasets/GRP-1/Images/03b9fcf6-RV_001_140.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:1319: "source_image": "Datasets/GRP-1/Images/03b9fcf6-RV_001_140.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:1341: "source_image": "Datasets/GRP-1/Images/042d1365-IMG_5571.jpeg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:1363: "source_image": "Datasets/GRP-1/Images/043a38a9-IMG20260105131300.jpg",`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\manifests\manifest.json:1385: "source_image": "Datasets/GRP-1/Images/045c881d-RV_001_63.jpeg",`

## pattern_bare_grp_images (WARN)

*GRP-n/Images path without the Datasets/MTSD prefix on the same line*

- `Scripts\Other-Scripts\GDPR-Compliance\README.md:52: GRP-1/Images/<name>.jpeg               # redacted copy (EXIF kept, orientation baked)`

## pattern_old_repo_name (WARN)

*reference to the pre-reorganisation repository layout*

- `Datasets\MTSD\Annotations\GRP-1\Final-QA\QA-GRP1.json:4: "source_file": "E:\\2. UM-Student\\MSC Dissertation\\MTSDataset\\TEST\\LabelStudioData\\export\\project-2-at-2`
- `Datasets\MTSD\Annotations\GRP-2\Final-QA\QA-GRP2.json:4: "source_file": "E:\\2. UM-Student\\MSC Dissertation\\MTSDataset\\Datasets\\Annotations\\GRP-2\\Final-QA\\QA-GR`
- `Datasets\MTSD\Annotations\GRP-3\Final-QA\QA-GRP3.json:4: "source_file": "E:\\2. UM-Student\\MSC Dissertation\\MTSDataset\\Datasets\\Annotations\\GRP-3\\Final-QA\\QA-GR`
- `Scripts\Automation\verify_repository_health.py:102: ("old_repo_name", "WARN", re.compile(r"MTSDataset|Dataset-Versions|AICOM-YOLO|Jupyter Notebooks[\\/]"),`
- `Scripts\MDWD-Scripts\MDWD-Analysis\MDWD-EDA-OG.ipynb:26: target_subfolder = os.path.join(versions_path, "AICOM-YOLOv26")`
- `Scripts\MDWD-Scripts\MDWD-Analysis\MDWD-EDA-OG.ipynb:68: renamed_folder = os.path.join(current_folder, "AICOM-YOLOv26")`
- `Scripts\MDWD-Scripts\MDWD-Analysis\MDWD-EDA-OG.ipynb:69: target_folder = os.path.join(versions_path, "AICOM-YOLOv26")`
- `Scripts\MDWD-Scripts\MDWD-Analysis\MDWD-EDA-OG.ipynb:137: DATASET_ROOT = "../../Versions/AICOM-YOLOv26"  # Path to the root folder of your YOLO dataset`
- `Scripts\MDWD-Scripts\MDWD-Analysis\README.md:18: | `DatasetVisualisation.ipynb`, `DatasetVisualisation-OG.ipynb` | **Legacy** notebooks from the pre-reorganisa`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:1: {"run_id": "convnext-20260703-021114-smoke", "variant": "convnext", "smoke_test": true, "start_time": "2026-07`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:2: {"run_id": "vjepa-20260703-021156-smoke", "variant": "vjepa", "smoke_test": true, "start_time": "2026-07-03T00`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:3: {"run_id": "dinov3-20260703-021232-smoke", "variant": "dinov3", "smoke_test": true, "start_time": "2026-07-03T`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:4: {"run_id": "vjepa-20260703-021534-smoke", "variant": "vjepa", "smoke_test": true, "start_time": "2026-07-03T00`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:5: {"run_id": "dinov3-20260703-021817", "variant": "dinov3", "smoke_test": false, "start_time": "2026-07-03T00:18`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:6: {"run_id": "vjepa-20260703-023332", "variant": "vjepa", "smoke_test": false, "start_time": "2026-07-03T00:33:4`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:7: {"run_id": "convnext-20260703-025354", "variant": "convnext", "smoke_test": false, "start_time": "2026-07-03T0`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:8: {"run_id": "dinov3-20260703-150747-smoke", "variant": "dinov3", "adaptation": "frozen", "smoke_test": true, "s`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:9: {"run_id": "vjepa-20260703-150755-smoke", "variant": "vjepa", "adaptation": "frozen", "smoke_test": true, "sta`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:10: {"run_id": "convnext_frozen-20260703-150808-smoke", "variant": "convnext_frozen", "adaptation": "frozen", "smo`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:11: {"run_id": "convnext-20260703-150817-smoke", "variant": "convnext", "adaptation": "finetune", "smoke_test": tr`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:12: {"run_id": "dinov3_lora-20260703-150833-smoke", "variant": "dinov3_lora", "adaptation": "lora", "smoke_test": `
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:13: {"run_id": "vjepa_lora-20260703-150838-smoke", "variant": "vjepa_lora", "adaptation": "lora", "smoke_test": tr`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:14: {"run_id": "convnext_frozen-20260703-151019-smoke", "variant": "convnext_frozen", "adaptation": "frozen", "smo`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:15: {"run_id": "dinov3-20260703-151103-smoke", "variant": "dinov3", "adaptation": "frozen", "smoke_test": true, "s`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:16: {"run_id": "vjepa-20260703-151111-smoke", "variant": "vjepa", "adaptation": "frozen", "smoke_test": true, "sta`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:17: {"run_id": "convnext_frozen-20260703-151123-smoke", "variant": "convnext_frozen", "adaptation": "frozen", "smo`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:18: {"run_id": "convnext-20260703-151131-smoke", "variant": "convnext", "adaptation": "finetune", "smoke_test": tr`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:19: {"run_id": "dinov3-20260704-010441", "variant": "dinov3", "adaptation": "frozen", "smoke_test": false, "start_`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:20: {"run_id": "vjepa-20260704-011610", "variant": "vjepa", "adaptation": "frozen", "smoke_test": false, "start_ti`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:21: {"run_id": "convnext_frozen-20260704-013448", "variant": "convnext_frozen", "adaptation": "frozen", "smoke_tes`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:22: {"run_id": "convnext-20260704-015608", "variant": "convnext", "adaptation": "finetune", "smoke_test": false, "`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:23: {"run_id": "dinov3_lora-20260704-020900", "variant": "dinov3_lora", "adaptation": "lora", "smoke_test": false,`
- `Scripts\MTSD-Scripts\AttributeClassification\outputs\experiment_log.json:24: {"run_id": "vjepa_lora-20260704-022043", "variant": "vjepa_lora", "adaptation": "lora", "smoke_test": false, "`
- `Scripts\MTSD-Scripts\update_annotation_paths.ps1:8: MTSDataset repository root, so their "source_image" entries look like:`

## pattern_abs_users_path (WARN)

*absolute Windows user path (fine for env docs, wrong in portable code)*

- `Documents\Other\CleanModelCache.md:134: Hugging Face hub cache: C:\Users\<you>\.cache\huggingface\hub`
- `Documents\Other\CloudflareTunnel.md:61: C:\Users\<YOUR_USERNAME>\.cloudflared\`
- `Documents\Other\CloudflareTunnel.md:162: C:\Users\<YOUR_USERNAME>\.cloudflared\cert.pem`
- `Documents\Other\CloudflareTunnel.md:186: C:\Users\<YOUR_USERNAME>\.cloudflared\<TUNNEL_ID>.json`
- `Documents\Other\CloudflareTunnel.md:222: C:\Users\<YOUR_USERNAME>\.cloudflared\config.yml`
- `Documents\Other\CloudflareTunnel.md:235: credentials-file: C:\Users\<YOUR_USERNAME>\.cloudflared\<TUNNEL_ID>.json`
- `Documents\Other\CloudflareTunnel.md:246: credentials-file: C:\Users\fridge\.cloudflared\f40eb9d6-4a86-4756-8eae-df45718d6865.json`
- `Scripts\MTSD-Scripts\AttributeClassification\inference\gradio_compare.py:4: PYTHONNOUSERSITE=1 C:/Users/fridge/anaconda3/envs/mtsd-attrcls/python.exe -s inference/gradio_compare.py`
- `Scripts\MTSD-Scripts\AttributeClassification\README.md:88: C:/Users/fridge/anaconda3/envs/mtsd-attrcls/python.exe -m pip install torch torchvision --index-url https://do`
- `Scripts\MTSD-Scripts\AttributeClassification\README.md:89: C:/Users/fridge/anaconda3/envs/mtsd-attrcls/python.exe -m pip install -r Requirements/requirements-attribute-c`
- `Scripts\MTSD-Scripts\AttributeClassification\README.md:98: PYTHONNOUSERSITE=1 C:/Users/fridge/anaconda3/envs/mtsd-attrcls/python.exe -s run_all.py`

## markdown_links (WARN)

*Relative links in key Markdown files resolve*

- `Requirements\CondaEnvironments\README.md: (../../Documents/CleanModelCache.md)`

*Read-only report; nothing was modified. Re-run via `python Scripts/Automation/verify_repository_health.py`.*
