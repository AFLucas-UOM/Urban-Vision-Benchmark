# MTSD model-training hyperparameters

This document consolidates the effective hyperparameters used to train models on the Maltese Traffic Sign Dataset (MTSD). It covers the two trainable MTSD pipelines in this repository:

1. supervised object detection (YOLO11, YOLO12, YOLO26 and RF-DETR); and
2. multi-attribute sign classification (DINOv3, V-JEPA 2.1, ConvNeXt and LingBot-Vision).

PromptDetect models are not included because they are evaluated zero-shot/prompt-based and are not trained on MTSD. Smoke tests and failed feasibility probes are also excluded from the full-run mappings below.

## 1. Supervised object detection

### 1.1 Shared and framework-specific training hyperparameters

| Hyperparameter | YOLO11 / YOLO12 / YOLO26 | RF-DETR N / S / M |
|---|---|---|
| Initialisation | Pretrained repository checkpoint | Pretrained repository checkpoint |
| Source-level data split | 80% train / 10% validation / 10% test; seed 42 | Same split and seed |
| Maximum epochs | 100 | 100 |
| Early stopping | Patience 10; validation-selected `best.pt` | Enabled; patience 10; minimum delta 0.001; best regular/EMA/total checkpoints retained |
| Optimizer | AdamW | AdamW |
| Base learning rate | 0.001 | 0.001 |
| Additional learning-rate settings | Final LR factor (`lrf`) 0.01; linear schedule (`cos_lr=false`) | Encoder LR 0.00015; ViT layer decay 0.8; component decay 0.7; step schedule with `lr_drop=100` |
| Weight decay | 0.0005 | 0.0005 |
| Warm-up | 3 epochs; warm-up momentum 0.8; warm-up bias LR 0.1 | 3 epochs, linear warm-up |
| Momentum / Adam betas | Ultralytics momentum parameter 0.937 | AdamW library defaults |
| Physical batch | 32 / 16 / 8 at 640 / 960 / 1280 px | 32 |
| Optimizer-effective batch | 64 | 32 |
| Gradient accumulation | 2 / 4 / 8 steps at 640 / 960 / 1280 px | 1 step |
| Automatic mixed precision | Enabled | Enabled |
| Determinism / seed | Deterministic mode enabled; seed 42 | Seed 42; deterministic flag is not exposed by the RF-DETR wrapper |
| DataLoader workers | 8 | 2 |
| Gradient checkpointing | Not enabled explicitly | Enabled |
| Gradient clipping | Ultralytics internal behaviour (not overridden) | Maximum norm 0.1 (RF-DETR default) |
| Dropout / drop path | Dropout 0.0; no layer freezing | Dropout 0.0; drop path 0.0 |
| Loss weights | Box 7.5; classification 0.5; DFL 1.5 | Classification 1.0; box 5; GIoU 2; focal alpha 0.25; auxiliary loss enabled; IA-BCE enabled |
| Matcher costs | Framework defaults | Classification 2; box 5; GIoU 2 |
| Queries / grouped DETR | Not applicable | 300 queries; 300 selected; 13 DETR groups |
| EMA | Ultralytics trainer-managed | Enabled; decay 0.993; tau 100 |
| Multi-scale training | Disabled (`multi_scale=0.0`) | Enabled with expanded scales; square resize divisible by 64; random resize via padding disabled |
| Online training augmentation | Fully disabled: HSV, rotation, translation, scale, shear, perspective, flips, BGR, mosaic, MixUp, CutMix, copy-paste, erasing and AutoAugment are all 0/off; `close_mosaic=10` is inert because mosaic is 0 | RF-DETR 1.3.0 internal random horizontal flip and random crop/resize composition remain active and are not fully controllable through its public API |
| Loader/input mode | Cache disabled; rectangular batches disabled; full dataset fraction 1.0 | Full dataset; COCO/Roboflow loader; square resizing divisible by 64 |
| Checkpoint interval | Ultralytics best/last checkpointing | Every 10 epochs |
| Device used for recorded final runs | CUDA (RTX 4090) | CUDA (RTX 4090) |

**Batch terminology.** The saved YOLO runs use `nbs=64`; therefore the physical batches 32/16/8 imply 2/4/8 accumulation steps and an optimizer-effective batch of 64. RF-DETR uses physical batch 32 with no accumulation, hence effective batch 32. This is the value recorded by the current run records and supersedes older prose that loosely called 32 the YOLO “effective batch”.

### 1.2 Detection model-to-hyperparameter mapping

All settings not shown as different in this table inherit Section 1.1. Resolution lists combine completed full training runs. `Mild` means the original offline recipe (`photometric-v1+motion-blur-v1`); `Strong` means `strong-offline-v2`; `None` means the unaugmented training split.

| Model | Scale | Training resolution(s), px | Physical batch(es) | Accumulation step(s) | Effective batch | Offline dataset regime(s) in full runs | Model-specific note |
|---|---:|---:|---:|---:|---:|---|---|
| YOLO11-N | n | 640 / 960 / 1280 | 32 / 16 / 8 | 2 / 4 / 8 | 64 | Mild / Strong | Same optimizer settings at every resolution |
| YOLO11-S | s | 640 / 960 / 1280 | 32 / 16 / 8 | 2 / 4 / 8 | 64 | None / Mild / Strong | Same optimizer settings at every resolution |
| YOLO11-M | m | 640 / 960 / 1280 | 32 / 16 / 8 | 2 / 4 / 8 | 64 | None / Mild / Strong | Same optimizer settings at every resolution |
| YOLO12-N | n | 640 / 960 / 1280 | 32 / 16 / 8 | 2 / 4 / 8 | 64 | Mild / Strong | Same optimizer settings at every resolution |
| YOLO12-S | s | 640 / 960 | 32 / 16 | 2 / 4 | 64 | None / Mild / Strong | A 1280-px follow-up was attempted but did not produce a completed full-run record |
| YOLO12-M | m | 640 | 32 | 2 | 64 | None / Mild / Strong | A two-epoch feasibility probe is not counted as a full training run |
| YOLO26-N | n | 640 / 960 / 1280 | 32 / 16 / 8 | 2 / 4 / 8 | 64 | Mild / Strong | Same optimizer settings at every resolution |
| YOLO26-S | s | 640 / 960 / 1280 | 32 / 16 / 8 | 2 / 4 / 8 | 64 | None / Mild / Strong | Same optimizer settings at every resolution |
| YOLO26-M | m | 640 / 960 / 1280 | 32 / 16 / 8 | 2 / 4 / 8 | 64 | None / Mild / Strong | Same optimizer settings at every resolution |
| YOLO26-L | l | 640 | 32 | 2 | 64 | Mild | Included as the thirteenth model in the original dissertation matrix |
| RF-DETR-N | nano | 384 | 32 | 1 | 32 | Mild / Strong | Native model resolution |
| RF-DETR-S | small | 512 / 640 | 32 | 1 | 32 | Mild / Strong | 512 is native; 640 is a resolution follow-up override |
| RF-DETR-M | medium | 576 / 704 | 32 | 1 | 32 | None / Mild / Strong | 576 is native; 704 is a resolution follow-up override |

The original 13-model dissertation matrix used 640 px for every YOLO model and the native 384/512/576 px resolutions for RF-DETR N/S/M. The later resolution and strong-augmentation experiments created the additional values shown above; they did not change the optimizer, learning-rate, warm-up, patience or seed settings.

### 1.3 Offline detection augmentation hyperparameters

| Regime / component | Hyperparameters |
|---|---|
| None | No offline copies; source training images only |
| Mild recipe identity | `photometric-v1+motion-blur-v1`; 3 offline copies per source training image; seed 42; no geometric augmentation |
| Mild photometric settings | Brightness 0.8–1.2; contrast 0.85–1.15; colour 0.9–1.1; Gaussian blur probability 0.5, sigma max 1.0; Gaussian noise probability 0.3, sigma max 0.02; JPEG probability 0.3, quality 70–92; gamma 0.9–1.1 |
| Mild motion blur | Applied to copy 3; horizontal; kernel size 9; blur weight 0.85 |
| Strong recipe identity/output | `strong-offline-v2`; 3 offline copies per source image; seed 42; JPEG quality 92; 8 preparation workers; maximum long edge 1920 px |
| Strong basic photometric | Brightness 0.65–1.35 (p=0.75); contrast 0.70–1.30 (p=0.75); saturation/colour 0.80–1.20 (p=0.70); gamma 0.70–1.35 (p=0.70) |
| Strong degradation | Gaussian blur p=0.50, sigma max 1.5; Gaussian noise p=0.50, sigma max 0.04; JPEG p=0.60, quality 45–90; motion blur p=0.35, kernels 7/9/11 |
| Strong illumination | Partial shadow p=0.25; gradient p=0.25; local change p=0.20; haze p=0.15 |
| Strong geometry | Translation fraction 0.10; scale 0.75–1.30; rotation −5° to +5°; shear 2°; perspective enabled with distortion fraction 0.02; geometric photometric p=0.65; maximum border fraction 0.18 |
| Strong random crop | Enabled; p=0.35; minimum retained fraction 0.85 |
| Strong horizontal flip | Disabled; p=0.0; no class remapping |
| Strong box filtering | Visible-area threshold 0.35; minimum width/height 2 px; minimum area 9 px² |
| Strong mosaic | p=0.40; canvas 1280×1280; centre range 0.40–0.60; photometric p=0.35; tiny-object area 16 px²; maximum tiny-object fraction 0.70 |
| Strong copy-paste | Enabled; p=0.20; rare-class maximum 700 instances; frequency quantile 0.30; maximum 2 pastes; scale 0.80–1.25; maximum sign width/height fractions 0.20/0.25; context fraction 0.18; edge feather 0.12; maximum IoU 0.20; maximum context IoU 0.10; 30 placement attempts; vertical jitter 0.15; photometric p=0.45 |
| Strong MixUp | Disabled; p=0.0 |
| Strong rejection policy | Maximum 8 attempts; reject empty samples |

## 2. MTSD multi-attribute classification

### 2.1 Shared classification hyperparameters

| Hyperparameter | Value |
|---|---|
| Task | Four simultaneous heads: view angle, mounting, condition and sign shape |
| Input crop preparation | Bounding-box padding 0.25 per side; minimum original crop dimension 20 px; JPEG quality 95 |
| Data split | 80% train / 10% validation / 10% test; deterministic source-image hashing with key `mtsd-attr-split-v1` |
| Maximum epochs | 150 |
| Optimizer | AdamW |
| Default head LR | 0.001 |
| ConvNeXt fine-tuning head LR | 0.0004 |
| Default trainable-backbone LR | 0.00005 |
| LoRA/adapter LR | 0.0002 |
| Weight decay | 0.05 |
| Scheduler | Per-optimizer-step linear warm-up followed by cosine decay to zero |
| Warm-up | 5 epochs |
| Physical / effective batch | 32×1 or 16×2; effective batch 32 for every full variant |
| DataLoader workers | 4 |
| Automatic mixed precision | Enabled |
| Gradient clipping | Global norm 1.0, applied immediately before the optimizer step |
| Early stopping | Validation mean macro-F1; mode=max; patience 10; minimum delta 0.001 |
| Best-checkpoint selection | Highest validation mean macro-F1 (any improvement, independently of early-stopping delta) |
| Loss | Class-weighted cross-entropy; inverse-frequency weights calculated from training split and normalised to mean 1 |
| Head loss weights | View angle 1.0; mounting 1.0; condition 1.0; sign shape 1.0 |
| Label smoothing | 0.0 |
| Online augmentation | Colour jitter (brightness 0.3, contrast 0.3, saturation 0.2, hue 0.02); rotation ±10°; random-resized-crop scale 0.8–1.0; horizontal flip disabled |
| Probe head | Linear; configured MLP hidden size 512 and dropout 0.2 are unused while `probe.type=linear` |
| Random seed | 42 (Python, NumPy and PyTorch CPU/CUDA) |
| OOM policy | No automatic batch-size reduction; a run fails and requires an explicit configuration change |

### 2.2 Classification model-to-hyperparameter mapping

All variants below use the shared settings in Section 2.1. “Backbone/adapter LR” is shown only when backbone or adapter parameters are trainable.

| Variant | Family / architecture | Adaptation | Resolution, px | Physical batch | Accumulation | Effective batch | Head LR | Backbone / adapter LR | LoRA configuration |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `dinov3_vitb_frozen` | DINOv3 ViT-B/16 | Frozen linear probe | 224 | 32 | 1 | 32 | 0.001 | — | — |
| `dinov3_vitl_frozen` | DINOv3 ViT-L/16 | Frozen linear probe | 224 | 32 | 1 | 32 | 0.001 | — | — |
| `dinov3_vitb_lora` | DINOv3 ViT-B/16 | LoRA | 224 | 32 | 1 | 32 | 0.001 | 0.0002 | r=8, α=16, dropout=0.05, targets `q_proj`/`v_proj` |
| `dinov3_vitl_lora` | DINOv3 ViT-L/16 | LoRA | 224 | 16 | 2 | 32 | 0.001 | 0.0002 | r=8, α=16, dropout=0.05, targets `q_proj`/`v_proj` |
| `vjepa21_vitb_frozen` | V-JEPA 2.1 ViT-B/16 | Frozen linear probe | 384 | 32 | 1 | 32 | 0.001 | — | — |
| `vjepa21_vitl_frozen` | V-JEPA 2.1 ViT-L/16 | Frozen linear probe | 384 | 32 | 1 | 32 | 0.001 | — | — |
| `vjepa21_vitb_lora` | V-JEPA 2.1 ViT-B/16 | LoRA | 384 | 16 | 2 | 32 | 0.001 | 0.0002 | r=8, α=16, dropout=0.05, target `qkv` |
| `vjepa21_vitl_lora` | V-JEPA 2.1 ViT-L/16 | LoRA | 384 | 16 | 2 | 32 | 0.001 | 0.0002 | r=8, α=16, dropout=0.05, target `qkv` |
| `convnext_base_frozen` | ConvNeXt-Base | Frozen linear probe | 224 | 32 | 1 | 32 | 0.001 | — | — |
| `convnext_large_frozen` | ConvNeXt-Large | Frozen linear probe | 224 | 32 | 1 | 32 | 0.001 | — | — |
| `convnext_base_finetune` | ConvNeXt-Base | Full fine-tuning | 224 | 32 | 1 | 32 | 0.0004 | 0.00005 | — |
| `convnext_large_finetune` | ConvNeXt-Large | Full fine-tuning | 224 | 16 | 2 | 32 | 0.0004 | 0.00005 | — |
| `convnext_base_lora` | ConvNeXt-Base | LoRA | 224 | 32 | 1 | 32 | 0.001 | 0.0002 | r=8, α=16, dropout=0.05, targets `block.3`/`block.5` |
| `convnext_large_lora` | ConvNeXt-Large | LoRA | 224 | 16 | 2 | 32 | 0.001 | 0.0002 | r=8, α=16, dropout=0.05, targets `block.3`/`block.5` |
| `lingbot_vitb_frozen` | LingBot-Vision ViT-B/16 | Frozen linear probe | 224 | 32 | 1 | 32 | 0.001 | — | — |
| `lingbot_vitl_frozen` | LingBot-Vision ViT-L/16 | Frozen linear probe | 224 | 32 | 1 | 32 | 0.001 | — | — |
| `lingbot_vitb_lora` | LingBot-Vision ViT-B/16 | LoRA | 224 | 16 | 2 | 32 | 0.001 | 0.0002 | r=8, α=16, dropout=0.05, target `qkv` |
| `lingbot_vitl_lora` | LingBot-Vision ViT-L/16 | LoRA | 224 | 16 | 2 | 32 | 0.001 | 0.0002 | r=8, α=16, dropout=0.05, target `qkv` |

V-JEPA uses a two-frame pseudo-clip made by repeating the still image. The 2.1 variants use the strict `torch.hub` backend with no fallback to V-JEPA 2.0. LingBot uses mean patch-token pooling. Frozen variants optimise only their four linear heads; fine-tuned ConvNeXt variants optimise the whole backbone and heads; LoRA variants optimise adapters and heads.

### 2.3 Legacy classification aliases retained in configuration

These six entries are retained for historical checkpoint compatibility and are excluded from the current 18-variant size-ablation profile. They inherit the shared classification settings above.

| Legacy variant | Effective configuration | Relationship to current table |
|---|---|---|
| `dinov3` | DINOv3 ViT-B/16, frozen, 224 px, batch 32×1 | Equivalent architecture/adaptation to `dinov3_vitb_frozen` |
| `dinov3_lora` | DINOv3 ViT-B/16, LoRA r=8/α=16/dropout=0.05 on `q_proj`/`v_proj`, 224 px, batch 32×1 | Equivalent architecture/adaptation to `dinov3_vitb_lora` |
| `vjepa` | V-JEPA ViT-L/16, frozen, two frames, 384 px for `torch.hub` (legacy fallback could use 256 px Transformers backend), batch 32×1 | Predecessor of strict `vjepa21_vitl_frozen` |
| `vjepa_lora` | V-JEPA ViT-L/16, LoRA r=8/α=16/dropout=0.05 on `qkv`, two frames, 384/256 px backend-dependent, batch 32×1 | Predecessor of strict `vjepa21_vitl_lora` |
| `convnext_frozen` | ConvNeXt-Tiny, frozen, 224 px, batch 32×1 | Historical Tiny-scale frozen probe |
| `convnext` | ConvNeXt-Tiny, full fine-tuning, 224 px, batch 32×1; head LR 0.0004; backbone LR 0.00005 | Historical Tiny-scale fine-tuning run |

## 3. Source-of-truth files

- Detection defaults and offline augmentation: [`Scripts/MTSD-Scripts/MTSD-SupervisedDetection/config/default.yaml`](../Scripts/MTSD-Scripts/MTSD-SupervisedDetection/config/default.yaml)
- Effective YOLO argument construction and disabled online augmentation: [`Scripts/MTSD-Scripts/MTSD-SupervisedDetection/mtsd_detection/train_yolo.py`](../Scripts/MTSD-Scripts/MTSD-SupervisedDetection/mtsd_detection/train_yolo.py)
- Effective RF-DETR argument construction and native resolutions: [`Scripts/MTSD-Scripts/MTSD-SupervisedDetection/mtsd_detection/train_rfdetr.py`](../Scripts/MTSD-Scripts/MTSD-SupervisedDetection/mtsd_detection/train_rfdetr.py)
- Detection model registry: [`Scripts/MTSD-Scripts/MTSD-SupervisedDetection/mtsd_detection/model_registry.py`](../Scripts/MTSD-Scripts/MTSD-SupervisedDetection/mtsd_detection/model_registry.py)
- Completed-run inventory and saved effective values: [`Documents/Final-Reports/MTSD-SupervisedDetection/mtsd_complete_experiment_matrix.csv`](Final-Reports/MTSD-SupervisedDetection/mtsd_complete_experiment_matrix.csv) and individual `Results/MTSD-Runs/*-MTSD/*/run_record.json` / `args.yaml` files
- Attribute-classification defaults, model variants and LoRA overrides: [`Scripts/MTSD-Scripts/AttributeClassification/config/default.yaml`](../Scripts/MTSD-Scripts/AttributeClassification/config/default.yaml)
- Attribute full-run metadata: [`Scripts/MTSD-Scripts/AttributeClassification/outputs/experiment_log.jsonl`](../Scripts/MTSD-Scripts/AttributeClassification/outputs/experiment_log.jsonl)
