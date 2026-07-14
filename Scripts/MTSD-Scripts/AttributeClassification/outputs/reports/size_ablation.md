# Model-size ablation: consolidated report

Generated 2026-07-13T21:20:23+00:00. Primary metric: **macro-F1** on the test split. Every table is labelled with its adaptation mode; frozen probes, LoRA adaptation, and full fine-tuning are never mixed in one table.

## Full matrix (all completed size-ablation variants)

| Variant | Family | Architecture | Size | Adaptation | Res | Params | Trainable | Best epoch | Val mean macro-F1 | Test mean macro-F1 | view_angle F1 | mounting F1 | condition F1 | sign_shape F1 | view_angle acc | mounting acc | condition acc | sign_shape acc | Train time (s) | Test im/s | Backend |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov3_vitl_lora | DINOv3 | ViT-L/16 | large | lora | 224 | 303,929,357 | 799,757 | 7 | 0.8890 | 0.8939 | 0.9302 | 0.9367 | 0.7343 | 0.9744 | 0.9370 | 0.9689 | 0.8487 | 0.9782 | 1310.2 | 44.07 | transformers |
| lingbot_vitl_lora | LingBot-Vision | ViT-L/16 | large | lora | 224 | 303,953,933 | 799,757 | 22 | 0.8892 | 0.8917 | 0.9163 | 0.9425 | 0.7381 | 0.9698 | 0.9243 | 0.9729 | 0.8576 | 0.9729 | 2043.5 | 56.98 | lingbot_vision |
| vjepa21_vitl_lora | V-JEPA | ViT-L/16 | large | lora | 384 | 305,480,717 | 799,757 | 30 | 0.8841 | 0.8877 | 0.9178 | 0.9367 | 0.7343 | 0.9618 | 0.9252 | 0.9706 | 0.8571 | 0.9706 | 5698.8 | 41.96 | torch_hub (2.1) |
| dinov3_vitb_lora | DINOv3 | ViT-B/16 | base | lora | 224 | 85,965,325 | 304,909 | 22 | 0.8639 | 0.8844 | 0.9193 | 0.9213 | 0.7354 | 0.9618 | 0.9269 | 0.9639 | 0.8555 | 0.9723 | 1738.0 | 40.73 | transformers |
| lingbot_vitb_lora | LingBot-Vision | ViT-B/16 | base | lora | 224 | 85,974,541 | 304,909 | 28 | 0.8713 | 0.8748 | 0.9054 | 0.9209 | 0.7129 | 0.9600 | 0.9139 | 0.9625 | 0.8438 | 0.9660 | 2198.3 | 60.5 | lingbot_vision |
| convnext_large_finetune | ConvNeXt | ConvNeXt-Large | large | finetune | 224 | 196,247,245 | 196,247,245 | 19 | 0.8620 | 0.8717 | 0.9236 | 0.9276 | 0.6679 | 0.9677 | 0.9319 | 0.9674 | 0.8340 | 0.9701 | 1520.7 | 57.53 | torchvision |
| convnext_base_finetune | ConvNeXt | ConvNeXt-Base | base | finetune | 224 | 87,577,741 | 87,577,741 | 21 | 0.8704 | 0.8670 | 0.9264 | 0.9256 | 0.6617 | 0.9543 | 0.9333 | 0.9660 | 0.8326 | 0.9625 | 2144.9 | 55.19 | torchvision |
| vjepa21_vitb_lora | V-JEPA | ViT-B/16 | base | lora | 384 | 87,138,061 | 304,909 | 25 | 0.8627 | 0.8614 | 0.9112 | 0.9206 | 0.6571 | 0.9566 | 0.9210 | 0.9630 | 0.8134 | 0.9664 | 2913.8 | 39.81 | torch_hub (2.1) |
| dinov3_vitl_frozen | DINOv3 | ViT-L/16 | large | frozen | 224 | 303,142,925 | 13,325 | 12 | 0.8084 | 0.8120 | 0.8581 | 0.8967 | 0.6033 | 0.8898 | 0.8672 | 0.9471 | 0.7487 | 0.9008 | 1344.9 | 45.2 | transformers |
| lingbot_vitl_frozen | LingBot-Vision | ViT-L/16 | large | frozen | 224 | 303,167,501 | 13,325 | 32 | 0.7928 | 0.8056 | 0.8700 | 0.8800 | 0.5709 | 0.9015 | 0.8773 | 0.9378 | 0.7328 | 0.9353 | 2994.8 | 38.5 | lingbot_vision |
| dinov3_vitb_frozen | DINOv3 | ViT-B/16 | base | frozen | 224 | 85,670,413 | 9,997 | 21 | 0.8023 | 0.8052 | 0.8672 | 0.8706 | 0.5978 | 0.8853 | 0.8756 | 0.9345 | 0.7815 | 0.9042 | 2203.5 | 44.35 | transformers |
| vjepa21_vitl_frozen | V-JEPA | ViT-L/16 | large | frozen | 384 | 304,694,285 | 13,325 | 14 | 0.7930 | 0.7979 | 0.8809 | 0.8560 | 0.5718 | 0.8827 | 0.8882 | 0.9269 | 0.7521 | 0.9067 | 2158.0 | 39.12 | torch_hub (2.1) |
| lingbot_vitb_frozen | LingBot-Vision | ViT-B/16 | base | frozen | 224 | 85,679,629 | 9,997 | 20 | 0.7958 | 0.7949 | 0.8575 | 0.8732 | 0.5571 | 0.8919 | 0.8647 | 0.9387 | 0.7076 | 0.9218 | 2137.6 | 41.06 | lingbot_vision |
| vjepa21_vitb_frozen | V-JEPA | ViT-B/16 | base | frozen | 384 | 86,843,149 | 9,997 | 17 | 0.7768 | 0.7843 | 0.8595 | 0.8470 | 0.5591 | 0.8715 | 0.8655 | 0.9202 | 0.7370 | 0.9092 | 2317.4 | 40.56 | torch_hub (2.1) |
| convnext_base_frozen | ConvNeXt | ConvNeXt-Base | base | frozen | 224 | 87,577,741 | 13,325 | 22 | 0.7734 | 0.7782 | 0.8473 | 0.8607 | 0.5349 | 0.8699 | 0.8563 | 0.9311 | 0.6849 | 0.8992 | 1650.2 | 38.44 | torchvision |
| convnext_large_frozen | ConvNeXt | ConvNeXt-Large | large | frozen | 224 | 196,247,245 | 19,981 | 22 | 0.7746 | 0.7733 | 0.8356 | 0.8366 | 0.5226 | 0.8985 | 0.8445 | 0.9118 | 0.6916 | 0.9042 | 2296.6 | 37.11 | torchvision |

## 1. Frozen linear probes, by family (adaptation: frozen)

### DINOv3 (frozen)

| Variant | Family | Architecture | Size | Adaptation | Res | Params | Trainable | Best epoch | Val mean macro-F1 | Test mean macro-F1 | view_angle F1 | mounting F1 | condition F1 | sign_shape F1 | view_angle acc | mounting acc | condition acc | sign_shape acc | Train time (s) | Test im/s | Backend |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov3_vitl_frozen | DINOv3 | ViT-L/16 | large | frozen | 224 | 303,142,925 | 13,325 | 12 | 0.8084 | 0.8120 | 0.8581 | 0.8967 | 0.6033 | 0.8898 | 0.8672 | 0.9471 | 0.7487 | 0.9008 | 1344.9 | 45.2 | transformers |
| dinov3_vitb_frozen | DINOv3 | ViT-B/16 | base | frozen | 224 | 85,670,413 | 9,997 | 21 | 0.8023 | 0.8052 | 0.8672 | 0.8706 | 0.5978 | 0.8853 | 0.8756 | 0.9345 | 0.7815 | 0.9042 | 2203.5 | 44.35 | transformers |

### V-JEPA (frozen)

| Variant | Family | Architecture | Size | Adaptation | Res | Params | Trainable | Best epoch | Val mean macro-F1 | Test mean macro-F1 | view_angle F1 | mounting F1 | condition F1 | sign_shape F1 | view_angle acc | mounting acc | condition acc | sign_shape acc | Train time (s) | Test im/s | Backend |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vjepa21_vitl_frozen | V-JEPA | ViT-L/16 | large | frozen | 384 | 304,694,285 | 13,325 | 14 | 0.7930 | 0.7979 | 0.8809 | 0.8560 | 0.5718 | 0.8827 | 0.8882 | 0.9269 | 0.7521 | 0.9067 | 2158.0 | 39.12 | torch_hub (2.1) |
| vjepa21_vitb_frozen | V-JEPA | ViT-B/16 | base | frozen | 384 | 86,843,149 | 9,997 | 17 | 0.7768 | 0.7843 | 0.8595 | 0.8470 | 0.5591 | 0.8715 | 0.8655 | 0.9202 | 0.7370 | 0.9092 | 2317.4 | 40.56 | torch_hub (2.1) |

### ConvNeXt (frozen)

| Variant | Family | Architecture | Size | Adaptation | Res | Params | Trainable | Best epoch | Val mean macro-F1 | Test mean macro-F1 | view_angle F1 | mounting F1 | condition F1 | sign_shape F1 | view_angle acc | mounting acc | condition acc | sign_shape acc | Train time (s) | Test im/s | Backend |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| convnext_base_frozen | ConvNeXt | ConvNeXt-Base | base | frozen | 224 | 87,577,741 | 13,325 | 22 | 0.7734 | 0.7782 | 0.8473 | 0.8607 | 0.5349 | 0.8699 | 0.8563 | 0.9311 | 0.6849 | 0.8992 | 1650.2 | 38.44 | torchvision |
| convnext_large_frozen | ConvNeXt | ConvNeXt-Large | large | frozen | 224 | 196,247,245 | 19,981 | 22 | 0.7746 | 0.7733 | 0.8356 | 0.8366 | 0.5226 | 0.8985 | 0.8445 | 0.9118 | 0.6916 | 0.9042 | 2296.6 | 37.11 | torchvision |

### LingBot-Vision (frozen)

| Variant | Family | Architecture | Size | Adaptation | Res | Params | Trainable | Best epoch | Val mean macro-F1 | Test mean macro-F1 | view_angle F1 | mounting F1 | condition F1 | sign_shape F1 | view_angle acc | mounting acc | condition acc | sign_shape acc | Train time (s) | Test im/s | Backend |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lingbot_vitl_frozen | LingBot-Vision | ViT-L/16 | large | frozen | 224 | 303,167,501 | 13,325 | 32 | 0.7928 | 0.8056 | 0.8700 | 0.8800 | 0.5709 | 0.9015 | 0.8773 | 0.9378 | 0.7328 | 0.9353 | 2994.8 | 38.5 | lingbot_vision |
| lingbot_vitb_frozen | LingBot-Vision | ViT-B/16 | base | frozen | 224 | 85,679,629 | 9,997 | 20 | 0.7958 | 0.7949 | 0.8575 | 0.8732 | 0.5571 | 0.8919 | 0.8647 | 0.9387 | 0.7076 | 0.9218 | 2137.6 | 41.06 | lingbot_vision |

## 2. LoRA adaptation, by family (adaptation: lora)

### DINOv3 (LoRA)

| Variant | Family | Architecture | Size | Adaptation | Res | Params | Trainable | Best epoch | Val mean macro-F1 | Test mean macro-F1 | view_angle F1 | mounting F1 | condition F1 | sign_shape F1 | view_angle acc | mounting acc | condition acc | sign_shape acc | Train time (s) | Test im/s | Backend |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov3_vitl_lora | DINOv3 | ViT-L/16 | large | lora | 224 | 303,929,357 | 799,757 | 7 | 0.8890 | 0.8939 | 0.9302 | 0.9367 | 0.7343 | 0.9744 | 0.9370 | 0.9689 | 0.8487 | 0.9782 | 1310.2 | 44.07 | transformers |
| dinov3_vitb_lora | DINOv3 | ViT-B/16 | base | lora | 224 | 85,965,325 | 304,909 | 22 | 0.8639 | 0.8844 | 0.9193 | 0.9213 | 0.7354 | 0.9618 | 0.9269 | 0.9639 | 0.8555 | 0.9723 | 1738.0 | 40.73 | transformers |

### V-JEPA (LoRA)

| Variant | Family | Architecture | Size | Adaptation | Res | Params | Trainable | Best epoch | Val mean macro-F1 | Test mean macro-F1 | view_angle F1 | mounting F1 | condition F1 | sign_shape F1 | view_angle acc | mounting acc | condition acc | sign_shape acc | Train time (s) | Test im/s | Backend |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vjepa21_vitl_lora | V-JEPA | ViT-L/16 | large | lora | 384 | 305,480,717 | 799,757 | 30 | 0.8841 | 0.8877 | 0.9178 | 0.9367 | 0.7343 | 0.9618 | 0.9252 | 0.9706 | 0.8571 | 0.9706 | 5698.8 | 41.96 | torch_hub (2.1) |
| vjepa21_vitb_lora | V-JEPA | ViT-B/16 | base | lora | 384 | 87,138,061 | 304,909 | 25 | 0.8627 | 0.8614 | 0.9112 | 0.9206 | 0.6571 | 0.9566 | 0.9210 | 0.9630 | 0.8134 | 0.9664 | 2913.8 | 39.81 | torch_hub (2.1) |

### LingBot-Vision (LoRA)

| Variant | Family | Architecture | Size | Adaptation | Res | Params | Trainable | Best epoch | Val mean macro-F1 | Test mean macro-F1 | view_angle F1 | mounting F1 | condition F1 | sign_shape F1 | view_angle acc | mounting acc | condition acc | sign_shape acc | Train time (s) | Test im/s | Backend |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lingbot_vitl_lora | LingBot-Vision | ViT-L/16 | large | lora | 224 | 303,953,933 | 799,757 | 22 | 0.8892 | 0.8917 | 0.9163 | 0.9425 | 0.7381 | 0.9698 | 0.9243 | 0.9729 | 0.8576 | 0.9729 | 2043.5 | 56.98 | lingbot_vision |
| lingbot_vitb_lora | LingBot-Vision | ViT-B/16 | base | lora | 224 | 85,974,541 | 304,909 | 28 | 0.8713 | 0.8748 | 0.9054 | 0.9209 | 0.7129 | 0.9600 | 0.9139 | 0.9625 | 0.8438 | 0.9660 | 2198.3 | 60.5 | lingbot_vision |

## 3. ConvNeXt full fine-tuning (adaptation: finetune)

| Variant | Family | Architecture | Size | Adaptation | Res | Params | Trainable | Best epoch | Val mean macro-F1 | Test mean macro-F1 | view_angle F1 | mounting F1 | condition F1 | sign_shape F1 | view_angle acc | mounting acc | condition acc | sign_shape acc | Train time (s) | Test im/s | Backend |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| convnext_large_finetune | ConvNeXt | ConvNeXt-Large | large | finetune | 224 | 196,247,245 | 196,247,245 | 19 | 0.8620 | 0.8717 | 0.9236 | 0.9276 | 0.6679 | 0.9677 | 0.9319 | 0.9674 | 0.8340 | 0.9701 | 1520.7 | 57.53 | torchvision |
| convnext_base_finetune | ConvNeXt | ConvNeXt-Base | base | finetune | 224 | 87,577,741 | 87,577,741 | 21 | 0.8704 | 0.8670 | 0.9264 | 0.9256 | 0.6617 | 0.9543 | 0.9333 | 0.9660 | 0.8326 | 0.9625 | 2144.9 | 55.19 | torchvision |

## 4. All frozen representation models, cross-family ranking (adaptation: frozen)

| Variant | Family | Architecture | Size | Adaptation | Res | Params | Trainable | Best epoch | Val mean macro-F1 | Test mean macro-F1 | view_angle F1 | mounting F1 | condition F1 | sign_shape F1 | view_angle acc | mounting acc | condition acc | sign_shape acc | Train time (s) | Test im/s | Backend |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov3_vitl_frozen | DINOv3 | ViT-L/16 | large | frozen | 224 | 303,142,925 | 13,325 | 12 | 0.8084 | 0.8120 | 0.8581 | 0.8967 | 0.6033 | 0.8898 | 0.8672 | 0.9471 | 0.7487 | 0.9008 | 1344.9 | 45.2 | transformers |
| lingbot_vitl_frozen | LingBot-Vision | ViT-L/16 | large | frozen | 224 | 303,167,501 | 13,325 | 32 | 0.7928 | 0.8056 | 0.8700 | 0.8800 | 0.5709 | 0.9015 | 0.8773 | 0.9378 | 0.7328 | 0.9353 | 2994.8 | 38.5 | lingbot_vision |
| dinov3_vitb_frozen | DINOv3 | ViT-B/16 | base | frozen | 224 | 85,670,413 | 9,997 | 21 | 0.8023 | 0.8052 | 0.8672 | 0.8706 | 0.5978 | 0.8853 | 0.8756 | 0.9345 | 0.7815 | 0.9042 | 2203.5 | 44.35 | transformers |
| vjepa21_vitl_frozen | V-JEPA | ViT-L/16 | large | frozen | 384 | 304,694,285 | 13,325 | 14 | 0.7930 | 0.7979 | 0.8809 | 0.8560 | 0.5718 | 0.8827 | 0.8882 | 0.9269 | 0.7521 | 0.9067 | 2158.0 | 39.12 | torch_hub (2.1) |
| lingbot_vitb_frozen | LingBot-Vision | ViT-B/16 | base | frozen | 224 | 85,679,629 | 9,997 | 20 | 0.7958 | 0.7949 | 0.8575 | 0.8732 | 0.5571 | 0.8919 | 0.8647 | 0.9387 | 0.7076 | 0.9218 | 2137.6 | 41.06 | lingbot_vision |
| vjepa21_vitb_frozen | V-JEPA | ViT-B/16 | base | frozen | 384 | 86,843,149 | 9,997 | 17 | 0.7768 | 0.7843 | 0.8595 | 0.8470 | 0.5591 | 0.8715 | 0.8655 | 0.9202 | 0.7370 | 0.9092 | 2317.4 | 40.56 | torch_hub (2.1) |
| convnext_base_frozen | ConvNeXt | ConvNeXt-Base | base | frozen | 224 | 87,577,741 | 13,325 | 22 | 0.7734 | 0.7782 | 0.8473 | 0.8607 | 0.5349 | 0.8699 | 0.8563 | 0.9311 | 0.6849 | 0.8992 | 1650.2 | 38.44 | torchvision |
| convnext_large_frozen | ConvNeXt | ConvNeXt-Large | large | frozen | 224 | 196,247,245 | 19,981 | 22 | 0.7746 | 0.7733 | 0.8356 | 0.8366 | 0.5226 | 0.8985 | 0.8445 | 0.9118 | 0.6916 | 0.9042 | 2296.6 | 37.11 | torchvision |
