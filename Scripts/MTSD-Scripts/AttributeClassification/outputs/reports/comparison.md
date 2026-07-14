# Attribute classification: consolidated comparison

Generated 2026-07-13T21:20:23+00:00. Primary metric: **macro-F1** on the test split (robust to class imbalance; accuracy shown for reference).

## Macro-F1 by attribute

| Variant | view_angle | mounting | condition | sign_shape | Mean |
|---|---|---|---|---|---|
| dinov3 | 0.8649 | 0.8458 | 0.5534 | 0.8663 | **0.7826** |
| dinov3_lora | 0.9129 | 0.8959 | 0.7346 | 0.9503 | **0.8734** |
| vjepa | 0.8823 | 0.8672 | 0.5110 | 0.8189 | **0.7699** |
| vjepa_lora | 0.9089 | 0.9054 | 0.5948 | 0.9507 | **0.8400** |
| convnext_frozen | 0.8426 | 0.8026 | 0.5673 | 0.8389 | **0.7629** |
| convnext | 0.9054 | 0.8734 | 0.7113 | 0.9663 | **0.8641** |
| dinov3_vitb_frozen | 0.8672 | 0.8706 | 0.5978 | 0.8853 | **0.8052** |
| dinov3_vitl_frozen | 0.8581 | 0.8967 | 0.6033 | 0.8898 | **0.8120** |
| dinov3_vitb_lora | 0.9193 | 0.9213 | 0.7354 | 0.9618 | **0.8844** |
| dinov3_vitl_lora | 0.9302 | 0.9367 | 0.7343 | 0.9744 | **0.8939** |
| vjepa21_vitb_frozen | 0.8595 | 0.8470 | 0.5591 | 0.8715 | **0.7843** |
| vjepa21_vitl_frozen | 0.8809 | 0.8560 | 0.5718 | 0.8827 | **0.7979** |
| vjepa21_vitb_lora | 0.9112 | 0.9206 | 0.6571 | 0.9566 | **0.8614** |
| vjepa21_vitl_lora | 0.9178 | 0.9367 | 0.7343 | 0.9618 | **0.8877** |
| convnext_base_frozen | 0.8473 | 0.8607 | 0.5349 | 0.8699 | **0.7782** |
| convnext_large_frozen | 0.8356 | 0.8366 | 0.5226 | 0.8985 | **0.7733** |
| convnext_base_finetune | 0.9264 | 0.9256 | 0.6617 | 0.9543 | **0.8670** |
| convnext_large_finetune | 0.9236 | 0.9276 | 0.6679 | 0.9677 | **0.8717** |
| lingbot_vitb_frozen | 0.8575 | 0.8732 | 0.5571 | 0.8919 | **0.7949** |
| lingbot_vitl_frozen | 0.8700 | 0.8800 | 0.5709 | 0.9015 | **0.8056** |
| lingbot_vitb_lora | 0.9054 | 0.9209 | 0.7129 | 0.9600 | **0.8748** |
| lingbot_vitl_lora | 0.9163 | 0.9425 | 0.7381 | 0.9698 | **0.8917** |

## Accuracy by attribute

| Variant | view_angle | mounting | condition | sign_shape |
|---|---|---|---|---|
| dinov3 | 0.8748 | 0.9324 | 0.7092 | 0.9026 |
| dinov3_lora | 0.9205 | 0.9563 | 0.8267 | 0.9602 |
| vjepa | 0.8907 | 0.9364 | 0.7371 | 0.8628 |
| vjepa_lora | 0.9165 | 0.9583 | 0.8028 | 0.9662 |
| convnext_frozen | 0.8549 | 0.9026 | 0.7291 | 0.8887 |
| convnext | 0.9145 | 0.9503 | 0.8566 | 0.9702 |
| dinov3_vitb_frozen | 0.8756 | 0.9345 | 0.7815 | 0.9042 |
| dinov3_vitl_frozen | 0.8672 | 0.9471 | 0.7487 | 0.9008 |
| dinov3_vitb_lora | 0.9269 | 0.9639 | 0.8555 | 0.9723 |
| dinov3_vitl_lora | 0.9370 | 0.9689 | 0.8487 | 0.9782 |
| vjepa21_vitb_frozen | 0.8655 | 0.9202 | 0.7370 | 0.9092 |
| vjepa21_vitl_frozen | 0.8882 | 0.9269 | 0.7521 | 0.9067 |
| vjepa21_vitb_lora | 0.9210 | 0.9630 | 0.8134 | 0.9664 |
| vjepa21_vitl_lora | 0.9252 | 0.9706 | 0.8571 | 0.9706 |
| convnext_base_frozen | 0.8563 | 0.9311 | 0.6849 | 0.8992 |
| convnext_large_frozen | 0.8445 | 0.9118 | 0.6916 | 0.9042 |
| convnext_base_finetune | 0.9333 | 0.9660 | 0.8326 | 0.9625 |
| convnext_large_finetune | 0.9319 | 0.9674 | 0.8340 | 0.9701 |
| lingbot_vitb_frozen | 0.8647 | 0.9387 | 0.7076 | 0.9218 |
| lingbot_vitl_frozen | 0.8773 | 0.9378 | 0.7328 | 0.9353 |
| lingbot_vitb_lora | 0.9139 | 0.9625 | 0.8438 | 0.9660 |
| lingbot_vitl_lora | 0.9243 | 0.9729 | 0.8576 | 0.9729 |

## Run details

| Variant | Family | Architecture | Size | Adaptation | Res | Best epoch | Stopped at | Stop reason | Val mean macro-F1 | Total params | Trainable | Trainable % | Train time (s) | Backend |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov3 | DINOv3 | ViT-B/16 | base | frozen | 224 | 17 | 22 | early_stopping | 0.7697 | 85,670,413 | 9,997 | 0.0117 | n/a | n/a |
| dinov3_lora | DINOv3 | ViT-B/16 | base | lora | 224 | 16 | 22 | early_stopping | 0.8327 | 85,965,325 | 304,909 | 0.3547 | n/a | n/a |
| vjepa | V-JEPA | ViT-L/16 | large | frozen | 384 | 19 | 29 | early_stopping | 0.7453 | 304,694,285 | 13,325 | 0.0044 | n/a | n/a |
| vjepa_lora | V-JEPA | ViT-L/16 | large | lora | 384 | 8 | 18 | early_stopping | 0.8380 | 305,480,717 | 799,757 | 0.2618 | n/a | n/a |
| convnext_frozen | ConvNeXt | ConvNeXt-Tiny | tiny | frozen | 224 | 37 | 47 | early_stopping | 0.7442 | 27,828,589 | 9,997 | 0.0359 | n/a | n/a |
| convnext | ConvNeXt | ConvNeXt-Tiny | tiny | finetune | 224 | 21 | 25 | early_stopping | 0.8212 | 27,828,589 | 27,828,589 | 100.0 | n/a | n/a |
| dinov3_vitb_frozen | DINOv3 | ViT-B/16 | base | frozen | 224 | 21 | 31 | early_stopping | 0.8023 | 85,670,413 | 9,997 | 0.0117 | 2203.5 | transformers |
| dinov3_vitl_frozen | DINOv3 | ViT-L/16 | large | frozen | 224 | 12 | 19 | early_stopping | 0.8084 | 303,142,925 | 13,325 | 0.0044 | 1344.9 | transformers |
| dinov3_vitb_lora | DINOv3 | ViT-B/16 | base | lora | 224 | 22 | 24 | early_stopping | 0.8639 | 85,965,325 | 304,909 | 0.3547 | 1738.0 | transformers |
| dinov3_vitl_lora | DINOv3 | ViT-L/16 | large | lora | 224 | 7 | 17 | early_stopping | 0.8890 | 303,929,357 | 799,757 | 0.2631 | 1310.2 | transformers |
| vjepa21_vitb_frozen | V-JEPA | ViT-B/16 | base | frozen | 384 | 17 | 27 | early_stopping | 0.7768 | 86,843,149 | 9,997 | 0.0115 | 2317.4 | torch_hub (2.1) |
| vjepa21_vitl_frozen | V-JEPA | ViT-L/16 | large | frozen | 384 | 14 | 24 | early_stopping | 0.7930 | 304,694,285 | 13,325 | 0.0044 | 2158.0 | torch_hub (2.1) |
| vjepa21_vitb_lora | V-JEPA | ViT-B/16 | base | lora | 384 | 25 | 32 | early_stopping | 0.8627 | 87,138,061 | 304,909 | 0.3499 | 2913.8 | torch_hub (2.1) |
| vjepa21_vitl_lora | V-JEPA | ViT-L/16 | large | lora | 384 | 30 | 40 | early_stopping | 0.8841 | 305,480,717 | 799,757 | 0.2618 | 5698.8 | torch_hub (2.1) |
| convnext_base_frozen | ConvNeXt | ConvNeXt-Base | base | frozen | 224 | 22 | 23 | early_stopping | 0.7734 | 87,577,741 | 13,325 | 0.0152 | 1650.2 | torchvision |
| convnext_large_frozen | ConvNeXt | ConvNeXt-Large | large | frozen | 224 | 22 | 32 | early_stopping | 0.7746 | 196,247,245 | 19,981 | 0.0102 | 2296.6 | torchvision |
| convnext_base_finetune | ConvNeXt | ConvNeXt-Base | base | finetune | 224 | 21 | 31 | early_stopping | 0.8704 | 87,577,741 | 87,577,741 | 100.0 | 2144.9 | torchvision |
| convnext_large_finetune | ConvNeXt | ConvNeXt-Large | large | finetune | 224 | 19 | 21 | early_stopping | 0.8620 | 196,247,245 | 196,247,245 | 100.0 | 1520.7 | torchvision |
| lingbot_vitb_frozen | LingBot-Vision | ViT-B/16 | base | frozen | 224 | 20 | 30 | early_stopping | 0.7958 | 85,679,629 | 9,997 | 0.0117 | 2137.6 | lingbot_vision |
| lingbot_vitl_frozen | LingBot-Vision | ViT-L/16 | large | frozen | 224 | 32 | 42 | early_stopping | 0.7928 | 303,167,501 | 13,325 | 0.0044 | 2994.8 | lingbot_vision |
| lingbot_vitb_lora | LingBot-Vision | ViT-B/16 | base | lora | 224 | 28 | 32 | early_stopping | 0.8713 | 85,974,541 | 304,909 | 0.3547 | 2198.3 | lingbot_vision |
| lingbot_vitl_lora | LingBot-Vision | ViT-L/16 | large | lora | 224 | 22 | 29 | early_stopping | 0.8892 | 303,953,933 | 799,757 | 0.2631 | 2043.5 | lingbot_vision |

## Per-class F1: view_angle

| Class | Support | dinov3 | dinov3_lora | vjepa | vjepa_lora | convnext_frozen | convnext | dinov3_vitb_frozen | dinov3_vitl_frozen | dinov3_vitb_lora | dinov3_vitl_lora | vjepa21_vitb_frozen | vjepa21_vitl_frozen | vjepa21_vitb_lora | vjepa21_vitl_lora | convnext_base_frozen | convnext_large_frozen | convnext_base_finetune | convnext_large_finetune | lingbot_vitb_frozen | lingbot_vitl_frozen | lingbot_vitb_lora | lingbot_vitl_lora |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Front | 185 | 0.8717 | 0.9173 | 0.8976 | 0.9022 | 0.8556 | 0.9202 | 0.8764 | 0.8682 | 0.9219 | 0.9333 | 0.8856 | 0.9001 | 0.9174 | 0.9178 | 0.8597 | 0.8401 | 0.9307 | 0.9345 | 0.8798 | 0.8777 | 0.9141 | 0.9216 |
| Back | 200 | 0.9196 | 0.9570 | 0.9173 | 0.9561 | 0.9118 | 0.9474 | 0.9130 | 0.9081 | 0.9665 | 0.9711 | 0.8891 | 0.9199 | 0.9634 | 0.9686 | 0.9009 | 0.8999 | 0.9664 | 0.9650 | 0.9010 | 0.9168 | 0.9563 | 0.9646 |
| Side | 118 | 0.8034 | 0.8644 | 0.8319 | 0.8684 | 0.7603 | 0.8485 | 0.8122 | 0.7979 | 0.8694 | 0.8861 | 0.8039 | 0.8227 | 0.8528 | 0.8670 | 0.7814 | 0.7667 | 0.8821 | 0.8712 | 0.7916 | 0.8156 | 0.8459 | 0.8627 |

## Per-class F1: mounting

| Class | Support | dinov3 | dinov3_lora | vjepa | vjepa_lora | convnext_frozen | convnext | dinov3_vitb_frozen | dinov3_vitl_frozen | dinov3_vitb_lora | dinov3_vitl_lora | vjepa21_vitb_frozen | vjepa21_vitl_frozen | vjepa21_vitb_lora | vjepa21_vitl_lora | convnext_base_frozen | convnext_large_frozen | convnext_base_finetune | convnext_large_finetune | lingbot_vitb_frozen | lingbot_vitl_frozen | lingbot_vitb_lora | lingbot_vitl_lora |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pole-Mounted | 444 | 0.9614 | 0.9752 | 0.9630 | 0.9761 | 0.9431 | 0.9721 | 0.9615 | 0.9688 | 0.9792 | 0.9819 | 0.9528 | 0.9570 | 0.9786 | 0.9830 | 0.9597 | 0.9474 | 0.9804 | 0.9813 | 0.9643 | 0.9633 | 0.9783 | 0.9843 |
| Wall-Mounted | 59 | 0.7302 | 0.8167 | 0.7714 | 0.8346 | 0.6621 | 0.7748 | 0.7797 | 0.8245 | 0.8635 | 0.8915 | 0.7411 | 0.7549 | 0.8625 | 0.8903 | 0.7616 | 0.7258 | 0.8707 | 0.8740 | 0.7821 | 0.7967 | 0.8636 | 0.9008 |

## Per-class F1: condition

| Class | Support | dinov3 | dinov3_lora | vjepa | vjepa_lora | convnext_frozen | convnext | dinov3_vitb_frozen | dinov3_vitl_frozen | dinov3_vitb_lora | dinov3_vitl_lora | vjepa21_vitb_frozen | vjepa21_vitl_frozen | vjepa21_vitb_lora | vjepa21_vitl_lora | convnext_base_frozen | convnext_large_frozen | convnext_base_finetune | convnext_large_finetune | lingbot_vitb_frozen | lingbot_vitl_frozen | lingbot_vitb_lora | lingbot_vitl_lora |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Good | 387 | 0.8216 | 0.8980 | 0.8763 | 0.9027 | 0.8487 | 0.9258 | 0.8806 | 0.8473 | 0.9198 | 0.9164 | 0.8403 | 0.8558 | 0.8990 | 0.9226 | 0.7963 | 0.8153 | 0.9090 | 0.9098 | 0.8244 | 0.8432 | 0.9125 | 0.9241 |
| Weathered | 85 | 0.4925 | 0.6161 | 0.3681 | 0.5636 | 0.4490 | 0.6199 | 0.5159 | 0.5389 | 0.6593 | 0.6597 | 0.4853 | 0.4856 | 0.5808 | 0.6591 | 0.4656 | 0.4229 | 0.6230 | 0.6299 | 0.4381 | 0.4870 | 0.6528 | 0.6667 |
| Heavily Damaged | 30 | 0.3462 | 0.6897 | 0.2887 | 0.3182 | 0.4043 | 0.5882 | 0.3969 | 0.4238 | 0.6271 | 0.6269 | 0.3519 | 0.3741 | 0.4915 | 0.6212 | 0.3429 | 0.3297 | 0.4531 | 0.4640 | 0.4088 | 0.3825 | 0.5734 | 0.6234 |

## Per-class F1: sign_shape

| Class | Support | dinov3 | dinov3_lora | vjepa | vjepa_lora | convnext_frozen | convnext | dinov3_vitb_frozen | dinov3_vitl_frozen | dinov3_vitb_lora | dinov3_vitl_lora | vjepa21_vitb_frozen | vjepa21_vitl_frozen | vjepa21_vitb_lora | vjepa21_vitl_lora | convnext_base_frozen | convnext_large_frozen | convnext_base_finetune | convnext_large_finetune | lingbot_vitb_frozen | lingbot_vitl_frozen | lingbot_vitb_lora | lingbot_vitl_lora |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Circular | 226 | 0.9273 | 0.9731 | 0.9065 | 0.9800 | 0.9099 | 0.9823 | 0.9229 | 0.9097 | 0.9809 | 0.9840 | 0.9268 | 0.9423 | 0.9799 | 0.9809 | 0.9176 | 0.9180 | 0.9740 | 0.9750 | 0.9345 | 0.9460 | 0.9729 | 0.9776 |
| Quadrangle | 148 | 0.9220 | 0.9695 | 0.8961 | 0.9664 | 0.9225 | 0.9764 | 0.9209 | 0.9184 | 0.9737 | 0.9816 | 0.9323 | 0.9245 | 0.9715 | 0.9764 | 0.9176 | 0.9217 | 0.9638 | 0.9762 | 0.9438 | 0.9530 | 0.9742 | 0.9805 |
| Triangular | 57 | 0.8621 | 0.9402 | 0.8772 | 0.9550 | 0.8780 | 0.9474 | 0.9075 | 0.9054 | 0.9736 | 0.9825 | 0.9096 | 0.8377 | 0.9647 | 0.9558 | 0.8946 | 0.9235 | 0.9545 | 0.9722 | 0.9426 | 0.9464 | 0.9602 | 0.9657 |
| Octagonal | 61 | 0.8346 | 0.9120 | 0.7006 | 0.9355 | 0.7967 | 0.9256 | 0.8127 | 0.8219 | 0.9444 | 0.9459 | 0.8228 | 0.8394 | 0.9104 | 0.9416 | 0.8200 | 0.7962 | 0.9301 | 0.9326 | 0.8153 | 0.8696 | 0.9284 | 0.9433 |
| Pentagon | 11 | 0.7857 | 0.9565 | 0.7143 | 0.9167 | 0.6875 | 1.0000 | 0.8627 | 0.8936 | 0.9362 | 0.9778 | 0.7660 | 0.8696 | 0.9565 | 0.9545 | 0.8000 | 0.9333 | 0.9492 | 0.9825 | 0.8235 | 0.7925 | 0.9643 | 0.9818 |

## Confusion matrices

- dinov3: [view_angle](../metrics/dinov3/confusion_test_view_angle.png), [mounting](../metrics/dinov3/confusion_test_mounting.png), [condition](../metrics/dinov3/confusion_test_condition.png), [sign_shape](../metrics/dinov3/confusion_test_sign_shape.png)
- dinov3_lora: [view_angle](../metrics/dinov3_lora/confusion_test_view_angle.png), [mounting](../metrics/dinov3_lora/confusion_test_mounting.png), [condition](../metrics/dinov3_lora/confusion_test_condition.png), [sign_shape](../metrics/dinov3_lora/confusion_test_sign_shape.png)
- vjepa: [view_angle](../metrics/vjepa/confusion_test_view_angle.png), [mounting](../metrics/vjepa/confusion_test_mounting.png), [condition](../metrics/vjepa/confusion_test_condition.png), [sign_shape](../metrics/vjepa/confusion_test_sign_shape.png)
- vjepa_lora: [view_angle](../metrics/vjepa_lora/confusion_test_view_angle.png), [mounting](../metrics/vjepa_lora/confusion_test_mounting.png), [condition](../metrics/vjepa_lora/confusion_test_condition.png), [sign_shape](../metrics/vjepa_lora/confusion_test_sign_shape.png)
- convnext_frozen: [view_angle](../metrics/convnext_frozen/confusion_test_view_angle.png), [mounting](../metrics/convnext_frozen/confusion_test_mounting.png), [condition](../metrics/convnext_frozen/confusion_test_condition.png), [sign_shape](../metrics/convnext_frozen/confusion_test_sign_shape.png)
- convnext: [view_angle](../metrics/convnext/confusion_test_view_angle.png), [mounting](../metrics/convnext/confusion_test_mounting.png), [condition](../metrics/convnext/confusion_test_condition.png), [sign_shape](../metrics/convnext/confusion_test_sign_shape.png)
- dinov3_vitb_frozen: [view_angle](../metrics/dinov3_vitb_frozen/confusion_test_view_angle.png), [mounting](../metrics/dinov3_vitb_frozen/confusion_test_mounting.png), [condition](../metrics/dinov3_vitb_frozen/confusion_test_condition.png), [sign_shape](../metrics/dinov3_vitb_frozen/confusion_test_sign_shape.png)
- dinov3_vitl_frozen: [view_angle](../metrics/dinov3_vitl_frozen/confusion_test_view_angle.png), [mounting](../metrics/dinov3_vitl_frozen/confusion_test_mounting.png), [condition](../metrics/dinov3_vitl_frozen/confusion_test_condition.png), [sign_shape](../metrics/dinov3_vitl_frozen/confusion_test_sign_shape.png)
- dinov3_vitb_lora: [view_angle](../metrics/dinov3_vitb_lora/confusion_test_view_angle.png), [mounting](../metrics/dinov3_vitb_lora/confusion_test_mounting.png), [condition](../metrics/dinov3_vitb_lora/confusion_test_condition.png), [sign_shape](../metrics/dinov3_vitb_lora/confusion_test_sign_shape.png)
- dinov3_vitl_lora: [view_angle](../metrics/dinov3_vitl_lora/confusion_test_view_angle.png), [mounting](../metrics/dinov3_vitl_lora/confusion_test_mounting.png), [condition](../metrics/dinov3_vitl_lora/confusion_test_condition.png), [sign_shape](../metrics/dinov3_vitl_lora/confusion_test_sign_shape.png)
- vjepa21_vitb_frozen: [view_angle](../metrics/vjepa21_vitb_frozen/confusion_test_view_angle.png), [mounting](../metrics/vjepa21_vitb_frozen/confusion_test_mounting.png), [condition](../metrics/vjepa21_vitb_frozen/confusion_test_condition.png), [sign_shape](../metrics/vjepa21_vitb_frozen/confusion_test_sign_shape.png)
- vjepa21_vitl_frozen: [view_angle](../metrics/vjepa21_vitl_frozen/confusion_test_view_angle.png), [mounting](../metrics/vjepa21_vitl_frozen/confusion_test_mounting.png), [condition](../metrics/vjepa21_vitl_frozen/confusion_test_condition.png), [sign_shape](../metrics/vjepa21_vitl_frozen/confusion_test_sign_shape.png)
- vjepa21_vitb_lora: [view_angle](../metrics/vjepa21_vitb_lora/confusion_test_view_angle.png), [mounting](../metrics/vjepa21_vitb_lora/confusion_test_mounting.png), [condition](../metrics/vjepa21_vitb_lora/confusion_test_condition.png), [sign_shape](../metrics/vjepa21_vitb_lora/confusion_test_sign_shape.png)
- vjepa21_vitl_lora: [view_angle](../metrics/vjepa21_vitl_lora/confusion_test_view_angle.png), [mounting](../metrics/vjepa21_vitl_lora/confusion_test_mounting.png), [condition](../metrics/vjepa21_vitl_lora/confusion_test_condition.png), [sign_shape](../metrics/vjepa21_vitl_lora/confusion_test_sign_shape.png)
- convnext_base_frozen: [view_angle](../metrics/convnext_base_frozen/confusion_test_view_angle.png), [mounting](../metrics/convnext_base_frozen/confusion_test_mounting.png), [condition](../metrics/convnext_base_frozen/confusion_test_condition.png), [sign_shape](../metrics/convnext_base_frozen/confusion_test_sign_shape.png)
- convnext_large_frozen: [view_angle](../metrics/convnext_large_frozen/confusion_test_view_angle.png), [mounting](../metrics/convnext_large_frozen/confusion_test_mounting.png), [condition](../metrics/convnext_large_frozen/confusion_test_condition.png), [sign_shape](../metrics/convnext_large_frozen/confusion_test_sign_shape.png)
- convnext_base_finetune: [view_angle](../metrics/convnext_base_finetune/confusion_test_view_angle.png), [mounting](../metrics/convnext_base_finetune/confusion_test_mounting.png), [condition](../metrics/convnext_base_finetune/confusion_test_condition.png), [sign_shape](../metrics/convnext_base_finetune/confusion_test_sign_shape.png)
- convnext_large_finetune: [view_angle](../metrics/convnext_large_finetune/confusion_test_view_angle.png), [mounting](../metrics/convnext_large_finetune/confusion_test_mounting.png), [condition](../metrics/convnext_large_finetune/confusion_test_condition.png), [sign_shape](../metrics/convnext_large_finetune/confusion_test_sign_shape.png)
- lingbot_vitb_frozen: [view_angle](../metrics/lingbot_vitb_frozen/confusion_test_view_angle.png), [mounting](../metrics/lingbot_vitb_frozen/confusion_test_mounting.png), [condition](../metrics/lingbot_vitb_frozen/confusion_test_condition.png), [sign_shape](../metrics/lingbot_vitb_frozen/confusion_test_sign_shape.png)
- lingbot_vitl_frozen: [view_angle](../metrics/lingbot_vitl_frozen/confusion_test_view_angle.png), [mounting](../metrics/lingbot_vitl_frozen/confusion_test_mounting.png), [condition](../metrics/lingbot_vitl_frozen/confusion_test_condition.png), [sign_shape](../metrics/lingbot_vitl_frozen/confusion_test_sign_shape.png)
- lingbot_vitb_lora: [view_angle](../metrics/lingbot_vitb_lora/confusion_test_view_angle.png), [mounting](../metrics/lingbot_vitb_lora/confusion_test_mounting.png), [condition](../metrics/lingbot_vitb_lora/confusion_test_condition.png), [sign_shape](../metrics/lingbot_vitb_lora/confusion_test_sign_shape.png)
- lingbot_vitl_lora: [view_angle](../metrics/lingbot_vitl_lora/confusion_test_view_angle.png), [mounting](../metrics/lingbot_vitl_lora/confusion_test_mounting.png), [condition](../metrics/lingbot_vitl_lora/confusion_test_condition.png), [sign_shape](../metrics/lingbot_vitl_lora/confusion_test_sign_shape.png)
