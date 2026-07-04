# Attribute classification: consolidated comparison

Generated 2026-07-04T00:41:46+00:00. Primary metric: **macro-F1** on the test split (robust to class imbalance; accuracy shown for reference).

## Macro-F1 by attribute

| Variant | view_angle | mounting | condition | sign_shape | Mean |
|---|---|---|---|---|---|
| dinov3 | 0.8649 | 0.8458 | 0.5534 | 0.8663 | **0.7826** |
| dinov3_lora | 0.9129 | 0.8959 | 0.7346 | 0.9503 | **0.8734** |
| vjepa | 0.8823 | 0.8672 | 0.5110 | 0.8189 | **0.7699** |
| vjepa_lora | 0.9089 | 0.9054 | 0.5948 | 0.9507 | **0.8400** |
| convnext_frozen | 0.8426 | 0.8026 | 0.5673 | 0.8389 | **0.7629** |
| convnext | 0.9054 | 0.8734 | 0.7113 | 0.9663 | **0.8641** |

## Accuracy by attribute

| Variant | view_angle | mounting | condition | sign_shape |
|---|---|---|---|---|
| dinov3 | 0.8748 | 0.9324 | 0.7092 | 0.9026 |
| dinov3_lora | 0.9205 | 0.9563 | 0.8267 | 0.9602 |
| vjepa | 0.8907 | 0.9364 | 0.7371 | 0.8628 |
| vjepa_lora | 0.9165 | 0.9583 | 0.8028 | 0.9662 |
| convnext_frozen | 0.8549 | 0.9026 | 0.7291 | 0.8887 |
| convnext | 0.9145 | 0.9503 | 0.8566 | 0.9702 |

## Run details

| Variant | Adaptation | Best epoch | Stopped at | Stop reason | Val mean macro-F1 | Total params | Trainable | Trainable % |
|---|---|---|---|---|---|---|---|---|
| dinov3 | frozen | 17 | 22 | early_stopping | 0.7697 | 85,670,413 | 9,997 | 0.0117 |
| dinov3_lora | lora | 16 | 22 | early_stopping | 0.8327 | 85,965,325 | 304,909 | 0.3547 |
| vjepa | frozen | 19 | 29 | early_stopping | 0.7453 | 304,694,285 | 13,325 | 0.0044 |
| vjepa_lora | lora | 8 | 18 | early_stopping | 0.8380 | 305,480,717 | 799,757 | 0.2618 |
| convnext_frozen | frozen | 37 | 47 | early_stopping | 0.7442 | 27,828,589 | 9,997 | 0.0359 |
| convnext | finetune | 21 | 25 | early_stopping | 0.8212 | 27,828,589 | 27,828,589 | 100.0 |

## Per-class F1: view_angle

| Class | Support | dinov3 | dinov3_lora | vjepa | vjepa_lora | convnext_frozen | convnext |
|---|---|---|---|---|---|---|---|
| Front | 185 | 0.8717 | 0.9173 | 0.8976 | 0.9022 | 0.8556 | 0.9202 |
| Back | 200 | 0.9196 | 0.9570 | 0.9173 | 0.9561 | 0.9118 | 0.9474 |
| Side | 118 | 0.8034 | 0.8644 | 0.8319 | 0.8684 | 0.7603 | 0.8485 |

## Per-class F1: mounting

| Class | Support | dinov3 | dinov3_lora | vjepa | vjepa_lora | convnext_frozen | convnext |
|---|---|---|---|---|---|---|---|
| Pole-Mounted | 444 | 0.9614 | 0.9752 | 0.9630 | 0.9761 | 0.9431 | 0.9721 |
| Wall-Mounted | 59 | 0.7302 | 0.8167 | 0.7714 | 0.8346 | 0.6621 | 0.7748 |

## Per-class F1: condition

| Class | Support | dinov3 | dinov3_lora | vjepa | vjepa_lora | convnext_frozen | convnext |
|---|---|---|---|---|---|---|---|
| Good | 387 | 0.8216 | 0.8980 | 0.8763 | 0.9027 | 0.8487 | 0.9258 |
| Weathered | 85 | 0.4925 | 0.6161 | 0.3681 | 0.5636 | 0.4490 | 0.6199 |
| Heavily Damaged | 30 | 0.3462 | 0.6897 | 0.2887 | 0.3182 | 0.4043 | 0.5882 |

## Per-class F1: sign_shape

| Class | Support | dinov3 | dinov3_lora | vjepa | vjepa_lora | convnext_frozen | convnext |
|---|---|---|---|---|---|---|---|
| Circular | 226 | 0.9273 | 0.9731 | 0.9065 | 0.9800 | 0.9099 | 0.9823 |
| Quadrangle | 148 | 0.9220 | 0.9695 | 0.8961 | 0.9664 | 0.9225 | 0.9764 |
| Triangular | 57 | 0.8621 | 0.9402 | 0.8772 | 0.9550 | 0.8780 | 0.9474 |
| Octagonal | 61 | 0.8346 | 0.9120 | 0.7006 | 0.9355 | 0.7967 | 0.9256 |
| Pentagon | 11 | 0.7857 | 0.9565 | 0.7143 | 0.9167 | 0.6875 | 1.0000 |

## Confusion matrices

- dinov3: [view_angle](../metrics/dinov3/confusion_test_view_angle.png), [mounting](../metrics/dinov3/confusion_test_mounting.png), [condition](../metrics/dinov3/confusion_test_condition.png), [sign_shape](../metrics/dinov3/confusion_test_sign_shape.png)
- dinov3_lora: [view_angle](../metrics/dinov3_lora/confusion_test_view_angle.png), [mounting](../metrics/dinov3_lora/confusion_test_mounting.png), [condition](../metrics/dinov3_lora/confusion_test_condition.png), [sign_shape](../metrics/dinov3_lora/confusion_test_sign_shape.png)
- vjepa: [view_angle](../metrics/vjepa/confusion_test_view_angle.png), [mounting](../metrics/vjepa/confusion_test_mounting.png), [condition](../metrics/vjepa/confusion_test_condition.png), [sign_shape](../metrics/vjepa/confusion_test_sign_shape.png)
- vjepa_lora: [view_angle](../metrics/vjepa_lora/confusion_test_view_angle.png), [mounting](../metrics/vjepa_lora/confusion_test_mounting.png), [condition](../metrics/vjepa_lora/confusion_test_condition.png), [sign_shape](../metrics/vjepa_lora/confusion_test_sign_shape.png)
- convnext_frozen: [view_angle](../metrics/convnext_frozen/confusion_test_view_angle.png), [mounting](../metrics/convnext_frozen/confusion_test_mounting.png), [condition](../metrics/convnext_frozen/confusion_test_condition.png), [sign_shape](../metrics/convnext_frozen/confusion_test_sign_shape.png)
- convnext: [view_angle](../metrics/convnext/confusion_test_view_angle.png), [mounting](../metrics/convnext/confusion_test_mounting.png), [condition](../metrics/convnext/confusion_test_condition.png), [sign_shape](../metrics/convnext/confusion_test_sign_shape.png)
