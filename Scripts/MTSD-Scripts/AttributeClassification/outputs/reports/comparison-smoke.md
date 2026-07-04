# Attribute classification: consolidated comparison

Generated 2026-07-03T13:16:18+00:00. Primary metric: **macro-F1** on the test split (robust to class imbalance; accuracy shown for reference).

## Macro-F1 by attribute

| Variant | view_angle | mounting | condition | sign_shape | Mean |
|---|---|---|---|---|---|
| dinov3 | 0.2688 | 0.5594 | 0.1411 | 0.0961 | **0.2664** |
| dinov3_lora | 0.2473 | 0.7037 | 0.3302 | 0.1987 | **0.3700** |
| vjepa | 0.3168 | 0.3594 | 0.2901 | 0.1263 | **0.2731** |
| vjepa_lora | 0.2410 | 0.3667 | 0.3319 | 0.2032 | **0.2857** |
| convnext_frozen | 0.2978 | 0.6807 | 0.2854 | 0.1140 | **0.3445** |
| convnext | 0.2586 | 0.6530 | 0.2802 | 0.0945 | **0.3216** |

## Accuracy by attribute

| Variant | view_angle | mounting | condition | sign_shape |
|---|---|---|---|---|
| dinov3 | 0.3281 | 0.6562 | 0.2222 | 0.1562 |
| dinov3_lora | 0.3125 | 0.8438 | 0.6508 | 0.2188 |
| vjepa | 0.3750 | 0.3750 | 0.7460 | 0.2656 |
| vjepa_lora | 0.3906 | 0.4062 | 0.7619 | 0.4688 |
| convnext_frozen | 0.3750 | 0.7656 | 0.4921 | 0.1562 |
| convnext | 0.3438 | 0.7344 | 0.4762 | 0.1250 |

## Run details

| Variant | Adaptation | Best epoch | Stopped at | Stop reason | Val mean macro-F1 | Total params | Trainable | Trainable % |
|---|---|---|---|---|---|---|---|---|
| dinov3 | frozen | 1 | 1 | max_epochs | 0.3134 | 85,670,413 | 9,997 | 0.0117 |
| dinov3_lora | lora | 1 | 1 | max_epochs | 0.3602 | 85,965,325 | 304,909 | 0.3547 |
| vjepa | frozen | 1 | 1 | max_epochs | 0.3068 | 304,694,285 | 13,325 | 0.0044 |
| vjepa_lora | lora | 1 | 1 | max_epochs | 0.2923 | 305,480,717 | 799,757 | 0.2618 |
| convnext_frozen | frozen | 1 | 1 | max_epochs | 0.3093 | 27,828,589 | 9,997 | 0.0359 |
| convnext | finetune | 1 | 1 | max_epochs | 0.3009 | 27,828,589 | 27,828,589 | 100.0 |

## Per-class F1: view_angle

| Class | Support | dinov3 | dinov3_lora | vjepa | vjepa_lora | convnext_frozen | convnext |
|---|---|---|---|---|---|---|---|
| Front | 21 | 0.4706 | 0.4225 | 0.0000 | 0.0000 | 0.4800 | 0.4533 |
| Back | 26 | 0.1290 | 0.2286 | 0.5217 | 0.5750 | 0.3226 | 0.3226 |
| Side | 17 | 0.2069 | 0.0909 | 0.4286 | 0.1481 | 0.0909 | 0.0000 |

## Per-class F1: mounting

| Class | Support | dinov3 | dinov3_lora | vjepa | vjepa_lora | convnext_frozen | convnext |
|---|---|---|---|---|---|---|---|
| Pole-Mounted | 54 | 0.7660 | 0.9074 | 0.4595 | 0.5250 | 0.8454 | 0.8211 |
| Wall-Mounted | 10 | 0.3529 | 0.5000 | 0.2593 | 0.2083 | 0.5161 | 0.4848 |

## Per-class F1: condition

| Class | Support | dinov3 | dinov3_lora | vjepa | vjepa_lora | convnext_frozen | convnext |
|---|---|---|---|---|---|---|---|
| Good | 54 | 0.3768 | 0.7800 | 0.8704 | 0.8624 | 0.6824 | 0.6667 |
| Weathered | 8 | 0.0000 | 0.2105 | 0.0000 | 0.1333 | 0.1739 | 0.1739 |
| Heavily Damaged | 1 | 0.0465 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Per-class F1: sign_shape

| Class | Support | dinov3 | dinov3_lora | vjepa | vjepa_lora | convnext_frozen | convnext |
|---|---|---|---|---|---|---|---|
| Circular | 26 | 0.0000 | 0.2000 | 0.0000 | 0.5672 | 0.0000 | 0.0000 |
| Quadrangle | 27 | 0.3265 | 0.2286 | 0.5172 | 0.4490 | 0.4103 | 0.3243 |
| Triangular | 6 | 0.1538 | 0.4167 | 0.0000 | 0.0000 | 0.0690 | 0.0714 |
| Octagonal | 2 | 0.0000 | 0.1481 | 0.1143 | 0.0000 | 0.0000 | 0.0000 |
| Pentagon | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0909 | 0.0769 |

## Confusion matrices

- dinov3: [view_angle](../metrics/dinov3-smoke/confusion_test_view_angle.png), [mounting](../metrics/dinov3-smoke/confusion_test_mounting.png), [condition](../metrics/dinov3-smoke/confusion_test_condition.png), [sign_shape](../metrics/dinov3-smoke/confusion_test_sign_shape.png)
- dinov3_lora: [view_angle](../metrics/dinov3_lora-smoke/confusion_test_view_angle.png), [mounting](../metrics/dinov3_lora-smoke/confusion_test_mounting.png), [condition](../metrics/dinov3_lora-smoke/confusion_test_condition.png), [sign_shape](../metrics/dinov3_lora-smoke/confusion_test_sign_shape.png)
- vjepa: [view_angle](../metrics/vjepa-smoke/confusion_test_view_angle.png), [mounting](../metrics/vjepa-smoke/confusion_test_mounting.png), [condition](../metrics/vjepa-smoke/confusion_test_condition.png), [sign_shape](../metrics/vjepa-smoke/confusion_test_sign_shape.png)
- vjepa_lora: [view_angle](../metrics/vjepa_lora-smoke/confusion_test_view_angle.png), [mounting](../metrics/vjepa_lora-smoke/confusion_test_mounting.png), [condition](../metrics/vjepa_lora-smoke/confusion_test_condition.png), [sign_shape](../metrics/vjepa_lora-smoke/confusion_test_sign_shape.png)
- convnext_frozen: [view_angle](../metrics/convnext_frozen-smoke/confusion_test_view_angle.png), [mounting](../metrics/convnext_frozen-smoke/confusion_test_mounting.png), [condition](../metrics/convnext_frozen-smoke/confusion_test_condition.png), [sign_shape](../metrics/convnext_frozen-smoke/confusion_test_sign_shape.png)
- convnext: [view_angle](../metrics/convnext-smoke/confusion_test_view_angle.png), [mounting](../metrics/convnext-smoke/confusion_test_mounting.png), [condition](../metrics/convnext-smoke/confusion_test_condition.png), [sign_shape](../metrics/convnext-smoke/confusion_test_sign_shape.png)
