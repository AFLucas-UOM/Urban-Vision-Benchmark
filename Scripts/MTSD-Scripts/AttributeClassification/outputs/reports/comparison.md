# Attribute classification: consolidated comparison

Generated 2026-07-03T01:09:47+00:00. Primary metric: **macro-F1** on the test split (robust to class imbalance; accuracy shown for reference).

## Macro-F1 by attribute

| Variant | view_angle | mounting | condition | sign_shape | Mean |
|---|---|---|---|---|---|
| dinov3 | 0.8594 | 0.8417 | 0.5777 | 0.8697 | **0.7871** |
| vjepa | 0.8751 | 0.8606 | 0.5294 | 0.8575 | **0.7806** |
| convnext | 0.9153 | 0.8691 | 0.6899 | 0.9660 | **0.8601** |

## Accuracy by attribute

| Variant | view_angle | mounting | condition | sign_shape |
|---|---|---|---|---|
| dinov3 | 0.8688 | 0.9264 | 0.7570 | 0.9066 |
| vjepa | 0.8827 | 0.9324 | 0.7171 | 0.9046 |
| convnext | 0.9245 | 0.9523 | 0.8526 | 0.9682 |

Per-class F1, supports, and confusion matrices: see outputs/metrics/<variant>/.
