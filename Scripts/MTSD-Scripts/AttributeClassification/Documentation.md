# Attribute Classification Methodology and Technical Audit

This document provides a detailed methodology description and correctness review for the `Scripts/MTSD-Scripts/AttributeClassification` experiment. It is written to be usable as source material for a dissertation methodology section, while also recording implementation-level details that affect reproducibility, interpretation, and validity.

## 1. Purpose of the Experiment

The attribute-classification experiment evaluates whether visual representations learned under different training paradigms can predict semantic and physical attributes of Maltese traffic signs from ground-truth sign crops. The task is strictly crop-level classification. It does not perform object detection, localisation, bounding-box regression, or instance segmentation.

The experiment predicts four attributes jointly:

| Attribute | Classes | Number of classes |
|---|---|---:|
| `view_angle` | `Front`, `Back`, `Side` | 3 |
| `mounting` | `Pole-Mounted`, `Wall-Mounted` | 2 |
| `condition` | `Good`, `Weathered`, `Heavily Damaged` | 3 |
| `sign_shape` | `Circular`, `Quadrangle`, `Triangular`, `Octagonal`, `Pentagon` | 5 |

The design is multi-task and multi-head: one image crop is passed through one backbone network, and the resulting feature vector is sent to four independent classification heads. Each head predicts one attribute. This is appropriate because the four labels describe different properties of the same traffic-sign instance, and sharing the visual representation avoids training four unrelated models over the same image data.

The experiment compares three representation strategies:

| Variant | Backbone | Training regime | Input size | Feature dim | Current run |
|---|---|---|---:|---:|---|
| `dinov3` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | Frozen self-supervised backbone, trained linear heads | 224 | 768 | `dinov3-20260703-021817` |
| `vjepa` | V-JEPA 2.1 `vjepa2_1_vit_large_384` through `torch.hub` | Frozen self-supervised video backbone, trained linear heads | 384 | 1024 | `vjepa-20260703-023332` |
| `convnext` | `torchvision` ConvNeXt-Tiny, ImageNet-pretrained | Fully fine-tuned supervised baseline | 224 | 768 | `convnext-20260703-025354` |

The comparison is therefore not a pure architecture comparison. It is a cross-paradigm comparison: frozen self-supervised representation probes are compared with a fully fine-tuned supervised ImageNet baseline. This is valid if framed as a comparison of practical representation-learning strategies, but it should not be described as a strictly equal training-budget or equal-adaptation comparison.

## 2. Repository Structure Reviewed

The implementation is organised as follows:

| Path | Role |
|---|---|
| `config/default.yaml` | Central source of paths, attributes, split settings, model choices, and hyperparameters. |
| `mtsd_attr/config.py` | YAML loading, repository-root path resolution, seeding, logging, per-model training overrides. |
| `mtsd_attr/data_manifest.py` | QA group discovery, crop extraction, deterministic split assignment, manifest maintenance. |
| `mtsd_attr/dataset.py` | Dataset class, image transforms, missing-label encoding, class-weight computation. |
| `mtsd_attr/multihead_model.py` | Shared multi-head classifier and masked multi-task cross-entropy loss. |
| `mtsd_attr/train_common.py` | Shared training loop, optimizer, scheduler, checkpointing, W&B logging, evaluation. |
| `mtsd_attr/evaluate.py` | Accuracy, macro-F1, per-class F1, confusion matrices, consolidated comparison reports. |
| `mtsd_attr/backbones/*.py` | Backbone wrappers for DINOv3, V-JEPA, and ConvNeXt-Tiny. |
| `train_dinov3.py`, `train_vjepa.py`, `train_convnext.py` | Thin entry scripts selecting one model variant. |
| `run_all.py` | End-to-end orchestration of manifest update, model training, and comparison report generation. |

The source code and README are broadly consistent. The README describes the main behaviour implemented in the code: QA-only data ingestion, deterministic image-level split assignment, crop padding, class-weighted masked multi-task loss, frozen probes for DINOv3 and V-JEPA, full fine-tuning for ConvNeXt, and macro-F1 as the primary metric.

## 3. Data Source and Inclusion Criteria

Only QA-approved annotation files are used. A group is included if and only if its annotation directory contains exactly one matching QA JSON under:

```text
Datasets/MTSD/Annotations/GRP-<N>/Final-QA/QA-GRP*.json
```

Raw annotator XML files are not used by this pipeline. The current manifest includes three QA-approved groups:

| Group | QA file | Images used | Crops kept | Too-small crops dropped | Dropped by configured attribute value |
|---|---|---:|---:|---:|---:|
| `GRP-1` | `Datasets/MTSD/Annotations/GRP-1/Final-QA/QA-GRP1.json` | 724 | 1,947 | 32 | 0 |
| `GRP-2` | `Datasets/MTSD/Annotations/GRP-2/Final-QA/QA-GRP2.json` | 630 | 1,554 | 65 | 2 |
| `GRP-3` | `Datasets/MTSD/Annotations/GRP-3/Final-QA/QA-GRP3.json` | 617 | 1,772 | 84 | 1 |
| **Total** |  | **1,971** | **5,273** | **181** | **3** |

The README states that four `Damaged-Unknown` shape instances were dropped as of GRP-1..3. The manifest currently records three crops dropped by configured value across GRP-2 and GRP-3. This small mismatch should be checked if the exact number is reported in the dissertation. The code is authoritative for the current run outputs because the manifest and metrics were generated from it.

### Crop extraction

Each annotation produces one crop, subject to filtering. The crop extraction uses:

| Setting | Value | Purpose |
|---|---:|---|
| `crop_padding` | `0.25` | Adds 25 percent of bounding-box width and height on each side to preserve context such as mounting and view angle. |
| `min_crop_size` | `16` pixels | Drops extremely small boxes that are unlikely to remain readable after resizing. |
| `crop_format` | `jpg` | Stores crops as JPEG. |
| `crop_quality` | `95` | Uses high JPEG quality to limit compression artefacts. |

The extraction code also checks image orientation. If the stored image dimensions do not match the annotation dimensions, it applies EXIF transposition when this makes the image match the annotation metadata. This is important because Label Studio annotations often refer to the EXIF-rotated visual view rather than the raw stored pixel orientation.

### Handling unknown and missing labels

Attribute values not listed in the configured vocabulary are handled according to:

```yaml
unknown_value_policy: mask
```

This means an unknown value is treated as a missing label for that head, rather than crashing the run. Missing labels are encoded as `-1` and masked in the loss and metrics. The exception is `sign_shape: Damaged-Unknown`, which is explicitly configured as a `drop_values` entry and causes the entire crop to be dropped.

Current missing-label counts are small:

| Split | Crops | Missing `view_angle` | Missing `mounting` | Missing `condition` | Missing `sign_shape` |
|---|---:|---:|---:|---:|---:|
| Train | 4,301 | 0 | 0 | 2 | 2 |
| Validation | 469 | 0 | 0 | 1 | 1 |
| Test | 503 | 0 | 0 | 1 | 0 |

This handling is correct for multi-task learning because it prevents a crop with one incomplete annotation from being discarded entirely, while ensuring that the missing attribute does not contribute an invalid loss or metric value.

## 4. Split Methodology

The split is deterministic and image-level rather than crop-level. This is a strong design choice because multiple signs can occur in the same source photo. If crops from the same photo were randomly split, train and test data could share background, lighting, viewpoint, and camera conditions, causing leakage.

The split function computes:

```text
u = sha256(hash_key + ":" + "GRP-<N>/<file_name>") / 2^64
```

and assigns:

```text
u < 0.8      -> train
u < 0.9      -> validation
otherwise    -> test
```

The configured split fractions are:

| Split | Fraction |
|---|---:|
| Train | 0.8 |
| Validation | 0.1 |
| Test | 0.1 |

The split salt is:

```yaml
hash_key: mtsd-attr-split-v1
```

This key should not be changed once experiments have started, because changing it would reshuffle every image and invalidate direct comparison with previous results.

Current split sizes are:

| Split | Crops | Source images |
|---|---:|---:|
| Train | 4,301 | 1,603 |
| Validation | 469 | 174 |
| Test | 503 | 194 |
| **Total** | **5,273** | **1,971** |

The realised crop proportions are approximately 81.6 percent train, 8.9 percent validation, and 9.5 percent test. This is close to the configured 80/10/10 target. Because assignment is image-level and hash-based, exact crop-level proportions are not guaranteed.

### Class distribution by split

The class distributions show significant imbalance, especially in `mounting`, `condition`, and `sign_shape`.

#### Train split

| Attribute | Class counts |
|---|---|
| `view_angle` | Front: 1,619; Back: 1,542; Side: 1,140 |
| `mounting` | Pole-Mounted: 3,821; Wall-Mounted: 480 |
| `condition` | Good: 3,239; Weathered: 791; Heavily Damaged: 269 |
| `sign_shape` | Circular: 1,935; Quadrangle: 1,148; Triangular: 608; Octagonal: 522; Pentagon: 86 |

#### Validation split

| Attribute | Class counts |
|---|---|
| `view_angle` | Front: 198; Back: 148; Side: 123 |
| `mounting` | Pole-Mounted: 414; Wall-Mounted: 55 |
| `condition` | Good: 329; Weathered: 114; Heavily Damaged: 25 |
| `sign_shape` | Circular: 221; Quadrangle: 116; Triangular: 63; Octagonal: 55; Pentagon: 13 |

#### Test split

| Attribute | Class counts |
|---|---|
| `view_angle` | Back: 200; Front: 185; Side: 118 |
| `mounting` | Pole-Mounted: 444; Wall-Mounted: 59 |
| `condition` | Good: 387; Weathered: 85; Heavily Damaged: 30 |
| `sign_shape` | Circular: 226; Quadrangle: 148; Octagonal: 61; Triangular: 57; Pentagon: 11 |

All classes are represented in all splits. This is important because macro-F1 and per-class F1 would be unstable or undefined if rare classes were absent. However, rare classes remain low-support, especially `Pentagon` and `Heavily Damaged`, so their F1 scores should be interpreted with wider uncertainty.

## 5. Image Preprocessing and Augmentation

The transform pipeline is defined in `mtsd_attr/dataset.py`.

All images are first padded to a square canvas with grey fill `(114, 114, 114)`. This preserves aspect ratio before resizing, which is particularly important for `sign_shape`; otherwise circular signs could be distorted into ellipses and shape classification would be artificially harmed.

### Training transforms

| Step | Value |
|---|---|
| Pad to square | Grey fill `(114, 114, 114)` |
| Random rotation | `10` degrees |
| Random resized crop | Target model size, scale `(0.8, 1.0)`, ratio `(0.9, 1.11)` |
| Horizontal flip | Disabled |
| Color jitter | Brightness `0.3`, contrast `0.3`, saturation `0.2`, hue `0.02` |
| Tensor conversion | `ToTensor()` |
| Normalisation | ImageNet mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)` |

Horizontal flipping is correctly disabled. For traffic signs, flipping may change text direction, symbol orientation, or viewpoint semantics, and therefore could introduce unrealistic training examples.

### Validation and test transforms

| Step | Value |
|---|---|
| Pad to square | Grey fill `(114, 114, 114)` |
| Resize | Direct resize to model input size |
| Tensor conversion | `ToTensor()` |
| Normalisation | ImageNet mean and standard deviation |

The validation and test transforms are deterministic, as expected.

## 6. Model Architecture

All variants share the same high-level classifier:

```text
input crop -> backbone -> pooled feature vector -> four independent heads
```

The heads are configured as:

```yaml
probe:
  type: linear
  mlp_hidden_dim: 512
  mlp_dropout: 0.2
```

Because `probe.type` is `linear`, the hidden dimension and dropout settings are present but unused in the current runs. Each attribute head is a single linear layer:

```text
Linear(feature_dim, number_of_classes_for_attribute)
```

The total number of trainable head parameters is small for the frozen variants:

| Variant | Total parameters | Trainable parameters | Interpretation |
|---|---:|---:|---|
| DINOv3 | 85.67M | approximately 0.01M | Backbone frozen; only linear heads trained. |
| V-JEPA | 304.69M | approximately 0.01M | Backbone frozen; only linear heads trained. |
| ConvNeXt-Tiny | 27.83M | 27.83M | Entire backbone and heads fine-tuned. |

### DINOv3

DINOv3 is loaded through Hugging Face `transformers`:

```yaml
hf_model_id: facebook/dinov3-vitb16-pretrain-lvd1689m
frozen: true
image_size: 224
```

The wrapper uses the model's `pooler_output` when available, otherwise the first token from `last_hidden_state`. The backbone is placed in eval mode, all backbone parameters are set to `requires_grad=False`, and backbone features are computed under `torch.no_grad()`.

This is a linear-probing setup. It tests the quality of the pretrained representation for traffic-sign attribute prediction without adapting the backbone.

### V-JEPA

V-JEPA is configured to use the `torch_hub` backend:

```yaml
backend: torch_hub
torch_hub_repo: facebookresearch/vjepa2
torch_hub_entrypoint: vjepa2_1_vit_large_384
torch_hub_checkpoint_url: https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt
num_frames: 2
frozen: true
image_size_torch_hub: 384
```

The current full run successfully used V-JEPA 2.1 through `torch_hub`, with:

```text
backend=torch_hub, num_frames=2, feature_dim=1024
```

Since V-JEPA is a video model, each static crop is repeated along the temporal axis to create a two-frame pseudo-clip. The output tokens are mean-pooled to produce a single feature vector.

This is a pragmatic adaptation of a video representation to a still-image attribute task. It is methodologically defensible, but it should be stated clearly because V-JEPA was not originally designed as a standard still-image classifier.

### ConvNeXt-Tiny

ConvNeXt-Tiny is loaded from `torchvision` with ImageNet-1K pretrained weights:

```yaml
backbone: convnext
frozen: false
image_size: 224
training_overrides:
  lr_heads: 4.0e-4
```

The model uses the ConvNeXt feature extractor and average pooling layer, producing a 768-dimensional feature vector. Unlike DINOv3 and V-JEPA, the full ConvNeXt backbone is trainable. The training override lowers the head learning rate from the global `1.0e-3` to `4.0e-4`; the backbone learning rate remains `5.0e-5`.

ConvNeXt is therefore a strong supervised fine-tuning baseline rather than a linear-probe baseline.

## 7. Loss Function

The experiment uses masked multi-task cross-entropy:

```text
L = sum_a lambda_a CE_a(logits_a, target_a)
```

where `a` indexes the four attributes and `lambda_a` is the configured head weight.

Current head weights are equal:

| Attribute | Loss weight |
|---|---:|
| `view_angle` | 1.0 |
| `mounting` | 1.0 |
| `condition` | 1.0 |
| `sign_shape` | 1.0 |

Missing labels are encoded as `-1` and ignored through `ignore_index=-1`. If an entire batch is missing labels for a head, that head is skipped for that step.

### Class-weighted cross-entropy

Class weighting is enabled:

```yaml
class_weighted_loss: true
```

Weights are computed from the training split only using:

```text
weight_c = N_labelled / (number_of_classes * count_c)
```

The weights used in the full runs were:

| Attribute | Class weights in configured class order |
|---|---|
| `view_angle` | `[0.886, 0.930, 1.258]` |
| `mounting` | `[0.563, 4.480]` |
| `condition` | `[0.442, 1.812, 5.327]` |
| `sign_shape` | `[0.444, 0.749, 1.414, 1.647, 9.998]` |

This is appropriate because the dataset is imbalanced. The weighting gives rare labels such as `Wall-Mounted`, `Heavily Damaged`, and `Pentagon` more influence during training. The main risk is that rare-class predictions can become noisy, especially when the rare class has very limited support. This is managed by reporting macro-F1 and per-class F1 rather than accuracy alone.

Label smoothing is disabled:

```yaml
label_smoothing: 0.0
```

This is acceptable for a first controlled comparison. If overconfidence or calibration were part of the research question, label smoothing or calibration metrics could be added, but they are not required for the current objective.

## 8. Optimisation Hyperparameters

The global training settings are:

| Hyperparameter | Value |
|---|---:|
| Epochs | 30 |
| Batch size | 32 |
| DataLoader workers | 4 |
| Optimizer | AdamW |
| Head learning rate | `1.0e-3` |
| Backbone learning rate | `5.0e-5` |
| Weight decay | `0.05` |
| Scheduler | Linear warmup followed by cosine decay |
| Warmup | 2 epochs |
| AMP | Enabled on CUDA using bfloat16 autocast |
| Gradient clipping | Global norm `1.0` |
| Early stopping patience | `0`, meaning disabled |
| Seed | 42 |

For ConvNeXt only:

| Hyperparameter | Value |
|---|---:|
| Head learning rate | `4.0e-4` |
| Backbone learning rate | `5.0e-5` |

The optimizer code creates separate parameter groups for heads and trainable backbone parameters. For frozen backbones, only the heads are present in the optimizer. For ConvNeXt, both the backbone and heads are optimized.

The scheduler runs per training step. It linearly increases the learning-rate multiplier during the warmup period and then decays it with a cosine schedule toward zero.

### Hyperparameter judgement

The hyperparameters are broadly sensible for the intended comparison:

1. **Thirty epochs** is reasonable for small linear probes and for fine-tuning a compact ConvNeXt model on 4,301 training crops. The validation curves show that all models reach their best validation macro-F1 within the 30-epoch budget.
2. **AdamW with weight decay 0.05** is standard for modern vision transformer and ConvNeXt-style training.
3. **Lower backbone learning rate** for ConvNeXt is appropriate because pretrained features should be adapted carefully.
4. **Class-weighted loss** is justified by severe imbalance in `mounting`, `condition`, and `sign_shape`.
5. **Macro-F1 selection** is correct for imbalanced multi-class classification.
6. **No early stopping** is acceptable because best-checkpoint selection is used. However, it costs extra compute after the best epoch and may be worth enabling if future runs become longer.

The most important methodological caveat is that ConvNeXt has much greater task-specific adaptation than DINOv3 and V-JEPA. If the dissertation wants to isolate representation quality, an additional ConvNeXt frozen-probe baseline, and possibly fine-tuned DINOv3/V-JEPA variants, would make the comparison more balanced.

## 9. Checkpoint Selection and Evaluation

The selection metric is:

```yaml
selection_metric: mean_macro_f1
```

In code, the best epoch is selected by the mean of the four validation macro-F1 scores, excluding heads with no labelled samples. This is correct for the problem because each attribute is treated as equally important, and macro-F1 prevents majority classes from dominating selection.

Two checkpoints are saved per variant:

| Checkpoint | Meaning |
|---|---|
| `best.pt` | Model state at the epoch with the highest validation mean macro-F1. |
| `last.pt` | Model state after the final training epoch. |

For frozen backbones, checkpoints omit backbone weights and save only the trained heads plus metadata. This is efficient and appropriate, provided the same backbone checkpoint remains available at evaluation time. For ConvNeXt, the full model state is saved.

The final test results are computed after reloading `best.pt`, not after using the final epoch. This is correct.

## 10. Metrics

The evaluation code reports, for each attribute:

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion correct; useful but can be misleading under imbalance. |
| Macro-F1 | Primary metric; averages per-class F1 equally. |
| Per-class F1 | Identifies whether rare classes are handled well. |
| Support | Number of labelled examples per class. |
| Confusion matrix | Error pattern analysis. |

Macro-F1 is the correct primary metric here. For example, `condition` is dominated by `Good`, so a model can obtain acceptable accuracy while performing poorly on `Weathered` and `Heavily Damaged`.

## 11. Current Full-Run Results

The consolidated test results are:

| Variant | `view_angle` macro-F1 | `mounting` macro-F1 | `condition` macro-F1 | `sign_shape` macro-F1 | Mean macro-F1 |
|---|---:|---:|---:|---:|---:|
| DINOv3 | 0.8594 | 0.8417 | 0.5777 | 0.8697 | 0.7871 |
| V-JEPA | 0.8751 | 0.8606 | 0.5294 | 0.8575 | 0.7806 |
| ConvNeXt-Tiny | 0.9153 | 0.8691 | 0.6899 | 0.9660 | 0.8601 |

The corresponding test accuracies are:

| Variant | `view_angle` accuracy | `mounting` accuracy | `condition` accuracy | `sign_shape` accuracy |
|---|---:|---:|---:|---:|
| DINOv3 | 0.8688 | 0.9264 | 0.7570 | 0.9066 |
| V-JEPA | 0.8827 | 0.9324 | 0.7171 | 0.9046 |
| ConvNeXt-Tiny | 0.9245 | 0.9523 | 0.8526 | 0.9682 |

Validation scores for the selected checkpoints were:

| Variant | Best epoch | Validation mean macro-F1 | Test mean macro-F1 |
|---|---:|---:|---:|
| DINOv3 | 12 | 0.7710 | 0.7871 |
| V-JEPA | 23 | 0.7512 | 0.7806 |
| ConvNeXt-Tiny | 29 | 0.8254 | 0.8601 |

The fact that ConvNeXt's best epoch is 29 suggests the 30-epoch budget was almost fully used. If future data increases or stronger augmentation is introduced, ConvNeXt may benefit from a slightly longer schedule. DINOv3 peaked earlier, at epoch 12, suggesting that its linear heads saturate more quickly.

### Per-class test F1

#### DINOv3

| Attribute | Class | Support | F1 |
|---|---|---:|---:|
| `view_angle` | Front | 185 | 0.8634 |
| `view_angle` | Back | 200 | 0.9132 |
| `view_angle` | Side | 118 | 0.8017 |
| `mounting` | Pole-Mounted | 444 | 0.9575 |
| `mounting` | Wall-Mounted | 59 | 0.7259 |
| `condition` | Good | 387 | 0.8615 |
| `condition` | Weathered | 85 | 0.5044 |
| `condition` | Heavily Damaged | 30 | 0.3673 |
| `sign_shape` | Circular | 226 | 0.9276 |
| `sign_shape` | Quadrangle | 148 | 0.9324 |
| `sign_shape` | Triangular | 57 | 0.8772 |
| `sign_shape` | Octagonal | 61 | 0.8254 |
| `sign_shape` | Pentagon | 11 | 0.7857 |

#### V-JEPA

| Attribute | Class | Support | F1 |
|---|---|---:|---:|
| `view_angle` | Front | 185 | 0.8918 |
| `view_angle` | Back | 200 | 0.9100 |
| `view_angle` | Side | 118 | 0.8235 |
| `mounting` | Pole-Mounted | 444 | 0.9606 |
| `mounting` | Wall-Mounted | 59 | 0.7606 |
| `condition` | Good | 387 | 0.8527 |
| `condition` | Weathered | 85 | 0.4098 |
| `condition` | Heavily Damaged | 30 | 0.3256 |
| `sign_shape` | Circular | 226 | 0.9369 |
| `sign_shape` | Quadrangle | 148 | 0.9210 |
| `sign_shape` | Triangular | 57 | 0.8500 |
| `sign_shape` | Octagonal | 61 | 0.8387 |
| `sign_shape` | Pentagon | 11 | 0.7407 |

#### ConvNeXt-Tiny

| Attribute | Class | Support | F1 |
|---|---|---:|---:|
| `view_angle` | Front | 185 | 0.9267 |
| `view_angle` | Back | 200 | 0.9596 |
| `view_angle` | Side | 118 | 0.8596 |
| `mounting` | Pole-Mounted | 444 | 0.9735 |
| `mounting` | Wall-Mounted | 59 | 0.7647 |
| `condition` | Good | 387 | 0.9237 |
| `condition` | Weathered | 85 | 0.6127 |
| `condition` | Heavily Damaged | 30 | 0.5333 |
| `sign_shape` | Circular | 226 | 0.9776 |
| `sign_shape` | Quadrangle | 148 | 0.9766 |
| `sign_shape` | Triangular | 57 | 0.9565 |
| `sign_shape` | Octagonal | 61 | 0.9194 |
| `sign_shape` | Pentagon | 11 | 1.0000 |

### Interpretation

ConvNeXt-Tiny is the strongest current model by mean test macro-F1. It outperforms both frozen self-supervised probes on all four attributes, with the largest advantage on `condition` and `sign_shape`.

DINOv3 and V-JEPA are close overall. V-JEPA performs slightly better on `view_angle` and `mounting`, while DINOv3 performs better on `condition` and `sign_shape`. This pattern is plausible: V-JEPA's video pretraining may help with viewpoint-like cues, while DINOv3's still-image representation may better encode object appearance and shape.

The hardest attribute is `condition`. All models show substantially lower macro-F1 for `condition` than for the other attributes. This is expected because condition labels are both imbalanced and visually subtle. `Heavily Damaged` has only 30 labelled test examples, and performance on this class is limited even for ConvNeXt.

The easiest attribute is `sign_shape`, especially for ConvNeXt. This is also plausible because shape is a strong geometric signal preserved by the square-padding transform.

## 12. Correctness Review

Overall, the implementation is technically coherent and suitable for the stated methodology. The most important correctness points are listed below.

### What appears correct

1. **QA-only ingestion is correctly implemented.** The code discovers QA-approved groups through `Final-QA/QA-GRP*.json` and excludes groups without QA files.
2. **Split assignment is image-level and deterministic.** This prevents leakage between crops from the same source image.
3. **Crop extraction preserves context.** The 25 percent padding is useful for mounting and view-angle cues.
4. **Aspect ratio is preserved before resizing.** Padding to square avoids distorting signs, which is especially important for shape classification.
5. **Missing labels are handled correctly.** Missing entries are masked in both training and evaluation.
6. **Class imbalance is addressed.** Inverse-frequency class weighting is computed from the training split only, avoiding validation/test leakage.
7. **Macro-F1 is the correct primary metric.** It is more informative than accuracy for imbalanced attributes.
8. **Best-checkpoint evaluation is implemented correctly.** Test metrics are computed from the validation-selected checkpoint.
9. **Frozen backbones are actually frozen.** DINOv3 and V-JEPA parameters are set to `requires_grad=False`, evaluated under `torch.no_grad()`, and kept in eval mode.
10. **ConvNeXt is correctly fine-tuned.** Its backbone is trainable and included in the optimizer with a lower learning rate.
11. **Smoke-test outputs are separated from full-run outputs.** This avoids contaminating real metrics with debugging metrics.
12. **Run metadata is logged.** The experiment log records run IDs, git commit, split sizes, group counts, best epoch, validation and test scores, and checkpoint path.

### Caveats to mention in the dissertation

1. **The comparison is asymmetric.** DINOv3 and V-JEPA are frozen linear probes, while ConvNeXt is fully fine-tuned. This is acceptable if framed as comparing representation-use strategies, but not if framed as an equal adaptation comparison.
2. **Rare-class metrics have high variance.** `Pentagon` has 11 test examples and `Heavily Damaged` has 30. A single prediction change can noticeably affect F1.
3. **Condition is intrinsically more subjective.** The lower `condition` scores may reflect both visual difficulty and annotation ambiguity.
4. **The manifest settings warning has a weakness.** If manifest-affecting settings change and `force=False`, the code warns that existing groups keep their old processing, but then writes the new settings into the manifest. This could hide a settings mismatch on later runs. Current results are not affected if the settings have not changed, but the behaviour should be fixed before major future reprocessing.
5. **No explicit validation checks enforce split-fraction sum or class presence.** The current manifest is fine, but future datasets would benefit from automated assertions or a saved data-audit table.
6. **DINOv3 preprocessing uses the shared ImageNet normalisation rather than a model-specific processor.** This is common and likely acceptable, but if exact model-card preprocessing differs, a processor-based transform could be tested.
7. **V-JEPA uses repeated still frames.** This is a practical static-image adaptation of a video model, not a native video evaluation.
8. **The exact dropped count for `Damaged-Unknown` should be reconciled.** Current manifest totals show three drops by configured value, while the README says four.

## 13. Suggested Methodology Text

The following text can be adapted directly into a dissertation methodology section:

> The attribute-classification experiment was formulated as a multi-task crop-level classification problem over QA-approved Maltese traffic-sign annotations. Each traffic-sign instance was represented by a ground-truth crop extracted from the final QA COCO annotations. Four attributes were predicted jointly: view angle, mounting type, physical condition, and sign shape. The model architecture consisted of a shared visual backbone followed by four independent classification heads, one per attribute. This allowed the model to learn or reuse a common visual representation while preserving separate output vocabularies for each semantic attribute.
>
> Only annotation groups containing a final QA JSON file were included. For each annotated bounding box, a crop was extracted with 25 percent padding on each side to retain contextual cues such as mounting structure and viewpoint. Crops with a minimum bounding-box side below 16 pixels were excluded. Instances labelled with the configured unlearnable shape value `Damaged-Unknown` were removed. Missing labels for individual attributes were retained as masked targets, allowing the remaining labels for the same crop to contribute to training.
>
> Data splitting was performed deterministically at source-image level using a salted SHA-256 hash of the group and image filename. This assigned images to train, validation, and test splits with target fractions 80 percent, 10 percent, and 10 percent respectively. Because all crops from the same source image inherit the same split, the procedure prevents leakage caused by multiple signs from the same photograph appearing across different splits. The current manifest contains 5,273 crops from 1,971 source images: 4,301 training crops, 469 validation crops, and 503 test crops.
>
> All crops were padded to a square canvas before resizing, preserving aspect ratio so that sign shape was not geometrically distorted. Training augmentation consisted of small random rotations, random resized cropping, and colour jitter. Horizontal flipping was disabled because mirrored traffic signs may be semantically unrealistic. Validation and test images used deterministic square padding, resizing, tensor conversion, and ImageNet normalisation.
>
> Three representation strategies were compared. DINOv3 used the gated `facebook/dinov3-vitb16-pretrain-lvd1689m` checkpoint as a frozen image backbone with trained linear attribute heads. V-JEPA used V-JEPA 2.1 through Meta's `torch.hub` interface as a frozen video backbone; each static crop was repeated along the temporal dimension to form a two-frame pseudo-clip, and output tokens were mean-pooled. ConvNeXt-Tiny used ImageNet-pretrained `torchvision` weights and was fully fine-tuned end to end, serving as a supervised baseline.
>
> Models were trained using AdamW, class-weighted cross-entropy, a batch size of 32, 30 epochs, two warmup epochs, and cosine learning-rate decay. The default head learning rate was `1e-3`, the backbone learning rate for trainable backbones was `5e-5`, and weight decay was `0.05`. ConvNeXt used a lower head learning rate of `4e-4`. Losses from the four heads were weighted equally. Class weights were computed from the training split only using inverse class frequency, reducing the effect of class imbalance in mounting, condition, and sign shape.
>
> Model selection used validation mean macro-F1, computed as the mean of the four per-attribute macro-F1 values. The selected checkpoint was then evaluated once on the held-out test split. Macro-F1 was used as the primary metric because several attributes were imbalanced; accuracy was reported as a secondary metric. Per-class F1 scores and confusion matrices were also produced to analyse rare-class behaviour.

## 14. Final Assessment

The experiment is in good methodological shape. The data ingestion, split logic, multi-task architecture, masked loss, class weighting, checkpoint selection, and metric reporting are all appropriate for the task.

The key interpretation point is that the current results show ConvNeXt-Tiny as the best-performing system, but ConvNeXt receives full fine-tuning while DINOv3 and V-JEPA are frozen probes. Therefore, the strongest defensible conclusion is:

```text
In the current setup, a fully fine-tuned supervised ConvNeXt-Tiny baseline outperforms frozen linear probes over DINOv3 and V-JEPA features for MTSD attribute classification, especially on condition and sign shape.
```

A stronger representation-learning conclusion would require additional balanced variants, such as frozen ConvNeXt probes and fine-tuned DINOv3/V-JEPA models. For the current dissertation methodology, however, the implementation is sufficiently rigorous as long as this comparison asymmetry and the rare-class limitations are stated clearly.
