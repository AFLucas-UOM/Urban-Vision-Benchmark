# MTSD Attribute Representation Learning

Multi-task, multi-head classification of Maltese traffic-sign **attributes** from
ground-truth crops, comparing adaptation regimes over shared pretrained
backbones. Part of the MSc AI dissertation on cross-paradigm visual perception
for municipal monitoring in Malta.

The project covers four backbone families (DINOv3, V-JEPA 2.1, ConvNeXt,
LingBot-Vision) in two experiment generations sharing one multi-head
architecture:

**Historical six-variant round** (completed on the GRP-1–GRP-3 snapshot; keys,
checkpoints, and metrics folders are preserved unchanged):

| Variant | Backbone | Adaptation | Trainable parameters |
| --- | --- | --- | --- |
| `dinov3` | DINOv3 ViT-B/16 | frozen | heads only |
| `dinov3_lora` | DINOv3 ViT-B/16 | LoRA (q_proj, v_proj) | LoRA adapters + heads |
| `vjepa` | V-JEPA 2.1 ViT-L/384 | frozen | heads only |
| `vjepa_lora` | V-JEPA 2.1 ViT-L/384 | LoRA (fused qkv) | LoRA adapters + heads |
| `convnext_frozen` | ConvNeXt-Tiny | frozen | heads only |
| `convnext` | ConvNeXt-Tiny | full fine-tune | backbone + heads |

**Model-size ablation matrix** (16 new variants; explicit names encode
family, size, and adaptation):

| Variant | Backbone | Model ID / entry point | Res | Adaptation |
| --- | --- | --- | ---: | --- |
| `dinov3_vitb_frozen` | DINOv3 ViT-B/16 | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 224 | frozen |
| `dinov3_vitl_frozen` | DINOv3 ViT-L/16 | `facebook/dinov3-vitl16-pretrain-lvd1689m` | 224 | frozen |
| `dinov3_vitb_lora` | DINOv3 ViT-B/16 | same as frozen | 224 | LoRA (q_proj, v_proj) |
| `dinov3_vitl_lora` | DINOv3 ViT-L/16 | same as frozen | 224 | LoRA (q_proj, v_proj) |
| `vjepa21_vitb_frozen` | V-JEPA 2.1 ViT-B/16 (~80M) | hub `vjepa2_1_vit_base_384` | 384 | frozen |
| `vjepa21_vitl_frozen` | V-JEPA 2.1 ViT-L/16 (~300M) | hub `vjepa2_1_vit_large_384` | 384 | frozen |
| `vjepa21_vitb_lora` | V-JEPA 2.1 ViT-B/16 | hub `vjepa2_1_vit_base_384` | 384 | LoRA (fused qkv) |
| `vjepa21_vitl_lora` | V-JEPA 2.1 ViT-L/16 | hub `vjepa2_1_vit_large_384` | 384 | LoRA (fused qkv) |
| `convnext_base_frozen` | ConvNeXt-Base | torchvision `convnext_base` | 224 | frozen |
| `convnext_large_frozen` | ConvNeXt-Large | torchvision `convnext_large` | 224 | frozen |
| `convnext_base_finetune` | ConvNeXt-Base | torchvision `convnext_base` | 224 | full fine-tune |
| `convnext_large_finetune` | ConvNeXt-Large | torchvision `convnext_large` | 224 | full fine-tune |
| `lingbot_vitb_frozen` | LingBot-Vision ViT-B/16 | `robbyant/lingbot-vision-vit-base` | 224 | frozen |
| `lingbot_vitl_frozen` | LingBot-Vision ViT-L/16 | `robbyant/lingbot-vision-vit-large` | 224 | frozen |
| `lingbot_vitb_lora` | LingBot-Vision ViT-B/16 | `robbyant/lingbot-vision-vit-base` | 224 | LoRA (fused qkv) |
| `lingbot_vitl_lora` | LingBot-Vision ViT-L/16 | `robbyant/lingbot-vision-vit-large` | 224 | LoRA (fused qkv) |

ConvNeXt-Tiny stays legacy-only (it is the historical baseline); LingBot has
no full fine-tuning (the ablation keeps SSL backbones at frozen/LoRA, and full
fine-tuning of a boundary-pretrained dense model is out of scope for this
comparison). The legacy variants are deliberately **excluded** from the
ablation profiles: `dinov3`/`dinov3_lora`, `vjepa`/`vjepa_lora` and
`convnext_frozen`/`convnext` would duplicate `dinov3_vitb_*`,
`vjepa21_vitl_*`, and ConvNeXt-Tiny runs.

Named experiment profiles (`run_all.py --profile NAME`):

| Profile | Variants | Content |
| --- | ---: | --- |
| `legacy_default` | 4 | exactly the historical default non-LoRA set |
| `size_ablation_frozen` | 8 | all frozen probes: DINOv3 B/L, V-JEPA 2.1 B/L, ConvNeXt Base/Large, LingBot B/L |
| `size_ablation_adapted` | 8 | DINOv3 B/L LoRA, V-JEPA 2.1 B/L LoRA, ConvNeXt Base/Large fine-tuned, LingBot B/L LoRA |
| `size_ablation_all` | 16 | union of the two |

The default experiment set (no `--profile`/`--variants`) is unchanged: the
four legacy non-LoRA variants.

Four attributes are predicted jointly, each by its own classification head on a
single shared backbone per variant:

| Attribute | Classes |
|---|---|
| `view_angle` | Front, Back, Side |
| `mounting` | Pole-Mounted, Wall-Mounted |
| `condition` | Good, Weathered, Heavily Damaged |
| `sign_shape` | Circular, Quadrangle, Triangular, Octagonal, Pentagon |

This experiment is strictly classification on ground-truth crops. It contains no
detection or localisation.

## Folder structure

```
Scripts/MTSD-Scripts/AttributeClassification/
  README.md                  this file
  config/default.yaml        the single source of truth for paths, attributes,
                             class vocabularies, hyperparameters, model variants
  mtsd_attr/                 the library package
    config.py                YAML loading, path resolution, seeding, logging
    variants.py              variant metadata, profiles, plan resolution
    data_manifest.py         group discovery, QA detection, crop extraction,
                             deterministic splits, manifest maintenance
    dataset.py               torch Dataset, transforms, class weights
    multihead_model.py       backbone -> N heads wrapper, masked multi-task loss
    train_common.py          the shared training loop (all variants; generic
                             gradient accumulation)
    evaluate.py              metrics, confusion matrices, consolidated +
                             size-ablation reports
    backbones/               dinov3 / vjepa / convnext / lingbot wrappers,
                             one shared interface
  train_variant.py           generic entry point: --variant <any config key>
  train_dinov3.py            compat wrapper: DINOv3 (frozen, probed)
  train_dinov3_lora.py       compat wrapper: DINOv3 (LoRA adapters + heads)
  train_vjepa.py             compat wrapper: V-JEPA 2/2.1 (frozen, probed)
  train_vjepa_lora.py        compat wrapper: V-JEPA 2.1 (LoRA adapters + heads)
  train_convnext.py          compat wrapper: ConvNeXt-Tiny (frozen; variant key convnext_frozen)
  train_convnext_finetuned.py  compat wrapper: ConvNeXt-Tiny (fine-tuned; variant key convnext)
  run_all.py                 manifest update + selected variants/profile + reports
  reset_outputs.py           archive/delete generated outputs for a fresh rerun
  inference/gradio_compare.py  side-by-side checkpoint inference UI
  tests/                     non-training validation (mocked backbones, profiles,
                             gradient accumulation, reports, plan safety)
  outputs/
    manifests/manifest.json  the unified crop manifest (append-only, see below)
    crops/GRP-*/             extracted sign crops (regenerable, git-ignored)
    checkpoints/<variant>/   best.pt and last.pt per variant (git-ignored);
                             every variant key owns its own isolated folder,
                             so legacy and ablation runs never collide
    metrics/<variant>/       per-split metrics JSON/CSV + confusion PNGs
    reports/                 comparison.{csv,md,png} (historical consolidated
                             comparison) + size_ablation.{csv,md} (new)
    logs/                    per-run log files (git-ignored)
    experiment_log.jsonl     one line per completed run, appended forever
```

The package is named `mtsd_attr` rather than `src` deliberately: Meta's vjepa2
torch.hub repository contains its own top-level package named `src`, and two
packages with the same import name cannot coexist in one process, which would
break V-JEPA 2.1 loading.

## Environment

Training runs in the dedicated conda env **mtsd-attrcls** (Python 3.11, CUDA 12.8
torch). Create it reproducibly from
[Requirements/CondaEnvironments/](../../../Requirements/CondaEnvironments/README.md)
(`.\setup_conda_env.ps1 -Name mtsd-attrcls`), or manually from the repository
root (pip pins in
[Requirements/requirements-attribute-classification.txt](../../../Requirements/requirements-attribute-classification.txt)):

```
conda create -n mtsd-attrcls python=3.11 -y
conda activate mtsd-attrcls
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r Requirements/requirements-attribute-classification.txt
```

Always run with user-site packages disabled (`-s` plus `PYTHONNOUSERSITE=1`),
because this machine has a shared `AppData\Roaming\Python\Python311` site-packages
that shadows env packages with incompatible versions:

```
cd Scripts/MTSD-Scripts/AttributeClassification
$env:PYTHONNOUSERSITE = "1"
python -s run_all.py
```

Weights & Biases logging uses `WANDB_API_KEY` from the repository-root `.env`
(loaded automatically via python-dotenv) and logs to the project
**MSc-MTSD-Attributes**. Smoke-test runs set `WANDB_MODE=disabled`. W&B is
controlled entirely from `config/default.yaml`:

- `wandb.enabled: false` disables logging completely (no key needed);
- `wandb.group` labels a fresh experiment round (e.g. `rerun-2026-07`) so new
  runs are cleanly separated from older ones in the W&B UI;
- `wandb.run_name_prefix` optionally prefixes every run name
  (names are otherwise `<variant>-<timestamp>`).

The LoRA variants additionally require `peft` (>= 0.19, already in
`Requirements/requirements-attribute-classification.txt`); the non-LoRA
variants never import it, so they run fine without LoRA-specific
initialisation. The LingBot variants require the `lingbot-vision` package
(pinned to a specific upstream commit in the same requirements file; installs
`omegaconf`, `opencv-python-headless`, `huggingface_hub` alongside). If it is
missing, the LingBot variants raise a clear `BackboneUnavailableError` with
install instructions and are skipped; nothing else is affected.

## How to run

List every configured variant / the named profiles (read-only, no data or
model access):

```
PYTHONNOUSERSITE=1 .../envs/mtsd-attrcls/python.exe -s run_all.py --list-variants
PYTHONNOUSERSITE=1 .../envs/mtsd-attrcls/python.exe -s run_all.py --list-profiles
```

Print the resolved experiment matrix for a selection without running anything
(**no manifest refresh, no model loading or downloads, no checkpoints, no
W&B**):

```
PYTHONNOUSERSITE=1 .../envs/mtsd-attrcls/python.exe -s run_all.py --plan --profile size_ablation_all
```

The default experiment set (manifest refresh, the four legacy non-LoRA
variants, comparison report) — unchanged historical behaviour:

```
PYTHONNOUSERSITE=1 .../envs/mtsd-attrcls/python.exe -s run_all.py
```

Run a named profile (frozen-only, adapted-only, or the complete ablation):

```
PYTHONNOUSERSITE=1 .../envs/mtsd-attrcls/python.exe -s run_all.py --profile size_ablation_frozen
PYTHONNOUSERSITE=1 .../envs/mtsd-attrcls/python.exe -s run_all.py --profile size_ablation_adapted
PYTHONNOUSERSITE=1 .../envs/mtsd-attrcls/python.exe -s run_all.py --profile size_ablation_all
```

Selection precedence is deterministic: `--variants` replaces the set entirely
and is mutually exclusive with `--profile`; `--profile` otherwise selects the
base set; without either, `run_all.variants` (the legacy default) is used;
`--include` appends extra variants to whichever base set was selected.

Add the optional legacy LoRA variants on top of the default set:

```
PYTHONNOUSERSITE=1 .../envs/mtsd-attrcls/python.exe -s run_all.py --include dinov3_lora vjepa_lora
```

Run an explicit set (replaces the default entirely — e.g. one variant only):

```
PYTHONNOUSERSITE=1 .../envs/mtsd-attrcls/python.exe -s run_all.py --variants dinov3_vitl_frozen
```

One variant via the generic entry point (works for every configured variant;
the historical thin scripts such as `train_convnext.py` remain as
compatibility wrappers):

```
PYTHONNOUSERSITE=1 .../envs/mtsd-attrcls/python.exe -s train_variant.py --variant lingbot_vitb_frozen
```

Re-evaluate an existing checkpoint (reconstructed from its stored metadata,
including adaptation mode and LoRA hyperparameters):

```
PYTHONNOUSERSITE=1 .../envs/mtsd-attrcls/python.exe -s -m mtsd_attr.evaluate --variant dinov3_lora --split test
```

Fast end-to-end pipeline verification (64 crops per split, 1 epoch, W&B off,
metrics and checkpoints quarantined into `<variant>-smoke/` folders):

```
PYTHONNOUSERSITE=1 .../envs/mtsd-attrcls/python.exe -s run_all.py --smoke-test
```

Manifest update only:

```
PYTHONNOUSERSITE=1 .../envs/mtsd-attrcls/python.exe -s -m mtsd_attr.data_manifest
```

## Resetting for a fresh rerun

`reset_outputs.py` clears **generated** experiment artefacts (checkpoints,
metrics, reports, logs, `experiment_log.jsonl`, local `wandb/` run dirs) so a
rerun starts clean. It is a dry run by default and never touches raw images,
QA annotations, `outputs/crops/`, or `outputs/manifests/` unless the
`--include-*` flags are passed explicitly:

```
python reset_outputs.py               # dry run: list what would be reset
python reset_outputs.py --archive     # move to outputs/_archive/<timestamp>/ (recommended)
python reset_outputs.py --delete      # permanent, asks for confirmation
```

For a fully fresh W&B round, also set `wandb.group` in `config/default.yaml`
(e.g. `rerun-2026-07`) and, if desired, delete old runs from the
MSc-MTSD-Attributes project in the W&B web UI (local `wandb/` dirs are already
covered by the reset script).

Rebuild the consolidated report from saved metrics:

```
PYTHONNOUSERSITE=1 .../envs/mtsd-attrcls/python.exe -s -m mtsd_attr.evaluate
```

All scripts accept `--config path/to/other.yaml` to run with a different config.

## Data schema and experiment snapshot (confirmed 2026-07-11)

Ground truth is **only** the QA-approved COCO JSONs:

```
Datasets/MTSD/Annotations/GRP-<N>/Final-QA/QA-GRP<N>.json
```

A group is **QA-approved if that file exists** (exactly one match of
`QA-GRP*.json` under `Final-QA/`). Raw annotator output
(`Fiverr-Annotations/*.xml`) is never read. The current annotation tree has
QA-approved JSONs for **GRP-1, GRP-2, GRP-3, GRP-5 and GRP-6** (3,098 images /
8,372 boxes); further groups join automatically once their `Final-QA` JSON
lands (no config change needed). The completed six-model comparison is
intentionally a **historical GRP-1--GRP-3 manifest snapshot** (1,971 source
images and 5,273 retained crops). Do not rerun only for GRP-5/GRP-6: first
finish the annotation-scope decision, then refresh the manifest and run one
separately labelled final round. In each QA JSON:

- `images[]`: `id`, `file_name`, `width`, `height`, and `source_image` (a
  repo-relative path into `Datasets/MTSD/GRP-<N>/Images/`). Width/height describe the
  EXIF-rotated view Label Studio annotated; crop extraction detects and applies
  the rotation when the stored pixels differ.
- `annotations[]`: `image_id`, `category_id` (sign type, unused here), `bbox`
  (`[x, y, w, h]` floats), and `attributes` containing `view_angle`, `mounting`,
  `condition`, `sign_shape` plus Label Studio bookkeeping.

## How new annotation groups are picked up

Every run (of any entry point) starts with group discovery. No code or config
changes are needed when a new group finishes QA:

1. All `GRP-*` folders under `Datasets/MTSD/` and `Datasets/MTSD/Annotations/` are
   scanned. Each is logged as **included** (Final-QA JSON present), **awaiting
   QA** (annotations but no Final-QA JSON), or **no annotations yet**.
2. For each QA-approved group the QA file's sha256 is compared with the hash
   recorded at ingestion. Unchanged groups are skipped (no re-hash of images,
   no re-cropping). New groups are ingested. A group whose QA file *changed* is
   re-ingested with a prominent warning, since results before/after may not be
   comparable.
3. Ingestion extracts one crop per annotation (bbox padded by
   `data.crop_padding`, default 25% per side, so mounting/context cues survive;
   aspect preserved) into `outputs/crops/GRP-<N>/ann_<id>.jpg` and appends one
   record per crop to `outputs/manifests/manifest.json`.

Exclusions during ingestion (all counted in the log): crops whose bbox min-side
is below `data.min_crop_size` (default 16 px — unreadable after resizing), and
crops whose `sign_shape` is `Damaged-Unknown` (3 instances in the historical
GRP-1--GRP-3 manifest; the latest four-group QA audit finds 5 in total;
decision 2026-07-03: dropped entirely). Crops missing a label for some attribute
are **kept**; the loss masks the missing head (see below).

## Split assignment and stability

Splits must stay comparable as groups arrive, so assignment is a **pure
function of the source image identity**, not a random shuffle:

```
u = sha256(splits.hash_key + ":" + "GRP-<N>/<file_name>") / 2^64
u < 0.8 -> train;  u < 0.9 -> val;  else test
```

- Assignment is per **source image**: every crop of the same photo (and any
  future augmented/duplicate view keyed to it) lands in the same split, so
  there is no crop-level leakage.
- Existing images keep their split forever; images from newly QA'd groups are
  assigned by the same rule on arrival. Re-running never reshuffles.
- Strict stratification is impossible under sticky hashing, so the manifest
  step instead logs the per-split class distribution of every attribute on
  every run; with the current data every class is represented in every split.
- `splits.hash_key` must never change once experiments have started; changing
  it (or seed/fractions/filters) triggers a settings-changed warning and
  requires a deliberate `--force` rebuild.

## Models and adaptation modes

All variants share one architecture: backbone -> pooled feature vector -> four
independent heads (one per attribute, sized by its vocabulary). Heads are
linear probes by default (`probe.type: mlp` switches to a one-hidden-layer MLP).
Each variant's `adaptation` field in the config selects the training regime:

- **frozen** — all backbone parameters have `requires_grad=False`, the
  backbone runs under `torch.no_grad()` and is pinned to `eval()` even during
  head training. Only the heads train. Checkpoints store heads only.
- **lora** — base backbone weights stay frozen; LoRA adapters (via `peft`) are
  injected into attention projections and train together with the heads.
  Injection is verified: if a configured target module matches nothing in the
  loaded architecture, the run fails with the list of available module names —
  it never silently falls back to frozen or full fine-tuning. Checkpoints
  store heads + adapter tensors, plus the adaptation mode and LoRA
  hyperparameters, so evaluation can reconstruct the model exactly.
- **finetune** — the whole network trains end to end, with separate learning
  rates for backbone (`lr_backbone`) and heads (`lr_heads`). LoRA-resident
  trainable parameters also use `lr_backbone` in the lora variants.

Backbones:

- **DINOv3** (`facebook/dinov3-vitb16-pretrain-lvd1689m` /
  `facebook/dinov3-vitl16-pretrain-lvd1689m`): loaded via transformers. Both
  checkpoints are **gated** on Hugging Face; request access for *each* model
  id and log in (`hf auth login`) first — if access is not granted the
  variant exits/skips with instructions rather than crashing the run (ViT-B
  access was granted for this machine's account on 2026-07-03; ViT-L needs
  its own grant). The LoRA variants target the per-block `q_proj` and
  `v_proj` Linear layers; ViT-B and ViT-L share the same DINOv3ViT block
  implementation (only depth/width differ), and `lora.py` verifies at load
  time that every configured target matched, failing loudly otherwise.
- **V-JEPA**: a video model — each image is repeated along the temporal axis
  to form a minimal pseudo-clip (`num_frames`, default 2, the tubelet size),
  and token embeddings are mean-pooled (768-d for ViT-B, 1024-d for ViT-L).
  Backend is a config switch:
  - `backend: torch_hub` (default) loads **V-JEPA 2.1** from Meta's GitHub
    repo via the official entry points `vjepa2_1_vit_base_384` /
    `vjepa2_1_vit_large_384` (each returns an `(encoder, predictor)` tuple;
    the encoder is used). As of 2026-07 Meta's hubconf ships a localhost
    placeholder checkpoint URL, so the real checkpoint is pre-cached from
    `torch_hub_checkpoint_url` (dl.fbaipublicfiles.com;
    `vjepa2_1_vitb_dist_vitG_384.pt` / `vjepa2_1_vitl_dist_vitG_384.pt`,
    ~1–4.8 GB) into the torch.hub cache first.
  - **Version policy**: the new `vjepa21_*` variants set
    `allow_backend_fallback: false` — a failed 2.1 hub load raises
    `BackboneUnavailableError` instead of silently substituting the V-JEPA
    2.0 transformers checkpoint, because a cross-version swap would
    invalidate the size ablation. Only the legacy `vjepa`/`vjepa_lora`
    variants retain the historical fallback-with-warning to
    `facebook/vjepa2-vitl-fpc64-256` (V-JEPA 2.0), kept solely for
    backwards compatibility and marked as legacy behaviour. The requested
    and actually loaded backend/version are recorded in every checkpoint,
    metrics bundle, and W&B config.
  - LoRA targets the fused `qkv` projections (Meta's ViT has no separate
    q/v Linears, so q, k and v are adapted jointly); the hub architecture is
    required — a LoRA variant fails explicitly rather than adapting the
    wrong layers of the 2.0 fallback.
- **ConvNeXt** (torchvision, ImageNet-1K `IMAGENET1K_V1` weights): a
  validated registry maps `model_size` to the official constructors —
  `convnext_tiny` (768-d, legacy only), `convnext_base` (1024-d),
  `convnext_large` (1536-d). The feature dimension is derived from the
  instantiated architecture, not hard-coded. Base/Large serve both frozen
  probes and the fully fine-tuned baselines.
- **LingBot-Vision** (Robbyant/Ant Group;
  `robbyant/lingbot-vision-vit-base` / `-vit-large`): loaded through the
  official `lingbot_vision.load_pretrained_backbone` interface (pip package
  `lingbot-vision`, pinned to a reviewed commit), which returns a
  `(model, embed_dim)` tuple — not a transformers AutoModel. The forward
  pass returns the model's token dictionary; the wrapper mean-pools the
  normalised patch tokens (`x_norm_patchtokens`, the representation
  documented for frozen readouts; `pooling: cls` switches to the CLS token).
  The embedding dimension comes from the loader (768 ViT-B / 1024 ViT-L) and
  is re-verified by a dummy forward. RoPE position embeddings make the input
  size flexible (224 configured; native pretraining used 512). LoRA targets
  the fused per-block `qkv` Linear (verified against the pinned
  `lingbot_vision/layers.py`). **Frozen and LoRA only** — full fine-tuning
  is deliberately not implemented for LingBot in this experiment. A missing
  package or checkpoint raises `BackboneUnavailableError` with instructions.

Every run logs a parameter breakdown (total, trainable, trainable %, backbone
vs heads vs LoRA) plus explicit variant metadata (family, architecture, size,
adaptation, model id, requested/loaded backend and version, resolution,
feature dim, physical/effective batch size) and stores it all in the
checkpoint, the experiment log, the metrics JSON, and W&B, so the regimes are
explicit: frozen ~0.01M trainable (heads only) · LoRA ~0.3–0.8M (adapters +
heads) · fine-tune 28M–198M (everything; ConvNeXt Tiny→Large).

### Batching, gradient accumulation, and VRAM

`training.gradient_accumulation_steps` (default 1) adds generic gradient
accumulation to the shared loop: the loss of each batch is scaled by its
accumulation group's size before backward, gradients are clipped immediately
before each optimiser step, and the optimiser + scheduler step once per group
(the final partial group of an epoch is handled correctly). Validation and
test behaviour are unchanged. Physical and effective batch sizes are logged
per epoch and recorded in all metadata.

Per-variant `training_overrides` keep `effective_batch_size =
batch_size × gradient_accumulation_steps = 32` everywhere. The shipped values
are **initial operational settings for a 24 GB RTX 4090** (32×1 for all
frozen probes and ConvNeXt-Base fine-tuning; 16×2 for the ViT-L LoRA
variants, ConvNeXt-Large fine-tuning, and all LingBot/V-JEPA-2.1 LoRA
variants); the smoke round establishes whether they fit. Expect the highest
VRAM pressure from `vjepa21_vitl_lora` (384 px, backprop through ViT-L) and
`convnext_large_finetune`. **There is no automatic OOM batch-size
reduction** — silently changing the batch would change experimental
conditions, so an OOM fails the run and the config must be adjusted
explicitly.

## Training schedule and early stopping

`training.max_epochs` (default **150**) is a maximum budget, not a target; the
cosine schedule spans `max_epochs x len(train_loader)` steps with a linear
warmup of `training.warmup_epochs` (default **5**) epochs. Early stopping
(`training.early_stopping`, enabled by default) monitors validation
`mean_macro_f1`: an improvement counts only if it exceeds the previous best by
`min_delta` (default 0.001); after `patience` (default 10) consecutive epochs
without a qualifying improvement, training stops. Best-checkpoint selection is
independent of early stopping (any strict improvement updates `best.pt`), test
metrics are never used for stopping or selection, and after training `best.pt`
is reloaded for the final test evaluation. Console logs, W&B
(`early_stopping/epochs_without_improvement`, `stop_reason`), the checkpoint
metadata, and the consolidated report all record the stopping epoch, best
epoch, and whether the run ended by early stopping or by exhausting
`max_epochs`.

### Loss

Joint loss = configurably weighted sum of per-head cross-entropies
(`training.head_loss_weights`, default equal). Missing labels are encoded as
-1 and masked out per head per batch (a handful of crops lack `condition` or
`sign_shape`). `training.class_weighted_loss: true` (default) applies
inverse-frequency class weights computed from the **training split** of the
current manifest — important because condition and mounting are heavily
imbalanced.

## Outputs and how to read them

- `outputs/metrics/<variant>/{val,test}_metrics.json` — accuracy, macro-F1,
  per-class F1, support, and confusion matrix per attribute, plus the run id.
- `outputs/metrics/<variant>/{split}_per_class_f1.csv` — flat per-class F1
  table for the dissertation.
- `outputs/metrics/<variant>/confusion_{split}_{attr}.png` — row-normalised
  confusion heatmaps with raw counts annotated.
- `outputs/reports/comparison.{csv,md}` and `comparison_macro_f1.png` — the
  consolidated comparison across **every variant with completed test metrics**
  (the report never assumes a fixed variant count): macro-F1 and accuracy per
  attribute, run details (family, architecture, size, adaptation, resolution,
  best/stopping epoch, stop reason, val score, parameter counts, training
  duration, loaded backend/version), per-class F1 tables with supports, and
  links to the confusion matrices. **Macro-F1 is the primary metric**: with
  Good ~78% of condition labels, accuracy rewards majority-class collapse,
  while macro-F1 weights rare classes (Heavily Damaged, Pentagon) equally.
- `outputs/reports/size_ablation.{csv,md}` — the model-size ablation report
  (distinct file names; the historical comparison files are never
  overwritten). Covers only the `size_ablation_all` profile members and
  groups them into: (1) frozen probes per family, (2) LoRA per family,
  (3) ConvNeXt full fine-tuning, (4) an all-frozen cross-family ranking —
  each table explicitly labelled with its adaptation mode. Rows carry
  variant, family, size, architecture, adaptation, resolution, parameter
  counts, best epoch, val/test mean macro-F1, per-head test macro-F1 and
  accuracy, training duration, test throughput, and backend/version.
- `outputs/experiment_log.jsonl` — one line per run: run id, variant,
  adaptation, git commit, groups and crop counts used, split sizes, parameter
  breakdown, best/stopping epoch and stop reason, val/test scores, checkpoint
  path. This is the longitudinal record as new groups arrive over time.
- Checkpoints: `best.pt` (highest val mean macro-F1 across heads) and
  `last.pt` per variant, with adaptation mode, LoRA hyperparameters, and the
  parameter breakdown in the metadata. Smoke-test runs write to
  `checkpoints/<variant>-smoke/` and `metrics/<variant>-smoke/`, never
  touching real-run outputs.
- W&B: per-epoch losses (total and per head), per-attribute val macro-F1 and
  accuracy, and final test summaries in project **MSc-MTSD-Attributes**.

## Reproducibility

- Seed (`seed: 42`) is applied to Python, NumPy, and torch (CPU+CUDA) at the
  start of every run and recorded in the manifest settings and W&B config.
- Split assignment is seedless-deterministic (content hashing), so it is
  reproducible independently of library RNGs.
- Residual non-determinism: cuDNN benchmark autotuning is enabled for speed
  (kernel selection may vary run to run), DataLoader workers interleave
  augmentation RNG, and bf16 autocast reductions are order-dependent. Expect
  metric jitter in the third decimal place rather than bit-exact repeats; the
  deterministic splits and logged config keep runs comparable.
- Every run logs: config snapshot, git commit, groups included with counts,
  split sizes, class weights, and start/end times (see `outputs/logs/`).
