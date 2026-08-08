# PromptDetect — Research Evaluation Interface

A local, research-grade Gradio application for evaluating **promptable
perception models** — **SAM 3**, **SAM 3.1**, **NVIDIA Cosmos Reason2**, and
**NVIDIA LocateAnything-3B** — for object localisation on urban-monitoring
imagery.

Built for MSc dissertation work on:

- **MDWD** — Maltese Domestic Waste Dataset
- **MTSD** — Maltese Traffic Sign Dataset

---

## Purpose

This is **not** a production system. It is a lightweight experimental interface
for investigating how text-promptable models respond to different
natural-language prompts on custom urban datasets, and for comparing their
output against classical detectors on box-level metrics.

You can:

- Run a single image + prompt and inspect boxes, masks, and timing.
- Compare up to 4 prompts on the same image (prompt-sensitivity analysis).
- Batch-process many images or a whole folder, with summary plots.
- Track per-prompt performance across a session.
- Export detections as CSV / JSON / COCO / YOLO.

For ground-truth-scored experiments, use the separate
[batch-evaluation workflow](../Scripts/Other-Scripts/PromptDetect/batch_evaluation/README.md).
The Gradio tabs described below are exploratory and do **not** score detections
against dataset ground truth. A five-image MDWD pilot (SAM 3 + Cosmos Reason2
2B) has verified the scoring pipeline; it is deliberately kept separate from
dissertation results until fixed, adequately sampled MDWD and MTSD evaluations
are run.

Fixed batch evaluations log online by default to the W&B project
`MSc-MDWD-MDWD-PromptDetect` in the same workspace as the existing detection
runs. Add `WANDB_API_KEY` and (if needed) `WANDB_ENTITY` to the repository-root
git-ignored `.env`; use `--wandb-mode offline` only for development runs that
must not upload to the workspace. Each completed evaluation uploads its full
result directory as a W&B artifact in addition to retaining the local atomic
outputs.

---

## Supported models

| UI option | Pipeline | Weights | Output |
|-----------|----------|---------|--------|
| **SAM 3** | Meta `sam3` native builder — one pass → detection + segmentation | `facebook/sam3` (`sam3.pt`) | boxes + masks |
| **SAM 3.1** | Same builder, updated checkpoint | `facebook/sam3.1` (`sam3.1_multiplex.pt`) | boxes + masks |
| **Cosmos Reason2 2B** | NVIDIA Cosmos Reason2 (Qwen3-VL) — JSON bbox grounding | `nvidia/Cosmos-Reason2-2B` | boxes only |
| **Cosmos Reason2 8B** | Same engine, larger checkpoint (~16 GB fp16) | `nvidia/Cosmos-Reason2-8B` | boxes only |
| **Cosmos Reason2 32B** | Same engine, largest checkpoint (~64 GB fp16; needs multi-GPU / CPU offload) | `nvidia/Cosmos-Reason2-32B` | boxes only |
| **LocateAnything 3B** | NVIDIA LocateAnything (Eagle) — dedicated grounding, special-token boxes | `nvidia/LocateAnything-3B` | boxes only |

All three Cosmos Reason2 sizes share the same Qwen3-VL architecture and the same
`CosmosReason2Engine`; only the checkpoint differs. The 2B and 8B fit a single
24 GB GPU (e.g. RTX 4090); the **32B does not** (~64 GB in fp16) and will fall
back to CPU/disk offload via `device_map="auto"` — it runs but is slow.

> **LocateAnything-3B runs in a transparent worker.** Its custom
> `trust_remote_code` modeling code is hard-bound to **transformers ~4.57.x** and
> can't share the transformers 5.x process the other models use. Selecting it
> automatically launches a worker in a dedicated `mtsd-la` env and proxies
> inference — no manual env switching. You just create that env once
> (see [Setup → LocateAnything-3B](#locateanything-3b-transparent-worker-one-time-env-setup)).

Select the model in the UI and click **Load Model** — no checkpoint paths to
manage. All weights are gated and download once from Hugging Face (see Setup).

### Why SAM 3 uses Meta's native package (not transformers)

SAM 3's `facebook/sam3` repo ships transformers-format weights, but **SAM 3.1's
`facebook/sam3.1` repo ships only the native `sam3.1_multiplex.pt` checkpoint —
there is no transformers `model.safetensors`** (confirmed on the model card).
So SAM 3.1 can *only* be loaded through Meta's `sam3` package. To keep SAM 3 and
SAM 3.1 directly comparable for the dissertation, **both** run through the same
native engine ([`Sam3NativeEngine`](../Scripts/Other-Scripts/PromptDetect/backend.py)), under bf16 autocast.

---

## How each flow works (end-to-end)

**SAM 3 / SAM 3.1** ([`Sam3NativeEngine`](../Scripts/Other-Scripts/PromptDetect/backend.py)):

```
text prompt + image
        │
        ▼
build_sam3_image_model(checkpoint=sam3[.1])   ← load detector + seg head
        │
        ▼
Sam3Processor.set_image(image)                ← preprocess (bf16 autocast)
        │
        ▼
Sam3Processor.set_text_prompt(prompt)         ← detect + segment in one pass
        │
        ▼
boxes (xyxy px) + scores + masks
        │
        ▼
filter_detections(...)                        ← confidence · area · count
```

**Cosmos Reason2** ([`CosmosReason2Engine`](../Scripts/Other-Scripts/PromptDetect/backend.py)):

```
text prompt + image
        │
        ▼
Qwen3VLProcessor.apply_chat_template(...)      ← "report bbox in JSON"
        │
        ▼
Qwen3VLForConditionalGeneration.generate
        │
        ▼
parse [{"bbox_2d":[x1,y1,x2,y2], "label":…}]   ← boxes only, score = 1.0
        │
        ▼
filter_detections(...)
```

**LocateAnything 3B** ([`LocateAnythingEngine`](../Scripts/Other-Scripts/PromptDetect/backend.py)):

```
text prompt + image
        │
        ▼
processor.py_apply_chat_template(...) + process_vision_info(...)
        │
        ▼
model.generate(generation_mode="hybrid", …)   ← Parallel Box Decoding
        │
        ▼
parse <box><x1><y1><x2><y2></box> tokens       ← 0–1000 grid, boxes only
        │
        ▼
filter_detections(...)
```

---

## Folder structure

```
Scripts/Other-Scripts/PromptDetect/          # source + the cleanup utility
├── app.py                — Gradio UI (5 tabs) + event wiring
├── backend.py            — model registry + inference engines (SAM 3/3.1, Cosmos, LocateAnything)
├── la_worker.py          — out-of-process LocateAnything worker (transformers 4.57 env)
├── utils.py              — visualisation, filtering, exports, logging, image I/O
├── clean_model_cache.ps1 — list / delete cached model weights (reclaim disk)
├── logs/                 — per-session run logs (auto-created, git-ignored)
└── exports/              — CSV / JSON / COCO / YOLO / plots (auto-created, git-ignored)

Requirements/             # pip dependency definitions (repo root)
├── requirements-promptdetect.txt     — main env (SAM 3/3.1 + Cosmos)
└── requirements-locate-anything.txt  — LocateAnything worker env

Requirements/CondaEnvironments/                     # reproducible env YAMLs + setup scripts
├── environment-mtsd-base.yml         — main env (pinned)
└── environment-mtsd-la.yml           — LocateAnything worker env (pinned)

Documents/                # documentation (repo root)
├── PromptDetect.md       — this file
├── ModelArchitectures.md — per-model architecture reference
└── CleanModelCache.md    — clean_model_cache.ps1 guide
```

The `sam3` Python package is installed into the environment (site-packages) —
it is **not** vendored in this folder. The app imports it like any dependency.

---

## Setup

The app assumes a **Conda environment named `mtsd-base`** (formerly `sam3`) (Python 3.12 tested).

> **Shortcut:** the whole setup below (env + PyTorch + dependencies + `sam3`
> package) is automated by the per-platform scripts in
> [Requirements/CondaEnvironments/](../Requirements/CondaEnvironments/README.md), e.g.
> `.\setup_conda_env.ps1 -Name mtsd-base`. The manual steps follow.

```bash
conda create -n mtsd-base python=3.12 -y
conda activate mtsd-base
```

### 1. Install PyTorch

Pick the build for your hardware from <https://pytorch.org/get-started/locally/>:

```bash
# Example — CUDA 12.8 (tested: torch 2.10.0 + torchvision 0.25.0)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 2. Install the app dependencies

From the repository root:

```bash
pip install -r Requirements/requirements-promptdetect.txt
```

### 3. Install the Meta `sam3` package (required for SAM 3 and SAM 3.1)

```bash
pip install git+https://github.com/facebookresearch/sam3.git
```

**Windows note (important).** The `sam3` package imports `triton`, which has no
official Windows wheels. Install the community port and keep setuptools below 81
(the package still uses the removed `pkg_resources`):

```bash
pip install triton-windows
pip install "setuptools<81"
```

These are already configured in the `mtsd-base` Conda env on this machine.

### 4. Get access to the gated weights

Every model repo is gated on Hugging Face. Request access on each one you intend
to use, then authenticate so the weights can download:

- SAM 3 — <https://huggingface.co/facebook/sam3>
- SAM 3.1 — <https://huggingface.co/facebook/sam3.1>
- Cosmos Reason2 2B — <https://huggingface.co/nvidia/Cosmos-Reason2-2B>
- Cosmos Reason2 8B — <https://huggingface.co/nvidia/Cosmos-Reason2-8B>
- Cosmos Reason2 32B — <https://huggingface.co/nvidia/Cosmos-Reason2-32B>
- LocateAnything 3B — <https://huggingface.co/nvidia/LocateAnything-3B>

```bash
hf auth login
```

Weights download automatically on first **Load Model** and are cached under
`~/.cache/huggingface/hub`. Approximate footprints: SAM 3 / SAM 3.1 ~3.5 GB each;
Cosmos Reason2 ~5 GB (2B) / ~16 GB (8B) / ~64 GB (32B); LocateAnything ~6 GB.

### LocateAnything-3B transparent worker (one-time env setup)

LocateAnything's `trust_remote_code` modeling code is hard-bound to
**transformers ~4.57.x** and can't share the transformers 5.x process the other
models use. PromptDetect handles this **automatically**: selecting
"LocateAnything 3B" launches a small worker ([`la_worker.py`](../Scripts/Other-Scripts/PromptDetect/la_worker.py)) in a
dedicated `mtsd-la` conda env and proxies inference to it over localhost
— you never switch environments by hand.

Create that env **once** — either via the setup scripts
(`.\setup_conda_env.ps1 -Name mtsd-la` from
[Requirements/CondaEnvironments/](../Requirements/CondaEnvironments/README.md)) or manually:

```bash
conda create -n mtsd-la python=3.12 -y
conda activate mtsd-la
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r Requirements/requirements-locate-anything.txt   # transformers==4.57.1 + decord/lmdb/peft
hf auth login                                                   # access to nvidia/LocateAnything-3B
conda deactivate
```

After that, just run the app in the main `mtsd-base` env and pick "LocateAnything 3B".
The worker starts on first load and is shut down (freeing its GPU memory) when you
switch to another model; its log is at `logs/la_worker.log`.

**How it works.** `LocateAnythingEngine` auto-detects the host transformers
version: on 4.57 it loads **in-process**; on 5.x it spawns the `mtsd-la`
**worker** and forwards `predict()` calls. The same engine runs the model in both
cases, so behaviour is identical. Large images are downscaled (longest side
≤ 1536 px) before MoonViT — its dense attention would otherwise exceed a 24 GB
GPU; boxes are on a 0–1000 grid, so the downscale does not affect coordinates.

---

## Running

```bash
conda activate mtsd-base
cd Scripts/Other-Scripts/PromptDetect
python app.py
```

Opens at **http://localhost:7860**.

In the UI: pick a **Model** → **Load Model** → provide an image and prompt →
**Run Inference**.

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `7860` | Server port |
| `--no-browser` | off | Do not auto-open the browser |
| `--share` | off | Create a public Gradio share link |

---

## Interface

| Tab | Purpose |
|-----|---------|
| **Single Image** | Primary tab — one image + one prompt. Visual overlay, detection table, timing, and one-click export (CSV / JSON / COCO / YOLO). |
| **Prompt Comparison** | Up to 4 prompts on the same image, side-by-side, for prompt-sensitivity analysis. |
| **Batch** | Many images, one prompt → `results.csv` in `exports/`. |
| **Dataset** | Scan a local folder; summary statistics + distribution plots (detections, box area, confidence). |
| **Analytics** | Session-level per-prompt aggregates (runs, total detections, mean confidence). |

**Boxes are always shown.** Segmentation masks are hidden by default (toggle
*Show segmentation masks*) and apply to SAM 3 / 3.1 only — Cosmos Reason2 and
LocateAnything return boxes only.

### Example prompts

- **MDWD (waste):** `waste`, `garbage bag`, `black garbage bag`,
  `orange CMD bag`, `recycling bag`, `organic waste`, `litter`
- **MTSD (signs):** `traffic sign`, `stop sign`, `speed limit sign`,
  `warning sign`, `damaged traffic sign`, `circular traffic sign`

---

## Exports & logs

Exports are written to `exports/` with a timestamped filename:

| Format | Description |
|--------|-------------|
| CSV | Flat table of box coordinates and scores |
| JSON | List of detection records |
| COCO JSON | Standard COCO instance-detection format |
| YOLO | Normalised `class cx cy w h` per line |

Each session writes a log to `logs/session_YYYYMMDD_HHMMSS.log`.

---

## Model storage & cleanup

Model weights are **not** stored in this folder — they download into the
**Hugging Face hub cache**
(`%USERPROFILE%\.cache\huggingface\hub\models--<org>--<name>\`, or
`$env:HF_HUB_CACHE` / `$env:HF_HOME\hub` if set). They are large: Cosmos 32B alone
is ~60 GB, and all PromptDetect models together approach ~100 GB.

To inspect or reclaim space, use the bundled
[`clean_model_cache.ps1`](../Scripts/Other-Scripts/PromptDetect/clean_model_cache.ps1) helper
(it touches only the PromptDetect models). Full usage — prerequisites, commands,
expected output, and troubleshooting — is in
**[Documents/Other/CleanModelCache.md](Other/CleanModelCache.md)**.

---

## Known limitations

- **All weights are gated** and require access approval + `hf auth login`.
  Without it, **Load Model** fails with a Hugging Face permissions error.
- **SAM 3.1 image weights are extracted from the multiplex (video) checkpoint.**
  Loading reports 4 missing FPN-conv keys (`backbone.vision_backbone.convs.3.*`);
  this is expected — Meta's loader uses `strict=False` and image inference is
  unaffected.
- **Cosmos Reason2 returns boxes only** (no masks) and **no real per-box
  confidence** — it's a generative VLM, not a detector. The backend assigns 1.0
  internally (for filtering), but the UI **hides** the confidence everywhere for
  such models (`has_confidence=False` on the model spec): the overlay shows just
  the label, the table omits the Confidence column, and the summary reads
  "Confidence: n/a" — a constant 1.0 would be misleading, not informative.
  Coordinates use the Qwen3-VL **0–1000 normalised grid** (verified by scale-
  invariance testing — *not* absolute pixels); the backend maps them back to
  image pixels in `_parse_json_boxes` and clamps to bounds.
- **LocateAnything-3B runs via a transparent worker** in a `mtsd-la`
  (`transformers==4.57.x`) env that you create once (see Setup); the main app
  spawns it automatically. It is **boxes only**, **no per-box confidence**
  (hidden in the UI), and reports on a 0–1000 grid (special tokens
  `<box>…</box>`, parsed by `_parse_box_tokens`). Two practical notes: inputs are
  **downscaled to ≤ 1536 px** (MoonViT's dense attention would otherwise exceed
  24 GB — boxes are normalised so coordinates are unaffected), and it is
  **slower** than the others (worker IPC + SDPA + autoregressive box decoding;
  the first inference also pays a warm-up cost).
- **Windows requires `triton-windows` + `setuptools<81`** for the `sam3`
  package; Triton kernels are only exercised by SAM 3's video path (unused here),
  so image inference runs under bf16 autocast on CUDA/CPU.
- Single-user local tool: model state and analytics are in-memory and reset on
  restart.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Load Model` fails with 401/403 / "gated repo" | `hf auth login` and request access to the model's HF repo. |
| `No module named 'triton'` (Windows) | `pip install triton-windows`. |
| `No module named 'pkg_resources'` | `pip install "setuptools<81"`. |
| `sam3.1 does not appear to have model.safetensors` | Expected — SAM 3.1 has no transformers weights. Ensure the `sam3` package is installed; it loads the native checkpoint. |
| `mat1 and mat2 must have the same dtype` (SAM 3/3.1) | The native model needs bf16 autocast on CUDA; the backend wraps this automatically. |
| Cosmos boxes look offset/scaled | Check the Qwen3-VL coordinate convention for your transformers version; adjust `_parse_json_boxes` / `_normalise_box` in `backend.py`. |
| LocateAnything "failed to build under transformers …" | Expected on the main env — use the `mtsd-la` env (`transformers==4.57.1`); see Setup. |
| LocateAnything `No module named 'decord' / 'lmdb' / 'peft'` | `pip install decord lmdb peft` in the LocateAnything env. |
| CUDA not used | Confirm a CUDA PyTorch build: `python -c "import torch; print(torch.cuda.is_available())"`. |

---

## Dissertation context

This tool supports the research objective of evaluating **prompt sensitivity**
in promptable detection/segmentation models on domain-specific urban-monitoring
datasets. Results feed into comparisons across SAM 3, SAM 3.1, Cosmos Reason2,
and LocateAnything under various prompt formulations.
