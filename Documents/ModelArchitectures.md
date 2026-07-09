# Model Architectures: SAM 3, SAM 3.1, Cosmos Reason2, and LocateAnything

Technical reference for the promptable perception models evaluated in this
dissertation. Covers architecture, loading strategy, end-to-end inference flow,
and the practical constraints that shaped each implementation.

---

## 1. SAM 3 (Segment Anything Model 3)

**Source:** Meta AI Research  
**HF repo:** `facebook/sam3`  
**Weights file:** `sam3.pt` (~3.5 GB)  
**Loaded via:** Meta's native `sam3` Python package (`build_sam3_image_model`)

### 1.1 What is it?

SAM 3 is Meta's third-generation promptable segmentation model. It is a
**text-grounded, one-pass detection + segmentation** system — unlike SAM 1/2
which required explicit point, box, or mask prompts, SAM 3 accepts a
**natural-language string** and returns both bounding boxes and segmentation
masks in a single forward pass.

### 1.2 Architecture

SAM 3 is built on a **hierarchical vision transformer** backbone that produces
multi-scale feature maps via a feature pyramid network (FPN). On top of this:

| Component | Role |
|-----------|------|
| **Vision backbone** (ViT-H / Hiera-L) | Encodes the input image at multiple scales using hierarchical attention |
| **FPN neck** | Fuses pyramid feature maps from backbone stages into a single set of detection features |
| **Text encoder** | Encodes the natural-language prompt into a sequence of embeddings (CLIP-like) |
| **Detection head** | Cross-attends text and image features → predicts boxes + class scores |
| **Mask decoder** | Takes the detection tokens and up-samples to per-object binary masks |

The key advance over SAM 2 is that **text grounding and mask generation share
the same forward pass**: the detector head outputs both boxes and the mask tokens
used by the decoder, so there is no separate grounding model (no BERT + GroundingDINO
stage).

### 1.3 Inference flow (this app)

```
Input: RGB image (any resolution) + text string (e.g. "garbage bag")
                            │
                            ▼
        Sam3Processor.set_image(pil_image)
            • Resize + normalise to model's input resolution
            • Run the vision backbone (bf16 on CUDA)
            • Cache image features (state object)
                            │
                            ▼
        Sam3Processor.set_text_prompt(state, prompt)
            • Tokenise + encode the text string
            • Cross-attend text ↔ image features
            • Detection head → raw boxes (xyxy, normalised) + scores
            • Mask decoder → per-box binary mask (model resolution)
                            │
                            ▼
        Post-processing (sam3_backend._normalise_box / _resize_mask)
            • Boxes: normalised [0,1] → absolute pixels; cxcywh → xyxy if needed
            • Masks: bilinear/nearest upsample → original image size
                            │
                            ▼
        filter_detections(...)
            • Drop detections below conf_threshold
            • Drop boxes smaller than min_area or larger than max_area
            • Keep at most max_detections
                            │
                            ▼
Output: boxes (px xyxy), scores [0,1], labels (= prompt text), masks (bool HxW)
```

### 1.4 Loading details

```python
# How backend.py loads SAM 3
from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
from sam3.model.sam3_image_processor import Sam3Processor

checkpoint = download_ckpt_from_hf(version="sam3")   # downloads sam3.pt
model = build_sam3_image_model(
    device=device, eval_mode=True,
    checkpoint_path=checkpoint,
    load_from_HF=False,
    enable_segmentation=True,
)
processor = Sam3Processor(model, device=device, confidence_threshold=0.05)
```

The `confidence_threshold=0.05` is deliberately low; the app re-filters at the
user-controlled threshold **after** inference so the same forward pass can serve
any threshold without rerunning the model.

### 1.5 Why bf16 autocast is required

SAM 3's vision backbone linear layers are initialised and stored in **bfloat16**.
Without autocast, activations entering those layers are float32, which causes a
dtype mismatch at the matrix-multiply level:

```
RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float
```

The fix is to wrap `set_image` and `set_text_prompt` in `torch.autocast`:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    state = processor.set_image(pil)
    out   = processor.set_text_prompt(state=state, prompt=prompt)
```

On CPU, autocast is skipped (CPU bf16 support is limited and the model is slower
anyway).

---

## 2. SAM 3.1

**Source:** Meta AI Research  
**HF repo:** `facebook/sam3.1`  
**Weights file:** `sam3.1_multiplex.pt` (~3.5 GB)  
**Loaded via:** Same Meta native `sam3` package, `version="sam3.1"`

### 2.1 What is it?

SAM 3.1 is an incremental update to SAM 3. The **architecture is identical**
(same backbone, FPN, text encoder, detection head, mask decoder). Only the
**checkpoint weights are updated** — trained on more data or for more steps,
yielding slightly different detection characteristics (empirically observed to
run faster in image mode: ~300 ms vs ~670 ms on a 4096×3072 image in this
setup).

### 2.2 The "multiplex" checkpoint

The SAM 3.1 checkpoint is called `sam3.1_multiplex.pt` because it is a
**multi-task (video + image) checkpoint** — the same file serves both SAM 3.1's
image detector and its video tracking pipeline. The `build_sam3_image_model`
builder extracts the image-relevant subgraph with `strict=False`, which means
four keys that exist only in the video path are reported as missing:

```
backbone.vision_backbone.convs.3.weight
backbone.vision_backbone.convs.3.bias
...
```

This is **expected and benign** — those convolutions belong to the video-only
temporal attention path and are not exercised during image inference.

### 2.3 Why there is no Hugging Face Transformers integration

`facebook/sam3.1`'s model card states explicitly:

> *"There is no Hugging Face Transformers integration for SAM 3.1."*

The repo ships only `sam3.1_multiplex.pt` — no `model.safetensors`, no
`config.json` in transformers format. Calling
`Sam3Model.from_pretrained("facebook/sam3.1")` raises:

```
OSError: facebook/sam3.1 does not appear to have a file named
pytorch_model.bin or model.safetensors
```

The only correct loading path is via `download_ckpt_from_hf(version="sam3.1")`
from the native `sam3` package, which fetches the `.pt` file directly.

### 2.4 Inference flow

Identical to SAM 3 — only the checkpoint path changes:

```python
checkpoint = download_ckpt_from_hf(version="sam3.1")   # downloads sam3.1_multiplex.pt
model = build_sam3_image_model(...)                      # same builder, same API
```

### 2.5 Why both SAM 3 and SAM 3.1 run through the same engine

Since SAM 3.1 *must* use the native package, and `Sam3Processor` provides the
same API for both checkpoints, **SAM 3 also uses the native engine** in this
app. This makes the two directly comparable:

- Same preprocessing pipeline
- Same bf16 autocast
- Same post-processing
- Only the learned weights differ

If SAM 3 were loaded via `transformers.Sam3Model.from_pretrained` and SAM 3.1
via the native builder, any observed difference could be an artefact of the two
different loading paths rather than a true model difference.

---

## 3. Cosmos Reason2

- **Source:** NVIDIA
- **HF repos:** `nvidia/Cosmos-Reason2-2B` · `-8B` · `-32B`
- **Weights:** ~5 GB (2B) / ~16 GB (8B) / ~64 GB (32B), float16 sharded safetensors
- **Loaded via:** `transformers.Qwen3VLForConditionalGeneration` + `AutoProcessor`

> **Size variants.** All three Cosmos Reason2 sizes share the *same* Qwen3-VL
> architecture, the same loading API, and the same `CosmosReason2Engine` in this
> app — only the checkpoint differs. 2B and 8B fit a single 24 GB GPU; the 32B
> (~64 GB fp16) does not and falls back to CPU/disk offload via
> `device_map="auto"` (functional but slow). The sections below describe the
> shared pipeline; swap the repo id to change size.
>
> **Not Cosmos3-Nano.** `nvidia/Cosmos3-Nano` is a *different* model —
> `Cosmos3ForConditionalGeneration` / `cosmos3_omni`, a 16B omnimodal
> text→image/video *generation* model loaded with `diffusers.DiffusionPipeline`
> (it ships a VAE, scheduler, and sound tokenizer). It has no transformers
> grounding class and no documented image+text→bbox detection API, so it is
> **not** wired into this detection tool.

### 3.1 What is it?

Cosmos Reason2 is NVIDIA's physics-informed world model, here used only for its
**open-vocabulary bounding box detection** capability. It is built on the
**Qwen3-VL** vision-language model architecture — a large language model (LLM)
augmented with a vision encoder that processes image patches as visual tokens.

Unlike SAM 3/3.1, Cosmos Reason2 does **not** do segmentation. It produces:
- **Bounding boxes** (xyxy on a **0–1000 normalised grid** — see §3.8)
- **Object labels** (from its own reasoning)
- **No per-box confidence** (every box is scored 1.0)

### 3.2 Architecture

```
Input: image + text prompt
    │
    ├── Vision encoder (ViT-based)
    │       • Patchify image into visual tokens (e.g. 14×14 patches → 196 tokens)
    │       • Project visual tokens to LLM embedding dimension
    │
    └── LLM decoder (Qwen3 transformer)
            • Interleaves visual tokens with text tokens
            • Autoregressively generates a text reply
            • Reply contains JSON-formatted bounding boxes
```

The grounding output format is an agreed-upon JSON schema rather than a
dedicated detection head. The model "knows" to report coordinates in absolute
pixels because the prompt instructs it to, and it was trained on grounding
data in this format.

### 3.3 Inference flow (this app)

```
Input: RGB image + text string (e.g. "traffic sign")
                        │
                        ▼
    Build conversation message:
        [{"role":"user","content":[
            {"type":"image","image": pil},
            {"type":"text","text":
              'Locate every object that matches "traffic sign". '
              'Report bbox in JSON as [{"bbox_2d":[x1,y1,x2,y2],"label":"..."}].'}
        ]}]
                        │
                        ▼
    Qwen3VLProcessor.apply_chat_template(messages, tokenize=True, ...)
        • Render the chat template (system/user/assistant turns)
        • Insert image as visual tokens
        • Tokenise text
        • Return input_ids + pixel_values tensors
                        │
                        ▼
    Qwen3VLForConditionalGeneration.generate(max_new_tokens=1024, do_sample=False)
        • Greedy decoding (deterministic)
        • Model writes its reasoning, then outputs JSON
                        │
                        ▼
    Decode generated tokens → reply string (skip prompt tokens)
                        │
                        ▼
    _parse_json_boxes(reply, img_w, img_h)
        • Scan for {"bbox_2d":[...], "label":"..."} objects via regex
        • Fall back to parsing a JSON array if no individual objects found
        • Map 0–1000 normalised grid → image pixels (see §3.8)
        • Clamp all boxes to image bounds
                        │
                        ▼
Output: boxes (px xyxy), scores (all 1.0), labels (model-generated), masks=[]
```

### 3.4 The grounding prompt

```python
_PROMPT = (
    'Locate every object that matches the description "{q}" in the image. '
    "Report bbox coordinates in JSON format as a list of "
    '{{"bbox_2d": [x1, y1, x2, y2], "label": "<name>"}}.'
)
```

The double braces `{{` / `}}` are Python f-string escapes for literal `{` / `}`.
The final string fed to the model reads, for prompt `"garbage bag"`:

> Locate every object that matches the description "garbage bag" in the image.
> Report bbox coordinates in JSON format as a list of
> {"bbox_2d": [x1, y1, x2, y2], "label": "<name>"}.

### 3.5 JSON box parser

Cosmos Reason2 is a thinking/reasoning model — it may produce several paragraphs
of reasoning text before the JSON, or wrap JSON in markdown fences. The parser
`_parse_json_boxes` handles this robustly:

1. **Primary pass:** scan for `{"bbox_2d":[...], "label":"..."}` objects using
   a regex that ignores surrounding text and markdown.
2. **Fallback:** if no individual objects found, extract the first `[...]` array
   and parse it as JSON.
3. For each object: read `bbox_2d` (or `bbox`) → check if normalised or absolute
   → convert → clamp to image.

```python
for match in re.finditer(r'\{[^{}]*"bbox_2d"[^{}]*\}', text, re.DOTALL):
    objs.append(json.loads(match.group(0)))
```

### 3.6 Loading details

```python
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "nvidia/Cosmos-Reason2-2B",
    dtype=torch.float16,        # fp16 for GPU memory efficiency
    device_map="auto",          # spread across available GPUs automatically
    attn_implementation="sdpa", # scaled dot-product attention (PyTorch built-in, faster)
).eval()
processor = AutoProcessor.from_pretrained("nvidia/Cosmos-Reason2-2B")
```

`device_map="auto"` uses Hugging Face Accelerate to shard the model across all
available GPUs, falling back to CPU if none. `attn_implementation="sdpa"` uses
PyTorch's fused SDPA kernel instead of the naive implementation.

### 3.7 Access

`nvidia/Cosmos-Reason2-2B` is a gated repo — NVIDIA must approve your HF
account before the weights download. Request access at
<https://huggingface.co/nvidia/Cosmos-Reason2-2B>, then authenticate:

```bash
huggingface-cli login
```

### 3.8 Coordinate convention (0–1000 normalised grid)

**Important:** Qwen3-VL — and therefore Cosmos Reason2 — reports bounding-box
coordinates on a fixed **0–1000 normalised grid**, *not* in absolute pixels.
A box `[353, 897, 384, 937]` means 35.3%–38.4% across and 89.7%–93.7% down the
image, regardless of its actual pixel dimensions.

This was confirmed empirically by a **scale-invariance test** — feeding the same
image at full and half resolution returns identical coordinates:

```
FULL  input 3009x3120  ->  [[353, 897, 384, 937]]
HALF  input 1504x1560  ->  [[354, 893, 385, 937]]   # essentially unchanged
```

If the values were absolute pixels they would have halved. They didn't, so the
grid is normalised.

**Mapping back to pixels** (`_parse_json_boxes`): each coordinate is divided by
1000 to get a [0, 1] fraction, then multiplied by the image width/height:

```python
vals = [float(v) for v in bb[:4]]
if max(vals) > 1.0:                  # 0..1000 grid -> [0, 1]
    vals = [v / 1000.0 for v in vals]
box = _normalise_box(vals, img_w, img_h)   # [0,1] -> pixels, clamp to bounds
```

> **Historical bug (fixed):** an earlier version treated the raw 0–1000 values
> as absolute pixels. On a 3000-px image this squashed every detection into the
> top-left ~1000×1000 corner and shrank it — boxes appeared "floating in the
> sky." The `/1000` mapping above corrects this. The `max(vals) > 1.0` guard
> means an occasional already-normalised (0–1) reply is left untouched.

---

## 4. LocateAnything-3B

**Source:** NVIDIA (Eagle VLM family)
**HF repo:** `nvidia/LocateAnything-3B`
**Weights:** ~6 GB (bf16, sharded safetensors)
**Loaded via:** `transformers.AutoModel.from_pretrained(..., trust_remote_code=True)`

### 4.1 What is it?

LocateAnything-3B is a **dedicated visual-grounding model** (not a general VLM
repurposed for detection). Vision encoder **MoonViT** + language model
**Qwen2.5-3B-Instruct**, trained specifically for detection, phrase grounding,
GUI grounding, text localisation, and pointing. Its core innovation is **Parallel
Box Decoding (PBD)** — predicting whole boxes in one parallel step instead of
token-by-token, for higher throughput.

It returns **boxes only** (no masks) and **no per-box confidence** (score = 1.0).

### 4.2 Loading — custom remote code

Unlike the Qwen3-VL VLMs, LocateAnything ships its own `trust_remote_code`
modeling files (`modeling_locateanything.py`, a vendored `modeling_qwen2.py`,
`modeling_vit.py`) and a **non-standard `generate()`** that takes `pixel_values`,
`tokenizer`, `generation_mode` and `image_grid_hws` directly.

```python
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
model     = AutoModel.from_pretrained(repo, dtype=torch.bfloat16, trust_remote_code=True)
```

It also imports `decord`, `lmdb`, and `peft` at load time (its remote modules
check for them), so those must be installed.

### 4.3 transformers version constraint → transparent worker

The vendored code is hard-bound to **transformers ~4.57**. Under transformers 5.x
it breaks in three independent ways, confirmed empirically:

1. `_check_and_adjust_attn_implementation()` — 5.x passes a new `allow_all_kernels`
   kwarg the overrides don't accept (on the LocateAnything *and* vendored Qwen2/MoonViT classes).
2. **RoPE config drift** — 5.x moved Qwen2's top-level `rope_theta` into the
   `rope_parameters`/`rope_scaling` dicts; the vendored code reads the old name.
3. **Tied-weights format** — 5.x expects `_tied_weights_keys` as a dict; the
   vendored code provides a list (`'list' object has no attribute 'keys'`).

The backend shims (1) and (2), but (3)+ are unbounded and patching them risks
silently mis-loading weights — unacceptable for evaluation. So LocateAnything
needs **transformers 4.57.x**, supplied by a one-time `mtsd-la` conda env.

To keep this **transparent**, `LocateAnythingEngine` auto-detects the host
transformers version and runs in one of two modes:

```
DetectionBackend.load("LocateAnything 3B")        (main env: transformers 5.x)
        │
        ▼
LocateAnythingEngine.load()
        │   _la_should_run_inprocess()?  → transformers 4.57 ?  → in-process
        │                                  else (5.x)          → worker
        ▼  (worker)
spawn  <conda>/envs/mtsd-la/python  la_worker.py --port <free>
        │   (sets PROMPTDETECT_LA_WORKER=1 → engine loads in-process *there*)
        ▼
poll  http://127.0.0.1:<port>/health  until {"ready": true}
        ▼
predict_raw → POST /predict {image_path, prompt}  → boxes/scores/labels (JSON)
        ▼
close() → POST /shutdown + terminate process      (frees the worker's GPU memory)
```

The same `LocateAnythingEngine` runs the model on both sides (the worker sets
`PROMPTDETECT_LA_WORKER=1` to force its in-process path), so behaviour is
identical. Images pass to the worker by temp-file **path** (same machine — avoids
base64 of large frames). `DetectionBackend.load()` calls `engine.close()` on the
previous model before loading a new one, so switching away from LocateAnything
shuts the worker down and frees VRAM. Helpers: `_find_env_python`, `_free_port`,
`_la_should_run_inprocess` in `backend.py`; the worker is `la_worker.py`.

### 4.4 Inference flow & output

```
text prompt + image (original W×H kept for box mapping)
        │
        ▼
downscale longest side ≤ 1536 px      ← MoonViT dense SDPA would otherwise OOM
        │                                a 24 GB GPU (~40 GB for one attention op
        │                                at 3 K px); the 0–1000 grid makes this safe
        ▼
processor.py_apply_chat_template + process_vision_info
        │
        ▼
model.generate(generation_mode="hybrid", max_new_tokens=2048, do_sample=False)
        │  (Parallel Box Decoding; MoonViT/Qwen2 fall back to SDPA — no flash/magi)
        ▼
reply text with special tokens:  <box><x1><y1><x2><y2></box>
        │
        ▼
_parse_box_tokens(reply, W, H)   ← ints on a 0–1000 grid → /1000 → original px
        │
        ▼
filter_detections(...)            ← score = 1.0, boxes only
```

The detection prompt template is
`"Locate all the instances that match the following description: {prompt}."`
Coordinates are integers on a **0–1000 normalised grid** (same convention as
Qwen-VL), mapped to pixels via `_parse_box_tokens` → `_normalise_box`. Because
the grid is normalised, the ≤ 1536 px downscale changes nothing about the output
coordinates — it only keeps MoonViT's attention within GPU memory. Inference is
**slower** than the other models (worker IPC + SDPA + autoregressive box
decoding, plus a first-call warm-up).

---

## 5. Side-by-side comparison

| Dimension | SAM 3 | SAM 3.1 | Cosmos Reason2 | LocateAnything 3B |
|-----------|-------|---------|----------------|-------------------|
| **Architecture** | Hiera ViT + FPN + detection/mask heads | Same as SAM 3 | Qwen3-VL (ViT + LLM) | MoonViT + Qwen2.5-3B + PBD |
| **Prompt type** | Short text string | Short text string | Free-form text | Short text / categories |
| **Output** | Boxes + binary masks | Boxes + binary masks | Boxes only | Boxes only (also points) |
| **Confidence scores** | Yes, [0, 1] per box | Yes, [0, 1] per box | No (fixed at 1.0) | No (fixed at 1.0) |
| **Coord convention** | px / normalised [0,1] | same | 0–1000 grid (JSON) | 0–1000 grid (`<box>` tokens) |
| **Weights** | `sam3.pt` (3.5 GB) | `sam3.1_multiplex.pt` (3.5 GB) | ~5 / 16 / 64 GB (2B/8B/32B) | ~6 GB |
| **Loading API** | Meta native `sam3` package | Same | `transformers` ≥4.57 | `AutoModel` trust_remote_code |
| **transformers** | 5.x OK | 5.x OK | ≥4.57 (5.x OK) | **~4.57.x** (auto worker in `mtsd-la`) |
| **Gated** | Yes (`facebook/sam3`) | Yes (`facebook/sam3.1`) | Yes (`nvidia/Cosmos-Reason2-*`) | Yes (`nvidia/LocateAnything-3B`) |

---

## 6. Shared post-processing

All three models pass through the same filtering stage in `utils.filter_detections`:

```
for each detection:
    if score < conf_threshold  → discard
    area = (x2-x1) * (y2-y1)
    if area < min_area or area > max_area → discard
    if total kept >= max_detections → stop
```

Boxes from all models are normalised to the same format by `_normalise_box`:

- If all coordinates are ≤ 1.0: interpreted as normalised → multiplied by (W, H)
- If x2 < x1 (CxCyWH format): converted to x1y1x2y2
- All coordinates clamped to image bounds

Cosmos Reason2 (`_parse_json_boxes`) and LocateAnything (`_parse_box_tokens`)
first divide their 0–1000 grid coordinates by 1000 before this step.

---

## 7. Windows-specific installation notes

### `sam3` package requirements

| Package | Why |
|---------|-----|
| `triton-windows` | The `sam3` package imports `triton` at module level (via `sam3.model.edt`). Official Triton has no Windows wheels. The community port `triton-windows` satisfies the import without running GPU kernels (the Triton kernels are only used in the video path, not image inference). |
| `setuptools<81` | setuptools 82 removed `pkg_resources`. The `sam3` package's `model_builder.py` still calls `pkg_resources.resource_filename(...)` at import time. |

Install order for a fresh environment (automated by
`Requirements/CondaEnvironments/setup_conda_env.ps1 -Name mtsd-base`):

```bash
conda create -n mtsd-base python=3.12 -y
conda activate mtsd-base
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r Requirements/requirements-promptdetect.txt
pip install triton-windows
pip install "setuptools<81"
pip install git+https://github.com/facebookresearch/sam3.git
```

---

## 8. Code references

| Concept | Location |
|---------|----------|
| Model registry | `Scripts/Other-Scripts/PromptDetect/backend.py` — `MODELS` dict |
| SAM 3 / 3.1 engine | `Scripts/Other-Scripts/PromptDetect/backend.py` — `Sam3NativeEngine` |
| Cosmos engine | `Scripts/Other-Scripts/PromptDetect/backend.py` — `CosmosReason2Engine` |
| LocateAnything engine | `Scripts/Other-Scripts/PromptDetect/backend.py` — `LocateAnythingEngine` |
| LocateAnything worker (4.57 env) | `Scripts/Other-Scripts/PromptDetect/la_worker.py` |
| Worker env helpers | `Scripts/Other-Scripts/PromptDetect/backend.py` — `_find_env_python`, `_free_port`, `_la_should_run_inprocess` |
| JSON box parser (Cosmos) | `Scripts/Other-Scripts/PromptDetect/backend.py` — `_parse_json_boxes` |
| Box-token parser (LocateAnything) | `Scripts/Other-Scripts/PromptDetect/backend.py` — `_parse_box_tokens` |
| transformers 4.57 shims (LocateAnything) | `Scripts/Other-Scripts/PromptDetect/backend.py` — `_shim_la_attn_impl`, `_patch_la_rope_config` |
| Box normalisation | `Scripts/Other-Scripts/PromptDetect/backend.py` — `_normalise_box` |
| Detection filtering | `Scripts/Other-Scripts/PromptDetect/utils.py` — `filter_detections` |
| Visualisation | `Scripts/Other-Scripts/PromptDetect/utils.py` — `draw_detections` |
| Gradio UI | `Scripts/Other-Scripts/PromptDetect/app.py` |
| LocateAnything env | `Requirements/requirements-locate-anything.txt` + `Requirements/CondaEnvironments/environment-mtsd-la.yml` |
