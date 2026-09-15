# Inference Speed / Deployment Benchmark — Status Report

> **Superseded status:** The benchmark has now been executed. The current
> consolidated results are in
> `Documents/Final-Tables/20260915-inference-benchmark/`; this document retains
> the original infrastructure plan for historical context.

*Generated 2026-07-06. Companion to `Scripts/Other-Scripts/Inference-Benchmark/inference_speed_benchmark.py`.
This report describes the benchmark **infrastructure and coverage**; it contains no
timing numbers yet, because no full benchmark run has been executed — dry-run
verification only, per the repository's safety policy.*

## 1. Why this benchmark exists

The dissertation compares computer-vision approaches for urban monitoring in Malta
not only on accuracy but on **deployment practicality**. This benchmark produces the
deployment half of that comparison: per-image latency (mean / median / p95),
throughput (FPS), cold-start load time, peak GPU memory, model file size and
parameter count — measured under one shared, reproducible protocol across all three
model families (supervised detectors, attribute classifiers, prompt-based
foundation models).

**Prior state:** no inference-speed benchmark existed in the repository. The only
timing signals were incidental (Ultralytics `speed_inference_ms` inside validation
results of the MDWD training runs, per-call `inference_ms` in the PromptDetect
batch evaluator). Neither is a controlled deployment measurement — no warmup
separation, no CUDA synchronisation discipline, no fixed sample, no memory or
model-size reporting. The dedicated benchmark was created on 2026-07-06.

## 2. Measurement protocol

- Fixed-seed sample of images per dataset/task (default seed 42); the exact file
  list is stored with every run (`benchmark_images_used.csv`), so any rerun or
  cross-model comparison uses identical inputs.
- Explicit warmup iterations (default 3), excluded from all statistics.
- `torch.cuda.synchronize()` before and after every timed call on CUDA.
- Model loading is timed separately (**cold start**) and never mixed into
  per-image latency.
- Batch size 1 by default (the deployment-relevant case for street-level
  monitoring); `--batch-sizes 1 4 8` opts into batched throughput where the API
  supports it (YOLO, attribute classifiers; RF-DETR and the prompt models expose
  single-image APIs and are pinned to batch 1 with a note).
- Where the underlying framework reports a preprocess/inference/postprocess
  breakdown (Ultralytics, PromptDetect backend), it is recorded alongside the
  wall-clock measurement; the attribute pipeline times its transform separately.
- Every run writes to a fresh timestamped folder under
  `Results/Inference-Benchmark/InferenceSpeed/` — previous results are never touched.

Hardware at the time of writing: single **NVIDIA GeForce RTX 4090**, CUDA available
in all three conda environments (`MDWD`: torch 2.10-dev/cu130; `mtsd-attrcls`:
torch 2.11/cu128; `mtsd-base`: torch 2.10/cu128). Hardware details are captured in
each run's `benchmark_config.json`, so results from other machines (e.g. the DGX)
remain distinguishable.

## 3. Model coverage

### Benchmark-ready now (checkpoints on disk, dependencies present)

| Group | Models | Checkpoints |
| --- | --- | --- |
| MDWD detection (env `MDWD`) | YOLO11 n/s/m/l · YOLO12 n/s/m · YOLO26 n/s/m/l (EUVIP) · YOLO26 n/s/m/l (`@dgx` duplicates trained on the DGX) · RF-DETR **nano** | `Results/MDWD-Runs/<suite>/E*/weights/best.pt`; RF-DETR `checkpoint_best_ema.pth` |
| MTSD attribute classification (env `mtsd-attrcls`) | dinov3, dinov3_lora, vjepa, vjepa_lora, convnext_frozen, convnext (fine-tuned) | `AttributeClassification/outputs/checkpoints/<variant>/best.pt` |
| PromptDetect (env `mtsd-base`) | SAM 3, SAM 3.1, Cosmos Reason2 2B/8B, LocateAnything 3B | weights load on demand (SAM: gated HF repo; LocateAnything: `mtsd-la` worker) |

Notes on coverage gaps within "ready": **YOLO12-large was never trained** (only
n/s/m exist), and **RF-DETR small/medium were never trained on MDWD** (only the
nano run `E001` exists). The benchmark reports what exists; if the dissertation
needs the full RF-DETR scale curve, those runs must be trained first.

### Opt-in only

- Cosmos Reason2 **32B**: refuses to run without `--allow-heavy`
  (VRAM/runtime cost; 32B needs multi-GPU or CPU offload).

### Pending (no results faked)

- **MTSD supervised detection** (YOLO12/YOLO26/RF-DETR): no trained checkpoints —
  `Results/MTSD-Runs/` is empty. The benchmark discovers this folder with the same
  scanner used for MDWD, so the moment the MTSD notebooks produce runs, the models
  appear in `--list-models` automatically. Until then they are listed as
  *pending* and selecting them records a pending row in the summary.

## 4. How to run (small first, then scale)

```bash
# 0. Inventory
python Scripts/Other-Scripts/Inference-Benchmark/inference_speed_benchmark.py --list-models

# 1. MDWD detection (MDWD env) — dry-run, then drop --dry-run
python Scripts/Other-Scripts/Inference-Benchmark/inference_speed_benchmark.py \
    --dataset MDWD --task detection --models yolo11 yolo12 yolo26 rf-detr \
    --split test --max-images 50 --batch-size 1 --dry-run

# 2. MTSD attribute classifiers (mtsd-attrcls env)
python Scripts/Other-Scripts/Inference-Benchmark/inference_speed_benchmark.py \
    --dataset MTSD --task attribute \
    --models dinov3 dinov3_lora vjepa vjepa_lora convnext_frozen convnext_finetuned \
    --max-images 50 --batch-size 1 --dry-run

# 3. PromptDetect (mtsd-base env; start with 10 images)
python Scripts/Other-Scripts/Inference-Benchmark/inference_speed_benchmark.py \
    --dataset PromptDetect --task prompt --models sam3 sam3.1 locateanything cosmos_reason2_2b \
    --max-images 10 --prompt "traffic sign" --dry-run

# 4. MTSD detection — after the MTSD notebooks have been trained
python Scripts/Other-Scripts/Inference-Benchmark/inference_speed_benchmark.py \
    --dataset MTSD --task detection --models yolo12 yolo26 rf-detr \
    --split test --max-images 50 --dry-run
```

## 5. Outputs per run

`Results/Inference-Benchmark/InferenceSpeed/<timestamp>/`:
`inference_speed_summary.csv` + `.md` (per-model metrics table),
`inference_speed_raw_timings.csv` (every per-image latency),
`benchmark_config.json` (full config + hardware), `benchmark_images_used.csv`,
`model_inventory.csv` (availability snapshot at run time), `latency_fps.png`.

## 6. Status checklist

- [x] Benchmark script implemented and dry-run verified (all three tasks)
- [x] Model inventory implemented (16 MDWD detection + 6 attribute checkpoints
      found ready; 6 prompt models registered; MTSD detection pending)
- [ ] MDWD detection speed benchmark **executed** — pending user confirmation
- [ ] Attribute classification speed benchmark **executed** — pending
- [ ] PromptDetect speed benchmark **executed** — pending (SAM/LocateAnything
      first; Cosmos 2B/8B next; 32B optional)
- [ ] MTSD detection speed benchmark — blocked on MTSD detection training
- [ ] Dissertation table/figure: accuracy-vs-latency (and vs model size) scatter
      combining this benchmark with the existing accuracy results — pending the
      runs above

## 7. Limitations to state in the dissertation

- Latency measured on a desktop RTX 4090 — representative of an edge-server
  deployment, not of embedded/mobile hardware; absolute numbers scale with GPU.
- Prompt-based VLMs (Cosmos, LocateAnything) expose no per-box confidence and no
  batching; their throughput is inherently single-stream.
- RF-DETR timing uses its high-level `predict()` API (single image), which
  includes the package's own pre/postprocessing — comparable to how it would be
  deployed, but not a pure forward-pass measurement.
- Attribute-classifier latency is per **crop**, not per full street image; an
  end-to-end sign-monitoring pipeline adds detector latency upstream.
