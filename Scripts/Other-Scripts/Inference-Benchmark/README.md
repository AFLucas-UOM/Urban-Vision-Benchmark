# Inference-Benchmark

Deployment-oriented benchmarks for the dissertation's comparative evaluation —
metrics beyond accuracy: latency, throughput, memory, model size.

This folder holds the **workstation / RTX 4090** benchmark. The NVIDIA Jetson
edge-device benchmark lives in
[`Scripts/Other-Scripts/Jetson-Benchmark/`](../Jetson-Benchmark/README.md) and
reuses this suite's timing discipline and seeded sample manifests so the two
devices can be compared where the protocol genuinely matches.

## `uvb_bench_core.py`

Shared primitives used by both suites: project-root discovery, percentile and
batching helpers, `timed_loop` (warm-up, CUDA synchronisation, per-item latency
collection), checkpoint integrity checks (size, SHA-256, Git-LFS pointer
detection) and CSV writing. `inference_speed_benchmark.py` imports them under
their original names, so its behaviour, CSV schemas and CLI are unchanged.

## `inference_speed_benchmark.py`

Benchmarks every trained model family in the repository with a shared,
reproducible protocol (fixed-seed image sample, explicit warmup,
CUDA-synchronised timing, cold-start reported separately, batch 1 by default).

| Task | Models discovered from | Images sampled from | Conda env |
| --- | --- | --- | --- |
| `detection` (MDWD) | `Results/MDWD-Runs/*/E*/` (YOLO `best.pt`, RF-DETR `checkpoint_best_*.pth`; `[OLD]` suites excluded) | `Datasets/MDWD/MDWD-YOLO26/<split>/images` | `MDWD` |
| `detection` (MTSD) | `Results/MTSD-Runs/` — **pending until trained** | `Datasets/MTSD/Prepared/MTSD-YOLO/<split>/images` | `MDWD` |
| `attribute` | `AttributeClassification/outputs/checkpoints/<variant>/best.pt` (smokes excluded unless `--include-smoke`) | `AttributeClassification/outputs/crops/**` | `mtsd-attrcls` |
| `prompt` | PromptDetect backend registry (weights load on demand) | MTSD group images / MDWD split images | `mtsd-base` |

### Commands

```bash
# Inventory: which checkpoints exist, what is pending
python Scripts/Other-Scripts/Inference-Benchmark/inference_speed_benchmark.py --list-models

# MDWD detection (dry-run first, then real; MDWD env)
python Scripts/Other-Scripts/Inference-Benchmark/inference_speed_benchmark.py \
    --dataset MDWD --task detection --models yolo11 yolo12 yolo26 rf-detr \
    --split test --max-images 50 --batch-size 1 --dry-run

# MTSD attribute classification (mtsd-attrcls env)
python Scripts/Other-Scripts/Inference-Benchmark/inference_speed_benchmark.py \
    --dataset MTSD --task attribute --models dinov3 dinov3_lora vjepa vjepa_lora convnext_frozen convnext_finetuned \
    --max-images 50 --batch-size 1 --dry-run

# PromptDetect (mtsd-base env; Cosmos 32B needs --allow-heavy)
python Scripts/Other-Scripts/Inference-Benchmark/inference_speed_benchmark.py \
    --dataset PromptDetect --task prompt --models sam3 locateanything \
    --max-images 10 --prompt "traffic sign" --dry-run
```

Drop `--dry-run` to execute. Useful extras: `--batch-sizes 1 4 8` (one summary
row per size; RF-DETR and prompt models are single-image APIs and stay at 1),
`--device cpu|cuda`, `--warmup N`, `--seed N`, `--include-smoke`.

### Outputs

Each real run writes a fresh timestamped folder
`Results/Inference-Benchmark/InferenceSpeed/<stamp>/` containing
`inference_speed_summary.csv` / `.md`, `inference_speed_raw_timings.csv`
(per-image latencies), `benchmark_config.json`, `benchmark_images_used.csv`
(the exact seeded sample), `model_inventory.csv` and `latency_fps.png`.

### Notes

- Model selection is explicit; nothing runs by default. Selectors match ids or
  families (`yolo26` = all sizes; `rf-detr`, `convnext_finetuned` are aliases).
- Selecting a **pending** model (e.g. MTSD detection) records it as pending in
  the summary instead of faking numbers.
- SAM 3/3.1 weights are HF-gated and download on first load; LocateAnything
  spawns its `mtsd-la` worker; Cosmos 32B requires `--allow-heavy`.
- Run the same command per environment — dependency failures are recorded per
  model in the summary rather than aborting the whole run.
- Every YOLO row is measured at `--imgsz` (640 by default) regardless of the
  resolution the checkpoint was trained at, and RF-DETR is constructed without
  an explicit resolution, so it runs at its package default (N 384 / S 512 /
  M 576). The Jetson benchmark checks both facts before declaring a row
  protocol-comparable.
