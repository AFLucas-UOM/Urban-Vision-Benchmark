# Jetson-Benchmark

NVIDIA Jetson edge-device **inference** benchmark for the dissertation's
deployment evidence: which selected models actually run on constrained edge
hardware, how fast, at what memory, power, energy and thermal cost, and how far
behind the completed RTX 4090 workstation benchmark they fall.

Nothing here trains, re-evaluates, augments or otherwise mutates anything.
Datasets, annotations, trained checkpoints, existing experiment directories and
existing final results are read-only, and every execution writes into a new
timestamped directory under `Results/Jetson-Benchmark/`.

## One command

Plug the SSD into the Jetson, open a terminal in the repository, and run:

```bash
./run_jetson_benchmark.sh
```

(equivalently `bash Scripts/Other-Scripts/Jetson-Benchmark/run_jetson_benchmark.sh`).

That single command performs, in order:

```text
hardware detection → repository preflight → environment creation/update
→ dependency verification → checkpoint discovery → dataset verification
→ smoke tests → required benchmark → optional zero-shot preflight
→ sustained telemetry → consolidation → figures → final report
```

You never activate an environment or run a sequence of Python programs.

### Useful variants

```bash
./run_jetson_benchmark.sh --dry-run           # plan only; safe on the workstation
./run_jetson_benchmark.sh --profile smoke     # a couple of inputs, one repeat
./run_jetson_benchmark.sh --track mdwd --track mtsd
./run_jetson_benchmark.sh --models mtsd_rfdetr_m_strong_trained
./run_jetson_benchmark.sh --resume Results/Jetson-Benchmark/<timestamp>
./run_jetson_benchmark.sh --skip-setup        # environments already built
./run_jetson_benchmark.sh --rebuild-env       # recreate the venvs from scratch
./run_jetson_benchmark.sh --allow-model-downloads   # permit HF / torch.hub fetches
./run_jetson_benchmark.sh --sudo-prompt --power-mode 0 --lock-clocks
./run_jetson_benchmark.sh --runtime tensorrt  # separate optimised-runtime rows
```

`--dry-run` inspects the hardware, resolves every checkpoint, verifies datasets
and environments, resolves the exact sample set and prints the execution plan
without loading a model or creating a run directory. It is the first thing to
run, and it works on a workstation as well as on the Jetson.

## The benchmark matrix

Encoded explicitly in [`config/jetson_benchmark.yaml`](config/jetson_benchmark.yaml).
Each user-facing request resolves to exactly one checkpoint.

| Track | Requested configurations |
| --- | --- |
| MDWD detection | YOLO26-S, YOLO26-L, RF-DETR-M, RF-DETR-S, YOLO12-S, YOLO11-S |
| MTSD detection | YOLO26-S/M strong @1280, RF-DETR-M/S strong @ trained resolution, YOLO12-S and YOLO11-S strong @1280 |
| Attribute classification | `dinov3_vitl_lora`, `vjepa21_vitl_lora`, `dinov3_vitb_lora`, `vjepa21_vitb_lora` |
| Zero-shot (optional, `auto`) | SAM 3, SAM 3.1, LocateAnything-3B, Cosmos Reason2 2B/8B — Cosmos 32B is excluded |

### Two configurations in this repository cannot be benchmarked

The dry run reports these as `missing_checkpoint` with the evidence, and they
are **not** substituted with a different model, resolution or augmentation
regime:

* **MDWD RF-DETR-M and RF-DETR-S** — only RF-DETR-N was trained on MDWD
  (`Results/MDWD-Runs/RF-DETR-EUVIP/` holds a single `E001_rfdetr-n_…` run, and
  the final MDWD detection table contains no RF-DETR rows at all).
* **MTSD YOLO12-S strong @1280** — the run directories exist
  (`FINAL_yolo12s_mtsd_strong_img1280_…` and
  `strongaug-wandb-followup-yolo12s-img1280-s42`) and carry weights, but the
  MTSD experiment matrix records both as `completion_status=partial` with no
  test metrics. Benchmarking them would attach a 640 or 960 accuracy figure to a
  1280 checkpoint, so the configuration is reported as missing instead.

## How a checkpoint is resolved

Fuzzy prefix matching is not used for the dissertation suite. Resolution order:

1. explicit final-report provenance
   (`Documents/Final-Reports/MTSD-SupervisedDetection/mtsd_complete_experiment_matrix.csv`,
   `Documents/Final-Tables/20260712-213548/mdwd_detection_results.csv`);
2. current final benchmark provenance
   (`Documents/Final-Tables/20260910-inference-benchmark/`);
3. run metadata on disk (`args.yaml`, `run_record.json`);
4. `FINAL_*` / non-probe naming;
5. the most authoritative immutable retained artifact.

`[OLD]`, `PROBE_`, `-smoke` and incomplete runs are excluded. If more than one
eligible run survives, the configuration is recorded as `ambiguous_checkpoint`
rather than guessed, and the newest timestamp is never used as a tie-break. The
resolution and its evidence are written to `model_manifest.csv`.

Input resolution always comes from the run's own metadata. RF-DETR is
benchmarked at the resolution it was actually trained and evaluated at (576 for
M, 512 for S in this repository) and is never forced to 1280.

## Two passes

**Pass A — protocol-compatible latency.** Reuses the exact seeded sample
manifest that the completed workstation benchmark recorded
(`Results/Inference-Benchmark/InferenceSpeed/<run>/benchmark_images_used.csv`),
at batch size 1 with the same warm-up count, over the same checkpoint. Those
manifests store absolute Windows paths; every entry is re-anchored on the
current checkout, so the same SSD works at any mount point. Three repeats by
default; individual repeats are kept and an extra pooled `repeat="all"` row is
added from the individual observations, never from an average of averages.

**Pass B — sustained edge telemetry.** Cycles the same deterministic sample for
at least 20 s per repeat (bounded above, so a pathological model cannot run
forever) with `tegrastats` sampling at 100 ms, which is what makes the power,
energy, memory and thermal numbers meaningful.

### Timing boundaries

| Boundary | Includes |
| --- | --- |
| `model_forward` | the model computation alone, CUDA-synchronised either side |
| `in_memory_pipeline` | already-decoded input → preprocessing → inference → postprocessing |
| `end_to_end_file` | SSD read and decode plus the whole in-memory pipeline |

These are never averaged together; each row states its own boundary. Both
detectors receive pre-decoded pixels for the in-memory boundary, so neither
framework is handed a file path while the other gets a decoded array.

The cross-device comparison uses, per engine, the boundary that **reproduces
what the workstation benchmark actually timed**: file paths for YOLO,
pre-loaded PIL images for RF-DETR, pre-transformed tensors for the attribute
classifiers, pre-decoded arrays for the prompt backends.

## When a comparison is valid

`jetson_vs_workstation.csv` computes a slowdown factor only when the checkpoint,
task, batch size, timing boundary and effective input resolution all match.
Everything else is marked `not_protocol_comparable` with the reason. Two rules
matter in this repository:

* the workstation benchmark ran **every** YOLO model at `imgsz=640`, including
  the 1280-trained MTSD checkpoints, so the MTSD @1280 configurations are not
  comparable;
* the workstation constructed RF-DETR without an explicit resolution, i.e. at
  the package default for the scale (N 384 / S 512 / M 576), so RF-DETR is
  comparable exactly when the trained resolution equals that default — which it
  does for the MTSD strong S and M runs.

Memory is reported side by side but never as a ratio: Jetson unified memory and
RTX 4090 VRAM allocation are not the same quantity.

## Power and energy

Rails are never blindly summed, because they can overlap hierarchically.

| Role | Meaning |
| --- | --- |
| `board_input_rail` | a genuine total-input rail exists (`VDD_IN`, `POM_5V_IN`) — board input power |
| `documented_disjoint_rail_sum` | a known non-overlapping rail set (e.g. the AGX Orin trio) — **module** power, explicitly not board input |
| `unresolved` | nothing defensible; total power and energy are left null, per-rail telemetry is kept, and board power is **not** estimated |

Energy is the trapezoidal integral of P(t) over the measured interval divided by
the number of processed items (image, crop, or image-prompt pair). An
idle-adjusted `dynamic_energy_j_per_item` is reported **in addition to**, never
instead of, the gross measured figure, and the subtraction method is recorded
with it.

## Optional TensorRT runtime

`--runtime tensorrt` (or `--profile dissertation_tensorrt`) adds a deployment-
optimised YOLO runtime as a **separate experiment**. It is never a prerequisite
for anything: the native PyTorch benchmark runs first and unchanged, and a
failed export cannot affect it because each TensorRT variant is its own
configuration measured in its own worker process.

* the engine is exported **on the Jetson itself**, FP16 by default, at the
  checkpoint's own input size;
* it is cached under `.cache/jetson/model_exports/`, keyed by checkpoint hash +
  input size + precision + TensorRT version, so an incompatible cached plan is
  never silently reused;
* export duration, engine path, precision, TensorRT and CUDA versions and
  whether the cache was reused are all recorded;
* a prediction parity check runs against the native checkpoint on one
  representative image before benchmarking, and its result is stored with the
  smoke record (it is a sanity check, not an accuracy evaluation);
* rows carry `runtime_backend=tensorrt`, land in their own Pareto group, and
  are excluded from the workstation comparison, whose baseline is native
  PyTorch.

Only YOLO models are exported. Giving one architecture an optimised runtime and
not the others would make a cross-model comparison meaningless. INT8 is out of
scope: this repository has no calibration protocol and one is not invented here.

## Outputs

```text
Results/Jetson-Benchmark/<timestamp>/
├── benchmark_config.json        effective configuration, profile, CLI overrides
├── hardware_snapshot.json       Jetson module, L4T/JetPack, memory, power mode, thermals
├── software_snapshot.json       CUDA / cuDNN / TensorRT / PyTorch / library versions
├── repository_snapshot.json     commit, branch, dirty state, SSD filesystem, free space
├── requested_coverage.csv       one row per requested configuration and its outcome
├── model_manifest.csv           the exact checkpoint each request resolved to, with evidence
├── sample_manifest.csv          the exact inputs measured (paths and hashes only)
├── sample_provenance.json       which workstation manifest each task reused
├── latency_summary.csv          one row per model / pass / repeat / timing boundary
├── telemetry_summary.csv        per-window power, energy, memory, utilisation, thermal
├── deployment_summary.csv       the compact per-configuration deployment view
├── raw_timings.csv.gz           every individual per-item latency observation
├── jetson_vs_workstation.csv    cross-device comparison with validity flags
├── memory_summary.csv           per-model memory, including residual after release
├── thermal_summary.csv          per-model start/mean/peak/end temperature and cooldown
├── failures.csv                 every failure with a structured status
├── zero_shot_preflight.json     why each optional model was or was not benchmarked
├── run_state.json               resumable state, written after every pass
├── benchmark_report.md          the dissertation-ready report
├── environment_freeze/          pip freeze per environment used
├── telemetry/                   idle baseline, parsed samples, raw tegrastats lines
├── logs/                        orchestrator log and per-model worker logs
└── figures/                     generated figures plus a manifest of skipped ones
```

No source imagery is copied and no thumbnails are produced: the outputs contain
identifiers, repository-relative paths and metrics only.

## Failure isolation and resumability

Each model runs in its own short-lived worker process
([`jetson_worker.py`](jetson_worker.py)) using the interpreter of the virtual
environment that owns its dependencies. A missing aarch64 wheel, a CUDA OOM or a
hard crash ends one model, not the experiment. Structured statuses:

```text
pending  preflight_ok  smoke_ok  benchmarking  ok  skipped
missing_checkpoint  missing_dataset  ambiguous_checkpoint  dependency_error
load_error  cuda_oom  runtime_error  thermal_abort  telemetry_unavailable
```

State is written atomically after every pass, so `--resume <run-directory>`
skips exactly the model/pass/repeat combinations that already completed
(`--rerun-completed` overrides). Ctrl+C produces a clean partial run: tegrastats
is stopped, worker process groups are terminated, and completed results are
consolidated.

## Zero-shot models are optional and non-blocking

`zero_shot.mode: auto` preflights each candidate — dependencies importable,
weights present in the repository-local cache (or downloads explicitly allowed),
enough available memory — and benchmarks only those that pass. Everything else
is recorded with a structured reason (`dependency_unavailable`,
`missing_local_weights`, `hf_auth_required`, `insufficient_memory`,
`unsupported_on_jetson`, `smoke_test_failed`, `manual_setup_required`) and the
required benchmark continues. The prompt text comes from the existing
dissertation protocol; no new prompt is invented. Zero-shot latency is reported
per **image-prompt pair**.

## Environments and network policy

Environments are created by
[`Requirements/Jetson/bootstrap_jetson.sh`](../../../Requirements/Jetson/bootstrap_jetson.sh)
on the SSD, with `--system-site-packages` so NVIDIA's JetPack PyTorch stays
importable, and with a generated pip constraints file that prevents any
dependency from replacing it. See
[`Requirements/Jetson/README.md`](../../../Requirements/Jetson/README.md).

Caches live under `.cache/jetson/` on the SSD, `WANDB_MODE=disabled`, and
Hugging Face / torch.hub downloads are off unless `--allow-model-downloads` is
passed. No inference image and no telemetry is uploaded anywhere.

## Relationship to the workstation benchmark

Both suites share one timing implementation:
[`Scripts/Other-Scripts/Inference-Benchmark/uvb_bench_core.py`](../Inference-Benchmark/uvb_bench_core.py)
holds warm-up, CUDA synchronisation, percentile and checkpoint-integrity
helpers. `inference_speed_benchmark.py` imports them under their original names,
so its results, CSV schemas and CLI are unchanged.

## Tests

The suite runs on a non-Jetson development machine and needs neither the private
datasets nor the trained checkpoints:

```bash
python -m pytest Scripts/Other-Scripts/Jetson-Benchmark/tests -q
```

It covers hardware-detection fallbacks, the tegrastats parser against four
Jetson generations plus truncated output, the power-rail resolver, energy
integration, model-request resolution and the exact MTSD selection rules,
Git-LFS pointer detection, result aggregation, resume-state handling, Pareto
calculation, workstation-baseline joining, configuration parsing, figure
generation and report assembly — plus a full end-to-end orchestration run
against a stubbed worker.
