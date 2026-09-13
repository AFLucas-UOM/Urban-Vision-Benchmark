# Jetson Environments

Python environments for the [Jetson edge-device benchmark](../../Scripts/Other-Scripts/Jetson-Benchmark/README.md),
created **on the external SSD** and designed around one rule:

> The NVIDIA software stack already installed on the device is never modified.

The repository's `Requirements/CondaEnvironments/` scripts are for x86 CUDA
workstations. They install PyTorch from `download.pytorch.org` wheels that do not
exist for aarch64+CUDA, so running them on a Jetson would at best fail and at
worst replace a working JetPack build with a CPU-only one. **Do not use them
here.**

## Files

| File | Purpose |
| --- | --- |
| [`bootstrap_jetson.sh`](bootstrap_jetson.sh) | Creates and updates the environments. Idempotent. |
| [`verify_jetson_stack.py`](verify_jetson_stack.py) | Read-only verification of the Jetson software stack. |
| [`requirements-detection.txt`](requirements-detection.txt) | Ultralytics + RF-DETR inference (MDWD and MTSD tracks). |
| [`requirements-attribute.txt`](requirements-attribute.txt) | DINOv3 / V-JEPA 2.1 + LoRA inference (attribute track). |
| [`requirements-prompt.txt`](requirements-prompt.txt) | Optional zero-shot prompted localisation. |

## Usage

The benchmark entry point runs the bootstrap for you:

```bash
./run_jetson_benchmark.sh
```

To run the setup on its own:

```bash
bash Requirements/Jetson/bootstrap_jetson.sh                  # all environments
bash Requirements/Jetson/bootstrap_jetson.sh --env detection  # just one
bash Requirements/Jetson/bootstrap_jetson.sh --check-only     # inspect, change nothing
bash Requirements/Jetson/bootstrap_jetson.sh --rebuild-env    # delete and recreate
bash Requirements/Jetson/bootstrap_jetson.sh --install-system-deps
```

Verify the stack at any time (this never modifies anything):

```bash
python3 Requirements/Jetson/verify_jetson_stack.py
python3 Requirements/Jetson/verify_jetson_stack.py --json
.venv-jetson-detection/bin/python Requirements/Jetson/verify_jetson_stack.py
```

Exit codes: `0` usable for a GPU benchmark, `1` a required check failed, `2` not
a Jetson.

## What is created

```text
<repository root>/
├── .venv-jetson-detection/     ultralytics, rfdetr, supervision, pycocotools
├── .venv-jetson-attribute/     transformers, peft, timm, safetensors
├── .venv-jetson-prompt/        transformers, accelerate (optional track)
└── .cache/jetson/
    ├── huggingface/            HF_HOME + HUGGINGFACE_HUB_CACHE
    ├── torch/                  TORCH_HOME (torch.hub checkpoints)
    ├── ultralytics/            YOLO_CONFIG_DIR
    ├── pip/                    PIP_CACHE_DIR
    ├── xdg/                    XDG_CACHE_HOME
    ├── model_exports/          TensorRT engines when the optional runtime is used
    ├── jetpack-constraints.txt generated pins that protect the JetPack stack
    ├── setup_state.json        what was installed, and the versions in play
    └── stack_verification.json the last stack verification result
```

Everything lives on the SSD, so a large download can never fill the Jetson's
internal storage. All of it is git-ignored.

Three separate environments exist because the three model stacks do not have to
be dependency-compatible with each other — in particular the prompt models are
by far the most likely to conflict, and isolating them is what lets the required
detector and attribute benchmark stay runnable when the prompt environment
cannot be built at all.

## How the JetPack stack is protected

The bootstrap **will never**:

* install `torch` or `torchvision` from PyPI;
* replace JetPack CUDA, cuDNN or TensorRT;
* upgrade NVIDIA drivers or the kernel;
* run `apt upgrade` or a distribution upgrade;
* run `sudo pip`;
* run `pip install --upgrade torch torchvision`;
* use the repository's x86 conda setup scripts.

Instead it applies three layers of protection:

1. **Inheritance.** Each environment is created with
   `python3 -m venv --system-site-packages`, so NVIDIA's system-installed
   `torch`, `torchvision` and CUDA-enabled `cv2` remain importable inside it.
2. **Constraints.** It reads the versions actually installed on the device and
   writes `.cache/jetson/jetpack-constraints.txt`:

   ```text
   torch==2.5.0
   torchvision==0.20.0
   numpy==1.26.4
   opencv-python==4.9.0
   opencv-python-headless==4.9.0
   ```

   Every `pip install` then runs with `--constraint` pointing at that file, so
   no transitive dependency of `ultralytics`, `rfdetr`, `transformers` or
   anything else can pull a different torch in. This is why the requirements
   files deliberately contain **no** `torch` line.
3. **Verification.** After installing, it re-checks
   `torch.cuda.is_available()` inside the environment and fails loudly if it
   regressed, naming the requirements file to inspect.

`opencv-python-headless` is installed **only** when no `cv2` is importable, so a
CUDA-enabled JetPack OpenCV is never shadowed.

## If PyTorch is missing or CPU-only

The bootstrap stops with a diagnostic and does not try to fix it. On a Jetson,
PyTorch must come from NVIDIA's JetPack build for your L4T release — installing
a PyPI wheel produces a CPU-only interpreter, and the benchmark refuses to
present a CPU run as a Jetson GPU result. Install NVIDIA's wheel for your
JetPack version first, then re-run.

`--device cpu` exists for debugging only and is never a valid dissertation
result.

## System packages

A fresh Jetson may lack the build prerequisites for Python wheels. With
`--install-system-deps` the bootstrap installs a minimal, explicitly listed set:

```text
python3-venv  python3-pip  python3-dev  build-essential  pkg-config
libjpeg-dev   zlib1g-dev
```

It runs `apt-get update` and that single `apt-get install`. It does not run
`apt upgrade`, does not change JetPack, does not install a different CUDA, and
does not touch NVIDIA driver or kernel packages.

## Deliberate omissions

| Package | Why it is not installed |
| --- | --- |
| `torch`, `torchvision` | Must come from JetPack; see above. |
| `sam3` (Meta native) | Built for x86 CUDA and imports `triton`; getting it onto aarch64 is dependency surgery, not an install. SAM 3 / 3.1 are then reported `dependency_unavailable` by the zero-shot preflight, which is a result rather than a failure. |
| `triton` | No supported Jetson wheel. Nothing in the required benchmark needs it. |
| `flash-attn` | No Jetson wheel; attention falls back to SDPA. |
| `wandb` | The edge benchmark is offline and locally authoritative. |
| `gradio`, `jupyterlab` | No UI or notebook is launched on the device. |
| `lingbot-vision` | Not part of the Jetson matrix; installs from a Git commit and pulls a second OpenCV that would fight the JetPack one. |
| `scikit-learn` | Only used for training-time metric computation. |

## Model weights

Weights are never downloaded implicitly. The benchmark runs offline
(`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) unless
`--allow-model-downloads` is passed. The supervised detection and attribute
tracks work entirely from the SSD's existing files once dependencies are
installed, with one caveat: the **V-JEPA 2.1 backbones load through
`torch.hub`** and need network access on first use. Run once with
`--allow-model-downloads` to populate `.cache/jetson/torch`, after which those
variants are fully offline too.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `python3-venv / ensurepip is missing` | Re-run with `--install-system-deps`, or `sudo apt-get install -y python3-venv`. |
| `No PyTorch is installed system-wide on this Jetson` | Install NVIDIA's JetPack PyTorch wheel for your L4T release first. |
| `torch.cuda.is_available() is False` | A CPU-only PyPI wheel is shadowing the JetPack build. Remove it and reinstall NVIDIA's wheel. |
| `CUDA is no longer available … after installation` | Something in a requirements file pulled torch. Re-run with `--rebuild-env`; the named requirements file is the place to look. |
| venv creation fails on an NTFS/exFAT SSD | POSIX permissions and symlinks misbehave there. Reformat the SSD as ext4. |
| The prompt environment fails | Expected on many Jetsons. The orchestrator prints `Optional zero-shot environment: SKIPPED (reason)` and continues; the required benchmark is unaffected. |
