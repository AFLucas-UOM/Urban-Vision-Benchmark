# Conda Environments

Reproducible Conda environment setup for every GPU/experiment component of the
Urban-Vision-Benchmark repository. Each environment has one YAML file (Python version +
pinned pip dependencies) and is created by one platform-appropriate setup
script that also installs the correct PyTorch build for your hardware.

Pinned versions mirror the working environments on the primary Windows machine
(captured 2026-07-03), so a rebuilt env matches the one the dissertation
experiments ran in.

## The environments

| Environment | YAML | Python | Used by |
|---|---|---|---|
| `MDWD` | [environment-mdwd.yml](environment-mdwd.yml) | 3.12 | Maltese Domestic Waste Dataset work — YOLO (ultralytics), VLM experiments, annotation notebooks |
| `mtsd-attrcls` | [environment-mtsd-attrcls.yml](environment-mtsd-attrcls.yml) | 3.11 | [Scripts/MTSD-Scripts/AttributeClassification/](../../Scripts/MTSD-Scripts/AttributeClassification/) — DINOv3 / V-JEPA / ConvNeXt attribute experiments |
| `mtsd-base` | [environment-mtsd-base.yml](environment-mtsd-base.yml) | 3.12 | [Scripts/Other-Scripts/PromptDetect/](../../Scripts/Other-Scripts/PromptDetect/) main app (SAM 3 / 3.1 + Cosmos Reason2) and the [GDPR-Compliance](../../Scripts/Other-Scripts/GDPR-Compliance/) `sam31` backend |
| `mtsd-la` | [environment-mtsd-la.yml](environment-mtsd-la.yml) | 3.12 | PromptDetect's LocateAnything-3B worker (`transformers==4.57.1`, spawned automatically — never activated by hand) |

The corresponding loose pip requirements (unpinned/manual installs) live in
[`Requirements/`](../../Requirements/) at the repository root; the YAMLs here
are the pinned, reproducible superset of those files.

## Why PyTorch is not in the YAMLs

`torch`/`torchvision` wheels differ per platform (CUDA 12.8, CUDA 13.0, CPU,
macOS MPS) and come from different package indexes, which a single portable
YAML cannot express. The setup scripts install the tested torch pair for each
env **after** creating it from the YAML:

| Environment | Torch pair | Default index (Windows / Linux) |
|---|---|---|
| `MDWD` | `torch==2.11.0` + `torchvision==0.26.0` | `https://download.pytorch.org/whl/cu130` |
| `mtsd-attrcls` | `torch==2.11.0` + `torchvision==0.26.0` | `https://download.pytorch.org/whl/cu128` |
| `mtsd-base` | `torch==2.10.0` + `torchvision==0.25.0` | `https://download.pytorch.org/whl/cu128` |
| `mtsd-la` | `torch==2.11.0` + `torchvision==0.26.0` | `https://download.pytorch.org/whl/cu128` |

On macOS the scripts install the same versions from the default PyPI index
(CPU / Apple MPS build). On machines without an NVIDIA GPU pass the CPU flag
(see below). The original `MDWD` env ran a torch 2.10 cu130 *nightly*; the
scripts install the closest stable build instead.

Other platform specifics handled automatically:

- `triton-windows` is installed only on Windows (environment marker in
  `environment-mtsd-base.yml`); Linux gets official `triton` with torch.
- The Meta `sam3` package (SAM 3 / 3.1) is installed into `mtsd-base` by the
  scripts *after* torch. On macOS it is skipped — it imports `triton`, which
  has no macOS wheels — so SAM 3 / 3.1 are unavailable there.
- `decord` has no Apple-Silicon wheels; the YAMLs install `eva-decord` on
  macOS arm64 instead.

## Create

Run from this folder (`Requirements/CondaEnvironments/`). Replace `mtsd-base` with
`MDWD`, `mtsd-attrcls`, `mtsd-la`, or `all`.

**Windows PowerShell**

```powershell
.\setup_conda_env.ps1 -Name mtsd-base          # CUDA (default)
.\setup_conda_env.ps1 -Name all -Cpu           # every env, CPU-only torch
```

**Windows Command Prompt (cmd)**

```bat
setup_conda_env.cmd mtsd-base
setup_conda_env.cmd all cpu
```

**macOS**

```bash
chmod +x setup_conda_env.sh      # first time only
./setup_conda_env.sh mtsd-base   # PyPI torch (CPU / MPS), sam3 skipped
```

**Linux**

```bash
chmod +x setup_conda_env.sh      # first time only
./setup_conda_env.sh mtsd-base   # CUDA (default)
./setup_conda_env.sh all --cpu   # every env, CPU-only torch
```

The scripts are idempotent: if the env already exists they update it from the
YAML with `--prune` instead of failing.

### After creating `mtsd-base`, `mtsd-la`, or `MDWD`

Model weights are gated on Hugging Face. Authenticate once inside the env and
request access to the model repos you need (full list and links in
[Documents/PromptDetect.md](../../Documents/PromptDetect.md)):

```bash
conda activate mtsd-base
hf auth login
```

## Activate

Identical on every platform (PowerShell, cmd, macOS, Linux):

```bash
conda activate mtsd-base     # or MDWD / mtsd-attrcls / mtsd-la
conda deactivate
```

`mtsd-la` never needs manual activation — PromptDetect spawns its worker
automatically.

## Update

After editing a YAML (or pulling changes), re-run the setup script (it
updates in place), or apply just the YAML by hand — identical on every
platform:

```bash
conda env update -n mtsd-base -f environment-mtsd-base.yml --prune
```

`--prune` removes packages that are no longer in the YAML. Re-run the setup
script instead of the raw command if torch or `sam3` also needs refreshing.

## Remove

Identical on every platform:

```bash
conda deactivate                       # if the env is active
conda env remove -n mtsd-base --yes
```

Model weights are cached outside the envs (in `~/.cache/huggingface/hub`), so
removing an env does not delete downloaded weights — see
[Documents/CleanModelCache.md](../../Documents/CleanModelCache.md) for
reclaiming that space.

## Notes

- **Windows user-site packages:** this machine has a shared
  `AppData\Roaming\Python` site-packages that can shadow env packages. Run
  AttributeClassification entry points with `PYTHONNOUSERSITE=1` and `python
  -s` (see [Scripts/MTSD-Scripts/AttributeClassification/README.md](../../Scripts/MTSD-Scripts/AttributeClassification/README.md)).
- **flash-attention:** the primary machine additionally carries a custom
  `flash_attn_3` build inside `mtsd-base`. It is optional (attention falls
  back to SDPA) and not pip-installable portably, so it is deliberately not
  part of the YAML.
- The general (non-GPU) tooling — EDA notebook, Label Studio scripts, atlas —
  needs no dedicated env; `pip install -r
  Requirements/requirements-general-tooling.txt` into any Python ≥ 3.10
  works (the Anaconda `base` env is used on the primary machine).
