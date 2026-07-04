#!/usr/bin/env bash
# ============================================================
# Creates (or updates) the project's Conda environments (macOS / Linux).
#
# Usage:
#   ./setup_conda_env.sh <MDWD|mtsd-attrcls|mtsd-base|mtsd-la|all> [--cpu]
#
#   --cpu   Linux only: install CPU-only PyTorch wheels (no NVIDIA GPU).
#           macOS always uses the default PyPI build (CPU / Apple MPS).
#
# Examples:
#   ./setup_conda_env.sh mtsd-base
#   ./setup_conda_env.sh all --cpu
# ============================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OS="$(uname -s)"

usage() {
    echo "Usage: $(basename "$0") <MDWD|mtsd-attrcls|mtsd-base|mtsd-la|all> [--cpu]" >&2
    exit 1
}

[ $# -ge 1 ] || usage
TARGET="$1"
CPU_ONLY="${2:-}"

if ! command -v conda >/dev/null 2>&1; then
    for candidate in "$HOME/anaconda3/bin/conda" "$HOME/miniconda3/bin/conda" \
                     "$HOME/miniforge3/bin/conda" "/opt/homebrew/Caskroom/miniforge/base/bin/conda"; do
        if [ -x "$candidate" ]; then PATH="$(dirname "$candidate"):$PATH"; break; fi
    done
fi
command -v conda >/dev/null 2>&1 || { echo "conda was not found on PATH or in the default install locations." >&2; exit 1; }

# ---- Per-environment definitions (tested versions, 2026-07) ----------------
#   <env>  <yaml>  <torch pins>  <CUDA wheel index (Linux default)>
env_spec() {
    case "$1" in
        MDWD)         echo "environment-mdwd.yml|torch==2.11.0 torchvision==0.26.0|https://download.pytorch.org/whl/cu130" ;;
        mtsd-attrcls) echo "environment-mtsd-attrcls.yml|torch==2.11.0 torchvision==0.26.0|https://download.pytorch.org/whl/cu128" ;;
        mtsd-base)    echo "environment-mtsd-base.yml|torch==2.10.0 torchvision==0.25.0|https://download.pytorch.org/whl/cu128" ;;
        mtsd-la)      echo "environment-mtsd-la.yml|torch==2.11.0 torchvision==0.26.0|https://download.pytorch.org/whl/cu128" ;;
        *) return 1 ;;
    esac
}

setup_one() {
    local name="$1" spec yaml torch index
    spec="$(env_spec "$name")" || usage
    yaml="$HERE/$(echo "$spec" | cut -d'|' -f1)"
    torch="$(echo "$spec" | cut -d'|' -f2)"
    index="$(echo "$spec" | cut -d'|' -f3)"

    echo
    echo "=== [$name] ==="
    if conda env list | awk '{print $1}' | grep -qx "$name"; then
        echo "[$name] exists - updating from $(basename "$yaml") (--prune)"
        conda env update -n "$name" -f "$yaml" --prune
    else
        echo "[$name] creating from $(basename "$yaml")"
        conda env create -f "$yaml"
    fi

    if [ "$OS" = "Darwin" ]; then
        # macOS: default PyPI wheels (CPU / Apple MPS); no CUDA index exists.
        echo "[$name] installing PyTorch ($torch) from PyPI (CPU/MPS build)"
        # shellcheck disable=SC2086
        conda run -n "$name" python -m pip install $torch
    else
        [ "$CPU_ONLY" = "--cpu" ] && index="https://download.pytorch.org/whl/cpu"
        echo "[$name] installing PyTorch ($torch) from $index"
        # shellcheck disable=SC2086
        conda run -n "$name" python -m pip install $torch --index-url "$index"
    fi

    if [ "$name" = "mtsd-base" ]; then
        if [ "$OS" = "Darwin" ]; then
            echo "[mtsd-base] SKIPPING the Meta sam3 package: it imports triton, which has no macOS wheels."
            echo "[mtsd-base] SAM 3 / SAM 3.1 will be unavailable on this machine; Cosmos Reason2 still works."
        else
            echo "[mtsd-base] installing Meta sam3 package (SAM 3 / SAM 3.1)"
            conda run -n mtsd-base python -m pip install "git+https://github.com/facebookresearch/sam3.git"
        fi
    fi

    echo "[$name] done. Activate with:  conda activate $name"
}

if [ "$TARGET" = "all" ]; then
    for name in MDWD mtsd-attrcls mtsd-base mtsd-la; do setup_one "$name"; done
else
    setup_one "$TARGET"
fi

echo
echo "Note: mtsd-base / mtsd-la / MDWD download gated Hugging Face weights on first use -"
echo "run 'hf auth login' inside the env once (see Documents/PromptDetect.md)."
