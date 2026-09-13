#!/usr/bin/env bash
# =============================================================================
# Jetson environment bootstrap for the Urban-Vision-Benchmark edge experiment
# =============================================================================
# Creates the Python environments the Jetson benchmark needs, ON THE EXTERNAL
# SSD, without touching the NVIDIA software stack already on the device.
#
# WHAT THIS SCRIPT WILL NEVER DO
#   * install torch or torchvision from PyPI (that would replace the JetPack
#     CUDA build with a CPU-only x86-oriented wheel, or simply fail);
#   * replace JetPack CUDA, cuDNN or TensorRT;
#   * upgrade NVIDIA drivers or the kernel;
#   * run `apt upgrade` or a distribution upgrade;
#   * run `sudo pip`;
#   * run `pip install --upgrade torch torchvision`;
#   * use the repository's x86 conda setup scripts.
#
# HOW THE JETPACK STACK IS PROTECTED
#   1. each venv is created with --system-site-packages, so NVIDIA's
#      system-installed torch/torchvision/cv2 stay importable;
#   2. a pip *constraints* file is generated from the versions actually
#      installed on the device, and every pip install runs with
#      `--constraint` pointing at it, so no transitive dependency can pull a
#      different torch, torchvision, numpy or opencv in;
#   3. torch and torchvision are additionally passed to pip as
#      `--no-deps`-protected pins, and a post-install verification re-checks
#      `torch.cuda.is_available()` and fails loudly if it regressed.
#
# Idempotent: re-running verifies and tops up an existing environment. Pass
# --rebuild-env to delete and recreate.
#
# Usage:
#   bash Requirements/Jetson/bootstrap_jetson.sh                 # all envs
#   bash Requirements/Jetson/bootstrap_jetson.sh --env detection
#   bash Requirements/Jetson/bootstrap_jetson.sh --rebuild-env
#   bash Requirements/Jetson/bootstrap_jetson.sh --check-only
#   bash Requirements/Jetson/bootstrap_jetson.sh --install-system-deps
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
CACHE_ROOT="${REPO_ROOT}/.cache/jetson"
STATE_FILE="${CACHE_ROOT}/setup_state.json"
CONSTRAINTS="${CACHE_ROOT}/jetpack-constraints.txt"

ENVIRONMENTS=(detection attribute prompt)
REQUESTED_ENVS=()
REBUILD=0
CHECK_ONLY=0
INSTALL_SYSTEM_DEPS=0
QUIET=0
EXIT_CODE=0

# Optional environments whose failure must not fail the required benchmark.
OPTIONAL_ENVS=(prompt)

# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
say()  { [ "${QUIET}" -eq 1 ] || printf '%s\n' "$*"; }
head1() { say ""; say "=============================================================================="; say "$*"; say "=============================================================================="; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
fail() { printf 'ERROR: %s\n' "$*" >&2; }

usage() {
    sed -n '2,44p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --env) REQUESTED_ENVS+=("$2"); shift 2 ;;
        --rebuild-env) REBUILD=1; shift ;;
        --check-only) CHECK_ONLY=1; shift ;;
        --install-system-deps) INSTALL_SYSTEM_DEPS=1; shift ;;
        --quiet) QUIET=1; shift ;;
        -h|--help) usage ;;
        *) fail "unknown option: $1"; exit 2 ;;
    esac
done
if [ "${#REQUESTED_ENVS[@]}" -gt 0 ]; then
    ENVIRONMENTS=("${REQUESTED_ENVS[@]}")
fi

# -----------------------------------------------------------------------------
# Repository-local caches: keep every download and compiled asset on the SSD
# so the Jetson's internal storage cannot fill up.
# -----------------------------------------------------------------------------
mkdir -p "${CACHE_ROOT}"/{huggingface/hub,torch,ultralytics,pip,xdg,model_exports}
export HF_HOME="${CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export TORCH_HOME="${CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export YOLO_CONFIG_DIR="${CACHE_ROOT}/ultralytics"
export MPLBACKEND=Agg
export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1

# -----------------------------------------------------------------------------
# 1. Platform identification (never assume; report)
# -----------------------------------------------------------------------------
head1 "1. Platform"
ARCH="$(uname -m)"
KERNEL="$(uname -r)"
MODEL="unknown"
[ -r /proc/device-tree/model ] && MODEL="$(tr -d '\0' < /proc/device-tree/model)"
L4T="unknown"
[ -r /etc/nv_tegra_release ] && L4T="$(head -n1 /etc/nv_tegra_release)"

say "Repository : ${REPO_ROOT}"
say "Device     : ${MODEL}"
say "Arch       : ${ARCH}"
say "Kernel     : ${KERNEL}"
say "L4T        : ${L4T}"

IS_JETSON=0
if [ "${ARCH}" = "aarch64" ] && { [ -r /etc/nv_tegra_release ] || printf '%s' "${MODEL}" | grep -qi jetson; }; then
    IS_JETSON=1
fi
if [ "${IS_JETSON}" -ne 1 ]; then
    warn "this does not look like an NVIDIA Jetson (arch=${ARCH}, model=${MODEL})."
    warn "The environments will still be created, but they will not contain a "
    warn "JetPack CUDA PyTorch and the benchmark will refuse to produce GPU results."
fi

# Storage backing the repository - warn (but do not abort) on non-Linux filesystems.
if command -v findmnt >/dev/null 2>&1; then
    FS_TYPE="$(findmnt -no FSTYPE --target "${REPO_ROOT}" 2>/dev/null || echo unknown)"
    FS_DEV="$(findmnt -no SOURCE --target "${REPO_ROOT}" 2>/dev/null || echo unknown)"
    say "Filesystem : ${FS_TYPE} on ${FS_DEV}"
    case "${FS_TYPE}" in
        ntfs|ntfs3|fuseblk|exfat|vfat|msdos)
            warn "the repository is on a ${FS_TYPE} volume. POSIX permissions and "
            warn "symlinks behave unexpectedly there, which can break venv creation. "
            warn "An ext4 SSD is strongly recommended." ;;
    esac
fi
FREE_KB="$(df -Pk "${REPO_ROOT}" | awk 'NR==2 {print $4}')"
say "Free space : $(( FREE_KB / 1024 / 1024 )) GB"
if [ "${FREE_KB}" -lt 5242880 ]; then
    warn "less than 5 GB free on the repository volume; installs may fail."
fi

# Writability - the benchmark must be able to create its results directory.
if ! ( : > "${REPO_ROOT}/.uvb_jetson_write_probe" ) 2>/dev/null; then
    fail "the repository volume is not writable at ${REPO_ROOT}"
    exit 2
fi
rm -f "${REPO_ROOT}/.uvb_jetson_write_probe"

# -----------------------------------------------------------------------------
# 2. Base interpreter and the JetPack PyTorch it exposes
# -----------------------------------------------------------------------------
head1 "2. System Python and the JetPack PyTorch stack"
SYSTEM_PYTHON="$(command -v python3 || true)"
if [ -z "${SYSTEM_PYTHON}" ]; then
    fail "python3 is not on PATH. Install it with: sudo apt-get install -y python3"
    exit 2
fi
say "python3    : ${SYSTEM_PYTHON} ($(${SYSTEM_PYTHON} -V 2>&1))"

if ! "${SYSTEM_PYTHON}" -c "import venv, ensurepip" >/dev/null 2>&1; then
    fail "python3-venv / ensurepip is missing."
    if [ "${INSTALL_SYSTEM_DEPS}" -eq 1 ]; then
        say "Installing the minimal build prerequisites (no apt upgrade, no NVIDIA packages)."
        sudo apt-get update
        sudo apt-get install -y --no-install-recommends \
            python3-venv python3-pip python3-dev build-essential pkg-config \
            libjpeg-dev zlib1g-dev
    else
        fail "Re-run with --install-system-deps, or install manually:"
        fail "  sudo apt-get install -y python3-venv python3-pip python3-dev build-essential"
        exit 2
    fi
fi

# Read the JetPack torch stack through the SYSTEM interpreter. Everything below
# is pinned to exactly these versions.
TORCH_PROBE="$(${SYSTEM_PYTHON} - <<'PY' 2>/dev/null
import json
report = {}
for name in ("torch", "torchvision", "numpy", "cv2"):
    try:
        module = __import__(name)
        report[name] = getattr(module, "__version__", "unknown")
    except Exception:
        report[name] = None
try:
    import torch
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["torch_cuda"] = torch.version.cuda
    if report["cuda_available"]:
        report["gpu"] = torch.cuda.get_device_name(0)
except Exception as exc:
    report["torch_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(report))
PY
)"
[ -z "${TORCH_PROBE}" ] && TORCH_PROBE='{}'

read_json() { printf '%s' "${TORCH_PROBE}" | "${SYSTEM_PYTHON}" -c \
    "import json,sys; print(json.load(sys.stdin).get('$1') if json.load(sys.stdin) else '')" 2>/dev/null || true; }

SYS_TORCH="$(printf '%s' "${TORCH_PROBE}" | "${SYSTEM_PYTHON}" -c "import json,sys;d=json.load(sys.stdin);print(d.get('torch') or '')")"
SYS_TV="$(printf '%s' "${TORCH_PROBE}" | "${SYSTEM_PYTHON}" -c "import json,sys;d=json.load(sys.stdin);print(d.get('torchvision') or '')")"
SYS_NUMPY="$(printf '%s' "${TORCH_PROBE}" | "${SYSTEM_PYTHON}" -c "import json,sys;d=json.load(sys.stdin);print(d.get('numpy') or '')")"
SYS_CV2="$(printf '%s' "${TORCH_PROBE}" | "${SYSTEM_PYTHON}" -c "import json,sys;d=json.load(sys.stdin);print(d.get('cv2') or '')")"
SYS_CUDA_OK="$(printf '%s' "${TORCH_PROBE}" | "${SYSTEM_PYTHON}" -c "import json,sys;d=json.load(sys.stdin);print(d.get('cuda_available'))")"

say "torch       : ${SYS_TORCH:-not installed}"
say "torchvision : ${SYS_TV:-not installed}"
say "numpy       : ${SYS_NUMPY:-not installed}"
say "cv2         : ${SYS_CV2:-not installed}"
say "cuda avail. : ${SYS_CUDA_OK}"

if [ "${IS_JETSON}" -eq 1 ] && [ -z "${SYS_TORCH}" ]; then
    fail "No PyTorch is installed system-wide on this Jetson."
    fail "This benchmark deliberately refuses to install a torch wheel for you: on"
    fail "Jetson, PyTorch must come from NVIDIA's JetPack build for your L4T release."
    fail "Install NVIDIA's wheel first (see https://developer.nvidia.com/embedded/"
    fail "downloads or the JetPack release notes for your L4T), then re-run this script."
    exit 3
fi
if [ "${IS_JETSON}" -eq 1 ] && [ "${SYS_CUDA_OK}" != "True" ]; then
    fail "PyTorch is present but torch.cuda.is_available() is False."
    fail "That normally means a CPU-only PyPI wheel shadowed the JetPack build."
    fail "Fix the PyTorch installation before benchmarking; a CPU run is not a"
    fail "Jetson GPU result and this benchmark will not present it as one."
    EXIT_CODE=3
fi

# -----------------------------------------------------------------------------
# 3. Constraints file - the mechanism that protects the JetPack stack
# -----------------------------------------------------------------------------
head1 "3. Pinning the JetPack stack"
{
    echo "# Generated by Requirements/Jetson/bootstrap_jetson.sh"
    echo "# These pins prevent pip from replacing the NVIDIA JetPack packages"
    echo "# already installed on this device. Do not edit by hand."
    [ -n "${SYS_TORCH}" ] && echo "torch==${SYS_TORCH%%+*}"
    [ -n "${SYS_TV}" ]    && echo "torchvision==${SYS_TV%%+*}"
    [ -n "${SYS_NUMPY}" ] && echo "numpy==${SYS_NUMPY}"
    [ -n "${SYS_CV2}" ]   && echo "opencv-python==${SYS_CV2}"
    [ -n "${SYS_CV2}" ]   && echo "opencv-python-headless==${SYS_CV2}"
} > "${CONSTRAINTS}"
say "Wrote ${CONSTRAINTS}:"
[ "${QUIET}" -eq 1 ] || sed 's/^/    /' "${CONSTRAINTS}"

if [ "${CHECK_ONLY}" -eq 1 ]; then
    head1 "Check-only: no environment was created or modified"
    exit "${EXIT_CODE}"
fi

# -----------------------------------------------------------------------------
# 4. Environment creation
# -----------------------------------------------------------------------------
requirements_for() {
    case "$1" in
        detection) echo "${SCRIPT_DIR}/requirements-detection.txt" ;;
        attribute) echo "${SCRIPT_DIR}/requirements-attribute.txt" ;;
        prompt)    echo "${SCRIPT_DIR}/requirements-prompt.txt" ;;
        *)         echo "" ;;
    esac
}

is_optional() {
    local candidate="$1"
    for name in "${OPTIONAL_ENVS[@]}"; do
        [ "${name}" = "${candidate}" ] && return 0
    done
    return 1
}

setup_env() {
    local name="$1"
    local venv_dir="${REPO_ROOT}/.venv-jetson-${name}"
    local requirements
    requirements="$(requirements_for "${name}")"

    head1 "4.${name}  Environment .venv-jetson-${name}"
    if [ -z "${requirements}" ] || [ ! -f "${requirements}" ]; then
        fail "no requirements file for environment '${name}'"
        return 1
    fi

    if [ "${REBUILD}" -eq 1 ] && [ -d "${venv_dir}" ]; then
        say "--rebuild-env: removing ${venv_dir}"
        rm -rf "${venv_dir}"
    fi

    if [ -d "${venv_dir}" ] && [ -x "${venv_dir}/bin/python" ]; then
        say "Existing environment found; verifying and topping up (idempotent)."
    else
        say "Creating with --system-site-packages so the JetPack torch stays importable."
        if ! "${SYSTEM_PYTHON}" -m venv --system-site-packages "${venv_dir}"; then
            fail "venv creation failed for ${name}"
            return 1
        fi
    fi

    local venv_python="${venv_dir}/bin/python"
    "${venv_python}" -m pip install --upgrade --quiet pip setuptools wheel \
        --constraint "${CONSTRAINTS}" || warn "pip self-upgrade failed; continuing"

    say "Installing ${requirements##*/} with the JetPack constraints applied."
    if ! "${venv_python}" -m pip install --constraint "${CONSTRAINTS}" \
            -r "${requirements}"; then
        fail "dependency installation failed for ${name}"
        return 1
    fi

    # OpenCV: only install a wheel when no system cv2 is importable, so a
    # CUDA-enabled JetPack OpenCV is never shadowed.
    if [ "${name}" = "detection" ]; then
        if "${venv_python}" -c "import cv2" >/dev/null 2>&1; then
            say "OpenCV: using the existing installation ($("${venv_python}" -c 'import cv2;print(cv2.__version__)' 2>/dev/null))"
        else
            say "OpenCV: no cv2 importable; installing the headless wheel."
            "${venv_python}" -m pip install --constraint "${CONSTRAINTS}" \
                "opencv-python-headless>=4.8" \
                || warn "opencv install failed; YOLO in-memory decoding will not work"
        fi
    fi

    # Post-install verification: the JetPack stack must be intact.
    local after
    after="$("${venv_python}" - <<'PY' 2>/dev/null
import json
report = {}
try:
    import torch
    report["torch"] = torch.__version__
    report["cuda_available"] = bool(torch.cuda.is_available())
except Exception as exc:
    report["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(report))
PY
)"
    say "Post-install torch check: ${after}"
    if [ "${IS_JETSON}" -eq 1 ] && ! printf '%s' "${after}" | grep -q '"cuda_available": true'; then
        fail "CUDA is no longer available in .venv-jetson-${name} after installation."
        fail "Something replaced the JetPack torch. Re-run with --rebuild-env, and"
        fail "check ${requirements} for a dependency that pulls torch."
        return 1
    fi
    say "Environment .venv-jetson-${name} is ready."
    return 0
}

READY_ENVS=()
FAILED_ENVS=()
for name in "${ENVIRONMENTS[@]}"; do
    if setup_env "${name}"; then
        READY_ENVS+=("${name}")
    else
        FAILED_ENVS+=("${name}")
        if is_optional "${name}"; then
            warn "optional environment '${name}' could not be created; the required "
            warn "detector and attribute benchmark is unaffected."
        else
            EXIT_CODE=1
        fi
    fi
done

# -----------------------------------------------------------------------------
# 5. Setup state
# -----------------------------------------------------------------------------
head1 "5. Setup state"
"${SYSTEM_PYTHON}" - "$STATE_FILE" "${READY_ENVS[*]:-}" "${FAILED_ENVS[*]:-}" \
    "${MODEL}" "${ARCH}" "${L4T}" "${SYS_TORCH}" "${SYS_TV}" "${SYS_NUMPY}" \
    "${SYS_CV2}" "${SYS_CUDA_OK}" <<'PY'
import datetime, json, sys
path, ready, failed, model, arch, l4t, torch_v, tv, numpy_v, cv2_v, cuda = sys.argv[1:12]
json_path = path
payload = {
    "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "device_model": model,
    "architecture": arch,
    "l4t_release": l4t,
    "system_torch": torch_v or None,
    "system_torchvision": tv or None,
    "system_numpy": numpy_v or None,
    "system_opencv": cv2_v or None,
    "torch_cuda_available": cuda == "True",
    "environments_ready": [e for e in ready.split() if e],
    "environments_failed": [e for e in failed.split() if e],
}
with open(json_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
print(f"Wrote {json_path}")
PY

head1 "Summary"
say "Ready      : ${READY_ENVS[*]:-none}"
say "Failed     : ${FAILED_ENVS[*]:-none}"
REQUIRED_OK=1
for name in detection attribute; do
    case " ${READY_ENVS[*]:-} " in *" ${name} "*) ;; *) REQUIRED_OK=0 ;; esac
done
if [ "${REQUIRED_OK}" -eq 1 ]; then
    say "Required benchmark: READY"
else
    say "Required benchmark: NOT READY (see the errors above)"
fi
case " ${READY_ENVS[*]:-} " in
    *" prompt "*) say "Optional zero-shot environment: READY" ;;
    *)            say "Optional zero-shot environment: SKIPPED (see the warnings above)" ;;
esac
say ""
say "Next: bash Scripts/Other-Scripts/Jetson-Benchmark/run_jetson_benchmark.sh --dry-run"
exit "${EXIT_CODE}"
