#!/usr/bin/env bash
# =============================================================================
# NVIDIA Jetson edge-device inference benchmark - single entry point
# =============================================================================
# Plug the SSD into a fresh Jetson, open a terminal in the repository, and run:
#
#     bash Scripts/Other-Scripts/Jetson-Benchmark/run_jetson_benchmark.sh
#
# (or ./run_jetson_benchmark.sh from the repository root - same thing).
#
# It orchestrates, in order:
#
#     hardware detection -> repository preflight -> environment creation/update
#     -> dependency verification -> checkpoint discovery -> dataset verification
#     -> smoke tests -> required benchmark -> optional zero-shot preflight
#     -> sustained telemetry -> consolidation -> figures -> final report
#
# The user never activates an environment or runs a sequence of Python programs.
#
# Inference only. Nothing is retrained, no dataset is modified, and every run
# writes into a NEW timestamped directory under Results/Jetson-Benchmark/.
#
# Common invocations
#     ... run_jetson_benchmark.sh --dry-run            # plan only; safe anywhere
#     ... run_jetson_benchmark.sh --profile smoke      # verify the stack quickly
#     ... run_jetson_benchmark.sh                      # full dissertation suite
#     ... run_jetson_benchmark.sh --track mdwd --track mtsd
#     ... run_jetson_benchmark.sh --resume Results/Jetson-Benchmark/<stamp>
#     ... run_jetson_benchmark.sh --skip-setup         # environments already built
#     ... run_jetson_benchmark.sh --rebuild-env        # recreate the venvs
#     ... run_jetson_benchmark.sh --sudo-prompt --power-mode 0 --lock-clocks
#
# Every other flag is passed straight through to run_jetson_benchmark.py
# (see --help there for the complete list).
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"
BOOTSTRAP="${REPO_ROOT}/Requirements/Jetson/bootstrap_jetson.sh"
VERIFY="${REPO_ROOT}/Requirements/Jetson/verify_jetson_stack.py"
ORCHESTRATOR="${SCRIPT_DIR}/run_jetson_benchmark.py"
CACHE_ROOT="${REPO_ROOT}/.cache/jetson"

SKIP_SETUP=0
REBUILD_ENV=0
SUDO_PROMPT=0
DRY_RUN=0
INSTALL_SYSTEM_DEPS=0
PASSTHROUGH=()

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-setup)          SKIP_SETUP=1; shift ;;
        --rebuild-env)         REBUILD_ENV=1; shift ;;
        --sudo-prompt)         SUDO_PROMPT=1; shift ;;
        --install-system-deps) INSTALL_SYSTEM_DEPS=1; shift ;;
        --dry-run)             DRY_RUN=1; PASSTHROUGH+=("$1"); shift ;;
        -h|--help)
            sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            echo
            echo "Orchestrator options:"
            "${REPO_ROOT}/.venv-jetson-detection/bin/python" "${ORCHESTRATOR}" --help \
                2>/dev/null || python3 "${ORCHESTRATOR}" --help
            exit 0 ;;
        *)                     PASSTHROUGH+=("$1"); shift ;;
    esac
done

# -----------------------------------------------------------------------------
# Repository-local caches and the offline/no-upload policy. Exported here so
# every child process - bootstrap, verifier, orchestrator, model workers -
# inherits exactly the same policy.
# -----------------------------------------------------------------------------
mkdir -p "${CACHE_ROOT}"/{huggingface/hub,torch,ultralytics,pip,xdg,model_exports}
export HF_HOME="${CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export TORCH_HOME="${CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export YOLO_CONFIG_DIR="${CACHE_ROOT}/ultralytics"
export ULTRALYTICS_CONFIG_DIR="${CACHE_ROOT}/ultralytics"
export MPLBACKEND=Agg
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

cd "${REPO_ROOT}" || { echo "ERROR: cannot enter ${REPO_ROOT}" >&2; exit 2; }

echo "=============================================================================="
echo " Urban-Vision-Benchmark - NVIDIA Jetson edge-device inference benchmark"
echo "=============================================================================="
echo " Repository : ${REPO_ROOT}"
echo " Caches     : ${CACHE_ROOT}  (nothing is written to the Jetson's internal disk)"
echo " Network    : W&B disabled; model downloads OFF unless --allow-model-downloads"
echo "=============================================================================="

# -----------------------------------------------------------------------------
# Ask for sudo ONCE, up front, only when the operator asked for something that
# genuinely needs it. The benchmark itself never prompts halfway through.
# -----------------------------------------------------------------------------
NEEDS_SUDO=0
for arg in "${PASSTHROUGH[@]:-}"; do
    case "${arg}" in
        --power-mode|--lock-clocks) NEEDS_SUDO=1 ;;
    esac
done
[ "${INSTALL_SYSTEM_DEPS}" -eq 1 ] && NEEDS_SUDO=1
if [ "${SUDO_PROMPT}" -eq 1 ] || [ "${NEEDS_SUDO}" -eq 1 ]; then
    if command -v sudo >/dev/null 2>&1; then
        echo
        echo "A platform command you requested (nvpmodel / jetson_clocks / apt) needs"
        echo "elevated privileges. Authenticating once now so the benchmark is not"
        echo "interrupted later."
        sudo -v || echo "WARNING: sudo authentication failed; those steps will be skipped."
    else
        echo "WARNING: sudo is not available; power-mode and clock changes will be skipped."
    fi
fi

# -----------------------------------------------------------------------------
# 1. Stack verification (read-only)
# -----------------------------------------------------------------------------
echo
echo ">>> Verifying the Jetson software stack (read-only)"
python3 "${VERIFY}" --output "${CACHE_ROOT}/stack_verification.json"
VERIFY_STATUS=$?
if [ "${VERIFY_STATUS}" -eq 1 ]; then
    echo
    echo "The Jetson stack verification reported a required failure (above)."
    echo "This script will not attempt to modify JetPack. Fix the reported condition"
    echo "and re-run."
    [ "${DRY_RUN}" -eq 1 ] || exit 3
fi

# -----------------------------------------------------------------------------
# 2. Environment creation / update (idempotent)
# -----------------------------------------------------------------------------
if [ "${SKIP_SETUP}" -eq 1 ]; then
    echo
    echo ">>> Skipping environment setup (--skip-setup)"
else
    echo
    echo ">>> Creating or updating the Jetson environments on the SSD"
    BOOTSTRAP_ARGS=()
    [ "${REBUILD_ENV}" -eq 1 ] && BOOTSTRAP_ARGS+=(--rebuild-env)
    [ "${INSTALL_SYSTEM_DEPS}" -eq 1 ] && BOOTSTRAP_ARGS+=(--install-system-deps)
    bash "${BOOTSTRAP}" "${BOOTSTRAP_ARGS[@]:-}"
    BOOTSTRAP_STATUS=$?
    if [ "${BOOTSTRAP_STATUS}" -ne 0 ]; then
        echo
        echo "Environment bootstrap reported problems (exit ${BOOTSTRAP_STATUS})."
        echo "The orchestrator will still run and will record, per configuration,"
        echo "exactly which environment was unusable."
    fi
fi

# -----------------------------------------------------------------------------
# 3. Orchestrator
# -----------------------------------------------------------------------------
# The orchestrator itself only needs the standard library plus matplotlib for
# figures; run it in the detection environment when one exists so figures work.
PYTHON="${REPO_ROOT}/.venv-jetson-detection/bin/python"
[ -x "${PYTHON}" ] || PYTHON="$(command -v python3)"
echo
echo ">>> Running the benchmark orchestrator with ${PYTHON}"
echo

"${PYTHON}" -s "${ORCHESTRATOR}" "${PASSTHROUGH[@]:-}"
STATUS=$?

echo
if [ "${STATUS}" -eq 0 ]; then
    echo "Benchmark finished. See Results/Jetson-Benchmark/<timestamp>/benchmark_report.md"
elif [ "${STATUS}" -eq 1 ]; then
    echo "Benchmark ended early (interrupted or partial). Completed results were kept;"
    echo "resume with --resume <run directory>."
else
    echo "Benchmark could not start (exit ${STATUS}); see the messages above."
fi
exit "${STATUS}"
