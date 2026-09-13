#!/usr/bin/env bash
# Root-level convenience wrapper for the NVIDIA Jetson edge-device benchmark.
#
#     cd /path/to/Urban-Vision-Benchmark
#     ./run_jetson_benchmark.sh
#
# Everything lives in Scripts/Other-Scripts/Jetson-Benchmark/; this file only
# forwards to it so the one-command workflow works from the repository root on
# whatever mount point the external SSD happens to get.
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "${ROOT}/Scripts/Other-Scripts/Jetson-Benchmark/run_jetson_benchmark.sh" "$@"
