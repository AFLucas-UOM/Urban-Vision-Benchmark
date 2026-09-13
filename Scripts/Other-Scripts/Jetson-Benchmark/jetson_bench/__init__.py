"""NVIDIA Jetson edge-device inference benchmark for the Urban-Vision-Benchmark.

Inference and deployment measurement only: nothing in this package trains,
augments, re-annotates or otherwise mutates datasets, annotations, trained runs
or existing evaluation artefacts. Every execution writes into a new timestamped
directory under ``Results/Jetson-Benchmark/``.

The package reuses the repository's completed workstation benchmark
(``Scripts/Other-Scripts/Inference-Benchmark/``) for its timing discipline via
``uvb_bench_core`` and for its seeded sample manifests, so the Jetson numbers
can be compared to the RTX 4090 baseline wherever the protocol genuinely
matches - and are explicitly marked ``not_protocol_comparable`` where it does
not.
"""

# Importing .paths first puts Scripts/Other-Scripts/Inference-Benchmark on
# sys.path, so every submodule can import the shared uvb_bench_core
# helpers regardless of which module is imported first.
from . import paths as paths  # noqa: F401

__all__ = ["__version__", "paths"]

__version__ = "1.0.0"
