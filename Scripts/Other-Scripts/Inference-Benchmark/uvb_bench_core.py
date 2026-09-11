#!/usr/bin/env python3
"""Shared benchmarking primitives for the Urban-Vision-Benchmark deployment suites.

These helpers were factored out of ``inference_speed_benchmark.py`` (the
completed workstation / RTX 4090 benchmark) so that the NVIDIA Jetson
edge-device benchmark under ``Scripts/Other-Scripts/Jetson-Benchmark/`` can
reuse the *same* timing discipline instead of growing a second, subtly
different implementation.

Nothing here changes the behaviour of the workstation benchmark: the functions
are byte-for-byte equivalent to the originals and are re-imported under their
original names, so existing results, CSV schemas and CLI behaviour are
unaffected.

Module contents
---------------
* project-root discovery (``Datasets/`` + ``Scripts/`` marker directories);
* percentile / batching / metric-list helpers;
* ``timed_loop`` - warmup, CUDA synchronisation, per-item latency collection;
* checkpoint integrity helpers (size, SHA-256, Git-LFS pointer detection);
* CSV writing.

Only the standard library is imported at module scope; ``torch`` is imported
lazily inside ``timed_loop`` exactly as it was in the original file.
"""

from __future__ import annotations

import csv
import hashlib
import statistics
import time
from pathlib import Path

__all__ = [
    "find_project_root",
    "percentile",
    "make_batches",
    "clear_metric_lists",
    "file_size_mb",
    "timed_loop",
    "write_csv",
    "sha256_file",
    "lfs_pointer_reason",
    "summarise_latencies",
]

# A Git-LFS pointer file is a tiny text stub; real checkpoints are megabytes.
LFS_POINTER_MAX_BYTES = 1024
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec"
# Anything below this is not a plausible trained checkpoint for this repository
# (the smallest real artifact - a LoRA adapter - is above 1 MB).
SUSPICIOUSLY_SMALL_BYTES = 64 * 1024


def find_project_root(start: Path) -> Path:
    """Walk upwards from *start* until the repository root is found.

    The root is identified by the two directories every checkout has,
    ``Datasets/`` and ``Scripts/`` - never by an absolute path, so the same
    code works from a Windows workstation and from an external SSD mounted at
    an arbitrary point on a Jetson.
    """
    for parent in [start, *start.parents]:
        if (parent / "Datasets").exists() and (parent / "Scripts").exists():
            return parent
    raise RuntimeError("Could not locate project root (needs Datasets/ and Scripts/).")


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile, matching the workstation benchmark exactly."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q / 100 * (len(ordered) - 1))))
    return ordered[index]


def make_batches(items: list, batch_size: int) -> list[list]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def clear_metric_lists(metrics: dict[str, list]) -> None:
    for values in metrics.values():
        values.clear()


def file_size_mb(path: str | Path | None) -> float | None:
    if not path:
        return None
    path = Path(path)
    return round(path.stat().st_size / 1024**2, 2) if path.exists() else None


def timed_loop(run_batch, batches: list, device: str, warmup: int,
               reset_after_warmup=None) -> dict:
    """Warmup then time ``run_batch(batch)`` over all batches, CUDA-synchronised.

    Warmup iterations are executed but excluded from the measurement; CUDA is
    synchronised immediately before and after every timed call so that
    asynchronous kernel launches cannot leak into the next interval.
    """
    import torch

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    for batch in batches[:min(max(0, warmup), len(batches))]:
        run_batch(batch)
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    if reset_after_warmup is not None:
        reset_after_warmup()

    per_image_latency_ms: list[float] = []
    total_images = 0
    wall_start = time.perf_counter()
    for batch in batches:
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_batch(batch)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        n = len(batch)
        total_images += n
        per_image_latency_ms.extend([elapsed * 1000 / n] * n)
    wall_total = time.perf_counter() - wall_start

    peak_mem_gb = (round(torch.cuda.max_memory_allocated() / 1024**3, 3)
                   if use_cuda else None)
    return {
        "timed_images": total_images,
        "mean_latency_ms": round(statistics.fmean(per_image_latency_ms), 2),
        "median_latency_ms": round(statistics.median(per_image_latency_ms), 2),
        "p95_latency_ms": round(percentile(per_image_latency_ms, 95), 2),
        "fps": round(total_images / wall_total, 2) if wall_total else 0.0,
        "total_wall_seconds": round(wall_total, 3),
        "peak_gpu_mem_gb": peak_mem_gb,
        "per_image_latency_ms": per_image_latency_ms,
    }


def summarise_latencies(values: list[float]) -> dict:
    """Distribution summary used by the Jetson benchmark's multi-repeat passes.

    Returns ``None`` for percentiles the sample size cannot support (p99 needs
    at least 100 observations) rather than inventing a number.
    """
    if not values:
        return {}
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    summary = {
        "n": len(values),
        "mean_ms": round(mean, 3),
        "median_ms": round(statistics.median(values), 3),
        "p50_ms": round(percentile(values, 50), 3),
        "p90_ms": round(percentile(values, 90), 3),
        "p95_ms": round(percentile(values, 95), 3),
        "p99_ms": round(percentile(values, 99), 3) if len(values) >= 100 else None,
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
        "std_ms": round(stdev, 3),
        "cv": round(stdev / mean, 4) if mean else None,
    }
    return summary


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str | None:
    """SHA-256 of a file, streamed so multi-hundred-MB checkpoints are safe."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lfs_pointer_reason(path: str | Path) -> str | None:
    """Return a reason string when *path* is not a usable real checkpoint.

    ``Path.exists()`` is not sufficient evidence: a Git-LFS pointer, a
    zero-byte placeholder or a truncated copy all "exist". Returns ``None``
    when the file looks like a genuine artifact.
    """
    path = Path(path)
    if not path.is_file():
        return "missing"
    size = path.stat().st_size
    if size == 0:
        return "empty_file"
    if size <= LFS_POINTER_MAX_BYTES:
        try:
            head = path.open("rb").read(LFS_POINTER_MAX_BYTES)
        except OSError as exc:  # pragma: no cover - unreadable file
            return f"unreadable ({exc.__class__.__name__})"
        if head.startswith(LFS_POINTER_PREFIX):
            return "git_lfs_pointer"
    if size < SUSPICIOUSLY_SMALL_BYTES:
        return f"suspiciously_small ({size} bytes)"
    return None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
