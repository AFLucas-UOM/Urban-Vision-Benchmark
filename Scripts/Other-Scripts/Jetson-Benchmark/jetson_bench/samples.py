"""Benchmark sample resolution and provenance.

The primary rule is reuse: where the completed workstation benchmark recorded
the exact seeded sample it measured
(``Results/Inference-Benchmark/InferenceSpeed/<run>/benchmark_images_used.csv``),
the Jetson measures the *same* files, so a statement such as

    "the same checkpoint and the same seeded benchmark image set were
     evaluated on the RTX 4090 and on the Jetson"

is literally true rather than approximately true.

Those historical manifests contain absolute Windows paths from the machine that
produced them. They are never used literally: every entry is re-anchored on the
current checkout via :func:`jetson_bench.paths.resolve_recorded_path`, so the
same SSD works at any mount point on any host.

Only identifiers, repository-relative paths and hashes are written to the
benchmark output. No private imagery is copied and no thumbnails are produced.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

from .paths import (
    ATTR_CROPS,
    MDWD_YOLO_DATASET,
    MTSD_GROUPS_ROOT,
    MTSD_STRONG_YOLO,
    MTSD_UNAUG_YOLO,
    PROJECT_ROOT,
    WORKSTATION_RUNS,
    repo_relative,
    resolve_recorded_path,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# The completed workstation benchmark runs, mapped to the (dataset, task) pair
# whose sample manifest they recorded. Taken from the provenance table in
# Scripts/Other-Scripts/Inference-Benchmark/consolidate_final_benchmark.py.
WORKSTATION_SAMPLE_RUNS: dict[tuple[str, str], tuple[str, ...]] = {
    ("MDWD", "detection"): ("20260910-163921",),
    ("MTSD", "detection"): ("20260910-164504", "20260910-221923", "20260910-224006"),
    ("MTSD", "attribute"): ("20260910-164007", "20260910-222307", "20260910-222038"),
    ("MTSD", "prompt"): ("20260910-165249",),
}


@dataclass
class SampleSet:
    """A resolved, ordered list of inputs plus its full provenance."""

    task: str
    dataset: str
    items: list[Path] = field(default_factory=list)
    source: str = ""                    # workstation_manifest | seeded_sample
    workstation_run: str | None = None
    workstation_manifest: str | None = None
    seed: int | None = None
    pool_root: str | None = None
    requested: int | None = None
    unresolved: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.items)

    def provenance(self) -> dict:
        return {
            "task": self.task,
            "dataset": self.dataset,
            "sample_source": self.source,
            "sample_count": self.count,
            "workstation_run": self.workstation_run,
            "workstation_manifest": self.workstation_manifest,
            "seed": self.seed,
            "pool_root": self.pool_root,
            "requested": self.requested,
            "unresolved_entries": len(self.unresolved),
            "unresolved_examples": self.unresolved[:5],
            "notes": self.notes,
        }

    def manifest_rows(self, with_hashes: bool = False) -> list[dict]:
        from uvb_bench_core import sha256_file

        rows = []
        for index, path in enumerate(self.items):
            row = {
                "index": index,
                "task": self.task,
                "dataset": self.dataset,
                "relative_path": repo_relative(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "sample_source": self.source,
                "workstation_run": self.workstation_run or "",
            }
            if with_hashes and path.exists():
                row["sha256"] = sha256_file(path)
            rows.append(row)
        return rows


def workstation_manifest_path(dataset: str, task: str) -> tuple[Path | None, str | None]:
    """Locate the workstation ``benchmark_images_used.csv`` for one task."""
    for run in WORKSTATION_SAMPLE_RUNS.get((dataset, task), ()):
        candidate = WORKSTATION_RUNS / run / "benchmark_images_used.csv"
        if candidate.is_file():
            return candidate, run
    return None, None


def read_workstation_manifest(path: Path) -> list[str]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    key = "image" if rows and "image" in rows[0] else (
        "relative_path" if rows and "relative_path" in rows[0] else None)
    if key is None:
        return []
    ordered = sorted(rows, key=lambda r: int(r.get("index", 0)))
    return [row[key] for row in ordered]


def pool_for(task: str, dataset: str, split: str = "test") -> tuple[list[Path], Path]:
    """The deterministic candidate pool used when no manifest can be reused.

    Mirrors the workstation benchmark's own sampling sources so a fallback
    sample is drawn from the same population.
    """
    if task == "attribute":
        root = ATTR_CROPS
        return sorted(root.rglob("*.jpg")), root
    if dataset == "MDWD":
        root = MDWD_YOLO_DATASET / split / "images"
        pool = sorted(p for p in root.iterdir()
                      if p.suffix.lower() in IMAGE_EXTENSIONS) if root.is_dir() else []
        return pool, root
    for root in (MTSD_STRONG_YOLO / split / "images", MTSD_UNAUG_YOLO / split / "images"):
        if root.is_dir():
            return sorted(p for p in root.iterdir()
                          if p.suffix.lower() in IMAGE_EXTENSIONS), root
    root = MTSD_GROUPS_ROOT
    pool = [p for p in sorted(root.glob("GRP-*/Images/*"))
            if p.suffix.lower() in IMAGE_EXTENSIONS]
    return pool, root


def resolve_sample(task: str, dataset: str, *, seed: int = 42, split: str = "test",
                   max_items: int | None = None, reuse_workstation: bool = True,
                   limit: int | None = None) -> SampleSet:
    """Resolve the input set for one task, preferring the workstation manifest.

    ``limit`` truncates the resolved set (used by the smoke profile) while
    preserving the manifest's order, so a smoke run is a strict prefix of the
    dissertation sample rather than a different sample.
    """
    sample = SampleSet(task=task, dataset=dataset, seed=seed)

    if reuse_workstation:
        manifest, run = workstation_manifest_path(dataset, task)
        if manifest is not None:
            recorded = read_workstation_manifest(manifest)
            resolved: list[Path] = []
            for entry in recorded:
                target = resolve_recorded_path(entry)
                if target is None:
                    sample.unresolved.append(entry)
                else:
                    resolved.append(target)
            if resolved:
                sample.items = resolved
                sample.source = "workstation_manifest"
                sample.workstation_run = run
                sample.workstation_manifest = repo_relative(manifest)
                sample.requested = len(recorded)
                if sample.unresolved:
                    sample.notes.append(
                        f"{len(sample.unresolved)} of {len(recorded)} recorded paths "
                        "could not be resolved against this checkout; the comparison "
                        "is marked partial rather than silently re-sampled")
                if limit and limit < len(sample.items):
                    sample.items = sample.items[:limit]
                    sample.notes.append(
                        f"truncated to the first {limit} manifest entries by profile")
                return sample
            sample.notes.append(
                f"workstation manifest {repo_relative(manifest)} resolved no files "
                "against this checkout")

    pool, root = pool_for(task, dataset, split)
    sample.pool_root = repo_relative(root)
    if not pool:
        sample.source = "unavailable"
        sample.notes.append(f"no benchmark inputs found under {sample.pool_root}")
        return sample
    requested = limit or max_items
    rng = random.Random(seed)
    if requested and requested < len(pool):
        sample.items = sorted(rng.sample(pool, requested))
    else:
        sample.items = pool
    sample.source = "seeded_sample"
    sample.requested = requested
    sample.notes.append(
        "workstation manifest unavailable; drew a fresh seeded sample - rows using "
        "this sample are not protocol-comparable to the workstation baseline")
    return sample


def resolve_prompt(config: dict) -> dict:
    """Recover the existing dissertation prompt definition for zero-shot timing.

    No new prompt is invented for this benchmark: the text comes from
    ``Scripts/Other-Scripts/PromptDetect/batch_evaluation/prompt_protocols/
    dissertation_protocol.yaml``.
    """
    from .config import load_yaml
    from .paths import PROMPT_PROTOCOLS

    zero_shot = config.get("zero_shot") or {}
    dataset = zero_shot.get("prompt_dataset", "MTSD")
    prompt_id = zero_shot.get("prompt_id", "mtsd-p12")
    protocol_path = PROMPT_PROTOCOLS / "dissertation_protocol.yaml"
    result = {
        "prompt_id": prompt_id,
        "prompt": None,
        "prompt_dataset": dataset,
        "prompt_source": repo_relative(protocol_path),
        "protocol_version": None,
        "conf_threshold": 0.30,
        "error": None,
    }
    if not protocol_path.is_file():
        result["error"] = "dissertation prompt protocol not found"
        return result
    try:
        protocol = load_yaml(protocol_path)
    except Exception as exc:  # pragma: no cover - malformed protocol
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    result["protocol_version"] = protocol.get("protocol_version")
    result["conf_threshold"] = float(
        (protocol.get("defaults") or {}).get("conf_threshold", 0.30))
    prompts = ((protocol.get("datasets") or {}).get(dataset) or {}).get("prompts") or []
    for entry in prompts:
        if str(entry.get("id")) == prompt_id:
            result["prompt"] = entry.get("prompt")
            result["prompt_group"] = entry.get("group")
            return result
    result["error"] = f"prompt id {prompt_id!r} not present in the {dataset} protocol"
    return result


__all__ = [
    "SampleSet", "resolve_sample", "resolve_prompt", "pool_for",
    "workstation_manifest_path", "read_workstation_manifest",
    "WORKSTATION_SAMPLE_RUNS", "PROJECT_ROOT",
]
