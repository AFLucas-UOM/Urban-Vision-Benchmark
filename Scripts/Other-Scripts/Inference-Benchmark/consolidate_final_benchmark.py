"""Create a traceable, deduplicated table of the completed inference benchmarks.

The source runs use the same summary schema.  Re-runs are deliberately given
higher precedence so that repaired measurements replace their failed rows,
while an absolute checkpoint path (or task/model for on-demand models) keeps
each evaluated artifact to one row.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
BENCH_ROOT = REPO / "Results" / "Inference-Benchmark" / "InferenceSpeed"
OUTPUT = REPO / "Documents" / "Final-Tables" / "20260915-inference-benchmark"

# Ordered by precedence.  The last three are the targeted successful repairs.
SOURCES = [
    ("20260910-163921", "MDWD object detectors", 10),
    ("20260910-164007", "MTSD attribute classifiers (initial)", 10),
    ("20260910-164504", "MTSD object detectors (initial)", 10),
    ("20260910-165249", "PromptDetect models", 10),
    ("20260910-221923", "MTSD RF-DETR compact-name repair", 20),
    ("20260910-222307", "V-JEPA import-isolation repair", 20),
    ("20260910-224006", "MTSD RF-DETR-M strong repair", 30),
    ("20260915-023903", "MDWD standalone RF-DETR-S/M", 40),
    ("20260915-023922", "MTSD RF-DETR-N/S/M 640-Strong", 40),
    ("20260915-024003", "MTSD YOLO 1280 inference", 50),
]

FIELDS = [
    "dataset", "task", "model", "suite", "run_identifier", "checkpoint",
    "input_size", "batch_size", "timed_images", "device", "gpu_name",
    "checkpoint_artifact_mb", "parameters", "mean_latency_ms",
    "median_latency_ms", "p95_latency_ms", "fps", "cold_start_s",
    "peak_gpu_mem_gb", "preprocess_ms", "inference_ms", "postprocess_ms",
    "source_run", "source_description", "notes",
]


def norm_path(value: str) -> str:
    return value.replace("/", "\\").casefold()


def row_key(row: dict[str, str]) -> str:
    checkpoint = row.get("checkpoint", "").strip()
    if checkpoint:
        return "checkpoint:" + norm_path(checkpoint)
    return "logical:" + "|".join(
        [row.get("dataset", ""), row.get("task", ""), row.get("model", "")]
    ).casefold()


def trained_run(notes: str) -> str:
    prefix = "trained run "
    if notes.startswith(prefix):
        return notes[len(prefix):].split(";", 1)[0].strip()
    return ""


def as_number(value: str) -> float | None:
    try:
        return float(value) if value.strip() else None
    except (TypeError, ValueError):
        return None


def fmt(value: str, decimals: int = 2) -> str:
    number = as_number(value)
    return "—" if number is None else f"{number:.{decimals}f}"


def checkpoint_label(row: dict[str, str]) -> str:
    checkpoint = row.get("checkpoint", "").strip()
    if not checkpoint:
        return "On-demand / repository-managed weights"
    try:
        return str(Path(checkpoint).resolve().relative_to(REPO))
    except ValueError:
        return checkpoint


def main() -> None:
    selected: dict[str, tuple[int, dict[str, str]]] = {}
    source_rows: list[dict[str, object]] = []
    for source_run, source_description, priority in SOURCES:
        summary = BENCH_ROOT / source_run / "inference_speed_summary.csv"
        if not summary.exists():
            raise FileNotFoundError(summary)
        with summary.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        source_rows.append({
            "source_run": source_run,
            "source_description": source_description,
            "rows": len(rows),
            "successful_rows": sum(r.get("status") == "ok" for r in rows),
            "priority": priority,
        })
        for raw in rows:
            if raw.get("status") != "ok":
                continue
            row = {field: raw.get(field, "") for field in raw}
            row["run_identifier"] = trained_run(row.get("notes", ""))
            row["checkpoint"] = checkpoint_label(row)
            row["checkpoint_artifact_mb"] = raw.get("model_file_mb", "")
            row["source_run"] = source_run
            row["source_description"] = source_description
            key = row_key(raw)
            old = selected.get(key)
            if old is None or priority >= old[0]:
                selected[key] = (priority, row)

    rows = [value[1] for value in selected.values()]
    rows.sort(key=lambda r: (r["dataset"], r["task"], r["suite"], r["model"], r["run_identifier"]))
    counts = Counter((row["dataset"], row["task"]) for row in rows)
    expected = {
        ("MDWD", "detection"): 18,
        ("MTSD", "detection"): 57,
        ("MTSD", "attribute"): 18,
        ("MTSD", "prompt"): 6,
    }
    if counts != expected:
        raise RuntimeError(f"Unexpected final coverage: {dict(counts)}; expected {expected}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / "all_model_inference_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    md: list[str] = [
        "# Completed inference benchmark — all models",
        "",
        f"**Coverage:** {len(rows)} unique evaluated model artifacts: "
        f"{counts[('MDWD', 'detection')]} MDWD detectors, "
        f"{counts[('MTSD', 'detection')]} MTSD detectors, "
        f"{counts[('MTSD', 'attribute')]} MTSD attribute classifiers, and "
        f"{counts[('MTSD', 'prompt')]} PromptDetect models.",
        "",
        "Each row is one successfully completed benchmark result. Targeted repair runs supersede earlier failed rows for the same checkpoint. Timing is only comparable within the same task/protocol; prompt models use a 10-image sample, all other groups use 50 images. All runs use seed 42, batch size 1, 3 warm-up iterations, CUDA on an NVIDIA GeForce RTX 4090.",
        "",
        "`checkpoint_artifact_mb` is the saved checkpoint artifact, not necessarily the full deployable model footprint. In particular, frozen and LoRA attribute-classifier checkpoints store only learned heads/adapters; use `parameters` to compare their instantiated model scale.",
        "",
        "## Coverage and provenance",
        "",
        "| Dataset | Task | Final artifacts | Protocol |",
        "|---|---:|---:|---|",
    ]
    for (dataset, task), count in sorted(counts.items()):
        protocol = "10 images" if task == "prompt" else "50 images"
        md.append(f"| {dataset} | {task} | {count} | {protocol}; seed 42; batch 1; warm-up 3 |");
    md.extend([
        "",
        "| Source run | Purpose | Successful rows in source |",
        "|---|---|---:|",
    ])
    for source in source_rows:
        md.append(f"| `{source['source_run']}` | {source['source_description']} | {source['successful_rows']} |")

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["task"])].append(row)
    for (dataset, task), grouped_rows in sorted(groups.items()):
        md.extend([
            "",
            f"## {dataset} — {task} ({len(grouped_rows)} models)",
            "",
            "| Model | Suite / trained run | Input | Artifact MB | Parameters | Mean ms | p95 ms | FPS | Cold start s | Peak GPU GB | Source |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for row in grouped_rows:
            suite_run = row["suite"] or row["run_identifier"] or "—"
            if row["suite"] and row["run_identifier"]:
                suite_run += f" / {row['run_identifier']}"
            md.append(
                "| {model} | {suite} | {input_size} | {size} | {params} | {mean} | {p95} | {fps} | {cold} | {gpu} | `{source}` |".format(
                    model=row["model"], suite=suite_run.replace("|", "/"),
                    input_size=row["input_size"] or "—",
                    size=fmt(row["checkpoint_artifact_mb"]),
                    params=(f"{int(float(row['parameters'])):,}" if as_number(row["parameters"]) is not None else "—"),
                    mean=fmt(row["mean_latency_ms"]), p95=fmt(row["p95_latency_ms"]),
                    fps=fmt(row["fps"]), cold=fmt(row["cold_start_s"]),
                    gpu=fmt(row["peak_gpu_mem_gb"]), source=row["source_run"],
                )
            )

    md_path = OUTPUT / "all_model_inference_benchmark.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUTPUT / "benchmark_provenance.json").write_text(
        json.dumps({"sources": source_rows, "final_counts": {f"{k[0]}::{k[1]}": v for k, v in counts.items()}}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print("Coverage:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
