"""Dissertation-ready Markdown report generation.

The report states what was measured and nothing else. Every section is derived
from the run's own artefacts; where a measurement is missing the report says so
and why, rather than omitting the row or filling it in.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def _fmt(value, digits: int = 2, dash: str = "—") -> str:
    if value is None or value == "":
        return dash
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _int(value, dash: str = "—") -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return dash


def _yes_no(value) -> str:
    if value is None:
        return "—"
    return "yes" if value else "no"


def _table(header: list[str], rows: list[list[str]], alignment: str = "") -> list[str]:
    if not rows:
        return []
    align = alignment or ("|" + "|".join(["---"] * len(header)) + "|")
    return ["| " + " | ".join(header) + " |", align] + [
        "| " + " | ".join(row) + " |" for row in rows]


def _platform_section(hardware: dict, software: dict, repository: dict,
                      policy: dict) -> list[str]:
    nvpmodel = hardware.get("nvpmodel") or {}
    storage = repository.get("storage") or {}
    lines = ["## Experimental Platform", ""]
    rows = [
        ["Jetson model", str(hardware.get("jetson_model") or "not detected")],
        ["Architecture", str(hardware.get("architecture"))],
        ["Linux kernel", str(hardware.get("kernel_release"))],
        ["Operating system", str(hardware.get("os_pretty_name") or hardware.get("os_name") or "—")],
        ["L4T release", str(hardware.get("l4t_version") or "not determinable")],
        ["JetPack", str(hardware.get("jetpack_version") or "not determinable from L4T")],
        ["CUDA", str(software.get("cuda_version") or "not determinable")],
        ["cuDNN", str(software.get("cudnn_version") or "not determinable")],
        ["TensorRT", str(software.get("tensorrt_version") or "not determinable")],
        ["PyTorch", str(software.get("torch_version") or "—")],
        ["torchvision", str(software.get("torchvision_version") or "—")],
        ["Ultralytics", str(software.get("ultralytics_version") or "—")],
        ["RF-DETR", str(software.get("rfdetr_version") or "—")],
        ["Python", str(software.get("python_version"))],
        ["CUDA available to PyTorch", _yes_no(software.get("torch_cuda_available"))],
        ["GPU reported by PyTorch", str(software.get("gpu_name") or "—")],
        ["Total system RAM (MB)", _fmt(hardware.get("system_ram_total_mb"), 0)],
        ["Available RAM at start (MB)", _fmt(hardware.get("system_ram_available_mb"), 0)],
        ["nvpmodel mode", f"{nvpmodel.get('nvpmodel_mode')} "
                          f"({nvpmodel.get('nvpmodel_mode_name') or 'name unavailable'})"],
        ["jetson_clocks", _yes_no(policy.get("jetson_clocks_enabled"))],
        ["Repository filesystem", str(storage.get("filesystem") or "not determinable")],
        ["Repository mount point", str(storage.get("mount_point") or "—")],
        ["Free space on that volume (GB)", _fmt(storage.get("free_gb"))],
        ["Git commit", str(repository.get("git_commit_short") or "—")],
        ["Git branch", str(repository.get("git_branch") or "—")],
        ["Working tree", "dirty" if repository.get("git_dirty") else "clean"],
    ]
    lines += _table(["Property", "Value"], rows)
    for warning in repository.get("storage_warnings") or []:
        lines += ["", f"> **Storage warning:** {warning}"]
    if not hardware.get("is_jetson"):
        lines += ["", "> **This run was not executed on a Jetson.** The hardware probe "
                      "reported: " + "; ".join(hardware.get("detection_notes") or []) +
                  ". Latency, power and thermal figures below therefore do not "
                  "describe Jetson edge behaviour."]
    return lines + [""]


def _protocol_section(config: dict, samples: dict, idle: dict, power_source: dict) -> list[str]:
    comparative = config.get("comparative_latency") or {}
    sustained = config.get("sustained_telemetry") or {}
    telemetry = config.get("telemetry") or {}
    cooldown = config.get("cooldown") or {}
    lines = ["## Protocol", "",
             "The experiment runs two logically distinct passes over the same "
             "deterministic inputs.", "",
             "**Pass A — protocol-compatible latency.** Reuses the seeded sample "
             "manifest recorded by the completed RTX 4090 benchmark, at batch size 1 "
             "with the same warm-up count, so the same checkpoint is measured over "
             "the same images on both devices.", "",
             "**Pass B — sustained edge telemetry.** Cycles the same deterministic "
             "sample for a fixed interval so power, energy, memory and thermal "
             "readings are taken over a stable workload rather than a sub-second burst.",
             ""]
    rows = [
        ["Seed", str(config.get("seed"))],
        ["Batch size", str(config.get("batch_size"))],
        ["Pass A repeats", str(comparative.get("repeats"))],
        ["Pass A warm-up", str(comparative.get("warmup"))],
        ["Pass A sample", "workstation manifest reused"
         if comparative.get("reuse_workstation_sample") else "fresh seeded sample"],
        ["Pass B repeats", str(sustained.get("repeats"))],
        ["Pass B minimum measured duration (s)", str(sustained.get("minimum_duration_seconds"))],
        ["Pass B maximum measured duration (s)", str(sustained.get("maximum_duration_seconds"))],
        ["Pass B warm-up items", str(sustained.get("warmup"))],
        ["tegrastats interval (ms)", str(telemetry.get("tegrastats_interval_ms"))],
        ["Idle baseline (s)", str(telemetry.get("idle_baseline_seconds"))],
        ["Cooldown between models", f"min {cooldown.get('min_seconds')}s, "
                                    f"target within {cooldown.get('target_delta_c')} C, "
                                    f"max {cooldown.get('max_seconds')}s"],
        ["Runtime backend", "native PyTorch"
         + (" + optional TensorRT (separate rows)"
            if (config.get("runtime") or {}).get("tensorrt_yolo") else "")],
    ]
    lines += _table(["Setting", "Value"], rows)

    lines += ["", "### Timing boundaries", "",
              "Timing boundaries are never averaged together. Each row records the "
              "boundary it was measured at.", ""]
    lines += _table(
        ["Boundary", "Includes"],
        [["`model_forward`", "the model computation alone, CUDA-synchronised either side"],
         ["`in_memory_pipeline`", "already-decoded input, model preprocessing, inference, postprocessing"],
         ["`end_to_end_file`", "SSD read and decode plus the whole in-memory pipeline"]])
    lines += ["", "The cross-device comparison uses, per engine, the boundary that "
                  "reproduces what the workstation benchmark actually timed: file-path "
                  "input for YOLO, pre-decoded images for RF-DETR, pre-transformed "
                  "tensors for the attribute classifiers and pre-decoded arrays for the "
                  "prompt backends.", ""]

    lines += ["### Sample provenance", ""]
    sample_rows = []
    for key, record in sorted(samples.items()):
        sample_rows.append([
            key, str(record.get("sample_count")), str(record.get("sample_source")),
            str(record.get("workstation_run") or "—"),
            str(record.get("unresolved_entries", 0))])
    lines += _table(["Task", "Items", "Source", "Workstation run", "Unresolved"],
                    sample_rows)
    lines += ["", "Only identifiers, repository-relative paths and hashes are stored; "
                  "no source imagery is copied into this report and no thumbnails are "
                  "generated.", ""]

    lines += ["### Power measurement basis", ""]
    lines += _table(["Property", "Value"], [
        ["Rail selection role", str(power_source.get("power_source_role"))],
        ["Rail(s) used", str(power_source.get("power_source_rail") or "—")],
        ["Is board input power", _yes_no(power_source.get("power_is_board_input"))],
        ["Reason", str(power_source.get("power_source_reason") or "—")],
        ["Idle baseline power (W)", _fmt(idle.get("idle_power_w_mean"))],
        ["Idle baseline peak temperature (C)", _fmt(idle.get("idle_temperature_peak_c"))],
    ])
    if power_source.get("power_source_role") == "unresolved":
        lines += ["", "> No defensible total-power rail was available on this platform, so "
                      "total power and energy fields are null. Per-rail telemetry is still "
                      "retained in `telemetry/`. Board power is not estimated."]
    elif not power_source.get("power_is_board_input"):
        lines += ["", "> Power is the sum of a documented, non-overlapping rail set. It is a "
                      "**module power** figure, not board input power, and is labelled as "
                      "such in every row."]
    return lines + [""]


def _coverage_section(coverage: list[dict]) -> list[str]:
    lines = ["## Requested Model Coverage", "",
             "Every configuration named in the benchmark matrix appears here, "
             "including the ones that could not be executed.", ""]
    rows = []
    for row in coverage:
        rows.append([
            row.get("track", ""),
            row.get("requested_model", ""),
            (row.get("checkpoint_relative_path") or "—"),
            _yes_no(row.get("checkpoint_real")),
            _fmt(row.get("checkpoint_mb"), 1),
            _yes_no(row.get("dataset_available")),
            _yes_no(row.get("environment_usable")),
            str(row.get("smoke_status", "—")),
            str(row.get("benchmark_status", row.get("status", "—"))),
            (row.get("reason") or "")[:160],
        ])
    lines += _table(["Track", "Requested model", "Resolved checkpoint", "Real?",
                     "MB", "Data", "Env", "Smoke", "Benchmark", "Reason if skipped"], rows)
    return lines + [""]


def _results_section(title: str, rows: list[dict], accuracy_label: str,
                     item_unit: str) -> list[str]:
    lines = [f"## {title}", ""]
    if not rows:
        return lines + ["No configuration in this track produced a measurement.", ""]
    table_rows = []
    for row in rows:
        table_rows.append([
            row.get("display_name", row.get("model", "")),
            str(row.get("input_resolution") or "—"),
            _fmt(row.get("predictive_metric_value"), 4),
            _fmt(row.get("latency_mean_ms")),
            _fmt(row.get("latency_p95_ms")),
            _fmt(row.get("throughput_items_s")),
            _fmt(row.get("peak_memory_mb"), 0),
            _fmt(row.get("mean_power_w")),
            _fmt(row.get("energy_j_per_item"), 4),
            _fmt(row.get("temperature_peak_c"), 1),
        ])
    lines += _table(
        ["Model", "Input", accuracy_label, f"Mean ms/{item_unit}", "p95 ms",
         f"{item_unit}/s", "Peak MB", "Mean W", f"J/{item_unit}", "Peak C"],
        table_rows)
    boundary = rows[0].get("timing_boundary")
    lines += ["", f"Latency is per {item_unit} at batch size 1, measured at the "
                  f"`{boundary}` boundary. Peak memory is "
                  f"{rows[0].get('peak_memory_basis', 'unspecified')}.", ""]
    return lines


def _pareto_section(rows: list[dict]) -> list[str]:
    lines = ["## Pareto-Efficient Configurations", "",
             "Frontiers are computed only inside protocol-compatible groups "
             "(same task, same accuracy metric, same timing boundary, same runtime "
             "backend). No frontier is computed across tasks.", ""]
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("predictive_metric_value") is None:
            continue
        groups.setdefault(row.get("pareto_group", "ungrouped"), []).append(row)
    if not groups:
        return lines + ["No group had enough resolved measurements to compute a frontier.", ""]
    for group, grouped in sorted(groups.items()):
        lines += [f"### {group}", ""]
        lines += _table(
            ["Model", "Accuracy", "Mean ms", "J/item", "Peak MB",
             "Pareto (latency)", "Pareto (energy)", "Pareto (memory)"],
            [[row.get("display_name", row.get("model", "")),
              _fmt(row.get("predictive_metric_value"), 4),
              _fmt(row.get("latency_mean_ms")),
              _fmt(row.get("energy_j_per_item"), 4),
              _fmt(row.get("peak_memory_mb"), 0),
              _yes_no(row.get("pareto_latency")),
              _yes_no(row.get("pareto_energy")),
              _yes_no(row.get("pareto_memory"))] for row in grouped])
        lines.append("")
    return lines


def _comparison_section(rows: list[dict]) -> list[str]:
    lines = ["## Workstation vs Jetson", ""]
    valid = [r for r in rows if r.get("comparison_valid")]
    invalid = [r for r in rows if not r.get("comparison_valid")]
    if valid:
        lines += _table(
            ["Model", "Task", "Jetson mean ms", "RTX 4090 mean ms", "Slowdown",
             "Jetson p95", "RTX 4090 p95", "Jetson items/s", "RTX 4090 items/s"],
            [[r.get("display_name", r.get("model", "")), str(r.get("task")),
              _fmt(r.get("jetson_mean_ms")), _fmt(r.get("rtx4090_mean_ms")),
              (f"{r['slowdown_x']:.2f}x" if r.get("slowdown_x") else "—"),
              _fmt(r.get("jetson_p95_ms")), _fmt(r.get("rtx4090_p95_ms")),
              _fmt(r.get("jetson_throughput")), _fmt(r.get("rtx4090_throughput"))]
             for r in valid])
        lines += ["", "These rows share a checkpoint, a seeded sample, batch size 1, a "
                      "timing boundary and an effective input resolution with the "
                      "workstation benchmark.", "",
                  "Memory is deliberately **not** expressed as a ratio: the Jetson figure "
                  "is unified system memory and the RTX 4090 figure is a discrete-VRAM "
                  "allocator peak. They are reported side by side in "
                  "`jetson_vs_workstation.csv`.", ""]
    else:
        lines += ["No configuration satisfied the protocol-compatibility conditions.", ""]
    if invalid:
        lines += ["### Not protocol-comparable", "",
                  "These Jetson measurements are valid in their own right but cannot be "
                  "divided by a workstation number:", ""]
        lines += _table(["Model", "Reason"],
                        [[r.get("display_name", r.get("model", "")),
                          (r.get("comparison_notes") or "")[:200]] for r in invalid])
        lines.append("")
    return lines


def _failures_section(coverage: list[dict], failures: list[dict]) -> list[str]:
    lines = ["## Models That Could Not Be Executed", ""]
    # Configurations that never reached a model load, and configurations that
    # loaded (or tried to) and then failed, are separate stories.
    pre_load = {"missing_checkpoint", "missing_dataset", "dependency_error",
                "ambiguous_checkpoint", "skipped", "pending"}
    blocked = [row for row in coverage if row.get("status") in pre_load]
    runtime = [row for row in failures
               if row.get("status") != "ok" and row.get("stage") != "preflight"]
    if not blocked and not runtime:
        return lines + ["Every requested configuration executed.", ""]
    if blocked:
        lines += ["### Blocked before loading", ""]
        lines += _table(["Model", "Status", "Reason"],
                        [[row.get("requested_model", ""), row.get("status", ""),
                          (row.get("reason") or "")[:300]] for row in blocked])
        lines.append("")
    if runtime:
        lines += ["### Failed during execution", ""]
        lines += _table(["Model", "Stage", "Status", "Error"],
                        [[row.get("display_name", row.get("model_id", "")),
                          row.get("stage", ""), row.get("status", ""),
                          (row.get("error") or "")[:300]] for row in runtime])
        lines.append("")
    return lines


LIMITATIONS = [
    "Exactly one Jetson device was tested. No claim is made about any other "
    "Jetson module, carrier board or thermal solution.",
    "Results apply to the recorded JetPack / L4T / CUDA / PyTorch stack, the "
    "recorded nvpmodel power mode and the recorded jetson_clocks state. A "
    "different power mode changes latency and power materially.",
    "Energy is derived from on-board INA3221 telemetry read through tegrastats, "
    "not from an external laboratory power meter, and inherits that "
    "instrumentation's accuracy.",
    "The tegrastats sampling interval bounds how finely power can be attributed; "
    "sub-interval transients are not resolved.",
    "Model-forward, in-memory-pipeline and end-to-end boundaries measure "
    "different things and are reported in separate columns; they must not be "
    "compared with each other.",
    "End-to-end figures include SSD reads that benefit from OS page caching. No "
    "privileged cache flushing was performed between images.",
    "Zero-shot results are per image-prompt pair. One query does not return the "
    "whole municipal taxonomy, so these numbers are not comparable to a "
    "supervised detector's per-image latency.",
    "Attribute classification is benchmarked on ground-truth crops. It therefore "
    "does not measure a complete municipal detector -> crop -> classifier "
    "pipeline, whose cost would be the detector plus the classifier per instance.",
    "A LoRA checkpoint's file size is the adapter only. The instantiated model "
    "includes the pretrained backbone, so `parameters_total` and the measured "
    "memory - not `checkpoint_mb` - describe the deployed footprint.",
    "Accuracy values are the repository's existing stored test metrics for the "
    "exact benchmarked checkpoint. No model was re-evaluated and no model was "
    "retrained for this experiment.",
    "TensorRT results, when present, are a separate optimised runtime and carry "
    "`runtime_backend=tensorrt`. They are never mixed into the native PyTorch "
    "comparison.",
]


def build_report(context: dict) -> str:
    """Assemble the full Markdown report from one run's artefacts."""
    lines = [
        "# NVIDIA Jetson Edge-Device Benchmark",
        "",
        f"Run `{context['run_id']}` — generated "
        f"{datetime.now().isoformat(timespec='seconds')}.",
        "",
        "Inference and deployment measurement only. No model was trained or "
        "re-evaluated, and no dataset, annotation, trained run or existing "
        "evaluation artefact was modified.",
        "",
    ]
    lines += _platform_section(context["hardware"], context["software"],
                               context["repository"], context.get("policy", {}))
    lines += _protocol_section(context["config"], context["samples"],
                               context.get("idle", {}), context.get("power_source", {}))
    lines += _coverage_section(context["coverage"])

    deployment = context["deployment"]
    ok = [r for r in deployment if r.get("status") == "ok"]
    lines += _results_section(
        "Supervised Detection — MDWD",
        [r for r in ok if r.get("dataset") == "MDWD" and r.get("task") == "detection"],
        "test mAP50-95", "image")
    lines += _results_section(
        "Supervised Detection — MTSD",
        [r for r in ok if r.get("dataset") == "MTSD" and r.get("task") == "detection"],
        "test mAP50-95", "image")
    lines += _results_section(
        "Attribute Classification",
        [r for r in ok if r.get("task") == "attribute"],
        "test mean macro-F1", "crop")
    lines += _results_section(
        "Optional Zero-Shot Localisation",
        [r for r in ok if r.get("task") == "prompt"],
        "macro mean F1", "image-prompt pair")

    lines += ["## Accuracy–Latency and Accuracy–Energy Trade-offs", "",
              "Per-task figures are written to `figures/`. Each task keeps its own "
              "accuracy axis: MDWD and MTSD use test mAP50-95, attribute "
              "classification uses four-head mean macro-F1, and prompt localisation "
              "uses macro mean F1 under the existing prompt protocol. These are not "
              "interchangeable and are never plotted on a shared axis.", ""]
    figure_rows = [[f.get("figure", ""), _yes_no(f.get("generated")),
                    (f.get("reason") or "")[:140]] for f in context.get("figures", [])]
    lines += _table(["Figure", "Generated", "Reason if skipped"], figure_rows)
    lines.append("")

    lines += ["## Memory Requirements", "",
              "Jetson memory is unified with the GPU, so PyTorch allocator figures are "
              "**not** the total device requirement. Both are reported:", ""]
    lines += _table(
        ["Model", "torch peak allocated MB", "torch peak reserved MB",
         "system RAM peak MB", "system RAM model delta MB", "Checkpoint MB", "Parameters"],
        [[r.get("display_name", r.get("model", "")),
          _fmt(r.get("torch_peak_allocated_mb"), 1),
          _fmt(r.get("torch_peak_reserved_mb"), 1),
          _fmt(r.get("system_ram_peak_mb"), 1),
          _fmt(r.get("system_ram_model_delta_mb"), 1),
          _fmt(r.get("checkpoint_mb"), 1),
          _int(r.get("parameters_total"))]
         for r in context.get("memory_rows", [])])
    lines.append("")

    lines += ["## Thermal Behaviour", ""]
    lines += _table(
        ["Model", "Start C", "Mean C", "Peak C", "End C", "Cooldown s", "Cooldown outcome"],
        [[r.get("display_name", r.get("model_id", "")),
          _fmt(r.get("temperature_start_c"), 1), _fmt(r.get("temperature_mean_c"), 1),
          _fmt(r.get("temperature_peak_c"), 1), _fmt(r.get("temperature_end_c"), 1),
          _fmt(r.get("cooldown_seconds"), 1), str(r.get("cooldown_outcome") or "—")]
         for r in context.get("thermal_rows", [])])
    lines.append("")

    lines += _comparison_section(context["comparison"])
    lines += _pareto_section(deployment)
    lines += _failures_section(context["coverage"], context.get("failures", []))

    lines += ["## Deployment Interpretation", ""]
    lines += context.get("interpretation", ["No interpretation could be derived from "
                                            "the available measurements."])
    lines.append("")

    lines += ["## Measurement Limitations", ""]
    lines += [f"{index}. {text}" for index, text in enumerate(LIMITATIONS, start=1)]
    lines.append("")

    lines += ["## Reproducibility Manifest", "",
              "Every file below is written by this run and is sufficient to recompute "
              "the aggregates, including confidence intervals, from the raw "
              "observations.", ""]
    lines += _table(["Artefact", "Contents"], [
        ["`benchmark_config.json`", "the effective configuration, including profile and CLI overrides"],
        ["`hardware_snapshot.json`", "Jetson module, L4T/JetPack, memory, power mode, thermal zones"],
        ["`software_snapshot.json`", "CUDA / cuDNN / TensorRT / PyTorch / library versions"],
        ["`repository_snapshot.json`", "commit, branch, dirty state, SSD filesystem and free space"],
        ["`requested_coverage.csv`", "one row per requested configuration with its outcome"],
        ["`model_manifest.csv`", "the exact checkpoint each user-facing request resolved to"],
        ["`sample_manifest.csv`", "the exact inputs measured, by repository-relative path"],
        ["`latency_summary.csv`", "one row per model / pass / repeat / timing boundary"],
        ["`telemetry_summary.csv`", "per-window power, energy, memory, utilisation, thermal"],
        ["`deployment_summary.csv`", "the compact per-configuration deployment view"],
        ["`raw_timings.csv.gz`", "every individual per-item latency observation"],
        ["`jetson_vs_workstation.csv`", "the cross-device comparison and its validity flags"],
        ["`failures.csv`", "every failure with a structured status and message"],
        ["`telemetry/`", "raw tegrastats lines and parsed samples"],
        ["`environment_freeze/`", "installed package lists for each environment used"],
        ["`logs/`", "per-model worker logs"],
        ["`figures/`", "generated figures plus the manifest of skipped ones"],
    ])
    lines.append("")
    return "\n".join(lines)


def derive_interpretation(deployment: list[dict], comparison: list[dict]) -> list[str]:
    """Factual observations derived only from this run's measurements."""
    statements: list[str] = []
    ok = [r for r in deployment if r.get("status") == "ok"]
    if not ok:
        return ["No configuration completed, so no deployment conclusion can be drawn."]

    for dataset, task, label in (("MDWD", "detection", "MDWD detection"),
                                 ("MTSD", "detection", "MTSD detection"),
                                 ("MTSD", "attribute", "MTSD attribute classification")):
        group = [r for r in ok if r.get("dataset") == dataset and r.get("task") == task
                 and r.get("latency_mean_ms")]
        if len(group) < 2:
            continue
        fastest = min(group, key=lambda r: r["latency_mean_ms"])
        slowest = max(group, key=lambda r: r["latency_mean_ms"])
        unit = fastest.get("item_unit", "item")
        statements.append(
            f"- For {label}, measured Jetson latency spans "
            f"{fastest['latency_mean_ms']:.1f} ms/{unit} "
            f"({fastest.get('display_name')}) to {slowest['latency_mean_ms']:.1f} "
            f"ms/{unit} ({slowest.get('display_name')}), a "
            f"{slowest['latency_mean_ms'] / fastest['latency_mean_ms']:.1f}x range.")
        scored = [r for r in group if r.get("predictive_metric_value") is not None]
        if len(scored) >= 2:
            best = max(scored, key=lambda r: r["predictive_metric_value"])
            cheapest = min(scored, key=lambda r: r["latency_mean_ms"])
            if best is not cheapest:
                delta = best["predictive_metric_value"] - cheapest["predictive_metric_value"]
                cost = best["latency_mean_ms"] / cheapest["latency_mean_ms"]
                statements.append(
                    f"  The most accurate {label} configuration "
                    f"({best.get('display_name')}, {best['predictive_metric_value']:.4f}) "
                    f"is {delta:+.4f} on the stored test metric relative to the fastest "
                    f"({cheapest.get('display_name')}) and costs {cost:.1f}x its "
                    f"Jetson latency.")
            else:
                statements.append(
                    f"  For {label} the most accurate stored configuration is also the "
                    f"fastest measured one, so no accuracy/latency trade-off arises "
                    f"within the benchmarked set.")

    energetic = [r for r in ok if r.get("energy_j_per_item") is not None]
    if energetic:
        cheapest = min(energetic, key=lambda r: r["energy_j_per_item"])
        priciest = max(energetic, key=lambda r: r["energy_j_per_item"])
        statements.append(
            f"- Measured energy per processed item ranges from "
            f"{cheapest['energy_j_per_item']:.4f} J ({cheapest.get('display_name')}) to "
            f"{priciest['energy_j_per_item']:.4f} J ({priciest.get('display_name')}).")
    else:
        statements.append(
            "- No energy figure could be derived on this platform, so no "
            "energy-based deployment statement is made.")

    valid = [r for r in comparison if r.get("comparison_valid") and r.get("slowdown_x")]
    if valid:
        factors = [r["slowdown_x"] for r in valid]
        statements.append(
            f"- Across the {len(valid)} protocol-compatible configuration(s), the Jetson "
            f"is {min(factors):.1f}x to {max(factors):.1f}x slower than the RTX 4090 "
            f"workstation under an identical batch-1 protocol.")
    else:
        statements.append(
            "- No configuration satisfied every protocol-compatibility condition, so no "
            "workstation slowdown factor is reported.")

    efficient = [r for r in ok if r.get("pareto_latency")]
    if efficient:
        statements.append(
            "- Pareto-efficient on accuracy vs latency within their own task group: "
            + ", ".join(sorted({str(r.get("display_name")) for r in efficient})) + ".")
    return statements
