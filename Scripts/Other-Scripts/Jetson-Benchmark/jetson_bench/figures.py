"""Dissertation-ready figures, generated headlessly.

Matplotlib runs on the ``Agg`` backend, so no display server is required. A
figure whose measurement is unavailable is skipped cleanly and the reason is
recorded in ``figures/figure_manifest.json`` rather than emitting an empty
plot.

Task metrics with different semantics are never combined on one accuracy axis:
MDWD mAP50-95, MTSD mAP50-95, attribute mean macro-F1 and prompt macro-mean F1
each get their own figure.
"""

from __future__ import annotations

from pathlib import Path

FIGURE_GROUPS = [
    # (dataset, task, filename prefix, axis label)
    ("MDWD", "detection", "MDWD", "MDWD test mAP50-95"),
    ("MTSD", "detection", "MTSD", "MTSD test mAP50-95"),
    ("MTSD", "attribute", "Attribute", "Attribute test mean macro-F1"),
    ("MTSD", "prompt", "ZeroShot", "Prompt-localisation macro mean F1"),
]

TRADEOFF_AXES = [
    ("latency_mean_ms", "Mean latency (ms)", "accuracy_vs_latency", "pareto_latency"),
    ("energy_j_per_item", "Energy per item (J)", "accuracy_vs_energy", "pareto_energy"),
    ("peak_memory_mb", "Peak memory (MB)", "accuracy_vs_memory", "pareto_memory"),
]


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _short_label(row: dict) -> str:
    name = row.get("display_name") or row.get("model_id") or "?"
    resolution = row.get("input_resolution")
    if resolution and row.get("task") == "detection":
        name = f"{name}"
    return name.replace("MDWD ", "").replace("MTSD ", "")


def _annotate(ax, xs, ys, labels) -> None:
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4),
                    fontsize=7.5, alpha=0.9)


def _tradeoff_figure(rows: list[dict], x_key: str, x_label: str, y_label: str,
                     pareto_key: str, title: str, output: Path) -> dict:
    usable = [r for r in rows
              if r.get("predictive_metric_value") is not None and r.get(x_key) is not None]
    if len(usable) < 2:
        return {"figure": output.name, "generated": False,
                "reason": f"needs at least two configurations with both "
                          f"{x_key} and a resolved predictive metric; have {len(usable)}"}
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    xs = [float(r[x_key]) for r in usable]
    ys = [float(r["predictive_metric_value"]) for r in usable]
    efficient = [bool(r.get(pareto_key)) for r in usable]
    ax.scatter([x for x, e in zip(xs, efficient) if not e],
               [y for y, e in zip(ys, efficient) if not e],
               s=58, color="#7a8796", label="dominated", zorder=3)
    ax.scatter([x for x, e in zip(xs, efficient) if e],
               [y for y, e in zip(ys, efficient) if e],
               s=88, color="#2a78d6", edgecolor="#12325c", linewidth=1.1,
               label="Pareto-efficient", zorder=4)
    frontier = sorted(((x, y) for x, y, e in zip(xs, ys, efficient) if e))
    if len(frontier) > 1:
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier],
                color="#2a78d6", alpha=0.45, linewidth=1.3, zorder=2)
    _annotate(ax, xs, ys, [_short_label(r) for r in usable])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return {"figure": output.name, "generated": True, "points": len(usable)}


def _bar_figure(rows: list[dict], value_key: str, y_label: str, title: str,
                output: Path, colour: str = "#1baf7a") -> dict:
    usable = [r for r in rows if r.get(value_key) is not None]
    if not usable:
        return {"figure": output.name, "generated": False,
                "reason": f"no configuration produced {value_key}"}
    plt = _pyplot()
    labels = [_short_label(r) for r in usable]
    values = [float(r[value_key]) for r in usable]
    fig, ax = plt.subplots(figsize=(max(6.5, 0.62 * len(labels) + 2.5), 4.4))
    ax.bar(labels, values, color=colour)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="x", rotation=38, labelsize=8)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return {"figure": output.name, "generated": True, "bars": len(usable)}


def _comparison_figure(rows: list[dict], output: Path) -> dict:
    usable = [r for r in rows if r.get("comparison_valid")
              and r.get("jetson_mean_ms") and r.get("rtx4090_mean_ms")]
    if not usable:
        return {"figure": output.name, "generated": False,
                "reason": "no protocol-comparable model pairs were measured"}
    plt = _pyplot()
    labels = [r.get("display_name") or r.get("model") for r in usable]
    jetson = [float(r["jetson_mean_ms"]) for r in usable]
    workstation = [float(r["rtx4090_mean_ms"]) for r in usable]
    positions = range(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(7.0, 0.72 * len(labels) + 2.5), 4.8))
    ax.bar([p - width / 2 for p in positions], workstation, width,
           label="RTX 4090 (workstation)", color="#7a8796")
    ax.bar([p + width / 2 for p in positions], jetson, width,
           label="Jetson", color="#2a78d6")
    for position, (j, w) in enumerate(zip(jetson, workstation)):
        ax.annotate(f"{j / w:.1f}x", (position, max(j, w)), ha="center",
                    textcoords="offset points", xytext=(0, 4), fontsize=8)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=8)
    ax.set_ylabel("Mean latency (ms), batch 1")
    ax.set_title("Protocol-compatible latency: workstation vs Jetson\n"
                 "(same checkpoint, same seeded sample, same timing boundary)",
                 fontsize=10.5)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return {"figure": output.name, "generated": True, "pairs": len(usable)}


def generate(deployment_rows: list[dict], comparison_rows: list[dict],
             figures_dir: Path) -> list[dict]:
    """Render every figure the measurements support; return a manifest."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    try:
        _pyplot()
    except Exception as exc:
        return [{"figure": "*", "generated": False,
                 "reason": f"matplotlib unavailable: {type(exc).__name__}: {exc}"}]

    ok_rows = [r for r in deployment_rows if r.get("status") == "ok"]
    for dataset, task, prefix, axis_label in FIGURE_GROUPS:
        group = [r for r in ok_rows if r.get("dataset") == dataset and r.get("task") == task]
        for x_key, x_label, suffix, pareto_key in TRADEOFF_AXES:
            output = figures_dir / f"{prefix}_{suffix}.png"
            if not group:
                manifest.append({"figure": output.name, "generated": False,
                                 "reason": f"no successful {dataset} {task} measurements"})
                continue
            manifest.append(_tradeoff_figure(
                group, x_key, x_label, axis_label, pareto_key,
                f"{dataset} {task}: accuracy vs {x_label.lower()} (Jetson)", output))

    manifest.append(_comparison_figure(comparison_rows,
                                       figures_dir / "workstation_vs_jetson_latency.png"))
    manifest.append(_bar_figure(ok_rows, "mean_power_w", "Mean power (W)",
                                "Jetson mean power during sustained inference",
                                figures_dir / "jetson_power_by_model.png", "#d2762a"))
    manifest.append(_bar_figure(ok_rows, "peak_memory_mb", "Peak memory (MB)",
                                "Jetson peak memory during inference",
                                figures_dir / "jetson_memory_by_model.png", "#5b52a3"))
    manifest.append(_bar_figure(ok_rows, "temperature_peak_c", "Peak temperature (C)",
                                "Jetson peak temperature during sustained inference",
                                figures_dir / "jetson_peak_temperature_by_model.png", "#b3453c"))
    manifest.append(_bar_figure(ok_rows, "energy_j_per_item", "Energy per item (J)",
                                "Jetson energy per processed item",
                                figures_dir / "jetson_energy_by_model.png", "#2f8f6b"))
    return manifest
