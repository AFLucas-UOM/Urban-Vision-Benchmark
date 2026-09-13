"""Power-rail resolution, energy integration and telemetry window summaries.

Jetson boards expose several INA3221 rails whose relationships differ per
platform. Some rails are hierarchical, so blindly summing every reported rail
double-counts power. This module therefore resolves a defensible power source
per platform and records *which* rail (or documented rail set) was used:

``board_input_rail``
    The platform exposes a genuine total-input rail (``VDD_IN`` on Orin Nano /
    Xavier NX carriers, ``POM_5V_IN`` on older Nano carriers). This is board
    input power.

``documented_disjoint_rail_sum``
    The platform exposes a known, documented, non-overlapping rail set whose
    sum is the module power (the AGX Orin trio ``VDD_GPU_SOC`` +
    ``VDD_CPU_CV`` + ``VIN_SYS_5V0`` is the common case). This is reported as a
    module-power estimate and is explicitly *not* labelled board input.

``unresolved``
    Nothing defensible is available. Per-rail telemetry is still retained, and
    total power / energy fields are left null with a recorded reason. No board
    power is ever fabricated.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from .tegrastats import TOTAL_RAIL_NAMES, TelemetrySample

# Rail sets that are known to be disjoint on their platform, so their sum is a
# meaningful module-power figure. Keys are frozensets of rail names.
DOCUMENTED_DISJOINT_RAIL_SETS: dict[frozenset[str], str] = {
    frozenset({"VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0"}):
        "AGX Orin module rails (GPU+SOC, CPU+CV, 5V system)",
    frozenset({"VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0", "NC"}):
        "AGX Orin module rails (GPU+SOC, CPU+CV, 5V system; NC unused)",
    frozenset({"VDD_CPU_GPU_CV", "VDD_SOC"}):
        "Orin/Xavier module rails (CPU+GPU+CV, SOC)",
    frozenset({"POM_5V_GPU", "POM_5V_CPU"}):
        "Jetson Nano module rails (5V GPU, 5V CPU)",
}

# Rails that are always ignored when forming a sum (unpopulated / reserved).
IGNORED_RAILS = {"NC"}


@dataclass(frozen=True)
class PowerSource:
    role: str                  # board_input_rail | documented_disjoint_rail_sum | unresolved
    rails: tuple[str, ...]
    label: str                 # human-readable rail expression, e.g. "VDD_IN"
    is_board_input: bool
    reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "power_source_role": self.role,
            "power_source_rail": self.label,
            "power_source_rails": list(self.rails),
            "power_is_board_input": self.is_board_input,
            "power_source_reason": self.reason,
        }


UNRESOLVED = PowerSource("unresolved", (), "", False,
                         "no board-input rail and no documented disjoint rail set")


def resolve_power_source(rail_names, strategy: str = "auto",
                         explicit_rails=None) -> PowerSource:
    """Choose which rail(s) represent the platform's power draw.

    ``rail_names`` is the set of rails ``tegrastats`` actually reported on this
    device. The result is recorded verbatim into every benchmark row so a
    reader always knows what ``mean_power_w`` means.
    """
    available = {name for name in rail_names if name not in IGNORED_RAILS}
    if not available:
        return PowerSource("unresolved", (), "", False, "no power rails reported")

    strategy = (strategy or "auto").lower()
    if strategy == "explicit":
        wanted = [r for r in (explicit_rails or [])]
        if not wanted:
            return PowerSource("unresolved", (), "", False,
                               "power.rail_strategy=explicit but power.rails is empty")
        missing = [r for r in wanted if r not in rail_names]
        if missing:
            return PowerSource("unresolved", (), "", False,
                               f"configured rails not reported by tegrastats: {missing}")
        board = len(wanted) == 1 and wanted[0] in TOTAL_RAIL_NAMES
        return PowerSource("explicit_rail_selection", tuple(wanted), "+".join(wanted),
                           board, "operator-selected rails")

    for candidate in TOTAL_RAIL_NAMES:
        if candidate in available:
            return PowerSource("board_input_rail", (candidate,), candidate, True,
                               "platform exposes a total board-input rail")

    if strategy == "total_rail_only":
        return PowerSource("unresolved", (), "", False,
                           "power.rail_strategy=total_rail_only and this platform "
                           "exposes no total board-input rail")

    # Match against the rails that carry power (ignored rails excluded) and,
    # as a fallback, against everything tegrastats reported.
    description = (DOCUMENTED_DISJOINT_RAIL_SETS.get(frozenset(available))
                   or DOCUMENTED_DISJOINT_RAIL_SETS.get(frozenset(rail_names)))
    if description:
        rails = tuple(sorted(available))
        return PowerSource("documented_disjoint_rail_sum", rails, "+".join(rails), False,
                           f"sum of {description}; module power, not board input")

    return PowerSource("unresolved", tuple(sorted(available)), "", False,
                       "rail set is not a recognised total rail or documented "
                       "disjoint set; rails may overlap, so they are not summed")


def sample_power_w(sample: TelemetrySample, source: PowerSource) -> float | None:
    """Instantaneous watts for one telemetry sample under the chosen source."""
    rails = sample.parsed.get("power_rails_mw") or {}
    if source.role == "unresolved" or not rails:
        return None
    values = [rails[name] for name in source.rails if name in rails]
    if not values:
        return None
    return sum(values) / 1000.0


def integrate_energy_j(samples: list[TelemetrySample], source: PowerSource) -> float | None:
    """Trapezoidal integral of P(t) over the sampled window, in joules.

    Returns ``None`` when fewer than two usable samples exist - a single
    telemetry sample cannot support an energy figure and is not extrapolated.
    """
    series = [(s.monotonic, sample_power_w(s, source)) for s in samples]
    series = [(t, p) for t, p in series if p is not None]
    if len(series) < 2:
        return None
    energy = 0.0
    for (t0, p0), (t1, p1) in zip(series, series[1:]):
        energy += (p1 + p0) / 2.0 * (t1 - t0)
    return round(energy, 4)


def _stats(values: list[float], prefix: str, ndigits: int = 3) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, round(0.95 * (len(ordered) - 1))))
    return {
        f"{prefix}_mean": round(statistics.fmean(values), ndigits),
        f"{prefix}_median": round(statistics.median(values), ndigits),
        f"{prefix}_p95": round(ordered[p95_index], ndigits),
        f"{prefix}_peak": round(max(values), ndigits),
        f"{prefix}_min": round(min(values), ndigits),
    }


def summarise_window(samples: list[TelemetrySample], source: PowerSource,
                     items_processed: int | None = None,
                     idle_power_w: float | None = None) -> dict:
    """Aggregate one measured interval's telemetry into reportable fields.

    ``energy_j_per_item`` is the *gross* measured energy divided by the number
    of processed items. ``dynamic_energy_j_per_item`` is reported separately
    and never replaces it: it subtracts the recorded idle baseline power over
    the same interval, which attributes only the additional draw caused by
    inference to the workload.
    """
    summary: dict = {"telemetry_samples": len(samples)}
    summary.update(source.as_dict())
    if not samples:
        summary["telemetry_status"] = "no_samples"
        return summary
    summary["telemetry_status"] = "ok"
    summary["window_seconds"] = round(samples[-1].monotonic - samples[0].monotonic, 4)

    powers = [p for p in (sample_power_w(s, source) for s in samples) if p is not None]
    summary.update(_stats(powers, "power_w"))

    per_rail: dict[str, list[float]] = {}
    for sample in samples:
        for name, milliwatts in (sample.parsed.get("power_rails_mw") or {}).items():
            per_rail.setdefault(name, []).append(milliwatts / 1000.0)
    if per_rail:
        summary["power_rail_mean_w"] = {name: round(statistics.fmean(v), 3)
                                        for name, v in sorted(per_rail.items())}
        summary["power_rail_peak_w"] = {name: round(max(v), 3)
                                        for name, v in sorted(per_rail.items())}

    energy = integrate_energy_j(samples, source)
    summary["energy_j"] = energy
    if energy is not None and items_processed:
        summary["energy_j_per_item"] = round(energy / items_processed, 5)
    else:
        summary["energy_j_per_item"] = None

    if (energy is not None and items_processed and idle_power_w is not None
            and summary.get("window_seconds")):
        dynamic = energy - idle_power_w * summary["window_seconds"]
        summary["idle_baseline_power_w"] = round(idle_power_w, 3)
        summary["dynamic_energy_j"] = round(dynamic, 4)
        summary["dynamic_energy_j_per_item"] = round(dynamic / items_processed, 5)
        summary["dynamic_energy_method"] = (
            "gross energy minus (recorded idle baseline power x measured window "
            "seconds); reported in addition to, never instead of, gross energy")
    else:
        summary["dynamic_energy_j_per_item"] = None

    gpu = [s.parsed["gpu_util_pct"] for s in samples if "gpu_util_pct" in s.parsed]
    summary.update(_stats(gpu, "gpu_util", 2))
    cpu = [s.parsed["cpu_util_mean_pct"] for s in samples if "cpu_util_mean_pct" in s.parsed]
    summary.update(_stats(cpu, "cpu_util", 2))
    gpu_freq = [s.parsed["gpu_freq_mhz"] for s in samples if "gpu_freq_mhz" in s.parsed]
    summary.update(_stats(gpu_freq, "gpu_freq_mhz", 1))
    emc = [s.parsed["emc_util_pct"] for s in samples if "emc_util_pct" in s.parsed]
    summary.update(_stats(emc, "emc_util", 2))

    ram = [s.parsed["ram_used_mb"] for s in samples if "ram_used_mb" in s.parsed]
    if ram:
        summary["system_ram_mean_mb"] = round(statistics.fmean(ram), 1)
        summary["system_ram_peak_mb"] = round(max(ram), 1)
        summary["system_ram_min_mb"] = round(min(ram), 1)
    totals = [s.parsed["ram_total_mb"] for s in samples if "ram_total_mb" in s.parsed]
    if totals:
        summary["system_ram_total_mb"] = round(max(totals), 1)
    swap = [s.parsed["swap_used_mb"] for s in samples if "swap_used_mb" in s.parsed]
    if swap:
        summary["swap_peak_mb"] = round(max(swap), 1)

    zone_series: dict[str, list[float]] = {}
    for sample in samples:
        for zone, value in (sample.parsed.get("temperatures_c") or {}).items():
            zone_series.setdefault(zone, []).append(value)
    if zone_series:
        summary["temperature_zone_mean_c"] = {z: round(statistics.fmean(v), 2)
                                              for z, v in sorted(zone_series.items())}
        summary["temperature_zone_peak_c"] = {z: round(max(v), 2)
                                              for z, v in sorted(zone_series.items())}
        overall = [s.parsed["temperature_max_c"] for s in samples
                   if "temperature_max_c" in s.parsed]
        if overall:
            summary["temperature_mean_c"] = round(statistics.fmean(overall), 2)
            summary["temperature_peak_c"] = round(max(overall), 2)
            summary["temperature_start_c"] = overall[0]
            summary["temperature_end_c"] = overall[-1]
    return summary


def idle_baseline(samples: list[TelemetrySample], source: PowerSource) -> dict:
    """Idle power / temperature baseline recorded before the model suite runs."""
    summary = summarise_window(samples, source)
    return {
        "idle_samples": summary.get("telemetry_samples", 0),
        "idle_power_w_mean": summary.get("power_w_mean"),
        "idle_power_w_peak": summary.get("power_w_peak"),
        "idle_temperature_mean_c": summary.get("temperature_mean_c"),
        "idle_temperature_peak_c": summary.get("temperature_peak_c"),
        "idle_temperature_zone_mean_c": summary.get("temperature_zone_mean_c"),
        "idle_system_ram_mean_mb": summary.get("system_ram_mean_mb"),
        "idle_gpu_util_mean": summary.get("gpu_util_mean"),
        **source.as_dict(),
    }
