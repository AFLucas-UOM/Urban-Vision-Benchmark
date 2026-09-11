"""``tegrastats`` parsing and background sampling.

Jetson telemetry is *not* ``nvidia-smi``. The platform-native tool is
``tegrastats``, and its line format differs between Tegra generations (Nano,
TX2, Xavier, Orin) and between L4T releases: rails are named differently, some
boards report ``mW`` suffixes and some do not, thermal zones come and go, and
several fields (``IRAM``, ``APE``, ``NVDEC``) only exist on some devices.

The parser is therefore field-by-field and tolerant: any field that is absent
is simply missing from the parsed sample, never an exception. The raw line is
always retained alongside the parsed values so nothing this parser fails to
understand is lost.

Representative lines this module is tested against live in
``tests/test_tegrastats_parser.py``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

TEGRASTATS_CANDIDATES = ("/usr/bin/tegrastats", "/home/nvidia/tegrastats", "tegrastats")

# Names that use the `NAME value/value` shape but are not power rails.
NON_POWER_KEYS = {
    "RAM", "SWAP", "IRAM", "EMC", "EMC_FREQ", "GR3D", "GR3D_FREQ", "GR3D2_FREQ",
    "APE", "NVENC", "NVDEC", "NVJPG", "NVJPG1", "OFA", "SE", "VIC", "MSENC",
    "CPU", "MTS", "VOLT", "cached", "lfb",
}
# Rails that carry total board / module input power on the platforms that have
# one. Ordered by preference.
TOTAL_RAIL_NAMES = ("VDD_IN", "POM_5V_IN", "VDD_SYS_IN", "SYS_IN", "TOTAL",
                    "POM_5V_IN_MAIN")

_RAM_RE = re.compile(r"\bRAM (?P<used>\d+)/(?P<total>\d+)(?P<unit>[kKmMgG])B")
_SWAP_RE = re.compile(r"\bSWAP (?P<used>\d+)/(?P<total>\d+)(?P<unit>[kKmMgG])B")
_LFB_RE = re.compile(r"\(lfb (?P<blocks>\d+)x(?P<size>\d+)(?P<unit>[kKmMgG])B\)")
_CPU_RE = re.compile(r"\bCPU \[(?P<body>[^\]]*)\]")
_CPU_CORE_RE = re.compile(r"(?P<util>\d+)%@(?P<freq>\d+)|(?P<off>off)")
_GR3D_RE = re.compile(r"\bGR3D(?P<index>\d*)_FREQ (?P<util>\d+)%(?:@\[?(?P<freq>\d+)\]?)?")
_EMC_RE = re.compile(r"\bEMC_FREQ (?P<util>\d+)%(?:@(?P<freq>\d+))?")
_TEMP_RE = re.compile(r"(?<![\w@])(?P<zone>[A-Za-z][A-Za-z0-9_]*)@(?P<value>-?\d+(?:\.\d+)?)C")
_POWER_RE = re.compile(
    r"(?<![\w@/])(?P<name>[A-Z][A-Z0-9_]*) (?P<cur>\d+)(?P<unit>mW)?/(?P<avg>\d+)(?:mW)?(?![\w.]|/)"
)
_TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})")

_UNIT_TO_MB = {"k": 1 / 1024, "m": 1.0, "g": 1024.0}


def _to_mb(value: str, unit: str) -> float:
    return round(float(value) * _UNIT_TO_MB[unit.lower()], 3)


def parse_tegrastats_line(line: str) -> dict:
    """Parse one ``tegrastats`` line into a flat dictionary.

    Unknown or absent fields are simply omitted. ``power_rails_mw`` maps rail
    name -> instantaneous milliwatts; ``temperatures_c`` maps thermal-zone name
    -> degrees Celsius. Sensors reporting the Jetson "disabled" sentinel
    (-256 C) are dropped rather than reported as a real reading.
    """
    sample: dict = {"raw": line.rstrip("\n")}
    if not line or not line.strip():
        return sample

    stamp = _TIMESTAMP_RE.match(line.strip())
    if stamp:
        sample["device_timestamp"] = stamp.group("stamp")

    ram = _RAM_RE.search(line)
    if ram:
        sample["ram_used_mb"] = _to_mb(ram.group("used"), ram.group("unit"))
        sample["ram_total_mb"] = _to_mb(ram.group("total"), ram.group("unit"))

    lfb = _LFB_RE.search(line)
    if lfb:
        sample["lfb_blocks"] = int(lfb.group("blocks"))
        sample["lfb_block_mb"] = _to_mb(lfb.group("size"), lfb.group("unit"))

    swap = _SWAP_RE.search(line)
    if swap:
        sample["swap_used_mb"] = _to_mb(swap.group("used"), swap.group("unit"))
        sample["swap_total_mb"] = _to_mb(swap.group("total"), swap.group("unit"))

    cpu = _CPU_RE.search(line)
    if cpu:
        utils, freqs, offline = [], [], 0
        for token in cpu.group("body").split(","):
            token = token.strip()
            if not token:
                continue
            if token.lower() == "off":
                offline += 1
                continue
            match = _CPU_CORE_RE.match(token)
            if match and match.group("util") is not None:
                utils.append(float(match.group("util")))
                freqs.append(float(match.group("freq")))
            elif re.fullmatch(r"\d+%", token):
                utils.append(float(token[:-1]))
        if utils:
            sample["cpu_util_mean_pct"] = round(sum(utils) / len(utils), 2)
            sample["cpu_util_max_pct"] = max(utils)
            sample["cpu_online_cores"] = len(utils)
        if freqs:
            sample["cpu_freq_mean_mhz"] = round(sum(freqs) / len(freqs), 1)
            sample["cpu_freq_max_mhz"] = max(freqs)
        sample["cpu_offline_cores"] = offline

    gpus = list(_GR3D_RE.finditer(line))
    if gpus:
        utils = [float(m.group("util")) for m in gpus]
        sample["gpu_util_pct"] = max(utils)
        freqs = [float(m.group("freq")) for m in gpus if m.group("freq")]
        if freqs:
            sample["gpu_freq_mhz"] = max(freqs)
    else:
        plain = re.search(r"\bGR3D_FREQ (?P<util>\d+)%", line)
        if plain:
            sample["gpu_util_pct"] = float(plain.group("util"))

    emc = _EMC_RE.search(line)
    if emc:
        sample["emc_util_pct"] = float(emc.group("util"))
        if emc.group("freq"):
            sample["emc_freq_mhz"] = float(emc.group("freq"))

    temperatures: dict[str, float] = {}
    for match in _TEMP_RE.finditer(line):
        value = float(match.group("value"))
        # -256 C is the Jetson sentinel for an unpopulated sensor.
        if value <= -100:
            continue
        temperatures[match.group("zone")] = value
    if temperatures:
        sample["temperatures_c"] = temperatures
        sample["temperature_max_c"] = max(temperatures.values())

    rails: dict[str, float] = {}
    rails_avg: dict[str, float] = {}
    for match in _POWER_RE.finditer(line):
        name = match.group("name")
        if name in NON_POWER_KEYS:
            continue
        # Without an explicit `mW` suffix, accept only names that look like a
        # rail so that e.g. `IRAM 0/252` style fields cannot be mistaken for one.
        if not match.group("unit") and not _looks_like_rail(name):
            continue
        rails[name] = float(match.group("cur"))
        rails_avg[name] = float(match.group("avg"))
    if rails:
        sample["power_rails_mw"] = rails
        sample["power_rails_avg_mw"] = rails_avg

    return sample


def _looks_like_rail(name: str) -> bool:
    return (name.startswith(("VDD", "POM", "VIN", "SYS", "NC", "PWR"))
            or name.endswith(("_IN", "_GPU", "_CPU", "_SOC", "_CV")))


def tegrastats_path() -> str | None:
    """Locate ``tegrastats``, or ``None`` on a non-Jetson machine."""
    for candidate in TEGRASTATS_CANDIDATES:
        resolved = shutil.which(candidate) if "/" not in candidate else (
            candidate if Path(candidate).exists() else None)
        if resolved:
            return resolved
    return None


@dataclass
class TelemetrySample:
    monotonic: float
    wall_clock: float
    parsed: dict


@dataclass
class TegrastatsMonitor:
    """Run ``tegrastats`` as a child process and collect timestamped samples.

    Every sample is stamped with ``time.monotonic()`` at the moment the line is
    read. ``time.monotonic()`` is ``CLOCK_MONOTONIC`` on Linux and is shared
    across processes, so the orchestrator's telemetry can be aligned exactly to
    a measured interval reported by a model worker in a different virtual
    environment.

    The process is terminated on every exit path - normal stop, model failure,
    Ctrl+C and uncaught exception (the class is also a context manager).
    """

    interval_ms: int = 100
    binary: str | None = None
    keep_raw: bool = True
    samples: list[TelemetrySample] = field(default_factory=list)
    unavailable_reason: str | None = None
    _process: subprocess.Popen | None = None
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def available(self) -> bool:
        return self._process is not None

    def start(self) -> bool:
        binary = self.binary or tegrastats_path()
        if not binary:
            self.unavailable_reason = "tegrastats_not_found"
            return False
        try:
            self._process = subprocess.Popen(
                [binary, "--interval", str(int(self.interval_ms))],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
        except OSError as exc:
            self.unavailable_reason = f"tegrastats_start_failed: {exc}"
            self._process = None
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, name="tegrastats", daemon=True)
        self._thread.start()
        return True

    def _reader(self) -> None:
        stream = self._process.stdout if self._process else None
        if stream is None:
            return
        try:
            for line in stream:
                if self._stop.is_set():
                    break
                parsed = parse_tegrastats_line(line)
                if not self.keep_raw:
                    parsed.pop("raw", None)
                sample = TelemetrySample(time.monotonic(), time.time(), parsed)
                with self._lock:
                    self.samples.append(sample)
        except (ValueError, OSError):
            # Stream closed while the process was being torn down.
            pass

    def stop(self) -> None:
        self._stop.set()
        process, self._process = self._process, None
        if process is not None:
            for action in (process.terminate, process.kill):
                if process.poll() is not None:
                    break
                try:
                    action()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    continue
            if process.stdout is not None:
                try:
                    process.stdout.close()
                except OSError:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def window(self, start_monotonic: float, end_monotonic: float) -> list[TelemetrySample]:
        """Samples whose read time falls inside a measured interval."""
        with self._lock:
            return [s for s in self.samples
                    if start_monotonic <= s.monotonic <= end_monotonic]

    def snapshot(self) -> TelemetrySample | None:
        with self._lock:
            return self.samples[-1] if self.samples else None

    def clear(self) -> None:
        with self._lock:
            self.samples.clear()

    def __enter__(self) -> "TegrastatsMonitor":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()
