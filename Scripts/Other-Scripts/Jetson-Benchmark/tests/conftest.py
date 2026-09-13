"""Shared fixtures for the Jetson-benchmark test suite.

Every test in this package runs on a non-Jetson development machine and needs
neither the private datasets nor the trained checkpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[2]
INFERENCE_BENCH = REPO_ROOT / "Scripts" / "Other-Scripts" / "Inference-Benchmark"

for path in (PACKAGE_ROOT, INFERENCE_BENCH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def default_config_path() -> Path:
    return PACKAGE_ROOT / "config" / "jetson_benchmark.yaml"


@pytest.fixture()
def tegrastats_lines() -> dict[str, str]:
    """Representative tegrastats lines from several Jetson generations.

    These are format fixtures, not measurements: they exist so the parser is
    exercised against the real layout differences between Tegra platforms and
    L4T releases rather than against one device's output.
    """
    return {
        # AGX Orin, L4T 35.x - power rails carry an explicit mW suffix, GR3D
        # frequency is bracketed, and unpopulated sensors report -256C.
        "agx_orin": (
            "11-29-2025 10:14:02 RAM 6421/30536MB (lfb 3x4MB) SWAP 0/15268MB (cached 0MB) "
            "CPU [24%@2201,18%@2201,12%@2201,9%@2201,7%@2201,6%@2201,5%@2201,4%@2201,"
            "3%@2201,2%@2201,1%@2201,0%@2201] EMC_FREQ 31%@3199 GR3D_FREQ 78%@[1300] "
            "NVDEC off NVJPG off NVJPG1 off VIC off OFA off APE 174 "
            "CV0@-256C CPU@57.593C SOC2@54.906C SOC0@53.625C CV1@-256C GPU@56.125C "
            "tj@57.593C SOC1@54.281C CV2@-256C "
            "VDD_GPU_SOC 9431mW/8102mW VDD_CPU_CV 3122mW/2874mW VIN_SYS_5V0 4835mW/4610mW "
            "NC 0mW/0mW"
        ),
        # Orin Nano developer kit - exposes a true total input rail (VDD_IN).
        "orin_nano": (
            "11-29-2025 10:14:02 RAM 3140/7620MB (lfb 12x4MB) SWAP 0/3810MB (cached 0MB) "
            "CPU [41%@1510,33%@1510,22%@1510,15%@1510,off,off] EMC_FREQ 18%@2133 "
            "GR3D_FREQ 61%@[624] NVDEC off NVJPG off "
            "CPU@48.906C SOC2@47.531C SOC0@46.812C GPU@47.25C tj@48.906C SOC1@47.156C "
            "VDD_IN 8942mW/8130mW VDD_CPU_GPU_CV 4210mW/3880mW VDD_SOC 2115mW/2006mW"
        ),
        # Xavier NX, L4T 32.x - no mW suffixes, plain GR3D_FREQ percentage.
        "xavier_nx": (
            "RAM 2673/7765MB (lfb 118x4MB) SWAP 0/3883MB (cached 0MB) "
            "CPU [12%@1190,8%@1190,4%@1190,off,off,off] EMC_FREQ 7% GR3D_FREQ 33% "
            "AO@35C GPU@33C PMIC@100C AUX@33.5C CPU@35C thermal@33.9C "
            "VDD_IN 3902/3611 VDD_CPU_GPU_CV 1331/1180 VDD_SOC 997/941"
        ),
        # Jetson Nano - IRAM field present, POM_5V_* rails, no total-only rail
        # name that the newer boards use.
        "nano": (
            "RAM 1276/3956MB (lfb 108x4MB) SWAP 0/1978MB (cached 0MB) "
            "IRAM 0/252kB(lfb 252kB) CPU [31%@1479,12%@1479,5%@1479,2%@1479] "
            "EMC_FREQ 14%@1600 GR3D_FREQ 22%@921 APE 25 "
            "PLL@21C CPU@23C PMIC@100C GPU@22.5C AO@28C thermal@22.5C "
            "POM_5V_IN 3478/3201 POM_5V_GPU 812/740 POM_5V_CPU 623/588"
        ),
        # A severely truncated line: the parser must degrade, not raise.
        "minimal": "RAM 900/3956MB CPU [5%@102] GR3D_FREQ 0%",
        # Only thermals, no power and no memory.
        "thermal_only": "CPU@41.5C GPU@40C tj@41.5C",
        "empty": "",
    }


@pytest.fixture()
def make_samples():
    """Build TelemetrySample objects with controlled monotonic timestamps."""
    from jetson_bench.tegrastats import TelemetrySample, parse_tegrastats_line

    def factory(lines, start: float = 100.0, step: float = 0.1):
        return [TelemetrySample(start + index * step, 1_700_000_000 + index * step,
                                parse_tegrastats_line(line))
                for index, line in enumerate(lines)]

    return factory
