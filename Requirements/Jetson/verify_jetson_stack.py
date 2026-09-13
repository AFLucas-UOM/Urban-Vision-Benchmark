#!/usr/bin/env python3
"""Verify that a Jetson's NVIDIA software stack is intact and usable.

Run this BEFORE installing anything, and again after the bootstrap. It never
modifies the system: it only reports.

    python3 Requirements/Jetson/verify_jetson_stack.py
    python3 Requirements/Jetson/verify_jetson_stack.py --json
    .venv-jetson-detection/bin/python Requirements/Jetson/verify_jetson_stack.py

Exit codes
    0  the stack is usable for a GPU benchmark
    1  a required condition failed (the report says which)
    2  this is not a Jetson (informational; useful on a workstation)

The checks deliberately stop at "report the problem". If the installed JetPack
is incompatible with the model software this script says so; it does not try to
repair JetPack, change CUDA, or install driver packages.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "Datasets").exists() and (parent / "Scripts").exists():
            return parent
    return start


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
JETSON_BENCH = PROJECT_ROOT / "Scripts" / "Other-Scripts" / "Jetson-Benchmark"
if str(JETSON_BENCH) not in sys.path:
    sys.path.insert(0, str(JETSON_BENCH))


class Check:
    def __init__(self, name: str, ok: bool | None, detail: str, required: bool = True):
        self.name, self.ok, self.detail, self.required = name, ok, detail, required

    @property
    def symbol(self) -> str:
        if self.ok is None:
            return "??"
        return "OK" if self.ok else ("FAIL" if self.required else "warn")

    def as_dict(self) -> dict:
        return {"check": self.name, "ok": self.ok, "required": self.required,
                "detail": self.detail}


def run(command: list[str], timeout: float = 15.0) -> str | None:
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return (completed.stdout or completed.stderr or "").strip() or None


def collect() -> tuple[list[Check], dict]:
    from jetson_bench.platform_info import (  # noqa: PLC0415
        cuda_versions, detect_jetson, detect_power_mode, memory_snapshot,
        thermal_zones, torch_snapshot,
    )

    jetson = detect_jetson()
    cuda = cuda_versions()
    torch_info = torch_snapshot()
    memory = memory_snapshot()
    power = detect_power_mode()
    checks: list[Check] = []

    checks.append(Check(
        "Jetson platform", jetson["is_jetson"],
        f"{jetson.get('jetson_model') or 'not detected'} "
        f"(arch {jetson.get('architecture')})",
        required=False))
    checks.append(Check(
        "aarch64 architecture", platform.machine() == "aarch64",
        platform.machine(), required=False))
    checks.append(Check(
        "L4T release", bool(jetson.get("l4t_version")),
        jetson.get("l4t_version") or "not determinable "
        "(/etc/nv_tegra_release absent and nvidia-l4t-core not installed)",
        required=False))
    checks.append(Check(
        "JetPack", bool(jetson.get("jetpack_version")),
        jetson.get("jetpack_version") or f"not mapped from L4T "
        f"{jetson.get('l4t_version')}", required=False))
    checks.append(Check(
        "CUDA toolkit", bool(cuda.get("cuda_version")),
        cuda.get("cuda_version") or "nvcc absent and no /usr/local/cuda version file",
        required=False))
    checks.append(Check(
        "cuDNN", bool(cuda.get("cudnn_version")),
        cuda.get("cudnn_version") or "not determinable", required=False))
    checks.append(Check(
        "TensorRT", bool(cuda.get("tensorrt_version")),
        cuda.get("tensorrt_version") or "not determinable "
        "(only needed for the optional --runtime tensorrt experiment)",
        required=False))

    checks.append(Check(
        "PyTorch importable", torch_info.get("torch_version") is not None,
        torch_info.get("torch_version") or f"import failed: {torch_info.get('error')}"))
    checks.append(Check(
        "torch.cuda.is_available()", bool(torch_info.get("torch_cuda_available")),
        f"{torch_info.get('torch_cuda_available')}"
        + (f"; GPU {torch_info.get('gpu_name')}" if torch_info.get("gpu_name") else "")
        + ("" if torch_info.get("torch_cuda_available") else
           " - a Jetson GPU benchmark requires CUDA. If PyTorch was installed "
           "from PyPI it is a CPU-only build; reinstall NVIDIA's JetPack wheel.")))
    checks.append(Check(
        "torchvision", torch_info.get("torchvision_version") is not None,
        torch_info.get("torchvision_version") or "not installed", required=False))

    if torch_info.get("torch_version") and cuda.get("cuda_version"):
        torch_cuda = (torch_info.get("torch_cuda_version") or "").split(".")[:2]
        system_cuda = cuda["cuda_version"].split(".")[:2]
        aligned = torch_cuda == system_cuda
        checks.append(Check(
            "PyTorch/CUDA alignment", aligned,
            f"torch was built for CUDA {torch_info.get('torch_cuda_version')}, "
            f"the system reports CUDA {cuda['cuda_version']}"
            + ("" if aligned else " - a mismatch usually means a non-JetPack wheel "
                                  "is installed"),
            required=False))

    total_ram = memory.get("system_ram_total_mb")
    checks.append(Check(
        "System memory", total_ram is not None and total_ram >= 3500,
        f"{total_ram} MB total, {memory.get('system_ram_available_mb')} MB available"
        if total_ram else "not readable", required=False))
    checks.append(Check(
        "tegrastats", shutil.which("tegrastats") is not None
        or Path("/usr/bin/tegrastats").exists(),
        "present" if (shutil.which("tegrastats") or Path("/usr/bin/tegrastats").exists())
        else "absent - power, thermal and unified-memory telemetry will be null",
        required=False))
    checks.append(Check(
        "nvpmodel", power["nvpmodel_available"],
        f"mode {power.get('nvpmodel_mode')} "
        f"({power.get('nvpmodel_mode_name') or 'name unavailable'})"
        if power["nvpmodel_available"] else "absent", required=False))
    checks.append(Check(
        "python3-venv", _venv_available(),
        "available" if _venv_available() else
        "missing - install with: sudo apt-get install -y python3-venv"))

    repository = {
        "repo_root": str(PROJECT_ROOT),
        "free_gb": round(shutil.disk_usage(PROJECT_ROOT).free / 1024**3, 2),
    }
    checks.append(Check(
        "Repository free space", repository["free_gb"] >= 5,
        f"{repository['free_gb']} GB free on the volume holding the repository",
        required=False))

    snapshot = {"jetson": jetson, "cuda": cuda, "torch": torch_info,
                "memory": memory, "power_mode": power,
                "thermal_zones_c": thermal_zones(), "repository": repository,
                "python": {"version": sys.version.split()[0],
                           "executable": sys.executable}}
    return checks, snapshot


def _venv_available() -> bool:
    try:
        import venv  # noqa: F401, PLC0415
        import ensurepip  # noqa: F401, PLC0415
        return True
    except ImportError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Also write the JSON snapshot to this path.")
    args = parser.parse_args()

    checks, snapshot = collect()
    payload = {"checks": [c.as_dict() for c in checks], "snapshot": snapshot}

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print("Jetson software-stack verification")
        print("=" * 72)
        for check in checks:
            print(f"  [{check.symbol:>4}] {check.name:<26} {check.detail}")
        print("=" * 72)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n",
                               encoding="utf-8")

    failures = [c for c in checks if c.required and c.ok is False]
    if not snapshot["jetson"]["is_jetson"]:
        if not args.json:
            print("\nThis machine is not a Jetson. The benchmark's --dry-run, the unit "
                  "tests and the analysis tooling still work here; latency, power and "
                  "thermal measurements do not.")
        return 2
    if failures:
        if not args.json:
            print("\nRequired checks failed:")
            for check in failures:
                print(f"  - {check.name}: {check.detail}")
            print("\nThis script does not attempt to repair the JetPack stack. Fix the "
                  "reported condition (or reflash the matching JetPack release) before "
                  "benchmarking.")
        return 1
    if not args.json:
        print("\nStack is usable for a Jetson GPU benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
