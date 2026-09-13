"""Jetson hardware / software / repository discovery.

Everything here is read-only and degrades gracefully: on a non-Jetson
development machine each probe records why it could not answer instead of
raising, so ``--dry-run`` and the unit tests work anywhere.

Deliberately **not** collected: serial numbers, MAC addresses, hostnames,
usernames or any other identifier that is not needed to reproduce a
measurement.
"""

from __future__ import annotations

import getpass
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .paths import PROJECT_ROOT
from .tegrastats import tegrastats_path

L4T_RELEASE_FILE = Path("/etc/nv_tegra_release")
DEVICE_TREE_MODEL = Path("/proc/device-tree/model")
NVPMODEL = "/usr/sbin/nvpmodel"
JETSON_CLOCKS = "/usr/bin/jetson_clocks"

# L4T release -> JetPack version. Only entries that are unambiguous are listed;
# an unknown L4T maps to None rather than to a guess.
L4T_TO_JETPACK = {
    "32.7": "4.6.x", "35.1": "5.0.2", "35.2": "5.1", "35.3": "5.1.1",
    "35.4": "5.1.2", "35.5": "5.1.3", "35.6": "5.1.4",
    "36.2": "6.0 DP", "36.3": "6.0", "36.4": "6.1", "36.4.3": "6.2",
    "38.1": "7.0", "38.2": "7.0",
}


def _run(command: list[str], timeout: float = 10.0) -> str | None:
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    output = (completed.stdout or "") + (completed.returncode and (completed.stderr or "") or "")
    return output.strip() or None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace").strip("\x00\n ") or None
    except OSError:
        return None


def _scrub(text: str | None) -> str | None:
    """Remove the current user name / home path from a captured string."""
    if not text:
        return text
    try:
        user = getpass.getuser()
    except Exception:  # pragma: no cover - no password database
        user = None
    home = str(Path.home())
    if home and home in text:
        text = text.replace(home, "~")
    if user and len(user) > 2:
        text = re.sub(rf"\b{re.escape(user)}\b", "<user>", text)
    return text


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

def detect_jetson() -> dict:
    """Identify the Jetson module/carrier, or explain why it is not one."""
    machine = platform.machine()
    model = _read_text(DEVICE_TREE_MODEL)
    l4t_raw = _read_text(L4T_RELEASE_FILE)
    compatible = _read_text(Path("/proc/device-tree/compatible"))

    is_tegra = bool(
        (model and "jetson" in model.lower())
        or (compatible and "tegra" in compatible.lower())
        or L4T_RELEASE_FILE.exists()
    )
    info = {
        "is_jetson": bool(is_tegra and machine == "aarch64"),
        "jetson_model": model,
        "architecture": machine,
        "platform_machine_is_aarch64": machine == "aarch64",
        "device_tree_compatible": compatible,
        "l4t_release_raw": l4t_raw,
        "l4t_version": None,
        "jetpack_version": None,
        "detection_notes": [],
    }
    if l4t_raw:
        match = re.search(r"R(\d+).*?REVISION:\s*([\d.]+)", l4t_raw)
        if match:
            revision = match.group(2).rstrip(".")
            info["l4t_version"] = f"{match.group(1)}.{revision}"
    if not info["l4t_version"]:
        # L4T can also be recovered from the nvidia-l4t-core package.
        package = _run(["dpkg-query", "--showformat=${Version}", "--show", "nvidia-l4t-core"])
        if package:
            info["l4t_version"] = package.split("-")[0]
            info["detection_notes"].append("l4t_version from dpkg nvidia-l4t-core")
    l4t = info["l4t_version"]
    if l4t:
        info["jetpack_version"] = (
            L4T_TO_JETPACK.get(l4t)
            or L4T_TO_JETPACK.get(".".join(l4t.split(".")[:2]))
        )
        if not info["jetpack_version"]:
            info["detection_notes"].append(
                f"JetPack version not determinable from L4T {l4t}")
    if not info["is_jetson"]:
        info["detection_notes"].append(
            f"not a Jetson: machine={machine!r}, "
            f"device-tree model={'present' if model else 'absent'}, "
            f"{L4T_RELEASE_FILE} {'present' if l4t_raw else 'absent'}")
    return info


def detect_power_mode() -> dict:
    """Current ``nvpmodel`` mode and the modes the platform offers."""
    info: dict = {"nvpmodel_available": False, "nvpmodel_mode": None,
                  "nvpmodel_mode_name": None, "nvpmodel_available_modes": None,
                  "jetson_clocks_available": False, "jetson_clocks_state": None,
                  "notes": []}
    binary = NVPMODEL if Path(NVPMODEL).exists() else shutil.which("nvpmodel")
    if binary:
        info["nvpmodel_available"] = True
        query = _run([binary, "-q"])
        if query:
            mode = re.search(r"NV Power Mode:\s*(.+)", query)
            number = re.search(r"^\s*(\d+)\s*$", query, re.MULTILINE)
            info["nvpmodel_mode_name"] = mode.group(1).strip() if mode else None
            info["nvpmodel_mode"] = int(number.group(1)) if number else None
        modes = _read_text(Path("/etc/nvpmodel.conf"))
        if modes:
            found = re.findall(r"^\s*<\s*POWER_MODEL\s+ID=(\d+)\s+NAME=(\S+)", modes, re.MULTILINE)
            if found:
                info["nvpmodel_available_modes"] = [
                    {"id": int(i), "name": n} for i, n in found]
    else:
        info["notes"].append("nvpmodel not present")

    clocks = JETSON_CLOCKS if Path(JETSON_CLOCKS).exists() else shutil.which("jetson_clocks")
    if clocks:
        info["jetson_clocks_available"] = True
        state = _run([clocks, "--show"], timeout=20)
        if state:
            info["jetson_clocks_state"] = _scrub(state)[:4000]
        else:
            info["notes"].append("jetson_clocks --show returned nothing "
                                 "(usually requires root)")
    else:
        info["notes"].append("jetson_clocks not present")
    return info


def set_power_mode(mode_id: int) -> dict:
    """Change the platform power mode, validating the requested ID first.

    ``/etc/nvpmodel.conf`` is never edited; only ``nvpmodel -m`` is called, and
    the before/after state is recorded.
    """
    before = detect_power_mode()
    result = {"requested_mode": mode_id, "before": before, "applied": False,
              "error": None, "after": None}
    if not before["nvpmodel_available"]:
        result["error"] = "nvpmodel is not available on this platform"
        return result
    modes = before.get("nvpmodel_available_modes")
    if modes and mode_id not in {m["id"] for m in modes}:
        result["error"] = (f"mode {mode_id} is not offered by this Jetson "
                           f"(available: {sorted(m['id'] for m in modes)})")
        return result
    binary = NVPMODEL if Path(NVPMODEL).exists() else shutil.which("nvpmodel")
    output = _run(["sudo", "-n", binary, "-m", str(mode_id)], timeout=60)
    result["command_output"] = _scrub(output)
    result["after"] = detect_power_mode()
    result["applied"] = result["after"].get("nvpmodel_mode") == mode_id
    if not result["applied"]:
        result["error"] = ("nvpmodel -m did not take effect (sudo credentials "
                           "may be required; run with --sudo-prompt)")
    return result


def set_jetson_clocks(enable: bool = True) -> dict:
    binary = JETSON_CLOCKS if Path(JETSON_CLOCKS).exists() else shutil.which("jetson_clocks")
    if not binary:
        return {"applied": False, "error": "jetson_clocks is not available"}
    command = ["sudo", "-n", binary] + ([] if enable else ["--restore"])
    output = _run(command, timeout=60)
    return {"applied": output is not None, "enabled": enable,
            "command_output": _scrub(output)}


def memory_snapshot() -> dict:
    """Total / available system RAM. Jetson memory is unified with the GPU."""
    info: dict = {}
    meminfo = _read_text(Path("/proc/meminfo"))
    if meminfo:
        values = dict(re.findall(r"^(\w+):\s+(\d+) kB", meminfo, re.MULTILINE))
        for key, field in (("MemTotal", "system_ram_total_mb"),
                           ("MemAvailable", "system_ram_available_mb"),
                           ("MemFree", "system_ram_free_mb"),
                           ("SwapTotal", "swap_total_mb"),
                           ("SwapFree", "swap_free_mb")):
            if key in values:
                info[field] = round(int(values[key]) / 1024, 1)
        return info
    try:
        import psutil  # noqa: PLC0415 - optional
        virtual = psutil.virtual_memory()
        info["system_ram_total_mb"] = round(virtual.total / 1024**2, 1)
        info["system_ram_available_mb"] = round(virtual.available / 1024**2, 1)
    except Exception:
        info["note"] = "system memory not readable (no /proc/meminfo, no psutil)"
    return info


def thermal_zones() -> dict[str, float]:
    """Temperatures from ``/sys/class/thermal`` (fallback when tegrastats is absent)."""
    zones: dict[str, float] = {}
    root = Path("/sys/class/thermal")
    if not root.is_dir():
        return zones
    for zone in sorted(root.glob("thermal_zone*")):
        name = _read_text(zone / "type") or zone.name
        raw = _read_text(zone / "temp")
        if raw is None:
            continue
        try:
            value = int(raw) / 1000.0
        except ValueError:
            continue
        if value <= -100:
            continue
        zones[name] = round(value, 2)
    return zones


def thermal_throttling() -> dict:
    """Whether the OS currently reports a thermal-throttling condition."""
    status: dict = {"throttled": False, "evidence": []}
    for path in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_cur_freq"):
        break
    for trip in Path("/sys/class/thermal").glob("thermal_zone*/cdev*_trip_point"):
        value = _read_text(trip)
        if value and value.strip() not in ("-1", ""):
            status["throttled"] = True
            status["evidence"].append(str(trip))
    soc_throttle = _read_text(Path("/sys/kernel/debug/bpmp/debug/soctherm/throt_en"))
    if soc_throttle:
        status["soctherm_throttle_enabled"] = soc_throttle
    return status


def hardware_snapshot() -> dict:
    """The full hardware record written into every benchmark run."""
    snapshot = {
        "architecture": platform.machine(),
        "kernel_release": platform.release(),
        "kernel_version": _scrub(platform.version()),
        "system": platform.system(),
        "cpu_count_logical": os.cpu_count(),
    }
    snapshot.update(detect_jetson())
    snapshot.update(memory_snapshot())
    snapshot["nvpmodel"] = detect_power_mode()
    snapshot["thermal_zones_c"] = thermal_zones()
    snapshot["tegrastats_path"] = tegrastats_path()
    os_release = _read_text(Path("/etc/os-release"))
    if os_release:
        fields = dict(re.findall(r'^(\w+)="?([^"\n]*)"?$', os_release, re.MULTILINE))
        snapshot["os_name"] = fields.get("NAME")
        snapshot["os_version"] = fields.get("VERSION")
        snapshot["os_pretty_name"] = fields.get("PRETTY_NAME")
    return snapshot


# ---------------------------------------------------------------------------
# Software
# ---------------------------------------------------------------------------

def _dpkg_version(package: str) -> str | None:
    output = _run(["dpkg-query", "--showformat=${Version}", "--show", package])
    return output.splitlines()[0].strip() if output else None


def cuda_versions() -> dict:
    info: dict = {"cuda_version": None, "cudnn_version": None,
                  "tensorrt_version": None, "sources": {}}
    nvcc = _run(["nvcc", "--version"])
    if nvcc:
        match = re.search(r"release (\d+\.\d+)", nvcc)
        if match:
            info["cuda_version"] = match.group(1)
            info["sources"]["cuda_version"] = "nvcc --version"
    if not info["cuda_version"]:
        version_txt = _read_text(Path("/usr/local/cuda/version.json")) or \
            _read_text(Path("/usr/local/cuda/version.txt"))
        if version_txt:
            match = re.search(r"(\d+\.\d+\.\d+)", version_txt)
            if match:
                info["cuda_version"] = match.group(1)
                info["sources"]["cuda_version"] = "/usr/local/cuda/version"
    for package, field in (("cuda-cudart-*", "cuda_version"),
                           ("libcudnn*", "cudnn_version"),
                           ("libnvinfer*", "tensorrt_version")):
        if info.get(field):
            continue
        listing = _run(["dpkg-query", "--showformat=${Package} ${Version}\\n",
                        "--show", package])
        if listing:
            versions = [line.split()[-1] for line in listing.splitlines() if " " in line]
            if versions:
                info[field] = sorted(versions)[0].split("-")[0]
                info["sources"][field] = f"dpkg {package}"
    if not info["cudnn_version"]:
        header = _read_text(Path("/usr/include/cudnn_version.h")) or \
            _read_text(Path("/usr/include/aarch64-linux-gnu/cudnn_version.h"))
        if header:
            parts = dict(re.findall(r"#define CUDNN_(MAJOR|MINOR|PATCHLEVEL)\s+(\d+)", header))
            if len(parts) == 3:
                info["cudnn_version"] = f"{parts['MAJOR']}.{parts['MINOR']}.{parts['PATCHLEVEL']}"
                info["sources"]["cudnn_version"] = "cudnn_version.h"
    if not info["tensorrt_version"]:
        try:
            import tensorrt  # noqa: PLC0415 - optional
            info["tensorrt_version"] = tensorrt.__version__
            info["sources"]["tensorrt_version"] = "python tensorrt module"
        except Exception:
            pass
    return info


def package_version(module_name: str) -> str | None:
    try:
        import importlib.metadata as metadata  # noqa: PLC0415
        return metadata.version(module_name)
    except Exception:
        pass
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def torch_snapshot() -> dict:
    """PyTorch / CUDA availability - the gate for a valid Jetson GPU benchmark."""
    info: dict = {"torch_version": None, "torch_cuda_available": False,
                  "torch_cuda_version": None, "torch_cudnn_version": None,
                  "gpu_name": None, "torchvision_version": None,
                  "error": None}
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info
    info["torch_version"] = torch.__version__
    info["torch_file"] = _scrub(getattr(torch, "__file__", None))
    try:
        info["torch_cuda_version"] = torch.version.cuda
    except Exception:
        pass
    try:
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if info["torch_cuda_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = ".".join(str(v) for v in torch.cuda.get_device_capability(0))
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_memory_mb"] = round(props.total_memory / 1024**2, 1)
    except Exception as exc:
        info["error"] = f"cuda probe failed: {type(exc).__name__}: {exc}"
    try:
        info["torch_cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        pass
    info["torchvision_version"] = package_version("torchvision")
    return info


def software_snapshot() -> dict:
    snapshot = {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "python_executable": _scrub(sys.executable),
        "virtualenv": _scrub(os.environ.get("VIRTUAL_ENV")),
    }
    snapshot.update(cuda_versions())
    snapshot.update(torch_snapshot())
    for name in ("ultralytics", "rfdetr", "supervision", "numpy", "opencv-python",
                 "pillow", "transformers", "peft", "timm", "matplotlib", "psutil",
                 "pycocotools"):
        snapshot[f"{name.replace('-', '_')}_version"] = package_version(name)
    return snapshot


# ---------------------------------------------------------------------------
# Repository / storage
# ---------------------------------------------------------------------------

def _git(*args: str) -> str | None:
    return _run(["git", "-C", str(PROJECT_ROOT), *args])


def filesystem_for(path: Path) -> dict:
    """Filesystem, mount device and free space for the volume holding *path*."""
    info: dict = {"path": str(path), "filesystem": None, "device": None,
                  "mount_point": None, "mount_options": None,
                  "free_gb": None, "total_gb": None, "notes": []}
    try:
        usage = shutil.disk_usage(path)
        info["total_gb"] = round(usage.total / 1024**3, 2)
        info["free_gb"] = round(usage.free / 1024**3, 2)
        info["used_gb"] = round(usage.used / 1024**3, 2)
    except OSError as exc:
        info["notes"].append(f"disk usage unavailable: {exc}")

    mounts = _read_text(Path("/proc/mounts"))
    if mounts:
        best = None
        resolved = str(Path(path).resolve())
        for line in mounts.splitlines():
            fields = line.split()
            if len(fields) < 4:
                continue
            device, mount_point, fs_type, options = fields[0], fields[1], fields[2], fields[3]
            if resolved == mount_point or resolved.startswith(mount_point.rstrip("/") + "/"):
                if best is None or len(mount_point) > len(best[1]):
                    best = (device, mount_point, fs_type, options)
        if best:
            info["device"], info["mount_point"] = best[0], best[1]
            info["filesystem"], info["mount_options"] = best[2], best[3]
    else:
        info["notes"].append("/proc/mounts not available on this platform")
    return info


NON_NATIVE_FILESYSTEMS = {"ntfs", "ntfs3", "fuseblk", "exfat", "vfat", "msdos"}


def repository_snapshot(run_output_root: Path | None = None) -> dict:
    """Repository identity, writability and the SSD it lives on."""
    snapshot = {
        "repo_path": str(PROJECT_ROOT),
        "repo_path_is_absolute": PROJECT_ROOT.is_absolute(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_commit_short": _git("rev-parse", "--short", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
    }
    status = _git("status", "--porcelain")
    snapshot["git_dirty"] = bool(status)
    snapshot["git_dirty_file_count"] = len(status.splitlines()) if status else 0
    snapshot["storage"] = filesystem_for(PROJECT_ROOT)

    fs_type = (snapshot["storage"].get("filesystem") or "").lower()
    warnings: list[str] = []
    if fs_type in NON_NATIVE_FILESYSTEMS:
        warnings.append(
            f"repository filesystem is {fs_type!r}: POSIX permissions, symlinks "
            "and virtual-environment creation may behave unexpectedly; a "
            "Linux-native filesystem (ext4/xfs/btrfs) is recommended")
    options = snapshot["storage"].get("mount_options") or ""
    if "ro" in options.split(","):
        warnings.append("repository volume is mounted read-only")
    snapshot["storage_warnings"] = warnings

    probe_root = run_output_root or (PROJECT_ROOT / "Results")
    snapshot["writable"] = _probe_writable(probe_root)
    return snapshot


def _probe_writable(directory: Path) -> dict:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".uvb_jetson_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return {"ok": True, "path": str(directory), "error": None}
    except OSError as exc:
        return {"ok": False, "path": str(directory), "error": f"{type(exc).__name__}: {exc}"}
