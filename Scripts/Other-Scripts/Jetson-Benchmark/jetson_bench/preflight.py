"""Repository, dataset, checkpoint and environment preflight.

Nothing is copied onto the Jetson's internal storage and no checkpoint is
duplicated: preflight only inspects. A checkpoint is not considered usable
merely because ``Path.exists()`` is true - a Git-LFS pointer, a zero-byte
placeholder or a truncated artifact all "exist".
"""

from __future__ import annotations

import importlib.util
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from uvb_bench_core import lfs_pointer_reason, sha256_file

from .models import BenchmarkSpec
from .paths import (
    ATTR_CROPS,
    MDWD_YOLO_DATASET,
    MTSD_GROUPS_ROOT,
    MTSD_STRONG_YOLO,
    MTSD_UNAUG_YOLO,
    PROJECT_ROOT,
    repo_relative,
    venv_dir,
    venv_python,
)

# Which Jetson virtual environment each engine runs in.
ENGINE_ENVIRONMENT = {
    "yolo": "detection",
    "yolo_tensorrt": "detection",
    "rfdetr": "detection",
    "attribute": "attribute",
    "prompt": "prompt",
}

# Modules each environment must be able to import for its track to run.
ENGINE_REQUIRED_MODULES = {
    "yolo": ("torch", "ultralytics", "numpy", "PIL"),
    # The engine is built on the device, so tensorrt itself must be importable.
    "yolo_tensorrt": ("torch", "ultralytics", "numpy", "PIL", "tensorrt"),
    "rfdetr": ("torch", "rfdetr", "numpy", "PIL"),
    "attribute": ("torch", "transformers", "peft", "numpy", "PIL"),
    "prompt": ("torch", "numpy", "PIL"),
}


@dataclass
class CheckpointReport:
    path: str | None = None
    relative_path: str | None = None
    exists: bool = False
    real: bool = False
    reason: str | None = None
    size_mb: float | None = None
    sha256: str | None = None

    def as_dict(self) -> dict:
        return {
            "checkpoint_relative_path": self.relative_path,
            "checkpoint_exists": self.exists,
            "checkpoint_real": self.real,
            "checkpoint_issue": self.reason,
            "checkpoint_mb": self.size_mb,
            "checkpoint_sha256": self.sha256,
        }


def inspect_checkpoint(path: str | Path | None, hash_it: bool = True) -> CheckpointReport:
    report = CheckpointReport()
    if not path:
        report.reason = "no checkpoint resolved"
        return report
    target = Path(path)
    report.path = str(target)
    report.relative_path = repo_relative(target)
    report.exists = target.is_file()
    reason = lfs_pointer_reason(target)
    report.reason = reason
    report.real = reason is None
    if report.exists:
        report.size_mb = round(target.stat().st_size / 1024**2, 3)
        if hash_it and report.real:
            report.sha256 = sha256_file(target)
    return report


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

@dataclass
class DatasetReport:
    key: str
    root: str
    available: bool
    item_count: int = 0
    note: str = ""

    def as_dict(self) -> dict:
        return {"dataset_key": self.key, "dataset_root": self.root,
                "dataset_available": self.available,
                "dataset_items": self.item_count, "dataset_note": self.note}


def _count_images(root: Path, limit: int = 100_000) -> int:
    if not root.is_dir():
        return 0
    count = 0
    for entry in root.iterdir():
        if entry.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            count += 1
            if count >= limit:
                break
    return count


def dataset_reports() -> dict[str, DatasetReport]:
    """Inspect the read-only data each track needs."""
    reports: dict[str, DatasetReport] = {}

    mdwd = MDWD_YOLO_DATASET / "test" / "images"
    reports["MDWD/detection"] = DatasetReport(
        "MDWD/detection", repo_relative(mdwd), mdwd.is_dir(), _count_images(mdwd),
        "" if mdwd.is_dir() else "MDWD YOLO test split not present")

    mtsd_root = next((r for r in (MTSD_STRONG_YOLO / "test" / "images",
                                  MTSD_UNAUG_YOLO / "test" / "images") if r.is_dir()), None)
    groups = sorted(MTSD_GROUPS_ROOT.glob("GRP-*/Images"))
    if mtsd_root is not None:
        reports["MTSD/detection"] = DatasetReport(
            "MTSD/detection", repo_relative(mtsd_root), True, _count_images(mtsd_root))
    elif groups:
        reports["MTSD/detection"] = DatasetReport(
            "MTSD/detection", repo_relative(MTSD_GROUPS_ROOT), True,
            sum(_count_images(g) for g in groups),
            "prepared MTSD split absent; raw group images available")
    else:
        reports["MTSD/detection"] = DatasetReport(
            "MTSD/detection", repo_relative(MTSD_STRONG_YOLO), False, 0,
            "no prepared MTSD test split and no raw group images")

    crops_present = ATTR_CROPS.is_dir() and any(ATTR_CROPS.rglob("*.jpg"))
    reports["MTSD/attribute"] = DatasetReport(
        "MTSD/attribute", repo_relative(ATTR_CROPS), crops_present,
        sum(1 for _ in ATTR_CROPS.rglob("*.jpg")) if crops_present else 0,
        "" if crops_present else
        "attribute crops are regenerable and git-ignored; regenerate them on the "
        "SSD before benchmarking the attribute track")

    reports["MTSD/prompt"] = DatasetReport(
        "MTSD/prompt", reports["MTSD/detection"].root,
        reports["MTSD/detection"].available, reports["MTSD/detection"].item_count)
    return reports


def dataset_key_for(spec: BenchmarkSpec) -> str:
    return f"{spec.dataset}/{spec.task}"


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentReport:
    name: str
    path: str
    exists: bool
    python: str | None
    modules: dict[str, bool] = field(default_factory=dict)
    cuda_available: bool | None = None
    torch_version: str | None = None
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.exists and self.python is not None and all(self.modules.values())

    def as_dict(self) -> dict:
        return {
            "environment": self.name,
            "environment_path": self.path,
            "environment_exists": self.exists,
            "environment_usable": self.usable,
            "environment_torch": self.torch_version,
            "environment_cuda_available": self.cuda_available,
            "environment_missing_modules": [m for m, ok in self.modules.items() if not ok],
            "environment_error": self.error,
        }


PROBE_SOURCE = r"""
import json, importlib.util, sys
modules = sys.argv[1].split(",") if len(sys.argv) > 1 else []
report = {"modules": {m: importlib.util.find_spec(m) is not None for m in modules if m}}
try:
    import torch
    report["torch_version"] = torch.__version__
    report["cuda_available"] = bool(torch.cuda.is_available())
    if report["cuda_available"]:
        report["gpu_name"] = torch.cuda.get_device_name(0)
except Exception as exc:
    report["torch_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(report))
"""


def probe_environment(name: str, modules: tuple[str, ...],
                      timeout: float = 180.0) -> EnvironmentReport:
    """Inspect one Jetson venv without importing anything into this process."""
    import json

    directory = venv_dir(name)
    interpreter = venv_python(name)
    report = EnvironmentReport(name=name, path=repo_relative(directory),
                              exists=directory.is_dir(),
                              python=str(interpreter) if interpreter.is_file() else None)
    if report.python is None:
        report.modules = {m: False for m in modules}
        report.error = "virtual environment not created yet (run the bootstrap)"
        return report
    try:
        completed = subprocess.run(
            [report.python, "-c", PROBE_SOURCE, ",".join(modules)],
            capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        report.error = f"{type(exc).__name__}: {exc}"
        report.modules = {m: False for m in modules}
        return report
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        report.error = (completed.stderr or completed.stdout or "probe produced no output")[-500:]
        report.modules = {m: False for m in modules}
        return report
    report.modules = payload.get("modules", {})
    report.torch_version = payload.get("torch_version")
    report.cuda_available = payload.get("cuda_available")
    if payload.get("torch_error"):
        report.error = payload["torch_error"]
    return report


def probe_current_environment(modules: tuple[str, ...]) -> EnvironmentReport:
    """Fallback probe of the interpreter this orchestrator is running in."""
    import sys

    report = EnvironmentReport(name="current", path=repo_relative(Path(sys.prefix)),
                               exists=True, python=sys.executable)
    report.modules = {m: importlib.util.find_spec(m) is not None for m in modules}
    try:
        import torch
        report.torch_version = torch.__version__
        report.cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        report.error = f"{type(exc).__name__}: {exc}"
    return report


def environment_reports(specs: list[BenchmarkSpec], use_venvs: bool = True,
                        timeout: float = 180.0) -> dict[str, EnvironmentReport]:
    needed: dict[str, set[str]] = {}
    for spec in specs:
        env = ENGINE_ENVIRONMENT.get(spec.engine)
        if env is None:
            continue
        needed.setdefault(env, set()).update(ENGINE_REQUIRED_MODULES.get(spec.engine, ()))
    reports = {}
    for env, modules in needed.items():
        ordered = tuple(sorted(modules))
        reports[env] = (probe_environment(env, ordered, timeout) if use_venvs
                        else probe_current_environment(ordered))
    return reports


# ---------------------------------------------------------------------------
# Coverage table
# ---------------------------------------------------------------------------

def build_coverage(specs: list[BenchmarkSpec], datasets: dict[str, DatasetReport],
                   environments: dict[str, EnvironmentReport],
                   hash_checkpoints: bool = True) -> list[dict]:
    """One row per requested configuration: the requested-coverage table.

    Also mutates each spec's status so a configuration that cannot run is
    recorded with a structured reason before anything is loaded.
    """
    rows = []
    for spec in specs:
        checkpoint = inspect_checkpoint(spec.checkpoint, hash_checkpoints)
        dataset = datasets.get(dataset_key_for(spec))
        environment = environments.get(ENGINE_ENVIRONMENT.get(spec.engine, ""))

        row = {
            "model_id": spec.model_id,
            "track": spec.track,
            "dataset": spec.dataset,
            "task": spec.task,
            "requested_model": spec.display_name,
            "family": spec.family,
            "scale": spec.scale,
            "augmentation": spec.augmentation,
            "input_resolution": spec.input_resolution,
            "run_directory": repo_relative(spec.run_directory) if spec.run_directory else None,
            **checkpoint.as_dict(),
            **(dataset.as_dict() if dataset else
               {"dataset_available": spec.task == "prompt", "dataset_note": "not applicable"}),
            **(environment.as_dict() if environment else {"environment": "n/a"}),
            "workstation_comparable": spec.workstation_comparable,
            "workstation_comparison_note": spec.workstation_reason,
            "predictive_metric_name": spec.predictive_metric_name,
            "predictive_metric_value": spec.predictive_metric_value,
            "predictive_metric_source": spec.predictive_metric_source,
            "predictive_protocol": spec.predictive_protocol,
            "resolution_evidence": "; ".join(spec.resolution_evidence),
            "smoke_status": "pending",
            "benchmark_status": "pending",
        }

        if spec.status in ("missing_checkpoint", "ambiguous_checkpoint"):
            pass
        elif spec.task == "prompt":
            spec.status = "preflight_ok"
        elif not checkpoint.exists:
            spec.status, spec.reason = "missing_checkpoint", "checkpoint file not found"
        elif not checkpoint.real:
            spec.status = "missing_checkpoint"
            spec.reason = f"checkpoint is not a usable artifact ({checkpoint.reason})"
        elif dataset is not None and not dataset.available:
            spec.status, spec.reason = "missing_dataset", dataset.note
        elif environment is not None and not environment.usable:
            missing = [m for m, ok in environment.modules.items() if not ok]
            spec.status = "dependency_error"
            spec.reason = (environment.error
                           or f"environment {environment.name} is missing {missing}")
        elif environment is not None and environment.cuda_available is False:
            spec.status = "dependency_error"
            spec.reason = (
                f"PyTorch in the {environment.name} environment reports "
                "torch.cuda.is_available() == False; a Jetson GPU benchmark requires "
                "CUDA. Use --device cpu only for debugging.")
        else:
            spec.status = "preflight_ok"

        row["status"] = spec.status
        row["reason"] = spec.reason
        row["selected"] = spec.status == "preflight_ok"
        rows.append(row)
    return rows
