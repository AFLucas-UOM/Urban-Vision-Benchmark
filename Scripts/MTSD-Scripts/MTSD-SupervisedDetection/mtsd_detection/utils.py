from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def find_repo_root(start: Path | None = None) -> Path:
    for candidate in [start or Path.cwd(), *(start or Path.cwd()).parents]:
        if (candidate / "Datasets").is_dir() and (candidate / "Scripts").is_dir():
            return candidate.resolve()
    raise RuntimeError("Could not find repository root containing Datasets/ and Scripts/.")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def hardware_info() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(), "processor": platform.processor(),
        "cpu_count": os.cpu_count(), "python": sys.version.replace("\n", " "),
    }
    try:
        import torch
        result.update(torch_version=torch.__version__, cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            result.update(cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                          gpu_vram_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
    except ImportError:
        result["torch_version"] = None
    return result

