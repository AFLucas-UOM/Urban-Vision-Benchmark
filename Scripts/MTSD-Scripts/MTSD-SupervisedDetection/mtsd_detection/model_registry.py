from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .utils import sha256_file

DEFAULT_ORDER = ["yolo12n", "yolo12s", "yolo12m", "yolo26n", "yolo26s", "yolo26m",
                 "yolo11n", "yolo11s", "yolo11m", "rfdetr-n", "rfdetr-s", "rfdetr-m", "yolo26l"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    scale: str
    checkpoint: str
    trainer: str


def registry() -> dict[str, ModelSpec]:
    result: dict[str, ModelSpec] = {}
    for family in ("11", "12", "26"):
        for scale in ("n", "s", "m", "l"):
            key = f"yolo{family}{scale}"
            result[key] = ModelSpec(key, f"YOLO{family}", scale, f"yolo{family}{scale}.pt", "ultralytics")
    for scale, name in (("n", "nano"), ("s", "small"), ("m", "medium"), ("large", "large")):
        key = f"rfdetr-{scale}"
        result[key] = ModelSpec(key, "RF-DETR", scale, f"rf-detr-{name}.pth", "rfdetr")
    return result


def resolve_checkpoint(spec: ModelSpec, repo_root: Path) -> Path:
    candidates = ([repo_root / "Models" / spec.family / spec.checkpoint,
                   repo_root / "Models" / spec.checkpoint] if spec.trainer == "ultralytics" else
                  [repo_root / "Models" / "RF-DETR" / spec.checkpoint, repo_root / "Models" / spec.checkpoint])
    return next((path for path in candidates if path.is_file()), candidates[0])


def resolve_checkpoint_info(spec: ModelSpec, repo_root: Path, require_local: bool = True) -> dict:
    path = resolve_checkpoint(spec, repo_root)
    if not path.is_file():
        if require_local:
            raise FileNotFoundError(
                f"Required repository checkpoint for {spec.key} is missing: {path}. "
                "Automatic downloads into the repository are disabled."
            )
        return {"path": str(path), "source": "remote_pretrained", "sha256": None, "exists": False}
    return {"path": str(path.resolve()), "source": "local_repository",
            "sha256": sha256_file(path), "exists": True}


def validate_model_keys(keys: list[str]) -> list[ModelSpec]:
    known = registry()
    unknown = [key for key in keys if key not in known]
    if unknown: raise ValueError(f"Unknown models {unknown}; available: {sorted(known)}")
    return [known[key] for key in keys]


def self_check(repo_root: Path, instantiate: bool = False) -> list[dict]:
    rows = []
    for key, spec in registry().items():
        checkpoint = resolve_checkpoint(spec, repo_root)
        row = {"model": key, "checkpoint": str(checkpoint), "exists": checkpoint.is_file(), "ok": checkpoint.is_file(),
               "source": "local_repository" if checkpoint.is_file() else "missing_local",
               "sha256": sha256_file(checkpoint) if checkpoint.is_file() else None}
        if instantiate and checkpoint.is_file():
            try:
                if spec.trainer == "ultralytics":
                    from ultralytics import YOLO
                    YOLO(str(checkpoint))
                else:
                    _rfdetr_class(spec)(pretrain_weights=str(checkpoint))
            except Exception as exc: row.update(ok=False, error=str(exc))
        rows.append(row)
    return rows


def _rfdetr_class(spec: ModelSpec):
    import rfdetr
    return {"n": rfdetr.RFDETRNano, "s": rfdetr.RFDETRSmall, "m": rfdetr.RFDETRMedium,
            "large": rfdetr.RFDETRLarge}[spec.scale]
