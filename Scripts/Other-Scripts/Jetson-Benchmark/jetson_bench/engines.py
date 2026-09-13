"""Model engines and timing boundaries for the Jetson benchmark.

This module runs inside a *worker* process in the virtual environment that
owns the model's dependencies, so a missing ARM wheel or a CUDA OOM in one
engine cannot take down the whole experiment.

Timing boundaries
-----------------
``model_forward``
    The model computation alone, CUDA-synchronised either side. For Ultralytics
    this is its own reported ``inference`` time; for the attribute classifier it
    is the forward pass over pre-transformed tensors.

``in_memory_pipeline``
    An already-decoded input through the model's own preprocessing, inference
    and postprocessing. Both detectors receive a *pre-decoded* image so neither
    framework is handed a file path while the other is handed a decoded array.

``end_to_end_file``
    SSD read plus decode plus the whole pipeline. OS page caching applies and is
    acknowledged in the report; no privileged cache flushing is attempted.

The workstation benchmark measured a different boundary per engine (YOLO from
file paths, RF-DETR from pre-loaded PIL images, attribute from pre-transformed
tensors, prompt from pre-loaded arrays). ``COMPARABLE_BOUNDARY`` records which
Jetson boundary reproduces it, and only that one is used for the cross-device
comparison.
"""

from __future__ import annotations

import gc
import statistics
import sys
import time
from pathlib import Path

# The comparison-relevant boundary per engine, chosen to reproduce exactly what
# the completed RTX 4090 benchmark timed for that engine.
COMPARABLE_BOUNDARY = {
    "yolo": "end_to_end_file",        # workstation passed file paths to predict()
    "rfdetr": "in_memory_pipeline",   # workstation passed pre-loaded PIL images
    "attribute": "model_forward",     # workstation timed the forward over tensors
    "prompt": "in_memory_pipeline",   # workstation passed pre-loaded numpy arrays
    # TensorRT is a separate optimised runtime; it keeps YOLO's boundary so its
    # rows are internally consistent, but it is never joined to the native
    # comparison (rows carry runtime_backend=tensorrt).
    "yolo_tensorrt": "end_to_end_file",
}

ITEM_UNIT = {"yolo": "image", "rfdetr": "image", "yolo_tensorrt": "image",
             "attribute": "crop", "prompt": "image_prompt_pair"}


# ---------------------------------------------------------------------------
# CUDA helpers
# ---------------------------------------------------------------------------

def cuda_available(device: str) -> bool:
    if not device.startswith("cuda"):
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def synchronise(device: str) -> None:
    if cuda_available(device):
        import torch
        torch.cuda.synchronize()


def reset_memory_stats(device: str) -> None:
    if cuda_available(device):
        import torch
        torch.cuda.reset_peak_memory_stats()


def torch_memory(device: str) -> dict:
    if not cuda_available(device):
        return {"torch_peak_allocated_mb": None, "torch_peak_reserved_mb": None}
    import torch
    return {
        "torch_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "torch_peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 2),
        "torch_current_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
    }


def release(device: str) -> None:
    """Release a model between runs: drop references, collect, empty the cache."""
    gc.collect()
    if cuda_available(device):
        import torch
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()


def process_rss_mb() -> float | None:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                try:
                    return round(int(line.split()[1]) / 1024, 2)
                except (IndexError, ValueError):
                    return None
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1024**2, 2)
    except Exception:
        return None


def system_ram_used_mb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    values = {}
    for line in meminfo.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":"):
            try:
                values[parts[0][:-1]] = int(parts[1])
            except ValueError:
                continue
    if "MemTotal" in values and "MemAvailable" in values:
        return round((values["MemTotal"] - values["MemAvailable"]) / 1024, 2)
    return None


# ---------------------------------------------------------------------------
# Timed measurement primitive
# ---------------------------------------------------------------------------

def measure(callable_per_item, items: list, device: str, warmup: int,
            boundary: str, *, min_seconds: float = 0.0, max_seconds: float | None = None,
            max_items: int | None = None) -> dict:
    """Time ``callable_per_item(item)`` at batch size 1, CUDA-synchronised.

    ``min_seconds`` turns the call into a sustained pass: the deterministic item
    list is cycled until the minimum duration is reached (bounded by
    ``max_seconds`` / ``max_items`` so a pathological model cannot run forever).

    Returns per-item latencies plus the monotonic window bounds, which the
    orchestrator uses to align tegrastats telemetry recorded in another process.
    """
    if not items:
        return {"status": "no_items", "latencies_ms": [], "timed_items": 0}

    for index in range(max(0, warmup)):
        callable_per_item(items[index % len(items)])
    synchronise(device)
    reset_memory_stats(device)

    latencies: list[float] = []
    indices: list[int] = []
    start_monotonic = time.monotonic()
    wall_start = time.perf_counter()
    position = 0
    while True:
        item_index = position % len(items)
        synchronise(device)
        t0 = time.perf_counter()
        callable_per_item(items[item_index])
        synchronise(device)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        indices.append(item_index)
        position += 1
        elapsed = time.perf_counter() - wall_start
        if position >= len(items) and elapsed >= min_seconds:
            break
        if max_items is not None and position >= max_items:
            break
        if max_seconds is not None and elapsed >= max_seconds:
            break
    wall_total = time.perf_counter() - wall_start
    end_monotonic = time.monotonic()

    return {
        "status": "ok",
        "timing_boundary": boundary,
        "latencies_ms": [round(v, 4) for v in latencies],
        "sample_indices": indices,
        "timed_items": len(latencies),
        "timed_seconds": round(wall_total, 4),
        "throughput_items_s": round(len(latencies) / wall_total, 3) if wall_total else None,
        "start_monotonic": start_monotonic,
        "end_monotonic": end_monotonic,
        "warmup": warmup,
        **torch_memory(device),
    }


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

class Engine:
    """Base engine: load once, expose one callable per timing boundary."""

    engine_key = ""

    def __init__(self, spec: dict, device: str):
        self.spec = spec
        self.device = device
        self.cold_start_s: float | None = None
        self.parameters_total: int | None = None
        self.parameters_trainable: int | None = None
        self.notes: list[str] = []
        self.extra: dict = {}

    def load(self) -> None:
        raise NotImplementedError

    def boundaries(self) -> dict:
        """Mapping ``boundary name -> (items, callable_per_item)``."""
        raise NotImplementedError

    def smoke(self) -> dict:
        raise NotImplementedError

    def close(self) -> None:
        release(self.device)


class YoloEngine(Engine):
    engine_key = "yolo"

    def load(self) -> None:
        from ultralytics import YOLO

        t0 = time.perf_counter()
        self.model = YOLO(self.spec["checkpoint"])
        self.model.to(self.device)
        synchronise(self.device)
        self.cold_start_s = round(time.perf_counter() - t0, 4)
        self.parameters_total = int(sum(p.numel() for p in self.model.model.parameters()))
        self.parameters_trainable = int(
            sum(p.numel() for p in self.model.model.parameters() if p.requires_grad))
        self.imgsz = int(self.spec.get("input_resolution") or 640)
        self.internal = {"preprocess": [], "inference": [], "postprocess": []}
        try:
            from ultralytics import __version__ as version
            self.extra["ultralytics_version"] = version
        except Exception:
            pass

    def _record_speed(self, results) -> None:
        for result in results if isinstance(results, list) else [results]:
            speed = getattr(result, "speed", None) or {}
            for key in self.internal:
                value = speed.get(key)
                if value is not None:
                    self.internal[key].append(float(value))

    def _predict_path(self, path: Path):
        results = self.model.predict(str(path), imgsz=self.imgsz, device=self.device,
                                     verbose=False)
        self._record_speed(results)
        return results

    def _predict_array(self, array):
        results = self.model.predict(array, imgsz=self.imgsz, device=self.device,
                                     verbose=False)
        self._record_speed(results)
        return results

    def boundaries(self) -> dict:
        import cv2

        paths = [Path(p) for p in self.spec["items"]]
        # Pre-decoded BGR arrays: both detectors receive decoded pixels, so the
        # in-memory boundary is comparable across frameworks.
        decoded = [cv2.imread(str(p)) for p in paths]
        missing = [str(p) for p, a in zip(paths, decoded) if a is None]
        if missing:
            raise RuntimeError(f"could not decode {len(missing)} benchmark image(s), "
                               f"first: {missing[0]}")
        return {
            "in_memory_pipeline": (decoded, self._predict_array),
            "end_to_end_file": (paths, self._predict_path),
        }

    def internal_breakdown(self) -> dict:
        return {f"ultralytics_{key}_ms": round(statistics.fmean(values), 4)
                for key, values in self.internal.items() if values}

    def reset_internal(self) -> None:
        for values in self.internal.values():
            values.clear()

    def smoke(self) -> dict:
        path = Path(self.spec["items"][0])
        results = self._predict_path(path)
        synchronise(self.device)
        first = results[0] if isinstance(results, list) else results
        boxes = getattr(first, "boxes", None)
        if boxes is None:
            raise RuntimeError("YOLO result has no `boxes` attribute - unexpected output")
        self.reset_internal()
        return {"detections": int(len(boxes)), "output": "ultralytics Results with boxes"}


class RfdetrEngine(Engine):
    engine_key = "rfdetr"

    def load(self) -> None:
        import rfdetr

        scale = str(self.spec.get("scale", "")).lower()
        model_class = {"n": rfdetr.RFDETRNano, "s": rfdetr.RFDETRSmall,
                       "m": rfdetr.RFDETRMedium}.get(scale)
        if model_class is None:
            raise RuntimeError(f"unknown RF-DETR scale {scale!r}")
        resolution = self.spec.get("input_resolution")
        kwargs = {"pretrain_weights": self.spec["checkpoint"], "device": self.device}
        if resolution:
            # RF-DETR is always run at the resolution it was trained/evaluated
            # at; it is never forced to a detector-wide 1280.
            kwargs["resolution"] = int(resolution)
        t0 = time.perf_counter()
        self.model = model_class(**kwargs)
        synchronise(self.device)
        self.cold_start_s = round(time.perf_counter() - t0, 4)
        try:
            parameters = list(self.model.model.model.parameters())
            self.parameters_total = int(sum(p.numel() for p in parameters))
            self.parameters_trainable = int(
                sum(p.numel() for p in parameters if p.requires_grad))
        except Exception:
            self.notes.append("RF-DETR parameter count not recoverable from this build")
        self.threshold = float(self.spec.get("conf_threshold", 0.30))
        self.notes.append("rfdetr .predict() is a single-image API; batch size is 1")
        try:
            import importlib.metadata as metadata
            self.extra["rfdetr_version"] = metadata.version("rfdetr")
        except Exception:
            pass

    def _predict_image(self, image):
        return self.model.predict(image, threshold=self.threshold)

    def _predict_path(self, path: Path):
        from PIL import Image
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        return self.model.predict(image, threshold=self.threshold)

    def boundaries(self) -> dict:
        from PIL import Image

        paths = [Path(p) for p in self.spec["items"]]
        loaded = []
        for path in paths:
            with Image.open(path) as handle:
                loaded.append(handle.convert("RGB"))
        return {
            "in_memory_pipeline": (loaded, self._predict_image),
            "end_to_end_file": (paths, self._predict_path),
        }

    def smoke(self) -> dict:
        detections = self._predict_path(Path(self.spec["items"][0]))
        synchronise(self.device)
        if not hasattr(detections, "xyxy"):
            raise RuntimeError("RF-DETR result has no `xyxy` attribute - unexpected output")
        return {"detections": int(len(detections.xyxy)),
                "output": "supervision Detections with xyxy"}


class AttributeEngine(Engine):
    """Reuses the repository's own attribute-classification construction path.

    The backbone builder, LoRA adaptation, multi-head classifier, transforms and
    checkpoint loader all come from ``mtsd_attr``; no second model-construction
    path is invented here.
    """

    engine_key = "attribute"

    def load(self) -> None:
        import torch

        attrcls_dir = self.spec["attrcls_dir"]
        if attrcls_dir not in sys.path:
            sys.path.insert(0, attrcls_dir)
        from mtsd_attr.backbones import build_backbone
        from mtsd_attr.config import adaptation_of, load_config
        from mtsd_attr.dataset import build_transforms
        from mtsd_attr.multihead_model import MultiHeadClassifier
        from mtsd_attr.train_common import _load_checkpoint_state

        ram_before = system_ram_used_mb()
        rss_before = process_rss_mb()
        t0 = time.perf_counter()
        checkpoint = torch.load(self.spec["checkpoint"], map_location="cpu",
                                weights_only=False)
        model_cfg = checkpoint["model_cfg"]
        adaptation = checkpoint.get("adaptation") or adaptation_of(model_cfg)
        backbone = build_backbone(model_cfg)
        model = MultiHeadClassifier(backbone, checkpoint["attributes"],
                                    checkpoint["probe"], adaptation)
        _load_checkpoint_state(model, checkpoint["model_state"], adaptation)
        self.model = model.to(self.device).eval()
        synchronise(self.device)
        self.cold_start_s = round(time.perf_counter() - t0, 4)

        self.transform = build_transforms(backbone.image_size, False,
                                          load_config()["training"]["augmentation"])
        self.image_size = int(backbone.image_size)
        self.adaptation = adaptation
        self.parameters_total = int(sum(p.numel() for p in self.model.parameters()))
        self.parameters_trainable = int(
            sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        # A LoRA checkpoint stores only adapters and heads: its file size is not
        # the deployed model footprint, because the pretrained backbone must be
        # instantiated as well. Both numbers are reported separately.
        self.extra["adapter_checkpoint_mb"] = round(
            Path(self.spec["checkpoint"]).stat().st_size / 1024**2, 3)
        self.extra["instantiated_ram_delta_mb"] = (
            round(system_ram_used_mb() - ram_before, 2)
            if ram_before is not None and system_ram_used_mb() is not None else None)
        self.extra["instantiated_process_rss_delta_mb"] = (
            round(process_rss_mb() - rss_before, 2)
            if rss_before is not None and process_rss_mb() is not None else None)
        self.extra["attribute_heads"] = list(checkpoint["attributes"])
        self.notes.append(
            "checkpoint stores adapters/heads only; the pretrained backbone is "
            "instantiated separately, so checkpoint size is not the deployed footprint")

    def _forward(self, tensor):
        import torch
        with torch.inference_mode():
            return self.model(tensor.unsqueeze(0).to(self.device))

    def _transform_forward(self, image):
        return self._forward(self.transform(image))

    def _file_forward(self, path: Path):
        from PIL import Image
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        return self._forward(self.transform(image))

    def boundaries(self) -> dict:
        from PIL import Image

        paths = [Path(p) for p in self.spec["items"]]
        images = []
        for path in paths:
            with Image.open(path) as handle:
                images.append(handle.convert("RGB"))
        tensors = [self.transform(image) for image in images]
        return {
            "model_forward": (tensors, self._forward),
            "in_memory_pipeline": (images, self._transform_forward),
            "end_to_end_file": (paths, self._file_forward),
        }

    def smoke(self) -> dict:
        outputs = self._file_forward(Path(self.spec["items"][0]))
        synchronise(self.device)
        if not isinstance(outputs, dict) or not outputs:
            raise RuntimeError("attribute model returned no per-head outputs")
        shapes = {head: list(value.shape) for head, value in outputs.items()}
        return {"heads": sorted(outputs), "output_shapes": shapes}


class PromptEngine(Engine):
    """Zero-shot prompted localisation through the existing PromptDetect backend."""

    engine_key = "prompt"

    def load(self) -> None:
        promptdetect_dir = self.spec["promptdetect_dir"]
        if promptdetect_dir not in sys.path:
            sys.path.insert(0, promptdetect_dir)
        from backend import DetectionBackend

        self.backend = DetectionBackend(device=self.device)
        label = self.spec["backend_label"]
        t0 = time.perf_counter()
        status = self.backend.load(label)
        synchronise(self.device)
        self.cold_start_s = round(time.perf_counter() - t0, 4)
        if not status.get("ok"):
            raise RuntimeError(f"model load failed: {status.get('error')}")
        self.prompt = self.spec["prompt"]
        self.threshold = float(self.spec.get("conf_threshold", 0.30))
        self.breakdown = {"preprocess": [], "inference": [], "postprocess": []}
        self.notes.append(
            f"latency is per image-prompt pair for prompt='{self.prompt}'; a single "
            "query does not return the whole municipal taxonomy")

    def _predict_array(self, array):
        result = self.backend.predict(image=array, text_prompt=self.prompt,
                                      conf_threshold=self.threshold)
        for key in self.breakdown:
            self.breakdown[key].append(result["runtime"][key] * 1000.0)
        return result

    def _predict_path(self, path: Path):
        import numpy as np
        from PIL import Image
        with Image.open(path) as handle:
            array = np.array(handle.convert("RGB"))
        return self._predict_array(array)

    def boundaries(self) -> dict:
        import numpy as np
        from PIL import Image

        paths = [Path(p) for p in self.spec["items"]]
        arrays = []
        for path in paths:
            with Image.open(path) as handle:
                arrays.append(np.array(handle.convert("RGB")))
        return {
            "in_memory_pipeline": (arrays, self._predict_array),
            "end_to_end_file": (paths, self._predict_path),
        }

    def internal_breakdown(self) -> dict:
        return {f"backend_{key}_ms": round(statistics.fmean(values), 4)
                for key, values in self.breakdown.items() if values}

    def reset_internal(self) -> None:
        for values in self.breakdown.values():
            values.clear()

    def smoke(self) -> dict:
        result = self._predict_path(Path(self.spec["items"][0]))
        synchronise(self.device)
        for key in ("boxes", "scores", "labels", "image_size"):
            if key not in result:
                raise RuntimeError(f"prompt backend result is missing {key!r}")
        self.reset_internal()
        return {"detections": len(result["boxes"]), "prompt": self.prompt}

    def close(self) -> None:
        # LocateAnything spawns a worker process; close it explicitly so no
        # background process survives this benchmark.
        try:
            self.backend._loaded.engine.close()  # noqa: SLF001 - no public API
        except Exception:
            pass
        super().close()


class TensorRTYoloEngine(YoloEngine):
    """Optional deployment-optimised YOLO runtime.

    This is a SEPARATE experiment, never a prerequisite for the native
    benchmark: its rows carry ``runtime_backend=tensorrt`` and are never joined
    to the native PyTorch comparison. A failure here cannot affect the native
    result, because it runs as its own worker process on its own spec.

    The engine is built on the Jetson itself (a TensorRT plan is specific to the
    device, the TensorRT version and the input shape), cached on the SSD, and
    keyed by checkpoint hash + input size + precision so an incompatible cached
    engine is never silently reused. INT8 is out of scope: no calibration
    protocol exists in this repository and one is not invented here.
    """

    engine_key = "yolo_tensorrt"

    def load(self) -> None:
        # Validate the requested precision before importing anything: an
        # unsupported precision is a configuration error, not a dependency one.
        precision = str(self.spec.get("precision", "fp16")).lower()
        if precision not in ("fp16", "fp32"):
            raise RuntimeError(
                f"unsupported TensorRT precision {precision!r}; INT8 is out of scope "
                "because this repository has no calibration protocol")
        self.precision = precision

        from ultralytics import YOLO

        checkpoint = Path(self.spec["checkpoint"])
        self.imgsz = int(self.spec.get("input_resolution") or 640)

        cache_dir = Path(self.spec["export_cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = (self.spec.get("checkpoint_sha256") or "nohash")[:16]
        engine_path = cache_dir / (
            f"{checkpoint.stem}.{digest}.img{self.imgsz}.{precision}."
            f"trt{self.spec.get('tensorrt_version', 'unknown')}.engine")

        self.extra["tensorrt_engine"] = str(engine_path)
        self.extra["tensorrt_precision"] = precision
        self.extra["tensorrt_version"] = self.spec.get("tensorrt_version")
        self.extra["cuda_version"] = self.spec.get("cuda_version")

        if engine_path.is_file():
            self.extra["tensorrt_export_seconds"] = 0.0
            self.extra["tensorrt_engine_reused"] = True
            self.notes.append(f"reused cached TensorRT engine {engine_path.name}")
        else:
            self.extra["tensorrt_engine_reused"] = False
            export_start = time.perf_counter()
            exporter = YOLO(str(checkpoint))
            produced = exporter.export(format="engine", imgsz=self.imgsz,
                                       half=(precision == "fp16"),
                                       device=self.device, verbose=False)
            self.extra["tensorrt_export_seconds"] = round(
                time.perf_counter() - export_start, 3)
            produced_path = Path(str(produced))
            if not produced_path.is_file():
                raise RuntimeError(f"TensorRT export produced no engine at {produced_path}")
            produced_path.replace(engine_path)
            del exporter
            release(self.device)
            self.notes.append(
                f"exported TensorRT {precision} engine in "
                f"{self.extra['tensorrt_export_seconds']}s")

        t0 = time.perf_counter()
        self.model = YOLO(str(engine_path), task="detect")
        synchronise(self.device)
        self.cold_start_s = round(time.perf_counter() - t0, 4)
        # A serialized plan carries no Python parameter list; the parameter
        # count is the native model's and is recorded from the spec instead of
        # being reported as zero.
        self.parameters_total = self.spec.get("parameters_total")
        self.parameters_trainable = None
        self.internal = {"preprocess": [], "inference": [], "postprocess": []}
        self.notes.append(
            "TensorRT is a separate optimised runtime; these rows carry "
            "runtime_backend=tensorrt and are not mixed into the native comparison")

    def smoke(self) -> dict:
        """Smoke plus a prediction parity check against the native checkpoint."""
        detail = super().smoke()
        detail["tensorrt_engine"] = self.extra.get("tensorrt_engine")
        if not self.spec.get("parity_check", True):
            detail["parity"] = {"status": "skipped_by_configuration"}
            return detail
        try:
            detail["parity"] = self._parity(Path(self.spec["items"][0]))
        except Exception as exc:  # noqa: BLE001 - parity is diagnostic, not fatal
            detail["parity"] = {"status": "parity_check_failed",
                                "error": f"{type(exc).__name__}: {exc}"}
        return detail

    def _parity(self, image_path: Path) -> dict:
        """Compare TensorRT and native detections on one representative image."""
        from ultralytics import YOLO

        threshold = float(self.spec.get("parity_iou_threshold", 0.5))
        trt = self.model.predict(str(image_path), imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]
        native_model = YOLO(self.spec["checkpoint"])
        native_model.to(self.device)
        native = native_model.predict(str(image_path), imgsz=self.imgsz,
                                      device=self.device, verbose=False)[0]
        trt_boxes = trt.boxes.xyxy.detach().cpu().tolist() if trt.boxes is not None else []
        native_boxes = (native.boxes.xyxy.detach().cpu().tolist()
                        if native.boxes is not None else [])
        matched = sum(1 for box in trt_boxes
                      if any(_iou(box, other) >= threshold for other in native_boxes))
        del native_model
        release(self.device)
        self.reset_internal()
        return {
            "status": "ok",
            "iou_threshold": threshold,
            "tensorrt_detections": len(trt_boxes),
            "native_detections": len(native_boxes),
            "matched_detections": matched,
            "match_rate": round(matched / len(trt_boxes), 4) if trt_boxes else None,
            "note": ("a parity check on one representative image; it is a sanity "
                     "check, not an accuracy evaluation"),
        }


def _iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a[:4]
    bx0, by0, bx1, by1 = b[:4]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    union = ((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection)
    return intersection / union if union > 0 else 0.0


ENGINES = {
    "yolo": YoloEngine,
    "rfdetr": RfdetrEngine,
    "attribute": AttributeEngine,
    "prompt": PromptEngine,
    "yolo_tensorrt": TensorRTYoloEngine,
}


def build_engine(spec: dict, device: str) -> Engine:
    engine_class = ENGINES.get(spec["engine"])
    if engine_class is None:
        raise RuntimeError(f"unknown engine {spec['engine']!r}")
    return engine_class(spec, device)
