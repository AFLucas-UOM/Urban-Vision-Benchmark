"""
backend.py — Inference backend for the PromptDetect evaluation tool.

This module exposes a single public class, :class:`DetectionBackend`, plus a
small model registry (:data:`MODELS`). The UI picks a model by its label
("SAM 3" / "SAM 3.1" / "Cosmos Reason2" / "LocateAnything 3B"); the backend
loads it on demand and
runs a uniform ``predict()`` regardless of which model is active.

Supported models
-----------------
SAM 3            — facebook/sam3             (Meta native builder, text grounding)
SAM 3.1          — facebook/sam3.1           (same builder + sam3.1 checkpoint)
Cosmos Reason2   — nvidia/Cosmos-Reason2-2B/8B/32B
                                             (Qwen3-VL VLM, JSON bbox grounding;
                                              2B/8B fit a 24 GB GPU, 32B needs
                                              multi-GPU or CPU offload)

End-to-end flow per family
---------------------------
SAM 3 / SAM 3.1 (shared native engine):
    text + image
        -> Sam3Processor.set_image(image)            # preprocess (bf16 autocast)
        -> Sam3Processor.set_text_prompt(prompt)     # detect + segment in one pass
        -> boxes (xyxy px) + scores + masks
        -> filter_detections                          # confidence / area / count

Cosmos Reason2 (vision-language grounding):
    text + image
        -> Qwen3VLProcessor.apply_chat_template(...)  # "report bbox in JSON"
        -> Qwen3VLForConditionalGeneration.generate
        -> parse JSON [{bbox_2d, label}, ...]
        -> filter_detections

Why SAM 3 uses Meta's native package (not transformers)
-------------------------------------------------------
SAM 3.1's weights ship **only** as the native ``sam3.1_multiplex.pt`` checkpoint
(no transformers ``model.safetensors``), so it can only be loaded through Meta's
``sam3`` package. To keep SAM 3 and SAM 3.1 directly comparable, SAM 3 uses the
same native engine. On Windows the package needs ``triton-windows`` and
``setuptools<81`` (see Documents/PromptDetect.md); inference runs under bf16 autocast.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from utils import filter_detections

logger = logging.getLogger("promptdetect")

ProgressFn = Callable[[float, str], None]


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# One place to declare the selectable models. To add a model (or switch a
# Cosmos size, e.g. -8B / -32B), edit an entry here; the UI builds its dropdown
# from MODELS.keys().

@dataclass(frozen=True)
class ModelSpec:
    label: str                    # UI label
    family: str                   # "sam3" | "cosmos" | "locate"
    source: str                   # native version ("sam3"/"sam3.1") or HF repo id
    notes: str = ""               # shown in the UI
    has_confidence: bool = True   # does the model emit real per-box scores?
    #   SAM 3/3.1 → True (detector scores in [0,1]).
    #   Cosmos / LocateAnything → False: generative VLMs with no per-box
    #   confidence, so the backend assigns 1.0 and the UI hides the value
    #   (a constant 1.0 would be misleading rather than informative).


MODELS: Dict[str, ModelSpec] = {
    "SAM 3": ModelSpec(
        label="SAM 3",
        family="sam3",
        source="sam3",
        notes="Meta SAM 3 — native text-grounded detection + segmentation.",
    ),
    "SAM 3.1": ModelSpec(
        label="SAM 3.1",
        family="sam3",
        source="sam3.1",
        notes="Meta SAM 3.1 — updated checkpoints, same prompting API.",
    ),
    # Cosmos Reason2 sizes — all share the Qwen3-VL architecture and the same
    # CosmosReason2Engine; only the checkpoint (and so VRAM cost) differs.
    "Cosmos Reason2 2B": ModelSpec(
        label="Cosmos Reason2 2B",
        family="cosmos",
        source="nvidia/Cosmos-Reason2-2B",
        notes="NVIDIA Cosmos Reason2 2B (Qwen3-VL) — boxes only. Fits a 24 GB GPU.",
        has_confidence=False,
    ),
    "Cosmos Reason2 8B": ModelSpec(
        label="Cosmos Reason2 8B",
        family="cosmos",
        source="nvidia/Cosmos-Reason2-8B",
        notes="NVIDIA Cosmos Reason2 8B (Qwen3-VL) — boxes only. ~16 GB fp16, fits a 24 GB GPU.",
        has_confidence=False,
    ),
    "Cosmos Reason2 32B": ModelSpec(
        label="Cosmos Reason2 32B",
        family="cosmos",
        source="nvidia/Cosmos-Reason2-32B",
        notes="NVIDIA Cosmos Reason2 32B (Qwen3-VL) — boxes only. ~64 GB fp16; "
              "needs multi-GPU or CPU offload (slow on a single 24 GB GPU).",
        has_confidence=False,
    ),
    "LocateAnything 3B": ModelSpec(
        label="LocateAnything 3B",
        family="locate",
        source="nvidia/LocateAnything-3B",
        notes="NVIDIA LocateAnything-3B (Eagle) — dedicated visual grounding "
              "(MoonViT + Qwen2.5-3B). Boxes only. ~6 GB, fits a 24 GB GPU.",
        has_confidence=False,
    ),
}

DEFAULT_MODEL = "SAM 3"


# ---------------------------------------------------------------------------
# Optional-dependency probing (cheap, import-time)
# ---------------------------------------------------------------------------

def _have(module: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module) is not None


_HAS_SAM3_PKG = _have("sam3")
_HAS_TRANSFORMERS = _have("transformers")


# ---------------------------------------------------------------------------
# Engines — one per loading/inference strategy
# ---------------------------------------------------------------------------

class _Engine:
    """Base class. Subclasses load a model and return raw detections."""

    source: str = ""        # human-readable weights source (path or repo id)
    detail: str = ""        # short description of the active pipeline

    def load(self, progress: ProgressFn) -> None:
        raise NotImplementedError

    def predict_raw(
        self, image: np.ndarray, text_prompt: str, conf_threshold: float,
    ) -> Tuple[List[List[float]], List[float], List[str], List[Optional[np.ndarray]]]:
        """Return (boxes_xyxy_pixels, scores, labels, masks)."""
        raise NotImplementedError

    def close(self) -> None:
        """Release resources (GPU memory, worker processes). Default: no-op."""
        pass


class Sam3NativeEngine(_Engine):
    """
    SAM 3 / SAM 3.1 via Meta's official ``sam3`` package.

    Both versions build the same image model; only the checkpoint differs
    (``sam3.pt`` vs ``sam3.1_multiplex.pt``, auto-downloaded from the gated HF
    repo). Inference runs the detector + segmentation head in a single pass
    under bf16 autocast and returns boxes, scores, and masks together.
    """

    def __init__(self, version: str, device: str) -> None:
        self.version = version            # "sam3" | "sam3.1"
        self.device = device
        self._processor: Any = None

    def load(self, progress: ProgressFn) -> None:
        if not _HAS_SAM3_PKG:
            raise RuntimeError(
                "The 'sam3' package is not installed. Install it with "
                "`pip install git+https://github.com/facebookresearch/sam3.git` "
                "(plus `triton-windows` and `setuptools<81` on Windows)."
            )
        from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
        from sam3.model.sam3_image_processor import Sam3Processor

        progress(0.2, f"Downloading {self.version} checkpoint…")
        checkpoint = download_ckpt_from_hf(version=self.version)

        progress(0.55, f"Building {self.version} image model…")
        model = build_sam3_image_model(
            device=self.device,
            eval_mode=True,
            checkpoint_path=checkpoint,
            load_from_HF=False,
            enable_segmentation=True,
        )
        # Low internal threshold; we re-filter per call in filter_detections.
        self._processor = Sam3Processor(model, device=self.device, confidence_threshold=0.05)
        self.source = checkpoint
        self.detail = f"Meta sam3 builder ({self.version}), bf16 autocast"
        progress(0.95, "Model loaded ✓")

    @torch.inference_mode()
    def predict_raw(self, image, text_prompt, conf_threshold):
        pil = Image.fromarray(image).convert("RGB")
        h, w = image.shape[:2]

        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.startswith("cuda") else _nullcontext()
        )
        with autocast:
            state = self._processor.set_image(pil)
            out = self._processor.set_text_prompt(state=state, prompt=text_prompt.strip())

        boxes = [_normalise_box(b, w, h) for b in _to_list(out.get("boxes"))]
        scores = [float(s) for s in _to_list(out.get("scores"))]
        masks = [
            _resize_mask(m, w, h) if m is not None else None
            for m in _to_mask_list(out.get("masks"))
        ]
        labels = [text_prompt] * len(boxes)
        return boxes, scores, labels, masks


class CosmosReason2Engine(_Engine):
    """
    NVIDIA Cosmos Reason2 (Qwen3-VL) used as a promptable detector.

    The model is asked to locate objects matching the prompt and to report
    bounding boxes as JSON. We parse ``[{"bbox_2d": [x1,y1,x2,y2], "label": …}]``
    from its reply. It returns boxes only (no masks) and no per-box confidence,
    so every detection is assigned a score of 1.0 (the confidence slider has no
    effect for this model).

    Coordinates follow the Qwen3-VL convention — a fixed 0..1000 **normalised**
    grid (not absolute pixels), confirmed by scale-invariance testing. They are
    mapped back to image pixels in ``_parse_json_boxes`` and clamped to bounds.
    """

    _PROMPT = (
        'Locate every object that matches the description "{q}" in the image. '
        "Report bbox coordinates in JSON format as a list of "
        '{{"bbox_2d": [x1, y1, x2, y2], "label": "<name>"}}.'
    )

    def __init__(self, repo_id: str, device: str) -> None:
        self.repo_id = repo_id
        self.device = device
        self._model: Any = None
        self._processor: Any = None

    def load(self, progress: ProgressFn) -> None:
        if not _HAS_TRANSFORMERS:
            raise RuntimeError("'transformers>=4.57' is required for Cosmos Reason2.")
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        progress(0.2, f"Loading {self.repo_id}…")
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.repo_id,
            dtype=torch.float16,
            device_map="auto" if self.device.startswith("cuda") else None,
            attn_implementation="sdpa",
        ).eval()
        self._processor = AutoProcessor.from_pretrained(self.repo_id)
        self.source = self.repo_id
        self.detail = "Qwen3-VL grounding (JSON bbox), boxes only"
        progress(0.95, "Model loaded ✓")

    @torch.inference_mode()
    def predict_raw(self, image, text_prompt, conf_threshold):
        pil = Image.fromarray(image).convert("RGB")
        h, w = image.shape[:2]

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": self._PROMPT.format(q=text_prompt.strip())},
            ],
        }]
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self._model.device)

        generated = self._model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        reply = self._processor.decode(
            generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        boxes, labels = _parse_json_boxes(reply, w, h, default_label=text_prompt.strip())
        scores = [1.0] * len(boxes)           # VLM gives no per-box confidence
        masks: List[Optional[np.ndarray]] = []  # boxes only
        return boxes, scores, labels, masks


class LocateAnythingEngine(_Engine):
    """
    NVIDIA LocateAnything-3B (Eagle VLM family) used as a promptable detector.

    A *dedicated* visual-grounding model — MoonViT vision encoder + Qwen2.5-3B
    language model, with Parallel Box Decoding (PBD). Loaded as a custom
    ``trust_remote_code`` model via ``AutoModel`` with a **non-standard**
    ``generate()`` (takes ``pixel_values``, ``tokenizer``, ``generation_mode``
    and ``image_grid_hws`` directly). Boxes are emitted as special tokens
    ``<box><x1><y1><x2><y2></box>`` on a 0..1000 normalised grid; no per-box
    confidence (score = 1.0), boxes only.

    Transparent dual mode
    ---------------------
    LocateAnything's remote code is bound to transformers ~4.57, incompatible
    with the 5.x the rest of the app uses. This engine therefore runs in one of
    two modes, chosen automatically — the user never switches environments:

    * **in-process** — when the host env is already transformers ~4.57 (the
      ``mtsd-la`` env, or inside the worker). Loads + infers directly.
    * **worker** — otherwise (the normal ``mtsd-base`` env): launches
      :mod:`la_worker` in the ``mtsd-la`` conda env and proxies
      inference over localhost HTTP.
    """

    # Multi-instance detection template (see the model card's worker.detect()).
    _PROMPT = "Locate all the instances that match the following description: {q}."
    # Conda envs carrying transformers 4.57.x, tried in order; the legacy
    # pre-rename env name is kept as a fallback for older machines.
    _ENVS = ("mtsd-la", "promptdetect-la")
    # MoonViT runs dense SDPA at native resolution; cap the longest side so the
    # attention fits a 24 GB GPU. Boxes are on a 0..1000 grid, so this downscale
    # does not affect the coordinates mapped back to the original image.
    _MAX_SIDE = 1536

    def __init__(self, repo_id: str, device: str) -> None:
        self.repo_id = repo_id
        self.device = device
        self._dtype = torch.bfloat16
        # in-process state
        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
        # worker state
        self._mode: str = ""        # "inproc" | "worker"
        self._proc: Any = None
        self._base_url: Optional[str] = None
        self._logf: Any = None
        self._tmpdir: Optional[str] = None

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def load(self, progress: ProgressFn) -> None:
        if _la_should_run_inprocess():
            self._mode = "inproc"
            self._load_inprocess(progress)
        else:
            self._mode = "worker"
            self._start_worker(progress)

    def predict_raw(self, image, text_prompt, conf_threshold):
        if self._mode == "worker":
            return self._predict_worker(image, text_prompt, conf_threshold)
        return self._predict_inprocess(image, text_prompt, conf_threshold)

    def close(self) -> None:
        self._stop_worker()

    # ------------------------------------------------------------------
    # In-process path (host transformers ~4.57)
    # ------------------------------------------------------------------

    def _load_inprocess(self, progress: ProgressFn) -> None:
        if not _HAS_TRANSFORMERS:
            raise RuntimeError("'transformers' is required for LocateAnything-3B.")
        from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        progress(0.2, f"Loading {self.repo_id} (trust_remote_code)…")
        self._tokenizer = AutoTokenizer.from_pretrained(self.repo_id, trust_remote_code=True)
        self._processor = AutoProcessor.from_pretrained(self.repo_id, trust_remote_code=True)

        # transformers 5.x compatibility shims (no-ops on 4.57): the base
        # PreTrainedModel passes `allow_all_kernels` to the model's overridden
        # _check_and_adjust_attn_implementation(), which 4.57-era code rejects.
        model_cls = get_class_from_dynamic_module(
            "modeling_locateanything.LocateAnythingForConditionalGeneration",
            self.repo_id, trust_remote_code=True,
        )
        _shim_la_attn_impl(model_cls)

        # 5.x relocated Qwen2's top-level `rope_theta`; restore it from the
        # authoritative value for the vendored Qwen2 code (no-op on 4.57).
        config = AutoConfig.from_pretrained(self.repo_id, trust_remote_code=True)
        _patch_la_rope_config(getattr(config, "text_config", None))

        progress(0.55, "Building LocateAnything model…")
        try:
            model = AutoModel.from_pretrained(
                self.repo_id, config=config, dtype=self._dtype, trust_remote_code=True,
            )
        except (TypeError, AttributeError) as exc:
            import transformers
            raise RuntimeError(
                f"LocateAnything-3B failed to build under transformers "
                f"{transformers.__version__}. Its bundled remote code targets "
                "transformers ~4.57.x and breaks on 5.x internals (attention / "
                "RoPE / tied-weights APIs). Run it in an environment with "
                "`transformers==4.57.1` plus `decord lmdb peft`. "
                f"(underlying error: {type(exc).__name__}: {exc})"
            ) from exc
        self._model = (model.to(self.device) if self.device.startswith("cuda") else model).eval()
        self.source = self.repo_id
        self.detail = "LocateAnything-3B in-process (MoonViT + Qwen2.5), hybrid PBD"
        progress(0.95, "Model loaded ✓")

    @torch.inference_mode()
    def _predict_inprocess(self, image, text_prompt, conf_threshold):
        h, w = image.shape[:2]                       # original dims for box mapping
        pil = _downscale_max_side(Image.fromarray(image).convert("RGB"), self._MAX_SIDE)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": self._PROMPT.format(q=text_prompt.strip())},
            ],
        }]
        # LocateAnything ships its own chat-template + vision-info helpers.
        text = self._processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        imgs, vids = self._processor.process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=imgs, videos=vids, return_tensors="pt"
        ).to(self._model.device)

        response = self._model.generate(
            pixel_values=inputs["pixel_values"].to(self._dtype),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=self._tokenizer,
            max_new_tokens=2048,
            use_cache=True,
            generation_mode="hybrid",   # MTP w/ AR fallback (model's default)
            do_sample=False,            # deterministic for evaluation
            verbose=False,
        )
        answer = response[0] if isinstance(response, tuple) else response
        if isinstance(answer, (list, tuple)):
            answer = answer[0]

        boxes = _parse_box_tokens(str(answer), w, h)   # 0..1000 grid → original px
        scores = [1.0] * len(boxes)             # no per-box confidence
        labels = [text_prompt] * len(boxes)
        masks: List[Optional[np.ndarray]] = []  # boxes only
        return boxes, scores, labels, masks

    # ------------------------------------------------------------------
    # Worker path (host transformers 5.x → run model in the mtsd-la env)
    # ------------------------------------------------------------------

    def _start_worker(self, progress: ProgressFn) -> None:
        import atexit
        import os
        import subprocess
        import urllib.request

        env_name, env_py = next(
            ((name, py) for name in self._ENVS
             if (py := _find_env_python(name))),
            (self._ENVS[0], None),
        )
        if not env_py:
            raise RuntimeError(
                f"LocateAnything needs the '{env_name}' conda env (transformers "
                f"4.57.x), which was not found. Create it once via the setup "
                f"scripts in Scripts/Other-Scripts/CondaEnvironments/ (see its README), or "
                f"manually:\n"
                f"  conda create -n {env_name} python=3.12 -y\n"
                f"  conda activate {env_name}\n"
                f"  pip install torch torchvision --index-url "
                f"https://download.pytorch.org/whl/cu128\n"
                f"  pip install -r Requirements/requirements-locate-anything.txt"
            )
        self._env_name = env_name

        worker = str(Path(__file__).with_name("la_worker.py"))
        port = _free_port()
        self._base_url = f"http://127.0.0.1:{port}"

        log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
        log_path = log_dir / "la_worker.log"
        self._logf = open(log_path, "w", encoding="utf-8")
        env = dict(os.environ, PYTHONIOENCODING="utf-8",
                   HF_HUB_DISABLE_SYMLINKS_WARNING="1")

        progress(0.15, f"Starting LocateAnything worker in '{env_name}'…")
        self._proc = subprocess.Popen(
            [env_py, "-u", worker, "--port", str(port),
             "--repo", self.repo_id, "--device", self.device],
            stdout=self._logf, stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parent), env=env,
        )
        atexit.register(self._stop_worker)

        # Poll until the worker has loaded the model (~30–90 s) or fails.
        deadline = time.time() + 360
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"LocateAnything worker exited early (code {self._proc.returncode}). "
                    f"See {log_path}."
                )
            try:
                with urllib.request.urlopen(self._base_url + "/health", timeout=2) as r:
                    st = json.loads(r.read().decode())
                if st.get("error"):
                    self._stop_worker()
                    raise RuntimeError(f"LocateAnything worker failed to load: {st['error']}")
                if st.get("ready"):
                    self.source = self.repo_id
                    self.detail = (f"LocateAnything worker · {env_name} "
                                   f"(transformers 4.57) · localhost:{port}")
                    progress(0.97, "Worker ready ✓")
                    return
            except RuntimeError:
                raise
            except Exception:
                pass                       # worker HTTP not up yet
            progress(0.6, "Loading LocateAnything in worker…")
            time.sleep(2)

        self._stop_worker()
        raise RuntimeError("LocateAnything worker did not become ready within 360s.")

    def _predict_worker(self, image, text_prompt, conf_threshold):
        import os
        import tempfile
        import urllib.request

        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp(prefix="la_worker_")
        frame = os.path.join(self._tmpdir, "frame.png")
        Image.fromarray(image).convert("RGB").save(frame)

        payload = json.dumps({
            "image_path": frame, "prompt": text_prompt, "conf": conf_threshold,
        }).encode("utf-8")
        req = urllib.request.Request(
            self._base_url + "/predict", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(r.read().decode())
        if "error" in out:
            raise RuntimeError(f"LocateAnything worker: {out['error']}")

        boxes = [[float(v) for v in b] for b in out.get("boxes", [])]
        scores = [float(s) for s in out.get("scores", [])]
        labels = [str(lab) for lab in out.get("labels", [])]
        masks: List[Optional[np.ndarray]] = []
        return boxes, scores, labels, masks

    def _stop_worker(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                import urllib.request
                urllib.request.urlopen(self._base_url + "/shutdown", timeout=2).read()
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._proc = None
        if getattr(self, "_logf", None):
            try:
                self._logf.close()
            except Exception:
                pass
            self._logf = None


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

@dataclass
class _Loaded:
    label: str
    engine: _Engine


class DetectionBackend:
    """
    Loads a selected model and runs a uniform text-prompted prediction.

    Typical use::

        backend = DetectionBackend()
        backend.load("SAM 3")
        result = backend.predict(image, "garbage bag")
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded: Optional[_Loaded] = None
        logger.info("DetectionBackend ready | device=%s", self.device)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self, model_label: str, progress: Optional[ProgressFn] = None) -> Dict[str, Any]:
        """
        Load the model identified by *model_label* (a key in MODELS).
        Returns a status dict: {ok, label, source, detail, device, error}.
        """
        prog = progress or (lambda p, m: None)
        if model_label not in MODELS:
            return self._status(False, model_label, error=f"Unknown model '{model_label}'.")

        # Free the previously-loaded model first — releases GPU memory and shuts
        # down any LocateAnything worker process so the GPU isn't double-booked.
        if self._loaded is not None:
            try:
                self._loaded.engine.close()
            except Exception:
                pass
            self._loaded = None
            if self.device.startswith("cuda"):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        spec = MODELS[model_label]
        prog(0.05, f"Preparing {spec.label}…")
        try:
            engine = self._build_engine(spec)
            engine.load(prog)
            self._loaded = _Loaded(label=spec.label, engine=engine)
            prog(1.0, "Ready")
            return self._status(True, spec.label, source=engine.source, detail=engine.detail)
        except Exception as exc:
            logger.error("Failed to load %s: %s", spec.label, exc, exc_info=True)
            self._loaded = None
            return self._status(False, spec.label, error=str(exc))

    def _build_engine(self, spec: ModelSpec) -> _Engine:
        if spec.family == "sam3":
            return Sam3NativeEngine(spec.source, self.device)
        if spec.family == "cosmos":
            return CosmosReason2Engine(spec.source, self.device)
        if spec.family == "locate":
            return LocateAnythingEngine(spec.source, self.device)
        raise RuntimeError(f"Unsupported model family: {spec.family}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        image: np.ndarray,
        text_prompt: str,
        conf_threshold: float = 0.30,
        max_detections: int = 50,
        min_area: float = 100.0,
        max_area: float = float("inf"),
    ) -> Dict[str, Any]:
        """
        Run the active model on one image.

        Returns a dict with: boxes ([x1,y1,x2,y2] px), scores, labels,
        masks (np.ndarray|None), image_size (w, h) and a runtime breakdown.
        """
        t0 = time.perf_counter()

        # -- preprocess --
        t_pre = time.perf_counter()
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        h, w = image.shape[:2]
        t_pre_end = time.perf_counter()

        if self._loaded is None:
            return _empty(time.perf_counter() - t0, w, h)

        # -- inference --
        t_inf = time.perf_counter()
        try:
            boxes, scores, labels, masks = self._loaded.engine.predict_raw(
                image, text_prompt, conf_threshold
            )
        except Exception as exc:
            logger.error("Inference failed: %s", exc, exc_info=True)
            return _empty(time.perf_counter() - t0, w, h)
        t_inf_end = time.perf_counter()

        # -- postprocess / filter --
        t_post = time.perf_counter()
        f_boxes, f_scores, f_labels, f_masks = filter_detections(
            boxes, scores, labels, masks,
            conf_threshold=conf_threshold,
            min_area=min_area,
            max_area=max_area if max_area > 0 else float("inf"),
            max_detections=max_detections,
        )
        t_post_end = time.perf_counter()

        return {
            "boxes": f_boxes,
            "scores": f_scores,
            "labels": f_labels,
            "masks": f_masks,
            "image_size": (w, h),
            "has_confidence": self.active_has_confidence,
            "runtime": {
                "preprocess": t_pre_end - t_pre,
                "inference": t_inf_end - t_inf,
                "postprocess": t_post_end - t_post,
                "total": time.perf_counter() - t0,
            },
        }

    # ------------------------------------------------------------------
    # Status / properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._loaded is not None

    @property
    def active_has_confidence(self) -> bool:
        """True if the active model emits real per-box confidence scores."""
        if self._loaded is None:
            return True
        spec = MODELS.get(self._loaded.label)
        return spec.has_confidence if spec else True

    @property
    def active_label(self) -> str:
        return self._loaded.label if self._loaded else "none"

    @property
    def device_info(self) -> str:
        if self.device.startswith("cuda"):
            try:
                return f"CUDA — {torch.cuda.get_device_name(0)}"
            except Exception:
                return "CUDA"
        return "CPU"

    def _status(
        self, ok: bool, label: str,
        source: str = "", detail: str = "", error: str = "",
    ) -> Dict[str, Any]:
        return {
            "ok": ok, "label": label, "source": source,
            "detail": detail, "device": self.device_info, "error": error,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *exc): return False


def _parse_json_boxes(
    text: str, img_w: int, img_h: int, default_label: str,
    coord_grid: float = 1000.0,
) -> Tuple[List[List[float]], List[str]]:
    """
    Extract ``{"bbox_2d": [x1,y1,x2,y2], "label": …}`` records from a VLM reply.

    The reply may contain reasoning text and markdown fences, so we scan for
    individual bbox objects rather than requiring strict JSON.

    Qwen3-VL (and Cosmos Reason2 on top of it) reports coordinates on a fixed
    ``0..coord_grid`` **normalised** grid (1000 by default) — *not* absolute
    pixels. Verified by scale-invariance: the same image at full vs half size
    returns identical coordinates. We therefore map the grid to [0, 1] and let
    ``_normalise_box`` scale to image pixels. (Values already in [0, 1] are left
    as-is, so an occasional already-normalised reply is handled too.)
    """
    objs: List[dict] = []
    for match in re.finditer(r'\{[^{}]*"bbox_2d"[^{}]*\}', text, re.DOTALL):
        try:
            objs.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    if not objs:  # fall back to a whole JSON array
        arr = re.search(r"\[.*\]", text, re.DOTALL)
        if arr:
            try:
                parsed = json.loads(arr.group(0))
                objs = [o for o in parsed if isinstance(o, dict)]
            except json.JSONDecodeError:
                pass

    boxes: List[List[float]] = []
    labels: List[str] = []
    for o in objs:
        bb = o.get("bbox_2d") or o.get("bbox")
        if not isinstance(bb, (list, tuple)) or len(bb) < 4:
            continue
        vals = [float(v) for v in bb[:4]]
        if max(vals) > 1.0:                       # 0..coord_grid -> [0, 1]
            vals = [v / coord_grid for v in vals]
        boxes.append(_normalise_box(vals, img_w, img_h))
        labels.append(str(o.get("label", default_label)))
    return boxes, labels


def _shim_la_attn_impl(model_cls: type) -> None:
    """
    Make LocateAnything's remote modeling code tolerate transformers 5.x.

    The remote code targets transformers ~4.57; the 5.x base ``PreTrainedModel``
    calls ``_check_and_adjust_attn_implementation`` with a new ``allow_all_kernels``
    kwarg that the remote overrides don't declare. The model bundles several such
    overrides (``LocateAnything…``, its own ``Qwen2…`` and MoonViT classes), so we
    patch *every* remote class that defines the method to drop unknown kwargs.
    Idempotent; a no-op for classes whose signature is already compatible.
    """
    import inspect
    import sys as _sys

    targets: set = set(model_cls.__mro__)
    for name, mod in list(_sys.modules.items()):
        if mod is not None and "transformers_modules" in name and "LocateAnything" in name:
            for _, obj in inspect.getmembers(mod, inspect.isclass):
                targets.add(obj)

    for klass in targets:
        orig = klass.__dict__.get("_check_and_adjust_attn_implementation")
        if orig is None or getattr(orig, "_la_shimmed", False):
            continue

        def patched(self, *args, _orig=orig, **kwargs):
            kwargs.pop("allow_all_kernels", None)
            return _orig(self, *args, **kwargs)

        patched._la_shimmed = True
        klass._check_and_adjust_attn_implementation = patched


def _patch_la_rope_config(text_config: Any) -> None:
    """
    Restore the legacy top-level RoPE fields LocateAnything's vendored Qwen2 code
    expects (``rope_theta``, ``rope_scaling``).

    transformers 5.x moved ``rope_theta`` into the ``rope_parameters`` /
    ``rope_scaling`` dicts and auto-fills a ``{"rope_theta", "rope_type"}`` dict
    where the checkpoint originally had ``rope_scaling=None``. The vendored 4.57
    modeling code reads the old top-level names, so we set ``rope_theta`` from the
    *authoritative* value carried in the new dict, and collapse the auto-filled
    default ``rope_scaling`` back to ``None`` to match the trained config.
    """
    tc = text_config
    if tc is None:
        return
    if getattr(tc, "rope_theta", None) is None:
        params = getattr(tc, "rope_parameters", None) or getattr(tc, "rope_scaling", None)
        theta = params.get("rope_theta") if isinstance(params, dict) else None
        tc.rope_theta = float(theta) if theta is not None else 1_000_000.0
    rs = getattr(tc, "rope_scaling", None)
    # Only a plain default {rope_theta, rope_type:"default"} block was synthesised
    # by 5.x; the original checkpoint used rope_scaling=None (no scaling).
    if isinstance(rs, dict) and rs.get("rope_type") in (None, "default") \
            and set(rs) <= {"rope_theta", "rope_type"}:
        tc.rope_scaling = None


def _parse_box_tokens(text: str, img_w: int, img_h: int) -> List[List[float]]:
    """
    Extract LocateAnything boxes ``<box><x1><y1><x2><y2></box>`` from the reply.

    Coordinates are integers on a 0..1000 normalised grid; we map them to [0, 1]
    and let ``_normalise_box`` scale to pixels and clamp to the image bounds.
    (Four-number groups are boxes; the model also emits two-number ``<box>``
    *points*, which are ignored here since this tool works with boxes.)
    """
    boxes: List[List[float]] = []
    for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", text):
        x1, y1, x2, y2 = (int(g) / 1000.0 for g in m.groups())
        boxes.append(_normalise_box([x1, y1, x2, y2], img_w, img_h))
    return boxes


def _downscale_max_side(pil: "Image.Image", max_side: int) -> "Image.Image":
    """Downscale a PIL image so its longest side is <= max_side (no upscaling)."""
    w, h = pil.size
    longest = max(w, h)
    if longest <= max_side:
        return pil
    scale = max_side / float(longest)
    return pil.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                      resample=Image.Resampling.BICUBIC)


def _la_should_run_inprocess() -> bool:
    """True when LocateAnything can load in this process (transformers ~4.57)."""
    import os
    if os.environ.get("PROMPTDETECT_LA_WORKER") == "1":
        return True
    try:
        import transformers
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        return (major, minor) == (4, 57)
    except Exception:
        return False


def _free_port() -> int:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _find_env_python(env_name: str) -> Optional[str]:
    """Locate the python executable of a sibling conda env, or None if absent."""
    import os
    import sys

    def py_of(env_dir: Path) -> Path:
        return env_dir / "python.exe" if os.name == "nt" else env_dir / "bin" / "python"

    candidates: List[Path] = []
    exe = Path(sys.executable)
    if exe.parent.parent.name.lower() == "envs":          # .../envs/<cur>/python
        candidates.append(exe.parent.parent / env_name)
    cp = os.environ.get("CONDA_PREFIX")
    if cp:
        p = Path(cp)
        if p.parent.name.lower() == "envs":
            candidates.append(p.parent / env_name)
        candidates.append(p / "envs" / env_name)          # base-env case
    for root in (os.environ.get("CONDA_ROOT"),
                 Path.home() / "anaconda3", Path.home() / "miniconda3",
                 Path.home() / "miniforge3"):
        if root:
            candidates.append(Path(root) / "envs" / env_name)

    for env_dir in candidates:
        py = py_of(env_dir)
        if py.exists():
            return str(py)
    return None


def _to_list(value: Any) -> List:
    """Convert a tensor / ndarray / iterable to a plain Python list."""
    if value is None:
        return []
    if hasattr(value, "cpu"):
        value = value.cpu().float().numpy() if hasattr(value, "float") else value.cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value) if hasattr(value, "__iter__") else []


def _to_mask_list(value: Any) -> List[Optional[np.ndarray]]:
    """Convert a batch of masks (tensor / ndarray / list) to List[ndarray|None]."""
    if value is None or (hasattr(value, "__len__") and len(value) == 0):
        return []
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 4:          # (N, 1, H, W)
            value = value[:, 0]
        elif value.ndim == 2:        # single (H, W)
            return [value.astype(bool)]
        return [value[i].astype(bool) for i in range(len(value))]
    out: List[Optional[np.ndarray]] = []
    for m in value:
        if m is None:
            out.append(None)
        else:
            if hasattr(m, "cpu"):
                m = m.cpu().numpy()
            out.append(np.asarray(m).astype(bool))
    return out


def _resize_mask(mask: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Resize a binary mask to the original image size when needed."""
    mask = np.asarray(mask).astype(bool)
    if mask.shape[:2] == (img_h, img_w):
        return mask
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (img_w, img_h), resample=Image.Resampling.NEAREST
    )
    return np.asarray(resized).astype(bool)


def _normalise_box(box: Any, img_w: int, img_h: int) -> List[float]:
    """Convert a box from any common format to pixel [x1, y1, x2, y2]."""
    if hasattr(box, "cpu"):
        box = box.cpu().float().numpy()
    if isinstance(box, np.ndarray):
        box = box.tolist()
    x1, y1, x2, y2 = [float(v) for v in box[:4]]

    if max(x1, y1, x2, y2) <= 1.0:            # normalised -> pixels
        x1, x2 = x1 * img_w, x2 * img_w
        y1, y2 = y1 * img_h, y2 * img_h
    if x2 < x1 or y2 < y1:                    # cxcywh -> xyxy
        cx, cy, bw, bh = x1, y1, x2, y2
        x1, y1 = cx - bw / 2, cy - bh / 2
        x2, y2 = cx + bw / 2, cy + bh / 2

    return [max(0.0, x1), max(0.0, y1), min(float(img_w), x2), min(float(img_h), y2)]


def _empty(elapsed: float, w: int = 0, h: int = 0) -> Dict[str, Any]:
    return {
        "boxes": [], "scores": [], "labels": [], "masks": [],
        "image_size": (w, h),
        "runtime": {"preprocess": 0.0, "inference": elapsed, "postprocess": 0.0, "total": elapsed},
    }
