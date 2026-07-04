"""
app.py — PromptDetect: Promptable Detection Evaluation Interface (Gradio)

A research-grade tool for evaluating promptable perception models
(SAM 3 / SAM 3.1 / NVIDIA Cosmos Reason2 / LocateAnything) for object
localisation on the MDWD and MTSD datasets.

Workflow
--------
1. Pick a model (SAM 3 / SAM 3.1 / Cosmos Reason2 / LocateAnything) and click "Load Model".
2. Provide an image and a text prompt.
3. Inspect detections, masks, timing, and export the results.

Tabs
----
  1. Single Image      — one image, one prompt (primary tab)
  2. Prompt Comparison — same image, up to 4 prompts side-by-side
  3. Batch             — many images, one prompt -> results.csv
  4. Dataset           — folder scan + summary statistics + plots
  5. Analytics         — session-level prompt performance

Run
---
  python app.py [--host 0.0.0.0] [--port 7860] [--no-browser] [--share]
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backend import DEFAULT_MODEL, MODELS, DetectionBackend
from utils import (
    create_detection_dataframe,
    draw_detections,
    export_coco_json,
    export_csv,
    export_json,
    export_yolo,
    load_image_from_url,
    log_run,
    setup_logger,
)

# ---------------------------------------------------------------------------
# Directories, logging, global state
# ---------------------------------------------------------------------------

LOG_DIR = "logs"
EXPORT_DIR = "exports"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

logger = setup_logger(LOG_DIR)

# Single-user local tool — module-level state is intentional.
backend = DetectionBackend()
session_history: List[Dict[str, Any]] = []      # recent runs (newest first)
prompt_analytics: Dict[str, Dict[str, Any]] = {}  # prompt -> aggregate stats
MAX_HISTORY = 20

# Preset prompts for the two target datasets.
WASTE_PROMPTS = [
    "waste", "garbage", "rubbish", "domestic waste",
    "garbage bag", "black garbage bag", "orange garbage bag",
    "orange CMD bag", "recyclable material", "recycling bag",
    "organic waste", "litter", "household waste",
]
SIGN_PROMPTS = [
    "traffic sign", "stop sign", "road sign",
    "damaged traffic sign", "bent traffic sign",
    "circular traffic sign", "warning sign",
    "speed limit sign", "yield sign", "regulatory sign",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _format_status(status: Dict[str, Any]) -> str:
    if status["ok"]:
        return (
            f"Status: ready\n"
            f"Model: {status['label']}\n"
            f"Pipeline: {status['detail']}\n"
            f"Source: {status['source']}\n"
            f"Device: {status['device']}"
        )
    return (
        f"Status: load failed\n"
        f"Model: {status['label']}\n"
        f"Device: {status['device']}\n"
        f"Error: {status['error']}"
    )


def load_model(model_label: str, progress: gr.Progress = gr.Progress()) -> str:
    """Load the selected model and report its status."""
    progress(0.0, desc=f"Loading {model_label}…")
    status = backend.load(model_label, progress=lambda p, m: progress(p, desc=m))
    progress(1.0, desc="Done")
    return _format_status(status)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _record_run(prompt: str, num_det: int, scores: List[float], runtime: Dict) -> None:
    session_history.insert(0, {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prompt": prompt,
        "num_detections": num_det,
        "mean_confidence": round(float(np.mean(scores)), 4) if scores else 0.0,
        "total_runtime_s": round(runtime.get("total", 0), 4),
    })
    del session_history[MAX_HISTORY:]

    pa = prompt_analytics.setdefault(prompt, {"runs": 0, "total_detections": 0, "total_confidence": 0.0})
    pa["runs"] += 1
    pa["total_detections"] += num_det
    pa["total_confidence"] += float(np.mean(scores)) if scores else 0.0


def _history_choices() -> List[str]:
    return [f"{h['prompt']}  ({h['num_detections']} det)" for h in session_history]


def _stats_text(runtime: Dict, num_det: int, scores: List[float],
                has_confidence: bool = True) -> str:
    # Generative VLMs have no real per-box score; show "n/a" instead of a
    # misleading constant 1.0.
    if has_confidence:
        mc = float(np.mean(scores)) if scores else 0.0
        conf = f"Mean conf: {mc:.3f}"
    else:
        conf = "Confidence: n/a"
    return (
        f"Detections: {num_det}  |  {conf}  |  "
        f"Pre: {runtime.get('preprocess', 0)*1000:.1f} ms  |  "
        f"Inf: {runtime.get('inference', 0)*1000:.1f} ms  |  "
        f"Post: {runtime.get('postprocess', 0)*1000:.1f} ms  |  "
        f"Total: {runtime.get('total', 0)*1000:.1f} ms"
    )


def _not_ready_msg() -> str:
    return "No model loaded. Select a model above and click 'Load Model'."


# ---------------------------------------------------------------------------
# Tab 1 — Single Image
# ---------------------------------------------------------------------------

def run_single_inference(
    image: Optional[np.ndarray],
    prompt: str,
    conf_threshold: float,
    max_detections: int,
    min_area: float,
    max_area: float,
    show_masks: bool,
    image_url: str,
) -> Tuple[Any, Any, pd.DataFrame, str, str, Any]:
    """Returns: original, annotated, df, stats, prompt (unchanged), history update."""
    if image is None and image_url.strip():
        try:
            image = load_image_from_url(image_url.strip())
        except Exception as exc:
            return None, None, pd.DataFrame(), f"URL error: {exc}", prompt, gr.update()

    if image is None:
        return None, None, pd.DataFrame(), "No image provided.", prompt, gr.update()
    if not prompt.strip():
        return image, None, pd.DataFrame(), "Enter a prompt.", prompt, gr.update()
    if not backend.is_ready:
        return image, None, pd.DataFrame(), _not_ready_msg(), prompt, gr.update()

    result = backend.predict(
        image=image,
        text_prompt=prompt.strip(),
        conf_threshold=conf_threshold,
        max_detections=max_detections,
        min_area=min_area,
        max_area=max_area if max_area > 0 else float("inf"),
    )

    boxes, scores, labels, masks = (
        result["boxes"], result["scores"], result["labels"], result["masks"]
    )
    has_conf = result.get("has_confidence", True)
    annotated = draw_detections(
        image.copy(), boxes, labels, scores,
        masks=masks if show_masks else None, show_masks=show_masks,
        show_scores=has_conf,
    )
    df = create_detection_dataframe(boxes, scores, labels, prompt.strip(),
                                    include_confidence=has_conf)
    stats = (f"{_stats_text(result['runtime'], len(boxes), scores, has_conf)}"
             f"\nModel: {backend.active_label}")

    _record_run(prompt.strip(), len(boxes), scores, result["runtime"])
    log_run(logger, prompt.strip(), "uploaded_image", len(boxes), result["runtime"])

    return image, annotated, df, stats, prompt, gr.update(choices=_history_choices())


def clear_single() -> Tuple:
    return None, None, None, pd.DataFrame(), ""


def load_url_image(url: str) -> Optional[np.ndarray]:
    if url.strip():
        try:
            return load_image_from_url(url.strip())
        except Exception as exc:
            logger.warning("URL load failed: %s", exc)
    return None


def export_detections(
    df: Optional[pd.DataFrame], fmt: str, prompt: str, image: Optional[np.ndarray],
) -> Tuple[Optional[str], str]:
    """Write the latest detections to a file and return path + UI status."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None, "No detection results available. Run inference before exporting."

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = prompt.strip().replace(" ", "_")[:30] if prompt.strip() else "export"
    rows = df.to_dict(orient="records")

    if fmt == "CSV":
        out = export_csv(df, os.path.join(EXPORT_DIR, f"{slug}_{ts}.csv"))
        return out, f"Exported CSV: {out}"
    if fmt == "JSON":
        out = export_json(rows, os.path.join(EXPORT_DIR, f"{slug}_{ts}.json"))
        return out, f"Exported JSON: {out}"
    if fmt == "COCO JSON":
        det = [{
            "filename": "image",
            "width": image.shape[1] if image is not None else 0,
            "height": image.shape[0] if image is not None else 0,
            "boxes": [[r["Xmin"], r["Ymin"], r["Xmax"], r["Ymax"]] for r in rows],
            "scores": [r.get("Confidence", 1.0) for r in rows],
        }]
        out = export_coco_json(det, os.path.join(EXPORT_DIR, f"{slug}_coco_{ts}.json"))
        return out, f"Exported COCO JSON: {out}"
    if fmt == "YOLO":
        boxes = [[r["Xmin"], r["Ymin"], r["Xmax"], r["Ymax"]] for r in rows]
        iw = int(image.shape[1]) if image is not None else 640
        ih = int(image.shape[0]) if image is not None else 640
        out = export_yolo(boxes, iw, ih, os.path.join(EXPORT_DIR, f"{slug}_{ts}.txt"))
        return out, f"Exported YOLO labels: {out}"
    return None, f"Unsupported export format: {fmt}"


# ---------------------------------------------------------------------------
# Tab 2 — Prompt Comparison
# ---------------------------------------------------------------------------

def run_prompt_comparison(
    image: Optional[np.ndarray], prompts_text: str,
    conf_threshold: float, max_detections: int,
    min_area: float, max_area: float, show_masks: bool,
) -> Tuple:
    empty = (None, "", None, "", None, "", None, "", pd.DataFrame(), "")
    if image is None:
        return empty[:-1] + ("No image uploaded.",)
    prompts = [p.strip() for p in prompts_text.strip().splitlines() if p.strip()][:4]
    if not prompts:
        return empty[:-1] + ("No prompts entered.",)
    if not backend.is_ready:
        return empty[:-1] + (_not_ready_msg(),)

    vis: List[Optional[np.ndarray]] = []
    caps: List[str] = []
    dfs: List[pd.DataFrame] = []

    for prompt in prompts:
        result = backend.predict(
            image=image, text_prompt=prompt,
            conf_threshold=conf_threshold,
            max_detections=max_detections, min_area=min_area,
            max_area=max_area if max_area > 0 else float("inf"),
        )
        has_conf = result.get("has_confidence", True)
        vis.append(draw_detections(
            image.copy(), result["boxes"], result["labels"], result["scores"],
            masks=result["masks"] if show_masks else None, show_masks=show_masks,
            show_scores=has_conf,
        ))
        n_det = len(result["boxes"])
        if n_det and has_conf:
            caps.append(f"**{prompt}** — {n_det} detection(s) | "
                        f"conf: {np.mean(result['scores']):.3f}")
        else:
            caps.append(f"**{prompt}** — {n_det} detection(s)")
        dfs.append(create_detection_dataframe(
            result["boxes"], result["scores"], result["labels"], prompt,
            include_confidence=has_conf,
        ))
        _record_run(prompt, len(result["boxes"]), result["scores"], result["runtime"])

    while len(vis) < 4:
        vis.append(None)
        caps.append("")

    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    summary = "\n".join(f"• **{p}** → {len(df)} det." for p, df in zip(prompts, dfs))
    return (vis[0], caps[0], vis[1], caps[1], vis[2], caps[2], vis[3], caps[3], combined, summary)


# ---------------------------------------------------------------------------
# Tab 3 — Batch
# ---------------------------------------------------------------------------

def run_batch_evaluation(
    files: Optional[List], prompt: str, conf_threshold: float,
    max_detections: int, min_area: float, max_area: float,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[pd.DataFrame, Optional[str], str]:
    if not files:
        return pd.DataFrame(), None, "No files uploaded."
    if not prompt.strip():
        return pd.DataFrame(), None, "Enter a prompt."
    if not backend.is_ready:
        return pd.DataFrame(), None, _not_ready_msg()

    records: List[Dict] = []
    n = len(files)
    for i, file in enumerate(files):
        fname = Path(file.name).name
        progress(i / n, desc=f"Processing {fname} ({i+1}/{n})")
        try:
            img = np.array(Image.open(file.name).convert("RGB"))
            result = backend.predict(
                image=img, text_prompt=prompt.strip(),
                conf_threshold=conf_threshold, max_detections=max_detections,
                min_area=min_area, max_area=max_area if max_area > 0 else float("inf"),
            )
            boxes, scores = result["boxes"], result["scores"]
            total_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
            records.append({
                "filename": fname, "prompt": prompt.strip(),
                "num_detections": len(boxes),
                "mean_confidence": round(float(np.mean(scores)), 4) if scores else 0.0,
                "total_box_area": int(total_area),
                "inference_ms": round(result["runtime"]["total"] * 1000, 1),
            })
        except Exception as exc:
            logger.error("Batch: %s failed — %s", fname, exc)
            records.append({
                "filename": fname, "prompt": prompt.strip(), "num_detections": -1,
                "mean_confidence": 0.0, "total_box_area": 0, "inference_ms": 0.0,
            })

    progress(1.0, desc="Done")
    df = pd.DataFrame(records)
    csv_path = os.path.join(EXPORT_DIR, f"batch_{datetime.now():%Y%m%d_%H%M%S}.csv")
    df.to_csv(csv_path, index=False)

    ok = df[df["num_detections"] >= 0]
    summary = (
        f"Processed {len(df)} images  |  Total detections: {ok['num_detections'].sum()}  |  "
        f"Avg/image: {ok['num_detections'].mean():.2f}  |  Saved → {csv_path}"
    )
    return df, csv_path, summary


# ---------------------------------------------------------------------------
# Tab 4 — Dataset
# ---------------------------------------------------------------------------

def run_dataset_evaluation(
    folder_path: str, prompt: str, conf_threshold: float, max_detections: int,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[Optional[str], pd.DataFrame, str]:
    folder = Path(folder_path.strip()) if folder_path.strip() else Path(".")
    if not folder.is_dir():
        return None, pd.DataFrame(), f"'{folder}' is not a valid directory."
    if not prompt.strip():
        return None, pd.DataFrame(), "Enter a prompt."
    if not backend.is_ready:
        return None, pd.DataFrame(), _not_ready_msg()

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    paths = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    if not paths:
        return None, pd.DataFrame(), f"No images found in {folder}"

    records: List[Dict] = []
    all_areas: List[float] = []
    all_confs: List[float] = []
    for i, p in enumerate(paths):
        progress(i / len(paths), desc=f"[{i+1}/{len(paths)}] {p.name}")
        try:
            img = np.array(Image.open(p).convert("RGB"))
            result = backend.predict(
                image=img, text_prompt=prompt.strip(),
                conf_threshold=conf_threshold, max_detections=max_detections,
            )
            boxes, scores = result["boxes"], result["scores"]
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            all_areas.extend(areas)
            all_confs.extend(scores)
            records.append({
                "filename": p.name, "num_detections": len(boxes),
                "mean_confidence": round(float(np.mean(scores)), 4) if scores else 0.0,
                "total_area_px2": int(sum(areas)),
                "mean_area_px2": round(float(np.mean(areas)), 1) if areas else 0.0,
            })
        except Exception as exc:
            logger.error("Dataset eval: %s — %s", p.name, exc)

    progress(1.0, desc="Generating plots…")
    df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Dataset Evaluation  ·  prompt: "{prompt.strip()}"', fontsize=13, fontweight="bold")
    axes[0].hist(df["num_detections"], bins=max(10, len(df) // 5), color="#4f86c6", edgecolor="white")
    axes[0].set_title("Detections per Image"); axes[0].set_xlabel("Count"); axes[0].set_ylabel("Images")
    if all_areas:
        axes[1].hist(all_areas, bins=40, color="#e07a5f", edgecolor="white")
        axes[1].set_title("Bounding Box Area Distribution"); axes[1].set_xlabel("Area (px²)"); axes[1].set_ylabel("Frequency")
    if all_confs:
        axes[2].hist(all_confs, bins=30, range=(0, 1), color="#81b29a", edgecolor="white")
        axes[2].set_title("Confidence Score Distribution"); axes[2].set_xlabel("Confidence"); axes[2].set_ylabel("Frequency")
    plt.tight_layout()
    plot_path = os.path.join(EXPORT_DIR, f"dataset_eval_{datetime.now():%Y%m%d_%H%M%S}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = (
        f"Images: {len(df)}  |  Total detections: {df['num_detections'].sum()}  |  "
        f"Avg/image: {df['num_detections'].mean():.2f}  |  Mean confidence: {np.mean(all_confs):.3f}"
        if all_confs else
        f"Images: {len(df)}  |  Total detections: {df['num_detections'].sum()}"
    )
    return plot_path, df, summary


# ---------------------------------------------------------------------------
# Tab 5 — Analytics
# ---------------------------------------------------------------------------

def get_prompt_analytics_df() -> pd.DataFrame:
    if not prompt_analytics:
        return pd.DataFrame()
    records = []
    for prompt, data in prompt_analytics.items():
        runs = data["runs"]
        records.append({
            "Prompt": prompt, "Runs": runs,
            "Total Detections": data["total_detections"],
            "Avg Det / Run": round(data["total_detections"] / runs, 2),
            "Mean Confidence": round(data["total_confidence"] / runs, 4),
        })
    return pd.DataFrame(records).sort_values("Total Detections", ascending=False).reset_index(drop=True)


def get_session_history_df() -> pd.DataFrame:
    return pd.DataFrame(session_history) if session_history else pd.DataFrame()


def clear_session_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    session_history.clear()
    prompt_analytics.clear()
    return pd.DataFrame(), pd.DataFrame()


def export_analytics_csv(df: Optional[pd.DataFrame]) -> Optional[str]:
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None
    path = os.path.join(EXPORT_DIR, f"prompt_analytics_{datetime.now():%Y%m%d_%H%M%S}.csv")
    return export_csv(df, path)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

HEADER = """
# PromptDetect — Research Evaluation Interface

Promptable detection &amp; grounding for urban-monitoring datasets &nbsp;·&nbsp;
**MDWD** (Maltese Domestic Waste) &nbsp;·&nbsp; **MTSD** (Maltese Traffic Signs)
"""

CSS = """
.stat-bar  { font-family: monospace; font-size: 0.82em; }
footer     { display: none !important; }
"""

# Gradio 6+ takes theme/css in launch(), not the Blocks constructor.
THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

# Shared slider config so every tab uses the same ranges/labels.
_CONF = dict(minimum=0.05, maximum=0.95, value=0.30, step=0.05, label="Confidence")
_MINA = dict(minimum=0, maximum=10000, value=100, step=50, label="Min Box Area")
_MAXA = dict(minimum=0, maximum=2_000_000, value=0, step=5000, label="Max Box Area (0 = ∞)")


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="PromptDetect") as demo:
        gr.Markdown(HEADER)

        # -- Model selection (shared across all tabs) --
        with gr.Group():
            with gr.Row():
                model_dd = gr.Dropdown(
                    label="Model",
                    choices=list(MODELS.keys()),
                    value=DEFAULT_MODEL,
                    scale=3,
                    info="Choose a backbone, then load it once per session.",
                )
                load_btn = gr.Button("Load Model", variant="primary", scale=1)
            model_info = gr.Markdown(
                "  ·  ".join(f"**{k}** — {v.notes}" for k, v in MODELS.items())
            )
            status_box = gr.Textbox(
                label="Model Status",
                value="No model loaded. Pick a model and click 'Load Model'.",
                interactive=False, lines=5, elem_classes=["stat-bar"],
            )

        with gr.Tabs():

            # ---- TAB 1 — Single Image ----
            with gr.TabItem("Single Image"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=320):
                        with gr.Row():
                            run_btn = gr.Button("Run Inference", variant="primary", scale=2)
                            clear_btn = gr.Button("Clear", scale=1)

                        with gr.Group():
                            gr.Markdown("#### Image")
                            image_in = gr.Image(
                                label="Upload · Drag & Drop · Paste",
                                type="numpy", sources=["upload", "clipboard"], height=260,
                            )
                            with gr.Row():
                                url_in = gr.Textbox(label="…or image URL", placeholder="https://.../image.jpg", scale=4)
                                url_btn = gr.Button("Load", size="sm", scale=1)

                        with gr.Group():
                            gr.Markdown("#### Prompt")
                            prompt_in = gr.Textbox(label="Text prompt", placeholder="e.g. orange CMD bag")
                            with gr.Row():
                                waste_dd = gr.Dropdown(label="MDWD presets", choices=WASTE_PROMPTS)
                                sign_dd = gr.Dropdown(label="MTSD presets", choices=SIGN_PROMPTS)
                            history_dd = gr.Dropdown(
                                label="Recent prompts", choices=[],
                                info="Reuse a prompt from this session.",
                            )

                        with gr.Accordion("Inference settings", open=False):
                            with gr.Row():
                                conf_sl = gr.Slider(**_CONF)
                                maxdet_sl = gr.Slider(1, 300, 50, step=1, label="Max Detections")
                            with gr.Row():
                                minarea_sl = gr.Slider(**_MINA)
                                maxarea_sl = gr.Slider(**_MAXA)
                            show_masks_cb = gr.Checkbox(
                                label="Show segmentation masks", value=False,
                                info="SAM 3 / 3.1 only — Cosmos Reason2 and LocateAnything return boxes only.",
                            )

                    with gr.Column(scale=2, min_width=520):
                        with gr.Tabs():
                            with gr.TabItem("Visual"):
                                with gr.Row():
                                    orig_out = gr.Image(label="Original", type="numpy", height=330)
                                    annot_out = gr.Image(label="Detections / Masks", type="numpy", height=330)
                            with gr.TabItem("Table"):
                                det_table = gr.DataFrame(label="Detections", interactive=False, wrap=True)

                        stats_out = gr.Textbox(label="Inference summary", interactive=False, lines=2, elem_classes=["stat-bar"])

                        with gr.Accordion("Export results", open=False):
                            with gr.Row():
                                fmt_radio = gr.Radio(["CSV", "JSON", "COCO JSON", "YOLO"], value="CSV", label="Format", scale=3)
                                export_btn = gr.Button("Export", size="sm", scale=1)
                            export_status = gr.Textbox(label="Export status", interactive=False, lines=2)
                            export_file = gr.File(label="File")

            # ---- TAB 2 — Prompt Comparison ----
            with gr.TabItem("Prompt Comparison"):
                gr.Markdown("Run **up to 4 prompts** on the same image to compare prompt sensitivity.")
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        cmp_img = gr.Image(label="Image", type="numpy", sources=["upload", "clipboard"], height=220)
                        cmp_prompts = gr.Textbox(
                            label="Prompts (one per line, max 4)",
                            placeholder="waste\ngarbage\nrubbish\ndomestic waste", lines=6,
                        )
                        with gr.Accordion("Inference settings", open=False):
                            with gr.Row():
                                cmp_conf = gr.Slider(**_CONF)
                                cmp_maxdet = gr.Slider(1, 200, 50, step=1, label="Max Detections")
                            with gr.Row():
                                cmp_minarea = gr.Slider(**_MINA)
                                cmp_maxarea = gr.Slider(**_MAXA)
                            cmp_masks = gr.Checkbox(label="Show masks", value=False)
                        cmp_run_btn = gr.Button("Compare Prompts", variant="primary")

                    with gr.Column(scale=3):
                        with gr.Row():
                            cmp_out_a = gr.Image(label="Prompt A", type="numpy", height=260)
                            cmp_out_b = gr.Image(label="Prompt B", type="numpy", height=260)
                        with gr.Row():
                            cmp_cap_a = gr.Markdown()
                            cmp_cap_b = gr.Markdown()
                        with gr.Row():
                            cmp_out_c = gr.Image(label="Prompt C", type="numpy", height=260)
                            cmp_out_d = gr.Image(label="Prompt D", type="numpy", height=260)
                        with gr.Row():
                            cmp_cap_c = gr.Markdown()
                            cmp_cap_d = gr.Markdown()
                cmp_summary = gr.Textbox(label="Summary", interactive=False, lines=5)
                cmp_table = gr.DataFrame(label="All detections", interactive=False)

            # ---- TAB 3 — Batch ----
            with gr.TabItem("Batch"):
                gr.Markdown("Run one prompt over **many images**. A `results.csv` is written to `exports/`.")
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        batch_files = gr.File(label="Images", file_count="multiple", file_types=["image"])
                        batch_prompt = gr.Textbox(label="Prompt", placeholder="garbage bag")
                        with gr.Accordion("Inference settings", open=False):
                            batch_conf = gr.Slider(**_CONF)
                            batch_maxdet = gr.Slider(1, 300, 50, step=1, label="Max Detections")
                            batch_minarea = gr.Slider(**_MINA)
                            batch_maxarea = gr.Slider(**_MAXA)
                        batch_run_btn = gr.Button("Run Batch", variant="primary")
                    with gr.Column(scale=2):
                        batch_summary = gr.Textbox(label="Summary", interactive=False, lines=3)
                        batch_table = gr.DataFrame(label="Per-image results", interactive=False)
                        batch_csv = gr.File(label="Download results.csv")

            # ---- TAB 4 — Dataset ----
            with gr.TabItem("Dataset"):
                gr.Markdown("Point to a **local image folder**. Generates summary stats and distribution plots.")
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        ds_folder = gr.Textbox(label="Folder path", placeholder="/path/to/MTSD/images")
                        ds_prompt = gr.Textbox(label="Prompt", placeholder="traffic sign")
                        with gr.Accordion("Inference settings", open=False):
                            ds_conf = gr.Slider(**_CONF)
                            ds_maxdet = gr.Slider(1, 300, 100, step=1, label="Max Detections")
                        ds_run_btn = gr.Button("Run Dataset Eval", variant="primary")
                    with gr.Column(scale=2):
                        ds_summary = gr.Textbox(label="Summary", interactive=False, lines=2)
                        ds_plot = gr.Image(label="Distribution plots", type="filepath", height=320)
                        ds_table = gr.DataFrame(label="Per-image results", interactive=False)

            # ---- TAB 5 — Analytics ----
            with gr.TabItem("Analytics"):
                gr.Markdown("Aggregated, session-level statistics for every prompt used so far.")
                with gr.Row():
                    refresh_btn = gr.Button("Refresh", variant="primary")
                    clear_an_btn = gr.Button("Clear Session")
                    an_export_btn = gr.Button("Export CSV")
                analytics_table = gr.DataFrame(label="Prompt performance", interactive=False)
                history_table = gr.DataFrame(label="Run history", interactive=False)
                an_export_file = gr.File(label="Download")

        # ---------------- Event bindings ----------------

        load_btn.click(fn=load_model, inputs=[model_dd], outputs=[status_box], show_progress=True)

        waste_dd.change(fn=lambda x: x, inputs=[waste_dd], outputs=[prompt_in])
        sign_dd.change(fn=lambda x: x, inputs=[sign_dd], outputs=[prompt_in])
        url_btn.click(fn=load_url_image, inputs=[url_in], outputs=[image_in])
        history_dd.change(fn=lambda x: x.split("  (")[0] if x else "", inputs=[history_dd], outputs=[prompt_in])

        run_inputs = [
            image_in, prompt_in, conf_sl, maxdet_sl,
            minarea_sl, maxarea_sl, show_masks_cb, url_in,
        ]
        run_outputs = [orig_out, annot_out, det_table, stats_out, prompt_in, history_dd]
        run_btn.click(fn=run_single_inference, inputs=run_inputs, outputs=run_outputs, show_progress=True)
        prompt_in.submit(fn=run_single_inference, inputs=run_inputs, outputs=run_outputs, show_progress=True)
        clear_btn.click(fn=clear_single, outputs=[image_in, orig_out, annot_out, det_table, stats_out])

        export_btn.click(
            fn=export_detections, inputs=[det_table, fmt_radio, prompt_in, orig_out],
            outputs=[export_file, export_status],
        )

        cmp_run_btn.click(
            fn=run_prompt_comparison,
            inputs=[cmp_img, cmp_prompts, cmp_conf, cmp_maxdet, cmp_minarea, cmp_maxarea, cmp_masks],
            outputs=[cmp_out_a, cmp_cap_a, cmp_out_b, cmp_cap_b, cmp_out_c, cmp_cap_c, cmp_out_d, cmp_cap_d, cmp_table, cmp_summary],
            show_progress=True,
        )

        batch_run_btn.click(
            fn=run_batch_evaluation,
            inputs=[batch_files, batch_prompt, batch_conf, batch_maxdet, batch_minarea, batch_maxarea],
            outputs=[batch_table, batch_csv, batch_summary], show_progress=True,
        )

        ds_run_btn.click(
            fn=run_dataset_evaluation,
            inputs=[ds_folder, ds_prompt, ds_conf, ds_maxdet],
            outputs=[ds_plot, ds_table, ds_summary], show_progress=True,
        )

        refresh_btn.click(fn=lambda: (get_prompt_analytics_df(), get_session_history_df()), outputs=[analytics_table, history_table])
        clear_an_btn.click(fn=clear_session_data, outputs=[analytics_table, history_table])
        an_export_btn.click(fn=export_analytics_csv, inputs=[analytics_table], outputs=[an_export_file])

    return demo


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM — Research Evaluation Interface")
    p.add_argument("--host", default="0.0.0.0", help="Bind address")
    p.add_argument("--port", default=7860, type=int, help="Port")
    p.add_argument("--no-browser", action="store_true", help="Do not open browser")
    p.add_argument("--share", action="store_true", help="Create a public share link")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ui = build_ui()
    ui.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        show_error=True,
        theme=THEME,
        css=CSS,
    )
