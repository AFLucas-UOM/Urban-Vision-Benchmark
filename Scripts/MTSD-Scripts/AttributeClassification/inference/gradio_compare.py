"""Minimal Gradio app for side-by-side multi-head checkpoint inference.

Run from Scripts/AttributeClassification:
    PYTHONNOUSERSITE=1 C:/Users/fridge/anaconda3/envs/mtsd-attrcls/python.exe -s inference/gradio_compare.py
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any

import torch
from PIL import Image
import gradio as gr


APP_DIR = Path(__file__).resolve().parent
SUBPROJECT_ROOT = APP_DIR.parent
if str(SUBPROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBPROJECT_ROOT))

from mtsd_attr.backbones import build_backbone  # noqa: E402
from mtsd_attr.config import adaptation_of, load_config  # noqa: E402
from mtsd_attr.dataset import build_transforms  # noqa: E402
from mtsd_attr.multihead_model import MultiHeadClassifier  # noqa: E402
from mtsd_attr.train_common import _load_checkpoint_state  # noqa: E402


FAMILY_ORDER = ("V-JEPA", "DINO", "ConvNeXt", "Unknown")


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    family: str
    label: str
    variant: str
    run_id: str


def _safe_rel(path: Path, root: Path = SUBPROJECT_ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _family_from_text(text: str) -> str | None:
    lower = text.lower()
    if any(token in lower for token in ("vjepa", "v-jepa", "jepa")):
        return "V-JEPA"
    if "dino" in lower:
        return "DINO"
    if any(token in lower for token in ("convnext", "conv-next", "conv")):
        return "ConvNeXt"
    return None


def _family_from_metadata(ckpt: dict[str, Any], path: Path) -> str | None:
    pieces = [
        str(ckpt.get("variant", "")),
        str(ckpt.get("run_id", "")),
        str(ckpt.get("model_cfg", {}).get("backbone", "")),
        path.as_posix(),
    ]
    return _family_from_text(" ".join(pieces))


def _candidate_search_roots(cfg: dict[str, Any]) -> list[Path]:
    roots = [
        cfg["paths"]["checkpoints_dir"],
        cfg["paths"]["subproject_root"] / "outputs",
        SUBPROJECT_ROOT / "outputs",
        Path.cwd() / "outputs",
        SUBPROJECT_ROOT,
    ]
    unique = []
    seen = set()
    for root in roots:
        root = Path(root)
        key = str(root.resolve())
        if root.exists() and key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


@lru_cache(maxsize=2)
def discover_checkpoints(include_smoke: bool = False) -> tuple[dict[str, list[CheckpointInfo]], dict[str, Any]]:
    cfg = load_config()
    grouped = {family: [] for family in FAMILY_ORDER}
    diagnostics = {
        "searched_roots": [str(root) for root in _candidate_search_roots(cfg)],
        "found_best": 0,
        "skipped_smoke": 0,
    }
    seen = set()
    for root in _candidate_search_roots(cfg):
        for path in root.rglob("best.pt"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            diagnostics["found_best"] += 1
            
            is_smoke = _is_smoke_checkpoint(resolved)
            if is_smoke:
                diagnostics["skipped_smoke"] += 1
                if not include_smoke:
                    continue

            ckpt = {}
            metadata_error = None
            try:
                ckpt = torch.load(resolved, map_location="cpu", weights_only=False)
            except Exception as exc:  # Discovery still falls back to path matching.
                metadata_error = str(exc)

            family = _family_from_metadata(ckpt, resolved) if ckpt else None
            family = family or _family_from_text(resolved.as_posix())
            if family not in grouped:
                family = "Unknown"

            variant = str(ckpt.get("variant") or resolved.parent.name)
            run_id = str(ckpt.get("run_id") or resolved.parent.name)
            rel = _safe_rel(resolved)
            suffix = f" [{metadata_error}]" if metadata_error else ""
            smoke_prefix = "[SMOKE] " if is_smoke else ""
            label = f"{smoke_prefix}{run_id} | {rel}{suffix}"
            grouped[family].append(
                CheckpointInfo(
                    path=resolved,
                    family=family,
                    label=label,
                    variant=variant,
                    run_id=run_id,
                )
            )

    for family in grouped:
        grouped[family].sort(key=lambda item: item.label.lower())
    return grouped, diagnostics


def _is_smoke_checkpoint(path: Path) -> bool:
    """Return True for checkpoints stored under a smoke run folder."""
    return any("smoke" in part.lower() for part in path.parts)


# Load initially with include_smoke=True as that is our workspace default for testing
CHECKPOINTS, DISCOVERY_DIAGNOSTICS = discover_checkpoints(include_smoke=True)


@lru_cache(maxsize=8)
def load_model(checkpoint_path: str, device_name: str):
    ckpt_path = Path(checkpoint_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["model_cfg"]
    attributes = ckpt.get("attributes")
    if not attributes:
        raise ValueError("Checkpoint does not contain attribute class mappings.")
    probe = ckpt.get("probe")
    if not probe:
        raise ValueError("Checkpoint does not contain probe/head configuration.")
    adaptation = ckpt.get("adaptation") or adaptation_of(model_cfg)

    backbone = build_backbone(model_cfg)
    model = MultiHeadClassifier(backbone, attributes, probe, adaptation)
    _load_checkpoint_state(model, ckpt["model_state"], adaptation)

    device = torch.device(device_name)
    model = model.to(device)
    model.eval()
    transform = build_transforms(
        backbone.image_size,
        False,
        load_config()["training"]["augmentation"],
    )
    return model, attributes, transform, ckpt


def _topk_items(probs: torch.Tensor, classes: list[str], k: int = 3) -> list[tuple[str, float]]:
    k = min(k, probs.numel())
    values, indices = torch.topk(probs, k=k)
    return [(classes[int(idx)], float(value)) for value, idx in zip(values, indices)]


def _flatten_checkpoints(grouped: dict[str, list[CheckpointInfo]]) -> list[CheckpointInfo]:
    return [
        info
        for family in FAMILY_ORDER
        for info in grouped[family]
    ]


def _empty_html(message: str) -> str:
    return _notice_html(f"<strong>{escape(message)}</strong>")


def _notice_html(body: str) -> str:
    return f"""
    <div style="padding:12px 14px;border:1px solid #30363d;border-radius:6px;background:#161b22;color:#c9d1d9;">
      {body}
    </div>
    """


def _get_family_badge(family: str) -> str:
    if family == "V-JEPA":
        bg, border, color = "rgba(168, 85, 247, 0.1)", "rgba(168, 85, 247, 0.2)", "#d8b4fe"
    elif family == "DINO":
        bg, border, color = "rgba(6, 182, 212, 0.1)", "rgba(6, 182, 212, 0.2)", "#99f6e4"
    elif family == "ConvNeXt":
        bg, border, color = "rgba(249, 115, 22, 0.1)", "rgba(249, 115, 22, 0.2)", "#fed7aa"
    else:
        bg, border, color = "rgba(107, 114, 128, 0.1)", "rgba(107, 114, 128, 0.2)", "#e5e7eb"
    return f"<span style='display:inline-block;padding:2px 6px;border-radius:4px;background:{bg};border:1px solid {border};color:{color};font-weight:600;font-size:10px;text-transform:uppercase;letter-spacing:0.02em;'>{escape(family)}</span>"


def _render_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return _empty_html("No results to display.")

    body = []
    previous_model = None
    for row in rows:
        if row["model"] != previous_model:
            badge = _get_family_badge(row["model"])
            body.append(
                "<tr>"
                f"<td colspan='6' style='background:rgba(31, 41, 55, 0.2);color:#c9d1d9;font-weight:600;padding:10px 12px;"
                "border-top:1px solid #30363d;'>"
                f"<div style='display:flex;align-items:center;gap:8px;'>{badge} <span style='font-size:12px;color:#8b949e;'>Checkpoints</span></div>"
                "</tr>"
            )
            previous_model = row["model"]

        confidence = row.get("confidence_value")
        if confidence is None:
            confidence_cell = f"<span style='color:#8b949e;'>{escape(row.get('confidence', ''))}</span>"
        else:
            pct = max(0.0, min(1.0, confidence)) * 100.0
            if confidence >= 0.8:
                bar_color = "#30a14e"
                text_color = "#30a14e"
            elif confidence >= 0.5:
                bar_color = "#1f6feb"
                text_color = "#58a6ff"
            else:
                bar_color = "#d29922"
                text_color = "#d29922"
                
            confidence_cell = (
                f"<div style='font-weight:600;color:{text_color};font-family:JetBrains Mono,monospace;font-size:12px;'>{confidence:.3f}</div>"
                "<div style='height:3px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden;margin-top:4px;width:80px;'>"
                f"<div style='width:{pct:.1f}%;height:100%;background:{bar_color};border-radius:2px;'></div>"
                "</div>"
            )

        topk = row.get("topk_items")
        if topk:
            topk_cell = " ".join(
                "<span style='display:inline-flex;align-items:center;gap:4px;padding:2px 6px;margin:2px 3px 2px 0;"
                "border:1px solid rgba(255,255,255,0.06);border-radius:4px;background:rgba(255,255,255,0.03);color:#8b949e;"
                "font-size:11.5px;'>"
                f"<span style='color:#c9d1d9;font-weight:500;'>{escape(label)}</span>"
                f"<span style='color:#58a6ff;font-family:JetBrains Mono,monospace;font-weight:600;'>{score*100:.0f}%</span></span>"
                for label, score in topk
            )
        else:
            topk_cell = f"<code style='font-family:JetBrains Mono,monospace;font-size:12px;color:#f85149;'>{escape(row.get('topk', ''))}</code>"

        checkpoint_name = escape(row['checkpoint'])
        if "smoke" in checkpoint_name.lower():
            checkpoint_name = (
                f"<span style='color:#f85149;font-weight:500;font-size:9px;border:1px solid rgba(248,81,73,0.15);"
                f"padding:0px 3px;border-radius:3px;background:rgba(248,81,73,0.03);margin-right:4px;vertical-align:middle;'>SMOKE</span>"
                f"<span style='vertical-align:middle;'>{checkpoint_name}</span>"
            )

        body.append(
            "<tr class='table-row'>"
            f"<td style='padding:8px 12px;border-top:1px solid #30363d;color:#8b949e;'>{_get_family_badge(row['model'])}</td>"
            f"<td style='padding:8px 12px;border-top:1px solid #30363d;font-family:JetBrains Mono,ui-monospace,monospace;font-size:11px;color:#8b949e;'>{checkpoint_name}</td>"
            f"<td style='padding:8px 12px;border-top:1px solid #30363d;color:#c9d1d9;font-weight:500;'>{escape(row['head'])}</td>"
            f"<td style='padding:8px 12px;border-top:1px solid #30363d;color:#58a6ff;font-weight:600;'>{escape(row['prediction'])}</td>"
            f"<td style='padding:8px 12px;border-top:1px solid #30363d;min-width:100px;'>{confidence_cell}</td>"
            f"<td style='padding:8px 12px;border-top:1px solid #30363d;'>{topk_cell}</td>"
            "</tr>"
        )

    return (
        "<div class='table-container'>"
        "<table class='custom-table'>"
        "<thead>"
        "<tr>"
        "<th>Family</th>"
        "<th>Checkpoint</th>"
        "<th>Head</th>"
        "<th>Prediction</th>"
        "<th>Confidence</th>"
        "<th>Top-k Predictions</th>"
        "</tr>"
        "</thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table>"
        "</div>"
    )


def _no_checkpoint_html(diagnostics: dict[str, Any]) -> str:
    found = diagnostics["found_best"]
    skipped = diagnostics["skipped_smoke"]
    roots = "<br>".join(escape(root) for root in diagnostics["searched_roots"])
    return (
        "<strong>No active best.pt checkpoints were discovered.</strong><br>"
        f"Found {found} best.pt file(s); skipped {skipped} smoke checkpoint(s).<br>"
        "If you only have smoke checkpoints, toggle <strong>Include Smoke Checkpoints</strong> on the left side."
        "<br><br><strong>Searched Roots:</strong><br>"
        f"{roots}"
    )


def run_comparison(
    image: Image.Image | None,
    vjepa_selected: list[str],
    dino_selected: list[str],
    convnext_selected: list[str],
    unknown_selected: list[str],
    include_smoke: bool,
):
    if image is None:
        return _empty_html("Please upload or paste an image.")

    selected_labels = []
    if vjepa_selected: selected_labels.extend(vjepa_selected)
    if dino_selected: selected_labels.extend(dino_selected)
    if convnext_selected: selected_labels.extend(convnext_selected)
    if unknown_selected: selected_labels.extend(unknown_selected)

    if not selected_labels:
        return _notice_html("<strong>No models selected.</strong> Please select at least one checkpoint checkbox in the sidebar.")

    grouped, diagnostics = discover_checkpoints(include_smoke=include_smoke)
    checkpoints = _flatten_checkpoints(grouped)
    selected_checkpoints = [ckpt for ckpt in checkpoints if ckpt.label in selected_labels]

    if not selected_checkpoints:
        return _notice_html("<strong>No matching checkpoints found.</strong> Try checking 'Include Smoke Checkpoints' or rescanning.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    rgb = image.convert("RGB")

    for info in selected_checkpoints:
        checkpoint_display = _safe_rel(info.path)
        try:
            model, attributes, transform, _ = load_model(str(info.path), str(device))
            tensor = transform(rgb).unsqueeze(0).to(device)
            with torch.inference_mode():
                logits_by_head = model(tensor)

            for head, logits in logits_by_head.items():
                classes = list(attributes[head]["classes"])
                probs = torch.softmax(logits[0].detach().cpu(), dim=0)
                confidence, pred_idx = probs.max(dim=0)
                rows.append({
                    "model": info.family,
                    "checkpoint": checkpoint_display,
                    "head": head,
                    "prediction": classes[int(pred_idx)],
                    "confidence_value": float(confidence),
                    "topk_items": _topk_items(probs, classes),
                })
        except Exception as exc:
            rows.append({
                "model": info.family,
                "checkpoint": checkpoint_display,
                "head": "Error",
                "prediction": f"{type(exc).__name__}: {exc}",
                "confidence": None,
                "topk": traceback.format_exc(limit=2).strip().splitlines()[-1],
            })

    return _render_table(rows)


def update_checkbox_choices(include_smoke: bool):
    grouped, diagnostics = discover_checkpoints(include_smoke=include_smoke)
    
    updates = []
    for family in FAMILY_ORDER:
        choices = [info.label for info in grouped[family]]
        updates.append(
            gr.CheckboxGroup(
                choices=choices,
                value=choices,
                visible=len(choices) > 0,
                label=f"{family} ({len(choices)} found)"
            )
        )
    
    return updates[0], updates[1], updates[2], updates[3]


def handle_refresh(include_smoke: bool):
    discover_checkpoints.cache_clear()
    return update_checkbox_choices(include_smoke)


def build_demo():
    theme = gr.themes.Default(
        primary_hue="neutral",
        secondary_hue="neutral",
        neutral_hue="slate",
    )

    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    .gradio-container {
        font-family: 'Inter', sans-serif !important;
        background-color: #0d1117 !important;
    }
    
    .app-header {
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 1.5rem;
        background: #161b22;
        border-radius: 8px;
        border: 1px solid #30363d;
    }
    
    .minimal-title {
        color: #f0f6fc;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.01em;
    }
    
    .subtitle {
        color: #8b949e;
        font-size: 0.95rem;
        margin-top: 0.4rem;
        font-weight: 400;
    }
    
    .sidebar-container {
        background: #161b22 !important;
        border-radius: 8px !important;
        border: 1px solid #30363d !important;
        padding: 1rem !important;
    }
    
    .main-container {
        background: #161b22 !important;
        border-radius: 8px !important;
        border: 1px solid #30363d !important;
        padding: 1rem !important;
    }
    
    button.primary-btn {
        background: #21262d !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        border-radius: 6px !important;
        transition: background-color 0.2s ease !important;
        padding: 8px 16px !important;
    }
    
    button.primary-btn:hover {
        background: #30363d !important;
        border-color: #8b949e !important;
    }
    
    button.primary-btn:active {
        background: #282e38 !important;
    }
    
    .table-container {
        overflow: auto;
        max-height: 75vh;
        border: 1px solid #30363d;
        border-radius: 8px;
        background: #161b22;
        margin-top: 1rem;
    }
    
    .custom-table {
        border-collapse: collapse;
        width: 100%;
        font-size: 13px;
        table-layout: auto;
        color: #c9d1d9;
        text-align: left;
    }
    
    .custom-table th {
        position: sticky;
        top: 0;
        background: #0d1117;
        color: #8b949e;
        z-index: 1;
        font-weight: 600;
        padding: 10px 12px;
        border-bottom: 1px solid #30363d;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 0.05em;
    }
    
    .table-row {
        transition: background-color 0.15s ease;
    }
    
    .table-row:hover {
        background-color: rgba(255, 255, 255, 0.015) !important;
    }
    
    .refresh-button-container button {
        background: #21262d !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
        font-weight: 500 !important;
        font-size: 0.75rem !important;
        transition: all 0.2s ease !important;
        border-radius: 4px !important;
        padding: 2px 8px !important;
        min-height: 24px !important;
        height: 24px !important;
        width: auto !important;
        min-width: auto !important;
    }
    
    .refresh-button-container button:hover {
        background: #30363d !important;
        border-color: #8b949e !important;
    }
    """

    with gr.Blocks(title="Traffic Sign Attribute Model Comparison") as demo:
        # Title Card
        gr.HTML(
            """
            <div class="app-header">
                <h1 class="minimal-title">Traffic Sign Attribute Comparison</h1>
                <p class="subtitle">Upload an image to run side-by-side inference across selected multi-head checkpoints</p>
            </div>
            """
        )

        # We start with include_smoke=True as default because only smoke checkpoints are present currently in workspace
        initial_grouped, _ = discover_checkpoints(include_smoke=True)

        vjepa_choices = [c.label for c in initial_grouped["V-JEPA"]]
        dino_choices = [c.label for c in initial_grouped["DINO"]]
        convnext_choices = [c.label for c in initial_grouped["ConvNeXt"]]
        unknown_choices = [c.label for c in initial_grouped["Unknown"]]

        with gr.Row():
            # Left Column (Controls & Selection)
            with gr.Column(scale=4, elem_classes=["sidebar-container"]):
                gr.Markdown("### Input Image")
                image = gr.Image(
                    label="Upload or Paste Traffic Sign Crop", 
                    type="pil", 
                    sources=["upload", "clipboard"]
                )

                gr.Markdown("### Settings")
                with gr.Row():
                    include_smoke_cb = gr.Checkbox(
                        label="Include Smoke Checkpoints",
                        value=True,
                        info="Scan and load best.pt from -smoke runs."
                    )
                    refresh_btn = gr.Button(
                        "Rescan Disk", 
                        size="sm", 
                        elem_classes=["refresh-button-container"]
                    )

                gr.Markdown("### Checkpoints to Compare")
                
                vjepa_chks = gr.CheckboxGroup(
                    label=f"V-JEPA ({len(vjepa_choices)} found)",
                    choices=vjepa_choices,
                    value=vjepa_choices,
                    visible=len(vjepa_choices) > 0
                )

                dino_chks = gr.CheckboxGroup(
                    label=f"DINO ({len(dino_choices)} found)",
                    choices=dino_choices,
                    value=dino_choices,
                    visible=len(dino_choices) > 0
                )

                convnext_chks = gr.CheckboxGroup(
                    label=f"ConvNeXt ({len(convnext_choices)} found)",
                    choices=convnext_choices,
                    value=convnext_choices,
                    visible=len(convnext_choices) > 0
                )

                unknown_chks = gr.CheckboxGroup(
                    label=f"Unknown ({len(unknown_choices)} found)",
                    choices=unknown_choices,
                    value=unknown_choices,
                    visible=len(unknown_choices) > 0
                )

                gr.Markdown("---")
                run_btn = gr.Button("Run Side-by-Side Comparison", variant="primary", elem_classes=["primary-btn"])

            # Right Column (Results Table)
            with gr.Column(scale=6, elem_classes=["main-container"]):
                gr.Markdown("### Inference Comparison Results")
                table = gr.HTML(
                    value=_empty_html("Please upload an image and click 'Run Side-by-Side Comparison' to display results.")
                )

        # Event callbacks
        include_smoke_cb.change(
            fn=update_checkbox_choices,
            inputs=[include_smoke_cb],
            outputs=[vjepa_chks, dino_chks, convnext_chks, unknown_chks]
        )

        refresh_btn.click(
            fn=handle_refresh,
            inputs=[include_smoke_cb],
            outputs=[vjepa_chks, dino_chks, convnext_chks, unknown_chks]
        )

        run_btn.click(
            fn=run_comparison,
            inputs=[image, vjepa_chks, dino_chks, convnext_chks, unknown_chks, include_smoke_cb],
            outputs=table
        )

    # Attach theme and css for Gradio 6 launch compatibility
    demo.theme = theme
    demo.css = custom_css
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(theme=getattr(demo, "theme", None), css=getattr(demo, "css", None))
