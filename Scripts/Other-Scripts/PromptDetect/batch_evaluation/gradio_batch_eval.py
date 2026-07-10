#!/usr/bin/env python3
"""Gradio front-end for PromptDetect batch evaluation.

A thin UI over run_batch_eval.run_evaluation: dataset + split dropdowns,
0-15 prompts (zero prompts for dry-run only), model checkboxes (none
pre-selected; Cosmos 32B gated behind an explicit opt-in), max-images limit
and a dry-run toggle. The existing PromptDetect app (app.py) is untouched.

Usage:
    python Scripts/Other-Scripts/PromptDetect/batch_evaluation/gradio_batch_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import gradio as gr
import pandas as pd

import config
from dataset_loader import load_ground_truth
from run_batch_eval import resolve_models, run_evaluation

LIGHT_MODELS = [label for label in config.available_models().values()
                if label not in config.HEAVY_MODEL_LABELS]
HEAVY_MODELS = sorted(config.HEAVY_MODEL_LABELS)


def parse_prompts(text: str, dry_run: bool) -> list[str]:
    prompts = [line.strip() for line in text.splitlines() if line.strip()]
    if not (config.MIN_PROMPTS <= len(prompts) <= config.MAX_PROMPTS):
        raise gr.Error(f"Provide {config.MIN_PROMPTS}-{config.MAX_PROMPTS} prompts "
                       f"(one per line); got {len(prompts)}.")
    if not dry_run and not prompts:
        raise gr.Error("A real evaluation needs at least one prompt. "
                       "Use dry run for a zero-prompt dataset/model sanity check.")
    return prompts


def launch_run(dataset, split, prompts_text, light_selected, heavy_selected,
               allow_heavy, max_images, conf_threshold, iou_threshold,
               save_visualizations, dry_run, progress=gr.Progress()):
    prompts = parse_prompts(prompts_text, bool(dry_run))
    selected = list(light_selected or [])
    if heavy_selected:
        if not allow_heavy:
            raise gr.Error("Heavy models selected but the 'I understand - allow heavy models' "
                           "toggle is off. Enable it to confirm, or deselect the heavy models.")
        selected += list(heavy_selected)
    if not selected:
        raise gr.Error("Select at least one model - nothing runs by default.")
    slugs = [config.model_slug(label) for label in selected]
    labels = resolve_models(slugs, allow_heavy=bool(allow_heavy))
    max_images = int(max_images) if max_images and int(max_images) > 0 else None

    if dry_run:
        gt = load_ground_truth(dataset, split, max_images)
        n_boxes = sum(len(r["boxes"]) for r in gt["records"])
        estimated = len(gt["records"]) * len(prompts) * len(labels)
        info = (
            f"**DRY RUN - no models loaded, nothing written.**\n\n"
            f"- Dataset: {gt['dataset']} `{gt['split']}` ({gt['source']})\n"
            f"- Images: {len(gt['records']):,} | GT boxes: {n_boxes:,} | "
            f"classes: {len(gt['class_names'])}\n"
            f"- Models: {', '.join(labels)}\n"
            f"- Prompts ({len(prompts)}): {', '.join(prompts) if prompts else '(none - dataset sanity check)'}\n"
            f"- A full run would make ~{estimated:,} predict() calls."
        )
        return info, pd.DataFrame(), ""

    def notify(fraction, message):
        progress(fraction, desc=message)

    run_dir, summary = run_evaluation(
        dataset, split, prompts, labels, max_images,
        float(conf_threshold), float(iou_threshold),
        len(load_ground_truth(dataset, split, max_images)["records"]) if save_visualizations else 0,
        notify,
    )
    table = pd.DataFrame(summary["per_prompt_metrics"])
    files = "\n".join(f"- `{p}`" for p in sorted(str(x.name) for x in Path(run_dir).iterdir()))
    visual_note = (f"\n\nVisual comparisons (all evaluated images): `{run_dir / 'visualizations' / 'index.html'}`"
                   if save_visualizations else "")
    info = f"**Run complete.** Results folder:\n\n`{run_dir}`{visual_note}\n\nFiles:\n{files}"
    return info, table, str(run_dir)


def build_app():
    with gr.Blocks(title="PromptDetect Batch Evaluation") as demo:
        gr.Markdown(
            "# PromptDetect - batch evaluation vs ground truth\n"
            "Systematic evaluation of prompt-based models against MDWD/MTSD annotations. "
            "Models must be selected explicitly; Cosmos 32B needs the extra "
            "opt-in toggle. Use **dry run** first to sanity-check the plan."
        )
        with gr.Row():
            with gr.Column(scale=4):
                dataset = gr.Dropdown(["MDWD", "MTSD"], value="MTSD", label="Dataset")
                split = gr.Dropdown(["train", "valid", "test", "all"], value="test",
                                    label="Split ('all' = MTSD QA annotations fallback)")
                prompts_text = gr.Textbox(
                    label=f"Prompts ({config.MIN_PROMPTS}-{config.MAX_PROMPTS}, one per line; 0 dry-run only)",
                    lines=5, placeholder="traffic sign\nstop sign\nwarning sign",
                )
                light_models = gr.CheckboxGroup(LIGHT_MODELS, label="Models (none pre-selected)")
                heavy_models = gr.CheckboxGroup(
                    HEAVY_MODELS, label="Heavy models (large VRAM / slow - opt-in)")
                allow_heavy = gr.Checkbox(
                    label="I understand - allow heavy models", value=False,
                    info="Required when any heavy model is selected.")
                with gr.Row():
                    max_images = gr.Number(label="Max images (0 = all)", value=25, precision=0)
                    save_visualizations = gr.Checkbox(
                        label="Save visual comparisons (GT vs predictions)", value=False,
                        info="Writes one side-by-side sheet for every evaluated image, model and prompt, plus visualizations/index.html.")
                with gr.Row():
                    conf_threshold = gr.Slider(0.0, 1.0, value=config.CONF_THRESHOLD,
                                               step=0.05, label="Confidence threshold")
                    iou_threshold = gr.Slider(0.3, 0.95, value=config.IOU_MATCH_THRESHOLD,
                                              step=0.05, label="IoU match threshold")
                dry_run = gr.Checkbox(label="Dry run (load dataset + plan only)", value=True)
                run_btn = gr.Button("Run batch evaluation", variant="primary")
            with gr.Column(scale=6):
                info = gr.Markdown("Configure a run on the left; results appear here.")
                table = gr.Dataframe(label="Per-model / per-prompt metrics", interactive=False)
                run_dir_box = gr.Textbox(label="Results folder", interactive=False)

        run_btn.click(
            launch_run,
            inputs=[dataset, split, prompts_text, light_models, heavy_models, allow_heavy,
                    max_images, conf_threshold, iou_threshold, save_visualizations, dry_run],
            outputs=[info, table, run_dir_box],
        )
    return demo


if __name__ == "__main__":
    build_app().launch(inbrowser=True)
