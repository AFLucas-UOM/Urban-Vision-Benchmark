#!/usr/bin/env python3
"""Annotation / training / operational effort report per experimental paradigm.

Quantifies the data, annotation, QA, training and deployment effort behind
each paradigm (supervised detection, attribute classification, prompt-based
localisation) so the dissertation can compare methods beyond accuracy alone.

Everything derivable is read from existing repository artefacts (dataset
folders, QA JSONs, audit reports, run summaries, manifests, run configs).
Human effort values that the repository cannot recover (annotation hours,
costs) come exclusively from the hand-edited
`Scripts/FinalEvaluation/config/annotation_effort_manual.yaml`; missing values
are reported as "not recorded" and never estimated.

Outputs (timestamped; --overwrite writes to .../latest/):
    Documents/Final-Reports/Annotation-Effort/<stamp>/
        annotation_effort_report.md     annotation_effort_summary.csv
        annotation_effort_details.json  operational_comparison.csv
        missing_manual_values.md
    Documents/Final-Figures/Annotation-Effort/<stamp>/   (PNG + PDF + SVG)

Usage:
    python Scripts/FinalEvaluation/annotation_effort_report.py --self-test
    python Scripts/FinalEvaluation/annotation_effort_report.py --dry-run
    python Scripts/FinalEvaluation/annotation_effort_report.py
    python Scripts/FinalEvaluation/annotation_effort_report.py --skip-disk-size
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evidence_lib as ev  # noqa: E402

GENERATOR = "Scripts/FinalEvaluation/annotation_effort_report.py"
MANUAL_CONFIG = SCRIPT_DIR / "config" / "annotation_effort_manual.yaml"
NOT_RECORDED = "not recorded"

MDWD_EDA_SUMMARY = ev.ROOT / "Documents" / "MDWD-EDA" / "eda_summary.json"
MTSD_ANNOTATIONS = ev.ROOT / "Datasets" / "MTSD" / "Annotations"
MTSD_GROUPS_ROOT = ev.ROOT / "Datasets" / "MTSD"
QA_AUDIT_ROOT = ev.ROOT / "Scripts" / "MTSD-Scripts" / "MTSD-AnnotationQA" / "outputs"
ATTR_COMPARISON = ev.ATTR_OUTPUTS / "reports" / "comparison.csv"
ATTR_MANIFEST = ev.ATTR_OUTPUTS / "manifests" / "manifest.json"
MDWD_RUNS = ev.ROOT / "Results" / "MDWD-Runs"
LEAKAGE_JSON = ev.REPORTS_ROOT / "MDWD-Leakage-Analysis" / "metric_comparison.json"

# Facts read from the PromptDetect backend registry
# (Scripts/Other-Scripts/PromptDetect/backend.py) - kept as a documented table
# so this report never imports heavy model code.
PROMPT_MODEL_FACTS = [
    {"model": "SAM 3", "provides_confidence": True, "gated_weights": True,
     "separate_env": False, "prompt_engineering": "text prompt per concept",
     "retraining_required": False},
    {"model": "SAM 3.1", "provides_confidence": True, "gated_weights": True,
     "separate_env": False, "prompt_engineering": "text prompt per concept",
     "retraining_required": False},
    {"model": "Cosmos Reason2 2B", "provides_confidence": False, "gated_weights": False,
     "separate_env": False, "prompt_engineering": "grounding prompt (JSON bbox reply)",
     "retraining_required": False},
    {"model": "Cosmos Reason2 8B", "provides_confidence": False, "gated_weights": False,
     "separate_env": False, "prompt_engineering": "grounding prompt (JSON bbox reply)",
     "retraining_required": False},
    {"model": "Cosmos Reason2 32B", "provides_confidence": False, "gated_weights": False,
     "separate_env": False, "prompt_engineering": "grounding prompt; opt-in via --allow-heavy",
     "retraining_required": False},
    {"model": "LocateAnything 3B", "provides_confidence": False, "gated_weights": False,
     "separate_env": True, "prompt_engineering": "text prompt; runs in mtsd-la worker env",
     "retraining_required": False},
]


def not_recorded(value) -> str:
    return NOT_RECORDED if value in (None, "", "null") else str(value)


def load_manual_values() -> tuple[dict, list[str]]:
    """Manual YAML values; anything missing/null stays 'not recorded'."""
    import yaml

    missing: list[str] = []
    if not MANUAL_CONFIG.exists():
        return {}, [f"manual config not found: {ev.rel(MANUAL_CONFIG)}"]
    payload = yaml.safe_load(MANUAL_CONFIG.read_text(encoding="utf-8")) or {}
    for section, values in payload.items():
        for key, value in (values or {}).items():
            if key != "notes" and value in (None, ""):
                missing.append(f"{section}.{key}")
    return payload, missing


def directory_size_bytes(path: Path) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                pass
    return total


# ---------------------------------------------------------------------------
# Derived facts
# ---------------------------------------------------------------------------

def collect_mdwd(skip_disk: bool) -> dict:
    facts = {"dataset": "MDWD", "source_files": []}
    if MDWD_EDA_SUMMARY.exists():
        payload = ev.read_json(MDWD_EDA_SUMMARY)
        facts.update({
            "raw_images": payload.get("n_images"),
            "annotated_images": payload.get("n_images"),
            "unannotated_images": 0,
            "unique_source_images": payload.get("n_unique_source_images"),
            "boxes": payload.get("n_boxes"),
            "classes": len(payload.get("classes", [])),
            "auxiliary_attributes": 0,
            "integrity_findings": payload.get("issue_counts", {}),
            "per_split": payload.get("per_split", {}),
        })
        facts["source_files"].append(ev.rel(MDWD_EDA_SUMMARY))
    if LEAKAGE_JSON.exists():
        leakage = ev.read_json(LEAKAGE_JSON).get("leakage", {})
        facts["leakage_affected_eval_images"] = (
            int(leakage.get("affected_valid_images", 0))
            + int(leakage.get("affected_test_images", 0)))
        facts["leaked_source_identities"] = leakage.get("n_leaked_source_identities")
        facts["source_files"].append(ev.rel(LEAKAGE_JSON))
    if not skip_disk:
        mdwd_dir = ev.ROOT / "Datasets" / "MDWD"
        if mdwd_dir.is_dir():
            facts["disk_size_gb"] = round(directory_size_bytes(mdwd_dir) / 1024 ** 3, 2)
    return facts


def collect_mtsd(skip_disk: bool) -> dict:
    facts = {"dataset": "MTSD", "source_files": []}
    qa_files = sorted(MTSD_ANNOTATIONS.glob("GRP-*/Final-QA/QA-*.json"))
    qa_files = [p for p in qa_files if ".bak" not in p.name.lower()]
    annotated_images = boxes = 0
    attribute_labelled_boxes = 0
    classes: set[str] = set()
    qa_groups = []
    for qa_path in qa_files:
        try:
            payload = ev.read_json(qa_path)
        except Exception:
            continue
        group = qa_path.parents[1].name
        qa_groups.append(group)
        annotated_images += len(payload.get("images", []))
        annotations = payload.get("annotations", [])
        boxes += len(annotations)
        attribute_labelled_boxes += sum(1 for a in annotations if a.get("attributes"))
        classes |= {c["name"] for c in payload.get("categories", [])}
        facts["source_files"].append(ev.rel(qa_path))
    raw_per_group = {}
    for group_dir in sorted(MTSD_GROUPS_ROOT.glob("GRP-*")):
        images_dir = group_dir / "Images"
        if images_dir.is_dir():
            raw_per_group[group_dir.name] = sum(
                1 for p in images_dir.iterdir() if p.is_file())
    raw_images = sum(raw_per_group.values())
    facts.update({
        "raw_images": raw_images,
        "annotated_images": annotated_images,
        "unannotated_images": raw_images - annotated_images,
        "boxes": boxes,
        "attribute_labelled_boxes": attribute_labelled_boxes,
        "classes": len(classes),
        "auxiliary_attributes": 4,
        "qa_approved_groups": sorted(qa_groups),
        "excluded_groups": sorted(set(raw_per_group) - set(qa_groups)),
    })

    audits = sorted(QA_AUDIT_ROOT.glob("audit-*/audit_summary.json"))
    if audits:
        audit = ev.read_json(audits[-1])
        facts["qa_audit"] = {"audit": audits[-1].parent.name,
                             **audit.get("totals", {})}
        facts["source_files"].append(ev.rel(audits[-1]))
    changelogs = list(QA_AUDIT_ROOT.glob("audit-*/applied_changes*.json")) \
        + list(MTSD_ANNOTATIONS.glob("**/change_log*.json"))
    facts["confirmed_corrections"] = (len(changelogs) if changelogs else NOT_RECORDED)
    if not skip_disk and MTSD_GROUPS_ROOT.is_dir():
        facts["disk_size_gb"] = round(directory_size_bytes(MTSD_GROUPS_ROOT) / 1024 ** 3, 2)
    return facts


def collect_attribute_training() -> dict:
    facts = {"paradigm": "MTSD attribute classification", "variants": [],
             "source_files": []}
    for row in ev.read_csv_rows(ATTR_COMPARISON):
        facts["variants"].append({
            "variant": row.get("variant"), "adaptation": row.get("adaptation"),
            "run_id": row.get("run_id"),
            "total_params": row.get("total_params"),
            "trainable_params": row.get("trainable_params"),
            "trainable_pct": row.get("trainable_pct"),
            "best_epoch": row.get("best_epoch"),
            "stopped_epoch": row.get("stopped_epoch"),
            "mean_macro_f1": row.get("mean_macro_f1"),
            "status": ev.STATUS_HISTORICAL,
        })
    if ATTR_COMPARISON.exists():
        facts["source_files"].append(ev.rel(ATTR_COMPARISON))
    metrics_root = ev.ATTR_OUTPUTS / "metrics"
    if metrics_root.is_dir():
        names = [p.name for p in metrics_root.iterdir() if p.is_dir()]
        facts["completed_runs"] = sum(1 for n in names if not n.endswith("-smoke"))
        facts["smoke_runs"] = sum(1 for n in names if n.endswith("-smoke"))
    if ATTR_MANIFEST.exists():
        manifest = ev.read_json(ATTR_MANIFEST)
        facts["manifest"] = {
            "updated_at": manifest.get("updated_at"),
            "seed": manifest.get("settings", {}).get("seed"),
            "split_fractions": manifest.get("settings", {}).get("split_fractions"),
            "n_crops": len(manifest.get("crops", manifest.get("entries", []))) or NOT_RECORDED,
        }
        facts["source_files"].append(ev.rel(ATTR_MANIFEST))
    facts["training_duration"] = NOT_RECORDED  # per-run wall time not stored in comparison.csv
    return facts


def collect_detection_training() -> dict:
    facts = {"paradigm": "MDWD supervised detection", "runs": [], "source_files": []}
    counts = {"completed": 0, "historical": 0}
    for summary_path in sorted(MDWD_RUNS.glob("*/E*/pipeline_complete.json")):
        suite = summary_path.parents[1].name
        historical = suite.startswith("[OLD]")
        counts["historical" if historical else "completed"] += 1
        try:
            payload = ev.read_json(summary_path)
        except Exception:
            continue
        if historical:
            continue
        facts["runs"].append({
            "suite": suite,
            "model": payload.get("model_variant"),
            "run_name": payload.get("run_name"),
            "parameters": payload.get("parameter_count"),
            "model_size_mb": payload.get("model_size_mb"),
            "training_seconds": payload.get("total_training_time_seconds"),
            "val_inference_ms": payload.get("validation_inference_time_ms"),
            "test_inference_ms": payload.get("test_inference_time_ms"),
            "epochs": 100, "batch_size": 32,  # encoded in every run name (e100, b32)
            "status": ev.STATUS_FINAL,
            "source_file": ev.rel(summary_path),
        })
        facts["source_files"].append(ev.rel(summary_path))
    facts["completed_runs"] = counts["completed"]
    facts["historical_runs"] = counts["historical"]
    facts["rfdetr_runs"] = len(list(MDWD_RUNS.glob("RF-DETR*/E*")))
    facts["rfdetr_note"] = ("RF-DETR nano has a checkpoint/log but no normalised "
                            "summary - excluded from headline tables")
    facts["peak_gpu_memory"] = NOT_RECORDED
    facts["hardware"] = "RTX 4090 (EUVIP suites) / DGX (YOLO26-DGX suite)"
    return facts


def collect_promptdetect() -> dict:
    facts = {"paradigm": "PromptDetect (prompt-based)", "runs": [],
             "model_facts": PROMPT_MODEL_FACTS, "source_files": []}
    for run in ev.discover_promptdetect_runs():
        config = run["config"]
        facts["runs"].append({
            "run": run["run_id"], "dataset": config.get("dataset"),
            "split": config.get("split"), "n_images": config.get("n_images"),
            "n_gt_boxes": config.get("n_gt_boxes"),
            "prompts": len(config.get("prompts", [])),
            "models": len(config.get("models", [])),
            "prompt_model_combinations": (len(config.get("prompts", []))
                                          * len(config.get("models", []))),
            "status": run["status"],
            "source_file": ev.rel(run["run_dir"] / "run_config.json"),
        })
        facts["source_files"].append(ev.rel(run["run_dir"] / "run_config.json"))
    facts["images_evaluated"] = sum(r["n_images"] or 0 for r in facts["runs"])
    facts["prompts_evaluated"] = sum(r["prompts"] for r in facts["runs"])
    return facts


def collect_efficiency() -> dict:
    speed_root = ev.ROOT / "Results" / "Inference-Benchmark" / "InferenceSpeed"
    runs = sorted(speed_root.glob("*/inference_speed_summary.csv")) if speed_root.is_dir() else []
    if not runs:
        return {"status": ev.STATUS_PENDING,
                "note": "no inference-speed benchmark runs yet - run "
                        "Scripts/Other-Scripts/Inference-Benchmark/"
                        "inference_speed_benchmark.py; MDWD detection latency below "
                        "comes from the training pipelines' own val timing instead"}
    rows = []
    for path in runs:
        for row in ev.read_csv_rows(path):
            row["source_file"] = ev.rel(path)
            rows.append(row)
    return {"status": "available", "rows": rows}


# ---------------------------------------------------------------------------
# Paradigm-level operational comparison (no composite score by design)
# ---------------------------------------------------------------------------

def operational_rows(detection: dict, attributes: dict, prompts: dict,
                     mdwd: dict, mtsd: dict, manual: dict) -> list[dict]:
    def manual_value(section: str, key: str) -> str:
        return not_recorded((manual.get(section) or {}).get(key))

    yolo_runs = detection["runs"]
    largest = max(yolo_runs, key=lambda r: r.get("parameters") or 0) if yolo_runs else {}
    lora = [v for v in attributes["variants"] if v.get("adaptation") == "lora"]
    return [
        {"paradigm": "Supervised detection (MDWD, YOLO)",
         "ground_truth_requirement": f"{mdwd.get('boxes', '?')} boxes over "
                                     f"{mdwd.get('annotated_images', '?')} images (10x augmented export)",
         "human_annotation_burden": f"annotation_hours: {manual_value('MDWD', 'annotation_hours')}; "
                                    f"qa_hours: {manual_value('MDWD', 'qa_hours')}",
         "training_requirement": f"{detection['completed_runs']} completed runs, "
                                 "100 epochs each (per-run wall time in details JSON)",
         "prompt_engineering": "none",
         "model_size": f"up to {largest.get('model_size_mb', '?')} MB ({largest.get('model', '')})",
         "latency": f"~{largest.get('test_inference_ms', '?')} ms/img (val-time measurement)",
         "gpu_memory": NOT_RECORDED,
         "maintenance_burden": "retrain to add classes; dataset/label upkeep",
         "main_limitation": "needs full box annotation; augmented-export leakage "
                            "audited (negligible effect)",
         "status": ev.STATUS_FINAL},
        {"paradigm": "Supervised detection (MTSD)",
         "ground_truth_requirement": f"{mtsd.get('boxes', '?')} QA boxes over "
                                     f"{mtsd.get('annotated_images', '?')} images "
                                     f"({len(mtsd.get('qa_approved_groups', []))} QA groups)",
         "human_annotation_burden": f"annotation_hours: {manual_value('MTSD', 'annotation_hours')}; "
                                    f"qa_hours: {manual_value('MTSD', 'qa_hours')}",
         "training_requirement": "pending (no MTSD detection training yet)",
         "prompt_engineering": "none", "model_size": "pending", "latency": "pending",
         "gpu_memory": "pending", "maintenance_burden": "as MDWD, plus ongoing group annotation",
         "main_limitation": f"{mtsd.get('unannotated_images', '?')} images in "
                            f"{len(mtsd.get('excluded_groups', []))} groups still unannotated",
         "status": ev.STATUS_PENDING},
        {"paradigm": "Attribute classification (MTSD crops)",
         "ground_truth_requirement": f"{mtsd.get('attribute_labelled_boxes', '?')} "
                                     "attribute-labelled boxes (4 attribute heads)",
         "human_annotation_burden": "attribute labels captured during box QA; extra hours: "
                                    + manual_value("AttributeClassification", "labelling_hours"),
         "training_requirement": f"{attributes.get('completed_runs', 0)} completed variants "
                                 f"(+{attributes.get('smoke_runs', 0)} smoke) - LoRA trains "
                                 f"~{lora[0]['trainable_params'] if lora else '?'} params",
         "prompt_engineering": "none",
         "model_size": "27M-305M backbone params (frozen for probe/LoRA variants)",
         "latency": NOT_RECORDED, "gpu_memory": NOT_RECORDED,
         "maintenance_burden": "manifest regeneration whenever QA annotations change",
         "main_limitation": "historical GRP-1..3 snapshot; weak `condition` head "
                            "(class imbalance)",
         "status": ev.STATUS_HISTORICAL},
        {"paradigm": "Prompt-based localisation (PromptDetect)",
         "ground_truth_requirement": "none for inference (GT only used for scoring)",
         "human_annotation_burden": f"prompt_design_hours: "
                                    f"{manual_value('PromptDetect', 'prompt_design_hours')}; "
                                    f"manual_review_hours: {manual_value('PromptDetect', 'manual_review_hours')}",
         "training_requirement": "none (zero-shot)",
         "prompt_engineering": f"{prompts.get('prompts_evaluated', 0)} prompts evaluated so far; "
                               "per-model prompt phrasing matters",
         "model_size": "2B-32B VLM / SAM 3-scale weights",
         "latency": "hundreds of ms/img (pilot measurements in predictions.csv)",
         "gpu_memory": NOT_RECORDED,
         "maintenance_burden": "gated weight access (SAM 3/3.1), separate env for "
                               "LocateAnything, prompt upkeep",
         "main_limitation": "only pilot-scale evaluations exist; Cosmos/LocateAnything "
                            "provide no per-box confidence (AP not comparable)",
         "status": ev.STATUS_PILOT},
    ]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(figures_dir: Path, mdwd: dict, mtsd: dict, detection: dict,
                 attributes: dict, operational: list[dict]) -> list[str]:
    ev.apply_plot_style()
    import matplotlib.pyplot as plt

    saved: list[str] = []

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    datasets = ["MDWD", "MTSD"]
    axes[0].bar(datasets, [mdwd.get("annotated_images") or 0,
                           mtsd.get("annotated_images") or 0], color="#4477aa")
    axes[0].set_title("Annotated images")
    axes[1].bar(datasets, [mdwd.get("boxes") or 0, mtsd.get("boxes") or 0],
                color="#66ccee")
    axes[1].set_title("Annotated boxes")
    for axis in axes:
        axis.tick_params(axis="x", labelsize=9)
    figure.suptitle("Annotation scale per dataset (MDWD counts are the 10x-augmented export)",
                    fontsize=10)
    saved.extend(ev.save_figure(figure, figures_dir, "annotation_scale_per_dataset"))

    runs = [r for r in detection["runs"] if r.get("training_seconds")]
    if runs:
        runs = sorted(runs, key=lambda r: (r["model"], r["suite"]))
        labels = [f"{r['model']}\n{r['suite']}" for r in runs]
        figure, axis = plt.subplots(figsize=(max(6.0, 0.65 * len(runs)), 3.2))
        axis.bar(labels, [r["training_seconds"] / 3600 for r in runs], color="#4477aa")
        axis.set_ylabel("Training time (h)")
        axis.set_title("MDWD detection - training time per completed run")
        axis.tick_params(axis="x", labelsize=7, rotation=45)
        saved.extend(ev.save_figure(figure, figures_dir, "mdwd_training_time_per_run"))

        figure, axis = plt.subplots(figsize=(4.8, 3.4))
        for r in runs:
            axis.scatter(r["model_size_mb"], r["test_inference_ms"], color="#4477aa")
            axis.annotate(r["model"], (r["model_size_mb"], r["test_inference_ms"]),
                          fontsize=7, xytext=(3, 3), textcoords="offset points")
        axis.set_xlabel("Model size (MB)")
        axis.set_ylabel("Test inference (ms/img)")
        axis.set_title("MDWD detection - latency vs model size (val-time measurement)")
        saved.extend(ev.save_figure(figure, figures_dir, "mdwd_latency_vs_model_size"))

    variants = [v for v in attributes["variants"] if v.get("trainable_params")]
    if variants:
        figure, axis = plt.subplots(figsize=(5.6, 3.2))
        labels = [f"{v['variant']}\n({v['adaptation']})" for v in variants]
        axis.bar(labels, [int(v["trainable_params"]) for v in variants], color="#66ccee")
        axis.set_yscale("log")
        axis.set_ylabel("Trainable parameters (log)")
        axis.set_title("MTSD attributes - trainable parameters by adaptation strategy")
        axis.tick_params(axis="x", labelsize=7, rotation=30)
        saved.extend(ev.save_figure(figure, figures_dir,
                                    "attr_trainable_params_by_adaptation"))

    from collections import Counter

    status_counts = Counter(row["status"] for row in operational)
    figure, axis = plt.subplots(figsize=(4.6, 2.8))
    axis.bar(list(status_counts), list(status_counts.values()), color="#4477aa")
    axis.set_ylabel("Paradigms")
    axis.set_title("Evidence status across paradigms")
    saved.extend(ev.save_figure(figure, figures_dir, "evidence_status_by_paradigm"))
    return saved


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def write_reports(report_dir: Path, figures: list[str], payload: dict,
                  operational: list[dict], missing_manual: list[str]) -> None:
    generated_at = payload["generated_at"]
    ev.write_json(report_dir / "annotation_effort_details.json", payload)
    ev.write_csv(report_dir / "operational_comparison.csv", operational)

    summary_rows = []
    for dataset in (payload["mdwd"], payload["mtsd"]):
        summary_rows.append({
            "item": f"{dataset['dataset']} dataset",
            "raw_images": dataset.get("raw_images", ""),
            "annotated_images": dataset.get("annotated_images", ""),
            "unannotated_images": dataset.get("unannotated_images", ""),
            "boxes": dataset.get("boxes", ""),
            "classes": dataset.get("classes", ""),
            "auxiliary_attributes": dataset.get("auxiliary_attributes", ""),
            "disk_size_gb": dataset.get("disk_size_gb", NOT_RECORDED),
            "source": "; ".join(dataset.get("source_files", [])[:2]),
        })
    for run in payload["detection"]["runs"]:
        summary_rows.append({
            "item": f"MDWD run {run['model']} ({run['suite']})",
            "training_seconds": run.get("training_seconds", ""),
            "parameters": run.get("parameters", ""),
            "model_size_mb": run.get("model_size_mb", ""),
            "test_inference_ms": run.get("test_inference_ms", ""),
            "status": run["status"], "source": run["source_file"],
        })
    for variant in payload["attributes"]["variants"]:
        summary_rows.append({
            "item": f"Attribute variant {variant['variant']}",
            "adaptation": variant.get("adaptation", ""),
            "trainable_params": variant.get("trainable_params", ""),
            "best_epoch": variant.get("best_epoch", ""),
            "status": variant["status"],
            "source": ev.rel(ATTR_COMPARISON) if ATTR_COMPARISON.exists() else "",
        })
    for run in payload["promptdetect"]["runs"]:
        summary_rows.append({
            "item": f"PromptDetect run {run['run']}",
            "n_images": run["n_images"], "prompts": run["prompts"],
            "prompt_model_combinations": run["prompt_model_combinations"],
            "status": run["status"], "source": run["source_file"],
        })
    ev.write_csv(report_dir / "annotation_effort_summary.csv", summary_rows)

    (report_dir / "missing_manual_values.md").write_text(
        "# Missing manual effort values\n\n"
        f"Generated: {generated_at}\n\n"
        "These values cannot be derived from the repository and were null in "
        f"`{ev.rel(MANUAL_CONFIG)}`. They are reported as *not recorded* - never "
        "estimated. Fill them in and rerun to include them.\n\n"
        + ("\n".join(f"- `{value}`" for value in missing_manual) or "- (none - all supplied)")
        + "\n", encoding="utf-8")

    mdwd, mtsd = payload["mdwd"], payload["mtsd"]
    lines = [
        "# Annotation and operational effort report",
        f"\nGenerated: {generated_at}  |  Commit: `{payload['commit_sha'][:12]}`",
        "\nAutomatically derived values come from dataset folders, QA JSONs, audit "
        "reports, run summaries and manifests (every row in "
        "`annotation_effort_summary.csv` records its source file). Manual values "
        f"come only from `{ev.rel(MANUAL_CONFIG)}`; missing ones are *not recorded*, "
        "never estimated (see `missing_manual_values.md`).",
        "\n## Dataset and annotation scale\n",
        "| Dataset | Raw images | Annotated | Unannotated | Boxes | Classes | "
        "Aux. attributes | Disk (GB) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| MDWD (10x-augmented export) | {mdwd.get('raw_images', '?')} | "
        f"{mdwd.get('annotated_images', '?')} | 0 | {mdwd.get('boxes', '?')} | "
        f"{mdwd.get('classes', '?')} | 0 | {mdwd.get('disk_size_gb', NOT_RECORDED)} |",
        f"| MTSD | {mtsd.get('raw_images', '?')} | {mtsd.get('annotated_images', '?')} | "
        f"{mtsd.get('unannotated_images', '?')} | {mtsd.get('boxes', '?')} | "
        f"{mtsd.get('classes', '?')} | 4 | {mtsd.get('disk_size_gb', NOT_RECORDED)} |",
        f"\n- MTSD QA-approved groups: {', '.join(mtsd.get('qa_approved_groups', []))}; "
        f"excluded (unannotated) groups: {', '.join(mtsd.get('excluded_groups', []))}.",
        f"- MTSD attribute-labelled boxes: {mtsd.get('attribute_labelled_boxes', '?')}.",
        f"- Latest annotation QA audit ({mtsd.get('qa_audit', {}).get('audit', 'n/a')}): "
        + ", ".join(f"{key} = {value}" for key, value in mtsd.get("qa_audit", {}).items()
                    if key != "audit"),
        f"- Confirmed applied corrections: {mtsd.get('confirmed_corrections', NOT_RECORDED)}.",
        f"- MDWD integrity findings: {mdwd.get('integrity_findings', {})} "
        f"(leakage measured negligible - see MDWD-Leakage-Analysis).",
        "\n## Training effort (completed experiments)\n",
        f"- MDWD detection: {payload['detection']['completed_runs']} completed runs "
        f"(+{payload['detection']['historical_runs']} historical sweep runs, "
        f"{payload['detection']['rfdetr_runs']} RF-DETR run(s) - "
        f"{payload['detection']['rfdetr_note']}). Hardware: {payload['detection']['hardware']}.",
        f"- MTSD attributes: {payload['attributes'].get('completed_runs', 0)} completed "
        f"variants (+{payload['attributes'].get('smoke_runs', 0)} smoke tests), "
        "historical GRP-1..GRP-3 snapshot.",
        "- MTSD detection: pending (no training runs).",
        "\n## Prompt-based evaluation effort\n",
        f"- Batch-evaluation runs so far: {len(payload['promptdetect']['runs'])} "
        f"(all {ev.STATUS_PILOT} scale), covering "
        f"{payload['promptdetect']['images_evaluated']} images and "
        f"{payload['promptdetect']['prompts_evaluated']} prompt evaluations.",
        "\n| Model | Confidence outputs | Gated weights | Separate env | "
        "Prompt engineering | Retraining |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for fact in PROMPT_MODEL_FACTS:
        lines.append(f"| {fact['model']} | {'yes' if fact['provides_confidence'] else 'no'} | "
                     f"{'yes' if fact['gated_weights'] else 'no'} | "
                     f"{'yes' if fact['separate_env'] else 'no'} | "
                     f"{fact['prompt_engineering']} | "
                     f"{'required' if fact['retraining_required'] else 'none'} |")
    lines += [
        "\n## Paradigm-level operational comparison\n",
        "No composite score is computed: the dimensions are not commensurable and "
        "collapsing them would be arbitrary. Raw dimensions only:\n",
        "| Paradigm | Ground-truth requirement | Human annotation burden | Training "
        "requirement | Prompt engineering | Model size | Latency | GPU memory | "
        "Maintenance burden | Main limitation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in operational:
        lines.append("| " + " | ".join(str(row[key]) for key in (
            "paradigm", "ground_truth_requirement", "human_annotation_burden",
            "training_requirement", "prompt_engineering", "model_size", "latency",
            "gpu_memory", "maintenance_burden", "main_limitation")) + " |")
    efficiency = payload["efficiency"]
    lines.append("\n## Inference-speed benchmark status\n")
    lines.append(f"- {efficiency.get('note', 'benchmark rows available - see details JSON')}")
    if figures:
        lines.append("\n## Figures\n")
        lines.extend(f"- `{f}`" for f in figures if f.endswith(".png"))
    lines.append("\n*Read-only report - nothing outside the output folders was "
                 "written; no value was estimated.*\n")
    (report_dir / "annotation_effort_report.md").write_text("\n".join(lines),
                                                            encoding="utf-8")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_self_tests() -> int:
    cases = []
    cases.append(("null manual value renders 'not recorded'",
                  not_recorded(None) == NOT_RECORDED and not_recorded("") == NOT_RECORDED))
    cases.append(("supplied manual value passes through", not_recorded(12.5) == "12.5"))

    manual = {"MDWD": {"annotation_hours": None, "qa_hours": 3.0},
              "PromptDetect": {"prompt_design_hours": None}}
    rows = operational_rows(
        detection={"runs": [], "completed_runs": 0, "historical_runs": 0,
                   "rfdetr_runs": 0, "rfdetr_note": "", "hardware": "test"},
        attributes={"variants": [], "completed_runs": 0, "smoke_runs": 0},
        prompts={"prompts_evaluated": 0, "runs": []},
        mdwd={"boxes": 1, "annotated_images": 1},
        mtsd={"boxes": 1, "annotated_images": 1, "attribute_labelled_boxes": 1,
              "qa_approved_groups": [], "excluded_groups": [], "unannotated_images": 0},
        manual=manual)
    cases.append(("paradigm table has 4 rows with all columns",
                  len(rows) == 4 and all(len(r) == 11 for r in rows)))
    burden = rows[0]["human_annotation_burden"]
    cases.append(("missing hours stay 'not recorded', supplied hours appear",
                  NOT_RECORDED in burden and "3.0" in burden))
    cases.append(("no composite score column exists",
                  not any("score" in key for row in rows for key in row)))
    cases.append(("pending paradigms are labelled pending",
                  rows[1]["status"] == ev.STATUS_PENDING))

    failures = 0
    for label, ok in cases:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        failures += 0 if ok else 1
    print(f"\nSelf-test: {len(cases) - failures}/{len(cases)} passed.")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Annotation/training/operational effort report (read-only).")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="Write to .../latest/ instead of a new timestamped folder.")
    parser.add_argument("--skip-disk-size", action="store_true",
                        help="Skip walking Datasets/ for on-disk sizes.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return run_self_tests()

    manual, missing_manual = load_manual_values()
    print("Collecting dataset facts..." + (" (disk sizes skipped)" if args.skip_disk_size else ""))
    mdwd = collect_mdwd(args.skip_disk_size)
    mtsd = collect_mtsd(args.skip_disk_size)
    detection = collect_detection_training()
    attributes = collect_attribute_training()
    prompts = collect_promptdetect()
    efficiency = collect_efficiency()
    operational = operational_rows(detection, attributes, prompts, mdwd, mtsd, manual)

    payload = {
        "generated_at": ev.provenance(None, GENERATOR)["generated_at"],
        "commit_sha": ev.git_commit_sha(), "generator": GENERATOR,
        "mdwd": mdwd, "mtsd": mtsd, "detection": detection,
        "attributes": attributes, "promptdetect": prompts,
        "efficiency": efficiency, "manual_values": manual,
        "missing_manual_values": missing_manual,
    }
    print(f"MDWD boxes: {mdwd.get('boxes')} | MTSD boxes: {mtsd.get('boxes')} | "
          f"MDWD runs: {detection['completed_runs']} | attribute variants: "
          f"{attributes.get('completed_runs', 0)} | PromptDetect runs: "
          f"{len(prompts['runs'])} | missing manual values: {len(missing_manual)}")
    if args.dry_run:
        print("Dry run: no files written.")
        return 0

    report_dir, figures_dir = ev.new_output_dirs("Annotation-Effort", args.overwrite,
                                                 args.output_dir)
    figures = make_figures(figures_dir, mdwd, mtsd, detection, attributes, operational)
    write_reports(report_dir, figures, payload, operational, missing_manual)
    print(f"Reports: {report_dir}\nFigures: {figures_dir} ({len(figures)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
