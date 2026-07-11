#!/usr/bin/env python3
"""Consolidate existing experiment results into dissertation-ready tables.

Collects whatever result files currently exist - nothing is recomputed and no
metric is ever invented; missing sections are marked *pending* with a note.

Sources scanned:
  * MDWD detection  : Results/MDWD-Results/<family>/Model-Size-Comparison/*summary*.csv
  * MTSD detection  : Results/MTSD-Results/<family>/Model-Size-Comparison/*summary*.csv (pending until trained)
  * MTSD attributes : Scripts/MTSD-Scripts/AttributeClassification/outputs/reports/comparison.csv
  * PromptDetect    : Results/PromptDetect/BatchEvaluation/*/*/evaluation_summary.json
  * Efficiency      : Results/Inference-Benchmark/InferenceSpeed/*/inference_speed_summary.csv (latest run)

Outputs (default: a fresh timestamped folder Documents/Final-Tables/<stamp>/;
--overwrite writes fixed filenames directly into Documents/Final-Tables/):
    mdwd_detection_results.csv     mtsd_detection_results.csv
    mtsd_attribute_results.csv     promptdetect_results.csv
    model_efficiency_results.csv   final_model_comparison_table.csv
    dissertation_results_summary.md

Usage:
    python Scripts/FinalEvaluation/export_dissertation_tables.py
    python Scripts/FinalEvaluation/export_dissertation_tables.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path = SCRIPT_DIR) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "Datasets").exists() and (parent / "Scripts").exists():
            return parent
    raise RuntimeError("Could not locate project root.")


ROOT = find_project_root()
TABLES_ROOT = ROOT / "Documents" / "Final-Tables"
MDWD_RESULTS = ROOT / "Results" / "MDWD-Results"
MTSD_RESULTS = ROOT / "Results" / "MTSD-Results"
ATTR_COMPARISON = (ROOT / "Scripts" / "MTSD-Scripts" / "AttributeClassification"
                   / "outputs" / "reports" / "comparison.csv")
PROMPT_RESULTS = ROOT / "Results" / "PromptDetect" / "BatchEvaluation"
SPEED_RESULTS = ROOT / "Results" / "Inference-Benchmark" / "InferenceSpeed"


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(dict.fromkeys(k for row in rows for k in row)) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _get(row: dict, *names: str) -> str:
    for name in names:
        if name in row and row[name] not in ("", None):
            return row[name]
    return ""


# ---------------------------------------------------------------------------
# Collectors (each returns rows + notes)
# ---------------------------------------------------------------------------

def collect_detection(results_root: Path, dataset: str) -> tuple[list[dict], list[str]]:
    rows, notes = [], []
    if not results_root.is_dir() or not any(results_root.iterdir()):
        return [], [f"{dataset} detection: no results yet - pending "
                    f"({'train via Scripts/MTSD-Scripts/MTSD-SupervisedNotebooks' if dataset == 'MTSD' else results_root})"]
    for summary_path in sorted(results_root.glob("*/Model-Size-Comparison/*summary*.csv")):
        family = summary_path.parts[len(results_root.parts)]
        source_rows = read_csv_rows(summary_path)
        if not source_rows:
            notes.append(f"{dataset} {family}: summary file empty "
                         f"({summary_path.relative_to(ROOT)}) - run's consolidation step never "
                         f"completed; metrics may exist in the run folder / W&B.")
            continue
        for row in source_rows:
            if row.get("status") not in (None, "", "completed", "success", "ok"):
                notes.append(f"{dataset} {family} {_get(row, 'model_variant', 'run_name')}: "
                             f"status={row.get('status')}")
            rows.append({
                "dataset": dataset,
                "model_family": family,
                "model_variant": _get(row, "model_variant", "run_name"),
                "parameters": _get(row, "parameter_count"),
                "model_size_mb": _get(row, "model_size_mb"),
                "precision": _get(row, "precision"),
                "recall": _get(row, "recall"),
                "f1": _get(row, "F1 Score", "f1"),
                "map50": _get(row, "mAP@50", "map50"),
                "map50_95": _get(row, "mAP@50-95", "map50_95"),
                "test_map50": _get(row, "test_mAP@50"),
                "test_map50_95": _get(row, "test_mAP@50-95"),
                "val_inference_ms": _get(row, "validation_inference_time_ms"),
                "train_seconds": _get(row, "total_training_time_seconds"),
                "run_path": _get(row, "best_weights", "run_name"),
                "wandb_url": _get(row, "wandb_url"),
                "source_file": str(summary_path.relative_to(ROOT)),
                "status": _get(row, "status") or "completed",
            })
    if not rows and not notes:
        notes.append(f"{dataset} detection: no summary CSVs found under {results_root}")
    return rows, notes


def collect_attributes() -> tuple[list[dict], list[str]]:
    source_rows = read_csv_rows(ATTR_COMPARISON)
    if not source_rows:
        return [], ["MTSD attribute classification: comparison.csv not found - pending "
                    "(run AttributeClassification/run_all.py)"]
    best = max(source_rows, key=lambda r: float(_get(r, "mean_macro_f1") or 0))
    rows = []
    for row in source_rows:
        rows.append({
            "variant": row["variant"],
            "backbone": row["variant"].split("_")[0],
            "adaptation": _get(row, "adaptation"),
            "mean_macro_f1": _get(row, "mean_macro_f1"),
            "view_angle_f1": _get(row, "view_angle"),
            "mounting_f1": _get(row, "mounting"),
            "condition_f1": _get(row, "condition"),
            "sign_shape_f1": _get(row, "sign_shape"),
            "total_params": _get(row, "total_params"),
            "trainable_params": _get(row, "trainable_params"),
            "best_model": "YES" if row is best else "",
            "run_id": _get(row, "run_id"),
            "source_file": str(ATTR_COMPARISON.relative_to(ROOT)),
            "status": "completed",
        })
    notes = [f"Best attribute model: {best['variant']} "
             f"(mean macro-F1 {best.get('mean_macro_f1')}); weakest head across all "
             f"variants is 'condition'."]
    return rows, notes


def collect_promptdetect() -> tuple[list[dict], list[str]]:
    rows, notes = [], []
    summaries = sorted(PROMPT_RESULTS.glob("*/*/evaluation_summary.json")) if PROMPT_RESULTS.is_dir() else []
    if not summaries:
        return [], ["PromptDetect: no batch-evaluation runs yet - pending "
                    "(run Scripts/Other-Scripts/PromptDetect/batch_evaluation/run_batch_eval.py)"]
    for summary_path in summaries:
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            notes.append(f"unreadable {summary_path.relative_to(ROOT)}: {exc}")
            continue
        for metric_row in payload.get("per_prompt_metrics", []):
            rows.append({
                "evaluation_protocol": payload.get("evaluation_protocol", "class-agnostic-exploratory"),
                "dataset": payload.get("dataset"),
                "split": payload.get("split"),
                "model": metric_row.get("model"),
                "prompt": metric_row.get("prompt"),
                "precision": metric_row.get("precision"),
                "recall": metric_row.get("recall"),
                "f1": metric_row.get("f1"),
                "ap50": metric_row.get("ap50"),
                "map50_95": metric_row.get("map50_95"),
                "mean_matched_iou": metric_row.get("mean_matched_iou"),
                "n_images": metric_row.get("n_images"),
                "run_folder": str(summary_path.parent.relative_to(ROOT)),
                "status": "completed",
            })
        protocol = payload.get("evaluation_protocol", "class-agnostic-exploratory")
        if protocol == "class-agnostic-exploratory":
            notes.append(f"{summary_path.parent.relative_to(ROOT)} is exploratory class-agnostic evidence")
    return rows, notes


def collect_efficiency() -> tuple[list[dict], list[str]]:
    runs = sorted(SPEED_RESULTS.glob("*/inference_speed_summary.csv")) if SPEED_RESULTS.is_dir() else []
    if not runs:
        return [], ["Model efficiency: no inference-speed benchmark runs yet - pending "
                    "(run Scripts/Other-Scripts/Inference-Benchmark/inference_speed_benchmark.py)"]
    rows = []
    for summary_path in runs:
        for row in read_csv_rows(summary_path):
            row["source_file"] = str(summary_path.relative_to(ROOT))
            rows.append(row)
    return rows, [f"Efficiency rows aggregated from {len(runs)} benchmark run(s)."]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Export dissertation-ready result tables (read-only sources).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Write fixed filenames directly into Documents/Final-Tables/ "
                             "(default: a new timestamped subfolder).")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = TABLES_ROOT if args.overwrite else TABLES_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    sections: dict[str, tuple[list[dict], list[str]]] = {
        "mdwd_detection_results": collect_detection(MDWD_RESULTS, "MDWD"),
        "mtsd_detection_results": collect_detection(MTSD_RESULTS, "MTSD"),
        "mtsd_attribute_results": collect_attributes(),
        "promptdetect_results": collect_promptdetect(),
        "model_efficiency_results": collect_efficiency(),
    }

    all_notes: dict[str, list[str]] = {}
    comparison_rows: list[dict] = []
    for name, (rows, notes) in sections.items():
        write_csv(out_dir / f"{name}.csv",
                  rows or [{"status": "pending", "note": "; ".join(notes)}])
        all_notes[name] = notes
        print(f"{name}: {len(rows)} rows" + (f" | notes: {len(notes)}" if notes else ""))

    # Cross-track comparison (long form; only completed rows).
    for row in sections["mdwd_detection_results"][0]:
        comparison_rows.append({"track": "MDWD detection", "model": row["model_variant"],
                                "key_metric": "val mAP@50-95", "value": row["map50_95"],
                                "secondary_metric": "val F1", "secondary_value": row["f1"],
                                "source": row["source_file"]})
    for row in sections["mtsd_attribute_results"][0]:
        comparison_rows.append({"track": "MTSD attributes", "model": row["variant"],
                                "key_metric": "mean macro-F1", "value": row["mean_macro_f1"],
                                "secondary_metric": "condition F1", "secondary_value": row["condition_f1"],
                                "source": row["source_file"]})
    for row in sections["promptdetect_results"][0]:
        comparison_rows.append({"track": f"PromptDetect ({row['dataset']})",
                                "model": f"{row['model']} | {row['prompt']}",
                                "key_metric": "F1", "value": row["f1"],
                                "secondary_metric": "AP@50", "secondary_value": row["ap50"],
                                "source": row["run_folder"]})
    write_csv(out_dir / "final_model_comparison_table.csv",
              comparison_rows or [{"status": "pending"}])

    # Markdown summary -----------------------------------------------------------
    lines = [
        "# Dissertation results summary",
        f"\nGenerated: {datetime.now().isoformat(timespec='seconds')} - consolidated from "
        "existing result files only (no metrics recomputed or invented).\n",
    ]
    titles = {
        "mdwd_detection_results": "MDWD supervised detection",
        "mtsd_detection_results": "MTSD supervised detection",
        "mtsd_attribute_results": "MTSD attribute classification",
        "promptdetect_results": "PromptDetect (prompt-based detection)",
        "model_efficiency_results": "Model efficiency (inference speed)",
    }
    for name, (rows, notes) in sections.items():
        lines.append(f"\n## {titles[name]}\n")
        if not rows:
            lines.append(f"**PENDING** - {'; '.join(notes)}")
            continue
        columns = [c for c in rows[0] if c not in ("source_file", "run_path", "wandb_url",
                                                    "run_folder", "checkpoint")][:10]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "---|" * len(columns))
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
        for note in notes:
            lines.append(f"\n> {note}")
        lines.append(f"\n*Full table incl. source paths: `{name}.csv`*")
    lines.append("\n---\n*Source files are recorded per row for traceability. Pending "
                 "sections list the command that produces them.*\n")
    (out_dir / "dissertation_results_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nTables written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
