"""Build dissertation-facing MTSD experiment tables from the read-only audit.

The audit remains the source of truth.  This script only reads the audit CSV
and writes regenerable comparison/cost/missingness tables under Documents.
"""
from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
AUDIT = REPO / "existing_mtSD_experiment_audit.csv"
REPORT_DIR = REPO / "Documents" / "Final-Reports" / "MTSD-SupervisedDetection"


def rows() -> list[dict[str, str]]:
    with AUDIT.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f(value: str) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def write_csv(path: Path, data: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for item in data:
        for key in item:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def write_md(path: Path, title: str, data: list[dict[str, object]]) -> None:
    keys = list(data[0]) if data else ["status"]
    lines = [f"# {title}", "", "Generated from `existing_mtSD_experiment_audit.csv`.", ""]
    if not data:
        lines.append("No matching rows are currently available.")
    else:
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("| " + " | ".join("---" for _ in keys) + " |")
        for item in data:
            lines.append("| " + " | ".join(str(item.get(key, "")).replace("|", "\\|") for key in keys) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def completed(data: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in data if row.get("completion_status") == "completed"]


def main() -> None:
    audit_rows = rows()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(AUDIT, REPORT_DIR / "mtsd_complete_experiment_matrix.csv")

    primary: list[dict[str, object]] = []
    # The main comparison is the Strong regime at the intended primary
    # resolution for each family; rows are retained even when historical runs
    # used a different evaluation split, which is stated in the audit.
    primary_sizes = {"YOLO11": "1280", "YOLO12": "640", "YOLO26": "1280", "RF-DETR": "384"}
    for row in completed(audit_rows):
        if row.get("augmentation_regime") != "Strong":
            continue
        if row.get("image_size") != primary_sizes.get(row.get("architecture", "")):
            continue
        primary.append({
            "architecture": row.get("architecture"), "scale": row.get("model_scale"),
            "image_size": row.get("image_size"), "augmentation": row.get("augmentation_regime"),
            "evaluation_split": row.get("evaluation_split"), "status": row.get("completion_status"),
            "val_mAP50": f(row.get("val_map50", "")), "val_mAP50_95": f(row.get("val_map50_95", "")),
            "val_precision": f(row.get("val_precision", "")), "val_recall": f(row.get("val_recall", "")),
            "epochs": row.get("epochs_completed"), "best_epoch": row.get("best_epoch"),
            "training_seconds": row.get("training_duration_seconds"), "run": row.get("run_path"),
        })
    primary.sort(key=lambda r: (str(r["architecture"]), str(r["scale"]), str(r["run"])))
    write_csv(REPORT_DIR / "mtsd_main_scale_comparison.csv", primary)
    write_md(REPORT_DIR / "mtsd_main_scale_comparison.md", "MTSD Main N/S/M Scale Comparison", primary)

    ablation: list[dict[str, object]] = []
    groups: defaultdict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in completed(audit_rows):
        groups[(row.get("architecture", ""), row.get("model_scale", ""), row.get("image_size", ""))].append(row)
    for key, group in sorted(groups.items()):
        regimes = {row.get("augmentation_regime") for row in group}
        if not {"Good", "Strong", "NoAug"}.issubset(regimes):
            continue
        for row in sorted(group, key=lambda r: (r.get("augmentation_regime", ""), r.get("run_path", ""))):
            ablation.append({
                "architecture": key[0], "scale": key[1], "image_size": key[2],
                "augmentation": row.get("augmentation_regime"), "evaluation_split": row.get("evaluation_split"),
                "val_mAP50": f(row.get("val_map50", "")), "val_mAP50_95": f(row.get("val_map50_95", "")),
                "training_seconds": row.get("training_duration_seconds"), "run": row.get("run_path"),
            })
    write_csv(REPORT_DIR / "mtsd_augmentation_ablation.csv", ablation)
    write_md(REPORT_DIR / "mtsd_augmentation_ablation.md", "MTSD Augmentation Ablation", ablation)

    resolution: list[dict[str, object]] = []
    by_model_regime: defaultdict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in completed(audit_rows):
        by_model_regime[(row.get("architecture", ""), row.get("model_scale", ""), row.get("augmentation_regime", ""))].append(row)
    for key, group in sorted(by_model_regime.items()):
        if len({row.get("image_size") for row in group}) < 2:
            continue
        for row in sorted(group, key=lambda r: (r.get("image_size", ""), r.get("run_path", ""))):
            resolution.append({
                "architecture": key[0], "scale": key[1], "augmentation": key[2],
                "image_size": row.get("image_size"), "evaluation_split": row.get("evaluation_split"),
                "val_mAP50": f(row.get("val_map50", "")), "val_mAP50_95": f(row.get("val_map50_95", "")),
                "training_seconds": row.get("training_duration_seconds"), "run": row.get("run_path"),
            })
    write_csv(REPORT_DIR / "mtsd_resolution_ablation.csv", resolution)
    write_md(REPORT_DIR / "mtsd_resolution_ablation.md", "MTSD Resolution Ablation", resolution)

    cost: list[dict[str, object]] = []
    for row in completed(audit_rows):
        cost.append({
            "architecture": row.get("architecture"), "scale": row.get("model_scale"),
            "augmentation": row.get("augmentation_regime"), "image_size": row.get("image_size"),
            "physical_batch": row.get("batch_size_physical"), "effective_batch": row.get("effective_batch_size"),
            "epochs_completed": row.get("epochs_completed"), "training_seconds": row.get("training_duration_seconds"),
            "seconds_per_epoch": row.get("time_per_epoch_seconds"),
            "peak_gpu_memory_mib": row.get("peak_training_gpu_memory_mib"),
            "inference_seconds": row.get("inference_seconds"), "inference_fps": row.get("inference_fps"),
            "run": row.get("run_path"),
        })
    cost.sort(key=lambda r: (str(r["architecture"]), str(r["scale"]), str(r["augmentation"]), str(r["image_size"]), str(r["run"])))
    write_csv(REPORT_DIR / "mtsd_computational_cost.csv", cost)
    write_md(REPORT_DIR / "mtsd_computational_cost.md", "MTSD Computational Cost", cost)

    missing: list[dict[str, object]] = []
    requested = [
        ("YOLO11", "n", "Strong", "1280"), ("YOLO26", "n", "Strong", "1280"),
        ("YOLO12", "n", "Strong", "640"), ("YOLO12", "s", "Strong", "640"),
        ("YOLO12", "m", "Strong", "640"), ("RF-DETR", "s", "Strong", "512"),
        ("RF-DETR", "m", "Strong", "576"), ("RF-DETR", "m", "NoAug", "576"),
        ("YOLO26", "s", "NoAug", "960"), ("YOLO26", "s", "NoAug", "1280"),
        ("YOLO26", "s", "Good", "960"), ("YOLO12", "s", "NoAug", "640"),
    ]
    for architecture, scale, regime, image_size in requested:
        matches = [row for row in audit_rows if row.get("architecture") == architecture and row.get("model_scale") == scale and row.get("augmentation_regime") == regime and row.get("image_size") == image_size]
        status = "completed" if any(row.get("completion_status") == "completed" for row in matches) else (matches[0].get("completion_status", "missing") if matches else "missing")
        missing.append({"architecture": architecture, "scale": scale, "augmentation": regime, "image_size": image_size, "status": status, "matching_runs": len(matches), "runs": "; ".join(row.get("run_path", "") for row in matches)})
    write_csv(REPORT_DIR / "mtsd_missing_failed_report.csv", missing)
    write_md(REPORT_DIR / "mtsd_missing_failed_report.md", "MTSD Missing and Failed Experiments", missing)


if __name__ == "__main__":
    main()
