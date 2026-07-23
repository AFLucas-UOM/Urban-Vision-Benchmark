#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNNER = HERE / "run_mtsd_supervised.py"
STATE_ROOT = ROOT / "Results" / "MTSD-Runs" / "Supervised-Matrix"
OUTPUT_ROOT = ROOT / "Results" / "MTSD-Results" / "Followup-Experiments"
PILOT_MODELS = ("yolo11s", "yolo11m", "yolo12m")
SMALL_METRIC_MIN_GAIN = 0.02
MAX_MEAN_MAP_DROP = 0.01


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _load_metrics_from_state(state_path: Path, models: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("final_status") != "completed":
        raise RuntimeError(f"Experiment state is not complete: {state_path}")
    metrics = {}
    for model in models:
        record_path = Path(state["models"][model]["run_dir"]) / "run_record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("unified_eval_status") != "completed":
            raise RuntimeError(f"Unified evaluation did not complete for {model}: {record_path}")
        metrics[model] = record["unified_test_metrics"]
    return metrics


def resolution_decision(
    baseline: dict[str, dict[str, Any]],
    pilot: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    comparisons = []
    for model in PILOT_MODELS:
        base = baseline[model]
        trial = pilot[model]
        comparisons.append({
            "model": model,
            "baseline_map50_95": base["map50_95"],
            "pilot_map50_95": trial["map50_95"],
            "map50_95_gain": trial["map50_95"] - base["map50_95"],
            "baseline_map_small": base["map_small"],
            "pilot_map_small": trial["map_small"],
            "map_small_gain": trial["map_small"] - base["map_small"],
            "baseline_ar_small": base["ar_small"],
            "pilot_ar_small": trial["ar_small"],
            "ar_small_gain": trial["ar_small"] - base["ar_small"],
        })
    mean = lambda key: sum(float(row[key]) for row in comparisons) / len(comparisons)
    mean_map_gain = mean("map50_95_gain")
    mean_small_ap_gain = mean("map_small_gain")
    mean_small_ar_gain = mean("ar_small_gain")
    proceed = (
        (mean_small_ap_gain >= SMALL_METRIC_MIN_GAIN or mean_small_ar_gain >= SMALL_METRIC_MIN_GAIN)
        and mean_map_gain >= -MAX_MEAN_MAP_DROP
    )
    return {
        "proceed_to_1280": proceed,
        "registered_rule": {
            "small_object_requirement": (
                f"mean AP-small gain >= {SMALL_METRIC_MIN_GAIN:.2f} OR "
                f"mean AR-small gain >= {SMALL_METRIC_MIN_GAIN:.2f} across all three pilots"
            ),
            "overall_guardrail": f"mean mAP50-95 loss must not exceed {MAX_MEAN_MAP_DROP:.2f}",
        },
        "mean_map50_95_gain": mean_map_gain,
        "mean_map_small_gain": mean_small_ap_gain,
        "mean_ar_small_gain": mean_small_ar_gain,
        "models": comparisons,
    }


def _baseline_metrics(path: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = {row["model"]: row for row in manifest["model_metrics"]}
    missing = [model for model in PILOT_MODELS if model not in rows]
    if missing:
        raise ValueError(f"Baseline unified evaluation is missing pilot models: {missing}")
    return rows


def _run(command: list[str], manifest: dict[str, Any], manifest_path: Path, stage: str) -> None:
    manifest["current_stage"] = stage
    manifest["stages"][stage] = {"status": "running", "command": command,
                                  "started_at": datetime.now(timezone.utc).isoformat()}
    _write_json(manifest_path, manifest)
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["stages"][stage].update(status="failed", error=str(exc),
                                          finished_at=datetime.now(timezone.utc).isoformat())
        _write_json(manifest_path, manifest)
        raise
    manifest["stages"][stage].update(status="completed", finished_at=datetime.now(timezone.utc).isoformat())
    _write_json(manifest_path, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run registered MTSD clean-ablation and resolution follow-ups")
    parser.add_argument("--baseline-evaluation", type=Path, required=True,
                        help="All-13 no-tourist evaluation_manifest.json used as the 640 baseline")
    parser.add_argument("--label", default=datetime.now().strftime("%Y%m%d-followups"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.label
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "experiment_manifest.json"
    baseline = _baseline_metrics(args.baseline_evaluation)
    ablation_label = f"{args.label}-clean-ablation"
    pilot960_label = f"{args.label}-resolution-960"
    pilot1280_label = f"{args.label}-resolution-1280"
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "baseline_evaluation": str(args.baseline_evaluation.resolve()),
        "excluded_evaluation_categories": ["Tourist Sign"],
        "training_taxonomy": "unchanged 12-class taxonomy to preserve causal comparability",
        "clean_ablation": {
            "models": ["yolo11m", "yolo12m", "yolo26m"],
            "dataset_variant": "unaugmented",
            "split_policy": "existing shared split manifest",
            "rf_detr_excluded": True,
        },
        "resolution_pilot": {
            "models": list(PILOT_MODELS),
            "dataset_variant": "augmented",
            "yolo_online_augmentation": "traffic_metric_push",
            "image_sizes": [960, "1280 conditional"],
            "fixed": {
                "ultralytics_optimizer_effective_batch": 64,
                "physical_batch_by_image_size": {"640": 32, "960": 16, "1280": 8},
                "optimizer": "AdamW", "seed": 42,
            },
            "decision_thresholds": {
                "mean_small_ap_or_ar_gain": SMALL_METRIC_MIN_GAIN,
                "maximum_mean_map50_95_drop": MAX_MEAN_MAP_DROP,
            },
        },
        "stages": {},
    }
    _write_json(manifest_path, manifest)
    common = [str(sys.executable), str(RUNNER), "--device", args.device, "--require-unified-eval"]
    _run(common + [
        "--dataset-variant", "unaugmented", "--matrix", "aug_ablation",
        "--run-label", ablation_label, "--wandb-group", "mtsd-clean-augmentation-ablation-v1",
    ], manifest, manifest_path, "clean_ablation")
    manifest["stages"]["clean_ablation"]["state"] = str((STATE_ROOT / ablation_label / "state.json").resolve())
    _write_json(manifest_path, manifest)
    _run(common + [
        "--final", "--dataset-variant", "augmented", "--models", *PILOT_MODELS,
        "--image-size", "960", "--yolo-online-augmentation", "traffic_metric_push",
        "--run-label", pilot960_label,
        "--wandb-group", "mtsd-resolution-960-pilot-v1",
    ], manifest, manifest_path, "resolution_960")
    pilot960_state = STATE_ROOT / pilot960_label / "state.json"
    pilot960 = _load_metrics_from_state(pilot960_state, PILOT_MODELS)
    decision = resolution_decision(baseline, pilot960)
    manifest["resolution_decision"] = decision
    manifest["stages"]["resolution_960"]["state"] = str(pilot960_state.resolve())
    _write_json(manifest_path, manifest)
    if decision["proceed_to_1280"]:
        _run(common + [
            "--final", "--dataset-variant", "augmented", "--models", *PILOT_MODELS,
            "--image-size", "1280", "--yolo-online-augmentation", "traffic_metric_push",
            "--run-label", pilot1280_label,
            "--wandb-group", "mtsd-resolution-1280-conditional-v1",
        ], manifest, manifest_path, "resolution_1280")
        manifest["stages"]["resolution_1280"]["state"] = str((STATE_ROOT / pilot1280_label / "state.json").resolve())
    else:
        manifest["stages"]["resolution_1280"] = {
            "status": "skipped", "reason": "The registered 960 small-object improvement rule was not met."
        }
    manifest.update(status="completed", current_stage=None, finished_at=datetime.now(timezone.utc).isoformat())
    _write_json(manifest_path, manifest)
    print(json.dumps({"status": "completed", "manifest": str(manifest_path),
                      "proceeded_to_1280": decision["proceed_to_1280"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
