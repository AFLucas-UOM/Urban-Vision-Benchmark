#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))

import config
from dataset_loader import ground_truth_index_rows, load_ground_truth
from persistence import combination_dir, compatible_complete, write_csv, write_json
from prompt_runner import prediction_boxes, run_models
from protocol import load_protocol, select_prompts, validate_vocabulary
from protocol_reporting import generate
from run_batch_eval import resolve_models
from targeted_metrics import evaluate_targeted

DEFAULT_PROTOCOL = HERE / "prompt_protocols" / "dissertation_protocol.yaml"


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256(); digest.update(path.read_bytes()); return digest.hexdigest()


def enforce_final_mtsd(gt: dict, spec: dict) -> dict:
    source_cfg = spec["source"]
    expected_source = (config.PROJECT_ROOT / source_cfg["path"]).resolve()
    actual_source = Path(gt["source"]).resolve()
    decisions = {"final_mode": True, "qa_fallback_refused": True,
                 "canonical_source_required": str(expected_source), "canonical_source_actual": str(actual_source)}
    if gt.get("source_type") != "prepared" or actual_source != expected_source:
        raise RuntimeError(f"Final MTSD evaluation requires canonical prepared source {expected_source}; got {actual_source}")
    manifest_path = Path(gt.get("manifest_path") or "")
    split_path = actual_source.parent / "split_manifest.csv"
    if not manifest_path.is_file() or not split_path.is_file():
        raise RuntimeError("Final MTSD evaluation requires prep_manifest.json and split_manifest.csv")
    manifest = gt.get("manifest") or json.loads(manifest_path.read_text(encoding="utf-8"))
    unsigned = dict(manifest); recorded_fingerprint = unsigned.pop("manifest_fingerprint", None)
    import hashlib
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    if hashlib.sha256(canonical.encode()).hexdigest() != recorded_fingerprint:
        raise RuntimeError("Prepared manifest fingerprint mismatch")
    if sha256(split_path) != manifest.get("split_manifest_sha256"):
        raise RuntimeError("Prepared split-manifest hash mismatch")
    if manifest.get("group_scope") != source_cfg.get("required_group_scope"):
        raise RuntimeError("Final MTSD evaluation refuses auto-discovered group scope")
    if manifest.get("approved_groups") != source_cfg.get("approved_groups"):
        raise RuntimeError("Prepared approved group scope differs from protocol lock")
    if manifest.get("annotation_source_mode") != source_cfg.get("required_annotation_source_mode"):
        raise RuntimeError("Final MTSD evaluation requires QA-only annotation source")
    gate = manifest.get("qa_gate", {})
    if not gate.get("resolved") or gate.get("override_used"):
        raise RuntimeError("Final MTSD evaluation requires a resolved, non-overridden QA gate")
    decisions.update(manifest_verified=True, split_manifest_verified=True,
                     group_scope=manifest["group_scope"], approved_groups=manifest["approved_groups"],
                     annotation_source_mode=manifest["annotation_source_mode"],
                     qa_gate_status=gate.get("resolution_status"))
    return decisions


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Targeted PromptDetect dissertation protocol")
    p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    p.add_argument("--dataset", choices=("MDWD", "MTSD", "both", "mdwd", "mtsd"), default="both")
    p.add_argument("--split", default="test"); p.add_argument("--models", nargs="+")
    p.add_argument("--prompt-ids", nargs="+"); p.add_argument("--prompt-group", choices=("class-targeted", "synonym-comparison", "broad", "optional-broad"))
    p.add_argument("--include-optional-prompts", action="store_true")
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--max-images", type=int); p.add_argument("--conf-threshold", type=float); p.add_argument("--iou-threshold", type=float)
    p.add_argument("--resume", type=Path); p.add_argument("--skip-completed", action="store_true")
    p.add_argument("--save-visualizations", nargs="?", const=-1, type=int, default=0)
    p.add_argument("--allow-heavy", action="store_true", help="Required for optional Cosmos Reason2 32B")
    p.add_argument("--final", action="store_true", help="For MTSD, enforce canonical unaugmented source, locked scope, verified manifests and resolved QA gate")
    p.add_argument("--allow-qa-fallback", action="store_true", help="Development-only MTSD fallback; incompatible with --final")
    p.add_argument("--run-label")
    p.add_argument("--reports-only", type=Path)
    return p


def _manifest_hash(gt: dict) -> str:
    if gt.get("manifest_path"): return sha256(Path(gt["manifest_path"]))
    data_yaml = Path(gt["source"]) / "data.yaml"
    return sha256(data_yaml) if data_yaml.is_file() else "qa-fallback-no-manifest"


def _dry_stats(gt: dict, prompts: list[dict]) -> list[dict]:
    rows = []
    for prompt in prompts:
        target = set(prompt["target_classes"])
        counts = [sum(box["class_name"] in target for box in record["boxes"]) for record in gt["records"]]
        rows.append({"prompt_id": prompt["id"], "prompt": prompt["prompt"], "group": prompt["group"],
                     "target_classes": prompt["target_classes"], "positive_images": sum(value > 0 for value in counts),
                     "negative_images": sum(value == 0 for value in counts), "target_boxes": sum(counts)})
    return rows


def _collect(run_dir: Path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    metrics, predictions, per_image, overlaps = [], [], [], []
    for directory in sorted((run_dir / "combinations").glob("*")):
        for path, target in ((directory / "metrics.json", metrics),):
            if path.is_file(): target.append(json.loads(path.read_text(encoding="utf-8")))
        for filename, target in (("predictions.csv", predictions), ("per_image.csv", per_image), ("fp_nontarget_overlap.csv", overlaps)):
            path = directory / filename
            if path.is_file() and path.stat().st_size:
                import csv
                with path.open(encoding="utf-8", newline="") as stream: target.extend(csv.DictReader(stream))
    return metrics, predictions, per_image, overlaps


def reports_only(run_dir: Path) -> int:
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    rows, predictions, per_image, overlaps = _collect(run_dir)
    write_csv(run_dir / "predictions.csv", predictions)
    generate(run_dir, cfg, rows, per_image, overlaps)
    print(run_dir); return 0


def run_dataset(args, protocol: dict, dataset: str, model_aliases: list[str]) -> Path:
    defaults = protocol["defaults"]
    conf = args.conf_threshold if args.conf_threshold is not None else float(defaults["conf_threshold"])
    iou = args.iou_threshold if args.iou_threshold is not None else float(defaults["iou_threshold"])
    max_images = 5 if args.smoke_test and args.max_images is None else args.max_images
    gt = load_ground_truth(dataset, args.split, max_images)
    spec = protocol["datasets"][dataset]
    validate_vocabulary(dataset, spec, gt["class_names"])
    prompts = select_prompts(spec, args.prompt_ids, args.prompt_group, args.include_optional_prompts)
    enforcement = enforce_final_mtsd(gt, spec) if args.final and dataset == "MTSD" else {"final_mode": bool(args.final)}
    if gt.get("source_type") == "qa_fallback" and not (args.allow_qa_fallback or args.dry_run or args.smoke_test):
        raise RuntimeError("MTSD QA fallback requires --allow-qa-fallback for development runs")
    models = resolve_models(model_aliases, args.allow_heavy)
    manifest_hash = _manifest_hash(gt)
    plan = {"dataset": dataset, "split": gt["split"], "source": gt["source"], "source_type": gt.get("source_type"),
            "images": len(gt["records"]), "models": models, "prompts": _dry_stats(gt, prompts),
            "predict_calls": len(gt["records"]) * len(models) * len(prompts)}
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return Path()
    if args.resume:
        run_dir = args.resume
        existing = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        for key, value in {"protocol_hash": protocol["protocol_hash"], "dataset_manifest_hash": manifest_hash,
                           "split": gt["split"], "conf_threshold": conf, "iou_threshold": iou}.items():
            if existing.get(key) != value: raise ValueError(f"Resume mismatch for {key}: {existing.get(key)!r} != {value!r}")
    else:
        label = args.run_label or ("smoke" if args.smoke_test else None)
        run_dir = config.new_run_dir(dataset, label)
        snapshot = dict(protocol); snapshot.pop("protocol_path", None)
        (run_dir / "protocol_snapshot.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
        (run_dir / "protocol_hash").write_text(protocol["protocol_hash"] + "\n", encoding="utf-8")
    run_config = {"evaluation_protocol": "targeted-v1", "protocol_version": protocol["protocol_version"],
                  "protocol_hash": protocol["protocol_hash"], "dataset": dataset, "split": gt["split"],
                  "source": gt["source"], "source_type": gt.get("source_type"), "dataset_manifest_hash": manifest_hash,
                  "dataset_manifest": gt.get("manifest"), "models": models, "prompt_ids": [row["id"] for row in prompts],
                  "approved_group_scope": (gt.get("manifest") or {}).get("approved_groups"),
                  "group_scope": (gt.get("manifest") or {}).get("group_scope"),
                  "final_enforcement": enforcement,
                  "conf_threshold": conf, "iou_threshold": iou, "max_detections": defaults["max_detections"],
                  "n_images": len(gt["records"]), "started_at": datetime.now(timezone.utc).isoformat()}
    write_json(run_dir / "run_config.json", run_config)
    write_csv(run_dir / "ground_truth_index.csv", ground_truth_index_rows(gt))
    by_text = {row["prompt"]: row for row in prompts}
    for model in models:
        pending = []
        for prompt in prompts:
            expected = {"protocol_hash": protocol["protocol_hash"], "dataset_manifest_hash": manifest_hash,
                        "model": model, "prompt_id": prompt["id"], "conf_threshold": conf,
                        "iou_threshold": iou, "split": gt["split"]}
            directory = combination_dir(run_dir, model, prompt["id"])
            if args.skip_completed and compatible_complete(directory, expected): continue
            pending.append(prompt)
        if not pending: continue
        def completed(model_label, prompt_text, prediction_rows, runtime):
            prompt = by_text[prompt_text]; directory = combination_dir(run_dir, model_label, prompt["id"])
            boxes = prediction_boxes(prediction_rows, model_label, prompt_text)
            result = evaluate_targeted(boxes, gt["records"], prompt["target_classes"], iou, config.MAP_IOU_RANGE)
            real_rows = [{**row, "prompt_id": prompt["id"]} for row in prediction_rows]
            write_csv(directory / "predictions.csv", real_rows)
            per_image = [{"model": model_label, "prompt_id": prompt["id"], "prompt": prompt_text, **row} for row in result["per_image"]]
            overlaps = [{"model": model_label, "prompt_id": prompt["id"], "prompt": prompt_text, **row} for row in result["fp_nontarget_overlap"]]
            write_csv(directory / "per_image.csv", per_image); write_csv(directory / "fp_nontarget_overlap.csv", overlaps)
            metric = {"dataset": dataset, "split": gt["split"], "model": model_label, "prompt_id": prompt["id"],
                      "prompt": prompt_text, "prompt_group": prompt["group"], "target_classes": prompt["target_classes"],
                      "prompt_class_confusion": result["prompt_class_confusion"],
                      **result["summary"], **runtime}
            infer = [float(row["inference_ms"]) for row in prediction_rows if row.get("inference_ms") not in (None, "")]
            metric["mean_inference_ms"] = round(sum(infer) / len(infer), 2) if infer else None
            metric["has_confidence"] = all(str(row.get("has_confidence", True)).lower() == "true" for row in prediction_rows)
            metric["ap_meaningful"] = metric["has_confidence"]
            write_json(directory / "metrics.json", metric)
            status = {"status": "completed", "protocol_hash": protocol["protocol_hash"], "dataset_manifest_hash": manifest_hash,
                      "model": model_label, "prompt_id": prompt["id"], "conf_threshold": conf, "iou_threshold": iou,
                      "split": gt["split"], "target_gt_count": metric["target_gt_boxes"], "tp": metric["tp"],
                      "fp": metric["fp"], "fn": metric["fn"], "finished_at": datetime.now(timezone.utc).isoformat(), **runtime}
            write_json(directory / "status.json", status)
        try:
            run_models(gt, [row["prompt"] for row in pending], [model], conf, int(defaults["max_detections"]),
                       on_combination_complete=completed)
        except Exception as exc:
            # Preserve every earlier prompt and make each unfinished combination
            # explicitly resumable instead of losing the whole model's progress.
            for prompt in pending:
                directory = combination_dir(run_dir, model, prompt["id"])
                status_path = directory / "status.json"
                if status_path.is_file():
                    continue
                write_json(status_path, {"status": "failed", "error": str(exc),
                    "protocol_hash": protocol["protocol_hash"], "dataset_manifest_hash": manifest_hash,
                    "model": model, "prompt_id": prompt["id"], "conf_threshold": conf,
                    "iou_threshold": iou, "split": gt["split"],
                    "finished_at": datetime.now(timezone.utc).isoformat()})
            print(f"FAILED {model}: {exc}", file=sys.stderr)
    metrics, predictions, per_image, overlaps = _collect(run_dir)
    write_csv(run_dir / "predictions.csv", predictions)
    run_config["finished_at"] = datetime.now(timezone.utc).isoformat()
    generate(run_dir, run_config, metrics, per_image, overlaps)
    print(run_dir); return run_dir


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.final and args.allow_qa_fallback:
        raise ValueError("--allow-qa-fallback is incompatible with --final")
    if args.reports_only: return reports_only(args.reports_only)
    protocol = load_protocol(args.protocol)
    aliases = args.models or list(protocol["models"]["primary"])
    datasets = ["MDWD", "MTSD"] if args.dataset.lower() == "both" else [args.dataset.upper()]
    if args.resume and len(datasets) != 1: raise ValueError("--resume requires one dataset")
    for dataset in datasets: run_dataset(args, protocol, dataset, aliases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
