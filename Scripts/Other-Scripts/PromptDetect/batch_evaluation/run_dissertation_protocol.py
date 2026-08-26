#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))

import config
from dataset_loader import ground_truth_index_rows, load_ground_truth
from persistence import combination_dir, compatible_complete, write_csv, write_json
from prompt_runner import prediction_boxes
from prompt_sensitivity import DEFAULT_CONSISTENCY_IOU, has_sensitivity_prompts, validate_family_gt_counts
from protocol import load_protocol, select_prompts, validate_vocabulary
from protocol_reporting import generate
from run_batch_eval import resolve_models
from sensitivity_reporting import generate_sensitivity
from targeted_metrics import evaluate_targeted
from wandb_utils import log_and_finish, log_combination, start_run, tracking_target

DEFAULT_PROTOCOL = HERE / "prompt_protocols" / "dissertation_protocol.yaml"
SENSITIVITY_PROTOCOL = HERE / "prompt_protocols" / "prompt_sensitivity_protocol.yaml"
COMBINATION_WORKER = HERE / "combination_worker.py"
EXECUTION_STRATEGY = "isolated-model-prompt-chunks-v1"


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
    p.add_argument("--protocol", type=Path, default=None,
                   help="Protocol YAML. Default: dissertation protocol; with --dataset both and no "
                        "explicit --protocol, the dissertation AND prompt-sensitivity protocols both "
                        "run as separate stages with separate run directories.")
    p.add_argument("--dataset", choices=("MDWD", "MTSD", "both", "mdwd", "mtsd"), default="both")
    p.add_argument("--split", default="test"); p.add_argument("--models", nargs="+")
    p.add_argument("--prompt-ids", nargs="+")
    p.add_argument("--prompt-group", help="Restrict to one prompt group; validated against the groups defined in the loaded protocol")
    p.add_argument("--include-optional-prompts", action="store_true")
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--max-images", type=int); p.add_argument("--conf-threshold", type=float); p.add_argument("--iou-threshold", type=float)
    p.add_argument("--consistency-iou", type=float, default=DEFAULT_CONSISTENCY_IOU,
                   help="IoU threshold for prompt-pair prediction-consistency matching (sensitivity runs)")
    p.add_argument("--resume", type=Path); p.add_argument("--skip-completed", action="store_true")
    p.add_argument("--save-visualizations", nargs="?", const=-1, type=int, default=None,
                   help="Save N deterministic TP/FP/FN-stratified comparison sheets per "
                        "model/prompt; omit N to save every evaluated image. If the flag is "
                        "absent, use the protocol default (10 in v2)")
    p.add_argument("--allow-heavy", action="store_true", help="Required for optional Cosmos Reason2 32B")
    p.add_argument("--final", action="store_true", help="For MTSD, enforce canonical unaugmented source, locked scope, verified manifests and resolved QA gate")
    p.add_argument("--allow-qa-fallback", action="store_true", help="Development-only MTSD fallback; incompatible with --final")
    p.add_argument("--run-label")
    p.add_argument("--reports-only", type=Path)
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online",
                   help="W&B tracking mode. Final evaluations must use online (default) for "
                        "the configured PromptDetect project; offline/disabled are development-only.")
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
        row = {"prompt_id": prompt["id"], "prompt": prompt["prompt"], "group": prompt["group"],
               "target_classes": prompt["target_classes"], "positive_images": sum(value > 0 for value in counts),
               "negative_images": sum(value == 0 for value in counts), "target_boxes": sum(counts)}
        if "sensitivity_family" in prompt:
            row["sensitivity_family"] = prompt["sensitivity_family"]
            row["variant_type"] = prompt["variant_type"]
        rows.append(row)
    return rows


def derive_detection_limit(gt: dict, defaults: dict) -> dict:
    """Resolve and document the per-image output cap for this dataset split."""
    policy = defaults.get("max_detections_policy", "fixed")
    counts = [(record["image_id"], len(record["boxes"])) for record in gt["records"]]
    busiest_id, busiest_count = max(counts, key=lambda item: item[1], default=(None, 0))
    if policy == "test-set-max-gt-per-image":
        if gt.get("split") != "test":
            raise ValueError(f"{policy} is defined only for the test split; got {gt.get('split')!r}")
        value = max(1, busiest_count)
    elif policy == "fixed":
        value = int(defaults["max_detections"])
    else:
        raise ValueError(f"Unknown max_detections_policy: {policy!r}")
    return {
        "value": value,
        "policy": policy,
        "source_split": gt.get("split"),
        "source_image_id": busiest_id,
        "source_image_gt_boxes": busiest_count,
    }


def worker_chunk_size(model: str, n_images: int, defaults: dict) -> int:
    """Resolve a model-specific process lifetime in images."""
    if model.startswith("SAM "):
        return n_images
    if model.startswith("Cosmos Reason2"):
        return int(defaults.get("cosmos_worker_chunk_size", config.COSMOS_WORKER_CHUNK_SIZE))
    return int(defaults.get("vlm_worker_chunk_size", config.VLM_WORKER_CHUNK_SIZE))


def worker_chunk_ranges(model: str, n_images: int, defaults: dict) -> list[tuple[int, int]]:
    """SAM uses one process/prompt; Cosmos uses stricter bounded workers."""
    if n_images <= 0:
        return []
    chunk_size = worker_chunk_size(model, n_images, defaults)
    if chunk_size <= 0:
        raise ValueError("worker chunk size must be positive")
    return [(start, min(start + chunk_size, n_images))
            for start in range(0, n_images, chunk_size)]


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file() or not path.stat().st_size:
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _chunk_complete(path: Path, expected: dict) -> bool:
    status_path = path / "status.json"
    predictions_path = path / "predictions.csv"
    if not status_path.is_file() or not predictions_path.is_file():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    differences = {key: (status.get(key), value) for key, value in expected.items()
                   if status.get(key) != value}
    if differences:
        raise ValueError(f"Incompatible completed worker chunk {path}: {differences}")
    return status.get("status") == "completed"


def _run_isolated_prediction(
    *, args, gt: dict, dataset: str, model: str, prompt: dict,
    directory: Path, identity: dict, conf: float, max_detections: int,
    nms_iou: float, defaults: dict,
) -> tuple[list[dict], dict, list[dict]]:
    """Run one combination through fresh worker processes and merge its chunks."""
    chunks_root = directory / "worker_chunks"
    rows: list[dict] = []
    load_ms: list[float] = []
    worker_statuses: list[dict] = []
    ranges = worker_chunk_ranges(model, len(gt["records"]), defaults)
    for start, end in ranges:
        chunk_dir = chunks_root / f"{start:06d}-{end - 1:06d}"
        chunk_expected = {
            **identity,
            "status": "completed",
            "execution_strategy": EXECUTION_STRATEGY,
            "image_start": start,
            "image_end": end,
        }
        if not _chunk_complete(chunk_dir, chunk_expected):
            chunk_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, str(COMBINATION_WORKER),
                "--dataset", dataset,
                "--split", gt["split"],
                "--model", model,
                "--prompt", prompt["prompt"],
                "--prompt-id", prompt["id"],
                "--conf-threshold", str(conf),
                "--max-detections", str(max_detections),
                "--nms-iou-threshold", str(nms_iou),
                "--image-start", str(start),
                "--image-end", str(end),
                "--output-dir", str(chunk_dir),
            ]
            if args.max_images is not None or args.smoke_test:
                command.extend(["--image-limit", str(len(gt["records"]))])
            environment = dict(os.environ)
            environment.setdefault("TOKENIZERS_PARALLELISM", "false")
            environment.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            environment["PROMPTDETECT_COSMOS_MAX_SIDE"] = str(config.cosmos_max_side(model))
            log_path = chunk_dir / "worker.log"
            with log_path.open("w", encoding="utf-8") as log_stream:
                completed = subprocess.run(
                    command,
                    cwd=config.PROJECT_ROOT,
                    env=environment,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            worker_status_path = chunk_dir / "worker_status.json"
            worker_status = (json.loads(worker_status_path.read_text(encoding="utf-8"))
                             if worker_status_path.is_file() else {})
            if completed.returncode != 0 or worker_status.get("status") != "completed":
                tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
                raise RuntimeError(
                    f"Worker failed for {model}/{prompt['id']} images {start}:{end} "
                    f"(exit {completed.returncode}): {worker_status.get('error', '')}\n{tail}"
                )
            write_json(chunk_dir / "status.json", {
                **chunk_expected,
                "model_status": worker_status.get("model_status"),
                "finished_at": worker_status.get("finished_at"),
            })
        worker_status = json.loads((chunk_dir / "worker_status.json").read_text(encoding="utf-8"))
        model_status = worker_status.get("model_status") or {}
        if model_status.get("model_load_ms") is not None:
            load_ms.append(float(model_status["model_load_ms"]))
        worker_statuses.append(worker_status)
        rows.extend(_read_csv_rows(chunk_dir / "predictions.csv"))
    runtime = {
        "model_load_ms": round(sum(load_ms), 1),
        "worker_processes": len(ranges),
        "worker_chunk_size": (None if model.startswith("SAM ") else
                              worker_chunk_size(model, len(gt["records"]), defaults)),
        "execution_strategy": EXECUTION_STRATEGY,
    }
    if model.startswith("Cosmos Reason2"):
        runtime.update(
            cosmos_max_side=config.cosmos_max_side(model),
            cosmos_execution_revision=config.COSMOS_EXECUTION_REVISION,
        )
    return rows, runtime, worker_statuses


def _image_ids_from_index(run_dir: Path) -> list[str]:
    import csv
    path = run_dir / "ground_truth_index.csv"
    if not path.is_file() or not path.stat().st_size: return []
    with path.open(encoding="utf-8", newline="") as stream:
        return sorted({row["image_id"] for row in csv.DictReader(stream)})


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


def reports_only(run_dir: Path, wandb_mode: str = "online") -> int:
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    wandb_run = start_run(run_dir, cfg, wandb_mode)
    rows, predictions, per_image, overlaps = _collect(run_dir)
    try:
        write_csv(run_dir / "predictions.csv", predictions)
        summary = generate(run_dir, cfg, rows, per_image, overlaps)
        if has_sensitivity_prompts(cfg.get("prompt_definitions") or []):
            summary = generate_sensitivity(run_dir, cfg, rows, predictions, _image_ids_from_index(run_dir))
        log_and_finish(wandb_run, run_dir, summary)
    except Exception:
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        raise
    print(run_dir); return 0


def run_dataset(args, protocol: dict, dataset: str, model_aliases: list[str]) -> Path:
    defaults = protocol["defaults"]
    conf = args.conf_threshold if args.conf_threshold is not None else float(defaults["conf_threshold"])
    iou = args.iou_threshold if args.iou_threshold is not None else float(defaults["iou_threshold"])
    nms_iou = float(defaults.get("nms_iou_threshold", iou))
    visualization_count = (int(defaults.get("visualizations_per_combination", 0))
                           if args.save_visualizations is None else args.save_visualizations)
    max_images = 5 if args.smoke_test and args.max_images is None else args.max_images
    # Resolve the cap against the complete canonical test set even for a smoke
    # slice; otherwise the cap would depend on whichever five images came first.
    full_gt = load_ground_truth(dataset, args.split)
    detection_limit = derive_detection_limit(full_gt, defaults)
    max_detections = int(detection_limit["value"])
    gt = dict(full_gt)
    if max_images is not None:
        gt["records"] = full_gt["records"][:max_images]
    spec = protocol["datasets"][dataset]
    validate_vocabulary(dataset, spec, gt["class_names"])
    prompts = select_prompts(spec, args.prompt_ids, args.prompt_group, args.include_optional_prompts)
    enforcement = enforce_final_mtsd(gt, spec) if args.final and dataset == "MTSD" else {"final_mode": bool(args.final)}
    if gt.get("source_type") == "qa_fallback" and not (args.allow_qa_fallback or args.dry_run or args.smoke_test):
        raise RuntimeError("MTSD QA fallback requires --allow-qa-fallback for development runs")
    models = resolve_models(model_aliases, args.allow_heavy)
    manifest_hash = _manifest_hash(gt)
    sensitivity = has_sensitivity_prompts(prompts)
    dry_rows = _dry_stats(gt, prompts)
    # Prompts inside one sensitivity family share target_classes, so their GT
    # counts must be identical; a mismatch means a broken family definition.
    family_gt = validate_family_gt_counts(dry_rows) if sensitivity else []
    plan = {"dataset": dataset, "split": gt["split"], "source": gt["source"], "source_type": gt.get("source_type"),
            "images": len(gt["records"]), "models": models, "prompts": dry_rows,
            "predict_calls": len(gt["records"]) * len(models) * len(prompts),
            "max_detections": max_detections,
            "max_detections_policy": detection_limit,
            "nms_iou_threshold": nms_iou,
            "execution_strategy": EXECUTION_STRATEGY,
            "vlm_worker_chunk_size": int(defaults.get("vlm_worker_chunk_size", config.VLM_WORKER_CHUNK_SIZE)),
            "cosmos_worker_chunk_size": int(defaults.get("cosmos_worker_chunk_size", config.COSMOS_WORKER_CHUNK_SIZE)),
            "cosmos_max_side_by_model": {
                model: config.cosmos_max_side(model)
                for model in models if model.startswith("Cosmos Reason2")
            },
            "cosmos_execution_revision": config.COSMOS_EXECUTION_REVISION,
            "visualizations_per_combination": ("all" if visualization_count == -1
                                                else visualization_count)}
    if sensitivity:
        plan["sensitivity_families"] = family_gt
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return Path()
    if args.resume:
        run_dir = args.resume
        existing = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        for key, value in {"protocol_hash": protocol["protocol_hash"], "dataset_manifest_hash": manifest_hash,
                           "split": gt["split"], "conf_threshold": conf, "iou_threshold": iou,
                           "max_detections": max_detections,
                           "max_detections_policy": detection_limit["policy"],
                           "nms_iou_threshold": nms_iou,
                           "execution_strategy": EXECUTION_STRATEGY}.items():
            if existing.get(key) != value: raise ValueError(f"Resume mismatch for {key}: {existing.get(key)!r} != {value!r}")
    else:
        base_label = args.run_label or ("smoke" if args.smoke_test else None)
        if sensitivity:  # keep sensitivity run directories visibly separate
            base_label = f"{base_label}-sensitivity" if base_label else "sensitivity"
        run_dir = config.new_run_dir(dataset, base_label)
        snapshot = dict(protocol); snapshot.pop("protocol_path", None)
        (run_dir / "protocol_snapshot.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
        (run_dir / "protocol_hash").write_text(protocol["protocol_hash"] + "\n", encoding="utf-8")
    run_config = {"evaluation_protocol": "prompt-sensitivity-v2" if sensitivity else "targeted-v2",
                  "protocol_version": protocol["protocol_version"],
                  "protocol_hash": protocol["protocol_hash"], "dataset": dataset, "split": gt["split"],
                  "source": gt["source"], "source_type": gt.get("source_type"), "dataset_manifest_hash": manifest_hash,
                  "dataset_manifest": gt.get("manifest"), "models": models, "prompt_ids": [row["id"] for row in prompts],
                  "prompt_definitions": prompts,
                  "approved_group_scope": (gt.get("manifest") or {}).get("approved_groups"),
                  "group_scope": (gt.get("manifest") or {}).get("group_scope"),
                  "final_enforcement": enforcement,
                  "conf_threshold": conf, "iou_threshold": iou,
                  "max_detections": max_detections,
                  "max_detections_policy": detection_limit["policy"],
                  "max_detections_provenance": detection_limit,
                  "nms_iou_threshold": nms_iou,
                  "execution_strategy": EXECUTION_STRATEGY,
                  "vlm_worker_chunk_size": int(defaults.get("vlm_worker_chunk_size", config.VLM_WORKER_CHUNK_SIZE)),
                  "cosmos_worker_chunk_size": int(defaults.get("cosmos_worker_chunk_size", config.COSMOS_WORKER_CHUNK_SIZE)),
                  "cosmos_max_side_by_model": {
                      model: config.cosmos_max_side(model)
                      for model in models if model.startswith("Cosmos Reason2")
                  },
                  "cosmos_execution_revision": config.COSMOS_EXECUTION_REVISION,
                  "wandb_run_revision": config.WANDB_RUN_REVISION,
                  "visualizations_per_combination": ("all" if visualization_count == -1
                                                      else visualization_count),
                  "consistency_iou_threshold": args.consistency_iou,
                  "n_images": len(gt["records"]), "started_at": datetime.now(timezone.utc).isoformat()}
    write_json(run_dir / "run_config.json", run_config)
    write_csv(run_dir / "ground_truth_index.csv", ground_truth_index_rows(gt))
    wandb_run = start_run(run_dir, run_config, args.wandb_mode)
    run_config["wandb"] = {
        "target": tracking_target(),
        "mode": args.wandb_mode,
        "run_id": getattr(wandb_run, "id", None),
        "url": getattr(wandb_run, "url", None),
    }
    write_json(run_dir / "run_config.json", run_config)

    def prompt_identity(model: str, prompt: dict) -> dict:
        """Resume/status fingerprint; stale prompt wording or family membership invalidates a cache hit."""
        identity = {"protocol_hash": protocol["protocol_hash"], "dataset_manifest_hash": manifest_hash,
                    "model": model, "prompt_id": prompt["id"], "prompt": prompt["prompt"],
                    "target_classes": prompt["target_classes"], "conf_threshold": conf,
                    "iou_threshold": iou, "max_detections": max_detections,
                    "max_detections_policy": detection_limit["policy"],
                    "nms_iou_threshold": nms_iou, "split": gt["split"]}
        if "sensitivity_family" in prompt:
            identity["sensitivity_family"] = prompt["sensitivity_family"]
            identity["variant_type"] = prompt["variant_type"]
        if model.startswith("Cosmos Reason2"):
            identity.update(
                cosmos_max_side=config.cosmos_max_side(model),
                cosmos_worker_chunk_size=int(defaults.get(
                    "cosmos_worker_chunk_size", config.COSMOS_WORKER_CHUNK_SIZE
                )),
                cosmos_execution_revision=config.COSMOS_EXECUTION_REVISION,
            )
        return identity

    def update_combination_visuals(model: str, prompt: dict,
                                   prediction_rows: list[dict]) -> None:
        if not visualization_count:
            return
        from visualizations import save_visualizations

        limit = None if visualization_count == -1 else visualization_count
        save_visualizations(
            run_dir, gt, prediction_rows, [model], [prompt], iou,
            max_images=limit,
        )

    for model in models:
        for prompt in prompts:
            directory = combination_dir(run_dir, model, prompt["id"])
            identity = prompt_identity(model, prompt)
            if args.skip_completed and compatible_complete(directory, identity):
                update_combination_visuals(
                    model, prompt, _read_csv_rows(directory / "predictions.csv")
                )
                continue
            try:
                prediction_rows, runtime, worker_statuses = _run_isolated_prediction(
                    args=args, gt=gt, dataset=dataset, model=model, prompt=prompt,
                    directory=directory, identity=identity, conf=conf,
                    max_detections=max_detections, nms_iou=nms_iou, defaults=defaults,
                )
                boxes = prediction_boxes(prediction_rows, model, prompt["prompt"])
                result = evaluate_targeted(
                    boxes, gt["records"], prompt["target_classes"], iou, config.MAP_IOU_RANGE
                )
                write_csv(directory / "predictions.csv", prediction_rows)
                per_image = [{"model": model, "prompt_id": prompt["id"],
                              "prompt": prompt["prompt"], **row} for row in result["per_image"]]
                overlaps = [{"model": model, "prompt_id": prompt["id"],
                             "prompt": prompt["prompt"], **row}
                            for row in result["fp_nontarget_overlap"]]
                write_csv(directory / "per_image.csv", per_image)
                write_csv(directory / "fp_nontarget_overlap.csv", overlaps)
                metric = {
                    "dataset": dataset, "split": gt["split"], "model": model,
                    "prompt_id": prompt["id"], "prompt": prompt["prompt"],
                    "prompt_group": prompt["group"], "target_classes": prompt["target_classes"],
                    "prompt_class_confusion": result["prompt_class_confusion"],
                    "max_detections": max_detections,
                    "max_detections_policy": detection_limit["policy"],
                    "nms_iou_threshold": nms_iou,
                    **result["summary"], **runtime,
                }
                if "sensitivity_family" in prompt:
                    metric["sensitivity_family"] = prompt["sensitivity_family"]
                    metric["variant_type"] = prompt["variant_type"]
                infer_by_image = {
                    row["image_id"]: float(row["inference_ms"])
                    for row in prediction_rows
                    if row.get("inference_ms") not in (None, "")
                }
                infer = list(infer_by_image.values())
                metric["mean_inference_ms"] = round(sum(infer) / len(infer), 2) if infer else None
                metric["has_confidence"] = all(
                    str(row.get("has_confidence", True)).lower() == "true"
                    for row in prediction_rows
                )
                metric["ap_meaningful"] = metric["has_confidence"]
                write_json(directory / "metrics.json", metric)
                status = {
                    "status": "completed", **identity,
                    "target_gt_count": metric["target_gt_boxes"],
                    "tp": metric["tp"], "fp": metric["fp"], "fn": metric["fn"],
                    "finished_at": datetime.now(timezone.utc).isoformat(), **runtime,
                }
                write_json(directory / "status.json", status)
                update_combination_visuals(model, prompt, prediction_rows)
                run_config.setdefault("combination_execution", []).append({
                    "model": model, "prompt_id": prompt["id"], **runtime,
                    "chunk_load_statuses": [row.get("model_status") for row in worker_statuses],
                })
                write_json(run_dir / "run_config.json", run_config)
                log_combination(wandb_run, metric)
                # Chunk predictions are only an interruption checkpoint. The
                # canonical merged combination file above is authoritative.
                for chunk_prediction in (directory / "worker_chunks").glob("*/predictions.csv"):
                    chunk_prediction.unlink(missing_ok=True)
            except Exception as exc:
                write_json(directory / "status.json", {
                    "status": "failed", "error": str(exc), **identity,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "execution_strategy": EXECUTION_STRATEGY,
                })
                print(f"FAILED {model}/{prompt['id']}: {exc}", file=sys.stderr)
    metrics, predictions, per_image, overlaps = _collect(run_dir)
    expected_combinations = len(models) * len(prompts)
    if len(metrics) != expected_combinations:
        run_config["status"] = "incomplete"
        run_config["finished_at"] = datetime.now(timezone.utc).isoformat()
        run_config["completed_combinations"] = len(metrics)
        run_config["expected_combinations"] = expected_combinations
        write_json(run_dir / "run_config.json", run_config)
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        raise RuntimeError(
            f"Incomplete evaluation: {len(metrics)}/{expected_combinations} combinations completed. "
            f"Resume {run_dir} with --skip-completed after fixing the failed worker."
        )
    write_csv(run_dir / "predictions.csv", predictions)
    if visualization_count:
        visualization_rows = _read_csv_rows(
            run_dir / "visualizations" / "visualization_index.csv"
        )
        run_config["visualizations"] = {
            "enabled": True,
            "per_combination_limit": ("all" if visualization_count == -1 else visualization_count),
            "count": len(visualization_rows),
            "folder": "visualizations",
            "index": "visualizations/index.html",
        }
    else:
        run_config["visualizations"] = {"enabled": False, "count": 0}
    run_config["status"] = "completed"
    run_config["finished_at"] = datetime.now(timezone.utc).isoformat()
    try:
        summary = generate(run_dir, run_config, metrics, per_image, overlaps)
        if sensitivity:
            summary = generate_sensitivity(run_dir, run_config, metrics, predictions,
                                           [record["image_id"] for record in gt["records"]])
        write_json(run_dir / "run_config.json", run_config)
        log_and_finish(wandb_run, run_dir, summary)
    except Exception:
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        raise
    print(run_dir); return run_dir


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.save_visualizations is not None and args.save_visualizations < -1:
        raise ValueError("--save-visualizations must be a positive count, 0, or omitted after the flag")
    if args.final and args.allow_qa_fallback:
        raise ValueError("--allow-qa-fallback is incompatible with --final")
    if args.final and not args.dry_run and args.wandb_mode != "online":
        raise ValueError("Final PromptDetect evaluations require --wandb-mode online so results "
                         "are uploaded to the configured PromptDetect W&B project")
    if args.reports_only: return reports_only(args.reports_only, args.wandb_mode)
    datasets = ["MDWD", "MTSD"] if args.dataset.lower() == "both" else [args.dataset.upper()]
    if args.resume and len(datasets) != 1: raise ValueError("--resume requires one dataset")
    # The standard both-dataset evaluation runs the classic dissertation protocol
    # AND the prompt-sensitivity protocol as separate stages, each with its own
    # run directories, outputs and protocol label. An explicit --protocol selects
    # a single protocol instead.
    dual_stage = args.dataset.lower() == "both" and args.protocol is None
    if dual_stage and (args.prompt_ids or args.prompt_group or args.resume):
        raise ValueError("--prompt-ids/--prompt-group/--resume are ambiguous across the dissertation "
                         "and prompt-sensitivity stages; pass --protocol to select one protocol")
    protocol_paths = [DEFAULT_PROTOCOL, SENSITIVITY_PROTOCOL] if dual_stage else [args.protocol or DEFAULT_PROTOCOL]
    for path in protocol_paths:
        protocol = load_protocol(path)
        if args.prompt_group:
            available = sorted({row["group"] for name in datasets
                                for row in protocol["datasets"][name]["prompts"]})
            if args.prompt_group not in available:
                raise ValueError(f"Unknown prompt group {args.prompt_group!r}; protocol "
                                 f"{protocol['protocol_version']} defines: {available}")
        aliases = args.models or list(protocol["models"]["primary"])
        if dual_stage:
            print(f"=== Stage: {protocol['protocol_version']} ({path.name}) ===")
        for dataset in datasets: run_dataset(args, protocol, dataset, aliases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
