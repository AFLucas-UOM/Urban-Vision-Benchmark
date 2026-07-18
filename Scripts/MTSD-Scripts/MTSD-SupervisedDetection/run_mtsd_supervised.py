#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))

from mtsd_detection.config import load_config
from mtsd_detection.dataset_validation import validate_prepared, validate_variant
from mtsd_detection.evaluate import export_rfdetr_predictions, export_yolo_predictions, safe_unified_evaluation
from mtsd_detection.manifests import sha256_file
from mtsd_detection.model_registry import DEFAULT_ORDER, resolve_checkpoint_info, validate_model_keys
from mtsd_detection.prepare_dataset import prepare, preparation_plan
from mtsd_detection.reporting import generate as generate_reports
from mtsd_detection.qa_gate import enforce_qa_gate, load_qa_gate
from mtsd_detection.state import atomic_write, load_compatible
from mtsd_detection.utils import fingerprint, git_commit, hardware_info, safe_name, set_global_seed
from mtsd_detection.wandb_utils import finish_run, log_dataset_artifact, start_run


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Canonical MTSD supervised detection pipeline")
    result.add_argument("--config", type=Path, default=HERE / "config" / "default.yaml")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--prepare-only", action="store_true")
    result.add_argument("--dataset-variant", choices=("augmented", "unaugmented", "both"), default="augmented")
    result.add_argument("--validate-prepared", action="store_true"); result.add_argument("--strict", action="store_true")
    result.add_argument("--smoke-test", action="store_true")
    result.add_argument("--matrix", choices=("dissertation", "aug_ablation")); result.add_argument("--models", nargs="+")
    result.add_argument("--resume", nargs="?", const="auto"); result.add_argument("--skip-completed", action="store_true")
    result.add_argument("--skip-failed", action="store_true"); result.add_argument("--run-label")
    result.add_argument("--reports-only", action="store_true")
    result.add_argument("--wandb-mode", choices=("online", "offline", "disabled")); result.add_argument("--wandb-group")
    result.add_argument("--device", choices=("auto", "cuda", "cpu"))
    result.add_argument("--allow-raw-xml-fallback", action="store_true")
    result.add_argument("--confirm-non-qa-data", action="store_true")
    result.add_argument("--final", action="store_true", help="Enforce locked scope, resolved QA gate, canonical variant, and strict validation")
    result.add_argument("--acknowledge-open-qa-gate", action="store_true", help="Development-only override; incompatible with --final")
    result.add_argument("--group-scope", choices=("auto", "explicit"), help="Override annotation scope for development/dry-run")
    result.add_argument("--approved-groups", nargs="+", help="Override explicit approved group list")
    result.add_argument("--require-unified-eval", action="store_true", help="Fail a model run if unified COCOeval fails")
    result.add_argument("--rebuild", action="store_true", help="Archive an existing prepared variant before rebuilding")
    return result


def _dataset(config: dict, variant: str) -> tuple[Path, dict, str]:
    if variant == "both": raise ValueError("Training requires one --dataset-variant, not both")
    title = "MTSD-Augmented" if variant == "augmented" else "MTSD-Unaugmented"
    root = Path(config["dataset"]["prepared_root"]) / title
    manifest_path = root / "prep_manifest.json"
    if not manifest_path.is_file(): raise FileNotFoundError(f"Prepared dataset missing: {manifest_path}")
    return root, json.loads(manifest_path.read_text(encoding="utf-8")), sha256_file(manifest_path)


def _resolve_resume(config: dict, value: str) -> Path:
    if value != "auto": return Path(value)
    states = sorted(Path(config["outputs"]["state_root"]).glob("*/state.json"), key=lambda p: p.stat().st_mtime)
    if not states: raise FileNotFoundError("No state.json found to resume")
    return states[-1]


def run_training(args, config: dict) -> int:
    from mtsd_detection import train_rfdetr, train_yolo
    keys = args.models or (config["matrix"][args.matrix] if args.matrix else (["yolo12n", "rfdetr-n"] if args.smoke_test else None))
    if not keys: raise ValueError("Training requires --matrix or --models (or --smoke-test)")
    specs = validate_model_keys(list(keys))
    gate = load_qa_gate(Path(config["annotations"]["qa_gate_file"]), Path(config["repo_root"]))
    enforce_qa_gate(gate, final=args.final, acknowledged_override=args.acknowledge_open_qa_gate)
    if args.final and args.dataset_variant != "augmented":
        raise PermissionError("Final dissertation training requires the augmented dataset variant")
    dataset_root, manifest, prep_hash = _dataset(config, args.dataset_variant)
    validation = validate_variant(dataset_root, strict=True, policy=config["validation"])
    if not validation["ok"]:
        raise RuntimeError(f"Prepared dataset strict validation failed: {validation}")
    if args.final:
        if config["annotations"].get("group_scope") != "explicit":
            raise PermissionError("Final training refuses auto-discovered annotation scope")
        if manifest.get("group_scope") != "explicit" or manifest.get("approved_groups") != config["annotations"].get("approved_groups"):
            raise PermissionError("Prepared manifest does not match the locked approved group scope")
        if manifest.get("annotation_source_mode") != "qa_only":
            raise PermissionError("Final dissertation training requires QA-only annotations")
    expected = {"requested_matrix": list(keys), "dataset_version": manifest["dataset_version"],
                "prep_manifest_sha256": prep_hash, "split_manifest_sha256": manifest["split_manifest_sha256"],
                "annotation_source_mode": manifest["annotation_source_mode"], "dataset_variant": args.dataset_variant,
                "config_fingerprint": config["config_fingerprint"]}
    if args.resume:
        state_path = _resolve_resume(config, args.resume); state = load_compatible(state_path, expected)
    else:
        label = safe_name(args.run_label or ("smoke" if args.smoke_test else datetime.now().strftime("%Y%m%d-%H%M%S")))
        state_path = Path(config["outputs"]["state_root"]) / label / "state.json"
        state = {**expected, "run_label": label, "started_at": datetime.now(timezone.utc).isoformat(),
                 "models": {key: {"status": "pending"} for key in keys}, "current_model": None, "final_status": "running"}
        if state_path.exists(): raise FileExistsError(f"State already exists: {state_path}")
        atomic_write(state_path, state)
    training = dict(config["training"])
    if args.device: training["device"] = args.device
    for index, spec in enumerate(specs, start=1):
        previous = state["models"][spec.key].get("status")
        # Completed immutable runs are never re-opened on resume; reruns require
        # a new run label/state and therefore cannot duplicate their W&B run.
        if previous == "completed": continue
        if previous == "failed" and args.skip_failed: continue
        slug = "mtsdqa1aug" if args.dataset_variant == "augmented" else "mtsdqa1noaug"
        image_size = (train_rfdetr.rfdetr_resolution(spec, training)
                      if spec.trainer == "rfdetr"
                      else (320 if args.smoke_test else int(training["image_size"])))
        name = safe_name(f"E{index:02d}_{spec.key}_{slug}_img{image_size}_eb32_e{2 if args.smoke_test else training['epochs']}_adamw_s42")
        run_dir = Path(config["outputs"]["runs_root"]) / f"{spec.family}-MTSD" / name
        state["current_model"] = spec.key; state["models"][spec.key] = {"status": "running", "run_dir": str(run_dir), "started_at": datetime.now(timezone.utc).isoformat()}; atomic_write(state_path, state)
        checkpoint_info = resolve_checkpoint_info(spec, Path(config["repo_root"]), require_local=True)
        checkpoint = Path(checkpoint_info["path"])
        online_policy = ("disabled-ultralytics-online-augmentation" if spec.trainer == "ultralytics"
                         else "rfdetr-1.3.0-internal-augmentation-uncontrolled")
        controlled = spec.trainer == "ultralytics"
        ablation_validity = ("clean-offline-augmentation-effect" if controlled
                             else "descriptive-only-uncontrolled-online-augmentation")
        record = {"model": spec.key, "family": spec.family, "scale": spec.scale, "trainer": spec.trainer,
                  "dataset_version": manifest["dataset_version"], "dataset_variant": args.dataset_variant,
                  "annotation_source_mode": manifest["annotation_source_mode"], "included_groups": [g["group"] for g in manifest["groups"] if g["status"] == "included"],
                  "prep_manifest_sha256": prep_hash, "split_manifest_sha256": manifest["split_manifest_sha256"],
                  "group_scope": manifest.get("group_scope"), "approved_groups": manifest.get("approved_groups"),
                  "qa_gate": {**gate, "override_used": args.acknowledge_open_qa_gate},
                  "offline_augmentation_variant": args.dataset_variant,
                  "online_augmentation_policy": online_policy, "online_augmentation_controlled": controlled,
                  "ablation_validity": ablation_validity,
                  "git_commit": git_commit(Path(config["repo_root"])), "hardware": hardware_info(), "run_dir": str(run_dir),
                  "checkpoint_path": checkpoint_info["path"], "checkpoint_source": checkpoint_info["source"],
                  "checkpoint_sha256": checkpoint_info["sha256"], "status": "running"}
        group = args.wandb_group or ("smoke" if args.smoke_test else config["wandb"]["group"])
        wandb_config = {**config, "run_metadata": record}
        wandb_run = start_run(wandb_config, name, group, args.wandb_mode or config["wandb"]["mode"],
                              [spec.family, spec.scale, manifest["dataset_version"], args.dataset_variant])
        if wandb_run is not None:
            record.update(wandb_run_id=getattr(wandb_run, "id", None),
                          wandb_url=getattr(wandb_run, "url", None),
                          wandb_project=config["wandb"]["project"], wandb_group=group)
            state["models"][spec.key]["wandb_run_id"] = record["wandb_run_id"]
            atomic_write(state_path, state)
            if config["wandb"].get("log_dataset_artifact", True):
                log_dataset_artifact(wandb_run, dataset_root, record)
        try:
            if not checkpoint.is_file(): raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
            set_global_seed(int(training["seed"]), bool(training["deterministic"]))
            if spec.trainer == "ultralytics":
                outcome = train_yolo.train(
                    spec, checkpoint, dataset_root / "MTSD-YOLO" / "data.yaml", run_dir,
                    training, args.smoke_test, wandb_run,
                    int(config["wandb"].get("log_interval_steps", 100)),
                )
            else:
                outcome = train_rfdetr.train(
                    spec, checkpoint, dataset_root / "MTSD-COCO", run_dir,
                    training, args.smoke_test, wandb_run,
                    int(config["wandb"].get("log_interval_steps", 100)),
                )
            predictions = run_dir / "unified_test_predictions.json"
            def unified_call():
                if spec.trainer == "ultralytics":
                    return export_yolo_predictions(Path(outcome["checkpoint_best"]), dataset_root / "MTSD-COCO", "test", predictions,
                                                   320 if args.smoke_test else int(training["image_size"]), training.get("device", "auto"))
                return export_rfdetr_predictions(spec, Path(outcome["checkpoint_best"]), dataset_root / "MTSD-COCO", "test", predictions,
                                                  train_rfdetr.rfdetr_resolution(spec, training), training.get("device", "auto"))
            unified_started = datetime.now(timezone.utc)
            unified_result = safe_unified_evaluation(unified_call, require=args.require_unified_eval)
            unified_result["unified_evaluation_seconds"] = (
                datetime.now(timezone.utc) - unified_started
            ).total_seconds()
            record.update(outcome, **unified_result, status="completed", finished_at=datetime.now(timezone.utc).isoformat())
            (run_dir / "run_record.json").write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
            state["models"][spec.key].update(status="completed", finished_at=record["finished_at"])
            finish_run(wandb_run, record)
        except Exception as exc:
            run_dir.mkdir(parents=True, exist_ok=True)
            record.update(status="failed", error=str(exc), traceback=traceback.format_exc(), finished_at=datetime.now(timezone.utc).isoformat())
            (run_dir / "run_record.json").write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
            state["models"][spec.key].update(status="failed", error=str(exc), finished_at=record["finished_at"])
            finish_run(wandb_run, record, 1)
            print(f"FAILED {spec.key}: {exc}", file=sys.stderr)
        atomic_write(state_path, state); gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except ImportError: pass
    state["current_model"] = None
    state["final_status"] = "completed" if all(v["status"] in {"completed", "skipped"} for v in state["models"].values()) else "completed_with_failures"
    state["finished_at"] = datetime.now(timezone.utc).isoformat(); atomic_write(state_path, state)
    print(json.dumps({"state": str(state_path), "status": state["final_status"]}, indent=2))
    return 0 if state["final_status"] == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except (ValueError, PermissionError, FileNotFoundError) as exc:
        # Policy violations (unexpected groups, unresolved QA gate, missing
        # sources) are expected states, not crashes: report them clearly.
        print(f"REFUSED: {exc}", file=sys.stderr)
        if "outside approved scope" in str(exc):
            print("\nNew Final-QA groups exist that are not yet approved. The required\n"
                  "sequence is: (1) run the annotation audit (scan_annotations.py),\n"
                  "(2) resolve its findings via review_app.py + apply_fixes.py,\n"
                  "(3) refresh the gate/scope (update_qa_gate.py --apply), then\n"
                  "(4) re-run this command. Nothing was prepared or modified.",
                  file=sys.stderr)
        return 2


def _main(argv: list[str] | None = None) -> int:
    arg_parser = parser(); argv = sys.argv[1:] if argv is None else argv
    if not argv: arg_parser.print_help(); return 0
    args = arg_parser.parse_args(argv); config = load_config(args.config)
    if args.final and args.acknowledge_open_qa_gate:
        raise ValueError("--final is incompatible with --acknowledge-open-qa-gate")
    if args.final and (args.allow_raw_xml_fallback or args.confirm_non_qa_data):
        raise ValueError("--final is incompatible with raw-XML fallback flags")
    if args.group_scope: config["annotations"]["group_scope"] = args.group_scope
    if args.approved_groups: config["annotations"]["approved_groups"] = args.approved_groups
    config["config_fingerprint"] = fingerprint({k: v for k, v in config.items() if k != "config_fingerprint"})
    if args.final and config["annotations"].get("group_scope") != "explicit":
        raise PermissionError("--final refuses annotations.group_scope=auto")
    if args.dry_run:
        plan = preparation_plan(config, args.allow_raw_xml_fallback, args.confirm_non_qa_data)
        plan["model_order"] = DEFAULT_ORDER; plan["selected_models"] = args.models or (config["matrix"].get(args.matrix) if args.matrix else None)
        print(json.dumps(plan, indent=2, default=str)); return 0
    if args.validate_prepared:
        result = validate_prepared(Path(config["dataset"]["prepared_root"]), args.strict, config["validation"])
        print(json.dumps(result, indent=2)); return 0 if result["ok"] else 1
    if args.prepare_only:
        if args.final and args.dataset_variant != "both":
            raise ValueError("Final preparation must build --dataset-variant both from one authoritative split")
        if args.allow_raw_xml_fallback:
            print("\nWARNING: RAW XML FALLBACK ACTIVE\nThis creates a mixed, non-final-QA dataset version.\n")
        result = prepare(config, args.dataset_variant, args.allow_raw_xml_fallback, args.confirm_non_qa_data,
                         args.rebuild, final=args.final,
                         acknowledge_open_qa_gate=args.acknowledge_open_qa_gate)
        print(json.dumps(result, indent=2, default=str)); return 0
    if args.reports_only:
        output = generate_reports(Path(config["outputs"]["results_root"]), Path(config["outputs"]["runs_root"]))
        print(output); return 0
    return run_training(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
