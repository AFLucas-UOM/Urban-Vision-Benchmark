"""Evaluate held-out MTSD test splits for completed RF-DETR runs missing them.

This is deliberately evaluation-only: it restores ``checkpoint_best`` and
uses RF-DETR's own COCO evaluator on the original run's test split.  It never
calls the optimiser or alters checkpoint weights.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, SequentialSampler

from mtsd_detection.model_registry import registry, _rfdetr_class


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNS = REPO / "Results" / "MTSD-Runs" / "RF-DETR-MTSD"


def find_targets() -> list[tuple[Path, dict]]:
    targets = []
    for record_path in RUNS.glob("*/run_record.json"):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            continue
        native = record.get("native_metrics", {}) or {}
        test = native.get("test", {}) if isinstance(native, dict) else {}
        rows = test.get("class_map", {}).get("test", []) if isinstance(test, dict) else []
        if any(isinstance(row, dict) and row.get("class") == "all" for row in rows):
            continue
        checkpoint = Path(record.get("checkpoint_best", ""))
        args = record.get("train_args", {}) or {}
        dataset_dir = Path(args.get("dataset_dir", ""))
        if not checkpoint.is_file() or not dataset_dir.is_dir():
            raise FileNotFoundError(f"Missing checkpoint or dataset for {record_path.parent}")
        targets.append((record_path, record))
    return targets


def evaluate(record_path: Path, record: dict) -> None:
    # Import package internals only in the CUDA training environment.
    from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
    from rfdetr.engine import evaluate as coco_evaluate
    from rfdetr.models import build_criterion_and_postprocessors
    from rfdetr import util

    args_record = record["train_args"]
    model_key = str(record.get("model", ""))
    spec = registry()[model_key]
    resolution = int((record.get("model_args", {}) or {}).get("resolution", 0) or args_record.get("resolution"))
    batch_size = int(args_record.get("batch_size", 1))
    dataset_dir = Path(args_record["dataset_dir"])
    checkpoint = Path(record["checkpoint_best"])
    categories = json.loads((dataset_dir / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))["categories"]
    categories = sorted(categories, key=lambda item: item["id"])
    class_names = [item["name"] for item in categories]

    # This exactly follows the RF-DETR trainer's evaluation setup, except that
    # it constructs the held-out ``test`` loader directly.
    model = _rfdetr_class(spec)(
        pretrain_weights=str(record.get("checkpoint_path")), resolution=resolution,
        gradient_checkpointing=True, device="cuda",
    )
    # Reuse the instantiated architecture's complete argument namespace.  This
    # retains architecture-specific fields such as patch size and positional
    # encoding, then supplies only evaluation-specific fields.
    args = model.model.args
    args.dataset_file = "roboflow"
    args.dataset_dir = str(dataset_dir)
    args.resolution = resolution
    args.batch_size = batch_size
    args.device = "cuda"
    args.num_classes = len(categories)
    args.class_names = class_names
    args.num_workers = 2
    args.square_resize_div_64 = True
    args.multi_scale = True
    args.expanded_scales = True
    args.do_random_resize_via_padding = False
    args.output_dir = str(record_path.parent / "native_test_evaluation")
    args.segmentation_head = False
    model.model.reinitialize_detection_head(len(categories))
    model.model.args.num_classes = len(categories)
    model.model.args.class_names = class_names
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    model.model.model.load_state_dict(state, strict=True)
    model.model.model.to("cuda").eval()
    criterion, postprocess = build_criterion_and_postprocessors(args)
    criterion.to("cuda").eval()
    dataset_test = build_dataset(image_set="test", args=args, resolution=resolution)
    loader = DataLoader(dataset_test, batch_size, sampler=SequentialSampler(dataset_test),
                        drop_last=False, collate_fn=util.misc.collate_fn, num_workers=2)
    started = time.perf_counter()
    stats, _ = coco_evaluate(model.model.model, criterion, postprocess, loader,
                             get_coco_api_from_dataset(dataset_test), torch.device("cuda"), args=args)
    elapsed = time.perf_counter() - started
    payload = stats["results_json"]
    aggregate = next(row for row in payload["class_map"] if row.get("class") == "all")
    native = record.setdefault("native_metrics", {})
    existing = native.get("test", {}) if isinstance(native.get("test"), dict) else {}
    native["test"] = {
        **existing,
        "class_map": {**(existing.get("class_map", {}) if isinstance(existing, dict) else {}), "test": payload["class_map"]},
        "map50": aggregate["map@50"], "map50_95": aggregate["map@50:95"],
        "precision": aggregate["precision"], "recall": aggregate["recall"],
    }
    record["native_test_evaluation"] = {
        "checkpoint": str(checkpoint), "dataset_dir": str(dataset_dir), "split": "test",
        "resolution": resolution, "batch_size": batch_size, "seconds": elapsed,
    }
    record["test_evaluation_deferred"] = False
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"DONE {record_path.parent.name}: mAP50-95={aggregate['map@50:95']:.6f}", flush=True)


def main() -> None:
    targets = find_targets()
    print(f"RF-DETR native-test evaluations required: {len(targets)}", flush=True)
    for record_path, record in targets:
        evaluate(record_path, record)


if __name__ == "__main__":
    main()
