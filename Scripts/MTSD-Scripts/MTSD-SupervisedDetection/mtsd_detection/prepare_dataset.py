from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from . import __version__
from .annotation_sources import CLASS_NAMES, discover_sources
from .augmentation import augment_image
from .dataset_validation import validate_prepared
from .manifests import sha256_file, write_csv, write_manifest
from .qa_gate import enforce_qa_gate, load_qa_gate
from .splitting import assign_splits
from .utils import git_commit

SPLITS = ("train", "valid", "test")


def _place(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def validate_records(raw: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    records, excluded = [], []
    findings: Counter[str] = Counter()
    for source in raw:
        path = Path(source["source_path"])
        if not path.is_file():
            findings["missing_image_file"] += 1
            excluded.append({"group": source["group"], "file_name": source["file_name"],
                             "reason": "missing_image_file", "expected_path": str(path)})
            continue
        try:
            with Image.open(path) as image:
                actual_width, actual_height = image.size
        except Exception as exc:
            findings["corrupt_image"] += 1
            excluded.append({"group": source["group"], "file_name": source["file_name"],
                             "reason": f"corrupt_image: {exc}"})
            continue
        # QA bboxes live in the annotation metadata coordinate space. Some JPEGs
        # carry EXIF orientation, so PIL's untransposed raw dimensions are swapped;
        # replacing QA dimensions here would corrupt otherwise valid boxes.
        width, height = int(source.get("width") or actual_width), int(source.get("height") or actual_height)
        if (width, height) != (actual_width, actual_height):
            findings["metadata_dimension_mismatch_exif"] += 1
        clean = []
        annotation_ids = []
        for annotation in source["raw_boxes"]:
            try:
                category = source["qa_id_map"][int(annotation["category_id"])]
                x, y, bw, bh = [float(v) for v in annotation["bbox"]]
            except Exception:
                findings["malformed_box_or_unknown_class"] += 1; continue
            x0, y0, x1, y1 = max(0.0, x), max(0.0, y), min(width, x + bw), min(height, y + bh)
            if (x0, y0, x1, y1) != (x, y, x + bw, y + bh): findings["box_clipped_to_image"] += 1
            if x1 - x0 < 1.0 or y1 - y0 < 1.0:
                findings["degenerate_box_after_clip"] += 1; continue
            clean.append({"class_id": category, "x0": x0, "y0": y0, "x1": x1, "y1": y1})
            annotation_ids.append(annotation.get("id"))
        if not clean: findings["image_without_valid_boxes"] += 1
        out_name = f"{source['group'].lower().replace('-', '')}_{Path(source['file_name']).name}"
        records.append({**source, "source_path": path, "width": width, "height": height,
                        "clean_boxes": clean, "source_annotation_ids": annotation_ids,
                        "out_name": out_name, "source_image_sha256": sha256_file(path)})
    return records, dict(findings), excluded


def _split_rows(records: list[dict[str, Any]], group_scope: str,
                approved_groups: list[str]) -> list[dict[str, Any]]:
    rows = []
    for record in sorted(records, key=lambda row: row["out_name"]):
        counts = Counter(CLASS_NAMES[box["class_id"]] for box in record["clean_boxes"])
        rows.append({
            "group": record["group"], "source_image_path": str(record["source_path"]),
            "source_filename": record["file_name"], "source_image_sha256": record["source_image_sha256"],
            "annotation_source_type": record["annotation_source_type"],
            "annotation_source_path": str(record["annotation_source_path"]),
            "annotation_source_sha256": record["annotation_source_sha256"],
            "source_annotation_ids": json.dumps(record["source_annotation_ids"]), "split": record["split"],
            "out_name": record["out_name"], "width": record["width"], "height": record["height"],
            "n_boxes": len(record["clean_boxes"]), "boxes_per_class": json.dumps(counts, sort_keys=True),
            "group_scope": group_scope, "approved_groups": json.dumps(approved_groups),
        })
    return rows


def _write_yolo(root: Path, records: list[dict[str, Any]], dataset_version: str, groups: list[str]) -> None:
    for record in records:
        image_path = root / record["split"] / "images" / record["out_name"]
        _place(record["source_path"], image_path)
        lines = []
        for box in record["clean_boxes"]:
            cx = (box["x0"] + box["x1"]) / 2 / record["width"]
            cy = (box["y0"] + box["y1"]) / 2 / record["height"]
            bw = (box["x1"] - box["x0"]) / record["width"]
            bh = (box["y1"] - box["y0"]) / record["height"]
            lines.append(f"{box['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        label_path = root / record["split"] / "labels" / f"{Path(record['out_name']).stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    payload = {"train": "../train/images", "val": "../valid/images", "test": "../test/images",
               "nc": len(CLASS_NAMES), "names": CLASS_NAMES,
               "mtsd": {"version": dataset_version, "source": f"MTSD ({', '.join(groups)})", "seed": 42}}
    (root / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _coco_payload(records: list[dict[str, Any]], split: str, version: str) -> dict[str, Any]:
    images, annotations = [], []
    annotation_id = 1
    for image_id, record in enumerate([r for r in records if r["split"] == split], start=1):
        images.append({"id": image_id, "file_name": record["out_name"], "width": record["width"],
                       "height": record["height"], "mtsd_group": record["group"]})
        for box in record["clean_boxes"]:
            width, height = box["x1"] - box["x0"], box["y1"] - box["y0"]
            annotations.append({"id": annotation_id, "image_id": image_id, "category_id": box["class_id"],
                                "bbox": [round(box["x0"], 4), round(box["y0"], 4), round(width, 4), round(height, 4)],
                                "area": round(width * height, 4), "iscrowd": 0, "segmentation": []})
            annotation_id += 1
    return {"info": {"description": f"MTSD {version} {split}"}, "licenses": [],
            "categories": [{"id": i, "name": name, "supercategory": "mtsd"} for i, name in enumerate(CLASS_NAMES)],
            "images": images, "annotations": annotations}


def _write_coco(root: Path, records: list[dict[str, Any]], version: str) -> None:
    for split in SPLITS:
        split_records = [row for row in records if row["split"] == split]
        for record in split_records: _place(record["source_path"], root / split / record["out_name"])
        path = root / split / "_annotations.coco.json"; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_coco_payload(records, split, version), indent=2) + "\n", encoding="utf-8")


def _augment(variant: Path, records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    yolo, coco = variant / "MTSD-YOLO", variant / "MTSD-COCO"
    coco_path = coco / "train" / "_annotations.coco.json"
    payload = json.loads(coco_path.read_text(encoding="utf-8"))
    next_image_id = max((row["id"] for row in payload["images"]), default=0) + 1
    next_annotation_id = max((row["id"] for row in payload["annotations"]), default=0) + 1
    for record in [row for row in records if row["split"] == "train"]:
        for copy_index in range(1, int(config["copies_per_image"]) + 1):
            stem, suffix = Path(record["out_name"]).stem, Path(record["out_name"]).suffix
            generated = f"{stem}_aug{copy_index}{suffix}"
            info = augment_image(record["source_path"], yolo / "train" / "images" / generated,
                                 record["out_name"], copy_index, int(config["seed"]), config)
            source_label = yolo / "train" / "labels" / f"{stem}.txt"
            target_label = yolo / "train" / "labels" / f"{Path(generated).stem}.txt"
            shutil.copy2(source_label, target_label)
            shutil.copy2(yolo / "train" / "images" / generated, coco / "train" / generated)
            payload["images"].append({"id": next_image_id, "file_name": generated,
                                      "width": record["width"], "height": record["height"],
                                      "mtsd_group": record["group"], "augmented_from": record["out_name"]})
            for box in record["clean_boxes"]:
                width, height = box["x1"] - box["x0"], box["y1"] - box["y0"]
                payload["annotations"].append({"id": next_annotation_id, "image_id": next_image_id,
                    "category_id": box["class_id"], "bbox": [box["x0"], box["y0"], width, height],
                    "area": width * height, "iscrowd": 0, "segmentation": []})
                next_annotation_id += 1
            next_image_id += 1
            rows.append({"source_out_name": record["out_name"], "source_image_hash": record["source_image_sha256"],
                         "split": "train", **info, "label_source": str(source_label)})
    coco_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return rows


def prepare(config: dict[str, Any], variants: str = "both", allow_raw_xml: bool = False,
            confirm_non_qa: bool = False, rebuild: bool = False, final: bool = False,
            acknowledge_open_qa_gate: bool = False) -> dict[str, Any]:
    root, annotations = Path(config["repo_root"]), Path(config["dataset"]["annotations_root"])
    gate = load_qa_gate(Path(config["annotations"]["qa_gate_file"]), root)
    enforce_qa_gate(gate, final=final, acknowledged_override=acknowledge_open_qa_gate)
    if final and config["annotations"].get("group_scope") != "explicit":
        raise PermissionError("Final preparation requires annotations.group_scope=explicit")
    annotation_cfg = config["annotations"]
    raw, groups, source_mode = discover_sources(
        annotations, root, allow_raw_xml, confirm_non_qa,
        group_scope=annotation_cfg.get("group_scope", "auto"),
        approved_groups=annotation_cfg.get("approved_groups", []),
        unexpected_group_policy=annotation_cfg.get("unexpected_group_policy", "warn"),
    )
    records, findings, excluded = validate_records(raw)
    assignment = assign_splits(records, config["split"]["ratios"], int(config["split"]["seed"]))
    for record in records: record["split"] = assignment[record["out_name"]]
    selected = ["augmented", "unaugmented"] if variants == "both" else [variants]
    prepared_root = Path(config["dataset"]["prepared_root"])
    split_rows = _split_rows(records, annotation_cfg.get("group_scope", "auto"),
                             annotation_cfg.get("approved_groups", []))
    outputs: dict[str, Any] = {}
    for variant_name in selected:
        title = "MTSD-Augmented" if variant_name == "augmented" else "MTSD-Unaugmented"
        target = prepared_root / title
        if target.exists():
            if not rebuild: raise FileExistsError(f"{target} exists; use --rebuild to archive and replace it")
            archive = prepared_root / "_archive" / f"{title}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            archive.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(target), str(archive))
        base = "mtsd-mixed-v1" if source_mode == "mixed" else str(config["dataset"]["version_base"])
        version = f"{base}-{'aug' if variant_name == 'augmented' else 'noaug'}"
        included = [row["group"] for row in groups if row["status"] == "included"]
        _write_yolo(target / "MTSD-YOLO", records, version, included)
        _write_coco(target / "MTSD-COCO", records, version)
        split_path = target / "split_manifest.csv"; write_csv(split_path, split_rows)
        aug_rows = _augment(target, records, config["augmentation"]) if variant_name == "augmented" else []
        aug_path = target / "augmentation_manifest.csv"
        write_csv(aug_path, aug_rows, list(aug_rows[0]) if aug_rows else ["source_out_name", "source_image_hash", "split", "copy_index", "ops_applied", "seed_material", "generated_filename", "generated_image_hash", "label_source"])
        per_split = {split: {"images": sum(r["split"] == split for r in records),
                             "boxes": sum(len(r["clean_boxes"]) for r in records if r["split"] == split)} for split in SPLITS}
        per_class = {split: {name: sum(1 for r in records if r["split"] == split for b in r["clean_boxes"] if b["class_id"] == i)
                             for i, name in enumerate(CLASS_NAMES)} for split in SPLITS}
        manifest = {
            "dataset_version": version, "prepared_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(root), "pipeline_version": __version__, "annotation_source_mode": source_mode,
            "group_scope": annotation_cfg.get("group_scope", "auto"),
            "approved_groups": annotation_cfg.get("approved_groups", []),
            "groups": groups, "class_names": CLASS_NAMES, "qa_id_to_contiguous": {str(i + 1): i for i in range(12)},
            "qa_gate": {**gate, "override_used": acknowledge_open_qa_gate},
            "split": {"algorithm_version": "v1", "ratios": config["split"]["ratios"], "seed": config["split"]["seed"], "mode": "random-per-group"},
            "counts": {"original_images": len(records), "original_boxes": sum(len(r["clean_boxes"]) for r in records),
                       "augmented_images": len(aug_rows), "final_images": len(records) + len(aug_rows),
                       "final_boxes": sum(len(r["clean_boxes"]) for r in records) + sum(len(r["clean_boxes"]) for r in records if r["split"] == "train") * (int(config["augmentation"]["copies_per_image"]) if aug_rows else 0),
                       "per_split": per_split, "per_class_per_split": per_class},
            "augmentation": {"enabled": bool(aug_rows), "recipe_version": config["augmentation"]["recipe_version"],
                             "copies_per_image": config["augmentation"]["copies_per_image"] if aug_rows else 0,
                             "ops": config["augmentation"]["ops"], "seed": config["augmentation"]["seed"]},
            "validation_findings": findings, "excluded_records": excluded,
            "outputs": {"yolo_path": str(target / "MTSD-YOLO"), "coco_path": str(target / "MTSD-COCO")},
            "split_manifest_sha256": sha256_file(split_path), "augmentation_manifest_sha256": sha256_file(aug_path),
            "cross_variant_split_manifest_sha256": sha256_file(split_path),
        }
        write_manifest(target / "prep_manifest.json", manifest); outputs[variant_name] = {"path": str(target), "version": version}
    validation = validate_prepared(prepared_root, strict=True, policy=config["validation"])
    if not validation["ok"]: raise RuntimeError("Prepared dataset validation failed:\n" + json.dumps(validation, indent=2))
    return {"source_mode": source_mode, "groups": groups, "records": len(records), "outputs": outputs, "validation": validation}


def preparation_plan(config: dict[str, Any], allow_raw_xml: bool = False,
                     confirm_non_qa: bool = False) -> dict[str, Any]:
    root = Path(config["repo_root"])
    annotation_cfg = config["annotations"]
    raw, groups, mode = discover_sources(
        Path(config["dataset"]["annotations_root"]), root, allow_raw_xml, confirm_non_qa,
        group_scope=annotation_cfg.get("group_scope", "auto"),
        approved_groups=annotation_cfg.get("approved_groups", []),
        unexpected_group_policy=annotation_cfg.get("unexpected_group_policy", "warn"),
    )
    records, findings, excluded = validate_records(raw)
    assignment = assign_splits(records, config["split"]["ratios"], int(config["split"]["seed"]))
    counts = Counter(assignment.values())
    gate = load_qa_gate(Path(annotation_cfg["qa_gate_file"]), root)
    return {"annotation_source_mode": mode, "group_scope": annotation_cfg.get("group_scope", "auto"),
            "approved_groups": annotation_cfg.get("approved_groups", []), "groups": groups, "usable_images": len(records),
            "planned_splits": dict(counts), "validation_findings": findings,
            "excluded_count": len(excluded), "variants": ["augmented", "unaugmented"],
            "qa_gate": gate}
