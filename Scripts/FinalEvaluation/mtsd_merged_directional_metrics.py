from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SUPERVISED = ROOT / "Scripts" / "MTSD-Scripts" / "MTSD-SupervisedDetection"
sys.path.insert(0, str(SUPERVISED))

from mtsd_detection.evaluate import coco_eval  # noqa: E402


RUNS = {
    "yolo11m": ROOT / "Results" / "MTSD-Runs" / "YOLO11-MTSD" / "E09_yolo11m_mtsdqa1aug_img640_eb32_e100_adamw_s42",
    "yolo12m": ROOT / "Results" / "MTSD-Runs" / "YOLO12-MTSD" / "E03_yolo12m_mtsdqa1aug_img640_eb32_e100_adamw_s42",
    "yolo26m": ROOT / "Results" / "MTSD-Runs" / "YOLO26-MTSD" / "E06_yolo26m_mtsdqa1aug_img640_eb32_e100_adamw_s42",
}

ANNOTATION_FILE = ROOT / "Datasets" / "MTSD" / "Prepared" / "MTSD-Augmented" / "MTSD-COCO" / "test" / "_annotations.coco.json"
IMAGE_DIR = ROOT / "Datasets" / "MTSD" / "Prepared" / "MTSD-Augmented" / "MTSD-COCO" / "test"
MERGE_SOURCE = "Tourist Sign"
MERGE_TARGET = "Directional Sign"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_category_maps(annotation: dict[str, Any]) -> tuple[dict[int, int], list[dict[str, Any]], dict[str, Any]]:
    categories = sorted(annotation["categories"], key=lambda row: int(row["id"]))
    source_id = next(int(row["id"]) for row in categories if row["name"] == MERGE_SOURCE)
    target_id = next(int(row["id"]) for row in categories if row["name"] == MERGE_TARGET)

    kept = [row for row in categories if int(row["id"]) != source_id]
    old_to_new_category: dict[int, int] = {}
    new_categories: list[dict[str, Any]] = []
    for new_id, category in enumerate(kept):
        old_id = int(category["id"])
        old_to_new_category[old_id] = new_id
        new_categories.append({**category, "id": new_id})
    old_to_new_category[source_id] = old_to_new_category[target_id]

    provenance = {
        "merge_source": MERGE_SOURCE,
        "merge_target": MERGE_TARGET,
        "source_category_id": source_id,
        "target_category_id": target_id,
        "original_category_count": len(categories),
        "merged_category_count": len(new_categories),
        "original_categories": [{"id": int(row["id"]), "name": row["name"]} for row in categories],
        "merged_categories": [{"id": int(row["id"]), "name": row["name"]} for row in new_categories],
        "old_to_new_category_id": {str(key): value for key, value in sorted(old_to_new_category.items())},
    }
    return old_to_new_category, new_categories, provenance


def remap_annotation(annotation: dict[str, Any], old_to_new_category: dict[int, int], categories: list[dict[str, Any]]) -> dict[str, Any]:
    remapped = {**annotation, "categories": categories}
    remapped_annotations = []
    for row in annotation["annotations"]:
        remapped_annotations.append({**row, "category_id": old_to_new_category[int(row["category_id"])]})
    remapped["annotations"] = remapped_annotations
    return remapped


def clean_coco_box(row: dict[str, Any], images: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    image = images.get(int(row.get("image_id", -1)))
    if image is None:
        return None
    x, y, w, h = [float(value) for value in row["bbox"]]
    x0 = max(0.0, min(x, float(image["width"])))
    y0 = max(0.0, min(y, float(image["height"])))
    x1 = max(0.0, min(x + w, float(image["width"])))
    y1 = max(0.0, min(y + h, float(image["height"])))
    if x1 <= x0 or y1 <= y0:
        return None
    return {**row, "bbox": [x0, y0, x1 - x0, y1 - y0]}


def export_yolo_predictions(checkpoint: Path, output: Path, image_size: int, old_to_new_category: dict[int, int]) -> dict[str, int]:
    from ultralytics import YOLO

    annotation = load_json(ANNOTATION_FILE)
    image_ids = {row["file_name"]: int(row["id"]) for row in annotation["images"]}
    images = {int(row["id"]): row for row in annotation["images"]}
    rows = []
    dropped = 0
    model = YOLO(str(checkpoint))
    for result in model.predict(source=str(IMAGE_DIR), imgsz=image_size, conf=0.001, iou=0.7, max_det=100, stream=True, verbose=False):
        image_id = image_ids.get(Path(result.path).name)
        if image_id is None:
            continue
        for xyxy, score, class_id in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
            x0, y0, x1, y1 = [float(value) for value in xyxy]
            row = {
                "image_id": image_id,
                "category_id": old_to_new_category[int(class_id)],
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": float(score),
            }
            cleaned = clean_coco_box(row, images)
            if cleaned is None:
                dropped += 1
            else:
                rows.append(cleaned)
    write_json(output, rows)
    return {"prediction_count": len(rows), "dropped_degenerate_predictions": dropped}


def remap_existing_predictions(input_path: Path, output: Path, old_to_new_category: dict[int, int]) -> dict[str, int]:
    annotation = load_json(ANNOTATION_FILE)
    images = {int(row["id"]): row for row in annotation["images"]}
    rows = []
    dropped = 0
    for row in load_json(input_path):
        remapped = {**row, "category_id": old_to_new_category[int(row["category_id"])]}
        cleaned = clean_coco_box(remapped, images)
        if cleaned is None:
            dropped += 1
        else:
            rows.append(cleaned)
    write_json(output, rows)
    return {"prediction_count": len(rows), "dropped_degenerate_predictions": dropped}


def row_for_csv(model: str, run_record: dict[str, Any], metrics: dict[str, Any], prediction_meta: dict[str, int], output_prediction: Path) -> dict[str, Any]:
    native = run_record.get("native_metrics", {}).get("test", {})
    return {
        "model": model,
        "family": run_record.get("family"),
        "scale": run_record.get("scale"),
        "dataset_version": run_record.get("dataset_version"),
        "taxonomy_variant": "tourist_sign_merged_into_directional_sign",
        "map50_95": metrics.get("map50_95"),
        "map50": metrics.get("map50"),
        "map75": metrics.get("map75"),
        "ar100": metrics.get("ar100"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "f1_score_threshold": metrics.get("f1_score_threshold"),
        "prediction_count": prediction_meta["prediction_count"],
        "dropped_degenerate_predictions": prediction_meta["dropped_degenerate_predictions"],
        "native_map50_95_original_taxonomy": native.get("metrics/mAP50-95(B)"),
        "native_map50_original_taxonomy": native.get("metrics/mAP50(B)"),
        "run_dir": str(RUNS[model]),
        "merged_predictions": str(output_prediction),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "Results" / "MTSD-Results" / "Merged-Directional")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse existing prediction files when present.")
    args = parser.parse_args()

    annotation = load_json(ANNOTATION_FILE)
    old_to_new_category, merged_categories, provenance = build_category_maps(annotation)
    output_dir = args.output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)

    merged_annotation = remap_annotation(annotation, old_to_new_category, merged_categories)
    merged_annotation_path = output_dir / "test_annotations_tourist_merged_directional.coco.json"
    write_json(merged_annotation_path, merged_annotation)

    rows = []
    run_metrics = {}
    for model, run_dir in RUNS.items():
        run_record = load_json(run_dir / "run_record.json")
        prediction_path = run_dir / "unified_test_predictions.json"
        merged_prediction_path = output_dir / f"{model}_merged_predictions.json"
        if args.reuse_existing and prediction_path.exists():
            prediction_meta = remap_existing_predictions(prediction_path, merged_prediction_path, old_to_new_category)
        else:
            prediction_meta = export_yolo_predictions(Path(run_record["checkpoint_best"]), merged_prediction_path, 640, old_to_new_category)
        metrics = coco_eval(merged_annotation_path, merged_prediction_path)
        run_metrics[model] = {"metrics": metrics, **prediction_meta}
        rows.append(row_for_csv(model, run_record, metrics, prediction_meta, merged_prediction_path))

    fields = list(rows[0])
    csv_path = output_dir / "merged_directional_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    write_json(output_dir / "merged_directional_metrics.json", {"provenance": provenance, "runs": rows, "raw_metrics": run_metrics})
    markdown = ["# MTSD supervised metrics: Tourist Sign merged into Directional Sign", ""]
    markdown.append(f"- Source annotation: `{ANNOTATION_FILE}`")
    markdown.append(f"- Original categories: {provenance['original_category_count']}")
    markdown.append(f"- Merged categories: {provenance['merged_category_count']}")
    markdown.append(f"- Merge: `{MERGE_SOURCE}` -> `{MERGE_TARGET}`")
    markdown.append("")
    markdown.append("| model | mAP50-95 | mAP50 | precision | recall | F1 | predictions | dropped degenerate |")
    markdown.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        markdown.append(
            f"| {row['model']} | {row['map50_95']:.6f} | {row['map50']:.6f} | {row['precision']:.6f} | "
            f"{row['recall']:.6f} | {row['f1']:.6f} | {row['prediction_count']} | {row['dropped_degenerate_predictions']} |"
        )
    (output_dir / "merged_directional_metrics.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
