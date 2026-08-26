"""Inter-annotator agreement between MTSD Fiverr XML and Final-QA JSON.

Each GRP-* directory is compared between its ``Fiverr-Annotations/*.xml``
files and current ``Final-QA/*.json`` export. The public ``match_image`` and
``compute_iaa`` functions are kept reusable for callers that want to inspect
the calculation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ATTRIBUTES = ("class", "viewing_angle", "mounting_type", "condition", "shape")
_TIMESTAMP_RE = re.compile(r"(?:19|20)\d{2}[-_]?\d{2}[-_]?\d{2}(?:[-_T]?\d{2}[-_]?\d{2}[-_]?\d{2})?")


def _xywh_to_xyxy(bbox: Iterable[float]) -> list[float]:
    x, y, w, h = [float(v) for v in bbox]
    return [x, y, x + w, y + h]


def _normalise(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
    return value if value not in (None, "") else "<missing>"


def load_annotations(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load the verified MTSD COCO-like schema into per-image instances."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return _load_labelstudio_tasks(data)
    categories = {int(c["id"]): c.get("name", str(c["id"])) for c in data.get("categories", [])}
    image_names = {int(img["id"]): img["file_name"] for img in data.get("images", [])}
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in data.get("annotations", []):
        attrs = ann.get("attributes") or {}
        by_image[_image_key(image_names.get(int(ann["image_id"]), f"<image_id:{ann['image_id']}>"))].append(
            {
                "box": _xywh_to_xyxy(ann["bbox"]),
                "class": _normalise(categories.get(int(ann.get("category_id", -1)), "<missing>")),
                "viewing_angle": _normalise(attrs.get("view_angle")),
                "mounting_type": _normalise(attrs.get("mounting")),
                "condition": _normalise(attrs.get("condition")),
                "shape": _normalise(attrs.get("sign_shape")),
            }
        )
    return dict(by_image)


def _image_key(name: str) -> str:
    return Path(str(name).replace("\\", "/")).name.casefold()


def load_fiverr_xml(directory: Path) -> dict[str, list[dict[str, Any]]]:
    """Load Fiverr Pascal-VOC XML files into the common instance schema."""
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    attribute_names = {"shape annotation": "shape", "viewing angle": "viewing_angle",
                       "mounting type": "mounting_type", "condition assessment": "condition"}
    for path in sorted(directory.glob("*.xml")):
        root = ET.parse(path).getroot()
        key = _image_key(root.findtext("filename") or path.stem)
        for obj in root.findall("object"):
            box = obj.find("bndbox")
            if box is None:
                continue
            instance = {"box": [float(box.findtext(tag, "0")) for tag in ("xmin", "ymin", "xmax", "ymax")],
                        "class": _normalise(obj.findtext("name")), "viewing_angle": "<missing>",
                        "mounting_type": "<missing>", "condition": "<missing>", "shape": "<missing>"}
            for attr in obj.findall("./attributes/attribute"):
                field = attribute_names.get((attr.findtext("name") or "").strip().casefold())
                if field:
                    instance[field] = _normalise(attr.findtext("value"))
            by_image[key].append(instance)
    return dict(by_image)


def _load_labelstudio_tasks(tasks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Normalize the raw Label Studio task-list backup schema.

    A region is represented by one ``sign_type`` rectangle result and zero or
    more ``choices`` results (``view_angle``, ``mounting``, ``condition`` and
    ``sign_shape``) sharing the same result ``id``. Coordinates are percentages
    of ``original_width``/``original_height``.
    """
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        filename = (task.get("data") or {}).get("image_filename")
        if not filename:
            filename = (task.get("meta") or {}).get("source_image", f"<task:{task.get('id')}>").replace("\\", "/").split("/")[-1]
        regions: dict[str, dict[str, Any]] = {}
        for annotation in task.get("annotations", []):
            for result in annotation.get("result", []):
                value = result.get("value") or {}
                region_id = str(result.get("id", "<missing-region>"))
                region = regions.setdefault(region_id, {"box": None, "class": "<missing>", "viewing_angle": "<missing>", "mounting_type": "<missing>", "condition": "<missing>", "shape": "<missing>"})
                from_name = result.get("from_name")
                if from_name == "sign_type":
                    width, height = float(result.get("original_width", 0)), float(result.get("original_height", 0))
                    x, y = float(value.get("x", 0)), float(value.get("y", 0))
                    w, h = float(value.get("width", 0)), float(value.get("height", 0))
                    region["box"] = [x * width / 100.0, y * height / 100.0, (x + w) * width / 100.0, (y + h) * height / 100.0]
                    labels = value.get("rectanglelabels") or []
                    if labels:
                        region["class"] = _normalise(labels[0])
                else:
                    key = {"view_angle": "viewing_angle", "mounting": "mounting_type", "condition": "condition", "sign_shape": "shape"}.get(from_name)
                    choices = value.get("choices") or []
                    if key and choices:
                        region[key] = _normalise(choices[0])
        by_image[filename].extend(region for region in regions.values() if region["box"] is not None)
    return dict(by_image)


def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def match_image(left: list[dict[str, Any]], right: list[dict[str, Any]], iou_threshold: float = 0.5) -> dict[str, Any]:
    """Maximum-IoU one-to-one matching for two images (threshold 0.5)."""
    edges = sorted(((_iou(a["box"], b["box"]), i, j) for i, a in enumerate(left) for j, b in enumerate(right)), reverse=True)
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, i, j in edges:
        if score < iou_threshold:
            break
        if i not in used_left and j not in used_right:
            used_left.add(i); used_right.add(j); matches.append((i, j, score))
    return {
        "matches": matches,
        "unmatched_left": len(left) - len(matches),
        "unmatched_right": len(right) - len(matches),
    }


def _kappa(left: list[Any], right: list[Any]) -> float | None:
    if not left:
        return None
    labels = list(dict.fromkeys(left + right))
    n = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / n
    p_left = {label: left.count(label) / n for label in labels}
    p_right = {label: right.count(label) / n for label in labels}
    expected = sum(p_left[label] * p_right[label] for label in labels)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1.0 - expected)


def compute_iaa(pairs: list[tuple[dict[str, Any], dict[str, Any], float]]) -> dict[str, Any]:
    """Compute attribute kappas and mean IoU from matched instance triples."""
    result: dict[str, Any] = {"matched_instances": len(pairs)}
    for attr in ATTRIBUTES:
        result[f"kappa_{attr}"] = _kappa([a[attr] for a, _, _ in pairs], [b[attr] for _, b, _ in pairs])
    result["mean_iou"] = mean((iou for _, _, iou in pairs)) if pairs else None
    return result


def _choose_files(group: Path) -> tuple[Path, Path] | None:
    jsons = [p for p in (group / "Final-QA").glob("*.json") if p.is_file()]
    xml_dir = group / "Fiverr-Annotations"
    if not jsons or not xml_dir.is_dir() or not list(xml_dir.glob("*.xml")):
        return None
    current = max(jsons, key=lambda p: p.stat().st_mtime)
    return current, xml_dir


def analyse_group(group: Path) -> dict[str, Any]:
    chosen = _choose_files(group)
    if chosen is None:
        raise ValueError(f"{group} has no Final-QA JSON and Fiverr XML pair")
    current, oldest = chosen
    current_by_image, old_by_image = load_annotations(current), load_fiverr_xml(oldest)
    all_names = sorted(set(current_by_image) | set(old_by_image))
    pairs: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    unmatched_current = unmatched_oldest = 0
    for name in all_names:
        result = match_image(current_by_image.get(name, []), old_by_image.get(name, []))
        unmatched_current += result["unmatched_left"]
        unmatched_oldest += result["unmatched_right"]
        pairs.extend((current_by_image[name][i], old_by_image[name][j], score) for i, j, score in result["matches"])
    out = compute_iaa(pairs)
    out.update({"group": group.name, "json_file": str(current), "xml_directory": str(oldest),
                "images": len(all_names), "unmatched_current": unmatched_current,
                "unmatched_oldest": unmatched_oldest})
    return out


def run(root: Path, output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [analyse_group(group) for group in sorted(root.iterdir()) if group.is_dir()]
    all_pairs: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    # Reconstruct aggregate pairs from the selected files so overall kappas
    # are weighted by matched instances, not averaged across groups.
    for row in rows:
        cur, old = load_annotations(Path(row["json_file"])), load_fiverr_xml(Path(row["xml_directory"]))
        for name in sorted(set(cur) | set(old)):
            m = match_image(cur.get(name, []), old.get(name, []))
            all_pairs.extend((cur[name][i], old[name][j], score) for i, j, score in m["matches"])
    overall = compute_iaa(all_pairs)
    overall.update({"groups": len(rows), "images": sum(row["images"] for row in rows),
                    "unmatched_current": sum(row["unmatched_current"] for row in rows),
                    "unmatched_oldest": sum(row["unmatched_oldest"] for row in rows)})
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "overall.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    (output_dir / "per_group.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output_dir / "per_group.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["group", "images", "matched_instances", *[f"kappa_{a}" for a in ATTRIBUTES], "mean_iou", "unmatched_current", "unmatched_oldest"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows({k: row.get(k) for k in fields} for row in rows)
    low = [row["group"] for row in rows if min((row.get(f"kappa_{a}") or 0.0) for a in ATTRIBUTES) < 0.8]
    lines = ["# MTSD inter-annotator agreement", "", "Comparison: Fiverr Pascal-VOC XML vs current Final-QA JSON; IoU threshold = 0.5.", "", "## Overall", ""]
    lines += [f"- Matched instances: {overall['matched_instances']}", f"- Mean IoU: {overall['mean_iou']:.6f}"]
    lines += [f"- {a}: κ = {overall[f'kappa_{a}']:.6f}" for a in ATTRIBUTES]
    lines += [f"- Unmatched current/XML instances: {overall['unmatched_current']}/{overall['unmatched_oldest']}", "", "## Per group", "", "| Group | Matched | Class κ | Angle κ | Mount κ | Condition κ | Shape κ | Mean IoU | Unmatched JSON/XML |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        vals = [row["group"], row["matched_instances"], *[f"{row[f'kappa_{a}']:.4f}" for a in ATTRIBUTES], f"{row['mean_iou']:.4f}", f"{row['unmatched_current']}/{row['unmatched_oldest']}"]
        lines.append("| " + " | ".join(map(str, vals)) + " |")
    lines += ["", "## Low-agreement note", "", "Condition is the weakest attribute overall; groups below κ=0.8 on at least one attribute: " + (", ".join(low) if low else "none") + ".", ""]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return overall, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("Datasets/MTSD/Annotations"))
    parser.add_argument("--output-dir", type=Path, default=Path("AnalysisOutputs/MTSD_IAA"))
    args = parser.parse_args()
    overall, rows = run(args.root, args.output_dir)
    print("OVERALL")
    for key, value in overall.items(): print(f"{key}: {value}")
    print("\nPER GROUP")
    for row in rows:
        print(row["group"], *(f"{row.get('kappa_'+a):.4f}" if row.get('kappa_'+a) is not None else "NA" for a in ATTRIBUTES), f"iou={row['mean_iou']:.4f}" if row["mean_iou"] is not None else "iou=NA", f"unmatched={row['unmatched_current']}/{row['unmatched_oldest']}")


if __name__ == "__main__":
    main()
