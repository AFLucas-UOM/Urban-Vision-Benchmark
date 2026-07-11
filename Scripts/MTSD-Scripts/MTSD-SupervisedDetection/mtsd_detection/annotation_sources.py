from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

from .utils import sha256_file

CLASS_NAMES = [
    "Pedestrian Crossing", "Stop Sign", "No Entry (One Way)", "Roundabout Ahead",
    "No Through Road (T-Junction)", "Blind-Spot Mirror (Convex Mirror)", "Street Sign",
    "Directional Sign", "Tourist Sign", "Auxiliary Sign", "Back-Unknown", "Other-Unknown",
]
LABEL_ALIASES = {"Stop": "Stop Sign"}


def _group_number(path: Path) -> int:
    try:
        return int(path.name.split("-")[-1])
    except ValueError:
        return 10**9


def _validate_categories(categories: list[dict[str, Any]], path: Path) -> tuple[list[str], dict[int, int]]:
    ordered = sorted((int(item["id"]), str(item["name"])) for item in categories)
    names = [name for _, name in ordered]
    if names != CLASS_NAMES:
        raise ValueError(f"Category vocabulary mismatch in {path}: {names!r}; expected {CLASS_NAMES!r}")
    return names, {qa_id: index for index, (qa_id, _) in enumerate(ordered)}


def parse_qa(path: Path, repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ValueError(f"Invalid QA JSON {path}: {exc}") from exc
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"QA JSON {path} has no list field {key!r}")
    _, id_map = _validate_categories(payload["categories"], path)
    by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        by_image[annotation.get("image_id")].append(annotation)
    group = path.parents[1].name
    records = []
    for image in payload["images"]:
        rel = str(image.get("source_image") or "").replace("\\", "/")
        source = repo_root / rel if rel else repo_root / "Datasets" / "MTSD" / group / str(image["file_name"])
        records.append({
            "group": group, "image_id": image["id"], "file_name": image["file_name"],
            "source_path": source, "width": image.get("width"), "height": image.get("height"),
            "raw_boxes": by_image.get(image["id"], []), "qa_id_map": id_map,
            "annotation_source_type": "final_qa", "annotation_source_path": path,
            "annotation_source_sha256": sha256_file(path),
        })
    meta = {"group": group, "status": "included", "source_type": "final_qa",
            "source_path": str(path), "source_sha256": sha256_file(path),
            "images": len(payload["images"]), "boxes": len(payload["annotations"])}
    return records, meta


def parse_voc_xml(path: Path, repo_root: Path, group: str) -> dict[str, Any]:
    try:
        root = ET.parse(path).getroot()
    except Exception as exc:
        raise ValueError(f"Invalid XML {path}: {exc}") from exc
    if root.tag != "annotation":
        raise ValueError(f"{path}: root must be <annotation>")
    try:
        width = int(float(root.findtext("size/width", "0")))
        height = int(float(root.findtext("size/height", "0")))
    except ValueError as exc:
        raise ValueError(f"{path}: non-numeric image size") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"{path}: image size must be positive")
    boxes = []
    for index, obj in enumerate(root.findall("object"), start=1):
        name = LABEL_ALIASES.get((obj.findtext("name") or "").strip(), (obj.findtext("name") or "").strip())
        if name not in CLASS_NAMES:
            raise ValueError(f"{path}: unknown class {name!r}")
        node = obj.find("bndbox")
        try:
            x0, y0, x1, y1 = (float(node.findtext(key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax"))
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"{path}: malformed bndbox") from exc
        if not (x0 < x1 and y0 < y1):
            raise ValueError(f"{path}: degenerate bndbox")
        boxes.append({"id": index, "category_id": CLASS_NAMES.index(name) + 1,
                      "bbox": [x0, y0, x1 - x0, y1 - y0]})
    filename = Path((root.findtext("filename") or path.with_suffix(".jpg").name).replace("\\", "/")).name
    candidates = [repo_root / "Datasets" / "MTSD" / group / "Images" / filename,
                  repo_root / "Datasets" / "MTSD" / group / filename,
                  repo_root / "Datasets" / "MTSD" / group / (root.findtext("filename") or "")]
    source = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    return {"group": group, "image_id": path.stem, "file_name": filename, "source_path": source,
            "width": width, "height": height, "raw_boxes": boxes,
            "qa_id_map": {i + 1: i for i in range(len(CLASS_NAMES))},
            "annotation_source_type": "raw_xml", "annotation_source_path": path,
            "annotation_source_sha256": sha256_file(path)}


def discover_sources(annotations_root: Path, repo_root: Path, allow_raw_xml: bool = False,
                     confirm_non_qa: bool = False, group_scope: str = "auto",
                     approved_groups: list[str] | None = None,
                     unexpected_group_policy: str = "warn") -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    mtsd_root = annotations_root.parent
    group_names = {p.name for p in mtsd_root.glob("GRP-*") if p.is_dir()}
    group_names.update(p.name for p in annotations_root.glob("GRP-*") if p.is_dir())
    groups = sorted(group_names, key=lambda name: int(name.split("-")[-1]))
    if group_scope not in {"auto", "explicit"}:
        raise ValueError("group_scope must be 'auto' or 'explicit'")
    approved = list(dict.fromkeys(approved_groups or []))
    if group_scope == "explicit" and not approved:
        raise ValueError("Explicit annotation scope requires approved_groups")
    qa_discovered = {
        group for group in groups
        if any(".bak" not in path.name.lower()
               for path in (annotations_root / group / "Final-QA").glob("QA-GRP*.json"))
    }
    unexpected = sorted(qa_discovered - set(approved), key=lambda name: int(name.split("-")[-1])) \
        if group_scope == "explicit" else []
    if unexpected and unexpected_group_policy == "fail":
        raise ValueError(f"Final-QA groups outside approved scope: {unexpected}")
    if unexpected_group_policy not in {"warn", "fail"}:
        raise ValueError("unexpected_group_policy must be warn or fail")
    selected_groups = approved if group_scope == "explicit" else groups
    records: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    used_raw = False
    if unexpected:
        metadata.extend({"group": group, "status": "unexpected", "reason": "Final-QA exists outside approved scope",
                         "source_type": "final_qa", "source_path": None, "source_sha256": None,
                         "images": 0, "boxes": 0} for group in unexpected)
    for group in selected_groups:
        group_dir = annotations_root / group
        qa_files = sorted(p for p in (group_dir / "Final-QA").glob("QA-GRP*.json") if ".bak" not in p.name.lower())
        if qa_files:
            if len(qa_files) != 1:
                raise ValueError(f"Expected one Final-QA JSON for {group}, found {qa_files}")
            rows, meta = parse_qa(qa_files[0], repo_root)
            records.extend(rows); metadata.append(meta)
            continue
        if group_scope == "explicit":
            raise FileNotFoundError(f"Approved group {group} has no valid Final-QA JSON")
        xml_files = sorted((group_dir / "Fiverr-Annotations").glob("*.xml"))
        if allow_raw_xml and xml_files:
            if not confirm_non_qa:
                raise PermissionError("Raw-XML fallback requires both --allow-raw-xml-fallback and --confirm-non-qa-data")
            rows, errors = [], []
            for xml in xml_files:
                try:
                    rows.append(parse_voc_xml(xml, repo_root, group))
                except ValueError as exc:
                    errors.append(str(exc))
            if errors:
                raise ValueError(f"Raw XML validation failed for {group}:\n" + "\n".join(errors[:20]))
            records.extend(rows); used_raw = True
            metadata.append({"group": group, "status": "included", "source_type": "raw_xml",
                             "source_path": str(group_dir / "Fiverr-Annotations"),
                             "source_sha256": None, "images": len(rows),
                             "boxes": sum(len(row["raw_boxes"]) for row in rows)})
        else:
            metadata.append({"group": group, "status": "skipped", "reason": "no Final-QA JSON",
                             "source_type": None, "source_path": None, "source_sha256": None,
                             "images": 0, "boxes": 0})
    if not records:
        raise FileNotFoundError(f"No usable annotations under {annotations_root}")
    return records, metadata, "mixed" if used_raw else "qa_only"
