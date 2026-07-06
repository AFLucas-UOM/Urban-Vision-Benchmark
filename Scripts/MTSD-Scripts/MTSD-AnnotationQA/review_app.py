#!/usr/bin/env python3
"""Gradio review UI for MTSD annotation QA findings.

Loads the most recent audit (or --audit <dir>) and lets you visually review:

* **Attribute fixes** - image with the annotation box drawn, the problematic
  field, and a vocabulary dropdown / free-text box to supply the correct value.
* **Duplicates** - both boxes overlaid (A red, B blue) with their attribute
  values side by side; choose which annotation to keep, keep both, or skip.

SAFETY: this app NEVER edits the QA annotation files. Every decision is
appended to ``reviewed_decisions.json`` inside the audit folder; the separate
``apply_fixes.py`` step (with backups, dry-run and confirmation) is the only
thing that modifies annotations.

Usage:
    python review_app.py                # review the latest audit
    python review_app.py --audit outputs/audit-20260706-120000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config  # noqa: E402
from annotation_schema import load_attribute_schema  # noqa: E402

SCHEMA = load_attribute_schema()
DISPLAY_WIDTH = 1100
CROP_PAD = 0.6  # context padding around the box, as a fraction of box size


# ---------------------------------------------------------------------------
# Audit + annotation loading
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=8)
def load_qa_file(rel_path: str) -> dict:
    data = json.loads((config.BASE_DIR / rel_path).read_text(encoding="utf-8-sig"))
    return {
        "images": {im.get("id"): im for im in data.get("images", [])},
        "annotations": {a.get("id"): a for a in data.get("annotations", [])},
        "categories": {c.get("id"): c.get("name") for c in data.get("categories", [])},
    }


def annotation_context(qa_file: str, annotation_id) -> tuple[dict | None, dict | None, dict]:
    """Return (annotation, image_entry, categories) for a finding row."""
    payload = load_qa_file(qa_file)
    annotation = None
    for key in (annotation_id, _as_int(annotation_id), str(annotation_id)):
        if key in payload["annotations"]:
            annotation = payload["annotations"][key]
            break
    image = payload["images"].get(annotation.get("image_id")) if annotation else None
    return annotation, image, payload["categories"]


def _as_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return value


def open_source_image(image_entry: dict) -> Image.Image | None:
    source = str(image_entry.get("source_image", "") or "").replace("\\", "/")
    path = config.BASE_DIR / source
    if not path.exists():
        return None
    return Image.open(path).convert("RGB")


def draw_finding(image_entry: dict, boxes: list[tuple[list[float], str]]) -> tuple[Image.Image | None, Image.Image | None]:
    """Full view + zoomed crop with the given [(bbox_xywh, colour), ...] drawn."""
    img = open_source_image(image_entry)
    if img is None:
        return None, None
    draw = ImageDraw.Draw(img)
    line = max(3, img.width // 400)
    x_min, y_min, x_max, y_max = img.width, img.height, 0, 0
    for bbox, colour in boxes:
        try:
            x, y, w, h = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        draw.rectangle([x, y, x + w, y + h], outline=colour, width=line)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x + w), max(y_max, y + h)

    if x_max > x_min and y_max > y_min:
        pad_x, pad_y = (x_max - x_min) * CROP_PAD, (y_max - y_min) * CROP_PAD
        crop = img.crop((
            max(0, int(x_min - pad_x)), max(0, int(y_min - pad_y)),
            min(img.width, int(x_max + pad_x)), min(img.height, int(y_max + pad_y)),
        ))
    else:
        crop = img.copy()

    if img.width > DISPLAY_WIDTH:
        img = img.resize((DISPLAY_WIDTH, int(img.height * DISPLAY_WIDTH / img.width)))
    if crop.width > DISPLAY_WIDTH:
        crop = crop.resize((DISPLAY_WIDTH, int(crop.height * DISPLAY_WIDTH / crop.width)))
    return img, crop


# ---------------------------------------------------------------------------
# Decisions persistence (append-or-update; the review app never edits QA files)
# ---------------------------------------------------------------------------

class DecisionStore:
    def __init__(self, audit_dir: Path) -> None:
        self.path = audit_dir / "reviewed_decisions.json"
        self.payload = {"audit_dir": str(audit_dir), "decisions": []}
        if self.path.exists():
            try:
                self.payload = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                backup = self.path.with_suffix(f".corrupt-{datetime.now():%Y%m%d-%H%M%S}.json")
                self.path.rename(backup)

    def _key(self, decision: dict) -> tuple:
        if decision["type"] == "set_attribute":
            return ("set", decision["qa_file"], str(decision["annotation_id"]), decision["attribute"])
        if decision["type"] == "delete_annotation":
            return ("del", decision["qa_file"], str(decision["annotation_id"]))
        return (decision["type"], decision["qa_file"],
                str(decision.get("annotation_id_a")), str(decision.get("annotation_id_b")))

    def record(self, decision: dict) -> int:
        decision["decided_at"] = datetime.now().isoformat(timespec="seconds")
        key = self._key(decision)
        kept = [d for d in self.payload["decisions"] if self._key(d) != key]
        # A delete decision supersedes any attribute fix on the same annotation.
        if decision["type"] == "delete_annotation":
            kept = [d for d in kept
                    if not (d["type"] == "set_attribute"
                            and d["qa_file"] == decision["qa_file"]
                            and str(d["annotation_id"]) == str(decision["annotation_id"]))]
        kept.append(decision)
        self.payload["decisions"] = kept
        self.path.write_text(json.dumps(self.payload, indent=2) + "\n", encoding="utf-8")
        return len(kept)


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_app(audit_dir: Path):
    import gradio as gr

    attribute_findings = read_csv(audit_dir / "missing_attributes.csv") + read_csv(
        audit_dir / "invalid_attributes.csv"
    )
    attribute_findings = [f for f in attribute_findings if f.get("attribute") in SCHEMA]
    duplicate_findings = read_csv(audit_dir / "duplicate_candidates.csv")
    store = DecisionStore(audit_dir)

    # ----- attribute tab callbacks -----
    def show_attribute(index: int):
        if not attribute_findings:
            return None, None, "No attribute findings in this audit.", gr.update(choices=[]), "", index
        index = max(0, min(index, len(attribute_findings) - 1))
        finding = attribute_findings[index]
        annotation, image, categories = annotation_context(finding["qa_file"], finding["annotation_id"])
        header = (
            f"**Finding {index + 1}/{len(attribute_findings)}** - group `{finding['group']}` - "
            f"image `{finding['image']}` - annotation `{finding['annotation_id']}`\n\n"
            f"Attribute **{finding['attribute']}** = `{finding['value']}` "
            f"(status: {finding.get('status', 'missing')}"
            + (f", suggestion: **{finding['suggestion']}**" if finding.get("suggestion") else "")
            + ")"
        )
        if annotation is None or image is None:
            return None, None, header + "\n\n*Annotation or image not found (re-run the scan?).*", \
                gr.update(choices=SCHEMA[finding["attribute"]]), "", index
        attrs = {k: v for k, v in (annotation.get("attributes") or {}).items()
                 if k not in config.NON_LABEL_ATTRIBUTE_KEYS}
        klass = categories.get(annotation.get("category_id"), "?")
        details = header + f"\n\nClass: **{klass}**\n\nCurrent attributes: `{json.dumps(attrs)}`"
        full, crop = draw_finding(image, [(annotation.get("bbox", []), "#e34948")])
        vocab = SCHEMA[finding["attribute"]]
        suggested = finding.get("suggestion") or None
        return full, crop, details, gr.update(choices=vocab, value=suggested), "", index

    def save_attribute(index: int, dropdown_value: str | None, free_text: str):
        if not attribute_findings:
            return "Nothing to save.", index
        finding = attribute_findings[max(0, min(index, len(attribute_findings) - 1))]
        new_value = (free_text or "").strip() or dropdown_value
        if not new_value:
            return "Pick a vocabulary value or type one before saving.", index
        count = store.record({
            "type": "set_attribute",
            "qa_file": finding["qa_file"],
            "annotation_id": _as_int(finding["annotation_id"]),
            "attribute": finding["attribute"],
            "old_value": finding["value"],
            "new_value": new_value,
        })
        return f"Saved: {finding['attribute']} -> '{new_value}' ({count} decisions recorded).", index

    def nav_attribute(index: int, step: int):
        return show_attribute(index + step)

    # ----- duplicate tab callbacks -----
    def show_duplicate(index: int):
        if not duplicate_findings:
            return None, None, "No duplicate findings in this audit.", index
        index = max(0, min(index, len(duplicate_findings) - 1))
        finding = duplicate_findings[index]
        ann_a, image, categories = annotation_context(finding["qa_file"], finding["annotation_id_a"])
        ann_b, _, _ = annotation_context(finding["qa_file"], finding["annotation_id_b"])
        header = (
            f"**Pair {index + 1}/{len(duplicate_findings)}** - `{finding['kind']}` - IoU {finding['iou']} - "
            f"group `{finding['group']}` - image `{finding['image']}`\n\n"
            f"<span style='color:#e34948'>A = annotation {finding['annotation_id_a']}</span> vs "
            f"<span style='color:#2a78d6'>B = annotation {finding['annotation_id_b']}</span>"
        )
        if ann_a is None or ann_b is None or image is None:
            return None, None, header + "\n\n*Annotation(s) or image missing (already fixed?).*", index

        def attr_row(ann):
            attrs = {k: v for k, v in (ann.get("attributes") or {}).items()
                     if k not in config.NON_LABEL_ATTRIBUTE_KEYS}
            return {"class": categories.get(ann.get("category_id"), "?"), **attrs}

        row_a, row_b = attr_row(ann_a), attr_row(ann_b)
        keys = sorted(set(row_a) | set(row_b), key=lambda k: (k != "class", k))
        table = ["| field | A (red) | B (blue) | same |", "| --- | --- | --- | --- |"]
        for key in keys:
            same = "yes" if row_a.get(key) == row_b.get(key) else "**NO**"
            table.append(f"| {key} | {row_a.get(key, '-')} | {row_b.get(key, '-')} | {same} |")
        details = header + "\n\n" + "\n".join(table)
        full, crop = draw_finding(
            image, [(ann_a.get("bbox", []), "#e34948"), (ann_b.get("bbox", []), "#2a78d6")]
        )
        return full, crop, details, index

    def decide_duplicate(index: int, decision: str):
        if not duplicate_findings:
            return "Nothing to decide.", index
        finding = duplicate_findings[max(0, min(index, len(duplicate_findings) - 1))]
        id_a, id_b = _as_int(finding["annotation_id_a"]), _as_int(finding["annotation_id_b"])
        if decision in ("keep_a", "keep_b"):
            delete_id = id_b if decision == "keep_a" else id_a
            keep_id = id_a if decision == "keep_a" else id_b
            count = store.record({
                "type": "delete_annotation",
                "qa_file": finding["qa_file"],
                "annotation_id": delete_id,
                "reason": f"duplicate of annotation {keep_id} (IoU {finding['iou']}, {finding['kind']})",
            })
            return f"Marked annotation {delete_id} for deletion, keeping {keep_id} ({count} decisions).", index
        count = store.record({
            "type": "keep_both",
            "qa_file": finding["qa_file"],
            "annotation_id_a": id_a,
            "annotation_id_b": id_b,
            "note": "reviewed: both annotations are legitimate",
        })
        return f"Recorded keep-both for {id_a}/{id_b} ({count} decisions).", index

    def nav_duplicate(index: int, step: int):
        return show_duplicate(index + step)

    # ----- layout -----
    with gr.Blocks(title="MTSD Annotation QA Review") as demo:
        gr.Markdown(
            f"# MTSD Annotation QA review\n"
            f"Audit: `{audit_dir}` - {len(attribute_findings)} attribute findings, "
            f"{len(duplicate_findings)} duplicate pairs.\n\n"
            f"Decisions are written to `reviewed_decisions.json`; **source annotations are "
            f"never modified here** - run `apply_fixes.py` afterwards."
        )
        with gr.Tab("Attribute fixes"):
            attr_index = gr.State(0)
            attr_details = gr.Markdown()
            with gr.Row():
                attr_full = gr.Image(label="Image (box in red)", type="pil")
                attr_crop = gr.Image(label="Zoom", type="pil")
            with gr.Row():
                attr_dropdown = gr.Dropdown(label="Correct value (controlled vocabulary)", choices=[])
                attr_freetext = gr.Textbox(label="...or type a value (overrides dropdown)")
            attr_status = gr.Markdown()
            with gr.Row():
                attr_prev = gr.Button("< Previous")
                attr_save = gr.Button("Save decision", variant="primary")
                attr_skip = gr.Button("Skip / Next >")
            attr_outputs = [attr_full, attr_crop, attr_details, attr_dropdown, attr_freetext, attr_index]
            attr_prev.click(lambda i: nav_attribute(i, -1), [attr_index], attr_outputs)
            attr_skip.click(lambda i: nav_attribute(i, +1), [attr_index], attr_outputs)
            attr_save.click(save_attribute, [attr_index, attr_dropdown, attr_freetext], [attr_status, attr_index])

        with gr.Tab("Duplicates"):
            dup_index = gr.State(0)
            dup_details = gr.Markdown()
            with gr.Row():
                dup_full = gr.Image(label="Image (A red, B blue)", type="pil")
                dup_crop = gr.Image(label="Zoom", type="pil")
            dup_status = gr.Markdown()
            with gr.Row():
                dup_prev = gr.Button("< Previous")
                dup_keep_a = gr.Button("Keep A (delete B)", variant="primary")
                dup_keep_b = gr.Button("Keep B (delete A)", variant="primary")
                dup_keep_both = gr.Button("Keep both")
                dup_skip = gr.Button("Skip / Next >")
            dup_outputs = [dup_full, dup_crop, dup_details, dup_index]
            dup_prev.click(lambda i: nav_duplicate(i, -1), [dup_index], dup_outputs)
            dup_skip.click(lambda i: nav_duplicate(i, +1), [dup_index], dup_outputs)
            dup_keep_a.click(lambda i: decide_duplicate(i, "keep_a"), [dup_index], [dup_status, dup_index])
            dup_keep_b.click(lambda i: decide_duplicate(i, "keep_b"), [dup_index], [dup_status, dup_index])
            dup_keep_both.click(lambda i: decide_duplicate(i, "keep_both"), [dup_index], [dup_status, dup_index])

        demo.load(lambda: show_attribute(0), outputs=attr_outputs)
        demo.load(lambda: show_duplicate(0), outputs=dup_outputs)
    return demo


def main() -> int:
    parser = argparse.ArgumentParser(description="Review MTSD annotation QA findings (writes decisions only).")
    parser.add_argument("--audit", type=Path, default=None, help="Audit folder (default: latest).")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    audit_dir = args.audit or config.latest_audit_dir()
    if audit_dir is None or not Path(audit_dir).is_dir():
        print("No audit found. Run scan_annotations.py first.")
        return 1
    demo = build_app(Path(audit_dir))
    demo.launch(server_port=args.port, inbrowser=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
