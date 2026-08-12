"""Human-readable GT-versus-prediction sheets for batch evaluation.

This module deliberately delegates all matching decisions to ``metrics`` so a
visual label such as TP, FP, or FN always has exactly the same meaning as the
numbers in ``per_image_metrics.csv``.
"""

from __future__ import annotations

import csv
import html
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import metrics


def _slug(value: str, limit: int = 80) -> str:
    """Stable, Windows-safe component for a visualisation filename."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip(".-")
    return (value or "unnamed")[:limit]


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str,
           color: str, font, image_width: int) -> None:
    """Draw a wrapped opaque label that stays within the image canvas."""
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > image_width - 12:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    text = "\n".join(lines)
    x = max(2, min(int(xy[0]), image_width - 4))
    y = max(2, int(xy[1]))
    bbox = draw.multiline_textbbox((x, y), text, font=font, spacing=2)
    draw.rectangle((bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1), fill=color)
    draw.multiline_text((x, y), text, fill="white", font=font, spacing=2)


def _duplicate_prediction(pred: dict, result: dict, gt_boxes: list[dict], threshold: float) -> bool:
    """Explain a metric FP as a duplicate when ``metrics.match_image`` did."""
    matched_gt = {id(gt) for _, gt, _ in result["matches"]}
    best_iou, best_gt = 0.0, None
    for gt in gt_boxes:
        iou = metrics.iou_xyxy(pred, gt)
        if iou > best_iou:
            best_iou, best_gt = iou, gt
    return best_gt is not None and id(best_gt) in matched_gt and best_iou >= threshold


def _target_boxes(record: dict, target_classes: set[str] | None) -> list[dict]:
    return [box for box in record["boxes"]
            if target_classes is None or box.get("class_name") in target_classes]


def _select_records(records: list[dict], by_image: dict[str, list[dict]],
                    target_classes: set[str] | None, iou_threshold: float,
                    limit: int | None) -> list[dict]:
    """Choose a deterministic mix of TP-, FP-, FN-, and clean-image examples."""
    if limit is None or limit >= len(records):
        return records
    if limit <= 0:
        return []
    buckets: list[list[dict]] = [[], [], [], []]
    for record in records:
        outcome = metrics.match_image(
            by_image.get(record["image_id"], []),
            _target_boxes(record, target_classes),
            iou_threshold,
        )
        if outcome["tp"]:
            buckets[0].append(record)
        if outcome["fp"]:
            buckets[1].append(record)
        if outcome["fn"]:
            buckets[2].append(record)
        if not (outcome["tp"] or outcome["fp"] or outcome["fn"]):
            buckets[3].append(record)

    selected: list[dict] = []
    selected_ids: set[str] = set()
    positions = [0] * len(buckets)
    while len(selected) < limit:
        added = False
        for bucket_index, bucket in enumerate(buckets):
            while (positions[bucket_index] < len(bucket)
                   and bucket[positions[bucket_index]]["image_id"] in selected_ids):
                positions[bucket_index] += 1
            if positions[bucket_index] >= len(bucket):
                continue
            record = bucket[positions[bucket_index]]
            positions[bucket_index] += 1
            selected.append(record)
            selected_ids.add(record["image_id"])
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
    if len(selected) < limit:
        for record in records:
            if record["image_id"] not in selected_ids:
                selected.append(record)
                selected_ids.add(record["image_id"])
                if len(selected) >= limit:
                    break
    return selected


def _draw_box(draw: ImageDraw.ImageDraw, box: dict, scale_x: float,
              scale_y: float, color: str, line: int) -> None:
    draw.rectangle((box["x0"] * scale_x, box["y0"] * scale_y,
                    box["x1"] * scale_x, box["y1"] * scale_y),
                   outline=color, width=line)


def _make_sheet(record: dict, predictions: list[dict], model: str, prompt: str,
                iou_threshold: float,
                target_classes: set[str] | None = None) -> tuple[Image.Image, dict]:
    """Return one targeted GT / prediction sheet and its metric-identical counts."""
    with Image.open(record["image_path"]) as source:
        base = source.convert("RGB")
    original_width, original_height = base.size
    # Keep thousands of qualitative samples practical to browse and upload.
    base.thumbnail((1100, 900), Image.Resampling.LANCZOS)
    scale_x = base.width / original_width
    scale_y = base.height / original_height
    gt_boxes = _target_boxes(record, target_classes)
    non_target_gt = [box for box in record["boxes"] if box not in gt_boxes]
    result = metrics.match_image(predictions, gt_boxes, iou_threshold)
    matched_predictions = {id(pred): iou for pred, _, iou in result["matches"]}
    matched_gt = {id(gt) for _, gt, _ in result["matches"]}

    gt_canvas, pred_canvas = base.copy(), base.copy()
    gt_draw, pred_draw = ImageDraw.Draw(gt_canvas), ImageDraw.Draw(pred_canvas)
    line = max(2, base.width // 450)
    font = _font(max(12, base.width // 75))

    for gt in non_target_gt:
        color = "#5f6368"
        _draw_box(gt_draw, gt, scale_x, scale_y, color, line)
        _label(gt_draw, (gt["x0"] * scale_x, max(0, gt["y0"] * scale_y - 20)),
               f"GT non-target: {gt.get('class_name', '?')}", color, font, base.width)
    for gt in gt_boxes:
        status = "MATCHED" if id(gt) in matched_gt else "FN"
        color = "#1976d2" if status == "MATCHED" else "#f57c00"
        _draw_box(gt_draw, gt, scale_x, scale_y, color, line)
        _label(gt_draw, (gt["x0"] * scale_x, max(0, gt["y0"] * scale_y - 20)),
               f"GT: {gt.get('class_name', '?')} | {status}", color, font, base.width)

    for pred in result["preds_sorted"]:
        is_tp = id(pred) in matched_predictions
        if is_tp:
            status, color = f"TP IoU={matched_predictions[id(pred)]:.2f}", "#188038"
        elif _duplicate_prediction(pred, result, gt_boxes, iou_threshold):
            status, color = "FP duplicate", "#c62828"
        else:
            status, color = "FP", "#c62828"
        _draw_box(pred_draw, pred, scale_x, scale_y, color, line)
        label = str(pred.get("predicted_label") or pred.get("prompt") or "prediction")
        if pred.get("has_confidence", True):
            confidence = f"conf={float(pred.get('score', 0.0)):.2f}"
        else:
            confidence = "conf=N/A"
        _label(pred_draw, (pred["x0"] * scale_x, max(0, pred["y0"] * scale_y - 20)),
               f"Pred: {label} | {confidence} | {status}", color, font, base.width)

    header_height = max(58, base.height // 12)
    sheet = Image.new("RGB", (base.width * 2, base.height + header_height), "#202124")
    header = ImageDraw.Draw(sheet)
    header_font = _font(max(14, base.width // 55))
    header.text((8, 5), f"{model} | {prompt}", fill="#f1f3f4", font=header_font)
    column_y = 31
    header.text((8, column_y), "TARGET GT (blue: matched; orange: FN; grey: non-target)", fill="white", font=font)
    header.text((base.width + 8, column_y), "PREDICTIONS (green: TP; red: FP)", fill="white", font=font)
    sheet.paste(gt_canvas, (0, header_height))
    sheet.paste(pred_canvas, (base.width, header_height))
    return sheet, {key: result[key] for key in ("tp", "fp", "fn", "duplicates")}


def save_visualizations(run_dir: Path, gt: dict, predictions: list[dict],
                        model_labels: list[str], prompts: list[str | dict],
                        iou_threshold: float, max_images: int | None = None) -> list[dict]:
    """Write/update representative sheets and the incrementally browsable index."""
    from prompt_runner import prediction_boxes

    root = run_dir / "visualizations"
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for model in model_labels:
        for prompt_spec in prompts:
            if isinstance(prompt_spec, dict):
                prompt = prompt_spec["prompt"]
                prompt_id = prompt_spec.get("id", "")
                target_classes = set(prompt_spec.get("target_classes") or [])
            else:
                prompt = prompt_spec
                prompt_id = ""
                target_classes = None
            by_image = prediction_boxes(predictions, model, prompt)
            records = _select_records(
                gt["records"], by_image, target_classes, iou_threshold, max_images,
            )
            for record in records:
                sheet, counts = _make_sheet(
                    record, by_image.get(record["image_id"], []), model, prompt,
                    iou_threshold, target_classes,
                )
                filename = "__".join((
                    _slug(gt["dataset"]), _slug(gt["split"]), _slug(Path(record["image_id"]).stem),
                    _slug(model), _slug(prompt_id or prompt),
                )) + ".jpg"
                relative = Path(_slug(model)) / _slug(prompt_id or prompt) / filename
                output = root / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                sheet.save(output, quality=88, optimize=True)
                sheet.close()
                rows.append({
                    "dataset": gt["dataset"], "split": gt["split"], "image_id": record["image_id"],
                    "image_path": str(record["image_path"]), "model": model,
                    "prompt_id": prompt_id, "prompt": prompt,
                    "target_classes": " | ".join(sorted(target_classes or [])),
                    "visualization": relative.as_posix(), **counts,
                })

    csv_path = root / "visualization_index.csv"
    # A dissertation run calls this after every completed combination. Preserve
    # prior combinations while replacing the current combination's rows, so
    # index.html is useful during a long-running evaluation and on resume.
    if csv_path.is_file() and csv_path.stat().st_size:
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
        replaced = {(row["model"], row.get("prompt_id", ""), row["prompt"]) for row in rows}
        rows = [row for row in existing
                if (row.get("model", ""), row.get("prompt_id", ""), row.get("prompt", ""))
                not in replaced] + rows
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["dataset", "split", "image_id", "image_path", "model", "prompt_id",
                  "prompt", "target_classes", "visualization", "tp", "fp", "fn", "duplicates"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    cards = "\n".join(
        f'<article><a href="{html.escape(row["visualization"])}"><img src="{html.escape(row["visualization"])}" loading="lazy"></a>'
        f'<p><b>{html.escape(row["model"])}</b> | {html.escape(row["prompt_id"])} | prompt: {html.escape(row["prompt"])}<br>'
        f'{html.escape(row["image_id"])} | TP {row["tp"]}, FP {row["fp"]}, FN {row["fn"]}, duplicates {row["duplicates"]}</p></article>'
        for row in rows
    )
    (root / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>PromptDetect comparisons</title>"
        "<style>body{font-family:system-ui;margin:20px;background:#f4f5f7}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:16px}article{background:white;padding:10px;border-radius:8px;box-shadow:0 1px 3px #bbb}img{width:100%;height:auto}p{margin:7px 0 0}</style>"
        "</head><body><h1>PromptDetect: targeted ground truth vs predictions</h1><p>Deterministic, outcome-stratified examples per model/prompt. Left: target GT (blue matched, orange FN) and non-target GT (grey). Right: predictions (green TP, red FP).</p><main>"
        + cards + "</main></body></html>\n", encoding="utf-8"
    )
    return rows
