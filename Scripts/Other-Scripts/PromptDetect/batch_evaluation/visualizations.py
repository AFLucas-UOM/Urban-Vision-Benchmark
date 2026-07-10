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


def _make_sheet(record: dict, predictions: list[dict], model: str, prompt: str,
                iou_threshold: float) -> tuple[Image.Image, dict]:
    """Return one side-by-side GT / prediction image and its match counts."""
    with Image.open(record["image_path"]) as source:
        base = source.convert("RGB")
    gt_boxes = record["boxes"]
    result = metrics.match_image(predictions, gt_boxes, iou_threshold)
    matched_predictions = {id(pred): iou for pred, _, iou in result["matches"]}
    matched_gt = {id(gt) for _, gt, _ in result["matches"]}

    gt_canvas, pred_canvas = base.copy(), base.copy()
    gt_draw, pred_draw = ImageDraw.Draw(gt_canvas), ImageDraw.Draw(pred_canvas)
    line = max(2, base.width // 450)
    font = _font(max(12, base.width // 75))

    for gt in gt_boxes:
        status = "MATCHED" if id(gt) in matched_gt else "FN"
        color = "#1976d2" if status == "MATCHED" else "#f57c00"
        gt_draw.rectangle((gt["x0"], gt["y0"], gt["x1"], gt["y1"]), outline=color, width=line)
        _label(gt_draw, (gt["x0"], max(0, gt["y0"] - 20)),
               f"GT: {gt.get('class_name', '?')} | {status}", color, font, base.width)

    for pred in result["preds_sorted"]:
        is_tp = id(pred) in matched_predictions
        if is_tp:
            status, color = f"TP IoU={matched_predictions[id(pred)]:.2f}", "#188038"
        elif _duplicate_prediction(pred, result, gt_boxes, iou_threshold):
            status, color = "FP duplicate", "#c62828"
        else:
            status, color = "FP", "#c62828"
        pred_draw.rectangle((pred["x0"], pred["y0"], pred["x1"], pred["y1"]), outline=color, width=line)
        label = str(pred.get("predicted_label") or pred.get("prompt") or "prediction")
        if pred.get("has_confidence", True):
            confidence = f"conf={float(pred.get('score', 0.0)):.2f}"
        else:
            confidence = "conf=N/A"
        _label(pred_draw, (pred["x0"], max(0, pred["y0"] - 20)),
               f"Pred: {label} | {confidence} | {status}", color, font, base.width)

    header_height = max(34, base.height // 18)
    sheet = Image.new("RGB", (base.width * 2, base.height + header_height), "#202124")
    header = ImageDraw.Draw(sheet)
    header_font = _font(max(14, base.width // 55))
    header.text((8, 7), "GROUND TRUTH  (blue: matched; orange: false negative)", fill="white", font=header_font)
    header.text((base.width + 8, 7), "PREDICTIONS  (green: TP; red: false positive)", fill="white", font=header_font)
    sheet.paste(gt_canvas, (0, header_height))
    sheet.paste(pred_canvas, (base.width, header_height))
    return sheet, {key: result[key] for key in ("tp", "fp", "fn", "duplicates")}


def save_visualizations(run_dir: Path, gt: dict, predictions: list[dict],
                        model_labels: list[str], prompts: list[str],
                        iou_threshold: float, max_images: int | None = None) -> list[dict]:
    """Write comparison sheets plus CSV and browser-friendly HTML index."""
    from prompt_runner import prediction_boxes

    root = run_dir / "visualizations"
    root.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    records = gt["records"] if max_images is None else gt["records"][:max_images]
    for model in model_labels:
        for prompt in prompts:
            by_image = prediction_boxes(predictions, model, prompt)
            for record in records:
                sheet, counts = _make_sheet(record, by_image.get(record["image_id"], []), model, prompt, iou_threshold)
                filename = "__".join((
                    _slug(gt["dataset"]), _slug(gt["split"]), _slug(Path(record["image_id"]).stem),
                    _slug(model), _slug(prompt),
                )) + ".jpg"
                relative = Path(_slug(model)) / _slug(prompt) / filename
                output = root / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                sheet.save(output, quality=92)
                rows.append({
                    "dataset": gt["dataset"], "split": gt["split"], "image_id": record["image_id"],
                    "image_path": str(record["image_path"]), "model": model, "prompt": prompt,
                    "visualization": relative.as_posix(), **counts,
                })

    csv_path = root / "visualization_index.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["dataset", "split", "image_id", "image_path", "model", "prompt", "visualization", "tp", "fp", "fn", "duplicates"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    cards = "\n".join(
        f'<article><a href="{html.escape(row["visualization"])}"><img src="{html.escape(row["visualization"])}" loading="lazy"></a>'
        f'<p><b>{html.escape(row["model"])}</b> | prompt: {html.escape(row["prompt"])}<br>'
        f'{html.escape(row["image_id"])} | TP {row["tp"]}, FP {row["fp"]}, FN {row["fn"]}, duplicates {row["duplicates"]}</p></article>'
        for row in rows
    )
    (root / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>PromptDetect comparisons</title>"
        "<style>body{font-family:system-ui;margin:20px;background:#f4f5f7}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:16px}article{background:white;padding:10px;border-radius:8px;box-shadow:0 1px 3px #bbb}img{width:100%;height:auto}p{margin:7px 0 0}</style>"
        "</head><body><h1>PromptDetect: ground truth vs predictions</h1><p>Left: GT (blue matched, orange FN). Right: predictions (green TP, red FP).</p><main>"
        + cards + "</main></body></html>\n", encoding="utf-8"
    )
    return rows
