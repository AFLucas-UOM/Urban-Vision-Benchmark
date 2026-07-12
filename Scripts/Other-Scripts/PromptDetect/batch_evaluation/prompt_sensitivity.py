"""Controlled prompt-sensitivity analysis for the PromptDetect dissertation pipeline.

A *sensitivity family* is a set of semantically equivalent prompt wordings
(one ``canonical`` plus paraphrase/instruction variants) that share exactly the
same ``target_classes``. Because the target ground truth is identical inside a
family, any metric difference between its prompts is attributable to wording
alone. This is distinct from:

* synonym comparison (alternative class names that may be semantically
  ambiguous, e.g. "no entry sign" vs "one way sign"), and
* prompt granularity (narrow class prompts vs broad superclass prompts),

both of which stay in the main dissertation protocol and are never mixed into
the paraphrase-sensitivity statistics computed here.

Two complementary analyses are provided:

1. **GT-based sensitivity** — spread (mean / population std / min / max /
   range) of the targeted detection metrics across the prompts of a family.
2. **Prediction consistency** — pairwise agreement between the raw prediction
   sets of two prompts in the same family (greedy IoU matching). This does NOT
   use ground truth and therefore measures output stability, not correctness.

All functions return full-precision values; rounding happens only at the
output boundary (see ``sensitivity_reporting``).
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from statistics import mean, median, pstdev
from typing import Any

from metrics import match_image

SENSITIVITY_GROUP = "paraphrase-sensitivity"
VARIANT_TYPES = ("canonical", "lexical-paraphrase", "descriptive-paraphrase", "instruction")
FAMILY_SIZE = 4
DEFAULT_CONSISTENCY_IOU = 0.5
SPREAD_METRICS = ("precision", "recall", "f1", "mean_matched_iou")


# --- protocol validation --------------------------------------------------------


def is_sensitivity_prompt(prompt: dict[str, Any]) -> bool:
    return (prompt.get("group") == SENSITIVITY_GROUP
            or "sensitivity_family" in prompt or "variant_type" in prompt)


def has_sensitivity_prompts(prompts: list[dict[str, Any]]) -> bool:
    return any(is_sensitivity_prompt(row) for row in prompts)


def validate_sensitivity_families(dataset: str, spec: dict[str, Any]) -> None:
    """Fail fast on malformed sensitivity families; no-op for classic protocols."""
    vocabulary = spec.get("class_vocabulary")
    families: dict[str, list[dict]] = defaultdict(list)
    for prompt in spec.get("prompts", []):
        if not is_sensitivity_prompt(prompt):
            continue
        for key in ("sensitivity_family", "variant_type"):
            if not prompt.get(key):
                raise ValueError(f"{dataset} prompt {prompt.get('id')!r} is missing sensitivity metadata: {key!r}")
        if prompt.get("group") != SENSITIVITY_GROUP:
            raise ValueError(f"{dataset} prompt {prompt['id']!r} carries sensitivity metadata but group "
                             f"{prompt.get('group')!r} != {SENSITIVITY_GROUP!r}")
        if prompt["variant_type"] not in VARIANT_TYPES:
            raise ValueError(f"{dataset} prompt {prompt['id']!r} has unknown variant_type "
                             f"{prompt['variant_type']!r}; expected one of {VARIANT_TYPES}")
        if vocabulary is not None:
            unknown = set(prompt.get("target_classes", [])) - set(vocabulary)
            if unknown:
                raise ValueError(f"{dataset} prompt {prompt['id']!r} has unknown target classes "
                                 f"{sorted(unknown)}; vocabulary={list(vocabulary)}")
        families[prompt["sensitivity_family"]].append(prompt)
    for family, prompts in sorted(families.items()):
        if len(prompts) != FAMILY_SIZE:
            raise ValueError(f"{dataset} sensitivity family {family!r} has {len(prompts)} prompts; "
                             f"exactly {FAMILY_SIZE} are required")
        texts = [str(row["prompt"]).strip().casefold() for row in prompts]
        if len(set(texts)) != len(texts):
            raise ValueError(f"{dataset} sensitivity family {family!r} contains duplicate prompt text")
        canonical = [row for row in prompts if row["variant_type"] == "canonical"]
        if len(canonical) != 1:
            raise ValueError(f"{dataset} sensitivity family {family!r} must have exactly one canonical "
                             f"prompt; found {len(canonical)}")
        target_sets = {tuple(row.get("target_classes", [])) for row in prompts}
        if len(target_sets) != 1:
            raise ValueError(f"{dataset} sensitivity family {family!r} mixes different target_classes: "
                             f"{sorted(target_sets)}")


# --- ground-truth consistency inside a family ------------------------------------


def _gt_count(row: dict[str, Any], *keys: str) -> int:
    for key in keys:
        if key in row and row[key] not in ("", None):
            return int(row[key])
    raise KeyError(f"None of {keys} present in row {row.get('prompt_id')!r}")


def validate_family_gt_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check every prompt in a family sees identical target GT; return family summary.

    ``rows`` are per-prompt rows carrying ``sensitivity_family``, ``target_classes``
    and the GT counts (``positive_images``, ``negative_images`` and
    ``target_boxes`` / ``target_gt_boxes``).
    """
    families: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("sensitivity_family"):
            families[row["sensitivity_family"]].append(row)
    summary = []
    for family, members in sorted(families.items()):
        counts = {(int(row["positive_images"]), int(row["negative_images"]),
                   _gt_count(row, "target_boxes", "target_gt_boxes")) for row in members}
        if len(counts) != 1:
            detail = {row["prompt_id"]: (row["positive_images"], row["negative_images"],
                                         _gt_count(row, "target_boxes", "target_gt_boxes")) for row in members}
            raise ValueError(f"Sensitivity family {family!r} does not share identical target GT "
                             f"across its prompts: {detail}")
        positive, negative, boxes = counts.pop()
        summary.append({"sensitivity_family": family, "target_classes": members[0]["target_classes"],
                        "n_prompts": len(members), "positive_images": positive,
                        "negative_images": negative, "target_gt_boxes": boxes})
    return summary


# --- GT-based sensitivity aggregation ---------------------------------------------


def _spread(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "std": pstdev(values), "min": min(values),
            "max": max(values), "range": max(values) - min(values)}


def relative_degradation(max_f1: float, min_f1: float) -> tuple[float, bool]:
    """(max-min)/max, guarded: a family where every variant failed returns (0.0, True)."""
    if max_f1 == 0:
        return 0.0, True
    return (max_f1 - min_f1) / max_f1, False


def sensitivity_rows(per_prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted((row for row in per_prompt if row.get("sensitivity_family")),
                  key=lambda row: (row["dataset"], row["model"], row["sensitivity_family"], row["prompt_id"]))


def family_statistics(per_prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per dataset x model x sensitivity_family with spread and degradation stats."""
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in sensitivity_rows(per_prompt):
        groups[(row["dataset"], row["model"], row["sensitivity_family"])].append(row)
    out = []
    for (dataset, model, family), members in sorted(groups.items()):
        members = sorted(members, key=lambda row: row["prompt_id"])  # deterministic tie-breaking
        stats: dict[str, Any] = {"dataset": dataset, "model": model, "sensitivity_family": family,
                                 "target_classes": members[0]["target_classes"], "n_prompts": len(members),
                                 "target_gt_boxes": members[0].get("target_gt_boxes"),
                                 "positive_images": members[0].get("positive_images"),
                                 "negative_images": members[0].get("negative_images")}
        for metric in SPREAD_METRICS:
            for name, value in _spread([float(row.get(metric) or 0.0) for row in members]).items():
                stats[f"{metric}_{name}"] = value
        best = max(members, key=lambda row: float(row.get("f1") or 0.0))
        worst = min(members, key=lambda row: float(row.get("f1") or 0.0))
        rel, all_failed = relative_degradation(float(best.get("f1") or 0.0), float(worst.get("f1") or 0.0))
        canonical = [row for row in members if row.get("variant_type") == "canonical"]
        instruction = [row for row in members if row.get("variant_type") == "instruction"]
        stats.update(
            best_prompt_id=best["prompt_id"], best_prompt=best["prompt"],
            best_variant_type=best.get("variant_type"),
            worst_prompt_id=worst["prompt_id"], worst_prompt=worst["prompt"],
            worst_variant_type=worst.get("variant_type"),
            f1_degradation_abs=float(best.get("f1") or 0.0) - float(worst.get("f1") or 0.0),
            f1_degradation_rel=rel, all_variants_failed=all_failed,
            canonical_f1=float(canonical[0].get("f1") or 0.0) if canonical else None,
            instruction_f1=float(instruction[0].get("f1") or 0.0) if instruction else None,
        )
        out.append(stats)
    return out


def variant_difference_rows(per_prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Canonical-vs-variant F1 differences (negative = variant worse than canonical)."""
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in sensitivity_rows(per_prompt):
        groups[(row["dataset"], row["model"], row["sensitivity_family"])].append(row)
    out = []
    for (dataset, model, family), members in sorted(groups.items()):
        canonical = next((row for row in members if row.get("variant_type") == "canonical"), None)
        if canonical is None:
            continue
        canonical_f1 = float(canonical.get("f1") or 0.0)
        for row in sorted(members, key=lambda item: item["prompt_id"]):
            if row.get("variant_type") == "canonical":
                continue
            variant_f1 = float(row.get("f1") or 0.0)
            out.append({"dataset": dataset, "model": model, "sensitivity_family": family,
                        "canonical_prompt_id": canonical["prompt_id"], "canonical_f1": canonical_f1,
                        "prompt_id": row["prompt_id"], "prompt": row["prompt"],
                        "variant_type": row.get("variant_type"), "variant_f1": variant_f1,
                        "canonical_f1_difference": variant_f1 - canonical_f1})
    return out


def model_macro_rows(family_stats: list[dict[str, Any]],
                     per_prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Macro sensitivity summary per dataset x model across its families."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in family_stats:
        groups[(row["dataset"], row["model"])].append(row)
    inference: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in sensitivity_rows(per_prompt):
        if row.get("mean_inference_ms") not in (None, ""):
            inference[(row["dataset"], row["model"])].append(float(row["mean_inference_ms"]))
    out = []
    for (dataset, model), families in sorted(groups.items()):
        families = sorted(families, key=lambda row: row["sensitivity_family"])
        worst = max(families, key=lambda row: row["f1_degradation_rel"])
        canonical = [row["canonical_f1"] for row in families if row["canonical_f1"] is not None]
        instruction = [row["instruction_f1"] for row in families if row["instruction_f1"] is not None]
        timings = inference.get((dataset, model), [])
        out.append({
            "dataset": dataset, "model": model, "n_families": len(families),
            "macro_mean_f1": mean(row["f1_mean"] for row in families),
            "mean_within_family_f1_std": mean(row["f1_std"] for row in families),
            "median_within_family_f1_std": median(row["f1_std"] for row in families),
            "mean_f1_range": mean(row["f1_range"] for row in families),
            "max_f1_range": max(row["f1_range"] for row in families),
            "mean_relative_f1_degradation": mean(row["f1_degradation_rel"] for row in families),
            "worst_family": worst["sensitivity_family"],
            "worst_family_relative_degradation": worst["f1_degradation_rel"],
            "macro_worst_prompt_f1": mean(row["f1_min"] for row in families),
            "macro_canonical_f1": mean(canonical) if canonical else None,
            "macro_instruction_f1": mean(instruction) if instruction else None,
            "mean_matched_iou_variability": mean(row["mean_matched_iou_std"] for row in families),
            "mean_inference_ms": mean(timings) if timings else None,
        })
    return out


# --- prediction-pair consistency (no ground truth involved) ------------------------


def pair_consistency(boxes_a: list[dict], boxes_b: list[dict], iou_threshold: float) -> dict[str, Any]:
    """Agreement between two prediction sets on one image (greedy IoU matching).

    Both-empty pairs are fully consistent by definition (F1 = Jaccard = 1.0).
    """
    n_a, n_b = len(boxes_a), len(boxes_b)
    if n_a == 0 and n_b == 0:
        return {"n_boxes_a": 0, "n_boxes_b": 0, "matched": 0,
                "pairwise_box_f1": 1.0, "pairwise_box_jaccard": 1.0,
                "mean_matched_pair_iou": None, "matched_iou_sum": 0.0,
                "unmatched_a": 0, "unmatched_b": 0, "both_empty": True}
    result = match_image(boxes_a, boxes_b, iou_threshold)
    matched = result["tp"]
    ious = [iou for _, _, iou in result["matches"]]
    return {"n_boxes_a": n_a, "n_boxes_b": n_b, "matched": matched,
            "pairwise_box_f1": 2 * matched / (n_a + n_b),
            "pairwise_box_jaccard": matched / (n_a + n_b - matched),
            "mean_matched_pair_iou": sum(ious) / len(ious) if ious else None,
            "matched_iou_sum": sum(ious),
            "unmatched_a": n_a - matched, "unmatched_b": n_b - matched, "both_empty": False}


def pairwise_image_rows(prompt_definitions: list[dict[str, Any]],
                        predictions_by_prompt: dict[str, dict[str, list[dict]]],
                        image_ids: list[str], model: str, dataset: str,
                        iou_threshold: float = DEFAULT_CONSISTENCY_IOU) -> list[dict[str, Any]]:
    """Per-image consistency rows for every prompt pair of every sensitivity family.

    ``predictions_by_prompt`` maps prompt_id -> image_id -> prediction boxes for
    one model; images with no predictions must simply be absent.
    """
    families: dict[str, list[dict]] = defaultdict(list)
    for prompt in prompt_definitions:
        if prompt.get("sensitivity_family"):
            families[prompt["sensitivity_family"]].append(prompt)
    rows = []
    for family, prompts in sorted(families.items()):
        ordered = sorted(prompts, key=lambda row: row["id"])
        for first, second in combinations(ordered, 2):
            preds_a = predictions_by_prompt.get(first["id"], {})
            preds_b = predictions_by_prompt.get(second["id"], {})
            for image_id in sorted(image_ids):
                comparison = pair_consistency(preds_a.get(image_id, []), preds_b.get(image_id, []),
                                              iou_threshold)
                rows.append({"dataset": dataset, "model": model, "sensitivity_family": family,
                             "prompt_id_a": first["id"], "prompt_id_b": second["id"],
                             "variant_type_a": first.get("variant_type"),
                             "variant_type_b": second.get("variant_type"),
                             "image_id": image_id, **comparison})
    return rows


def aggregate_pair_consistency(image_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per dataset x model x family x prompt pair (macro mean over images)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in image_rows:
        groups[(row["dataset"], row["model"], row["sensitivity_family"],
                row["prompt_id_a"], row["prompt_id_b"])].append(row)
    out = []
    for (dataset, model, family, id_a, id_b), members in sorted(groups.items()):
        matched = sum(row["matched"] for row in members)
        iou_sum = sum(row["matched_iou_sum"] for row in members)
        out.append({"dataset": dataset, "model": model, "sensitivity_family": family,
                    "prompt_id_a": id_a, "prompt_id_b": id_b,
                    "variant_type_a": members[0]["variant_type_a"],
                    "variant_type_b": members[0]["variant_type_b"],
                    "n_images": len(members),
                    "n_both_empty": sum(row["both_empty"] for row in members),
                    "mean_pairwise_box_f1": mean(row["pairwise_box_f1"] for row in members),
                    "mean_pairwise_box_jaccard": mean(row["pairwise_box_jaccard"] for row in members),
                    "mean_matched_pair_iou": iou_sum / matched if matched else None,
                    "total_matched": matched,
                    "total_unmatched_a": sum(row["unmatched_a"] for row in members),
                    "total_unmatched_b": sum(row["unmatched_b"] for row in members)})
    return out


def aggregate_family_consistency(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Macro mean over prompt pairs, per dataset x model x sensitivity family."""
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in pair_rows:
        groups[(row["dataset"], row["model"], row["sensitivity_family"])].append(row)
    out = []
    for (dataset, model, family), members in sorted(groups.items()):
        ious = [row["mean_matched_pair_iou"] for row in members if row["mean_matched_pair_iou"] is not None]
        out.append({"dataset": dataset, "model": model, "sensitivity_family": family,
                    "n_pairs": len(members),
                    "mean_pairwise_box_f1": mean(row["mean_pairwise_box_f1"] for row in members),
                    "mean_pairwise_box_jaccard": mean(row["mean_pairwise_box_jaccard"] for row in members),
                    "mean_matched_pair_iou": mean(ious) if ious else None,
                    "min_pairwise_box_f1": min(row["mean_pairwise_box_f1"] for row in members),
                    "max_pairwise_box_f1": max(row["mean_pairwise_box_f1"] for row in members)})
    return out


def aggregate_model_consistency(family_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Macro mean over families, per dataset x model."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in family_rows:
        groups[(row["dataset"], row["model"])].append(row)
    out = []
    for (dataset, model), members in sorted(groups.items()):
        ious = [row["mean_matched_pair_iou"] for row in members if row["mean_matched_pair_iou"] is not None]
        out.append({"dataset": dataset, "model": model, "n_families": len(members),
                    "macro_pairwise_box_f1": mean(row["mean_pairwise_box_f1"] for row in members),
                    "macro_pairwise_box_jaccard": mean(row["mean_pairwise_box_jaccard"] for row in members),
                    "macro_matched_pair_iou": mean(ious) if ious else None,
                    "least_consistent_family": min(members, key=lambda row: row["mean_pairwise_box_f1"])["sensitivity_family"],
                    "min_family_pairwise_box_f1": min(row["mean_pairwise_box_f1"] for row in members)})
    return out


def predictions_by_prompt_for_model(prediction_rows: list[dict[str, Any]], model: str) -> dict[str, dict[str, list[dict]]]:
    """Group raw prediction rows (in-memory or read back from CSV) for one model.

    Zero-detection marker rows (empty ``x0``) are skipped; numeric fields are
    coerced so CSV-sourced strings behave identically to in-memory floats.
    """
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in prediction_rows:
        if row.get("model") != model or row.get("x0") in ("", None) or not row.get("prompt_id"):
            continue
        grouped[str(row["prompt_id"])][str(row["image_id"])].append({
            "x0": float(row["x0"]), "y0": float(row["y0"]),
            "x1": float(row["x1"]), "y1": float(row["y1"]),
            "score": float(row["score"]) if row.get("score") not in ("", None) else 0.0,
        })
    return {prompt_id: dict(images) for prompt_id, images in grouped.items()}
