#!/usr/bin/env python3
"""Bootstrap confidence intervals and paired comparisons over stored outcomes.

Computes seeded percentile-bootstrap uncertainty for the repository's stored
sample-level results. It never reruns inference and never derives uncertainty
from a single aggregate number: when sample-level inputs do not exist the
analysis is reported as PENDING with the command that would produce them.

Supported analyses and their resampling units:
  * MTSD attribute classification - per-head accuracy / macro-F1 / per-class
    F1 CIs. Per-sample (true, pred) outcomes are reconstructed exactly from
    the stored test confusion matrices (each crop contributes one cell), so
    the resampling unit is the test crop. Cross-head mean macro-F1 CIs and
    paired variant comparisons need per-crop prediction dumps that the
    pipeline does not store - reported as missing. Results carry the
    historical GRP-1..GRP-3 snapshot label.
  * PromptDetect batch evaluation - image-level CIs for precision / recall /
    F1 / matched IoU / FP-per-image / FN-per-image per (model, prompt), and
    paired model or prompt comparisons on the same images. Current runs are
    pilots and are labelled so.
  * MDWD supervised detection - image-level CIs for precision / recall / F1 /
    matched IoU per stored checkpoint evaluation (leakage-analysis
    predictions over the full official valid/test splits), plus explicit
    paired model comparisons via --model-a/--model-b. Confidence-ranked AP
    intervals are deliberately not produced here (see missing_inputs.md).

Method: percentile intervals over --n-bootstrap resamples (default 2000),
fixed --seed, paired resampling for comparisons (identical resampled indices
feed both sides). "Significance" is never declared; paired reports state the
exact criterion (whether the CI excludes zero) and the proportion of
resamples favouring A.

Outputs (timestamped; --overwrite writes to .../latest/):
    Documents/Final-Reports/Statistical-Uncertainty/<stamp>/
        bootstrap_summary.md   bootstrap_results.csv
        bootstrap_results.json bootstrap_config.json  missing_inputs.md
    Documents/Final-Figures/Statistical-Uncertainty/<stamp>/  (PNG + PDF + SVG)

Usage:
    python Scripts/FinalEvaluation/bootstrap_uncertainty.py --self-test
    python Scripts/FinalEvaluation/bootstrap_uncertainty.py --dry-run
    python Scripts/FinalEvaluation/bootstrap_uncertainty.py
    python Scripts/FinalEvaluation/bootstrap_uncertainty.py --dataset PromptDetect-MDWD `
        --model-a "SAM 3.1" --model-b "Cosmos Reason2 2B"
    python Scripts/FinalEvaluation/bootstrap_uncertainty.py --dataset MDWD `
        --model-a yolo26l@YOLO26-EUVIP --model-b yolo12m@YOLO12-EUVIP
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evidence_lib as ev  # noqa: E402

GENERATOR = "Scripts/FinalEvaluation/bootstrap_uncertainty.py"
MIN_CLASS_SUPPORT = 20  # per-class F1 CIs need at least this many samples

CSV_FIELDS = ["workstream", "status", "analysis", "model", "comparison", "run",
              "split", "metric", "point", "ci_low", "ci_high",
              "observed_difference", "p_a_better", "n_units", "unit",
              "n_bootstrap", "confidence_level", "method", "source_file", "notes"]


# ---------------------------------------------------------------------------
# Metric functions over resampling units
# ---------------------------------------------------------------------------

def accuracy_of(samples: list[tuple[int, int]]) -> float:
    return sum(1 for t, p in samples if t == p) / len(samples) if samples else 0.0


def macro_f1_of(samples: list[tuple[int, int]], n_classes: int) -> float:
    f1_values = []
    for class_index in range(n_classes):
        tp = sum(1 for t, p in samples if t == class_index and p == class_index)
        fp = sum(1 for t, p in samples if t != class_index and p == class_index)
        fn = sum(1 for t, p in samples if t == class_index and p != class_index)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall)
                         if precision + recall else 0.0)
    return sum(f1_values) / n_classes if n_classes else 0.0


def single_class_f1(samples: list[tuple[int, int]], class_index: int) -> float:
    tp = sum(1 for t, p in samples if t == class_index and p == class_index)
    fp = sum(1 for t, p in samples if t != class_index and p == class_index)
    fn = sum(1 for t, p in samples if t == class_index and p != class_index)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


# Detection/PromptDetect units: (tp, fp, fn, iou_sum, n_matched) per image.

def micro_metric(units: list[tuple], name: str) -> float:
    tp = sum(u[0] for u in units)
    fp = sum(u[1] for u in units)
    fn = sum(u[2] for u in units)
    if name == "precision":
        return tp / (tp + fp) if tp + fp else 0.0
    if name == "recall":
        return tp / (tp + fn) if tp + fn else 0.0
    if name == "f1":
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if name == "mean_matched_iou":
        matched = sum(u[4] for u in units)
        return sum(u[3] for u in units) / matched if matched else 0.0
    if name == "fp_per_image":
        return fp / len(units) if units else 0.0
    if name == "fn_per_image":
        return fn / len(units) if units else 0.0
    raise ValueError(name)


def image_units_from_matches(pd_metrics, preds_by_image: dict, gt_by_image: dict,
                             iou_threshold: float, class_aware: bool) -> dict[str, tuple]:
    units = {}
    for image_id, gts in gt_by_image.items():
        result = ev.match_detections(pd_metrics, preds_by_image.get(image_id, []),
                                     gts, iou_threshold, class_aware=class_aware)
        ious = [g["iou"] for g in result["gt_records"] if g["matched"]]
        units[image_id] = (result["tp"], result["fp"], result["fn"],
                           sum(ious), len(ious))
    return units


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

def ci_row(*, workstream, status, model, run, split, metric, ci, unit, source,
           notes="") -> dict:
    return {"workstream": workstream, "status": status, "analysis": "ci",
            "model": model, "comparison": "", "run": run, "split": split,
            "metric": metric, "point": ci["point"], "ci_low": ci["low"],
            "ci_high": ci["high"], "observed_difference": "", "p_a_better": "",
            "n_units": ci["n_units"], "unit": unit,
            "n_bootstrap": ci["n_bootstrap"],
            "confidence_level": ci.get("confidence_level", ""),
            "method": ci["method"], "source_file": source, "notes": notes}


def paired_row(*, workstream, status, comparison, run, split, metric, result,
               unit, source, notes="") -> dict:
    excludes_zero = result["low"] > 0 or result["high"] < 0
    interpretation = (
        f"CI {'excludes' if excludes_zero else 'includes'} zero at "
        f"{result.get('confidence_level', 0.95):.0%}; "
        f"A better in {result['p_a_better']:.1%} of resamples")
    return {"workstream": workstream, "status": status, "analysis": "paired",
            "model": "", "comparison": comparison, "run": run, "split": split,
            "metric": metric, "point": "", "ci_low": result["low"],
            "ci_high": result["high"],
            "observed_difference": result["observed_difference"],
            "p_a_better": result["p_a_better"], "n_units": result["n_units"],
            "unit": unit, "n_bootstrap": result["n_bootstrap"],
            "confidence_level": result.get("confidence_level", ""),
            "method": result["method"], "source_file": source,
            "notes": (notes + "; " if notes else "") + interpretation}


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def analyse_attributes(args, missing: list[str]) -> list[dict]:
    variants = ev.discover_attribute_variants(include_smoke=False)
    if not variants:
        missing.append("MTSD attributes: no completed variants with stored test "
                       "metrics - run the attribute pipeline first.")
        return []
    rows = []
    for variant in variants:
        payload = variant["payload"]
        source = ev.rel(variant["metrics_path"])
        run_id = payload.get("run_id", variant["variant"])
        for head, head_payload in payload.get("attributes", {}).items():
            confusion = head_payload.get("confusion_matrix")
            classes = head_payload.get("classes", [])
            if not confusion or not classes:
                missing.append(f"MTSD attributes {variant['variant']}/{head}: no "
                               "stored confusion matrix - CI skipped.")
                continue
            samples = ev.confusion_to_samples(confusion)
            notes = ("samples reconstructed from stored confusion matrix; iid-crop "
                     "assumption, no capture-group clustering possible")
            common = dict(workstream="MTSD-attributes", status=variant["status"],
                          model=variant["variant"], run=run_id, split="test",
                          unit="test crop", source=source, notes=notes)
            rows.append(ci_row(metric=f"{head}/accuracy",
                               ci=ev.bootstrap_ci(samples, accuracy_of,
                                                  args.n_bootstrap, args.seed,
                                                  args.confidence_level), **common))
            rows.append(ci_row(metric=f"{head}/macro_f1",
                               ci=ev.bootstrap_ci(
                                   samples, lambda s: macro_f1_of(s, len(classes)),
                                   args.n_bootstrap, args.seed,
                                   args.confidence_level), **common))
            for class_index, class_name in enumerate(classes):
                support = int(head_payload.get("support", {}).get(class_name, 0))
                if support < MIN_CLASS_SUPPORT:
                    missing.append(
                        f"MTSD attributes {variant['variant']}/{head}/{class_name}: "
                        f"support {support} < {MIN_CLASS_SUPPORT} - per-class CI "
                        "skipped (insufficient support).")
                    continue
                rows.append(ci_row(
                    metric=f"{head}/f1[{class_name}]",
                    ci=ev.bootstrap_ci(
                        samples, lambda s, i=class_index: single_class_f1(s, i),
                        args.n_bootstrap, args.seed, args.confidence_level),
                    **common))
    missing.append(
        "MTSD attributes: cross-head mean macro-F1 CIs and paired variant "
        "comparisons need per-crop prediction dumps (crops aligned across heads/"
        "variants), which the training pipeline does not store. Add a per-sample "
        "prediction export to `mtsd_attr` evaluation and rerun the final-scope "
        "attribute round to enable them.")
    return rows


def analyse_promptdetect(args, pd_metrics, dataset: str, missing: list[str]) -> list[dict]:
    runs = ev.discover_promptdetect_runs(dataset)
    if args.run not in ("", "latest"):
        runs = [r for r in runs if r["run_dir"] == Path(args.run)] or runs[-1:]
    else:
        runs = runs[-1:]
    if not runs:
        missing.append(f"PromptDetect {dataset}: no batch-evaluation runs - run "
                       "run_batch_eval.py first; CIs pending.")
        return []
    run = runs[0]
    data = ev.load_promptdetect_run(run["run_dir"])
    config = data["config"]
    iou_threshold = float(config.get("iou_threshold", 0.5))
    source = ev.rel(run["run_dir"] / "predictions.csv")
    status = run["status"]
    if status == ev.STATUS_PILOT:
        missing.append(
            f"PromptDetect {dataset} run `{run['run_id']}` is a PILOT "
            f"({run['n_images']} images): intervals are extremely wide by "
            "construction and validate the pipeline only. Final CIs need the "
            "full fixed-protocol evaluation.")

    gt_boxes = {image_id: record["boxes"]
                for image_id, record in data["gt_by_image"].items()}
    pairs = sorted({(model, prompt) for (model, prompt, _) in data["predictions"]})
    units_by_pair: dict[tuple, dict[str, tuple]] = {}
    rows = []
    for model, prompt in pairs:
        preds_by_image = {image_id: data["predictions"].get((model, prompt, image_id), [])
                          for image_id in gt_boxes}
        units = image_units_from_matches(pd_metrics, preds_by_image, gt_boxes,
                                         iou_threshold, class_aware=False)
        units_by_pair[(model, prompt)] = units
        no_confidence = all(not p["has_confidence"]
                            for preds in preds_by_image.values() for p in preds) \
            and any(preds_by_image.values())
        notes = ("no per-box confidence - P/R/F1/IoU only, no AP interval"
                 if no_confidence else "")
        unit_list = list(units.values())
        for metric in ("precision", "recall", "f1", "mean_matched_iou",
                       "fp_per_image", "fn_per_image"):
            rows.append(ci_row(
                workstream=f"PromptDetect-{dataset}", status=status, model=model,
                run=f"{run['run_id']} | {prompt}", split=config.get("split", ""),
                metric=metric,
                ci=ev.bootstrap_ci(unit_list, lambda u, m=metric: micro_metric(u, m),
                                   args.n_bootstrap, args.seed, args.confidence_level),
                unit="image", source=source, notes=notes))

    # Paired comparisons on the same images -----------------------------------
    comparisons = []
    if args.model_a and args.model_b:
        shared_prompts = sorted({p for (m, p) in pairs if m == args.model_a}
                                & {p for (m, p) in pairs if m == args.model_b})
        comparisons += [((args.model_a, p), (args.model_b, p),
                         f"{args.model_a} vs {args.model_b} | prompt: {p}")
                        for p in shared_prompts]
    if args.prompt_a and args.prompt_b:
        shared_models = sorted({m for (m, p) in pairs if p == args.prompt_a}
                               & {m for (m, p) in pairs if p == args.prompt_b})
        comparisons += [((m, args.prompt_a), (m, args.prompt_b),
                         f"prompt '{args.prompt_a}' vs '{args.prompt_b}' | {m}")
                        for m in shared_models]
    if not comparisons:
        models = sorted({m for m, _ in pairs})
        shared = [p for p in {p for _, p in pairs}
                  if all((m, p) in units_by_pair for m in models)]
        if len(models) == 2 and shared:
            comparisons = [((models[0], p), (models[1], p),
                            f"{models[0]} vs {models[1]} | prompt: {p}")
                           for p in sorted(shared)]
    for key_a, key_b, label in comparisons:
        units_a, units_b = units_by_pair[key_a], units_by_pair[key_b]
        shared_images = sorted(set(units_a) & set(units_b))
        paired_units = [(units_a[i], units_b[i]) for i in shared_images]
        for metric in ("f1", "recall", "mean_matched_iou"):
            result = ev.paired_bootstrap(
                paired_units,
                lambda u, m=metric: micro_metric([x[0] for x in u], m),
                lambda u, m=metric: micro_metric([x[1] for x in u], m),
                args.n_bootstrap, args.seed, args.confidence_level)
            rows.append(paired_row(
                workstream=f"PromptDetect-{dataset}", status=status,
                comparison=label, run=run["run_id"], split=config.get("split", ""),
                metric=metric, result=result, unit="image (paired)", source=source))
    return rows


def analyse_mdwd_detection(args, pd_metrics, missing: list[str]) -> list[dict]:
    runs = ev.discover_leakage_prediction_runs()
    if not runs:
        missing.append(
            "MDWD detection: no stored per-image predictions - produce them with "
            "`python Scripts/FinalEvaluation/mdwd_leakage_sensitivity.py "
            "--run-inference --models best-per-family`; CIs pending.")
        return []
    rows = []
    gt_cache: dict[str, dict] = {}
    units_by_model: dict[tuple, dict[str, tuple]] = {}
    for run in runs:
        split = run["split"]
        if split not in gt_cache:
            gt_cache[split] = ev.load_mdwd_split_gt(split)
        gt = gt_cache[split]
        preds = ev.load_leakage_predictions(run["predictions"], gt["class_names"],
                                            min_score=args.working_conf)
        units = image_units_from_matches(
            pd_metrics, preds, {stem: r["boxes"] for stem, r in gt["records"].items()},
            iou_threshold=0.5, class_aware=True)
        model = f"{run['model']}@{run['suite']}"
        units_by_model[(model, split)] = units
        unit_list = list(units.values())
        for metric in ("precision", "recall", "f1", "mean_matched_iou"):
            rows.append(ci_row(
                workstream="MDWD-detection", status=ev.STATUS_FINAL, model=model,
                run=run["run_dir"].name, split=split, metric=metric,
                ci=ev.bootstrap_ci(unit_list, lambda u, m=metric: micro_metric(u, m),
                                   args.n_bootstrap, args.seed, args.confidence_level),
                unit="image", source=ev.rel(run["predictions"]),
                notes=f"working conf {args.working_conf}, IoU 0.5, class-aware matching"))
    if args.model_a and args.model_b:
        for split in ("valid", "test"):
            key_a, key_b = (args.model_a, split), (args.model_b, split)
            if key_a not in units_by_model or key_b not in units_by_model:
                continue
            units_a, units_b = units_by_model[key_a], units_by_model[key_b]
            shared = sorted(set(units_a) & set(units_b))
            paired_units = [(units_a[i], units_b[i]) for i in shared]
            for metric in ("f1", "recall", "mean_matched_iou"):
                result = ev.paired_bootstrap(
                    paired_units,
                    lambda u, m=metric: micro_metric([x[0] for x in u], m),
                    lambda u, m=metric: micro_metric([x[1] for x in u], m),
                    args.n_bootstrap, args.seed, args.confidence_level)
                rows.append(paired_row(
                    workstream="MDWD-detection", status=ev.STATUS_FINAL,
                    comparison=f"{args.model_a} vs {args.model_b}",
                    run="leakage-analysis stored predictions", split=split,
                    metric=metric, result=result, unit="image (paired)",
                    source=ev.rel(ev.LEAKAGE_RUNS)))
    missing.append(
        "MDWD detection: confidence-ranked AP/mAP intervals are deliberately not "
        "bootstrapped here - the official mAP protocol evaluates a ranking over "
        "the whole split, and a defensible image-level AP bootstrap over the "
        "full conf-0.001 prediction set is computationally heavy in pure Python. "
        "P/R/F1/matched-IoU intervals above use the documented operating point "
        "instead; headline mAP values remain the Ultralytics point estimates.")
    return rows


# ---------------------------------------------------------------------------
# Figures / reports
# ---------------------------------------------------------------------------

def make_figures(rows: list[dict], figures_dir: Path) -> list[str]:
    ev.apply_plot_style()
    import matplotlib.pyplot as plt

    saved = []
    for workstream in sorted({r["workstream"] for r in rows}):
        ci_rows = [r for r in rows if r["workstream"] == workstream
                   and r["analysis"] == "ci" and r["metric"].endswith(("f1", "macro_f1"))
                   and "[" not in r["metric"]]
        if not ci_rows:
            continue
        labels = [f"{r['model']} {r['split']} {r['metric']}"
                  + (f" ({r['run'].split(' | ')[-1]})" if " | " in str(r["run"]) else "")
                  for r in ci_rows]
        figure, axis = plt.subplots(
            figsize=(6.6, max(2.4, 0.34 * len(ci_rows) + 1.0)))
        y_positions = range(len(ci_rows))
        axis.errorbar([r["point"] for r in ci_rows], list(y_positions),
                      xerr=[[r["point"] - r["ci_low"] for r in ci_rows],
                            [r["ci_high"] - r["point"] for r in ci_rows]],
                      fmt="o", color="#4477aa", ecolor="#4477aa", capsize=3)
        axis.set_yticks(list(y_positions))
        axis.set_yticklabels(labels, fontsize=7)
        axis.invert_yaxis()
        axis.set_xlim(0, 1.02)
        axis.set_xlabel("Metric with 95% percentile bootstrap CI")
        axis.set_title(f"{workstream} - F1-family point estimates with CIs")
        safe = workstream.lower().replace("-", "_")
        saved.extend(ev.save_figure(figure, figures_dir, f"{safe}_ci"))

    paired = [r for r in rows if r["analysis"] == "paired" and r["metric"] == "f1"]
    if paired:
        figure, axis = plt.subplots(figsize=(6.6, max(2.2, 0.5 * len(paired) + 1.0)))
        labels = [f"{r['comparison']} ({r['workstream']}, {r['split']})" for r in paired]
        centers = [r["observed_difference"] for r in paired]
        axis.errorbar(centers, range(len(paired)),
                      xerr=[[c - r["ci_low"] for c, r in zip(centers, paired)],
                            [r["ci_high"] - c for c, r in zip(centers, paired)]],
                      fmt="s", color="#b2182b", capsize=3)
        axis.axvline(0, color="grey", linewidth=0.8)
        axis.set_yticks(range(len(paired)))
        axis.set_yticklabels(labels, fontsize=7)
        axis.invert_yaxis()
        axis.set_xlabel("F1 difference (A - B) with 95% CI")
        axis.set_title("Paired comparisons (same resampled units on both sides)")
        saved.extend(ev.save_figure(figure, figures_dir, "paired_differences"))
    return saved


def write_reports(report_dir: Path, figures: list[str], rows: list[dict],
                  missing: list[str], args) -> None:
    generated_at = ev.provenance(None, GENERATOR)["generated_at"]
    ev.write_csv(report_dir / "bootstrap_results.csv", rows, CSV_FIELDS)
    ev.write_json(report_dir / "bootstrap_results.json",
                  {"generated_at": generated_at, "commit_sha": ev.git_commit_sha(),
                   "generator": GENERATOR, "rows": rows})
    ev.write_json(report_dir / "bootstrap_config.json", {
        "generated_at": generated_at, "generator": GENERATOR,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "interval_method": "percentile (BCa not implemented - percentile only, "
                           "stated in every row)",
        "resampling_units": {"detection/PromptDetect": "image",
                             "attributes": "test crop (reconstructed from stored "
                                           "confusion matrices)"},
        "pairing": "paired comparisons resample identical unit indices for both sides",
        "significance_policy": "no result is labelled 'statistically significant'; "
                               "reports state whether the CI excludes zero and the "
                               "proportion of resamples favouring A",
    })
    (report_dir / "missing_inputs.md").write_text(
        "# Bootstrap uncertainty - missing or pending inputs\n\n"
        f"Generated: {generated_at}\n\n"
        + "\n".join(f"- {note}" for note in missing) + "\n", encoding="utf-8")

    lines = [
        "# Bootstrap uncertainty summary",
        f"\nGenerated: {generated_at}  |  Commit: `{ev.git_commit_sha()[:12]}`",
        f"\nSeeded percentile bootstrap ({args.n_bootstrap} resamples, seed "
        f"{args.seed}, {args.confidence_level:.0%} intervals) over stored "
        "sample-level outcomes. Resampling units, exclusions and pairing rules: "
        "`bootstrap_config.json`; gaps: `missing_inputs.md`.",
    ]
    for workstream in sorted({r["workstream"] for r in rows}):
        ws_rows = [r for r in rows if r["workstream"] == workstream]
        lines.append(f"\n## {workstream} - status: **{ws_rows[0]['status']}**\n")
        ci_rows = [r for r in ws_rows if r["analysis"] == "ci"]
        if ci_rows:
            lines.append("| Model | Run | Split | Metric | Point | 95% CI | Units |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for r in ci_rows:
                lines.append(f"| {r['model']} | {r['run']} | {r['split']} | "
                             f"{r['metric']} | {r['point']} | "
                             f"[{r['ci_low']}, {r['ci_high']}] | {r['n_units']} |")
        paired = [r for r in ws_rows if r["analysis"] == "paired"]
        if paired:
            lines.append("\n| Comparison | Split | Metric | Observed diff | 95% CI | "
                         "P(A better) | Units |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for r in paired:
                lines.append(f"| {r['comparison']} | {r['split']} | {r['metric']} | "
                             f"{r['observed_difference']:+} | [{r['ci_low']}, "
                             f"{r['ci_high']}] | {r['p_a_better']} | {r['n_units']} |")
            lines.append("\n> Interpretation criterion: a difference is only called "
                         "robust when its percentile CI excludes zero; the "
                         "P(A better) column is the raw proportion of resamples, "
                         "not a p-value.")
    if figures:
        lines.append("\n## Figures\n")
        lines.extend(f"- `{f}`" for f in figures if f.endswith(".png"))
    lines.append("\n*Read-only analysis - no inference was run; pending analyses "
                 "are listed in `missing_inputs.md`.*\n")
    (report_dir / "bootstrap_summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_self_tests() -> int:
    cases = []
    values = [0.0, 0.25, 0.5, 0.75, 1.0]
    cases.append(("percentile interpolation", ev.percentile(values, 0.5) == 0.5
                  and ev.percentile(values, 0.0) == 0.0))

    units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    mean = lambda u: sum(u) / len(u)  # noqa: E731
    ci_1 = ev.bootstrap_ci(units, mean, n_bootstrap=200, seed=7)
    ci_2 = ev.bootstrap_ci(units, mean, n_bootstrap=200, seed=7)
    ci_3 = ev.bootstrap_ci(units, mean, n_bootstrap=200, seed=8)
    cases.append(("same seed reproduces identical CI", ci_1 == ci_2))
    cases.append(("different seed changes resamples",
                  (ci_1["low"], ci_1["high"]) != (ci_3["low"], ci_3["high"])))
    cases.append(("CI brackets the point estimate",
                  ci_1["low"] <= ci_1["point"] <= ci_1["high"]))
    constant = ev.bootstrap_ci([5, 5, 5, 5], mean, n_bootstrap=50, seed=1)
    cases.append(("constant data gives degenerate CI",
                  constant["low"] == constant["high"] == constant["point"] == 5))

    paired_units = [(0.7, 0.6)] * 30
    paired = ev.paired_bootstrap(
        paired_units, lambda u: mean([x[0] for x in u]),
        lambda u: mean([x[1] for x in u]), n_bootstrap=100, seed=3)
    cases.append(("constant paired difference is exact",
                  abs(paired["observed_difference"] - 0.1) < 1e-9
                  and abs(paired["low"] - 0.1) < 1e-9
                  and paired["p_a_better"] == 1.0))

    confusion = [[8, 2], [1, 9]]
    samples = ev.confusion_to_samples(confusion)
    cases.append(("confusion reconstruction preserves counts and accuracy",
                  len(samples) == 20 and abs(accuracy_of(samples) - 17 / 20) < 1e-9))
    cases.append(("macro-F1 on symmetric confusion",
                  abs(macro_f1_of(samples, 2)
                      - ((2 * (8 / 9) * 0.8 / ((8 / 9) + 0.8)
                          + 2 * (9 / 11) * 0.9 / ((9 / 11) + 0.9)) / 2)) < 1e-9))
    cases.append(("single-class F1 matches manual computation",
                  abs(single_class_f1(samples, 0)
                      - 2 * (8 / 9) * 0.8 / ((8 / 9) + 0.8)) < 1e-9))

    micro_units = [(2, 1, 0, 1.8, 2), (1, 0, 1, 0.9, 1)]
    cases.append(("micro precision/recall from image units",
                  abs(micro_metric(micro_units, "precision") - 3 / 4) < 1e-9
                  and abs(micro_metric(micro_units, "recall") - 3 / 4) < 1e-9
                  and abs(micro_metric(micro_units, "mean_matched_iou") - 0.9) < 1e-9))
    cases.append(("empty units are safe",
                  micro_metric([], "f1") == 0.0
                  and ev.bootstrap_ci([], lambda u: 0.0,
                                      n_bootstrap=10, seed=1)["n_units"] == 0))

    failures = 0
    for label, ok in cases:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        failures += 0 if ok else 1
    print(f"\nSelf-test: {len(cases) - failures}/{len(cases)} passed.")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap uncertainty over stored sample-level outcomes (read-only).")
    parser.add_argument("--dataset", default="all",
                        choices=["MDWD", "MTSD-attributes", "PromptDetect-MDWD",
                                 "PromptDetect-MTSD", "all"])
    parser.add_argument("--task", default="auto",
                        choices=["detection", "attributes", "auto"])
    parser.add_argument("--run", default="latest")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-a", default="")
    parser.add_argument("--model-b", default="")
    parser.add_argument("--prompt-a", default="")
    parser.add_argument("--prompt-b", default="")
    parser.add_argument("--working-conf", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return run_self_tests()

    pd_metrics = ev.load_pd_metrics()
    rows: list[dict] = []
    missing: list[str] = []
    wants = lambda name: args.dataset in ("all", name)  # noqa: E731

    if wants("MTSD-attributes") and args.task in ("attributes", "auto"):
        print("Bootstrapping MTSD attribute heads (reconstructed samples)...")
        rows += analyse_attributes(args, missing)
    for dataset in ("MDWD", "MTSD"):
        if wants(f"PromptDetect-{dataset}") and args.task in ("detection", "auto"):
            print(f"Bootstrapping PromptDetect {dataset} (image level)...")
            rows += analyse_promptdetect(args, pd_metrics, dataset, missing)
    if wants("MDWD") and args.task in ("detection", "auto"):
        print("Bootstrapping MDWD detection (image level, stored predictions)...")
        rows += analyse_mdwd_detection(args, pd_metrics, missing)
    missing.append("MTSD supervised detection: PENDING - no training runs exist yet.")

    print(f"\nCI/paired rows: {len(rows)}; pending/missing notes: {len(missing)}")
    if args.dry_run:
        print("Dry run: no files written.")
        return 0
    report_dir, figures_dir = ev.new_output_dirs("Statistical-Uncertainty",
                                                 args.overwrite, args.output_dir)
    figures = make_figures(rows, figures_dir) if rows else []
    write_reports(report_dir, figures, rows, missing, args)
    print(f"Reports: {report_dir}\nFigures: {figures_dir} ({len(figures)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
