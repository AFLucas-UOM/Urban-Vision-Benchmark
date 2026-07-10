#!/usr/bin/env python3
"""Build the static, read-only dissertation evidence dashboard.

Generates a fully offline HTML site (plain HTML + CSS + vanilla JS, no CDN,
no telemetry, no backend) that indexes the repository's dissertation
evidence: datasets, experiments, final tables, figures, robustness slices,
statistical uncertainty, annotation effort, failure cases, limitations and a
machine-readable provenance index (data/evidence_index.json).

The dashboard is a *viewer*: it never runs training or inference, never
modifies evidence, and renders missing evidence as PENDING instead of
failing. Statuses come from explicit rules and
`Scripts/FinalEvaluation/config/research_questions.yaml` - a result file
existing is never enough to call something "final".

Outputs:
    Documents/Final-Reports/Dissertation-Dashboard/<stamp>/
        index.html   assets/style.css   assets/app.js   data/evidence_index.json
    --overwrite writes to Documents/Final-Reports/Dissertation-Dashboard/latest/

Usage:
    python Scripts/FinalEvaluation/build_dissertation_dashboard.py --self-test
    python Scripts/FinalEvaluation/build_dissertation_dashboard.py --dry-run
    python Scripts/FinalEvaluation/build_dissertation_dashboard.py
    python Scripts/FinalEvaluation/build_dissertation_dashboard.py --overwrite
"""

from __future__ import annotations

import argparse
import html
import os
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evidence_lib as ev  # noqa: E402

GENERATOR = "Scripts/FinalEvaluation/build_dissertation_dashboard.py"
DASH_ROOT = ev.REPORTS_ROOT / "Dissertation-Dashboard"
RQ_CONFIG = SCRIPT_DIR / "config" / "research_questions.yaml"
STATUS_VOCAB = {ev.STATUS_FINAL, ev.STATUS_HISTORICAL, ev.STATUS_PILOT,
                ev.STATUS_SMOKE, ev.STATUS_PENDING, ev.STATUS_EXCLUDED}
BADGE_CLASS = {ev.STATUS_FINAL: "final", ev.STATUS_HISTORICAL: "hist",
               ev.STATUS_PILOT: "pilot", ev.STATUS_SMOKE: "smoke",
               ev.STATUS_PENDING: "pending", ev.STATUS_EXCLUDED: "excluded"}


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def latest_subdir(base: Path) -> Path | None:
    """Most recently modified run folder under a tool's output root."""
    if not base.is_dir():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir()]
    return max(candidates, key=lambda p: p.stat().st_mtime, default=None)


# ---------------------------------------------------------------------------
# Evidence collection (read-only)
# ---------------------------------------------------------------------------

def load_research_questions() -> dict:
    import yaml

    if not RQ_CONFIG.exists():
        return {"project": {}, "questions": []}
    payload = yaml.safe_load(RQ_CONFIG.read_text(encoding="utf-8")) or {}
    for question in payload.get("questions", []):
        if question.get("status") not in STATUS_VOCAB:
            question["status"] = ev.STATUS_PENDING
    return payload


def collect_datasets() -> dict:
    data = {"mdwd": {}, "mtsd": {}, "integrity": {}, "leakage": {}, "warnings": []}
    eda = ev.ROOT / "Documents" / "MDWD-EDA" / "eda_summary.json"
    if eda.exists():
        payload = ev.read_json(eda)
        data["mdwd"] = {"images": payload.get("n_images"),
                        "sources": payload.get("n_unique_source_images"),
                        "boxes": payload.get("n_boxes"),
                        "classes": payload.get("classes", []),
                        "issues": payload.get("issue_counts", {}),
                        "source": ev.rel(eda)}
    qa_files = [p for p in sorted((ev.ROOT / "Datasets" / "MTSD" / "Annotations")
                                  .glob("GRP-*/Final-QA/QA-*.json"))
                if ".bak" not in p.name.lower()]
    images = boxes = 0
    groups = []
    for qa_path in qa_files:
        try:
            payload = ev.read_json(qa_path)
        except Exception:
            continue
        groups.append(qa_path.parents[1].name)
        images += len(payload.get("images", []))
        boxes += len(payload.get("annotations", []))
    all_groups = sorted(p.name for p in (ev.ROOT / "Datasets" / "MTSD").glob("GRP-*"))
    data["mtsd"] = {"qa_groups": groups, "images": images, "boxes": boxes,
                    "unannotated_groups": sorted(set(all_groups) - set(groups)),
                    "source": "Datasets/MTSD/Annotations/GRP-*/Final-QA"}
    integrity = ev.REPORTS_ROOT / "dataset_integrity_report.json"
    if integrity.exists():
        payload = ev.read_json(integrity)
        data["integrity"] = {"overall": payload.get("overall"),
                             "generated_at": payload.get("generated_at"),
                             "sections": [{"name": s.get("name"),
                                           "status": s.get("status")}
                                          for s in payload.get("sections", [])],
                             "source": ev.rel(integrity)}
    leakage = ev.REPORTS_ROOT / "MDWD-Leakage-Analysis" / "metric_comparison.json"
    if leakage.exists():
        payload = ev.read_json(leakage).get("leakage", {})
        data["leakage"] = {"n_sources": payload.get("n_leaked_source_identities"),
                           "valid_affected": payload.get("affected_valid_images"),
                           "test_affected": payload.get("affected_test_images"),
                           "verdict": "negligible (max delta 0.48 pp mAP50-95)",
                           "source": ev.rel(leakage.parent / "mdwd_leakage_sensitivity_report.md")
                           if (leakage.parent / "mdwd_leakage_sensitivity_report.md").exists()
                           else ev.rel(leakage)}
    if data["mtsd"].get("unannotated_groups"):
        data["warnings"].append(
            f"MTSD groups without Final-QA annotations: "
            f"{', '.join(data['mtsd']['unannotated_groups'])}")
    if data["mdwd"].get("issues"):
        data["warnings"].append(f"MDWD integrity findings: {data['mdwd']['issues']} "
                                "(leakage measured negligible)")
    return data


def collect_experiments() -> list[dict]:
    cards: list[dict] = []

    def card(**kw) -> dict:
        base = {"title": "", "status": ev.STATUS_PENDING, "dataset": "", "model": "",
                "run_id": "", "source": "", "test_status": "", "main_metric": "",
                "limitations": "", "headline_use": "no", "task": ""}
        base.update(kw)
        cards.append(base)
        return base

    # MDWD supervised detection (final) --------------------------------------
    for summary in sorted((ev.ROOT / "Results" / "MDWD-Results")
                          .glob("*/Model-Size-Comparison/*summary*.csv")):
        rows = ev.read_csv_rows(summary)
        rows = [r for r in rows if r.get("model_variant")]
        if not rows:
            continue
        family = summary.parts[len((ev.ROOT / "Results" / "MDWD-Results").parts)]
        best = max(rows, key=lambda r: float(r.get("mAP@50-95") or 0))
        card(title=f"MDWD supervised detection - {family}", status=ev.STATUS_FINAL,
             dataset="MDWD (Roboflow v20)", task="detection",
             model=", ".join(r["model_variant"] for r in rows),
             run_id=best.get("run_name", ""), source=ev.rel(summary),
             test_status="evaluated on official valid + test splits",
             main_metric=f"best {best['model_variant']}: val mAP50-95 "
                         f"{float(best.get('mAP@50-95') or 0):.4f} / test "
                         f"{float(best.get('test_mAP@50-95') or 0):.4f}",
             limitations="augmented-export leakage measured negligible",
             headline_use="yes")
    card(title="MDWD RF-DETR normalisation", status=ev.STATUS_EXCLUDED,
         dataset="MDWD", task="detection", model="rfdetr-nano",
         run_id="RF-DETR-EUVIP/E001",
         source="Results/MDWD-Runs/RF-DETR-EUVIP",
         test_status="COCO-evaluator log only",
         main_metric="not normalised into the consolidated tables",
         limitations="single nano run, different evaluator; exclude from "
                     "cross-family quantitative claims until normalised",
         headline_use="no")
    card(title="MTSD supervised detection", status=ev.STATUS_PENDING,
         dataset="MTSD (QA groups)", task="detection",
         limitations="prepared dataset + training notebooks not yet run",
         source="Scripts/MTSD-Scripts/MTSD-SupervisedNotebooks",
         headline_use="no")

    # Attribute classification ------------------------------------------------
    comparison = ev.ATTR_OUTPUTS / "reports" / "comparison.csv"
    rows = ev.read_csv_rows(comparison)
    if rows:
        best = max(rows, key=lambda r: float(r.get("mean_macro_f1") or 0))
        card(title="MTSD attribute classification (historical round)",
             status=ev.STATUS_HISTORICAL, dataset="MTSD GRP-1..GRP-3 snapshot",
             task="attributes", model=", ".join(r["variant"] for r in rows),
             run_id=best.get("run_id", ""), source=ev.rel(comparison),
             test_status="single test evaluation per variant",
             main_metric=f"best {best['variant']}: mean macro-F1 "
                         f"{float(best.get('mean_macro_f1') or 0):.4f}",
             limitations="valid only for the GRP-1..3 snapshot; weak condition head",
             headline_use="only with historical-scope label")
    card(title="MTSD attribute classification (final-scope round)",
         status=ev.STATUS_PENDING, dataset="MTSD (final annotation scope)",
         task="attributes",
         limitations="deferred until the MTSD annotation scope is frozen",
         source="Scripts/MTSD-Scripts/AttributeClassification", headline_use="no")

    # PromptDetect -------------------------------------------------------------
    prompt_runs = ev.discover_promptdetect_runs()
    for run in prompt_runs:
        config = run["config"]
        card(title=f"PromptDetect {config.get('dataset')} run {run['run_id']}",
             status=run["status"], dataset=str(config.get("dataset")),
             task="prompt-based detection",
             model=", ".join(config.get("models", [])),
             run_id=run["run_id"],
             source=ev.rel(run["run_dir"] / "evaluation_summary.json"),
             test_status=f"{config.get('n_images')} images / "
                         f"{config.get('n_gt_boxes')} GT boxes ({config.get('split')})",
             main_metric=f"{len(config.get('prompts', []))} prompt(s)",
             limitations="pilot scale - pipeline validation only"
             if run["status"] == ev.STATUS_PILOT else "",
             headline_use="no" if run["status"] == ev.STATUS_PILOT else "yes")
    if not any(r["config"].get("dataset") == "MTSD" for r in prompt_runs):
        card(title="PromptDetect MTSD evaluation", status=ev.STATUS_PENDING,
             dataset="MTSD", task="prompt-based detection",
             limitations="needs the prepared MTSD detection dataset first",
             source="Scripts/Other-Scripts/PromptDetect/batch_evaluation",
             headline_use="no")

    # Benchmarks / new analysis tools ------------------------------------------
    speed = ev.ROOT / "Results" / "Inference-Benchmark" / "InferenceSpeed"
    speed_runs = sorted(speed.glob("*/inference_speed_summary.csv")) if speed.is_dir() else []
    card(title="Inference-speed benchmark",
         status=ev.STATUS_FINAL if speed_runs else ev.STATUS_PENDING,
         dataset="MDWD / MTSD / PromptDetect models", task="efficiency",
         source=ev.rel(speed_runs[-1]) if speed_runs
         else "Scripts/Other-Scripts/Inference-Benchmark",
         limitations="" if speed_runs else "tooling ready, benchmark not executed",
         headline_use="yes" if speed_runs else "no")
    for title, base, filename in [
            ("Robustness slice analysis", ev.REPORTS_ROOT / "Robustness-Slices",
             "robustness_slice_summary.md"),
            ("Statistical uncertainty (bootstrap)",
             ev.REPORTS_ROOT / "Statistical-Uncertainty", "bootstrap_summary.md"),
            ("Annotation / operational effort",
             ev.REPORTS_ROOT / "Annotation-Effort", "annotation_effort_report.md")]:
        latest = latest_subdir(base)
        available = latest is not None and (latest / filename).exists()
        card(title=title,
             status=ev.STATUS_FINAL if available else ev.STATUS_PENDING,
             dataset="cross-track", task="analysis",
             run_id=latest.name if latest else "",
             source=ev.rel(latest / filename) if available else ev.rel(base),
             limitations="" if available else "not generated yet",
             main_metric="derived analysis over stored evidence",
             headline_use="yes (analysis layer)" if available else "no")
    return cards


def collect_tables() -> dict:
    tables: dict[str, dict] = {}
    latest = latest_subdir(ev.TABLES_ROOT)
    if latest is None:
        return tables
    for csv_path in sorted(latest.glob("*.csv")):
        rows = ev.read_csv_rows(csv_path)
        pending = bool(rows) and rows[0].get("status") == "pending"
        tables[csv_path.stem] = {"rows": rows[:60], "n_rows": len(rows),
                                 "pending": pending, "source": ev.rel(csv_path),
                                 "tables_run": latest.name}
    if "promptdetect_results" in tables and tables["promptdetect_results"]["pending"] \
            and ev.discover_promptdetect_runs():
        tables["promptdetect_results"]["stale_note"] = (
            "PromptDetect runs now exist but this table snapshot predates them - "
            "re-run export_dissertation_tables.py.")
    return tables


def collect_figures() -> list[dict]:
    figures = []
    for base in sorted(ev.FIGURES_ROOT.glob("*")) if ev.FIGURES_ROOT.is_dir() else []:
        latest = latest_subdir(base)
        if latest is None:
            continue
        for png in sorted(latest.glob("*.png")):
            entry = {"title": png.stem.replace("_", " "),
                     "png": ev.rel(png), "category": base.name,
                     "run": latest.name, "source_dir": ev.rel(latest),
                     "generating_script": f"Scripts/FinalEvaluation ({base.name})"}
            for extension in ("pdf", "svg"):
                sibling = png.with_suffix(f".{extension}")
                if sibling.exists():
                    entry[extension] = ev.rel(sibling)
            figures.append(entry)
    attr_figure = ev.ATTR_OUTPUTS / "reports" / "comparison_macro_f1.png"
    if attr_figure.exists():
        figures.append({"title": "MTSD attributes - macro-F1 comparison "
                                 "(historical round)",
                        "png": ev.rel(attr_figure), "category": "AttributeClassification",
                        "run": "GRP-1..3 snapshot", "source_dir": ev.rel(attr_figure.parent),
                        "generating_script": "AttributeClassification/run_all.py"})
    for run in ev.discover_promptdetect_runs():
        for name in ("metrics_bars.png", "confusion_matrix.png"):
            plot = run["run_dir"] / name
            if plot.exists():
                figures.append({"title": f"PromptDetect {run['run_id']} - "
                                         f"{plot.stem.replace('_', ' ')} ({run['status']})",
                                "png": ev.rel(plot), "category": "PromptDetect",
                                "run": run["run_id"], "source_dir": ev.rel(run["run_dir"]),
                                "generating_script": "batch_evaluation/run_batch_eval.py"})
    return figures


def collect_tool_json(base: Path, filename: str) -> dict | None:
    latest = latest_subdir(base)
    if latest and (latest / filename).exists():
        payload = ev.read_json(latest / filename)
        payload["_source"] = ev.rel(latest / filename)
        payload["_run"] = latest.name
        return payload
    return None


def collect_failure_cases() -> list[dict]:
    items = []
    for run in ev.discover_promptdetect_runs():
        index = run["run_dir"] / "visualizations" / "visualization_index.csv"
        for row in ev.read_csv_rows(index):
            fp = int(row.get("fp") or 0)
            fn = int(row.get("fn") or 0)
            duplicates = int(row.get("duplicates") or 0)
            error = ("false positive" if fp else "") or ("missed" if fn else "") \
                or ("duplicate" if duplicates else "") or "clean"
            items.append({"dataset": row.get("dataset", ""),
                          "model": row.get("model", ""),
                          "prompt": row.get("prompt", ""),
                          "error_type": error, "status": run["status"],
                          "image": ev.rel(run["run_dir"] / "visualizations"
                                          / row["visualization"]),
                          "run": run["run_id"],
                          "counts": f"TP {row.get('tp')} / FP {fp} / FN {fn} / dup {duplicates}"})
    error_root = ev.FIGURES_ROOT / "ErrorAnalysis"
    for index in sorted(error_root.glob("*/*/error_case_index.csv")) if error_root.is_dir() else []:
        for row in ev.read_csv_rows(index):
            image = row.get("output_image") or row.get("image") or ""
            items.append({"dataset": index.parts[len(error_root.parts)],
                          "model": row.get("model", index.parent.name),
                          "prompt": "", "error_type": row.get("category", ""),
                          "status": ev.STATUS_FINAL,
                          "image": ev.rel(index.parent / image) if image else "",
                          "run": index.parent.name, "counts": ""})
    return items


def collect_limitations(datasets: dict, experiments: list[dict],
                        robustness: dict | None, effort: dict | None) -> list[str]:
    limitations = [
        "MTSD attribute results are a historical GRP-1..GRP-3 snapshot; the "
        "final-scope attribute round is pending and results must not be "
        "generalised to the expanded corpus.",
        "MTSD supervised detection is pending (no training runs; Results/MTSD-* empty).",
        "All existing PromptDetect evaluations are pilots (<100 images, exploratory "
        "prompts) - pipeline validation only, not dissertation evidence.",
        "RF-DETR has a single nano-scale run whose metrics are not normalised into "
        "the consolidated tables; it is excluded from cross-family claims.",
        "Cosmos Reason2 and LocateAnything emit no per-box confidence; their AP "
        "collapses to a single operating point and is not comparable to "
        "confidence-ranked AP (compare on precision/recall/F1/matched IoU).",
        "Manual annotation-hour/cost values are not recorded; the effort report "
        "shows them as 'not recorded' rather than estimating them.",
        "Low-support robustness slices are marked insufficient_support and must "
        "not be quoted as findings.",
    ]
    if datasets.get("mdwd", {}).get("issues"):
        limitations.append(
            f"MDWD integrity findings {datasets['mdwd']['issues']} - split leakage "
            "was independently measured as negligible (max 0.48 pp); 15 "
            "out-of-range boxes and 5 empty annotations are train-split only.")
    if datasets.get("mtsd", {}).get("unannotated_groups"):
        limitations.append(
            "Unannotated MTSD groups remain: "
            + ", ".join(datasets["mtsd"]["unannotated_groups"]) + ".")
    return limitations


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

STYLE = """
:root{--bg:#f7f8fa;--card:#fff;--ink:#1c2733;--muted:#5c6b7a;--line:#dde3ea;
--accent:#31597f;--final:#2e7d32;--hist:#8a6d1a;--pilot:#a35c00;--smoke:#7b1fa2;
--pending:#78909c;--excluded:#b23b3b}
*{box-sizing:border-box}body{margin:0;font:14px/1.55 "Segoe UI",system-ui,sans-serif;
background:var(--bg);color:var(--ink)}
header{background:var(--accent);color:#fff;padding:14px 22px}
header h1{margin:0;font-size:18px}header p{margin:4px 0 0;font-size:12px;opacity:.85}
.layout{display:flex;min-height:100vh}
nav{width:215px;flex:0 0 auto;background:#fff;border-right:1px solid var(--line);
padding:14px;position:sticky;top:0;align-self:flex-start;max-height:100vh;overflow:auto}
nav a{display:block;color:var(--accent);text-decoration:none;padding:4px 6px;
border-radius:4px;font-size:13px}nav a:hover{background:var(--bg)}
main{flex:1;padding:18px 26px;max-width:1180px}
section{margin-bottom:34px}h2{font-size:16px;border-bottom:1px solid var(--line);
padding-bottom:6px}h3{font-size:14px;margin:16px 0 6px}
.controls{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 16px}
.controls input,.controls select{padding:6px 8px;border:1px solid var(--line);
border-radius:5px;font-size:13px;background:#fff}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:12px 14px;font-size:13px}
.card h4{margin:0 0 6px;font-size:13.5px}.card dl{margin:0}
.card dt{color:var(--muted);font-size:11px;text-transform:uppercase;
letter-spacing:.03em;margin-top:6px}.card dd{margin:0}
.badge{display:inline-block;padding:1px 9px;border-radius:10px;color:#fff;
font-size:11px;font-weight:600;vertical-align:middle}
.badge.final{background:var(--final)}.badge.hist{background:var(--hist)}
.badge.pilot{background:var(--pilot)}.badge.smoke{background:var(--smoke)}
.badge.pending{background:var(--pending)}.badge.excluded{background:var(--excluded)}
table{border-collapse:collapse;width:100%;font-size:12.5px;background:#fff}
th,td{border:1px solid var(--line);padding:5px 8px;text-align:left;
vertical-align:top}th{background:#eef2f6}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:6px}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px}
.gallery figure{margin:0;background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:8px;font-size:12px}
.gallery img{width:100%;height:auto;border-radius:4px}
details{background:var(--card);border:1px solid var(--line);border-radius:6px;
padding:8px 12px;margin:8px 0}summary{cursor:pointer;font-weight:600;font-size:13px}
.note{color:var(--muted);font-size:12px}.warn{color:#a35c00}
a{color:var(--accent)}footer{padding:16px 26px;color:var(--muted);font-size:12px}
.hidden{display:none!important}
@media(max-width:800px){.layout{flex-direction:column}nav{width:100%;position:static;
display:flex;flex-wrap:wrap;gap:2px}}
"""

APP_JS = """
function normalise(t){return (t||'').toLowerCase();}
function applyFilters(){
  const q=normalise(document.getElementById('search').value);
  const status=document.getElementById('f-status').value;
  const dataset=normalise(document.getElementById('f-dataset').value);
  const model=normalise(document.getElementById('f-model').value);
  document.querySelectorAll('[data-searchable]').forEach(el=>{
    const text=normalise(el.textContent);
    const okQ=!q||text.includes(q);
    const okS=!status||el.dataset.status===status;
    const okD=!dataset||normalise(el.dataset.dataset).includes(dataset);
    const okM=!model||normalise(el.dataset.model).includes(model);
    el.classList.toggle('hidden',!(okQ&&okS&&okD&&okM));
  });
}
window.addEventListener('DOMContentLoaded',()=>{
  ['search','f-status','f-dataset','f-model'].forEach(id=>{
    const el=document.getElementById(id);
    if(el){el.addEventListener('input',applyFilters);el.addEventListener('change',applyFilters);}
  });
});
"""


class Renderer:
    def __init__(self, out_dir: Path, repo_root: Path = ev.ROOT):
        self.out_dir = out_dir
        self.repo_root = repo_root

    def href(self, repo_rel: str) -> str:
        """Relative link from the dashboard to a repo file; refuses escapes.

        Returns "" (rendered as plain text) when the target would escape the
        repository or when no relative path exists (e.g. an --output-dir on a
        different drive) - absolute machine paths are never emitted.
        """
        if not repo_rel:
            return ""
        target = (self.repo_root / repo_rel).resolve()
        try:
            target.relative_to(self.repo_root)
            return Path(os.path.relpath(target, self.out_dir)).as_posix()
        except ValueError:
            return ""

    def link(self, repo_rel: str, label: str | None = None) -> str:
        href = self.href(repo_rel)
        if not href:
            return esc(label or repo_rel)
        return f'<a href="{esc(href)}">{esc(label or repo_rel)}</a>'

    @staticmethod
    def badge(status: str) -> str:
        return (f'<span class="badge {BADGE_CLASS.get(status, "pending")}">'
                f'{esc(status.upper())}</span>')

    def table(self, rows: list[dict], columns: list[str] | None = None,
              max_rows: int = 40) -> str:
        if not rows:
            return '<p class="note">PENDING - no rows available.</p>'
        columns = columns or list(rows[0].keys())
        body = "".join(
            "<tr>" + "".join(f"<td>{esc(row.get(c, ''))}</td>" for c in columns) + "</tr>"
            for row in rows[:max_rows])
        more = (f'<p class="note">... {len(rows) - max_rows} more rows in the '
                'source CSV.</p>' if len(rows) > max_rows else "")
        header = "".join(f"<th>{esc(c)}</th>" for c in columns)
        return (f'<div class="scroll"><table><thead><tr>{header}</tr></thead>'
                f"<tbody>{body}</tbody></table></div>{more}")


def render(model: dict, out_dir: Path, repo_root: Path = ev.ROOT) -> list[Path]:
    r = Renderer(out_dir, repo_root)
    meta = model["meta"]
    parts: list[str] = []
    nav_items: list[tuple[str, str]] = []

    def section(section_id: str, title: str, body: str) -> None:
        nav_items.append((section_id, title))
        parts.append(f'<section id="{section_id}"><h2>{esc(title)}</h2>{body}</section>')

    # 1. Overview -------------------------------------------------------------
    datasets = model["datasets"]
    overview = f"""
    <p>{esc(meta.get('purpose', ''))}</p>
    <div class="cards">
      <article class="card"><h4>MDWD</h4><p>{esc(datasets['mdwd'].get('images', 'PENDING'))}
        images / {esc(datasets['mdwd'].get('boxes', '?'))} boxes /
        {esc(datasets['mdwd'].get('sources', '?'))} unique sources
        ({len(datasets['mdwd'].get('classes', []))} classes).</p></article>
      <article class="card"><h4>MTSD</h4><p>{esc(datasets['mtsd'].get('images', 'PENDING'))}
        QA images / {esc(datasets['mtsd'].get('boxes', '?'))} boxes in groups
        {esc(', '.join(datasets['mtsd'].get('qa_groups', [])) or 'none')}.</p></article>
      <article class="card"><h4>Snapshot</h4><p>Commit
        <code>{esc(meta['commit_sha'][:12])}</code><br>Generated {esc(meta['generated_at'])}
        <br>Evidence items: {len(model['evidence_index'])}</p></article>
    </div>"""
    section("overview", "Project overview", overview)

    # 2. Research questions ----------------------------------------------------
    body = ['<div class="cards">']
    for question in model["questions"]:
        body.append(f"""<article class="card" data-searchable data-status="{esc(question['status'])}"
        data-dataset="{esc(' '.join(question.get('datasets', [])))}" data-model="">
        <h4>{esc(question['id'])} {r.badge(question['status'])}</h4>
        <p>{esc(question['question'])}</p>
        <dl><dt>Experiments</dt><dd>{'<br>'.join(esc(e) for e in question.get('experiments', []))}</dd>
        <dt>Tables / figures</dt><dd>{'<br>'.join(esc(t) for t in
            question.get('tables', []) + question.get('figures', []))}</dd>
        <dt>Limitations</dt><dd>{esc(question.get('limitations', ''))}</dd></dl></article>""")
    body.append("</div>")
    section("questions", "Research questions and evidence matrix", "".join(body))

    # 3. Dataset status ----------------------------------------------------------
    integrity = datasets.get("integrity", {})
    leakage = datasets.get("leakage", {})
    body = ["<div class='cards'>"]
    body.append(f"""<article class="card"><h4>Integrity report</h4>
      <p>Overall: <strong>{esc(integrity.get('overall', 'PENDING'))}</strong>
      ({esc(integrity.get('generated_at', 'not generated'))})</p>
      <p>{'<br>'.join(esc(s['name']) + ': ' + esc(s['status'])
                      for s in integrity.get('sections', []))}</p>
      <p>{r.link(integrity.get('source', ''), 'dataset_integrity_report.json')
          if integrity else ''}</p></article>""")
    body.append(f"""<article class="card"><h4>MDWD split leakage</h4>
      <p>{esc(leakage.get('n_sources', 'PENDING'))} leaked source identities
      ({esc(leakage.get('valid_affected', '?'))} valid / {esc(leakage.get('test_affected', '?'))}
      test images) - measured effect: <strong>{esc(leakage.get('verdict', 'pending'))}</strong></p>
      <p>{r.link(leakage.get('source', ''), 'leakage sensitivity report') if leakage else ''}</p>
      </article>""")
    body.append(f"""<article class="card"><h4>Warnings</h4>
      <p class="warn">{'<br>'.join(esc(w) for w in datasets.get('warnings', [])) or 'none'}</p>
      </article>""")
    body.append("</div>")
    section("datasets", "Dataset status", "".join(body))

    # 4. Experiment status ----------------------------------------------------------
    body = ["<div class='cards'>"]
    for card in model["experiments"]:
        body.append(f"""<article class="card" data-searchable
        data-status="{esc(card['status'])}" data-dataset="{esc(card['dataset'])}"
        data-model="{esc(card['model'])}">
        <h4>{esc(card['title'])} {r.badge(card['status'])}</h4><dl>
        <dt>Dataset scope</dt><dd>{esc(card['dataset'])}</dd>
        <dt>Model(s)</dt><dd>{esc(card['model'] or '-')}</dd>
        <dt>Run</dt><dd>{esc(card['run_id'] or '-')}</dd>
        <dt>Test-set status</dt><dd>{esc(card['test_status'] or '-')}</dd>
        <dt>Main metric</dt><dd>{esc(card['main_metric'] or '-')}</dd>
        <dt>Limitations</dt><dd>{esc(card['limitations'] or '-')}</dd>
        <dt>Headline use</dt><dd>{esc(card['headline_use'])}</dd>
        <dt>Source</dt><dd>{r.link(card['source']) if card['source'] else '-'}</dd>
        </dl></article>""")
    body.append("</div>")
    section("experiments", "Experiment status", "".join(body))

    # 5. Final quantitative evidence -------------------------------------------------
    tables = model["tables"]
    titles = {"mdwd_detection_results": "Supervised detection - MDWD",
              "mtsd_detection_results": "Supervised detection - MTSD",
              "mtsd_attribute_results": "Attribute classification (historical snapshot)",
              "promptdetect_results": "Prompt-based localisation",
              "model_efficiency_results": "Efficiency (inference speed)",
              "final_model_comparison_table": "Cross-track long-form comparison "
                                              "(per-track metrics, not one leaderboard)"}
    body = []
    if not tables:
        body.append('<p class="note">PENDING - run export_dissertation_tables.py '
                    'to generate Documents/Final-Tables.</p>')
    for name, title in titles.items():
        table = tables.get(name)
        if table is None:
            continue
        body.append(f"<h3>{esc(title)}</h3>")
        if table.get("stale_note"):
            body.append(f'<p class="warn">{esc(table["stale_note"])}</p>')
        if table["pending"]:
            note = table["rows"][0].get("note", "") if table["rows"] else ""
            body.append(f'<p class="note">PENDING - {esc(note)}</p>')
        else:
            body.append(r.table(table["rows"]))
        body.append(f'<p class="note">Source: {r.link(table["source"])} '
                    f'(tables run {esc(table["tables_run"])})</p>')
    section("evidence", "Final quantitative evidence", "".join(body))

    # 6. Figures -----------------------------------------------------------------
    body = ["<div class='gallery'>"]
    for figure in model["figures"]:
        downloads = " | ".join(
            r.link(figure[ext], ext.upper()) for ext in ("png", "pdf", "svg")
            if figure.get(ext))
        body.append(f"""<figure data-searchable data-status="" data-dataset="{esc(figure['category'])}"
        data-model=""><a href="{esc(r.href(figure['png']))}">
        <img loading="lazy" src="{esc(r.href(figure['png']))}" alt="{esc(figure['title'])}"></a>
        <figcaption><strong>{esc(figure['title'])}</strong><br>
        <span class="note">{esc(figure['category'])} / {esc(figure['run'])} -
        {esc(figure['generating_script'])}</span><br>{downloads}</figcaption></figure>""")
    body.append("</div>")
    if not model["figures"]:
        body = ['<p class="note">PENDING - no publication figures generated yet.</p>']
    section("figures", "Figures", "".join(body))

    # 7. Robustness ---------------------------------------------------------------
    robustness = model["robustness"]
    if robustness:
        rows = robustness.get("rows", [])
        aggregate = [row for row in rows if row.get("slice_dimension") == "aggregate"]
        deviations = sorted(
            (row for row in rows
             if row.get("support_ok") == "ok" and row.get("delta_vs_aggregate")
             not in ("", None) and row.get("metric") in ("f1", "recall")),
            key=lambda row: row.get("delta_vs_aggregate", 0))[:15]
        insufficient = sum(1 for row in rows if row.get("support_ok") != "ok")
        body = (f'<p>Run <code>{esc(robustness["_run"])}</code> - {len(rows)} slice rows, '
                f'{insufficient} below minimum support (marked, never highlighted). '
                f'Source: {r.link(robustness["_source"])}</p>'
                "<h3>Aggregates</h3>"
                + r.table(aggregate, ["workstream", "model", "split", "metric",
                                      "value", "n_images", "n_objects", "status"])
                + "<h3>Largest slice degradations (sufficient support)</h3>"
                + r.table(deviations, ["workstream", "model", "split",
                                       "slice_dimension", "slice_value", "metric",
                                       "value", "aggregate_value",
                                       "delta_vs_aggregate"]))
    else:
        body = ('<p class="note">PENDING - run '
                '<code>python Scripts/FinalEvaluation/robustness_slice_analysis.py</code>.</p>')
    section("robustness", "Robustness analysis", body)

    # 8. Statistical uncertainty ------------------------------------------------------
    bootstrap = model["bootstrap"]
    if bootstrap:
        rows = bootstrap.get("rows", [])
        ci_rows = [row for row in rows if row.get("analysis") == "ci"]
        paired = [row for row in rows if row.get("analysis") == "paired"]
        body = (f'<p>Run <code>{esc(bootstrap["_run"])}</code>; percentile bootstrap, '
                f'seeded. Source: {r.link(bootstrap["_source"])}</p>'
                "<h3>Confidence intervals</h3>"
                + r.table(ci_rows, ["workstream", "status", "model", "split", "metric",
                                    "point", "ci_low", "ci_high", "n_units", "unit"])
                + ("<h3>Paired comparisons</h3>"
                   + r.table(paired, ["workstream", "comparison", "split", "metric",
                                      "observed_difference", "ci_low", "ci_high",
                                      "p_a_better", "n_units"]) if paired else "")
                + '<details><summary>Methodological notes</summary><p class="note">'
                  'Percentile intervals; paired comparisons resample identical unit '
                  'indices on both sides; no result is labelled "statistically '
                  'significant" - the criterion reported is whether the CI excludes '
                  'zero. Pilot-run intervals validate the pipeline only.</p></details>')
    else:
        body = ('<p class="note">PENDING - run '
                '<code>python Scripts/FinalEvaluation/bootstrap_uncertainty.py</code>.</p>')
    section("uncertainty", "Statistical uncertainty", body)

    # 9. Annotation / operational effort ------------------------------------------------
    effort = model["effort"]
    if effort:
        operational = effort.get("operational_rows", [])
        missing_manual = effort.get("missing_manual_values", [])
        body = (f'<p>Run <code>{esc(effort["_run"])}</code>. Source: '
                f'{r.link(effort["_source"])}</p>'
                + r.table(operational)
                + f'<p class="note">Manual values not recorded: '
                  f'{esc(", ".join(missing_manual) or "none")} (never estimated).</p>')
    else:
        body = ('<p class="note">PENDING - run '
                '<code>python Scripts/FinalEvaluation/annotation_effort_report.py</code>.</p>')
    section("effort", "Annotation and operational effort", body)

    # 10. Failure cases ---------------------------------------------------------------
    cases = model["failure_cases"]
    if cases:
        body = ["<div class='gallery'>"]
        for case in cases:
            body.append(f"""<figure data-searchable data-status="{esc(case['status'])}"
            data-dataset="{esc(case['dataset'])}" data-model="{esc(case['model'])}">
            <a href="{esc(r.href(case['image']))}">
            <img loading="lazy" src="{esc(r.href(case['image']))}" alt=""></a>
            <figcaption><strong>{esc(case['model'])}</strong> {r.badge(case['status'])}<br>
            <span class="note">{esc(case['dataset'])} | prompt: {esc(case['prompt'] or '-')} |
            {esc(case['error_type'])} | {esc(case['counts'])}<br>run {esc(case['run'])}
            </span></figcaption></figure>""")
        body.append("</div>")
        body = "".join(body)
    else:
        body = ('<p class="note">PENDING - no failure-case galleries generated yet '
                '(failure_case_sampler.py / PromptDetect visualisations).</p>')
    section("failures", "Failure cases (existing assets only)", body)

    # 11. Limitations ------------------------------------------------------------------
    body = "<ul>" + "".join(f"<li>{esc(l)}</li>" for l in model["limitations"]) + "</ul>"
    section("limitations", "Limitations and unresolved work", body)

    # 12. Provenance --------------------------------------------------------------------
    body = ('<p>Machine-readable index: '
            '<a href="data/evidence_index.json">data/evidence_index.json</a></p>'
            + r.table(model["evidence_index"],
                      ["title", "evidence_type", "status", "dataset", "model",
                       "source_file", "generating_script", "headline_use"], max_rows=100))
    section("provenance", "Provenance", body)

    # Assemble ------------------------------------------------------------------------
    nav = "".join(f'<a href="#{sid}">{esc(title)}</a>' for sid, title in nav_items)
    statuses = "".join(f'<option value="{esc(s)}">{esc(s)}</option>'
                       for s in sorted(STATUS_VOCAB))
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(meta.get('title', 'Dissertation evidence dashboard'))}</title>
<link rel="stylesheet" href="assets/style.css"></head><body>
<header><h1>{esc(meta.get('title', ''))}</h1>
<p>{esc(meta.get('degree', ''))} - read-only evidence dashboard - commit
<code>{esc(meta['commit_sha'][:12])}</code> - generated {esc(meta['generated_at'])}</p></header>
<div class="layout"><nav>{nav}</nav><main>
<div class="controls">
<input id="search" type="search" placeholder="Search evidence...">
<select id="f-status"><option value="">status: all</option>{statuses}</select>
<input id="f-dataset" type="text" placeholder="dataset filter">
<input id="f-model" type="text" placeholder="model filter">
</div>
{''.join(parts)}
</main></div>
<footer>Static, offline dashboard - no telemetry, no external requests, no
modification controls. Generated by {esc(GENERATOR)}.</footer>
<script src="assets/app.js"></script></body></html>"""

    (out_dir / "assets").mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    written = []
    for path, content in [(out_dir / "index.html", page),
                          (out_dir / "assets" / "style.css", STYLE),
                          (out_dir / "assets" / "app.js", APP_JS)]:
        path.write_text(content, encoding="utf-8")
        written.append(path)
    ev.write_json(out_dir / "data" / "evidence_index.json",
                  {"generated_at": meta["generated_at"], "commit_sha": meta["commit_sha"],
                   "generator": GENERATOR, "items": model["evidence_index"]})
    written.append(out_dir / "data" / "evidence_index.json")
    return written


# ---------------------------------------------------------------------------
# Evidence index
# ---------------------------------------------------------------------------

def build_evidence_index(model: dict) -> list[dict]:
    items = []
    stamp = model["meta"]["generated_at"]
    sha = model["meta"]["commit_sha"]

    def item(title, evidence_type, status, source, script, *, dataset="", task="",
             model_name="", experiment_id="", snapshot="", headline="no"):
        items.append({"title": title, "evidence_type": evidence_type,
                      "status": status, "source_file": source,
                      "generating_script": script, "experiment_id": experiment_id,
                      "dataset": dataset, "task": task, "model": model_name,
                      "snapshot": snapshot, "commit_sha": sha,
                      "generated_at": stamp, "headline_use": headline})

    for card in model["experiments"]:
        item(card["title"], "experiment", card["status"], card["source"],
             "see source", dataset=card["dataset"], task=card["task"],
             model_name=card["model"], experiment_id=card["run_id"],
             headline=card["headline_use"])
    for name, table in model["tables"].items():
        item(f"Final table: {name}", "table",
             ev.STATUS_PENDING if table["pending"] else ev.STATUS_FINAL,
             table["source"], "Scripts/FinalEvaluation/export_dissertation_tables.py",
             snapshot=table["tables_run"],
             headline="per-row status applies" if not table["pending"] else "no")
    for figure in model["figures"]:
        item(figure["title"], "figure", ev.STATUS_FINAL, figure["png"],
             figure["generating_script"], dataset=figure["category"],
             snapshot=figure["run"], headline="figure")
    for question in model["questions"]:
        item(f"{question['id']}: {question['question'][:80]}", "research question",
             question["status"], ev.rel(RQ_CONFIG), GENERATOR,
             dataset=", ".join(question.get("datasets", [])),
             headline="matrix entry")
    return items


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_self_tests() -> int:
    import json
    import re
    import tempfile

    cases = []
    with tempfile.TemporaryDirectory() as tmp:
        # Synthetic repo root inside the temp dir: the self-test needs neither
        # the real datasets nor the real repo, and stays on one drive.
        repo_root = Path(tmp) / "repo"
        (repo_root / "Documents" / "Final-Reports").mkdir(parents=True)
        (repo_root / "README.md").write_text("stub", encoding="utf-8")
        out_dir = repo_root / "Documents" / "Final-Reports" / "dash"
        out_dir.mkdir()
        minimal = {
            "meta": {"title": "T", "degree": "D", "purpose": "P",
                     "commit_sha": "abc123def456", "generated_at": "2026-01-01T00:00:00"},
            "questions": [{"id": "RQ1", "question": "Q?", "status": "pending",
                           "datasets": ["MDWD"], "experiments": [], "tables": [],
                           "figures": [], "limitations": "none"}],
            "datasets": {"mdwd": {}, "mtsd": {}, "integrity": {}, "leakage": {},
                         "warnings": []},
            "experiments": [{"title": "E1", "status": "pilot", "dataset": "MDWD",
                             "model": "m", "run_id": "r", "source": "README.md",
                             "test_status": "", "main_metric": "", "task": "t",
                             "limitations": "", "headline_use": "no"}],
            "tables": {}, "figures": [], "robustness": None, "bootstrap": None,
            "effort": None, "failure_cases": [], "limitations": ["L1"],
        }
        minimal["evidence_index"] = build_evidence_index(minimal)
        written = render(minimal, out_dir, repo_root)
        page = (out_dir / "index.html").read_text(encoding="utf-8")

        cases.append(("all four artefacts written",
                      {p.name for p in written} ==
                      {"index.html", "style.css", "app.js", "evidence_index.json"}))
        cases.append(("missing sections render PENDING, build does not crash",
                      page.count("PENDING") >= 4))
        cases.append(("no machine-absolute paths leak into the page",
                      not re.search(r"[A-Za-z]:[\\/](Users|2\. UM)", page)))
        cases.append(("no external hosts referenced",
                      "http://" not in page.replace("http://localhost", "")
                      and "https://" not in page))
        index = json.loads((out_dir / "data" / "evidence_index.json")
                           .read_text(encoding="utf-8"))
        cases.append(("evidence index parses with valid statuses",
                      all(i["status"] in STATUS_VOCAB for i in index["items"])))
        cases.append(("pilot experiment is not headline-allowed",
                      all(i["headline_use"] == "no" for i in index["items"]
                          if i["evidence_type"] == "experiment")))

        renderer = Renderer(out_dir, repo_root)
        cases.append(("href stays inside the repository",
                      renderer.href("README.md").endswith("README.md")
                      and renderer.href("../outside.txt") == ""))
        href = renderer.href("Documents/Final-Reports/repository_health_check.md")
        cases.append(("href is relative (portable)", not Path(href).is_absolute()))
        # Out-dir against the real repo root: on another drive this exercises
        # the ValueError fallback; either way no absolute path may be emitted.
        cross = Renderer(Path(tmp) / "x", ev.ROOT).href("README.md")
        cases.append(("href never emits absolute or drive-lettered paths",
                      cross == "" or not re.match(r"^([A-Za-z]:|/)", cross)))

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
        description="Build the static dissertation evidence dashboard (read-only).")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="Write to Dissertation-Dashboard/latest/ instead of "
                             "a new timestamped folder.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return run_self_tests()

    print("Collecting evidence (read-only)...")
    rq = load_research_questions()
    datasets = collect_datasets()
    experiments = collect_experiments()
    tables = collect_tables()
    figures = collect_figures()
    robustness = collect_tool_json(ev.REPORTS_ROOT / "Robustness-Slices",
                                   "robustness_slice_results.json")
    bootstrap = collect_tool_json(ev.REPORTS_ROOT / "Statistical-Uncertainty",
                                  "bootstrap_results.json")
    effort_details = collect_tool_json(ev.REPORTS_ROOT / "Annotation-Effort",
                                       "annotation_effort_details.json")
    if effort_details:
        run_dir = ev.ROOT / Path(effort_details["_source"]).parent
        effort_details["operational_rows"] = ev.read_csv_rows(
            run_dir / "operational_comparison.csv")
        effort_details["missing_manual_values"] = effort_details.get(
            "missing_manual_values", [])
    failure_cases = collect_failure_cases()

    model = {
        "meta": {**rq.get("project", {}), "commit_sha": ev.git_commit_sha(),
                 "generated_at": datetime.now().isoformat(timespec="seconds")},
        "questions": rq.get("questions", []),
        "datasets": datasets, "experiments": experiments, "tables": tables,
        "figures": figures, "robustness": robustness, "bootstrap": bootstrap,
        "effort": effort_details, "failure_cases": failure_cases,
        "limitations": collect_limitations(datasets, experiments, robustness,
                                           effort_details),
    }
    model["evidence_index"] = build_evidence_index(model)

    print(f"Experiments: {len(experiments)} | tables: {len(tables)} | figures: "
          f"{len(figures)} | failure cases: {len(failure_cases)} | evidence items: "
          f"{len(model['evidence_index'])}")
    if args.dry_run:
        print("Dry run: no files written.")
        return 0

    stamp = "latest" if args.overwrite else ev.timestamp()
    out_dir = (args.output_dir or DASH_ROOT) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    render(model, out_dir)
    print(f"Dashboard: {out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
