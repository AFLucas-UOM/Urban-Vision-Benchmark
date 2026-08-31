#!/usr/bin/env python3
"""Urban Vision Benchmark - safe, repository-root launcher.

This file intentionally uses only the Python standard library.  If Rich is
installed it improves the presentation, but it is never required.

Local web applications (Gradio apps) are *managed*: the launcher starts them
in their own process group, polls the local HTTP port until the server is
actually ready (no fixed sleep), opens the browser only after a successful
start, tracks the process, and offers reopen / restart / stop from the menu.
Closing a browser tab never stops a server - that is impossible to detect
reliably - so stopping is always an explicit launcher action.  Every exit
path (menu quit, Ctrl+C, exceptions, direct CLI runs) terminates the full
launcher-owned process tree, including `conda run` intermediaries and any
worker processes they spawned.  The launcher only ever terminates processes
it started itself; a foreign process that happens to own a configured port is
reported, never killed.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import io
import json
import os
import platform
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import webbrowser

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

if os.name == "nt":
    os.system("")  # Enable VT100 escape sequences in Windows cmd/powershell


ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / ".uvb_launcher_state.json"
LOG_DIR = ROOT / ".uvb_launcher_logs"
WORKFLOW_REGISTRY_FILE = ROOT / "Scripts/Automation/workflow_targets.json"
EXECUTED_NOTEBOOK_DIR = ROOT / "Scripts/Automation/executed"

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    RICH = True
    CONSOLE = Console()
except ImportError:  # pragma: no cover - exercised on minimal installations
    RICH = False
    CONSOLE = None


# ---------------------------------------------------------------------------
# Cache git info once at startup so we don't shell out on every frame redraw
# ---------------------------------------------------------------------------
def _init_repo_status() -> tuple[str, str]:
    branch = commit = "unavailable"
    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip() or "detached"
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return branch, commit

_REPO_BRANCH, _REPO_COMMIT = _init_repo_status()


@dataclass(frozen=True)
class Tool:
    id: str
    title: str
    category: str
    description: str
    script: str | None = None
    args: tuple[str, ...] = ()
    cwd: str = "."
    env: str | None = None
    port: int | None = None
    port_arg: str | None = None
    no_browser_arg: str | None = None
    url: str | None = None
    browser: bool = False
    heavy: bool = False
    confirm: bool = False
    read_only: bool = True
    ready_timeout: float = 90.0
    notes: str = ""
    instruction: str | None = None
    kind: str = "python"
    no_user_site: bool = False
    training: bool = False
    workflow_target: str | None = None
    redirect_id: str | None = None

    @property
    def path(self) -> Path | None:
        return ROOT / self.script if self.script else None

    @property
    def is_web_app(self) -> bool:
        return self.port is not None and self.url is not None


def T(**kwargs) -> Tool:
    return Tool(**kwargs)


WEB_APP_NOTE = ("Closing the browser tab does NOT stop the server; "
                "stop or restart it from the launcher.")

CURATED_TOOLS = [
    # ---- Local web applications (long-running, launcher-managed) ----------
    T(id="promptdetect",
      title="PromptDetect — single-image prompted localisation (local web app)",
      category="Local Web Applications",
      description="Gradio app: run one image + one text prompt through SAM 3/3.1, LocateAnything 3B or Cosmos Reason2 and inspect the boxes/masks. Models are chosen and loaded inside the app.",
      script="Scripts/Other-Scripts/PromptDetect/app.py", cwd="Scripts/Other-Scripts/PromptDetect",
      env="mtsd-base", port=7860, port_arg="--port", no_browser_arg="--no-browser",
      url="http://127.0.0.1:{port}", browser=True, heavy=True, confirm=True, ready_timeout=180.0,
      notes="May use substantial GPU memory; selecting LocateAnything also starts its mtsd-la worker process (owned and cleaned up with the app)."),
    T(id="promptdetect-batch-ui",
      title="PromptDetect — batch prompt evaluation UI (local web app)",
      category="Local Web Applications",
      description="Gradio app for scoring prompt sets against MDWD or MTSD ground truth. A dry-run mode inside the app validates plans without loading models; real runs load heavy models.",
      script="Scripts/Other-Scripts/PromptDetect/batch_evaluation/gradio_batch_eval.py",
      cwd="Scripts/Other-Scripts/PromptDetect/batch_evaluation",
      env="mtsd-base", port=7860, port_arg="--port", no_browser_arg="--no-browser",
      url="http://127.0.0.1:{port}", browser=True, heavy=True, confirm=True, ready_timeout=180.0,
      notes="Cosmos Reason2 32B stays gated inside the UI."),
    T(id="attr-ui",
      title="MTSD attributes — checkpoint comparison (local web app)",
      category="Local Web Applications",
      description="Gradio app comparing trained MTSD multi-attribute classifier checkpoints side by side on one sign crop: DINOv3, V-JEPA 2.1, ConvNeXt and LingBot-Vision backbones, base/large sizes, frozen / LoRA / fine-tuned.",
      script="Scripts/MTSD-Scripts/AttributeClassification/inference/gradio_compare.py",
      cwd="Scripts/MTSD-Scripts/AttributeClassification",
      env="mtsd-attrcls", port=7860, port_arg="--port", no_browser_arg="--no-browser",
      url="http://127.0.0.1:{port}", browser=True, heavy=True, confirm=True, ready_timeout=180.0,
      no_user_site=True,
      notes="Needs trained checkpoints under AttributeClassification/outputs/checkpoints."),
    T(id="mtsd-review",
      title="MTSD QA — visual review of audit findings (local web app)",
      category="Local Web Applications",
      description="Gradio app for reviewing the latest annotation audit: image + crop overlays, attribute corrections against the controlled vocabularies, duplicate keep/delete decisions. Saves review decisions only; never edits annotation JSON.",
      script="Scripts/MTSD-Scripts/MTSD-AnnotationQA/review_app.py", cwd="Scripts/MTSD-Scripts/MTSD-AnnotationQA",
      env="mtsd-base", port=7861, port_arg="--port", no_browser_arg="--no-browser",
      url="http://127.0.0.1:{port}", browser=True, confirm=True, read_only=False,
      notes="Writes reviewed_decisions.json in the audit folder; annotations remain unchanged."),

    # ---- Annotation and QA tools -------------------------------------------
    T(id="mtsd-scan", title="MTSD QA — audit Final-QA annotations (read-only)", category="Annotation and QA Tools",
      description="Read every group's Final-QA JSON and report duplicate boxes, missing/invalid attributes and broken references. No files are changed in dry-run mode.",
      script="Scripts/MTSD-Scripts/MTSD-AnnotationQA/scan_annotations.py", cwd=".", env="mtsd-base",
      args=("--dry-run",), read_only=True),
    T(id="mtsd-apply-dry", title="MTSD QA — preview reviewed fixes (dry-run)", category="Annotation and QA Tools",
      description="Show exactly what the latest review decisions would change. This is a dry run; annotations stay untouched.",
      script="Scripts/MTSD-Scripts/MTSD-AnnotationQA/apply_fixes.py", cwd="Scripts/MTSD-Scripts/MTSD-AnnotationQA",
      env="mtsd-base", args=("--decisions", "latest", "--dry-run"), read_only=True),
    T(id="mtsd-apply", title="MTSD QA — apply reviewed fixes (writes annotations)", category="Annotation and QA Tools",
      description="Apply the latest approved review decisions to Final-QA annotations, with backups and two confirmation prompts.",
      script="Scripts/MTSD-Scripts/MTSD-AnnotationQA/apply_fixes.py", cwd="Scripts/MTSD-Scripts/MTSD-AnnotationQA",
      env="mtsd-base", args=("--decisions", "latest", "--apply"), confirm=True, read_only=False,
      notes="This modifies annotation JSON only after the launcher and the tool both confirm; --yes is never used."),
    T(id="qa-gate-refresh-dry", title="MTSD QA gate — preview scope refresh (dry-run)", category="Annotation and QA Tools",
      description="Discover current Final-QA groups and the newest annotation audit, and preview the resulting approved_groups/qa_gate.yaml refresh. Nothing is written.",
      script="Scripts/MTSD-Scripts/MTSD-SupervisedDetection/update_qa_gate.py", cwd=".", env="MDWD",
      args=("--dry-run",), read_only=True),
    T(id="qa-gate-refresh-apply", title="MTSD QA gate — apply scope refresh (writes config)", category="Annotation and QA Tools",
      description="Write the discovered approved_groups scope and qa_gate.yaml resolution status. Never prepares data, trains, or runs the audit itself.",
      script="Scripts/MTSD-Scripts/MTSD-SupervisedDetection/update_qa_gate.py", cwd=".", env="MDWD",
      args=("--apply",), confirm=True, read_only=False,
      notes="Rewrites config/default.yaml and config/qa_gate.yaml in place with no backup copies; both files are atomically replaced and reloaded to confirm they still agree."),
    T(id="labelstudio", title="Label Studio — show documented workflow (no launch)", category="Annotation and QA Tools",
      description="Show the documented start, import and reload steps without starting Label Studio or touching its database.",
      instruction="Documents/LabelStudioWorkflow.md",
      notes="The existing start script can stop another process on port 8080 and writes local Label Studio state."),

    # ---- Dataset analysis and EDA ------------------------------------------
    T(id="mdwd-eda", title="MDWD — sample EDA (50 images, writes analysis output)", category="Dataset Analysis and EDA",
      description="Run a small 50-image MDWD scan and write a separate analysis output. Charts are skipped for a quick, safe pass.",
      script="Scripts/MDWD-Scripts/MDWD-Analysis/run_eda.py", cwd=".", env="MDWD",
      args=("--max-images", "50", "--skip-charts",), confirm=True,
      notes="This writes EDA output only; it does not alter the dataset or experiment results."),
    T(id="mtsd-stats", title="MTSD — image and annotation totals (read-only)", category="Dataset Analysis and EDA",
      description="Print per-group image counts, annotation counts and review workload estimates. No CSV is written.",
      script="Scripts/MTSD-Scripts/MTSD-Analysis/count_image_annotation_stats.py", cwd=".", env="mtsd-base",
      args=("--no-csv",)),
    T(id="mtsd-gps", title="MTSD — GPS metadata audit (read-only)", category="Dataset Analysis and EDA",
      description="Scan MTSD images and print whether EXIF/GPS metadata is present. No report file is requested.",
      script="Scripts/MTSD-Scripts/MTSD-Analysis/check_gps_tags.py", cwd=".", env="mtsd-base"),
    T(id="mdwd-samples", title="MDWD — render sample annotations (writes images)", category="Dataset Analysis and EDA",
      description="Create two annotated sample images per dataset split for visual inspection.",
      script="Scripts/MDWD-Scripts/MDWD-Analysis/visualise_samples.py", cwd=".", env="MDWD",
      args=("--per-split", "2"), confirm=True,
      notes="Writes sample images under the analysis output; source data is untouched."),
    T(id="mtsd-map", title="MTSD — rebuild capture atlas HTML", category="Dataset Analysis and EDA",
      description="Create the interactive HTML map from the cached MTSD image inventory.",
      script="Scripts/MTSD-Scripts/MTSD-Analysis/mtsd_mapper.py", cwd=".", env="mtsd-base", args=(), confirm=True,
      notes="Updates Documents/MTSD-EDA/MTSD_mapped.html only; use Open MTSD map to view it."),

    # ---- Evaluation and benchmarks -----------------------------------------
    T(id="attr-variants-list", title="MTSD attributes — list configured variants (read-only)", category="Evaluation and Benchmarks",
      description="Print every configured attribute-classifier variant (all four backbone families, sizes and adaptation modes) with its metadata. No model loading, no manifest refresh.",
      script="Scripts/MTSD-Scripts/AttributeClassification/run_all.py", cwd="Scripts/MTSD-Scripts/AttributeClassification",
      env="mtsd-attrcls", args=("--list-variants",), read_only=True, no_user_site=True),
    T(id="attr-size-plan", title="MTSD attributes — 18-variant size/adaptation plan (read-only)", category="Evaluation and Benchmarks",
      description="Print the resolved 18-variant size/adaptation matrix (DINOv3, V-JEPA 2.1, ConvNeXt, LingBot-Vision; base/large; frozen/LoRA/fine-tuned; batch settings, output dirs). No manifest refresh, no model loading or downloads, no training.",
      script="Scripts/MTSD-Scripts/AttributeClassification/run_all.py", cwd="Scripts/MTSD-Scripts/AttributeClassification",
      env="mtsd-attrcls", args=("--plan", "--profile", "size_ablation_all"), read_only=True,
      no_user_site=True),
    T(id="batch-cli", title="PromptDetect — batch evaluation plan (dry-run, read-only)", category="Evaluation and Benchmarks",
      description="Validate a small MTSD/SAM3 evaluation plan: 25 images, one neutral prompt, no model load and no output files.",
      script="Scripts/Other-Scripts/PromptDetect/batch_evaluation/run_batch_eval.py",
      cwd="Scripts/Other-Scripts/PromptDetect/batch_evaluation", env="mtsd-base",
      args=("--dataset", "MTSD", "--split", "test", "--prompts", "traffic sign", "--models", "sam3", "--max-images", "25", "--dry-run"),
      read_only=True),
    T(id="mtsd-supervised-plan", title="MTSD supervised detection — 13-model plan (dry-run, read-only)", category="Evaluation and Benchmarks",
      description="Discover Final-QA annotations and print the canonical split and 13-model plan (YOLO11/12/26, RF-DETR) without writing files.",
      script="Scripts/MTSD-Scripts/MTSD-SupervisedDetection/run_mtsd_supervised.py", cwd=".", env="MDWD",
      args=("--dry-run",), read_only=True),
    T(id="promptdetect-dissertation-plan", title="PromptDetect dissertation protocol — plan (dry-run, read-only)", category="Evaluation and Benchmarks",
      description="Validate the fixed targeted prompt protocol and print positive/negative counts without loading models.",
      script="Scripts/Other-Scripts/PromptDetect/batch_evaluation/run_dissertation_protocol.py", cwd=".", env="mtsd-base",
      args=("--dataset", "both", "--split", "test", "--dry-run"), read_only=True),
    T(id="inference-benchmark", title="Inference benchmark — list models (read-only)", category="Evaluation and Benchmarks",
      description="List the supported benchmark models and their availability. It does not load a model or run inference.",
      script="Scripts/Other-Scripts/Inference-Benchmark/inference_speed_benchmark.py", cwd=".", env="MDWD",
      args=("--list-models",)),
    T(id="integrity", title="Final dataset integrity — check (dry-run, read-only)", category="Evaluation and Benchmarks",
      description="Check MDWD and MTSD dataset consistency and print the result. No integrity report is written in dry-run mode.",
      script="Scripts/FinalEvaluation/final_dataset_integrity_check.py", cwd=".", env="MDWD",
      args=("--dataset", "all", "--dry-run")),
    T(id="failure-sampler-help", title="Failure-case sampler — usage help (read-only)", category="Evaluation and Benchmarks",
      description="Show how to sample existing prediction failures without launching a sampling job.",
      script="Scripts/FinalEvaluation/failure_case_sampler.py", cwd="Scripts/FinalEvaluation", env="MDWD",
      args=("--help",)),
    T(id="robustness-slices", title="Robustness slices — plan (dry-run, read-only)", category="Evaluation and Benchmarks",
      description="Compute per-slice performance from stored predictions and print the summary. No model runs; nothing is written in dry-run mode.",
      script="Scripts/FinalEvaluation/robustness_slice_analysis.py", cwd=".", env="MDWD",
      args=("--dry-run",)),
    T(id="bootstrap-ci", title="Bootstrap uncertainty — plan (dry-run, read-only)", category="Evaluation and Benchmarks",
      description="Compute bootstrap confidence intervals over stored sample-level outcomes and print the counts. No model runs; nothing is written in dry-run mode.",
      script="Scripts/FinalEvaluation/bootstrap_uncertainty.py", cwd=".", env="MDWD",
      args=("--dry-run", "--n-bootstrap", "200")),
    T(id="effort-report", title="Annotation effort report (writes timestamped report)", category="Evaluation and Benchmarks",
      description="Derive dataset/annotation/training/operational effort facts and write a timestamped report. It reads existing artefacts only.",
      script="Scripts/FinalEvaluation/annotation_effort_report.py", cwd=".", env="MDWD",
      args=("--skip-disk-size",), confirm=True,
      notes="Writes a new timestamped folder under Documents/Final-Reports/Annotation-Effort; never trains or runs inference."),

    # ---- Documentation and reports -----------------------------------------
    T(id="dashboard", title="Build dissertation evidence dashboard (writes HTML)", category="Documentation and Reports",
      description="Generate the static offline evidence dashboard (HTML) from existing reports, tables and figures.",
      script="Scripts/FinalEvaluation/build_dissertation_dashboard.py", cwd=".", env="MDWD", args=(), confirm=True,
      notes="Writes a new timestamped folder under Documents/Final-Reports/Dissertation-Dashboard; viewer only — never trains or runs inference."),
    T(id="open-final-reports", title="Open final reports folder", category="Documentation and Reports",
      description="Open Documents/Final-Reports in File Explorer.", instruction="Documents/Final-Reports"),
    T(id="open-eda", title="Open EDA output folders", category="Documentation and Reports",
      description="Open the MDWD and MTSD EDA output folders in File Explorer.",
      instruction="Documents/MDWD-EDA\nDocuments/MTSD-EDA"),
    T(id="open-map", title="Open MTSD capture atlas (HTML)", category="Documentation and Reports",
      description="Open the generated MTSD capture atlas HTML in your default browser.",
      instruction="Documents/MTSD-EDA/MTSD_mapped.html"),

    # ---- Repository maintenance --------------------------------------------
    T(id="health", title="Repository health — check (dry-run, read-only)", category="Repository Maintenance",
      description="Check paths, required folders, notebooks and repository conventions without writing health reports.",
      script="Scripts/Automation/verify_repository_health.py", cwd=".", env="mtsd-base", args=("--dry-run",)),
    T(id="gdpr-help", title="GDPR redaction — safe usage help (read-only)", category="Repository Maintenance",
      description="Show preview/apply instructions without loading detectors, generating previews or changing images.",
      script="Scripts/Other-Scripts/GDPR-Compliance/redact.py", cwd="Scripts/Other-Scripts/GDPR-Compliance",
      env="mtsd-base", args=("--help",), confirm=True,
      notes="Preview creates a separate output tree. Applying redactions is deliberately not exposed here."),
]


# The automation registry is the authoritative inventory of principal
# workflows.  Curated entries above remain the shortest route to common safe
# actions; this generated section makes every registered workflow discoverable
# without maintaining a second, incomplete list by hand.
WORKFLOW_ALIASES = {
    "PromptDetect-App": "promptdetect",
    "MTSD-AnnotationQA-Review-UI": "mtsd-review",
    "AttrCls-Compare-UI": "attr-ui",
}
REGISTRY_LOAD_ERROR: str | None = None


def _workflow_is_read_only(spec: dict) -> bool:
    if spec.get("training") or spec.get("type") == "notebook":
        return False
    args = {str(arg).lower() for arg in spec.get("defaultArgs", [])}
    return bool(args & {"--dry-run", "--help", "--list", "--list-models", "--list-variants", "--plan"})


def _load_workflow_tools() -> list[Tool]:
    global REGISTRY_LOAD_ERROR
    try:
        payload = json.loads(WORKFLOW_REGISTRY_FILE.read_text(encoding="utf-8"))
        targets = payload["targets"]
        if not isinstance(targets, dict):
            raise ValueError("'targets' must be an object")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        REGISTRY_LOAD_ERROR = str(exc)
        return []

    generated: list[Tool] = []
    for name, spec in targets.items():
        if not isinstance(spec, dict):
            REGISTRY_LOAD_ERROR = f"target {name!r} is not an object"
            continue
        read_only = _workflow_is_read_only(spec)
        training = bool(spec.get("training"))
        redirect_id = WORKFLOW_ALIASES.get(name)
        generated.append(T(
            id=f"workflow:{name}",
            title=name,
            category="Complete Workflow Registry",
            description=str(spec.get("description", "Registered repository workflow.")),
            script=str(spec.get("path", "")) or None,
            args=tuple(str(arg) for arg in spec.get("defaultArgs", [])),
            cwd=".",
            env=str(spec.get("env", "")) or None,
            heavy=training,
            confirm=training or not read_only,
            read_only=read_only,
            kind=str(spec.get("type", "python")),
            no_user_site=bool(spec.get("noUserSite")),
            training=training,
            workflow_target=name,
            redirect_id=redirect_id,
            notes=(
                "Registry-controlled heavy/training workflow; a real run requires explicit confirmation."
                if training else
                "Loaded from Scripts/Automation/workflow_targets.json."
            ),
        ))
    return generated


WORKFLOW_TOOLS = _load_workflow_tools()
TOOLS = CURATED_TOOLS + WORKFLOW_TOOLS
TOOL_BY_ID = {tool.id.lower(): tool for tool in TOOLS}


def find_tool(tool_id: str) -> Tool | None:
    """Resolve a tool ID case-insensitively."""
    return TOOL_BY_ID.get(tool_id.lower())


def effective_tool(tool: Tool) -> Tool:
    """Resolve a registry alias to its curated managed implementation."""
    if not tool.redirect_id:
        return tool
    target = find_tool(tool.redirect_id)
    if target is None:
        raise RuntimeError(f"Launcher alias {tool.id!r} points to missing tool {tool.redirect_id!r}")
    return target


def say(message: str = "", style: str | None = None) -> None:
    if RICH:
        CONSOLE.print(message, style=style)
    else:
        print(message)


def repo_status() -> tuple[str, str]:
    return _REPO_BRANCH, _REPO_COMMIT


# ---------------------------------------------------------------------------
# Rendering helpers — double-buffered to eliminate flicker
# ---------------------------------------------------------------------------

def _render_to_string(render_fn) -> str:
    """Capture all Rich output produced by *render_fn* into a plain string."""
    buf = io.StringIO()
    tmp_console = Console(file=buf, force_terminal=True, color_system="truecolor", width=CONSOLE.width if CONSOLE else 120)
    render_fn(tmp_console)
    return buf.getvalue()


def _paint(frame: str) -> None:
    """Write a full frame to the terminal without any visible flicker."""
    sys.stdout.write("\033[H" + frame + "\033[J")
    sys.stdout.flush()


UVB_LOGO = """██╗   ██╗██╗   ██╗        ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗███╗   ███╗ █████╗ ██████╗ ██╗  ██╗
██║   ██║██║   ██║        ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║████╗ ████║██╔══██╗██╔══██╗██║ ██╔╝
██║   ██║██║   ██║███████╗██████╔╝█████╗  ██╔██╗ ██║██║     ███████║██╔████╔██║███████║██████╔╝█████╔╝
██║   ██║╚██╗ ██╔╝╚══════╝██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║██║╚██╔╝██║██╔══██║██╔══██╗██╔═██╗
╚██████╔╝ ╚████╔╝         ██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║██║ ╚═╝ ██║██║  ██║██║  ██║██║  ██╗
 ╚═════╝   ╚═══╝          ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝"""


def _render_header(c: Console) -> None:
    """Render the UVB banner + repo metadata into *c*."""
    branch, commit = repo_status()
    c.print(Panel.fit(
        f"[bold bright_cyan]{UVB_LOGO}[/]\n\n"
        "[bold white]URBAN VISION BENCHMARK[/]  [dim]// local research workspace[/]",
        border_style="bright_blue", padding=(1, 4),
    ))
    c.print(
        f"  [dim]branch[/] [bold cyan]{branch}[/]   [dim]commit[/] [bold white]{commit}[/]   "
        f"[dim]python[/] [bold white]{Path(sys.executable).name}[/]   [dim]platform[/] [bold white]{platform.system()}[/]\n"
    )


def header() -> None:
    """Print the header directly (used outside the navigation loop)."""
    branch, commit = repo_status()
    if RICH:
        _render_header(CONSOLE)
    else:
        print(UVB_LOGO + "\nURBAN VISION BENCHMARK // local research workspace")
        print(f"repo={ROOT.name} branch={branch} commit={commit} python={sys.executable} os={platform.system()}")


def clear_screen() -> None:
    sys.stdout.write("\033[H\033[J")
    sys.stdout.flush()


@contextlib.contextmanager
def fullscreen_tui():
    """Enter alternate screen buffer and hide cursor for a glitch-free TUI."""
    sys.stdout.write("\033[?1049h\033[?25l")
    sys.stdout.flush()
    try:
        yield
    finally:
        sys.stdout.write("\033[?1049l\033[?25h")
        sys.stdout.flush()


def read_key() -> str:
    """Read one navigation key without requiring Enter; works in Windows Terminal."""
    if os.name == "nt":
        import msvcrt

        key = msvcrt.getwch()
        if key in {"\x00", "\xe0"}:
            key = msvcrt.getwch()
            return {"H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT"}.get(key, "")
        return {"\r": "ENTER", "\x1b": "ESC", "\x08": "BACK", "\x03": "CTRL_C"}.get(key, key.lower())

    if not sys.stdin.isatty():
        return input().strip().lower()
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
        if key == "\x1b" and select.select([sys.stdin], [], [], 0.05)[0]:
            key += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
    return {"\x1b[A": "UP", "\x1b[B": "DOWN", "\r": "ENTER", "\n": "ENTER", "\x1b": "ESC", "\x7f": "BACK", "\x03": "CTRL_C"}.get(key, key.lower())


def tool_badges(tool: Tool) -> str:
    badges: list[str] = []
    display_tool = effective_tool(tool) if tool.redirect_id else tool
    if display_tool.is_web_app:
        badges.append(f"LOCAL WEB APP :{display_tool.port}")
    elif tool.kind == "notebook":
        badges.append("NOTEBOOK COPY")
    if tool.training:
        badges.append("HEAVY/TRAINING")
    if tool.env:
        badges.append(tool.env)
    if tool.confirm:
        badges.append("CONFIRM")
    elif tool.read_only:
        badges.append("READ-ONLY")
    return "  ·  ".join(badges)


def navigation_menu(title: str, subtitle: str, options: Sequence[tuple[str, str, str]], *, allow_search: bool = False, allow_quit: bool = False) -> tuple[str, int | None]:
    """Return (action, selected index); keyboard controls are shown on screen."""
    if not options:
        return "back", None
    selected = 0

    controls = "↑/↓ move   Enter select   Esc back"
    if allow_search:
        controls += "   / search"
    if allow_quit:
        controls += "   q quit"

    with fullscreen_tui():
        while True:
            terminal_height = shutil.get_terminal_size(fallback=(120, 36)).lines
            max_visible = max(3, min(10, (terminal_height - 18) // 3))
            start = min(max(0, selected - max_visible // 2), max(0, len(options) - max_visible))
            stop = min(len(options), start + max_visible)
            visible_options = options[start:stop]
            position = f"items {start + 1}–{stop} of {len(options)}" if len(options) > max_visible else ""
            if RICH:
                def _build(c: Console, *, _sel=selected, _opts=visible_options,
                           _offset=start, _title=title, _sub=subtitle,
                           _ctrl=controls, _position=position) -> None:
                    _render_header(c)
                    c.print(f" [dim]{_sub}[/]\n")

                    table = Table(show_header=False, show_edge=False, box=None, padding=(0, 1, 0, 0))
                    table.add_column("Pointer", justify="right", style="bold bright_cyan", width=2)
                    table.add_column("Main")

                    for local_idx, (label, description, badge) in enumerate(_opts):
                        idx = _offset + local_idx
                        if idx == _sel:
                            ptr = "▶"
                            t_text = f"[bold bright_cyan]{label}[/]"
                            d_text = f"[white]{description}[/]"
                        else:
                            ptr = " "
                            t_text = f"[white]{label}[/]"
                            d_text = f"[dim]{description}[/]"
                        b_text = f"  [dim]{badge}[/]" if badge else ""
                        row = f"{t_text}{b_text}\n  {d_text}"
                        table.add_row(ptr, row)
                        if local_idx < len(_opts) - 1:
                            table.add_row("", "")

                    c.print(Panel(
                        table,
                        title=f"[bold white]{_title}[/]",
                        title_align="left",
                        subtitle=f"[dim]{_position + '   ' if _position else ''}{_ctrl}[/]",
                        subtitle_align="left",
                        border_style="bright_black",
                        padding=(1, 2),
                    ))

                frame = _render_to_string(_build)
            else:
                lines: list[str] = []
                lines.append(f"\n{title}")
                lines.append(subtitle)
                lines.append("")
                for local_idx, (label, description, badge) in enumerate(visible_options):
                    idx = start + local_idx
                    pointer = ">" if idx == selected else " "
                    suffix = f"  [{badge}]" if badge else ""
                    lines.append(f"{pointer} {label}{suffix}")
                    lines.append(f"    {description}")
                lines.append(f"\n{position + '   ' if position else ''}{controls}")
                frame = "\n".join(lines) + "\n"

            _paint(frame)

            key = read_key()
            if key in {"UP", "k"}:
                selected = (selected - 1) % len(options)
            elif key in {"DOWN", "j"}:
                selected = (selected + 1) % len(options)
            elif key == "ENTER":
                return "select", selected
            elif allow_search and key == "/":
                return "search", None
            elif key == "CTRL_C":
                raise KeyboardInterrupt
            elif key in {"ESC", "BACK", "b"}:
                return "quit" if allow_quit else "back", None
            elif allow_quit and key == "q":
                return "quit", None


# ---------------------------------------------------------------------------
# Command construction and environment discovery
# ---------------------------------------------------------------------------

def command_for(tool: Tool, port: int | None = None, *, managed: bool = False,
                notebook_output: Path | None = None) -> list[str]:
    """Build the exact command line for a tool.

    When *managed* is true (a launcher-managed web app), the app's
    no-browser flag is appended so the app never opens its own tab; the
    launcher opens the browser exactly once, after readiness.
    """
    if tool.script is None:
        return []
    python_cmd = [sys.executable]
    if tool.env:
        conda = find_conda()
        if conda:
            python_cmd = [str(conda), "run", "-n", tool.env, "python"]
    if tool.no_user_site:
        python_cmd.append("-s")
    if tool.kind == "notebook":
        output = notebook_output or (
            EXECUTED_NOTEBOOK_DIR /
            f"{tool.workflow_target or tool.id}-<timestamp>.ipynb"
        )
        return python_cmd + [
            "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
            str(ROOT / tool.script),
            "--output", output.name,
            "--output-dir", str(output.parent),
            "--ExecutePreprocessor.timeout=14400",
        ]
    args = list(tool.args)
    if tool.port_arg and port:
        args += [tool.port_arg, str(port)]
    if managed and tool.no_browser_arg:
        args.append(tool.no_browser_arg)
    script_path = Path(tool.script)
    cwd_path = Path(tool.cwd)
    try:
        script_arg = str(script_path.relative_to(cwd_path))
    except ValueError:
        script_arg = str(script_path)
    return python_cmd + [script_arg] + args


def display_command(cmd: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(cmd)) if os.name == "nt" else shlex.join(cmd)


def find_conda() -> Path | None:
    found = shutil.which("conda")
    if found:
        return Path(found)
    candidates = [
        Path.home() / "anaconda3/bin/conda",
        Path.home() / "miniconda3/bin/conda",
        Path.home() / "mambaforge/bin/conda",
        Path.home() / "anaconda3/Scripts/conda.exe",
        Path.home() / "miniconda3/Scripts/conda.exe",
        Path(os.environ.get("PROGRAMDATA", "C:/ProgramData")) / "anaconda3/Scripts/conda.exe",
        Path(os.environ.get("LOCALAPPDATA", "C:/Users/Default/AppData/Local")) / "anaconda3/Scripts/conda.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def conda_envs() -> set[str]:
    conda = find_conda()
    if not conda:
        return set()
    try:
        output = subprocess.check_output(
            [str(conda), "env", "list", "--json"],
            text=True, stderr=subprocess.STDOUT, timeout=20,
        )
        payload = json.loads(output)
        paths = payload.get("envs", [])
        root_prefix = payload.get("root_prefix")
    except (OSError, subprocess.SubprocessError, ValueError):
        return set()
    names = {Path(path).name for path in paths if path != root_prefix}
    if paths:
        names.add("base")
    return names


def environment_issue(tool: Tool) -> str | None:
    """Return a clear preflight error when a named Conda environment is absent.

    A missing environment used to surface as a brief ``conda run`` failure after
    the launcher had already tried to start a workflow.  Catching it here keeps
    the menu responsive and tells the researcher exactly what needs installing.
    """
    if not tool.env:
        return None
    conda = find_conda()
    if conda is None:
        return None
    available = conda_envs()
    if tool.env in available:
        return None
    found = ", ".join(sorted(available)) if available else "none detected"
    if os.name == "nt":
        setup = (
            ".\\Requirements\\CondaEnvironments\\setup_conda_env.ps1 "
            f"-Name {tool.env}"
        )
    else:
        setup = f"bash Requirements/CondaEnvironments/setup_conda_env.sh {tool.env}"
    return (
        f"Required Conda environment '{tool.env}' is not installed. "
        f"Detected environments: {found}. "
        f"From the repository root, run: {setup}"
    )


# ---------------------------------------------------------------------------
# Port utilities
# ---------------------------------------------------------------------------

def port_is_listening(port: int, host: str = "127.0.0.1", timeout: float = 0.25) -> bool:
    """True when something accepts TCP connections on host:port."""
    with socket.socket() as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def port_info(port: int) -> str:
    return "IN USE" if port_is_listening(port) else "free"


def find_free_port(preferred: int, attempts: int = 25) -> int | None:
    """Return the first port >= preferred that can actually be bound."""
    for candidate in range(preferred, min(preferred + attempts, 65536)):
        try:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", candidate))
            return candidate
        except OSError:
            continue
    return None


def describe_port_owner(port: int) -> str | None:
    """Best-effort description of the process listening on *port* (never kills)."""
    try:
        if os.name == "nt":
            out = subprocess.check_output(["netstat", "-ano", "-p", "tcp"], text=True, timeout=10)
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[0] == "TCP" and parts[1].endswith(f":{port}") and parts[3] == "LISTENING":
                    pid = parts[4]
                    try:
                        tl = subprocess.check_output(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], text=True, timeout=10)
                        name = tl.split(",")[0].strip('" ') if "," in tl else "?"
                    except (OSError, subprocess.SubprocessError):
                        name = "?"
                    return f"PID {pid} ({name})"
        else:
            out = subprocess.check_output(["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpc"], text=True, timeout=10)
            pid = name = "?"
            for token in out.split():
                if token.startswith("p"):
                    pid = token[1:]
                elif token.startswith("c"):
                    name = token[1:]
            return f"PID {pid} ({name})"
    except (OSError, subprocess.SubprocessError):
        pass
    return None


# ---------------------------------------------------------------------------
# Process-tree termination and PID verification (only ever applied to
# processes this launcher started, or PIDs verified to match a stored
# launcher command line)
# ---------------------------------------------------------------------------

def terminate_process_tree(pid: int) -> None:
    """Terminate *pid* and its complete descendant tree, deterministically."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, text=True)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.kill(pid, signal.SIGTERM)


def pid_command_line(pid: int) -> str | None:
    """Return the command line of *pid*, or None when it does not exist."""
    try:
        if os.name == "nt":
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
                text=True, timeout=15, stderr=subprocess.DEVNULL)
            return out.strip() or None
        cmdline = Path(f"/proc/{pid}/cmdline")
        if cmdline.exists():
            return cmdline.read_bytes().replace(b"\x00", b" ").decode(errors="replace").strip() or None
        out = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="],
            text=True, timeout=10, stderr=subprocess.DEVNULL,
        )
        return out.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


# ---------------------------------------------------------------------------
# Managed local web applications
# ---------------------------------------------------------------------------

@dataclass
class ManagedApp:
    tool: Tool
    process: subprocess.Popen
    port: int
    url: str
    command: list[str]
    log_path: Path
    started_at: str
    log_handle: object = None

    @property
    def running(self) -> bool:
        return self.process.poll() is None


def _default_spawn(cmd: list[str], cwd: Path, log_handle) -> subprocess.Popen:
    kwargs: dict = {"cwd": str(cwd), "stdout": log_handle, "stderr": subprocess.STDOUT,
                    "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        # A new process group detaches the child from the console Ctrl+C
        # group, so an interrupt in the launcher never half-kills the app;
        # cleanup is always the explicit taskkill of the whole tree.
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)


def _default_browser_open(url: str) -> bool:
    try:
        return bool(webbrowser.open(url))
    except Exception:
        return False


class AppManager:
    """Tracks and controls every local web application the launcher started.

    Only processes spawned through this manager are ever terminated. The
    manager persists {pid, port, command} to the state file so that a crashed
    launcher can *report* leftovers on the next start; a persisted PID is
    offered for termination only after its current command line has been
    re-read from the OS and matches the stored launcher command.
    """

    def __init__(self, *, spawn: Callable = _default_spawn,
                 terminate: Callable[[int], None] = terminate_process_tree,
                 port_check: Callable[[int], bool] = port_is_listening,
                 browser_open: Callable[[str], bool] = _default_browser_open,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 state_file: Path = STATE_FILE,
                 log_dir: Path = LOG_DIR):
        self.apps: dict[str, ManagedApp] = {}
        self._spawn = spawn
        self._terminate = terminate
        self._port_check = port_check
        self._browser_open = browser_open
        self._sleep = sleep
        self._clock = clock
        self._state_file = state_file
        self._log_dir = log_dir

    # -- state file -------------------------------------------------------
    def _write_state(self) -> None:
        payload = {"apps": {
            app.tool.id: {"pid": app.process.pid, "port": app.port,
                          "command": display_command(app.command),
                          "started_at": app.started_at}
            for app in self.apps.values() if app.running}}
        try:
            if payload["apps"]:
                temp = self._state_file.with_suffix(self._state_file.suffix + ".tmp")
                temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                temp.replace(self._state_file)
            elif self._state_file.exists():
                self._state_file.unlink()
        except OSError:
            pass

    def read_stale_state(self) -> list[dict]:
        """Entries persisted by a previous launcher run (not this process)."""
        try:
            payload = json.loads(self._state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [{"tool_id": tool_id, **info}
                for tool_id, info in payload.get("apps", {}).items()]

    def clear_state(self) -> None:
        with contextlib.suppress(OSError):
            if self._state_file.exists():
                self._state_file.unlink()

    # -- lifecycle ----------------------------------------------------------
    def running_app(self, tool_id: str) -> ManagedApp | None:
        app = self.apps.get(tool_id)
        if app and not app.running:
            self._forget(tool_id)
            return None
        return app

    def running_apps(self) -> list[ManagedApp]:
        for tool_id in list(self.apps):
            self.running_app(tool_id)
        return list(self.apps.values())

    def _forget(self, tool_id: str) -> None:
        app = self.apps.pop(tool_id, None)
        if app and app.log_handle:
            with contextlib.suppress(OSError):
                app.log_handle.close()
        self._write_state()

    def start(self, tool: Tool, port: int) -> ManagedApp:
        """Spawn the app process (no readiness wait, no browser)."""
        cmd = command_for(tool, port, managed=True)
        self._log_dir.mkdir(exist_ok=True)
        log_path = self._log_dir / f"{tool.id.replace(':', '-')}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.log"
        log_handle = open(log_path, "w", encoding="utf-8", errors="replace")
        process = self._spawn(cmd, ROOT / tool.cwd, log_handle)
        app = ManagedApp(tool=tool, process=process, port=port,
                         url=tool.url.format(port=port), command=cmd,
                         log_path=log_path,
                         started_at=datetime.now().isoformat(timespec="seconds"),
                         log_handle=log_handle)
        self.apps[tool.id] = app
        self._write_state()
        return app

    def wait_until_ready(self, app: ManagedApp, timeout: float,
                         poll_interval: float = 0.5,
                         on_tick: Callable[[float], None] | None = None) -> str:
        """Poll the app's port until ready. Returns 'ready', 'exited' or 'timeout'."""
        deadline = self._clock() + timeout
        while True:
            if not app.running:
                return "exited"
            if self._port_check(app.port):
                return "ready"
            if self._clock() >= deadline:
                return "timeout"
            if on_tick:
                on_tick(max(0.0, deadline - self._clock()))
            self._sleep(poll_interval)

    def open_browser(self, app: ManagedApp) -> bool:
        return self._browser_open(app.url)

    def stop(self, tool_id: str, wait_timeout: float = 15.0) -> bool:
        """Terminate the app's whole process tree; True when it is gone."""
        app = self.apps.get(tool_id)
        if app is None:
            return True
        if app.running:
            self._terminate(app.process.pid)
            try:
                app.process.wait(timeout=wait_timeout)
            except subprocess.TimeoutExpired:
                return False
        self._forget(tool_id)
        return True

    def stop_all(self) -> None:
        for tool_id in list(self.apps):
            self.stop(tool_id)
        self._write_state()

    def log_tail(self, app: ManagedApp, lines: int = 15) -> list[str]:
        try:
            content = app.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return content[-lines:]
        except OSError:
            return []


MANAGER = AppManager()


def _cleanup_at_exit() -> None:  # pragma: no cover - process teardown
    MANAGER.stop_all()

atexit.register(_cleanup_at_exit)


# ---------------------------------------------------------------------------
# Stale-state handling (previous launcher crashed or was killed hard)
# ---------------------------------------------------------------------------

def report_stale_state(manager: AppManager, *, interactive: bool) -> None:
    entries = manager.read_stale_state()
    if not entries:
        return
    say("\nA previous launcher session left state behind:", "yellow")
    verified: list[dict] = []
    for entry in entries:
        pid, stored_cmd = int(entry["pid"]), entry.get("command", "")
        current_cmd = pid_command_line(pid)
        if current_cmd is None:
            say(f"  {entry['tool_id']}: PID {pid} no longer exists — stale entry dropped.")
        elif stored_cmd and _commands_equivalent(current_cmd, stored_cmd):
            say(f"  {entry['tool_id']}: PID {pid} still running on port {entry.get('port')} "
                f"(verified as a launcher-started app).", "yellow")
            verified.append(entry)
        else:
            say(f"  {entry['tool_id']}: PID {pid} now belongs to a DIFFERENT process — "
                "not touched, stale entry dropped.", "yellow")
    if verified and interactive:
        if confirm("Stop these leftover launcher-started applications now?"):
            for entry in verified:
                terminate_process_tree(int(entry["pid"]))
                say(f"  stopped {entry['tool_id']} (PID {entry['pid']}).")
        else:
            say("  Left running. They keep their ports until stopped manually.")
    elif verified:
        say("  Re-run interactively to stop them, or stop them manually.", "yellow")
    manager.clear_state()


def _commands_equivalent(current: str, stored: str) -> bool:
    """Compare command lines loosely (quoting differs between APIs)."""
    normalise = lambda s: " ".join(s.replace('"', "").split()).lower()
    return normalise(stored) in normalise(current) or normalise(current) in normalise(stored)


# ---------------------------------------------------------------------------
# Tool information display
# ---------------------------------------------------------------------------

def show_tool(tool: Tool, port: int | None = None,
              notebook_output: Path | None = None) -> None:
    if tool.instruction:
        say(f"\n{tool.title}\n{tool.description}\n", "bold cyan")
        for item in tool.instruction.splitlines():
            path = ROOT / item
            if path.is_dir():
                say(f"  Open: {path}")
            elif path.is_file() and path.suffix.lower() in {".md", ".txt"}:
                say(f"  Read: {path}")
            else:
                say(f"  {item}")
        if tool.id == "labelstudio":
            say("\nRead the guide before starting Label Studio. The documented launcher uses port 8080 and may stop an existing listener.", "yellow")
        return
    cmd = command_for(
        tool, port, managed=tool.is_web_app,
        notebook_output=notebook_output,
    )
    say(f"\n{tool.title}\n{tool.description}", "bold cyan")
    say(f"Command: {display_command(cmd)}")
    say(f"Working directory: {ROOT / tool.cwd}")
    if tool.env:
        say(f"Conda environment: {tool.env}")
    if tool.no_user_site:
        say("Python user-site packages: disabled (-s)")
    if tool.kind == "notebook":
        say("Notebook source: preserved; execution writes a timestamped copy")
    if tool.is_web_app:
        say(f"URL: {tool.url.format(port=port or tool.port)}  (port {port or tool.port}: {port_info(port or tool.port)})")
        say(f"Note: {WEB_APP_NOTE}", "yellow")
    if tool.notes:
        say(f"Note: {tool.notes}", "yellow")


def confirm(prompt: str) -> bool:
    try:
        return input(f"{prompt} [y/N]: ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


# ---------------------------------------------------------------------------
# Port resolution for web apps
# ---------------------------------------------------------------------------

def resolve_port(tool: Tool, manager: AppManager) -> tuple[int | None, str]:
    """Choose a verified port for a web app.

    Returns (port, message). port is None when launching must be refused.
    Assumes the same-tool-already-running case was handled by the caller.
    """
    preferred = tool.port
    if not port_is_listening(preferred):
        return preferred, ""
    # The preferred port is occupied. Was it one of OUR apps (different tool)?
    owner_app = next((a for a in manager.running_apps() if a.port == preferred), None)
    if owner_app is not None:
        owner_desc = f"the running launcher app '{owner_app.tool.id}'"
    else:
        owner = describe_port_owner(preferred)
        owner_desc = f"an unrelated process ({owner})" if owner else "an unrelated process"
    if tool.port_arg:
        alternate = find_free_port(preferred + 1)
        if alternate is not None:
            return alternate, (f"Port {preferred} is in use by {owner_desc}; "
                               f"using verified free port {alternate} instead.")
        return None, f"Port {preferred} is in use by {owner_desc} and no free alternate port was found."
    return None, (f"Port {preferred} is in use by {owner_desc}. This application does not "
                  "support an alternate port, so the launch was refused. Free the port and retry.")


# ---------------------------------------------------------------------------
# Running tools
# ---------------------------------------------------------------------------

def run_instruction_tool(tool: Tool, dry_run: bool = False) -> None:
    show_tool(tool)
    if dry_run or not tool.id.startswith("open-"):
        return
    for item in tool.instruction.splitlines():
        target = ROOT / item
        if target.exists():
            if target.is_file() and target.suffix.lower() in {".html", ".htm"}:
                if not _default_browser_open(target.as_uri()):
                    say(f"Could not open a browser; open manually: {target}", "yellow")
            elif os.name == "nt":
                os.startfile(str(target))  # type: ignore[attr-defined]
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                try:
                    subprocess.run([opener, str(target)], check=False)
                except OSError as exc:
                    say(f"Could not open automatically ({exc}); open manually: {target}", "yellow")
        else:
            say(f"Missing: {target}", "red")


def run_cli_tool(tool: Tool, dry_run: bool = False, *, assume_yes: bool = False) -> int:
    """Run a non-web tool in the foreground, with full-tree cleanup on Ctrl+C."""
    if tool.path and not tool.path.exists():
        say(f"Missing script: {tool.path}", "red")
        return 2
    notebook_output = None
    if tool.kind == "notebook":
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        notebook_output = EXECUTED_NOTEBOOK_DIR / f"{tool.workflow_target or tool.id}-{stamp}.ipynb"
    cmd = command_for(tool, notebook_output=notebook_output)
    show_tool(tool, notebook_output=notebook_output)
    if dry_run:
        return 0
    issue = environment_issue(tool)
    if issue:
        say(issue, "red")
        return 2
    if tool.confirm and not assume_yes and not confirm(
        "Proceed with this HEAVY/TRAINING workflow?" if tool.training else "Proceed with this launch?"
    ):
        say("Cancelled.", "yellow")
        return 0
    if tool.env and not find_conda():
        if tool.heavy:
            say("Conda was not found; this launch needs its documented environment and was not started.", "red")
            return 2
        say("Conda was not found; using the current Python interpreter for this safe command.", "yellow")
    process = None
    try:
        if notebook_output is not None:
            notebook_output.parent.mkdir(parents=True, exist_ok=True)
        process = _default_spawn_foreground(cmd, ROOT / tool.cwd)
        code = process.wait()
        say(f"\nFinished with exit code {code}.", "green" if code == 0 else "red")
        if code == 0 and notebook_output is not None:
            say(f"Executed notebook copy: {notebook_output}", "green")
        return code
    except KeyboardInterrupt:
        say("\nInterrupted — stopping the tool's full process tree...", "yellow")
        if process is not None:
            terminate_process_tree(process.pid)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=15)
        say("Stopped.", "yellow")
        return 130
    except OSError as exc:
        say(f"Could not launch: {exc}", "red")
        return 1


def _default_spawn_foreground(cmd: list[str], cwd: Path) -> subprocess.Popen:
    kwargs: dict = {"cwd": str(cwd)}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)


def launch_web_app(tool: Tool, manager: AppManager, *, dry_run: bool = False,
                   open_browser: bool = True, ready_timeout: float | None = None,
                   interactive: bool = True, assume_yes: bool = False) -> ManagedApp | None:
    """Start a managed local web application with readiness polling.

    The browser is opened only after the app's port actually accepts
    connections; a failed or timed-out startup opens no browser tab.
    """
    if tool.path and not tool.path.exists():
        say(f"Missing script: {tool.path}", "red")
        return None

    existing = manager.running_app(tool.id)
    if existing is not None:
        if interactive:
            handle_running_app(tool, manager)
        else:
            say(f"'{tool.id}' is already running at {existing.url} (PID {existing.process.pid}).", "yellow")
            say("Reopen it in the browser, or stop it from the launcher menu — a duplicate was not launched.")
        return existing

    if dry_run:
        show_tool(tool, tool.port)
        return None

    port, message = resolve_port(tool, manager)
    if message:
        say(message, "yellow" if port else "red")
    if port is None:
        return None

    show_tool(tool, port)
    issue = environment_issue(tool)
    if issue:
        say(issue, "red")
        return None
    if tool.confirm and not assume_yes:
        if not interactive:
            say("This launch requires confirmation; re-run with --yes from a non-interactive shell.", "red")
            return None
        if not confirm("Proceed with this launch?"):
            say("Cancelled.", "yellow")
            return None
    if tool.env and not find_conda():
        say("Conda was not found; this local web application needs its documented environment and was not started.", "red")
        return None

    app = manager.start(tool, port)
    timeout = ready_timeout if ready_timeout is not None else tool.ready_timeout
    say(f"Starting… polling {app.url} until the server is ready (timeout {int(timeout)}s). Log: {app.log_path}")

    def _tick(remaining: float) -> None:
        sys.stdout.write(f"\r  waiting for {app.url}  ({int(remaining)}s left) ")
        sys.stdout.flush()

    status = manager.wait_until_ready(app, timeout, on_tick=_tick if interactive else None)
    sys.stdout.write("\r" + " " * 60 + "\r")

    if status == "exited":
        code = app.process.poll()
        say(f"Startup FAILED: the application exited with code {code} before becoming ready.", "red")
        for line in manager.log_tail(app):
            say(f"  | {line}", "dim" if RICH else None)
        say(f"Full log: {app.log_path}", "yellow")
        manager.stop(tool.id)
        return None
    if status == "timeout":
        say(f"Startup TIMED OUT after {int(timeout)}s: {app.url} never became ready.", "red")
        say("The process was stopped and no browser tab was opened. "
            f"Inspect the log and retry (a heavy model load may need --ready-timeout): {app.log_path}", "yellow")
        manager.stop(tool.id)
        return None

    say(f"Ready: {app.url}  (PID {app.process.pid})", "green")
    say(WEB_APP_NOTE, "yellow")
    if open_browser and tool.browser:
        if not manager.open_browser(app):
            say(f"Could not open a browser automatically; open manually: {app.url}", "yellow")
    return app


def handle_running_app(tool: Tool, manager: AppManager) -> None:
    """Menu for a web app that is already running: reopen / restart / stop."""
    while True:
        app = manager.running_app(tool.id)
        if app is None:
            say(f"'{tool.id}' is no longer running.", "yellow")
            return
        action, index = navigation_menu(
            f"{tool.title} — RUNNING",
            f"{app.url}  ·  PID {app.process.pid}  ·  started {app.started_at}  ·  {WEB_APP_NOTE}",
            [
                ("Reopen in browser", "Open the running application's URL in a new browser tab.", ""),
                ("Show status and recent log", "Print process status, URL and the last log lines.", ""),
                ("Restart", "Stop the full process tree, then start the application again.", ""),
                ("Stop", "Terminate the application's full process tree (conda run, server, workers).", ""),
                ("Back (keep running)", "Return to the launcher; the application keeps running.", ""),
            ],
        )
        if action != "select" or index is None or index == 4:
            return
        clear_screen(); header()
        if index == 0:
            if not manager.open_browser(app):
                say(f"Could not open a browser automatically; open manually: {app.url}", "yellow")
            else:
                say(f"Reopened {app.url}", "green")
        elif index == 1:
            say(f"\n{tool.title}", "bold cyan")
            say(f"  status : {'running' if app.running else 'exited'}")
            say(f"  URL    : {app.url}")
            say(f"  PID    : {app.process.pid}")
            say(f"  started: {app.started_at}")
            say(f"  log    : {app.log_path}")
            for line in manager.log_tail(app):
                say(f"  | {line}")
        elif index == 2:
            say("Restarting…", "yellow")
            if not manager.stop(tool.id):
                say("The old process tree did not terminate in time; not restarting.", "red")
            else:
                launch_web_app(tool, manager)
            input("\nPress Enter to continue...")
            return
        elif index == 3:
            if manager.stop(tool.id):
                say("Stopped (full process tree).", "green")
            else:
                say("The process tree did not terminate within the wait window; check Task Manager.", "red")
            input("\nPress Enter to continue...")
            return
        input("\nPress Enter to continue...")


def run_tool(tool: Tool, dry_run: bool = False, *, manager: AppManager | None = None,
             open_browser: bool = True, ready_timeout: float | None = None,
             interactive: bool = True, assume_yes: bool = False) -> int:
    manager = manager or MANAGER
    tool = effective_tool(tool)
    if tool.instruction:
        run_instruction_tool(tool, dry_run)
        return 0
    if tool.is_web_app:
        launch_web_app(tool, manager, dry_run=dry_run, open_browser=open_browser,
                       ready_timeout=ready_timeout, interactive=interactive,
                       assume_yes=assume_yes)
        return 0
    return run_cli_tool(tool, dry_run, assume_yes=assume_yes)


# ---------------------------------------------------------------------------
# Listing, doctor, menus
# ---------------------------------------------------------------------------

def list_tools() -> None:
    for category in dict.fromkeys(t.category for t in TOOLS):
        say(f"\n{category}", "bold cyan")
        for tool in [x for x in TOOLS if x.category == category]:
            suffix = f"  [{tool.env}]" if tool.env else ""
            if tool.port:
                suffix += f"  :{tool.port}"
            say(f"  {tool.id:<28} {tool.title}{suffix}")
            say(f"  {'':<28} {tool.description}", "dim")


def search_tools(query: str) -> list[Tool]:
    terms = query.lower().split()
    if not terms:
        return []
    matches = []
    for tool in TOOLS:
        haystack = " ".join((
            tool.id, tool.title, tool.category, tool.description, tool.notes,
            tool.script or "", " ".join(tool.args),
        )).lower()
        if "promptdetect" in haystack:
            haystack += " prompted prompting zero-shot sensitivity paraphrase"
        if all(term in haystack for term in terms):
            matches.append(tool)
    return matches


def catalog_findings() -> list[str]:
    findings: list[str] = []
    ids = [tool.id.lower() for tool in TOOLS]
    duplicates = sorted({tool_id for tool_id in ids if ids.count(tool_id) > 1})
    findings.extend(f"duplicate tool id: {tool_id}" for tool_id in duplicates)
    for tool in TOOLS:
        if tool.script and not tool.path.exists():
            findings.append(f"missing script for {tool.id}: {tool.script}")
        if not (ROOT / tool.cwd).is_dir():
            findings.append(f"missing working directory for {tool.id}: {tool.cwd}")
        if tool.kind not in {"python", "notebook"}:
            findings.append(f"unsupported kind for {tool.id}: {tool.kind}")
        if tool.redirect_id and find_tool(tool.redirect_id) is None:
            findings.append(f"broken alias for {tool.id}: {tool.redirect_id}")
    if REGISTRY_LOAD_ERROR:
        findings.append(f"workflow registry load error: {REGISTRY_LOAD_ERROR}")
    return findings


def doctor() -> None:
    say("UVB launcher doctor", "bold cyan")
    conda = find_conda()
    say(f"  conda: {'OK - ' + str(conda) if conda else 'NOT FOUND'}", "green" if conda else "yellow")
    envs = conda_envs()
    for env in sorted({t.env for t in TOOLS if t.env}):
        say(f"  env {env}: {'OK' if env in envs else 'not detected'}", "green" if env in envs else "yellow")
    for port in sorted({t.port for t in TOOLS if t.port}):
        say(f"  port {port}: {port_info(port)}")
    important = [t for t in TOOLS if t.script]
    missing = [t.script for t in important if t.path and not t.path.exists()]
    say(f"  configured scripts: {len(important) - len(missing)}/{len(important)} present", "green" if not missing else "red")
    if missing:
        for item in missing:
            say(f"    missing: {item}", "red")
    findings = catalog_findings()
    say(
        f"  launcher catalog: {'OK' if not findings else str(len(findings)) + ' issue(s)'}; "
        f"{len(CURATED_TOOLS)} curated actions + {len(WORKFLOW_TOOLS)} registered workflows",
        "green" if not findings else "red",
    )
    for finding in findings:
        say(f"    {finding}", "red")
    stale = MANAGER.read_stale_state()
    say(f"  launcher state file: {'PRESENT — ' + str(len(stale)) + ' recorded app(s) from a previous session' if stale else 'clean'}",
        "yellow" if stale else "green")
    env_file = ROOT / ".env"
    env_has_wandb = False
    if env_file.exists():
        try:
            env_has_wandb = any(
                line.strip().startswith("WANDB_API_KEY=") and line.strip().split("=", 1)[1].strip()
                for line in env_file.read_text(encoding="utf-8").splitlines()
            )
        except OSError:
            pass
    say(f"  .env: {'present (secret values not printed)' if env_file.exists() else 'not present'}")
    say(f"  W&B key: {'present (value hidden)' if (os.getenv('WANDB_API_KEY') or env_has_wandb) else 'not detected'}")
    say("  sqlite guard: label_studio.sqlite3 is never opened by this launcher")


CATEGORY_DESCRIPTIONS = {
    "Local Web Applications": ("Long-running local apps served on 127.0.0.1. The launcher starts them, waits for readiness, "
                               "opens the browser, and can reopen, restart or stop them. Closing a browser tab never stops a server."),
    "Annotation and QA Tools": "Audit, review and carefully apply MTSD annotation fixes; Label Studio stays instructions-only.",
    "Dataset Analysis and EDA": "Inspect MDWD and MTSD structure, metadata, samples and map outputs.",
    "Evaluation and Benchmarks": "Validate benchmark plans, integrity checks and qualitative-analysis tooling.",
    "Documentation and Reports": "Open the folders and HTML outputs produced by the repository tools.",
    "Repository Maintenance": "Run read-only repository checks and view safe GDPR workflow guidance.",
    "Complete Workflow Registry": ("Every target from Scripts/Automation/workflow_targets.json, including notebooks, "
                                   "dataset preparation, final exports, full inference and explicitly gated training."),
}


def running_apps_menu(manager: AppManager) -> None:
    while True:
        apps = manager.running_apps()
        if not apps:
            say("No launcher-managed applications are running.", "yellow")
            input("\nPress Enter to return...")
            return
        options = [(f"{a.tool.title}", f"{a.url}  ·  PID {a.process.pid}  ·  started {a.started_at}",
                    f"LOCAL WEB APP :{a.port}") for a in apps]
        action, index = navigation_menu(
            f"Running local web applications ({len(apps)})",
            WEB_APP_NOTE + "  Select an application to reopen, restart or stop it.",
            options,
        )
        if action != "select" or index is None:
            return
        handle_running_app(apps[index].tool, manager)


def menu() -> None:
    manager = MANAGER
    while True:
        categories = list(dict.fromkeys(t.category for t in TOOLS))
        running = manager.running_apps()
        category_options = []
        if running:
            category_options.append((f"Running local web applications ({len(running)})",
                                     "Reopen, restart or stop the applications this launcher started.",
                                     "MANAGED"))
        category_options += [(category, CATEGORY_DESCRIPTIONS[category],
                              f"{sum(t.category == category for t in TOOLS)} tools") for category in categories]
        action, index = navigation_menu(
            "Choose a workspace",
            "Everything below is repository-local. Items marked CONFIRM need an explicit go-ahead.",
            category_options,
            allow_search=True,
            allow_quit=True,
        )
        if action == "quit":
            if manager.running_apps():
                say("Stopping all launcher-managed applications before exit…", "yellow")
                manager.stop_all()
                say("All managed applications stopped.", "green")
            return
        if action == "search":
            clear_screen()
            header()
            try:
                query = input("\nSearch tools (name or purpose; blank to cancel)> ").strip().lower()
            except EOFError:
                continue
            if not query:
                continue
            matches = search_tools(query)
            if not matches:
                say("No matching tools.", "yellow")
                input("Press Enter to return...")
                continue
            action, match_index = navigation_menu(
                "Search results",
                f"Matching: {query}",
                [(t.title, t.description, tool_badges(t)) for t in matches],
            )
            if action == "select" and match_index is not None:
                clear_screen(); header()
                run_tool(matches[match_index])
                input("\nPress Enter to return to the launcher...")
            continue
        if index is None:
            continue
        if running and index == 0:
            running_apps_menu(manager)
            continue
        category_index = index - (1 if running else 0)
        selected = [t for t in TOOLS if t.category == categories[category_index]]
        action, tool_index = navigation_menu(
            categories[category_index],
            "Choose an action. The label tells you whether it is read-only, a local web app, or requires confirmation.",
            [(t.title, t.description, tool_badges(t)) for t in selected],
        )
        if action == "select" and tool_index is not None:
            clear_screen(); header()
            run_tool(selected[tool_index])
            input("\nPress Enter to return to the launcher...")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safe Urban Vision Benchmark launcher (local web apps are managed: "
                    "readiness-polled startup, reopen/restart/stop, deterministic process-tree cleanup)")
    parser.add_argument("tool", nargs="?", help="direct tool id, e.g. promptdetect, attr-ui, health")
    parser.add_argument("--list", action="store_true", help="list configured tools")
    parser.add_argument("--search", metavar="QUERY", help="list tools matching all words in QUERY")
    parser.add_argument("--doctor", action="store_true", help="check environments, ports, paths and leftover state")
    parser.add_argument("--dry-run", action="store_true", help="print configured commands without running them")
    parser.add_argument("--yes", action="store_true",
                        help="accept the launcher's confirmation gate (underlying tools may still prompt)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser after a web app becomes ready")
    parser.add_argument("--ready-timeout", type=float, default=None,
                        help="seconds to wait for a local web app to become ready (default: per-tool)")
    args = parser.parse_args()
    if args.list:
        list_tools(); return 0
    if args.search is not None:
        matches = search_tools(args.search)
        if not matches:
            say(f"No tools match: {args.search}", "yellow")
            return 1
        for tool in matches:
            say(f"{tool.id:<42} {tool.title}  [{tool_badges(tool)}]")
        return 0
    if args.doctor:
        doctor(); return 0
    if args.dry_run:
        if args.tool:
            tool = find_tool(args.tool)
            if not tool:
                say(f"Unknown tool id: {args.tool}", "red"); return 2
            run_tool(tool, dry_run=True, assume_yes=args.yes)
        else:
            for tool in TOOLS:
                if tool.script:
                    show_tool(effective_tool(tool), effective_tool(tool).port)
        return 0
    try:
        if args.tool:
            selected_tool = find_tool(args.tool)
            tool = effective_tool(selected_tool) if selected_tool else None
            if not tool:
                say(f"Unknown tool id: {args.tool}. Use --list.", "red"); return 2
            report_stale_state(MANAGER, interactive=sys.stdin.isatty())
            if tool.is_web_app:
                app = launch_web_app(tool, MANAGER, open_browser=not args.no_browser,
                                     ready_timeout=args.ready_timeout,
                                     interactive=sys.stdin.isatty(), assume_yes=args.yes)
                if app is None:
                    return 1
                say("Press Ctrl+C to stop the application and exit.", "yellow")
                try:
                    while app.running:
                        time.sleep(1.0)
                    say(f"The application exited on its own (code {app.process.poll()}).", "yellow")
                except KeyboardInterrupt:
                    say("\nStopping the application (full process tree)…", "yellow")
                finally:
                    MANAGER.stop_all()
                    say("Cleaned up.", "green")
                return 0
            return run_tool(tool, interactive=sys.stdin.isatty(), assume_yes=args.yes)
        report_stale_state(MANAGER, interactive=sys.stdin.isatty())
        menu()
        return 0
    except KeyboardInterrupt:
        say("\nInterrupted — stopping all launcher-managed applications…", "yellow")
        return 130
    finally:
        MANAGER.stop_all()


if __name__ == "__main__":
    raise SystemExit(main())
