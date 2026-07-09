#!/usr/bin/env python3
"""Urban Vision Benchmark - safe, repository-root launcher.

This file intentionally uses only the Python standard library.  If Rich is
installed it improves the presentation, but it is never required.
"""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

if os.name == "nt":
    os.system("")  # Enable VT100 escape sequences in Windows cmd/powershell


ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / ".uvb_launcher_state.json"  # reserved; no state is written yet

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    RICH = True
    CONSOLE = Console()
except ImportError:  # pragma: no cover - exercised on minimal installations
    RICH = False
    CONSOLE = None


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
    url: str | None = None
    browser: bool = False
    heavy: bool = False
    confirm: bool = False
    read_only: bool = True
    notes: str = ""
    instruction: str | None = None

    @property
    def path(self) -> Path | None:
        return ROOT / self.script if self.script else None


def T(**kwargs) -> Tool:
    return Tool(**kwargs)


TOOLS = [
    T(id="promptdetect", title="PromptDetect — manual image inference", category="Model and Inference UIs", description="Open a local Gradio app to test a prompt against one image. You choose and load the model inside the app.", script="Scripts/Other-Scripts/PromptDetect/app.py", cwd="Scripts/Other-Scripts/PromptDetect", env="mtsd-base", port=7860, port_arg="--port", url="http://localhost:{port}", browser=True, heavy=True, confirm=True, notes="May use substantial GPU memory; LocateAnything also starts its mtsd-la worker."),
    T(id="promptdetect-batch-ui", title="PromptDetect — batch evaluation UI", category="Model and Inference UIs", description="Open a local Gradio app for scoring prompts against MDWD or MTSD ground truth. Dry-run is available in the app.", script="Scripts/Other-Scripts/PromptDetect/batch_evaluation/gradio_batch_eval.py", cwd="Scripts/Other-Scripts/PromptDetect/batch_evaluation", env="mtsd-base", port=7860, url="http://localhost:{port}", browser=True, heavy=True, confirm=True, notes="Can run real inference when requested; Cosmos 32B stays gated in the UI."),
    T(id="attr-ui", title="MTSD attributes — checkpoint comparison", category="Model and Inference UIs", description="Open a local Gradio app to compare existing attribute-classifier checkpoints on an image.", script="Scripts/MTSD-Scripts/AttributeClassification/inference/gradio_compare.py", cwd="Scripts/MTSD-Scripts/AttributeClassification", env="mtsd-attrcls", port=7860, url="http://localhost:{port}", browser=True, heavy=True, confirm=True, notes="This UI has no port flag; it normally uses 7860 and needs available model checkpoints."),
    T(id="mtsd-scan", title="MTSD QA — audit annotations", category="Annotation and QA Tools", description="Read the Final-QA JSONs and report duplicate boxes, missing attributes and broken references. No files are changed.", script="Scripts/MTSD-Scripts/MTSD-AnnotationQA/scan_annotations.py", cwd=".", env="mtsd-base", args=("--dry-run",), read_only=True),
    T(id="mtsd-review", title="MTSD QA — review audit findings", category="Annotation and QA Tools", description="Open a local review UI for the latest audit. It saves review decisions only; it never edits annotation JSON.", script="Scripts/MTSD-Scripts/MTSD-AnnotationQA/review_app.py", cwd="Scripts/MTSD-Scripts/MTSD-AnnotationQA", env="mtsd-base", port=7861, port_arg="--port", url="http://localhost:{port}", browser=True, confirm=True, read_only=False, notes="Writes reviewed_decisions.json in the audit folder; annotations remain unchanged."),
    T(id="mtsd-apply-dry", title="MTSD QA — preview reviewed fixes", category="Annotation and QA Tools", description="Show exactly what the latest review decisions would change. This is a dry run; annotations stay untouched.", script="Scripts/MTSD-Scripts/MTSD-AnnotationQA/apply_fixes.py", cwd="Scripts/MTSD-Scripts/MTSD-AnnotationQA", env="mtsd-base", args=("--decisions", "latest", "--dry-run"), read_only=True),
    T(id="mtsd-apply", title="MTSD QA — apply reviewed fixes", category="Annotation and QA Tools", description="Apply the latest approved review decisions to Final-QA annotations, with backups and two confirmation prompts.", script="Scripts/MTSD-Scripts/MTSD-AnnotationQA/apply_fixes.py", cwd="Scripts/MTSD-Scripts/MTSD-AnnotationQA", env="mtsd-base", args=("--decisions", "latest", "--apply"), confirm=True, read_only=False, notes="This modifies annotation JSON only after the launcher and the tool both confirm; --yes is never used."),
    T(id="labelstudio", title="Label Studio — show workflow", category="Annotation and QA Tools", description="Show the documented start, import and reload steps without starting Label Studio or touching its database.", instruction="Documents/LabelStudioWorkflow.md", notes="The existing start script can stop another process on port 8080 and writes local Label Studio state."),
    T(id="mdwd-eda", title="MDWD — sample EDA", category="Dataset Analysis and EDA", description="Run a small 50-image MDWD scan and write a separate analysis output. Charts are skipped for a quick, safe pass.", script="Scripts/MDWD-Scripts/MDWD-Analysis/run_eda.py", cwd=".", env="MDWD", args=("--max-images", "50", "--skip-charts",), confirm=True, notes="This writes EDA output only; it does not alter the dataset or experiment results."),
    T(id="mtsd-stats", title="MTSD — image and annotation totals", category="Dataset Analysis and EDA", description="Print per-group image counts, annotation counts and review workload estimates. No CSV is written.", script="Scripts/MTSD-Scripts/MTSD-Analysis/count_image_annotation_stats.py", cwd=".", env="mtsd-base", args=("--no-csv",)),
    T(id="mtsd-gps", title="MTSD — GPS metadata audit", category="Dataset Analysis and EDA", description="Scan MTSD images and print whether EXIF/GPS metadata is present. No report file is requested.", script="Scripts/MTSD-Scripts/MTSD-Analysis/check_gps_tags.py", cwd=".", env="mtsd-base"),
    T(id="mdwd-samples", title="MDWD — render sample annotations", category="Dataset Analysis and EDA", description="Create two annotated sample images per dataset split for visual inspection.", script="Scripts/MDWD-Scripts/MDWD-Analysis/visualise_samples.py", cwd=".", env="MDWD", args=("--per-split", "2"), confirm=True, notes="Writes sample images under the analysis output; source data is untouched."),
    T(id="mtsd-map", title="MTSD — rebuild capture atlas", category="Dataset Analysis and EDA", description="Create the interactive HTML map from the cached MTSD image inventory.", script="Scripts/MTSD-Scripts/MTSD-Analysis/mtsd_mapper.py", cwd=".", env="mtsd-base", args=(), confirm=True, notes="Updates Documents/MTSD-EDA/MTSD_mapped.html only; use Open MTSD map to view it."),
    T(id="batch-cli", title="PromptDetect — batch plan dry-run", category="Evaluation and Benchmarks", description="Validate a small MTSD/SAM3 evaluation plan: 25 images, no prompts, no model load and no output files.", script="Scripts/Other-Scripts/PromptDetect/batch_evaluation/run_batch_eval.py", cwd="Scripts/Other-Scripts/PromptDetect/batch_evaluation", env="mtsd-base", args=("--dataset", "MTSD", "--split", "test", "--prompts", "--models", "sam3", "--max-images", "25", "--dry-run"), heavy=True, confirm=True),
    T(id="inference-benchmark", title="Inference benchmark — list models", category="Evaluation and Benchmarks", description="List the supported benchmark models and their availability. It does not load a model or run inference.", script="Scripts/Other-Scripts/Inference-Benchmark/inference_speed_benchmark.py", cwd=".", env="MDWD", args=("--list-models",)),
    T(id="integrity", title="Final dataset integrity — dry-run", category="Evaluation and Benchmarks", description="Check MDWD and MTSD dataset consistency and print the result. No integrity report is written in dry-run mode.", script="Scripts/FinalEvaluation/final_dataset_integrity_check.py", cwd=".", env="MDWD", args=("--dataset", "all", "--dry-run")),
    T(id="failure-sampler-help", title="Failure-case sampler — usage help", category="Evaluation and Benchmarks", description="Show how to sample existing prediction failures without launching a sampling job.", script="Scripts/FinalEvaluation/failure_case_sampler.py", cwd="Scripts/FinalEvaluation", env="MDWD", args=("--help",)),
    T(id="health", title="Repository health — dry-run", category="Repository Maintenance", description="Check paths, required folders, notebooks and repository conventions without writing health reports.", script="Scripts/Automation/verify_repository_health.py", cwd=".", env="mtsd-base", args=("--dry-run",)),
    T(id="gdpr-help", title="GDPR redaction — safe usage help", category="Repository Maintenance", description="Show preview/apply instructions without loading detectors, generating previews or changing images.", script="Scripts/Other-Scripts/GDPR-Compliance/redact.py", cwd="Scripts/Other-Scripts/GDPR-Compliance", env="mtsd-base", args=("--help",), confirm=True, notes="Preview creates a separate output tree. Applying redactions is deliberately not exposed here."),
    T(id="open-final-reports", title="Open final reports", category="Documentation and Reports", description="Open Documents/Final-Reports in File Explorer.", instruction="Documents/Final-Reports"),
    T(id="open-eda", title="Open EDA outputs", category="Documentation and Reports", description="Open the MDWD and MTSD EDA output folders in File Explorer.", instruction="Documents/MDWD-EDA\nDocuments/MTSD-EDA"),
    T(id="open-map", title="Open MTSD map", category="Documentation and Reports", description="Open the generated MTSD capture atlas HTML in your default browser.", instruction="Documents/MTSD-EDA/MTSD_mapped.html"),
]


def say(message: str = "", style: str | None = None) -> None:
    if RICH:
        CONSOLE.print(message, style=style)
    else:
        print(message)


def repo_status() -> tuple[str, str]:
    branch = commit = "unavailable"
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip() or "detached"
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return branch, commit


def header() -> None:
    branch, commit = repo_status()
    logo = """██╗   ██╗██╗   ██╗██████╗
██║   ██║██║   ██║██╔══██╗
██║   ██║██║   ██║██████╔╝
██║   ██║╚██╗ ██╔╝██╔══██╗
╚██████╔╝ ╚████╔╝ ██████╔╝
 ╚═════╝   ╚═══╝  ╚═════╝"""
    if RICH:
        CONSOLE.print(Panel.fit(
            f"[bold bright_cyan]{logo}[/]\n\n"
            "[bold white]URBAN VISION BENCHMARK[/]  [dim]// local research workspace[/]",
            border_style="bright_blue", padding=(1, 4),
        ))
        CONSOLE.print(
            f"  [dim]branch[/] [bold cyan]{branch}[/]   [dim]commit[/] [bold white]{commit}[/]   "
            f"[dim]python[/] [bold white]{Path(sys.executable).name}[/]   [dim]platform[/] [bold white]{platform.system()}[/]\n"
        )
    else:
        print(logo + "\nURBAN VISION BENCHMARK // local research workspace")
        print(f"repo={ROOT.name} branch={branch} commit={commit} python={sys.executable} os={platform.system()}")



def clear_screen() -> None:
    sys.stdout.write("\033[H\033[2J")
    sys.stdout.flush()


def read_key() -> str:
    """Read one navigation key without requiring Enter; works in Windows Terminal."""
    if os.name == "nt":
        import msvcrt

        key = msvcrt.getwch()
        if key in {"\x00", "\xe0"}:
            key = msvcrt.getwch()
            return {"H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT"}.get(key, "")
        return {"\r": "ENTER", "\x1b": "ESC", "\x08": "BACK"}.get(key, key.lower())

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
    return {"\x1b[A": "UP", "\x1b[B": "DOWN", "\r": "ENTER", "\n": "ENTER", "\x1b": "ESC", "\x7f": "BACK"}.get(key, key.lower())


def tool_badges(tool: Tool) -> str:
    badges: list[str] = []
    if tool.port:
        badges.append(f"LOCAL:{tool.port}")
    if tool.env:
        badges.append(tool.env)
    if tool.confirm:
        badges.append("CONFIRM")
    elif tool.read_only:
        badges.append("READ-ONLY")
    return "  ·  ".join(badges)


def navigation_menu(title: str, subtitle: str, options: Sequence[tuple[str, str, str]], *, allow_search: bool = False, allow_quit: bool = False) -> tuple[str, int | None]:
    """Return (action, selected index); keyboard controls are shown on screen."""
    selected = 0
    while True:
        clear_screen()
        header()
        
        controls = "↑/↓ move   Enter select   Esc back"
        if allow_search:
            controls += "   / search"
        if allow_quit:
            controls += "   q quit"

        if RICH:
            say(f" [dim]{subtitle}[/]\n")
            table = Table(show_header=False, show_edge=False, box=None, padding=(0, 1, 0, 0))
            table.add_column("Pointer", justify="right", style="bold bright_cyan", width=2)
            table.add_column("Main")
            
            for index, (label, description, badge) in enumerate(options):
                if index == selected:
                    pointer = "▶"
                    title_text = f"[bold bright_cyan]{label}[/]"
                    desc_text = f"[white]{description}[/]"
                    badge_text = f"[dim]{badge}[/]" if badge else ""
                else:
                    pointer = " "
                    title_text = f"[white]{label}[/]"
                    desc_text = f"[dim]{description}[/]"
                    badge_text = f"[dim]{badge}[/]" if badge else ""
                
                row_main = f"{title_text}  {badge_text}\n  {desc_text}" if badge_text else f"{title_text}\n  {desc_text}"
                table.add_row(pointer, row_main)
                # Add spacing row
                if index < len(options) - 1:
                    table.add_row("", "")
                    
            CONSOLE.print(Panel(
                table,
                title=f"[bold white]{title}[/]",
                title_align="left",
                subtitle=f"[dim]{controls}[/]",
                subtitle_align="left",
                border_style="bright_black",
                padding=(1, 2)
            ))
        else:
            say(f"\n{title}")
            say(subtitle)
            say("")
            for index, (label, description, badge) in enumerate(options):
                pointer = "❯" if index == selected else " "
                suffix = f"  [{badge}]" if badge else ""
                say(f"{pointer} {label}{suffix}\n    {description}")
            say(f"\n{controls}")
            
        key = read_key()
        if key in {"UP", "k"}:
            selected = (selected - 1) % len(options)
        elif key in {"DOWN", "j"}:
            selected = (selected + 1) % len(options)
        elif key == "ENTER":
            return "select", selected
        elif allow_search and key == "/":
            return "search", None
        elif key in {"ESC", "BACK", "b"}:
            return "quit" if allow_quit else "back", None
        elif allow_quit and key == "q":
            return "quit", None


def command_for(tool: Tool, port: int | None = None) -> list[str]:
    if tool.script is None:
        return []
    python_cmd = [sys.executable]
    if tool.env:
        conda = find_conda()
        if conda:
            python_cmd = [str(conda), "run", "-n", tool.env, "python"]
    args = list(tool.args)
    if tool.port_arg and port:
        args += [tool.port_arg, str(port)]
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
    for candidate in (Path.home() / "anaconda3/Scripts/conda.exe", Path.home() / "miniconda3/Scripts/conda.exe", Path(os.environ.get("PROGRAMDATA", "C:/ProgramData")) / "anaconda3/Scripts/conda.exe"):
        if candidate.exists():
            return candidate
    return None


def conda_envs() -> set[str]:
    conda = find_conda()
    if not conda:
        return set()
    try:
        output = subprocess.check_output([str(conda), "env", "list"], text=True, stderr=subprocess.STDOUT, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return set()
    return {line.split()[0] for line in output.splitlines() if line and not line.lstrip().startswith("#") and line.split()}


def port_info(port: int) -> str:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        free = sock.connect_ex(("127.0.0.1", port)) != 0
    return "free" if free else "IN USE"


def show_tool(tool: Tool, port: int | None = None) -> None:
    if tool.instruction:
        say(f"\n{tool.title}\n{tool.description}\n", "bold cyan")
        for item in tool.instruction.splitlines():
            path = ROOT / item
            if path.is_dir():
                say(f"  Open: {path}")
            elif path.is_file() and path.suffix.lower() in {".md", ".txt"}:
                say(f"  Read: {path}")
            elif path.is_file():
                webbrowser.open(path.as_uri())
            else:
                say(f"  {item}")
        if tool.id == "labelstudio":
            say("\nRead the guide before starting Label Studio. The documented launcher uses port 8080 and may stop an existing listener.", "yellow")
        return
    cmd = command_for(tool, port)
    say(f"\n{tool.title}\n{tool.description}", "bold cyan")
    say(f"Command: {display_command(cmd)}")
    say(f"Working directory: {ROOT / tool.cwd}")
    if tool.env:
        say(f"Conda environment: {tool.env}")
    if tool.port:
        say(f"URL: {tool.url.format(port=port or tool.port)}  ({port_info(port or tool.port)})")
    if tool.notes:
        say(f"Note: {tool.notes}", "yellow")


def confirm(prompt: str) -> bool:
    return input(f"{prompt} [y/N]: ").strip().lower() in {"y", "yes"}


def choose_port(tool: Tool, dry_run: bool = False) -> int | None:
    if not tool.port:
        return None
    if dry_run:
        return tool.port
    if port_info(tool.port) == "free":
        return tool.port
    say(f"Port {tool.port} is already in use.", "yellow")
    if not tool.port_arg:
        return tool.port if confirm("Continue anyway (the app may fail to bind)?") else None
    choice = input("Enter an alternate port, or press Enter to cancel: ").strip()
    if not choice:
        return None
    try:
        selected = int(choice)
        if not 1 <= selected <= 65535:
            raise ValueError
        return selected
    except ValueError:
        say("Invalid port; cancelled.", "red")
        return None


def run_tool(tool: Tool, dry_run: bool = False) -> None:
    if tool.path and not tool.path.exists():
        say(f"Missing script: {tool.path}", "red")
        return
    port = choose_port(tool, dry_run)
    if tool.port and port is None:
        return
    if tool.instruction:
        show_tool(tool, port)
        if not dry_run and tool.id.startswith("open-"):
            for item in tool.instruction.splitlines():
                target = ROOT / item
                if target.exists():
                    if target.is_file() and target.suffix.lower() in {".html", ".htm"}:
                        webbrowser.open(target.as_uri())
                    elif os.name == "nt":
                        os.startfile(str(target))  # type: ignore[attr-defined]
                    else:
                        subprocess.run(["xdg-open", str(target)], check=False)
        return
    cmd = command_for(tool, port)
    show_tool(tool, port)
    if dry_run:
        return
    if tool.confirm and not confirm("Proceed with this launch?"):
        say("Cancelled.", "yellow")
        return
    if tool.env and not find_conda():
        if tool.heavy or tool.port:
            say("Conda was not found; this launch needs its documented environment and was not started.", "red")
            return
        say("Conda was not found; using the current Python interpreter for this safe command.", "yellow")
    try:
        if tool.browser and tool.url and port:
            threading.Timer(2.0, lambda: webbrowser.open(tool.url.format(port=port))).start()
            say("Browser will open shortly. Keep this terminal open; stop the app with Ctrl+C.", "green")
        elif tool.port:
            say("Keep this terminal open; stop the app with Ctrl+C.", "green")
        result = subprocess.run(cmd, cwd=ROOT / tool.cwd)
        say(f"\nFinished with exit code {result.returncode}.", "green" if result.returncode == 0 else "red")
    except KeyboardInterrupt:
        say("\nStopped by user.", "yellow")
    except OSError as exc:
        say(f"Could not launch: {exc}", "red")


def list_tools() -> None:
    for category in dict.fromkeys(t.category for t in TOOLS):
        say(f"\n{category}", "bold cyan")
        for tool in [x for x in TOOLS if x.category == category]:
            suffix = f"  [{tool.env}]" if tool.env else ""
            if tool.port:
                suffix += f"  :{tool.port}"
            say(f"  {tool.id:<24} {tool.title}{suffix}")
            say(f"  {'':<24} {tool.description}", "dim")


def doctor() -> None:
    say("UVB launcher doctor", "bold cyan")
    conda = find_conda()
    say(f"  conda: {'OK - ' + str(conda) if conda else 'NOT FOUND'}", "green" if conda else "yellow")
    envs = conda_envs()
    for env in ("MDWD", "mtsd-attrcls", "mtsd-base", "mtsd-la"):
        say(f"  env {env}: {'OK' if env in envs else 'not detected'}", "green" if env in envs else "yellow")
    for port in sorted({t.port for t in TOOLS if t.port}):
        say(f"  port {port}: {port_info(port)}")
    important = [t for t in TOOLS if t.script]
    missing = [t.script for t in important if t.path and not t.path.exists()]
    say(f"  configured scripts: {len(important) - len(missing)}/{len(important)} present", "green" if not missing else "red")
    if missing:
        for item in missing:
            say(f"    missing: {item}", "red")
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
    say(f"  sqlite guard: label_studio.sqlite3 is never opened by this launcher")


def menu() -> None:
    category_descriptions = {
        "Model and Inference UIs": "Open local browser tools for interactive model inference and checkpoint comparison.",
        "Annotation and QA Tools": "Audit, review and carefully apply MTSD annotation fixes; Label Studio stays instructions-only.",
        "Dataset Analysis and EDA": "Inspect MDWD and MTSD structure, metadata, samples and map outputs.",
        "Evaluation and Benchmarks": "Validate benchmark plans, integrity checks and qualitative-analysis tooling.",
        "Repository Maintenance": "Run read-only repository checks and view safe GDPR workflow guidance.",
        "Documentation and Reports": "Open the folders and HTML outputs produced by the repository tools.",
    }
    while True:
        categories = list(dict.fromkeys(t.category for t in TOOLS))
        category_options = [(category, category_descriptions[category], f"{sum(t.category == category for t in TOOLS)} tools") for category in categories]
        action, index = navigation_menu(
            "Choose a workspace",
            "Everything below is repository-local. Items marked CONFIRM need an explicit go-ahead.",
            category_options,
            allow_search=True,
            allow_quit=True,
        )
        if action == "quit":
            return
        if action == "search":
            clear_screen()
            header()
            query = input("\nSearch tools (name or purpose; blank to cancel)> ").strip().lower()
            matches = [t for t in TOOLS if query in (t.id + " " + t.title + " " + t.description).lower()]
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
                run_tool(matches[match_index])
                input("\nPress Enter to return to the launcher...")
            continue
        if index is None:
            continue
        selected = [t for t in TOOLS if t.category == categories[index]]
        action, tool_index = navigation_menu(
            categories[index],
            "Choose an action. The label tells you whether it is read-only, local, or requires confirmation.",
            [(t.title, t.description, tool_badges(t)) for t in selected],
        )
        if action == "select" and tool_index is not None:
            run_tool(selected[tool_index])
            input("\nPress Enter to return to the launcher...")


def main() -> int:
    parser = argparse.ArgumentParser(description="Safe Urban Vision Benchmark launcher")
    parser.add_argument("tool", nargs="?", help="direct tool id, e.g. promptdetect, attr-ui, health")
    parser.add_argument("--list", action="store_true", help="list configured tools")
    parser.add_argument("--doctor", action="store_true", help="check environments, ports and paths")
    parser.add_argument("--dry-run", action="store_true", help="print configured commands without running them")
    args = parser.parse_args()
    if args.list:
        list_tools(); return 0
    if args.doctor:
        doctor(); return 0
    if args.dry_run:
        if args.tool:
            tool = next((t for t in TOOLS if t.id == args.tool), None)
            if not tool: say(f"Unknown tool id: {args.tool}", "red"); return 2
            run_tool(tool, dry_run=True)
        else:
            for tool in TOOLS:
                if tool.script:
                    show_tool(tool, tool.port)
        return 0
    if args.tool:
        tool = next((t for t in TOOLS if t.id == args.tool), None)
        if not tool: say(f"Unknown tool id: {args.tool}. Use --list.", "red"); return 2
        run_tool(tool); return 0
    menu(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
