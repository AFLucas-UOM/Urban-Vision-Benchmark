#!/usr/bin/env python3
"""Repository health / path verification for Urban-Vision-Benchmark.

Read-only checks over structure, code and documentation:

  * expected top-level, dataset, scripts, results, models and documents folders;
  * key README files present;
  * every notebook is valid JSON; every Python file compiles;
  * suspicious path patterns (old pre-reorganisation layouts, bare GRP paths,
    absolute C:/Users paths, legacy repo names);
  * cross-dataset contamination (MDWD paths inside MTSD supervised notebook
    code cells and vice versa);
  * hardcoded W&B keys outside .env; .env is git-ignored;
  * Markdown relative links that no longer resolve;
  * every target in Scripts/Automation/workflow_targets.json exists on disk.

Outputs (unless --dry-run):
    Documents/Final-Reports/repository_health_check.md
    Documents/Final-Reports/repository_health_check.json
(fixed names by design - this is a living "current health" report; each run
embeds its timestamp).

Usage:
    python Scripts/Automation/verify_repository_health.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import py_compile
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path = SCRIPT_DIR) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "Datasets").exists() and (parent / "Scripts").exists():
            return parent
    raise RuntimeError("Could not locate project root.")


ROOT = find_project_root()
REPORT_DIR = ROOT / "Documents" / "Final-Reports"

EXPECTED_DIRS = [
    "Datasets/MDWD/MDWD-YOLO26", "Datasets/MDWD/MDWD-RFDETR",
    "Datasets/MTSD/Annotations", "Datasets/MTSD/GRP-1",
    "Documents/MDWD-EDA", "Documents/MTSD-EDA",
    "Models", "Requirements",
    "Results/MDWD-Results", "Results/MDWD-Runs", "Results/MTSD-Results", "Results/MTSD-Runs",
    "Scripts/MDWD-Scripts/MDWD-Analysis", "Scripts/MDWD-Scripts/MDWD-SupervisedNotebooks",
    "Scripts/MTSD-Scripts/AttributeClassification", "Scripts/MTSD-Scripts/MTSD-Analysis",
    "Scripts/MTSD-Scripts/MTSD-AnnotationQA", "Scripts/MTSD-Scripts/MTSD-SupervisedNotebooks",
    "Scripts/Other-Scripts/PromptDetect/batch_evaluation",
    "Scripts/Other-Scripts/GDPR-Compliance", "Requirements/CondaEnvironments",
    "Scripts/Other-Scripts/Inference-Benchmark", "Scripts/Automation",
]
EXPECTED_READMES = [
    "README.md",
    "Scripts/MTSD-Scripts/README.md",
    "Scripts/MTSD-Scripts/AttributeClassification/README.md",
    "Scripts/MTSD-Scripts/MTSD-AnnotationQA/README.md",
    "Scripts/MDWD-Scripts/MDWD-Analysis/README.md",
    "Scripts/Other-Scripts/PromptDetect/batch_evaluation/README.md",
    "Requirements/CondaEnvironments/README.md",
    "Scripts/Other-Scripts/Inference-Benchmark/README.md",
    "Scripts/Automation/README.md",
    "Scripts/FinalEvaluation/README.md",
]
# Result folders that must keep existing (preservation canaries).
EXPECTED_RESULT_DIRS = [
    "Results/MDWD-Runs/YOLO26-EUVIP", "Results/MDWD-Runs/YOLO12-EUVIP",
    "Results/MDWD-Runs/YOLO11-EUVIP", "Results/MDWD-Runs/RF-DETR-EUVIP",
    "Results/MDWD-Results/YOLO26/Model-Size-Comparison",
    "Scripts/MTSD-Scripts/AttributeClassification/outputs/checkpoints",
    "Scripts/MTSD-Scripts/AttributeClassification/outputs/reports",
]

# Directories never scanned for content problems (historical records / bulk).
EXCLUDED_DIR_TOKENS = {
    ".git", "__pycache__", "wandb", "runs", "logs", "executed", "crops",
    "checkpoints", "GeneratedCSVs", "Figures", "SampleAnnotationImages",
    "Final-Reports", "_archive", "node_modules", ".ipynb_checkpoints",
}
LEGACY_FILES = {  # documented legacy artefacts - old paths inside are expected
    "Scripts/MDWD-Scripts/MDWD-Analysis/DatasetVisualisation.ipynb",
    "Scripts/MDWD-Scripts/MDWD-Analysis/DatasetVisualisation-OG.ipynb",
}

SUSPICIOUS_PATTERNS = [
    # (name, severity, compiled regex, description)
    ("old_grp_path", "FAIL", re.compile(r"Datasets[\\/]{1,2}GRP-\d"),
     "pre-migration dataset path (Datasets/GRP-n instead of Datasets/MTSD/GRP-n)"),
    ("bare_grp_images", "WARN", re.compile(r"(?<![\w/\\])GRP-\d+[\\/]{1,2}Images"),
     "GRP-n/Images path without the Datasets/MTSD prefix on the same line"),
    ("old_repo_name", "WARN", re.compile(r"MTSDataset|Dataset-Versions|AICOM-YOLO|Jupyter Notebooks[\\/]"),
     "reference to the pre-reorganisation repository layout"),
    ("abs_users_path", "WARN", re.compile(r"[A-Za-z]:[\\/]{1,2}Users[\\/]"),
     "absolute Windows user path (fine for env docs, wrong in portable code)"),
    ("wandb_key", "FAIL", re.compile(r"wandb_v1_[A-Za-z0-9_\-]{8,}|WANDB_API_KEY\s*=\s*[\"'][A-Za-z0-9_\-]{12,}"),
     "hardcoded W&B API key outside .env"),
]
CROSS_CHECKS = [
    ("Scripts/MTSD-Scripts/MTSD-SupervisedNotebooks", re.compile(r"Datasets[\\/]{1,2}MDWD|MDWD-Runs|MDWD-Results"),
     "MDWD path inside MTSD supervised notebook code"),
    ("Scripts/MDWD-Scripts/MDWD-SupervisedNotebooks", re.compile(r"Datasets[\\/]{1,2}MTSD|MTSD-Runs|MTSD-Results"),
     "MTSD path inside MDWD supervised notebook code"),
]


def rel_posix(path: Path) -> str:
    """Repository-relative POSIX path for matching allowlisted records."""
    return str(path.relative_to(ROOT)).replace("\\", "/")


def is_allowed_pattern_hit(name: str, path: Path, line: str) -> bool:
    """Known provenance/history/migration references that stay auditable.

    These are intentionally not rewritten: they document where data or run
    records came from, or show the legacy path shape handled by migration
    tooling, rather than paths used by current executable workflows.
    """
    rel = rel_posix(path)
    if name == "old_grp_path":
        return rel in {
            "Scripts/MTSD-Scripts/README.md",
            "Scripts/MTSD-Scripts/update_annotation_paths.ps1",
        } and '"source_image"' in line
    if name == "old_repo_name":
        if rel == "Scripts/Automation/verify_repository_health.py":
            return True
        if rel.startswith("Datasets/MTSD/Annotations/") and "/Final-QA/QA-GRP" in rel \
                and '"source_file"' in line:
            return True
        if rel == "Scripts/MTSD-Scripts/AttributeClassification/outputs/experiment_log.json":
            return True
        if rel == "Scripts/MTSD-Scripts/update_annotation_paths.ps1" \
                and "MTSDataset repository root" in line:
            return True
        if rel == "Scripts/MDWD-Scripts/MDWD-Analysis/MDWD-EDA-OG.ipynb":
            return True
        if rel == "Scripts/MDWD-Scripts/MDWD-Analysis/README.md" and "Legacy" in line:
            return True
    return False


def is_excluded(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if rel_posix(path) in LEGACY_FILES:
        return True
    return any(part in EXCLUDED_DIR_TOKENS or part.startswith("[OLD]") for part in rel.parts) \
        or ".bak" in path.name or ".pre-migration" in path.name or ".pre-qa-fix" in path.name


def scan_files() -> list[Path]:
    """Source/docs/config files worth scanning for path problems."""
    files: list[Path] = []
    for base in ("Scripts", "Documents", "Requirements"):
        for ext in ("*.py", "*.md", "*.ps1", "*.yaml", "*.yml", "*.json", "*.ipynb",
                    "*.sh", "*.cmd", "*.txt"):
            files.extend(p for p in (ROOT / base).rglob(ext) if not is_excluded(p))
    files.extend(p for p in ROOT.glob("*.md"))
    files.extend((ROOT / "Datasets" / "MTSD" / "Annotations").rglob("QA-*.json"))
    return sorted({p for p in files if p.is_file() and not is_excluded(p)})


def file_text(path: Path) -> str:
    """Text content; for notebooks, cell sources only (not outputs)."""
    if path.suffix == ".ipynb":
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
            return "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))
        except Exception:
            return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def notebook_code_text(path: Path) -> str:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
        return "\n".join("".join(c.get("source", []))
                         for c in nb.get("cells", []) if c.get("cell_type") == "code")
    except Exception:
        return ""


def run_checks() -> list[dict]:
    checks: list[dict] = []

    def add(name: str, status: str, details: list[str], description: str = "") -> None:
        checks.append({"check": name, "status": status,
                       "description": description, "details": details})

    # 1. structure ------------------------------------------------------------
    missing = [d for d in EXPECTED_DIRS if not (ROOT / d).is_dir()]
    add("expected_folders", "FAIL" if missing else "PASS", missing,
        "All expected dataset/scripts/results/documents folders exist")
    missing = [f for f in EXPECTED_READMES if not (ROOT / f).is_file()]
    add("key_readmes", "WARN" if missing else "PASS", missing, "Key README files present")
    missing = [d for d in EXPECTED_RESULT_DIRS if not (ROOT / d).is_dir()]
    add("result_folders_preserved", "FAIL" if missing else "PASS", missing,
        "Existing experimental result folders still present")

    # 2. notebooks valid JSON ---------------------------------------------------
    bad = []
    notebooks = [p for p in (ROOT / "Scripts").rglob("*.ipynb") if not is_excluded(p)]
    for nb in notebooks:
        try:
            json.loads(nb.read_text(encoding="utf-8"))
        except Exception as exc:
            bad.append(f"{nb.relative_to(ROOT)}: {exc}")
    add("notebooks_valid_json", "FAIL" if bad else "PASS", bad,
        f"{len(notebooks)} notebooks parse as JSON")

    # 3. python compiles ---------------------------------------------------------
    bad = []
    py_files = [p for p in (ROOT / "Scripts").rglob("*.py") if not is_excluded(p)]
    for py in py_files:
        try:
            py_compile.compile(str(py), doraise=True)
        except Exception as exc:
            bad.append(f"{py.relative_to(ROOT)}: {exc}")
    add("python_compiles", "FAIL" if bad else "PASS", bad,
        f"{len(py_files)} Python files compile")

    # 4. suspicious patterns -----------------------------------------------------
    files = scan_files()
    for name, severity, pattern, description in SUSPICIOUS_PATTERNS:
        hits = []
        for path in files:
            if name == "wandb_key" and path.name == ".env":
                continue
            for line_no, line in enumerate(file_text(path).splitlines(), 1):
                if pattern.search(line) and not is_allowed_pattern_hit(name, path, line):
                    hits.append(f"{path.relative_to(ROOT)}:{line_no}: {line.strip()[:110]}")
        add(f"pattern_{name}", severity if hits else "PASS", hits[:60], description)

    # 5. cross-dataset contamination (code cells only) ------------------------------
    for folder, pattern, description in CROSS_CHECKS:
        hits = []
        for nb in sorted((ROOT / folder).glob("*.ipynb")):
            for line_no, line in enumerate(notebook_code_text(nb).splitlines(), 1):
                if pattern.search(line):
                    hits.append(f"{nb.relative_to(ROOT)}:code:{line_no}: {line.strip()[:110]}")
        add(f"cross_{Path(folder).name}", "FAIL" if hits else "PASS", hits, description)

    # 6. markdown links resolve -----------------------------------------------------
    link_pattern = re.compile(r"\]\(([^)#\s]+?)\)")
    broken = []
    md_files = [ROOT / f for f in EXPECTED_READMES if (ROOT / f).is_file()]
    md_files += [p for p in (ROOT / "Documents").glob("*.md")]
    for md in md_files:
        for target in link_pattern.findall(md.read_text(encoding="utf-8", errors="replace")):
            if target.startswith(("http", "mailto:", "<")):
                continue
            resolved = (md.parent / target.replace("%20", " ")).resolve()
            if not resolved.exists():
                broken.append(f"{md.relative_to(ROOT)}: ({target})")
    add("markdown_links", "WARN" if broken else "PASS", broken,
        "Relative links in key Markdown files resolve")

    # 7. .env safety --------------------------------------------------------------
    details = []
    status = "PASS"
    if (ROOT / ".env").exists():
        result = subprocess.run(["git", "check-ignore", ".env"], cwd=ROOT,
                                capture_output=True, text=True)
        if result.returncode != 0:
            status = "FAIL"
            details.append(".env exists but is NOT git-ignored!")
    add("env_gitignored", status, details, ".env is git-ignored")

    # 8. automation targets exist ------------------------------------------------
    registry_path = SCRIPT_DIR / "workflow_targets.json"
    missing = []
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))["targets"]
        missing = [f"{name}: {spec['path']}" for name, spec in registry.items()
                   if not (ROOT / spec["path"]).exists()]
        add("automation_targets", "FAIL" if missing else "PASS", missing,
            f"All {len(registry)} workflow-runner targets exist on disk")
    else:
        add("automation_targets", "WARN", ["workflow_targets.json not found"], "")

    return checks


def write_reports(checks: list[dict], generated_at: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    overall = ("FAIL" if any(c["status"] == "FAIL" for c in checks)
               else "WARN" if any(c["status"] == "WARN" for c in checks) else "PASS")
    payload = {"generated_at": generated_at, "overall": overall, "checks": checks}
    (REPORT_DIR / "repository_health_check.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Repository health check",
        f"\nGenerated: {generated_at}  |  Overall: **{overall}**",
        "\n| Check | Status | Findings |", "| --- | --- | --- |",
    ]
    for check in checks:
        lines.append(f"| {check['check']} | {check['status']} | {len(check['details'])} |")
    for check in checks:
        if check["details"]:
            lines.append(f"\n## {check['check']} ({check['status']})\n")
            if check["description"]:
                lines.append(f"*{check['description']}*\n")
            lines.extend(f"- `{d}`" for d in check["details"])
    lines.append("\n*Read-only report; nothing was modified. Re-run via "
                 "`python Scripts/Automation/verify_repository_health.py`.*\n")
    (REPORT_DIR / "repository_health_check.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify repository structure/path health (read-only).")
    parser.add_argument("--dry-run", action="store_true", help="Print results; write no report files.")
    args = parser.parse_args()

    generated_at = datetime.now().isoformat(timespec="seconds")
    checks = run_checks()
    overall = ("FAIL" if any(c["status"] == "FAIL" for c in checks)
               else "WARN" if any(c["status"] == "WARN" for c in checks) else "PASS")
    print(f"Repository health: {overall}\n")
    for check in checks:
        marker = {"PASS": "+", "WARN": "!", "FAIL": "X"}[check["status"]]
        print(f" [{marker}] {check['check']:<34} {check['status']:<5} ({len(check['details'])} findings)")
        for detail in check["details"][:5]:
            print(f"      - {detail}")
        if len(check["details"]) > 5:
            print(f"      ... {len(check['details']) - 5} more")
    if not args.dry_run:
        write_reports(checks, generated_at)
        print(f"\nReports written to {REPORT_DIR / 'repository_health_check.md'} (+ .json)")
    else:
        print("\nDry run: no report files written.")
    return 0 if overall != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
