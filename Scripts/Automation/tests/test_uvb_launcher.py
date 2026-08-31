"""Catalog and command-construction tests for the repository launcher."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import launch_uvb  # noqa: E402


def test_catalog_ids_paths_and_working_directories_are_valid():
    ids = [tool.id.lower() for tool in launch_uvb.TOOLS]
    assert len(ids) == len(set(ids))
    assert not launch_uvb.catalog_findings()

    for tool in launch_uvb.TOOLS:
        if tool.script:
            assert (REPO_ROOT / tool.script).is_file(), tool.id
        assert (REPO_ROOT / tool.cwd).is_dir(), tool.id


def test_complete_workflow_registry_is_exposed_once():
    registry = json.loads(launch_uvb.WORKFLOW_REGISTRY_FILE.read_text(encoding="utf-8"))["targets"]
    exposed = {tool.workflow_target for tool in launch_uvb.WORKFLOW_TOOLS}
    assert exposed == set(registry)
    assert len(launch_uvb.WORKFLOW_TOOLS) == len(registry)


def test_managed_registry_apps_redirect_to_curated_apps():
    for workflow_name, curated_id in launch_uvb.WORKFLOW_ALIASES.items():
        alias = launch_uvb.find_tool(f"workflow:{workflow_name}")
        curated = launch_uvb.find_tool(curated_id)
        assert alias is not None and curated is not None
        assert launch_uvb.effective_tool(alias) is curated
        assert curated.is_web_app


def test_notebook_command_preserves_source_and_targets_executed_copy(tmp_path):
    tool = launch_uvb.find_tool("workflow:MDWD-EDA-Notebook")
    assert tool is not None and tool.kind == "notebook"
    output = tmp_path / "executed.ipynb"
    command = launch_uvb.command_for(tool, notebook_output=output)
    assert command[-1] == "--ExecutePreprocessor.timeout=14400"
    assert "nbconvert" in command
    assert "--execute" in command
    assert str(REPO_ROOT / tool.script) in command
    assert output.name in command
    assert str(output.parent) in command


def test_attribute_commands_disable_user_site_and_current_plan_is_18_variants():
    plan = launch_uvb.find_tool("attr-size-plan")
    assert plan is not None
    command = launch_uvb.command_for(plan)
    assert "-s" in command
    assert "18-variant" in plan.description

    registry_plan = launch_uvb.find_tool("workflow:AttrCls-Plan")
    assert registry_plan is not None
    assert "18-variant" in registry_plan.description
    assert "-s" in launch_uvb.command_for(registry_plan)


def test_promptdetect_dry_run_has_a_prompt_value():
    tool = launch_uvb.find_tool("batch-cli")
    assert tool is not None
    index = tool.args.index("--prompts")
    assert tool.args[index + 1] == "traffic sign"
    assert "--dry-run" in tool.args


def test_search_matches_all_terms_case_insensitively():
    matches = launch_uvb.search_tools("prompt sensitivity")
    assert matches
    assert all("prompt" in (tool.title + tool.description).lower() for tool in matches)


def test_missing_conda_environment_is_reported_before_launch(monkeypatch):
    tool = launch_uvb.find_tool("promptdetect")
    assert tool is not None
    monkeypatch.setattr(launch_uvb, "find_conda", lambda: Path("/mock/conda"))
    monkeypatch.setattr(launch_uvb, "conda_envs", lambda: {"base", "MTSD"})

    issue = launch_uvb.environment_issue(tool)

    assert issue is not None
    assert "mtsd-base" in issue
    assert "MTSD" in issue
