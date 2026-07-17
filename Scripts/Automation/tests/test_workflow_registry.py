"""Registry-consistency checks for Scripts/Automation/workflow_targets.json.

Fails automatically on: invalid JSON, duplicate target names, missing
required fields, unsupported target types, paths that do not exist on disk,
type/extension mismatches, unknown conda environments, and malformed
optional fields (defaultArgs, training, noUserSite).
"""

import json
from pathlib import Path

import pytest

AUTOMATION_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = AUTOMATION_DIR.parents[1]
REGISTRY_PATH = AUTOMATION_DIR / "workflow_targets.json"

KNOWN_ENVS = {"MDWD", "mtsd-base", "mtsd-attrcls", "mtsd-la"}
KNOWN_TYPES = {"python", "notebook"}
REQUIRED_FIELDS = {"type", "path", "env", "training", "description"}
OPTIONAL_FIELDS = {"defaultArgs", "noUserSite"}


def _reject_duplicates(pairs):
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"Duplicate key in registry JSON: {key!r}")
        seen[key] = value
    return seen


@pytest.fixture(scope="module")
def registry():
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"),
                         object_pairs_hook=_reject_duplicates)
    assert "targets" in payload and payload["targets"]
    return payload["targets"]


def test_registry_parses_without_duplicate_names(registry):
    assert len(registry) >= 30


def test_every_target_has_required_fields_and_known_type(registry):
    for name, spec in registry.items():
        missing = REQUIRED_FIELDS - set(spec)
        assert not missing, f"{name}: missing fields {sorted(missing)}"
        assert spec["type"] in KNOWN_TYPES, f"{name}: unsupported type {spec['type']!r}"
        unknown = set(spec) - REQUIRED_FIELDS - OPTIONAL_FIELDS
        assert not unknown, f"{name}: unknown fields {sorted(unknown)}"


def test_every_target_path_exists_with_matching_extension(registry):
    for name, spec in registry.items():
        path = REPO_ROOT / spec["path"]
        assert path.is_file(), f"{name}: path missing on disk: {spec['path']}"
        expected_suffix = ".py" if spec["type"] == "python" else ".ipynb"
        assert path.suffix == expected_suffix, \
            f"{name}: type {spec['type']} but file is {path.suffix}"


def test_environments_flags_and_args_are_well_formed(registry):
    for name, spec in registry.items():
        assert spec["env"] in KNOWN_ENVS, f"{name}: unknown env {spec['env']!r}"
        assert isinstance(spec["training"], bool), f"{name}: training must be bool"
        assert isinstance(spec["description"], str) and spec["description"].strip(), \
            f"{name}: empty description"
        if "defaultArgs" in spec:
            assert isinstance(spec["defaultArgs"], list) and \
                all(isinstance(a, str) for a in spec["defaultArgs"]), \
                f"{name}: defaultArgs must be a list of strings"
        if "noUserSite" in spec:
            assert isinstance(spec["noUserSite"], bool), f"{name}: noUserSite must be bool"


def test_attrcls_targets_disable_user_site_packages(registry):
    for name, spec in registry.items():
        if "AttributeClassification" in spec["path"] and spec["type"] == "python":
            assert spec.get("noUserSite") is True, \
                f"{name}: mtsd-attrcls targets must set noUserSite (PYTHONNOUSERSITE=1 / python -s)"
            assert spec["env"] == "mtsd-attrcls", f"{name}: wrong env for AttributeClassification"


def test_training_targets_are_the_expected_gated_set(registry):
    gated = sorted(name for name, spec in registry.items() if spec["training"])
    for expected in ("MTSD-Supervised-Matrix", "MTSD-Aug-Ablation",
                     "AttrCls-Size-Ablation-All",
                     "PromptDetect-Dissertation-MDWD"):
        assert expected in gated, f"{expected} must remain gated behind -AllowTraining"
    for name in gated:
        # Every gated target must say so in its description (TRAIN/heavy/gated/loads).
        text = registry[name]["description"].lower()
        assert any(k in text for k in ("train", "heavy", "gated", "loads")), \
            f"{name}: gated target description must state why it is gated"


def test_principal_workflows_are_covered(registry):
    names = set(registry)
    required = {
        # MDWD
        "MDWD-EDA", "MDWD-YOLO12-Notebook", "MDWD-YOLO26-Notebook",
        "MDWD-RFDETR-Notebook", "MDWD-Leakage-Sensitivity",
        # MTSD detection + prompting
        "MTSD-AnnotationQA-Scan", "MTSD-AnnotationQA-Review-UI",
        "MTSD-AnnotationQA-Apply-Preview", "MTSD-Refresh-QA-Gate",
        "MTSD-Prepare-CLI", "MTSD-Validate-Prepared", "MTSD-Dataset-Audit",
        "MTSD-Supervised-Smoke", "MTSD-Supervised-Matrix", "MTSD-Aug-Ablation",
        "PromptDetect-App", "PromptDetect-BatchEval",
        "PromptDetect-Dissertation-Plan", "PromptDetect-Dissertation-MDWD",
        "PromptDetect-Dissertation-MTSD", "Final-Dataset-Integrity",
        # Attribute classification
        "AttrCls-List-Variants", "AttrCls-Plan",
        "AttrCls-Size-Ablation-Frozen", "AttrCls-Size-Ablation-Adapted",
        "AttrCls-Size-Ablation-All", "AttrCls-Smoke", "AttrCls-Compare-UI",
        # Maintenance
        "Repo-Health-Check",
    }
    missing = sorted(required - names)
    assert not missing, f"Registry lacks principal workflow targets: {missing}"
