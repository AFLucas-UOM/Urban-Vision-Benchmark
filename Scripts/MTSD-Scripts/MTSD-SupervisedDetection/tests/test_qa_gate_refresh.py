import json
from pathlib import Path

import pytest
import yaml

import update_qa_gate as qa_refresh
from mtsd_detection.annotation_sources import CLASS_NAMES


def _qa_payload(categories=None):
    return {
        "images": [{"id": 1, "file_name": "a.jpg", "source_image": "a.jpg", "width": 10, "height": 10}],
        "annotations": [],
        "categories": categories or [{"id": i + 1, "name": name} for i, name in enumerate(CLASS_NAMES)],
    }


def _write_qa(root: Path, group: str, categories=None, name: str | None = None) -> Path:
    path = root / group / "Final-QA" / (name or f"QA-{group.replace('-', '')}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_qa_payload(categories)), encoding="utf-8")
    return path


def _write_default_config(path: Path, approved_groups: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "dataset:\n  annotations_root: Datasets/MTSD/Annotations\n  prepared_root: Datasets/MTSD/Prepared\n"
        "  version_base: mtsd-qa-v1\n"
        "annotations:\n  source_mode: qa_only\n  group_scope: explicit\n"
        f"  approved_groups: {qa_refresh._flow_list(approved_groups)}\n"
        "  unexpected_group_policy: fail\n  allow_raw_xml_fallback: false\n"
        "  require_non_qa_confirmation: true\n  qa_gate_file: qa_gate.yaml\n"
        "split:\n  algorithm_version: v1\n  ratios: {train: 0.8, valid: 0.1, test: 0.1}\n  seed: 42\n"
        "augmentation:\n  recipe_version: photometric-v1\n  copies_per_image: 2\n  seed: 42\n  ops: {}\n"
        "training:\n  epochs: 1\n  image_size: 640\n  effective_batch: 32\n  optimizer: AdamW\n"
        "  lr0: 0.001\n  lrf: 0.01\n  weight_decay: 0.0005\n  warmup_epochs: 3.0\n  patience: 10\n"
        "  seed: 42\n  deterministic: true\n  device: auto\n  per_model_overrides: {}\n"
        "matrix:\n  dissertation: []\n  aug_ablation: []\n"
        "validation:\n  require_all_classes_in_test: false\n"
        "  minimum_boxes_per_class: {train: 0, valid: 0, test: 0}\n  low_support_warning_threshold: 0\n"
        "wandb:\n  project: MSc-MTSD-SupervisedDetection\n  entity_env: WANDB_ENTITY\n  mode: disabled\n"
        "  group: test\n"
        "outputs:\n  runs_root: Results/Runs\n  results_root: Results/Results\n  state_root: Results/State\n",
        encoding="utf-8",
    )


def _write_qa_gate(path: Path, approved_scope: list[str], audited_groups: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gate = {
        "gate_version": "qa-gate-v1", "audit_path": "x", "audit_timestamp": "2020-01-01T00:00:00",
        "audited_groups": audited_groups or [], "approved_scope": approved_scope,
        "finding_counts": {"invalid_attribute_values": 0, "duplicate_candidates": 0},
        "resolution_status": "unresolved", "approved_by": None, "acknowledgement": "placeholder",
    }
    path.write_text(qa_refresh.render_qa_gate(gate), encoding="utf-8")


def _write_audit(audit_root: Path, name: str, groups: list[str], generated_at: str,
                 invalid_attr: int = 0, dup: int = 0, missing: int = 0, ref: int = 0) -> Path:
    audit_dir = audit_root / name
    audit_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "qa_files": [f"Datasets/MTSD/Annotations/{g}/Final-QA/QA-{g.replace('-', '')}.json" for g in groups],
        "generated_at": generated_at,
        "totals": {"missing_attribute_findings": missing, "invalid_attribute_findings": invalid_attr,
                   "duplicate_pair_findings": dup, "reference_problem_findings": ref},
    }
    (audit_dir / "audit_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return audit_dir


class _Layout:
    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.annotations_root = tmp_path / "Annotations"
        self.audit_root = tmp_path / "AuditOutputs"
        self.default_config = tmp_path / "default.yaml"
        self.qa_gate = tmp_path / "qa_gate.yaml"

    def run(self, *extra_args, apply: bool = False, as_json: bool = False) -> int:
        args = ["--apply" if apply else "--dry-run",
                "--annotations-root", str(self.annotations_root),
                "--audit-root", str(self.audit_root),
                "--default-config", str(self.default_config),
                "--qa-gate", str(self.qa_gate)]
        if as_json:
            args.append("--json")
        return qa_refresh.main([*args, *extra_args])


@pytest.fixture
def layout(tmp_path):
    return _Layout(tmp_path)


# 1. Existing valid QA groups are discovered and sorted numerically.
def test_discovery_sorted_numerically(tmp_path):
    root = tmp_path / "Annotations"
    for group in ("GRP-10", "GRP-2", "GRP-1"):
        _write_qa(root, group)
    discovery = qa_refresh.discover_final_qa_groups(root)
    assert list(discovery["valid"]) == ["GRP-1", "GRP-2", "GRP-10"]


# 2. A newly added valid Final-QA group is added to approved scope.
def test_apply_adds_newly_discovered_group(layout):
    _write_qa(layout.annotations_root, "GRP-1")
    _write_qa(layout.annotations_root, "GRP-2")
    _write_default_config(layout.default_config, ["GRP-1"])
    _write_qa_gate(layout.qa_gate, ["GRP-1"], audited_groups=["GRP-1"])
    assert layout.run(apply=True) == 0
    updated = yaml.safe_load(layout.default_config.read_text(encoding="utf-8"))
    assert updated["annotations"]["approved_groups"] == ["GRP-1", "GRP-2"]
    gate = yaml.safe_load(layout.qa_gate.read_text(encoding="utf-8"))
    assert gate["approved_scope"] == ["GRP-1", "GRP-2"]


# 3. Backup JSON files are ignored.
def test_backup_files_ignored(tmp_path):
    root = tmp_path / "Annotations"
    valid = _write_qa(root, "GRP-1")
    (valid.parent / "QA-GRP1.json.bak").write_text("{}", encoding="utf-8")
    (valid.parent / "QA-GRP1.json.pre-migration-20260101.bak").write_text("{}", encoding="utf-8")
    discovery = qa_refresh.discover_final_qa_groups(root)
    assert list(discovery["valid"]) == ["GRP-1"]


# 4. Multiple active QA JSONs in one group fail.
def test_multiple_active_qa_files_raise(tmp_path):
    root = tmp_path / "Annotations"
    _write_qa(root, "GRP-1")
    _write_qa(root, "GRP-1", name="QA-GRP1-second.json")
    with pytest.raises(ValueError, match="multiple active Final-QA JSON files"):
        qa_refresh.discover_final_qa_groups(root)


def test_multiple_active_qa_files_fail_cli(layout):
    _write_qa(layout.annotations_root, "GRP-1")
    _write_qa(layout.annotations_root, "GRP-1", name="QA-GRP1-second.json")
    _write_default_config(layout.default_config, ["GRP-1"])
    _write_qa_gate(layout.qa_gate, ["GRP-1"])
    assert layout.run() == 2


# 5. Invalid JSON is rejected (soft: excluded, run still succeeds).
def test_invalid_json_is_rejected_softly(layout):
    path = layout.annotations_root / "GRP-1" / "Final-QA" / "QA-GRP1.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not valid json", encoding="utf-8")
    _write_default_config(layout.default_config, [])
    _write_qa_gate(layout.qa_gate, [])
    assert layout.run() == 0
    discovery = qa_refresh.discover_final_qa_groups(layout.annotations_root)
    assert discovery["valid"] == {}
    assert "GRP-1" in discovery["invalid"]


# 6. Class-vocabulary mismatch is rejected (hard failure).
def test_vocabulary_mismatch_raises(tmp_path):
    root = tmp_path / "Annotations"
    _write_qa(root, "GRP-1", categories=[{"id": 1, "name": "Not A Real Class"}])
    with pytest.raises(ValueError, match="Category vocabulary mismatch"):
        qa_refresh.discover_final_qa_groups(root)


def test_vocabulary_mismatch_fails_cli(layout):
    _write_qa(layout.annotations_root, "GRP-1", categories=[{"id": 1, "name": "Not A Real Class"}])
    _write_default_config(layout.default_config, [])
    _write_qa_gate(layout.qa_gate, [])
    assert layout.run() == 2


# 7. New approved group absent from the latest audit keeps the gate unresolved.
def test_new_group_absent_from_audit_keeps_unresolved(layout):
    _write_qa(layout.annotations_root, "GRP-1")
    _write_qa(layout.annotations_root, "GRP-2")
    _write_audit(layout.audit_root, "audit-1", ["GRP-1"], "2026-01-01T00:00:00")
    _write_default_config(layout.default_config, ["GRP-1"])
    _write_qa_gate(layout.qa_gate, ["GRP-1"], audited_groups=["GRP-1"])
    assert layout.run(apply=True) == 0
    gate = yaml.safe_load(layout.qa_gate.read_text(encoding="utf-8"))
    assert gate["resolution_status"] == "unresolved"
    assert "GRP-2" in gate["acknowledgement"]


# 8. Remaining invalid attributes keep the gate unresolved.
def test_remaining_invalid_attributes_keep_unresolved(layout):
    _write_qa(layout.annotations_root, "GRP-1")
    _write_audit(layout.audit_root, "audit-1", ["GRP-1"], "2026-01-01T00:00:00", invalid_attr=3)
    _write_default_config(layout.default_config, ["GRP-1"])
    _write_qa_gate(layout.qa_gate, ["GRP-1"])
    assert layout.run(apply=True) == 0
    gate = yaml.safe_load(layout.qa_gate.read_text(encoding="utf-8"))
    assert gate["resolution_status"] == "unresolved"
    assert gate["finding_counts"]["invalid_attribute_values"] == 3


# 9. Remaining duplicate candidates keep the gate unresolved.
def test_remaining_duplicates_keep_unresolved(layout):
    _write_qa(layout.annotations_root, "GRP-1")
    _write_audit(layout.audit_root, "audit-1", ["GRP-1"], "2026-01-01T00:00:00", dup=2)
    _write_default_config(layout.default_config, ["GRP-1"])
    _write_qa_gate(layout.qa_gate, ["GRP-1"])
    assert layout.run(apply=True) == 0
    gate = yaml.safe_load(layout.qa_gate.read_text(encoding="utf-8"))
    assert gate["resolution_status"] == "unresolved"
    assert gate["finding_counts"]["duplicate_candidates"] == 2


# 10. Full audit coverage with zero findings resolves the gate.
def test_full_coverage_zero_findings_resolves(layout):
    _write_qa(layout.annotations_root, "GRP-1")
    _write_qa(layout.annotations_root, "GRP-2")
    _write_audit(layout.audit_root, "audit-1", ["GRP-1", "GRP-2"], "2026-01-01T00:00:00")
    _write_default_config(layout.default_config, ["GRP-1", "GRP-2"])
    _write_qa_gate(layout.qa_gate, ["GRP-1", "GRP-2"], audited_groups=["GRP-1", "GRP-2"])
    assert layout.run(apply=True) == 0
    gate = yaml.safe_load(layout.qa_gate.read_text(encoding="utf-8"))
    assert gate["resolution_status"] == "resolved"
    assert "no unresolved critical findings" in gate["acknowledgement"]


# 11. Missing audit leaves the gate unresolved without fabricating metadata.
def test_missing_audit_no_fabrication(layout):
    _write_qa(layout.annotations_root, "GRP-1")
    _write_default_config(layout.default_config, ["GRP-1"])
    _write_qa_gate(layout.qa_gate, ["GRP-1"])
    assert layout.run(apply=True) == 0
    gate = yaml.safe_load(layout.qa_gate.read_text(encoding="utf-8"))
    assert gate["resolution_status"] == "unresolved"
    assert gate["audit_path"] is None
    assert gate["audit_timestamp"] is None
    assert gate["audited_groups"] == []
    assert gate["finding_counts"]["invalid_attribute_values"] is None
    assert gate["finding_counts"]["duplicate_candidates"] is None


# 12. Dry run writes no files.
def test_dry_run_writes_nothing(layout):
    _write_qa(layout.annotations_root, "GRP-1")
    _write_qa(layout.annotations_root, "GRP-2")
    _write_default_config(layout.default_config, ["GRP-1"])
    _write_qa_gate(layout.qa_gate, ["GRP-1"])
    before_default = layout.default_config.read_text(encoding="utf-8")
    before_gate = layout.qa_gate.read_text(encoding="utf-8")
    assert layout.run(apply=False) == 0
    assert layout.default_config.read_text(encoding="utf-8") == before_default
    assert layout.qa_gate.read_text(encoding="utf-8") == before_gate


# 13. Apply updates only the intended YAML fields.
def test_apply_touches_only_approved_groups_in_default_config(layout):
    _write_qa(layout.annotations_root, "GRP-1")
    _write_qa(layout.annotations_root, "GRP-2")
    _write_default_config(layout.default_config, ["GRP-1"])
    _write_qa_gate(layout.qa_gate, ["GRP-1"])
    before = yaml.safe_load(layout.default_config.read_text(encoding="utf-8"))
    assert layout.run(apply=True) == 0
    after = yaml.safe_load(layout.default_config.read_text(encoding="utf-8"))
    for section in ("dataset", "split", "augmentation", "training", "matrix", "validation", "wandb", "outputs"):
        assert after[section] == before[section]
    assert after["annotations"]["group_scope"] == before["annotations"]["group_scope"]
    assert after["annotations"]["unexpected_group_policy"] == before["annotations"]["unexpected_group_policy"]
    assert after["annotations"]["approved_groups"] == ["GRP-1", "GRP-2"]


# 14. default.yaml and qa_gate.yaml scopes match after apply.
def test_scopes_match_after_apply(layout):
    _write_qa(layout.annotations_root, "GRP-1")
    _write_qa(layout.annotations_root, "GRP-3")
    _write_default_config(layout.default_config, [])
    _write_qa_gate(layout.qa_gate, [])
    assert layout.run(apply=True) == 0
    default_scope = yaml.safe_load(layout.default_config.read_text(encoding="utf-8"))["annotations"]["approved_groups"]
    gate_scope = yaml.safe_load(layout.qa_gate.read_text(encoding="utf-8"))["approved_scope"]
    assert default_scope == gate_scope == ["GRP-1", "GRP-3"]


# 15. Re-running apply with no changes is idempotent.
def test_reapply_is_idempotent(layout):
    _write_qa(layout.annotations_root, "GRP-1")
    _write_default_config(layout.default_config, [])
    _write_qa_gate(layout.qa_gate, [])
    assert layout.run(apply=True) == 0
    first_default = layout.default_config.read_text(encoding="utf-8")
    first_gate = layout.qa_gate.read_text(encoding="utf-8")
    assert layout.run(apply=True) == 0
    assert layout.default_config.read_text(encoding="utf-8") == first_default
    assert layout.qa_gate.read_text(encoding="utf-8") == first_gate


# Extra: dry-run before a no-op apply reports nothing to modify.
def test_idempotent_dry_run_reports_no_files_to_modify(layout, capsys):
    _write_qa(layout.annotations_root, "GRP-1")
    _write_default_config(layout.default_config, [])
    _write_qa_gate(layout.qa_gate, [])
    assert layout.run(apply=True) == 0
    capsys.readouterr()
    assert layout.run(apply=False, as_json=True) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["files_that_would_be_modified"] == []


# Extra: --dry-run and --apply are mutually exclusive.
def test_dry_run_and_apply_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        qa_refresh.build_argparser().parse_args(["--dry-run", "--apply"])


# Extra: neither flag behaves like --dry-run (writes nothing).
def test_no_flags_behaves_like_dry_run(layout):
    _write_qa(layout.annotations_root, "GRP-1")
    _write_default_config(layout.default_config, [])
    _write_qa_gate(layout.qa_gate, [])
    before = layout.default_config.read_text(encoding="utf-8")
    args = ["--annotations-root", str(layout.annotations_root), "--audit-root", str(layout.audit_root),
            "--default-config", str(layout.default_config), "--qa-gate", str(layout.qa_gate)]
    assert qa_refresh.main(args) == 0
    assert layout.default_config.read_text(encoding="utf-8") == before


# Extra: applying with zero valid Final-QA groups refuses rather than emptying the scope.
def test_apply_refuses_empty_scope(layout):
    _write_default_config(layout.default_config, ["GRP-1"])
    _write_qa_gate(layout.qa_gate, ["GRP-1"])
    assert layout.run(apply=True) == 2
    assert yaml.safe_load(layout.default_config.read_text(encoding="utf-8"))["annotations"]["approved_groups"] == ["GRP-1"]


# Extra: no --apply required approver, approved_by stays null.
def test_approved_by_is_always_null(layout):
    _write_qa(layout.annotations_root, "GRP-1")
    _write_default_config(layout.default_config, [])
    _write_qa_gate(layout.qa_gate, [])
    assert layout.run(apply=True) == 0
    gate = yaml.safe_load(layout.qa_gate.read_text(encoding="utf-8"))
    assert gate["approved_by"] is None


# Extra: the newest audit is chosen by generated_at, not directory name/mtime order.
def test_newest_audit_chosen_by_generated_at(layout):
    _write_qa(layout.annotations_root, "GRP-1")
    _write_audit(layout.audit_root, "audit-a", ["GRP-1"], "2026-01-01T00:00:00", invalid_attr=9)
    _write_audit(layout.audit_root, "audit-b-older-name-newer-time", ["GRP-1"], "2026-06-01T00:00:00", invalid_attr=0)
    audit = qa_refresh.discover_latest_audit(layout.audit_root)
    assert audit["timestamp"] == "2026-06-01T00:00:00"
    assert audit["finding_counts"]["invalid_attribute_values"] == 0


# Extra: an audit_summary.json that exists but cannot be parsed is a structural error.
def test_unparseable_audit_summary_raises(layout):
    audit_dir = layout.audit_root / "audit-broken"
    audit_dir.mkdir(parents=True)
    (audit_dir / "audit_summary.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="unparseable"):
        qa_refresh.discover_latest_audit(layout.audit_root)
