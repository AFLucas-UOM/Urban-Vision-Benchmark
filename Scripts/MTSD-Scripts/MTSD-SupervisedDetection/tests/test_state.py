import json

import pytest

from mtsd_detection.state import atomic_write, compatibility_diff, load_compatible


def test_atomic_state_roundtrip_and_resume_fingerprint_checks(tmp_path):
    expected = {"requested_matrix": ["yolo12n"], "dataset_version": "v",
                "prep_manifest_sha256": "p", "split_manifest_sha256": "s",
                "annotation_source_mode": "qa_only", "dataset_variant": "augmented",
                "config_fingerprint": "c"}
    path = tmp_path / "state.json"; atomic_write(path, {**expected, "models": {}})
    assert load_compatible(path, expected)["dataset_version"] == "v"
    changed = {**expected, "split_manifest_sha256": "changed"}
    assert "split_manifest_sha256" in compatibility_diff(json.loads(path.read_text()), changed)
    with pytest.raises(ValueError, match="split_manifest_sha256"):
        load_compatible(path, changed)
