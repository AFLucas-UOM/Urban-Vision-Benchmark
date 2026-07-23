import os

from sync_shared_yolo_images import apply_sync, find_mismatches


def test_sync_unlinks_target_before_copying_without_mutating_hardlink_source(tmp_path):
    source_variant = tmp_path / "MTSD-Augmented"
    target_variant = tmp_path / "MTSD-Unaugmented"
    canonical = tmp_path / "canonical.jpg"
    canonical.write_bytes(b"canonical")
    for split in ("train", "valid", "test"):
        source_dir = source_variant / "MTSD-YOLO" / split / "images"
        target_dir = target_variant / "MTSD-YOLO" / split / "images"
        source_dir.mkdir(parents=True)
        target_dir.mkdir(parents=True)
        (source_dir / "a.jpg").write_bytes(b"repaired")
        os.link(canonical, target_dir / "a.jpg")
    rows = find_mismatches(source_variant, target_variant)
    assert len(rows) == 3
    apply_sync(rows, target_variant)
    assert canonical.read_bytes() == b"canonical"
    assert not find_mismatches(source_variant, target_variant)
