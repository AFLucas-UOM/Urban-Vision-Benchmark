"""Group discovery, QA detection, crop extraction, and manifest maintenance.

A group GRP-N is QA-approved iff Datasets/MTSD/Annotations/GRP-N/Final-QA/ contains a
COCO JSON matching the configured pattern (currently QA-GRP*.json). This module
scans for all groups on every run, ingests newly approved ones, skips groups
whose QA file content hash is unchanged, and re-ingests (with a warning) groups
whose QA file changed. The result is a single manifest JSON with one record per
sign crop, carrying per-attribute labels and a deterministic split assignment.

Split assignment is a pure function of the source image identity
(sha256(hash_key + group/file_name) mapped to [0, 1) against the configured
fractions), so images keep their split forever regardless of when their group
arrives, and crops from the same photo can never straddle splits.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from .config import REPO_ROOT, load_config, setup_logging

import logging

log = logging.getLogger("mtsd_attr")

MANIFEST_VERSION = 1


def _utcnow():
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _group_sort_key(name):
    """Sort group names numerically (GRP-2 before GRP-10)."""
    m = re.search(r"(\d+)$", name)
    return (int(m.group(1)) if m else 0, name)


def file_sha256(path):
    """Return the sha256 hex digest of a file's content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_of(image_key, hash_key, fractions):
    """Deterministically assign a source image to train/val/test.

    Args:
        image_key: Stable identifier, e.g. "GRP-1/0062fb9d-IMG_4749.jpeg".
        hash_key: Salt from the config; changing it reshuffles every split.
        fractions: Mapping split name -> fraction, iterated in insertion order.

    Returns:
        The split name whose cumulative fraction bucket contains the image's
        hash value in [0, 1).
    """
    digest = hashlib.sha256(f"{hash_key}:{image_key}".encode("utf-8")).hexdigest()
    u = int(digest[:16], 16) / float(1 << 64)
    cumulative = 0.0
    names = list(fractions)
    for name in names:
        cumulative += fractions[name]
        if u < cumulative:
            return name
    return names[-1]


def discover_groups(cfg):
    """Scan the data and annotations roots and classify every group.

    Returns:
        dict mapping group name -> {"status": one of "qa_approved",
        "awaiting_qa", "no_annotations", "qa_ambiguous"; "qa_file": Path or None}.
    """
    pattern = cfg["data"]["group_dir_pattern"]
    qa_subdir = cfg["data"]["qa_subdir"]
    qa_pattern = cfg["data"]["qa_filename_pattern"]
    groups = {}
    for d in cfg["paths"]["data_root"].glob(pattern):
        if d.is_dir() and d.name != "Annotations":
            groups[d.name] = {"status": "no_annotations", "qa_file": None}
    for d in cfg["paths"]["annotations_root"].glob(pattern):
        if not d.is_dir():
            continue
        qa_files = sorted((d / qa_subdir).glob(qa_pattern))
        if len(qa_files) == 1:
            groups[d.name] = {"status": "qa_approved", "qa_file": qa_files[0]}
        elif len(qa_files) > 1:
            groups[d.name] = {"status": "qa_ambiguous", "qa_file": None}
        else:
            groups[d.name] = {"status": "awaiting_qa", "qa_file": None}
    return dict(sorted(groups.items(), key=lambda kv: _group_sort_key(kv[0])))


def log_group_summary(groups):
    """Log which groups were found and why each is included or excluded."""
    reasons = {
        "qa_approved": "INCLUDED (Final-QA JSON present)",
        "awaiting_qa": "excluded: annotations exist but no Final-QA JSON yet",
        "no_annotations": "excluded: no annotation folder yet",
        "qa_ambiguous": "excluded: multiple QA JSONs in Final-QA, resolve manually",
    }
    log.info("Group discovery: %d groups found", len(groups))
    for name, info in groups.items():
        log.info("  %-8s %s", name, reasons[info["status"]])
    ambiguous = [n for n, i in groups.items() if i["status"] == "qa_ambiguous"]
    if ambiguous:
        log.warning("Ambiguous QA state for %s; these groups are NOT ingested", ambiguous)


def load_manifest(path):
    """Load an existing manifest JSON, or return a fresh empty structure."""
    if Path(path).is_file():
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    return {
        "version": MANIFEST_VERSION,
        "created_at": _utcnow(),
        "updated_at": None,
        "settings": None,
        "groups": {},
        "records": [],
    }


def _current_settings(cfg):
    """Extract the manifest-affecting settings for change detection."""
    return {
        "seed": cfg["seed"],
        "split_fractions": cfg["splits"]["fractions"],
        "hash_key": cfg["splits"]["hash_key"],
        "crop_padding": cfg["data"]["crop_padding"],
        "min_crop_size": cfg["data"]["min_crop_size"],
        "attributes": {
            name: {
                "classes": spec["classes"],
                "drop_values": spec.get("drop_values", []),
            }
            for name, spec in cfg["attributes"].items()
        },
    }


def _open_image_oriented(path, expected_size):
    """Open an image, applying EXIF orientation only if needed to match annotations.

    Label Studio annotates the EXIF-rotated view, so the QA JSON's width/height
    tell us which orientation the bbox coordinates live in.

    Returns:
        (PIL.Image, warning_or_None)
    """
    img = Image.open(path)
    img.load()
    if img.size == expected_size:
        return img, None
    transposed = ImageOps.exif_transpose(img)
    if transposed.size == expected_size:
        return transposed, None
    return img, (
        f"size mismatch for {path.name}: file {img.size}, "
        f"annotation {expected_size}; crop coordinates may be wrong"
    )


def _extract_group_crops(cfg, group, qa_file, crops_root):
    """Extract padded crops for one QA-approved group and build its records.

    Returns:
        (records, stats) where stats counts drops and warnings for logging.
    """
    padding = cfg["data"]["crop_padding"]
    min_size = cfg["data"]["min_crop_size"]
    fmt = cfg["data"]["crop_format"]
    quality = cfg["data"]["crop_quality"]
    attributes = cfg["attributes"]
    policy = cfg["data"]["unknown_value_policy"]
    fractions = cfg["splits"]["fractions"]
    hash_key = cfg["splits"]["hash_key"]

    with open(qa_file, encoding="utf-8-sig") as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco["images"]}
    categories = {c["id"]: c["name"] for c in coco.get("categories", [])}

    by_image = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    group_dir = crops_root / group
    group_dir.mkdir(parents=True, exist_ok=True)

    records = []
    stats = Counter()
    for image_id, anns in by_image.items():
        im_meta = images[image_id]
        source_rel = str(im_meta["source_image"]).replace("\\", "/")
        source_path = REPO_ROOT / source_rel
        image_key = f"{group}/{im_meta['file_name']}"
        split = split_of(image_key, hash_key, fractions)

        keep = []
        for ann in anns:
            w, h = ann["bbox"][2], ann["bbox"][3]
            if min(w, h) < min_size:
                stats["dropped_too_small"] += 1
                continue
            labels = {}
            dropped = False
            for attr, spec in attributes.items():
                value = (ann.get("attributes") or {}).get(attr)
                if value in spec.get("drop_values", []):
                    dropped = True
                    break
                if value is None or value == "":
                    labels[attr] = None
                    stats[f"missing_{attr}"] += 1
                elif value in spec["classes"]:
                    labels[attr] = value
                else:
                    if policy == "error":
                        raise ValueError(
                            f"{group} annotation {ann['id']}: unknown {attr} "
                            f"value {value!r} not in configured classes"
                        )
                    labels[attr] = None
                    stats[f"unknown_{attr}"] += 1
                    log.warning(
                        "%s annotation %s: unknown %s value %r masked as missing",
                        group, ann["id"], attr, value,
                    )
            if dropped:
                stats["dropped_by_value"] += 1
                continue
            keep.append((ann, labels))
        if not keep:
            continue

        if not source_path.is_file():
            stats["missing_source_images"] += 1
            log.warning("%s: source image not found, skipped: %s", group, source_rel)
            continue
        img, warning = _open_image_oriented(
            source_path, (im_meta["width"], im_meta["height"])
        )
        if warning:
            stats["orientation_warnings"] += 1
            log.warning("%s: %s", group, warning)
        img = img.convert("RGB")
        iw, ih = img.size

        for ann, labels in keep:
            x, y, w, h = ann["bbox"]
            px, py = w * padding, h * padding
            left = max(0, int(round(x - px)))
            top = max(0, int(round(y - py)))
            right = min(iw, int(round(x + w + px)))
            bottom = min(ih, int(round(y + h + py)))
            crop = img.crop((left, top, right, bottom))
            crop_name = f"ann_{ann['id']:06d}.{fmt}"
            crop_path = group_dir / crop_name
            crop.save(crop_path, quality=quality)
            records.append(
                {
                    "crop_id": f"{image_key}#{ann['id']}",
                    "group": group,
                    "image_key": image_key,
                    "source_image": source_rel,
                    "annotation_id": ann["id"],
                    "category": categories.get(ann["category_id"]),
                    "bbox": [round(v, 2) for v in ann["bbox"]],
                    "crop_path": str(crop_path.relative_to(crops_root.parent.parent)),
                    "labels": labels,
                    "split": split,
                }
            )
            stats["crops"] += 1
        stats["images"] += 1
        img.close()
    return records, stats


def label_distribution(records, attributes):
    """Count labels per attribute per split for a list of records.

    Returns:
        dict attr -> split -> Counter of class values ("<missing>" for None).
    """
    dist = {attr: {} for attr in attributes}
    for rec in records:
        for attr in attributes:
            value = rec["labels"].get(attr) or "<missing>"
            dist[attr].setdefault(rec["split"], Counter())[value] += 1
    return dist


def log_distribution(records, attributes):
    """Log per-attribute totals and per-split class distributions."""
    dist = label_distribution(records, attributes)
    splits = sorted({r["split"] for r in records})
    log.info("Manifest totals: %d crops from %d source images",
             len(records), len({r["image_key"] for r in records}))
    for split in splits:
        n = sum(1 for r in records if r["split"] == split)
        log.info("  split %-5s: %d crops (%.1f%%)", split, n, 100 * n / len(records))
    for attr in attributes:
        labelled = sum(
            c for by_split in dist[attr].values()
            for v, c in by_split.items() if v != "<missing>"
        )
        log.info("Attribute %s: %d labelled crops", attr, labelled)
        for split in splits:
            counts = dist[attr].get(split, Counter())
            desc = ", ".join(f"{v}={c}" for v, c in counts.most_common())
            log.info("    %-5s %s", split, desc)


def update_manifest(cfg, force=False):
    """Discover groups, ingest new or changed QA-approved ones, save the manifest.

    Groups already in the manifest whose QA file hash is unchanged are skipped.
    A changed hash triggers a re-ingest of that group (old records and crops for
    it are replaced) with a prominent warning, since it can shift metrics.

    Args:
        cfg: Config dict from load_config.
        force: Re-ingest every QA-approved group regardless of stored hashes.

    Returns:
        The up-to-date manifest dict.
    """
    manifest_path = cfg["paths"]["manifest"]
    crops_root = cfg["paths"]["crops_dir"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    crops_root.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    settings = _current_settings(cfg)
    if manifest["settings"] is not None and manifest["settings"] != settings and not force:
        log.warning(
            "Manifest-affecting settings changed since the manifest was built "
            "(splits, filters, or vocabularies). Existing groups keep their old "
            "processing; run with --force to rebuild everything consistently."
        )
    if force:
        manifest["groups"] = {}
        manifest["records"] = []
    manifest["settings"] = settings

    groups = discover_groups(cfg)
    log_group_summary(groups)

    for name, info in groups.items():
        if info["status"] != "qa_approved":
            continue
        qa_hash = file_sha256(info["qa_file"])
        known = manifest["groups"].get(name)
        if known and known["qa_sha256"] == qa_hash:
            log.info("%s: unchanged (QA hash match), keeping %d existing crops",
                     name, known["n_crops"])
            continue
        if known:
            log.warning(
                "%s: QA file content CHANGED since ingestion on %s; re-ingesting. "
                "Metrics from earlier runs may not be comparable for this group.",
                name, known["ingested_at"],
            )
            manifest["records"] = [r for r in manifest["records"] if r["group"] != name]
        log.info("%s: ingesting from %s", name, info["qa_file"].name)
        records, stats = _extract_group_crops(cfg, name, info["qa_file"], crops_root)
        manifest["records"].extend(records)
        manifest["groups"][name] = {
            "qa_file": str(info["qa_file"].relative_to(REPO_ROOT)),
            "qa_sha256": qa_hash,
            "ingested_at": _utcnow(),
            "n_images": stats["images"],
            "n_crops": stats["crops"],
            "n_dropped_too_small": stats["dropped_too_small"],
            "n_dropped_by_value": stats["dropped_by_value"],
        }
        log.info(
            "%s: %d crops from %d images (dropped: %d below min size, %d by "
            "drop_values; orientation warnings: %d)",
            name, stats["crops"], stats["images"], stats["dropped_too_small"],
            stats["dropped_by_value"], stats["orientation_warnings"],
        )

    stale = [g for g in manifest["groups"] if g not in groups
             or groups[g]["status"] != "qa_approved"]
    for name in stale:
        log.warning("%s: previously ingested but QA file no longer present; "
                    "keeping existing records (delete manually if intended)", name)

    manifest["records"].sort(key=lambda r: (_group_sort_key(r["group"]), r["crop_id"]))
    manifest["updated_at"] = _utcnow()
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    log.info("Manifest saved to %s", manifest_path)
    log_distribution(manifest["records"], cfg["attributes"])
    return manifest


def main():
    """CLI entry point: update (or force-rebuild) the manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild the manifest from scratch")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg["paths"]["subproject_root"] / "outputs" / "manifest.log")
    update_manifest(cfg, force=args.force)


if __name__ == "__main__":
    main()
