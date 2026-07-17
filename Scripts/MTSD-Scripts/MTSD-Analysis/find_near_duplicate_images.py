#!/usr/bin/env python3
"""Find visually duplicate images by comparing decoded CONTENT, not file bytes.

Companion to dedupe_exact_images.py, which only catches byte-identical files.
This script decodes every image and detects two tiers of duplicates:

1. PIXEL-EXACT  - identical decoded RGB pixels (after EXIF-orientation
                  normalisation). Catches the same photo re-saved with a
                  different encoder, stripped/changed EXIF, renamed, etc.
2. PERCEPTUAL   - visually near-identical images found via two perceptual
                  hashes (dHash and pHash) compared with Hamming distance.
                  Catches re-encodes, resizes, slight crops and burst shots.
                  These are CANDIDATES: always confirm visually in the report.

By default nothing is ever modified or deleted - this is a read-only scanner.
It writes an HTML report with side-by-side thumbnails per cluster so duplicates
can be confirmed by eye. If you want to clean up interactively, pass --review.
Review mode is also dry-run by default; add --delete to actually remove files.

Usage
-----
    # Default: scan GRP-10, write near_duplicates_report.html in the cwd
    python find_near_duplicate_images.py

    # Custom dataset / report location / sensitivity
    python find_near_duplicate_images.py --images-dir path/to/Images \
        --report out.html --threshold 10

    # Review each duplicate cluster. Dry-run: shows what would be deleted.
    python find_near_duplicate_images.py --review

    # Review and actually delete selected files.
    python find_near_duplicate_images.py --review --delete

    # Also remove matching image rows and annotations from QA-GRP*.json.
    python find_near_duplicate_images.py --review --delete --update-annotations

    # Launch a Gradio visual review UI.
    python find_near_duplicate_images.py --ui

Threshold guide (Hamming distance on 64-bit hashes): 0-4 = almost certainly
the same image; 5-10 = very similar (re-encode or burst shot); >12 = usually
just similar scenes. Default pairs are flagged when EITHER hash distance is
<= --threshold (default 10).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger("neardupe")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGES_DIR = REPO_ROOT / "Datasets" / "MTSD" / "GRP-10" / "Images"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
THUMB_WIDTH = 220  # px, thumbnails embedded in the HTML report


# --------------------------------------------------------------------------- #
# Per-image fingerprints
# --------------------------------------------------------------------------- #
@dataclass
class Fingerprint:
    path: Path
    width: int
    height: int
    file_size: int
    mtime: float
    pixel_sha: str  # sha256 of decoded RGB pixels (orientation-normalised)
    dhash: int  # 64-bit difference hash
    phash: int  # 64-bit DCT perceptual hash


def _dct_matrix(n: int = 32) -> np.ndarray:
    k = np.arange(n)[:, None]
    m = np.arange(n)[None, :]
    c = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * m + 1) * k / (2 * n))
    c[0] /= np.sqrt(2.0)
    return c


_DCT = _dct_matrix(32)


def dhash64(gray: Image.Image) -> int:
    """Classic 8x8 difference hash: brightness gradient direction per pixel."""
    a = np.asarray(gray.resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int(np.packbits(bits).tobytes().hex(), 16)


def phash64(gray: Image.Image) -> int:
    """DCT perceptual hash: low-frequency structure vs. its median."""
    a = np.asarray(gray.resize((32, 32), Image.LANCZOS), dtype=np.float64)
    freq = _DCT @ a @ _DCT.T
    low = freq[:8, :8].flatten()[1:]  # drop the DC coefficient
    bits = low > np.median(low)
    return int(np.packbits(np.concatenate([bits, [False]])).tobytes().hex(), 16)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def fingerprint_image(path: Path) -> Fingerprint | None:
    """Decode one image and compute all fingerprints. None if unreadable."""
    try:
        stat = path.stat()
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            rgb = img.convert("RGB")
            pixel_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
            gray = rgb.convert("L")
            return Fingerprint(
                path=path,
                width=rgb.width,
                height=rgb.height,
                file_size=stat.st_size,
                mtime=stat.st_mtime,
                pixel_sha=pixel_sha,
                dhash=dhash64(gray),
                phash=phash64(gray),
            )
    except (OSError, ValueError) as exc:
        log.warning("Cannot decode %s (%s) - skipped", path, exc)
        return None


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
@dataclass
class Pair:
    a: Fingerprint
    b: Fingerprint
    d_dhash: int
    d_phash: int
    pixel_exact: bool

    @property
    def score(self) -> int:
        return min(self.d_dhash, self.d_phash)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[Path, Path] = {}

    def find(self, x: Path) -> Path:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: Path, b: Path) -> None:
        self.parent[self.find(a)] = self.find(b)


def find_pairs(prints: list[Fingerprint], threshold: int) -> list[Pair]:
    """All pairs that are pixel-exact or within the perceptual threshold."""
    pairs: list[Pair] = []
    for i in range(len(prints)):
        for j in range(i + 1, len(prints)):
            a, b = prints[i], prints[j]
            exact = a.pixel_sha == b.pixel_sha
            dd = hamming(a.dhash, b.dhash)
            dp = hamming(a.phash, b.phash)
            if exact or dd <= threshold or dp <= threshold:
                pairs.append(Pair(a, b, dd, dp, exact))
    return pairs


def cluster_pairs(pairs: list[Pair]) -> list[list[Fingerprint]]:
    """Union-find the pairs into clusters for display."""
    uf = UnionFind()
    by_path: dict[Path, Fingerprint] = {}
    for pair in pairs:
        uf.union(pair.a.path, pair.b.path)
        by_path[pair.a.path] = pair.a
        by_path[pair.b.path] = pair.b
    clusters: dict[Path, list[Fingerprint]] = {}
    for path, fp in by_path.items():
        clusters.setdefault(uf.find(path), []).append(fp)
    return sorted(
        (sorted(members, key=lambda f: f.path.name) for members in clusters.values()),
        key=lambda c: c[0].path.name,
    )


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
def thumbnail_data_uri(path: Path) -> str:
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((THUMB_WIDTH, THUMB_WIDTH * 4))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except OSError:
        return ""


def cluster_pair_lookup(pairs: list[Pair]) -> dict[frozenset[Path], Pair]:
    return {frozenset({p.a.path, p.b.path}): p for p in pairs}


def write_report(
    report_path: Path,
    images_dir: Path,
    clusters: list[list[Fingerprint]],
    pairs: list[Pair],
    threshold: int,
    total_scanned: int,
) -> None:
    lookup = cluster_pair_lookup(pairs)
    parts: list[str] = [
        "<meta charset='utf-8'><title>Near-duplicate report</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:1.5rem;background:#fafafa}"
        ".cluster{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0;background:#fff}"
        ".imgs{display:flex;flex-wrap:wrap;gap:1rem}.card{max-width:240px;font-size:12px}"
        "img{max-width:220px;border:1px solid #999;display:block;margin-bottom:4px}"
        ".exact{color:#b00;font-weight:700}.meta{color:#444}"
        "table{border-collapse:collapse;font-size:12px;margin-top:.5rem}"
        "td,th{border:1px solid #ddd;padding:2px 8px}</style>",
        f"<h1>Near-duplicate report — {images_dir.name}</h1>",
        f"<p>Scanned <b>{total_scanned}</b> images in <code>{images_dir}</code> "
        f"on {datetime.now():%Y-%m-%d %H:%M}. Perceptual threshold: Hamming ≤ "
        f"{threshold} on 64-bit dHash/pHash. Found <b>{len(clusters)}</b> "
        f"cluster(s) / <b>{len(pairs)}</b> pair(s). "
        "<span class='exact'>PIXEL-EXACT</span> = decoded pixels are identical "
        "(true duplicate); others are visual candidates — confirm by eye.</p>",
    ]
    for idx, members in enumerate(clusters, start=1):
        exact = any(
            lookup[key].pixel_exact
            for key in (
                frozenset({a.path, b.path})
                for i, a in enumerate(members)
                for b in members[i + 1 :]
            )
            if key in lookup
        )
        tag = " <span class='exact'>[PIXEL-EXACT]</span>" if exact else ""
        parts.append(
            f"<div class='cluster'><h2>Cluster {idx} ({len(members)} images){tag}</h2>"
            "<div class='imgs'>"
        )
        for fp in members:
            parts.append(
                "<div class='card'>"
                f"<img src='{thumbnail_data_uri(fp.path)}' loading='lazy'>"
                f"<b>{fp.path.name}</b><br>"
                f"<span class='meta'>{fp.width}x{fp.height} · "
                f"{fp.file_size:,} B · "
                f"{datetime.fromtimestamp(fp.mtime):%Y-%m-%d %H:%M}</span></div>"
            )
        parts.append("</div><table><tr><th>pair</th><th>dHash</th><th>pHash</th>"
                     "<th>pixel-exact</th></tr>")
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                pair = lookup.get(frozenset({a.path, b.path}))
                if pair is None:
                    continue
                parts.append(
                    f"<tr><td>{a.path.name} ↔ {b.path.name}</td>"
                    f"<td>{pair.d_dhash}</td><td>{pair.d_phash}</td>"
                    f"<td>{'YES' if pair.pixel_exact else ''}</td></tr>"
                )
        parts.append("</table></div>")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(parts), encoding="utf-8")
    log.info("Report written: %s", report_path)


# --------------------------------------------------------------------------- #
# Interactive cleanup
# --------------------------------------------------------------------------- #
def describe_member(index: int, fp: Fingerprint) -> str:
    return (
        f"{index}. {fp.path.name} "
        f"({fp.width}x{fp.height}, {fp.file_size:,} B, "
        f"{datetime.fromtimestamp(fp.mtime):%Y-%m-%d %H:%M})"
    )


def qa_files_for_image(path: Path) -> list[Path]:
    """Return QA annotation files for the dataset group containing an image."""
    parts = path.resolve().parts
    group = next((part for part in reversed(parts) if part.startswith("GRP-")), None)
    if group is None:
        return []
    qa_dir = REPO_ROOT / "Datasets" / "MTSD" / "Annotations" / group / "Final-QA"
    return sorted(qa_dir.glob("QA-*.json"))


def remove_from_annotation_json(paths: list[Path], *, actually_update: bool) -> tuple[int, int]:
    """Remove image rows and their annotations from matching QA COCO JSON files."""
    names_by_qa: dict[Path, set[str]] = {}
    for path in paths:
        for qa_file in qa_files_for_image(path):
            names_by_qa.setdefault(qa_file, set()).add(path.name)

    removed_images = 0
    removed_annotations = 0
    for qa_file, names in names_by_qa.items():
        with qa_file.open(encoding="utf-8-sig") as handle:
            payload = json.load(handle)

        images = payload.get("images", [])
        annotations = payload.get("annotations", [])
        removed_ids = {
            image.get("id")
            for image in images
            if Path(str(image.get("file_name", ""))).name in names
            or Path(str(image.get("source_image", ""))).name in names
        }
        if not removed_ids:
            continue

        kept_images = [image for image in images if image.get("id") not in removed_ids]
        kept_annotations = [
            annotation for annotation in annotations if annotation.get("image_id") not in removed_ids
        ]
        image_delta = len(images) - len(kept_images)
        annotation_delta = len(annotations) - len(kept_annotations)
        removed_images += image_delta
        removed_annotations += annotation_delta

        if actually_update:
            backup = qa_file.with_suffix(
                qa_file.suffix + f".bak-{datetime.now():%Y%m%d-%H%M%S}"
            )
            shutil.copy2(qa_file, backup)
            payload["images"] = kept_images
            payload["annotations"] = kept_annotations
            qa_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            log.info(
                "Updated %s: removed %d image row(s), %d annotation(s). Backup: %s",
                qa_file,
                image_delta,
                annotation_delta,
                backup,
            )
        else:
            log.info(
                "DRY RUN would update %s: remove %d image row(s), %d annotation(s)",
                qa_file,
                image_delta,
                annotation_delta,
            )

    return removed_images, removed_annotations


def delete_paths(
    paths: list[Path],
    *,
    actually_delete: bool,
    update_annotations: bool = False,
) -> int:
    deleted = 0
    annotation_targets: list[Path] = []
    for path in paths:
        if actually_delete:
            try:
                path.unlink()
                log.info("Deleted %s", path)
                deleted += 1
                annotation_targets.append(path)
            except OSError as exc:
                log.error("Could not delete %s (%s)", path, exc)
        else:
            log.info("DRY RUN would delete %s", path)
            deleted += 1
            annotation_targets.append(path)
    if update_annotations:
        remove_from_annotation_json(annotation_targets, actually_update=actually_delete)
    return deleted


def scan_images(
    images_dir: Path,
    threshold: int,
    recursive: bool,
) -> tuple[list[Fingerprint], list[Pair], list[list[Fingerprint]]]:
    walker = images_dir.rglob("*") if recursive else images_dir.glob("*")
    paths = [p for p in sorted(walker) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    log.info("Decoding and fingerprinting %d images ...", len(paths))

    prints: list[Fingerprint] = []
    for i, path in enumerate(paths, start=1):
        fp = fingerprint_image(path)
        if fp is not None:
            prints.append(fp)
        if i % 100 == 0 or i == len(paths):
            log.info("  %d/%d done", i, len(paths))

    pairs = find_pairs(prints, threshold)
    clusters = cluster_pairs(pairs)
    return prints, pairs, clusters


def review_clusters(
    clusters: list[list[Fingerprint]],
    *,
    actually_delete: bool,
    update_annotations: bool,
) -> int:
    """Prompt the user to keep one file or delete one file per duplicate cluster."""
    if not sys.stdin.isatty():
        log.error("--review needs an interactive terminal.")
        return 2

    planned_or_deleted = 0
    mode = "DELETE" if actually_delete else "DRY RUN"
    print(f"\nInteractive duplicate review ({mode})")
    print("Commands: kN = keep image N and delete the rest, dN = delete only image N, s = skip, q = quit")
    if not actually_delete:
        print("Dry-run only. Re-run with --review --delete after checking the choices.\n")
    else:
        print("Files you choose will be permanently deleted.\n")

    for cluster_index, members in enumerate(clusters, start=1):
        print(f"Cluster {cluster_index}/{len(clusters)} ({len(members)} images)")
        for index, fp in enumerate(members, start=1):
            print("  " + describe_member(index, fp))

        while True:
            choice = input("Choose [kN/dN/s/q]: ").strip().lower()
            if choice == "s":
                print("Skipped.\n")
                break
            if choice == "q":
                print("Stopped review.")
                return planned_or_deleted
            if len(choice) < 2 or choice[0] not in {"k", "d"} or not choice[1:].isdigit():
                print("Enter kN, dN, s, or q. Example: k1 keeps image 1 and deletes the rest.")
                continue

            selected = int(choice[1:]) - 1
            if selected < 0 or selected >= len(members):
                print(f"Choose a number from 1 to {len(members)}.")
                continue

            if choice[0] == "k":
                targets = [fp.path for i, fp in enumerate(members) if i != selected]
                print(f"Keeping {members[selected].path.name}; deleting {len(targets)} other file(s).")
            else:
                targets = [members[selected].path]
                print(f"Deleting {members[selected].path.name}.")

            confirm = input("Confirm? [y/N]: ").strip().lower()
            if confirm == "y":
                planned_or_deleted += delete_paths(
                    targets,
                    actually_delete=actually_delete,
                    update_annotations=update_annotations,
                )
                print()
                break
            print("Not confirmed; choose again.")

    return planned_or_deleted


# --------------------------------------------------------------------------- #
# Gradio UI
# --------------------------------------------------------------------------- #
def cluster_choice(index: int, members: list[Fingerprint]) -> str:
    names = ", ".join(fp.path.name for fp in members[:3])
    if len(members) > 3:
        names += f", +{len(members) - 3} more"
    return f"Cluster {index}: {len(members)} image(s) - {names}"


def selected_cluster_index(choice: str | None) -> int:
    if not choice:
        return 0
    try:
        return int(choice.split(":", 1)[0].removeprefix("Cluster ").strip()) - 1
    except (ValueError, IndexError):
        return 0


def selected_image_index(choice: str | None) -> int | None:
    if not choice:
        return None
    try:
        return int(choice.split(".", 1)[0].strip()) - 1
    except (ValueError, IndexError):
        return None


def ui_state(
    images_dir: Path,
    threshold: int,
    recursive: bool,
    prints: list[Fingerprint],
    pairs: list[Pair],
    clusters: list[list[Fingerprint]],
) -> dict[str, object]:
    return {
        "images_dir": str(images_dir),
        "threshold": threshold,
        "recursive": recursive,
        "prints": prints,
        "pairs": pairs,
        "clusters": clusters,
    }


def state_clusters(state: dict[str, object] | None) -> list[list[Fingerprint]]:
    if not state:
        return []
    clusters = state.get("clusters", [])
    return clusters if isinstance(clusters, list) else []


def component_update(**kwargs):
    import gradio as gr

    return gr.update(**kwargs)


def render_cluster(
    state: dict[str, object] | None,
    choice: str | None,
) -> tuple[list[tuple[str, str]], object, str]:
    clusters = state_clusters(state)
    if not clusters:
        return [], component_update(choices=[], value=None), "No duplicate cluster selected."

    cluster_index = max(0, min(selected_cluster_index(choice), len(clusters) - 1))
    members = clusters[cluster_index]
    gallery = [
        (
            str(fp.path),
            f"{i}. {fp.path.name}\n{fp.width}x{fp.height} | {fp.file_size:,} B",
        )
        for i, fp in enumerate(members, start=1)
    ]
    image_choices = [
        f"{i}. {fp.path.name} ({fp.width}x{fp.height}, {fp.file_size:,} B)"
        for i, fp in enumerate(members, start=1)
    ]
    details = [
        f"### Cluster {cluster_index + 1} of {len(clusters)}",
        "",
        "| # | file | size | modified |",
        "|---:|---|---:|---|",
    ]
    for i, fp in enumerate(members, start=1):
        details.append(
            f"| {i} | `{html.escape(fp.path.name)}` | "
            f"{fp.width}x{fp.height}, {fp.file_size:,} B | "
            f"{datetime.fromtimestamp(fp.mtime):%Y-%m-%d %H:%M} |"
        )
    return gallery, component_update(choices=image_choices, value=image_choices[0] if image_choices else None), "\n".join(details)


def scan_for_ui(
    images_dir_text: str,
    threshold: float,
    recursive: bool,
) -> tuple[dict[str, object] | None, str, object, list[tuple[str, str]], object, str]:
    images_dir = Path(images_dir_text).expanduser().resolve()
    if not images_dir.is_dir():
        return None, f"Images directory not found: `{images_dir}`", component_update(choices=[], value=None), [], component_update(choices=[], value=None), ""

    threshold_int = int(threshold)
    prints, pairs, clusters = scan_images(images_dir, threshold_int, recursive)
    exact_pairs = [p for p in pairs if p.pixel_exact]
    state = ui_state(images_dir, threshold_int, recursive, prints, pairs, clusters)
    choices = [cluster_choice(i, members) for i, members in enumerate(clusters, start=1)]
    selected = choices[0] if choices else None
    gallery, image_update, details = render_cluster(state, selected)
    summary = (
        f"Scanned **{len(prints)}** images. Found **{len(clusters)}** cluster(s), "
        f"**{len(exact_pairs)}** pixel-exact pair(s), and "
        f"**{len(pairs) - len(exact_pairs)}** perceptual pair(s)."
    )
    if not clusters:
        summary += "\n\nNo duplicate candidates found at this threshold."
    return state, summary, component_update(choices=choices, value=selected), gallery, image_update, details


def apply_ui_action(
    state: dict[str, object] | None,
    cluster_choice_text: str | None,
    image_choice_text: str | None,
    actually_delete: bool,
    update_annotations: bool,
    action: str,
) -> tuple[dict[str, object] | None, str, object, list[tuple[str, str]], object, str]:
    clusters = state_clusters(state)
    if not clusters:
        return state, "No clusters are loaded. Scan first.", component_update(choices=[], value=None), [], component_update(choices=[], value=None), ""

    cluster_index = max(0, min(selected_cluster_index(cluster_choice_text), len(clusters) - 1))
    members = clusters[cluster_index]
    image_index = selected_image_index(image_choice_text)
    if image_index is None or image_index < 0 or image_index >= len(members):
        gallery, image_update, details = render_cluster(state, cluster_choice_text)
        return state, "Select an image in the current cluster first.", component_update(choices=[cluster_choice(i, c) for i, c in enumerate(clusters, start=1)], value=cluster_choice_text), gallery, image_update, details

    if action == "keep_selected":
        targets = [fp.path for i, fp in enumerate(members) if i != image_index]
        action_text = f"keep `{members[image_index].path.name}` and delete {len(targets)} other file(s)"
    else:
        targets = [members[image_index].path]
        action_text = f"delete `{members[image_index].path.name}`"

    changed = delete_paths(
        targets,
        actually_delete=actually_delete,
        update_annotations=update_annotations,
    )
    status = "Deleted" if actually_delete else "Dry-run would delete"
    message = f"{status} **{changed}** file(s): {action_text}."
    if update_annotations:
        message += " Annotation JSON cleanup was included."

    if actually_delete and changed:
        target_set = set(targets)
        updated_members = [fp for fp in members if fp.path not in target_set and fp.path.exists()]
        if len(updated_members) > 1:
            clusters[cluster_index] = updated_members
        else:
            del clusters[cluster_index]

    choices = [cluster_choice(i, cluster) for i, cluster in enumerate(clusters, start=1)]
    next_index = min(cluster_index, len(clusters) - 1)
    next_choice = choices[next_index] if choices else None
    gallery, image_update, details = render_cluster(state, next_choice)
    return state, message, component_update(choices=choices, value=next_choice), gallery, image_update, details


def skip_ui_cluster(
    state: dict[str, object] | None,
    cluster_choice_text: str | None,
) -> tuple[str, object, list[tuple[str, str]], object, str]:
    clusters = state_clusters(state)
    if not clusters:
        return "No clusters are loaded. Scan first.", component_update(choices=[], value=None), [], component_update(choices=[], value=None), ""

    current_index = max(0, min(selected_cluster_index(cluster_choice_text), len(clusters) - 1))
    next_index = current_index + 1
    if next_index >= len(clusters):
        next_index = 0
        message = "Skipped current cluster. Wrapped back to the first cluster."
    else:
        message = "Skipped current cluster. Nothing was deleted."

    choices = [cluster_choice(i, cluster) for i, cluster in enumerate(clusters, start=1)]
    next_choice = choices[next_index]
    gallery, image_update, details = render_cluster(state, next_choice)
    return message, component_update(choices=choices, value=next_choice), gallery, image_update, details


def build_ui(default_images_dir: Path, default_threshold: int, default_recursive: bool):
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit(
            "Gradio is not installed. Install it with: pip install gradio"
        ) from exc

    with gr.Blocks(title="Near Duplicate Image Review") as demo:
        gr.Markdown("# Near Duplicate Image Review")
        with gr.Row():
            images_dir = gr.Textbox(
                value=str(default_images_dir),
                label="Images directory",
                scale=4,
            )
            threshold = gr.Slider(
                minimum=0,
                maximum=20,
                value=default_threshold,
                step=1,
                label="Perceptual threshold",
                scale=1,
            )
            recursive = gr.Checkbox(value=default_recursive, label="Recursive")
        with gr.Row():
            scan_button = gr.Button("Scan", variant="primary")
            actually_delete = gr.Checkbox(
                value=False,
                label="Actually delete files",
                info="Leave off for dry-run review.",
            )
            update_annotations = gr.Checkbox(
                value=False,
                label="Update annotation JSON",
                info="Remove matching image rows and annotations from QA-GRP*.json.",
            )

        state = gr.State(None)
        summary = gr.Markdown()
        cluster_dropdown = gr.Dropdown(label="Duplicate cluster", choices=[], interactive=True)
        gallery = gr.Gallery(
            label="Images in selected cluster",
            columns=[2, 3, 4],
            object_fit="contain",
            height="auto",
            preview=True,
        )
        image_radio = gr.Radio(label="Selected image", choices=[], interactive=True)
        details = gr.Markdown()
        with gr.Row():
            skip_button = gr.Button("Skip cluster")
            keep_button = gr.Button("Keep selected, delete rest", variant="primary")
            delete_button = gr.Button("Delete selected only", variant="stop")

        scan_button.click(
            fn=scan_for_ui,
            inputs=[images_dir, threshold, recursive],
            outputs=[state, summary, cluster_dropdown, gallery, image_radio, details],
        )
        cluster_dropdown.change(
            fn=render_cluster,
            inputs=[state, cluster_dropdown],
            outputs=[gallery, image_radio, details],
        )
        skip_button.click(
            fn=skip_ui_cluster,
            inputs=[state, cluster_dropdown],
            outputs=[summary, cluster_dropdown, gallery, image_radio, details],
        )
        keep_button.click(
            fn=lambda s, c, i, d, a: apply_ui_action(s, c, i, d, a, "keep_selected"),
            inputs=[state, cluster_dropdown, image_radio, actually_delete, update_annotations],
            outputs=[state, summary, cluster_dropdown, gallery, image_radio, details],
        )
        delete_button.click(
            fn=lambda s, c, i, d, a: apply_ui_action(s, c, i, d, a, "delete_selected"),
            inputs=[state, cluster_dropdown, image_radio, actually_delete, update_annotations],
            outputs=[state, summary, cluster_dropdown, gallery, image_radio, details],
        )

    return demo


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only scanner for visually duplicate images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Max Hamming distance (on either hash) to flag a pair.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("near_duplicates_report.html"),
        help="Output HTML report path.",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Interactively choose one duplicate to keep or delete after scanning.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete files chosen in --review mode. Without this, review is a dry run.",
    )
    parser.add_argument(
        "--update-annotations",
        action="store_true",
        help="Also remove deleted images and their annotations from matching QA-GRP*.json files.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch a Gradio visual review UI instead of running the terminal reviewer.",
    )
    parser.add_argument(
        "--server-name",
        default="127.0.0.1",
        help="Host interface for --ui.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Port for --ui.",
    )
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    args = parse_args(argv)
    if args.delete and not args.review and not args.ui:
        log.error("--delete only works with --review.")
        return 2

    images_dir = args.images_dir.resolve()
    if not images_dir.is_dir():
        log.error("Images directory not found: %s", images_dir)
        return 2

    if args.ui:
        demo = build_ui(images_dir, args.threshold, args.recursive)
        demo.launch(server_name=args.server_name, server_port=args.server_port)
        return 0

    prints, pairs, clusters = scan_images(images_dir, args.threshold, args.recursive)
    exact_pairs = [p for p in pairs if p.pixel_exact]

    log.info(
        "Result: %d pixel-exact pair(s), %d perceptual pair(s) <= %d, "
        "%d cluster(s) total",
        len(exact_pairs),
        len(pairs) - len(exact_pairs),
        args.threshold,
        len(clusters),
    )
    for pair in sorted(pairs, key=lambda p: (not p.pixel_exact, p.score)):
        kind = "PIXEL-EXACT" if pair.pixel_exact else f"dHash={pair.d_dhash} pHash={pair.d_phash}"
        log.info("  %s <-> %s  [%s]", pair.a.path.name, pair.b.path.name, kind)

    if pairs:
        write_report(
            args.report.resolve(), images_dir, clusters, pairs, args.threshold, len(prints)
        )
        if args.review:
            changed = review_clusters(
                clusters,
                actually_delete=args.delete,
                update_annotations=args.update_annotations,
            )
            action = "Deleted" if args.delete else "Dry-run planned"
            log.info("%s %d file(s).", action, changed)
    else:
        log.info("No duplicate candidates found at threshold %d.", args.threshold)
        if args.review:
            log.info("Nothing to review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
