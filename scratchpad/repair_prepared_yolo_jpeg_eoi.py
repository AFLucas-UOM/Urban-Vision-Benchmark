#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


SUFFIXES = {".jpg", ".jpeg"}


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def needs_eoi(path: Path) -> bool:
    with path.open("rb") as handle:
        if path.stat().st_size < 2:
            return True
        handle.seek(-2, os.SEEK_END)
        return handle.read(2) != b"\xff\xd9"


def repair_file(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve()
    if not is_inside(resolved, allowed_root):
        raise PermissionError(f"Refusing repair outside prepared root: {resolved}")
    tmp = resolved.with_name(f".{resolved.name}.eoi.tmp")
    with resolved.open("rb") as source, tmp.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
        target.write(b"\xff\xd9")
    os.replace(tmp, resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description="Append missing JPEG EOI markers in prepared YOLO images.")
    parser.add_argument("--prepared-root", type=Path, default=Path("Datasets/MTSD/Prepared"))
    parser.add_argument("--remove-label-caches", action="store_true")
    args = parser.parse_args()

    prepared = args.prepared_root.resolve()
    roots = [
        prepared / "MTSD-Augmented" / "MTSD-YOLO",
        prepared / "MTSD-Unaugmented" / "MTSD-YOLO",
    ]
    checked = 0
    repaired: list[str] = []
    for root in roots:
        root = root.resolve()
        if not is_inside(root, prepared):
            raise PermissionError(f"YOLO root must stay inside prepared root: {root}")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUFFIXES:
                continue
            checked += 1
            if needs_eoi(path):
                repair_file(path, prepared)
                repaired.append(str(path.resolve()))

    deleted_caches: list[str] = []
    if args.remove_label_caches:
        for cache in sorted(prepared.rglob("labels.cache")):
            if not cache.is_file():
                continue
            resolved = cache.resolve()
            if not is_inside(resolved, prepared):
                raise PermissionError(f"Refusing cache delete outside prepared root: {resolved}")
            resolved.unlink()
            deleted_caches.append(str(resolved))

    print(json.dumps({
        "checked_images": checked,
        "repaired_images": len(repaired),
        "deleted_label_caches": len(deleted_caches),
        "examples": repaired[:10],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
