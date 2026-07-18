"""Dependency-free canonical MTSD split implementation."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable, Mapping, Any

ALGORITHM_VERSION = "v2-hash-grouped"


def assign_splits(records: Iterable[Mapping[str, Any]], ratios: Mapping[str, float], seed: int) -> dict[str, str]:
    """Assign deterministic per-group splits while keeping exact image duplicates together."""
    rng = random.Random(seed)
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[str(record["group"])].append(record)
    assignment: dict[str, str] = {}
    for _, group_records in sorted(by_group.items()):
        by_hash: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in group_records:
            by_hash[str(record.get("source_image_sha256") or record["out_name"])].append(record)
        units = [
            sorted(rows, key=lambda r: str(r["out_name"]))
            for _, rows in sorted(by_hash.items(), key=lambda item: min(str(r["out_name"]) for r in item[1]))
        ]
        rng.shuffle(units)
        n_train = round(len(group_records) * float(ratios["train"]))
        n_valid = round(len(group_records) * float(ratios["valid"]))
        index = 0
        for unit in units:
            split = (
                "train" if index < n_train else "valid" if index < n_train + n_valid else "test"
            )
            for record in unit:
                assignment[str(record["out_name"])] = split
            index += len(unit)
    return assignment
