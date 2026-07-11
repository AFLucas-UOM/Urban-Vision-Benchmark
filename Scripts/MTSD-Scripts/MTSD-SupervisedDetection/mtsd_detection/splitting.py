"""Dependency-free canonical MTSD split implementation."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable, Mapping, Any

ALGORITHM_VERSION = "v1"


def assign_splits(records: Iterable[Mapping[str, Any]], ratios: Mapping[str, float], seed: int) -> dict[str, str]:
    """Match the original notebook: one RNG, groups sorted, records sorted then shuffled."""
    rng = random.Random(seed)
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[str(record["group"])].append(record)
    assignment: dict[str, str] = {}
    for _, group_records in sorted(by_group.items()):
        shuffled = sorted(group_records, key=lambda r: str(r["out_name"]))
        rng.shuffle(shuffled)
        n_train = round(len(shuffled) * float(ratios["train"]))
        n_valid = round(len(shuffled) * float(ratios["valid"]))
        for index, record in enumerate(shuffled):
            assignment[str(record["out_name"])] = (
                "train" if index < n_train else "valid" if index < n_train + n_valid else "test"
            )
    return assignment

