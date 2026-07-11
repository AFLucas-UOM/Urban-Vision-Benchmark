from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _flatten(record: dict[str, Any]) -> dict[str, Any]:
    test = record.get("unified_test_metrics", {})
    native = record.get("native_metrics", {}).get("test", {})
    return {"model": record.get("model"), "family": record.get("family"), "scale": record.get("scale"),
            "dataset_version": record.get("dataset_version"), "status": record.get("status"),
            "map50_95": test.get("map50_95"), "map50": test.get("map50"), "map75": test.get("map75"),
            "map50_95_native": native.get("metrics/mAP50-95(B)", native.get("map50_95")),
            "map50_native": native.get("metrics/mAP50(B)", native.get("map50")),
            "native_source": record.get("trainer"), "training_seconds": record.get("training_seconds"),
            "offline_augmentation_variant": record.get("offline_augmentation_variant"),
            "online_augmentation_policy": record.get("online_augmentation_policy"),
            "online_augmentation_controlled": record.get("online_augmentation_controlled"),
            "ablation_validity": record.get("ablation_validity"),
            "parameters": record.get("parameters"), "inference_ms": record.get("inference_ms"),
            "run_dir": record.get("run_dir"), "error": record.get("error")}


def generate(results_root: Path, runs_root: Path) -> Path:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in runs_root.rglob("run_record.json")]
    rows = [_flatten(record) for record in records]
    output = results_root / "Supervised-Consolidated" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    fields = list(rows[0]) if rows else ["model", "family", "scale", "dataset_version", "status"]
    with (output / "consolidated.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    (output / "consolidated.json").write_text(json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8")
    header = "| " + " | ".join(fields) + " |\n|" + "|".join(["---"] * len(fields)) + "|\n"
    body = "".join("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |\n" for row in rows)
    (output / "consolidated.md").write_text("# MTSD supervised consolidated results\n\n" + header + body, encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        completed = [row for row in rows if row.get("map50_95") is not None]
        for name, x, y, title in (("accuracy-vs-latency", "inference_ms", "map50_95", "Accuracy vs latency"),
                                  ("accuracy-vs-parameters", "parameters", "map50_95", "Accuracy vs parameters")):
            points = [row for row in completed if row.get(x) is not None]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter([row[x] for row in points], [row[y] for row in points])
            for row in points: ax.annotate(row["model"], (row[x], row[y]))
            ax.set(xlabel=x, ylabel=y, title=title); fig.tight_layout()
            fig.savefig(output / f"{name}.png", dpi=200); fig.savefig(output / f"{name}.pdf"); plt.close(fig)
    except ImportError:
        pass
    return output
