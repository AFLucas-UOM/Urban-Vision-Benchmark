"""Weights & Biases tracking for the PromptDetect evaluation pipeline.

The repository result directory remains the authoritative, resumable record.
W&B receives the same run configuration, scalar comparison metrics, summary
tables, and a complete result-directory artifact so final MDWD/MTSD runs can
be compared in one project without making the evaluator depend on W&B for
its local persistence.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import config


WANDB_ENTITY = "mark-bugeja-university-of-malta"
WANDB_ENTITY_ENV = "WANDB_ENTITY"
WANDB_PROJECT = "MSc-MDWD-MDWD-PromptDetect"


def _load_dotenv() -> None:
    """Load repository-local secrets without ever printing them."""
    try:
        from dotenv import load_dotenv

        load_dotenv(config.PROJECT_ROOT / ".env", override=False)
    except ImportError:
        # The pinned environment includes python-dotenv, but keeping this
        # optional makes offline unit tests and source checks lightweight.
        pass


def _run_name(run_dir: Path, run_config: dict[str, Any]) -> str:
    return f"{run_config.get('dataset', 'dataset')}-{run_config.get('evaluation_protocol', 'protocol')}-{run_dir.name}"


def _run_group(run_config: dict[str, Any]) -> str:
    return f"promptdetect-{run_config.get('evaluation_protocol', 'unknown')}-{run_config.get('split', 'unknown')}"


def _tags(run_config: dict[str, Any]) -> list[str]:
    tags = [
        "promptdetect",
        str(run_config.get("evaluation_protocol", "unknown")),
        str(run_config.get("dataset", "unknown")).lower(),
        str(run_config.get("split", "unknown")).lower(),
    ]
    if run_config.get("final_enforcement", {}).get("final_mode"):
        tags.append("final")
    else:
        tags.append("development")
    return tags


def _wandb_id(run_dir: Path, run_config: dict[str, Any]) -> str:
    # The run directory is immutable and unique, so using it as the W&B ID
    # makes reports-only regeneration and resume continue the same W&B run.
    raw = _run_name(run_dir, run_config).lower()
    # W&B permanently tombstones deleted IDs. A short execution revision lets
    # an intentionally replaced run retain its stable local directory and
    # display name while publishing to a fresh, resumable tracker ID.
    revision = str(run_config.get("wandb_run_revision") or "").strip()
    if revision:
        raw = f"{raw}-{revision}"
    return config.model_slug(raw)[:64]


def tracking_target() -> str:
    entity = os.getenv(WANDB_ENTITY_ENV) or WANDB_ENTITY
    return f"{entity}/{WANDB_PROJECT}"


def start_run(run_dir: Path, run_config: dict[str, Any], mode: str = "online") -> Any:
    """Start a W&B run, failing loudly when online tracking is unavailable.

    ``mode=offline`` is available for development and creates a local W&B
    queue; it does not claim that results were uploaded to the workstation.
    ``mode=disabled`` is intended only for source/tests and returns ``None``.
    """
    if mode == "disabled":
        return None
    _load_dotenv()
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B is required for PromptDetect evaluation tracking. Install "
            "wandb in the mtsd-base environment or use --wandb-mode disabled "
            "only for a non-final development check."
        ) from exc

    init_config = dict(run_config)
    init_config["wandb_target"] = tracking_target()
    init_config["wandb_mode"] = mode
    entity = os.getenv(WANDB_ENTITY_ENV) or WANDB_ENTITY
    (run_dir / "wandb").mkdir(parents=True, exist_ok=True)
    try:
        return wandb.init(
            entity=entity,
            project=WANDB_PROJECT,
            name=_run_name(run_dir, run_config),
            id=_wandb_id(run_dir, run_config),
            resume="allow",
            group=_run_group(run_config),
            job_type="evaluation",
            tags=_tags(run_config),
            config=init_config,
            dir=str(run_dir / "wandb"),
            mode=mode,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not initialize W&B target {tracking_target()} in {mode!r} mode: {exc}"
        ) from exc


def _number(value: Any) -> float | int | None:
    try:
        if value in (None, "") or isinstance(value, bool):
            return None
        number = float(value)
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return None


def _numeric_payload(prefix: str, row: dict[str, Any], keys: tuple[str, ...]) -> dict[str, float | int]:
    payload: dict[str, float | int] = {}
    for key in keys:
        number = _number(row.get(key))
        if number is not None:
            payload[f"{prefix}/{key}"] = number
    return payload


def log_combination(run: Any, metric: dict[str, Any]) -> None:
    """Log one completed model × prompt combination immediately."""
    if run is None:
        return
    payload = _numeric_payload(
        "combination",
        metric,
        ("tp", "fp", "fn", "duplicates", "precision", "recall", "f1",
         "accuracy", "mean_matched_iou", "ap50", "map50_95", "mean_inference_ms"),
    )
    if payload:
        # The prompt/model identity is stored with the row in the summary
        # tables and artifact. These scalars provide a compact dashboard trace.
        payload["combination/target_gt_boxes"] = int(metric.get("target_gt_boxes") or 0)
        try:
            run.log(payload)
        except Exception as exc:
            # Local atomic combination files remain authoritative; a transient
            # W&B transport error must not invalidate already-completed work.
            print(f"W&B combination logging skipped: {exc}")


def _read_csv(path: Path) -> tuple[list[str], list[list[Any]]]:
    if not path.is_file() or not path.stat().st_size:
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return [], []
    columns = list(rows[0])
    return columns, [[row.get(column, "") for column in columns] for row in rows]


def _log_table(run: Any, key: str, path: Path) -> None:
    import wandb

    columns, data = _read_csv(path)
    if columns:
        run.log({key: wandb.Table(columns=columns, data=data)})


def _log_final_metrics(run: Any, run_dir: Path, summary: dict[str, Any]) -> None:
    """Log comparison-ready summary scalars and tables for either protocol."""
    import wandb

    protocol = summary.get("evaluation_protocol", "targeted-v2")
    if str(protocol).startswith("prompt-sensitivity-"):
        table_files = (
            "prompt_sensitivity_per_prompt.csv",
            "prompt_sensitivity_per_family.csv",
            "prompt_sensitivity_per_model.csv",
            "prompt_sensitivity_variant_differences.csv",
            "prompt_consistency_per_model.csv",
            "prompt_consistency_per_family.csv",
            "prompt_pair_consistency.csv",
        )
        model_rows = summary.get("per_model_sensitivity", [])
        for row in model_rows:
            model = config.model_slug(str(row.get("model", "unknown")))
            for key in ("macro_mean_f1", "mean_f1_range", "mean_relative_f1_degradation",
                        "macro_canonical_f1", "macro_instruction_f1", "mean_inference_ms"):
                number = _number(row.get(key))
                if number is not None:
                    run.log({f"final/{model}/{key}": number})
    else:
        table_files = (
            "per_prompt_metrics.csv",
            "per_model_summary.csv",
            "per_target_class_metrics.csv",
            "prompt_group_summary.csv",
            "prompt_vs_class_confusion.csv",
            "fp_nontarget_overlap.csv",
        )
        model_rows = summary.get("per_model_summary", [])
        for row in model_rows:
            model = config.model_slug(str(row.get("model", "unknown")))
            for key in ("precision", "recall", "f1", "accuracy", "mean_matched_iou", "mean_inference_ms"):
                number = _number(row.get(key))
                if number is not None:
                    run.log({f"final/{model}/{key}": number})

    for filename in table_files:
        _log_table(run, f"tables/{Path(filename).stem}", run_dir / filename)

    run.summary.update({
        "dataset": summary.get("dataset"),
        "split": summary.get("split"),
        "evaluation_protocol": protocol,
        "protocol_version": summary.get("protocol_version"),
        "protocol_hash": summary.get("protocol_hash"),
        "dataset_manifest_hash": summary.get("dataset_manifest_hash"),
        "n_images": summary.get("n_images"),
        "models": summary.get("models"),
        "wandb_target": tracking_target(),
    })


def log_and_finish(run: Any, run_dir: Path, summary: dict[str, Any], exit_code: int = 0) -> None:
    """Upload final tables and every generated result as one W&B artifact."""
    if run is None:
        return
    import wandb

    _log_final_metrics(run, run_dir, summary)
    metadata = {
        "dataset": summary.get("dataset"),
        "split": summary.get("split"),
        "evaluation_protocol": summary.get("evaluation_protocol"),
        "protocol_hash": summary.get("protocol_hash"),
        "dataset_manifest_hash": summary.get("dataset_manifest_hash"),
        "tracking_target": tracking_target(),
    }
    artifact = wandb.Artifact(f"{run_dir.name}-results", type="promptdetect-results", metadata=metadata)
    added = False
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or "wandb" in path.relative_to(run_dir).parts:
            continue
        artifact.add_file(str(path), name=str(path.relative_to(run_dir)).replace("\\", "/"))
        added = True
    if added:
        run.log_artifact(artifact)
    run.finish(exit_code=exit_code)
