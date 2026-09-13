"""Unambiguous resolution of the requested benchmark matrix to exact checkpoints.

This repository contains many similarly named runs (``E0xx_``, ``FINAL_``,
``PROBE_``, ``strongaug-wandb-*``, ``[OLD]`` suites, per-resolution follow-ups).
Fuzzy prefix matching is therefore *not* used for the dissertation suite. Each
user-facing request such as

    MTSD / YOLO26-S / strong / 1280

is resolved to exactly one run directory using, in order:

1. explicit final-report provenance
   (``Documents/Final-Reports/MTSD-SupervisedDetection/mtsd_complete_experiment_matrix.csv``
   and the MDWD final detection table);
2. current final benchmark provenance
   (``Documents/Final-Tables/20260910-inference-benchmark/``);
3. run metadata on disk (``args.yaml`` / ``run_record.json`` / ``results.json``);
4. FINAL / non-probe naming;
5. the most authoritative immutable retained artifact.

If more than one eligible run survives, the request is recorded as
``ambiguous_checkpoint``. If none does, it is ``missing_checkpoint``. Neither
is silently substituted with a different model, resolution or augmentation
regime.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .paths import (
    ATTR_CHECKPOINTS,
    ATTR_SIZE_ABLATION,
    MDWD_DETECTION_RESULTS,
    MDWD_RUNS,
    MTSD_EXPERIMENT_MATRIX,
    MTSD_RUNS,
    PROJECT_ROOT,
    WORKSTATION_FINAL_TABLE,
    repo_relative,
    resolve_recorded_path,
)

# RF-DETR's high-level constructors default to these input resolutions. The
# workstation benchmark constructed RF-DETR without an explicit resolution, so
# a Jetson row is only protocol-comparable when the trained resolution it uses
# equals the package default for that scale.
RFDETR_DEFAULT_RESOLUTION = {"n": 384, "s": 512, "m": 576, "large": 728}

RFDETR_CHECKPOINT_PREFERENCE = ("checkpoint_best_total.pth", "checkpoint_best_ema.pth",
                                "checkpoint_best_regular.pth", "checkpoint.pth")

FAMILY_LABELS = {"yolo11": "YOLO11", "yolo12": "YOLO12", "yolo26": "YOLO26",
                 "rfdetr": "RF-DETR"}

ATTRIBUTE_LABELS = {
    "dinov3_vitl_lora": "DINOv3 ViT-L/16 + LoRA",
    "dinov3_vitb_lora": "DINOv3 ViT-B/16 + LoRA",
    "vjepa21_vitl_lora": "V-JEPA 2.1 ViT-L/16 + LoRA",
    "vjepa21_vitb_lora": "V-JEPA 2.1 ViT-B/16 + LoRA",
}

# Run-name markers that disqualify a run from the dissertation suite.
EXCLUDED_RUN_MARKERS = ("[OLD]", "PROBE_", "-smoke", "smoke", "probe")


@dataclass
class BenchmarkSpec:
    """One fully resolved benchmark target, or a recorded reason it is not one."""

    model_id: str                      # stable identifier, e.g. mtsd_yolo26_s_strong_1280
    display_name: str
    track: str                         # mdwd | mtsd | attributes | zero-shot
    dataset: str                       # MDWD | MTSD
    task: str                          # detection | attribute | prompt
    family: str
    scale: str
    engine: str                        # yolo | rfdetr | attribute | prompt
    adaptation: str = "supervised"
    augmentation: str = ""
    input_resolution: int | None = None
    checkpoint: str | None = None      # absolute path
    run_directory: str | None = None
    suite: str = ""
    status: str = "pending"            # see runner.STATUSES
    reason: str | None = None
    candidates: list[str] = field(default_factory=list)
    resolution_evidence: list[str] = field(default_factory=list)
    predictive_metric_name: str | None = None
    predictive_metric_value: float | None = None
    predictive_metric_source: str | None = None
    predictive_protocol: str | None = None
    workstation_comparable: bool = False
    workstation_reason: str = "not_evaluated"
    extra: dict = field(default_factory=dict)

    @property
    def checkpoint_path(self) -> Path | None:
        return Path(self.checkpoint) if self.checkpoint else None

    def to_row(self) -> dict:
        row = asdict(self)
        row["checkpoint_relative_path"] = repo_relative(self.checkpoint) if self.checkpoint else None
        row["run_directory"] = repo_relative(self.run_directory) if self.run_directory else None
        row["candidates"] = "; ".join(self.candidates)
        row["resolution_evidence"] = "; ".join(self.resolution_evidence)
        row.pop("checkpoint", None)
        row.pop("extra", None)
        return row


# ---------------------------------------------------------------------------
# Evidence sources
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _float(value) -> float | None:
    try:
        text = str(value).strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def _normalise_run_path(recorded: str) -> str:
    return str(recorded).replace("\\", "/").strip().rstrip("/")


def mtsd_experiment_rows() -> list[dict]:
    """Rows of the canonical MTSD supervised-detection experiment matrix."""
    return _read_csv(MTSD_EXPERIMENT_MATRIX)


def mdwd_result_rows() -> list[dict]:
    return _read_csv(MDWD_DETECTION_RESULTS)


def workstation_rows() -> list[dict]:
    return _read_csv(WORKSTATION_FINAL_TABLE)


def is_excluded_run(name: str) -> bool:
    lowered = name.lower()
    return any(marker.lower() in lowered for marker in EXCLUDED_RUN_MARKERS)


def yolo_checkpoint(run_dir: Path) -> Path | None:
    candidate = run_dir / "weights" / "best.pt"
    return candidate if candidate.is_file() else None


def rfdetr_checkpoint(run_dir: Path) -> Path | None:
    for name in RFDETR_CHECKPOINT_PREFERENCE:
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    return None


def run_imgsz(run_dir: Path) -> int | None:
    """Input size recorded by the run itself, never assumed."""
    args = run_dir / "args.yaml"
    if args.is_file():
        match = re.search(r"^imgsz:\s*(\d+)", args.read_text(encoding="utf-8", errors="replace"),
                          re.MULTILINE)
        if match:
            return int(match.group(1))
    record = run_dir / "run_record.json"
    if record.is_file():
        try:
            data = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        for key in ("image_size", "imgsz", "resolution"):
            value = _find_key(data, key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
    return None


def _find_key(node, key):
    if isinstance(node, dict):
        if key in node and isinstance(node[key], (int, float, str)):
            return node[key]
        for value in node.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_key(value, key)
            if found is not None:
                return found
    return None


# ---------------------------------------------------------------------------
# MDWD supervised detection
# ---------------------------------------------------------------------------

def _mdwd_candidates(family: str, scale: str, suite_hint: str | None) -> list[Path]:
    if not MDWD_RUNS.is_dir():
        return []
    token = f"rfdetr-{scale}" if family == "rfdetr" else f"{family}{scale}"
    matches: list[Path] = []
    for suite in sorted(p for p in MDWD_RUNS.iterdir() if p.is_dir()):
        if is_excluded_run(suite.name):
            continue
        if suite_hint and suite.name != suite_hint:
            continue
        for run in sorted(p for p in suite.iterdir() if p.is_dir()):
            if is_excluded_run(run.name):
                continue
            parts = run.name.split("_")
            variant = parts[1] if len(parts) > 1 else run.name
            if variant.lower() != token:
                continue
            checkpoint = rfdetr_checkpoint(run) if family == "rfdetr" else yolo_checkpoint(run)
            if checkpoint is not None:
                matches.append(run)
    return matches


def resolve_mdwd(request: dict) -> BenchmarkSpec:
    family = str(request["family"]).lower()
    scale = str(request["scale"]).lower()
    suite_hint = request.get("suite")
    label = f"{FAMILY_LABELS.get(family, family.upper())}-{scale.upper()}"
    spec = BenchmarkSpec(
        model_id=f"mdwd_{family}_{scale}",
        display_name=f"MDWD {label}",
        track="mdwd", dataset="MDWD", task="detection",
        family=family, scale=scale,
        engine="rfdetr" if family == "rfdetr" else "yolo",
        augmentation="roboflow-v20",
    )
    candidates = _mdwd_candidates(family, scale, suite_hint)
    spec.candidates = [repo_relative(c) for c in candidates]
    if not candidates:
        spec.status = "missing_checkpoint"
        spec.reason = (
            f"no retained MDWD run with a usable checkpoint for {label}"
            + (f" in suite {suite_hint}" if suite_hint else "")
            + "; the requested configuration was not trained for MDWD in this repository")
        return spec
    if len(candidates) > 1:
        # Prefer the suite named by the configuration, then the canonical EUVIP
        # suite that the final MDWD detection table was produced from.
        preferred = [c for c in candidates if c.parent.name.endswith("-EUVIP")]
        if len(preferred) == 1:
            candidates = preferred
            spec.resolution_evidence.append(
                "selected the canonical -EUVIP suite over duplicate copies")
        else:
            spec.status = "ambiguous_checkpoint"
            spec.reason = (f"{len(candidates)} eligible MDWD runs remain after applying "
                           f"the requested constraints; refusing to guess")
            return spec
    run = candidates[0]
    checkpoint = rfdetr_checkpoint(run) if family == "rfdetr" else yolo_checkpoint(run)
    spec.run_directory = str(run)
    spec.suite = run.parent.name
    spec.checkpoint = str(checkpoint)
    spec.input_resolution = run_imgsz(run) or _imgsz_from_name(run.name) or 640
    spec.resolution_evidence.append(
        f"input size {spec.input_resolution} recovered from run metadata ({run.name})")
    spec.status = "preflight_pending"
    _attach_mdwd_accuracy(spec, run)
    return spec


def _imgsz_from_name(name: str) -> int | None:
    match = re.search(r"img(\d{3,4})", name)
    return int(match.group(1)) if match else None


def _attach_mdwd_accuracy(spec: BenchmarkSpec, run: Path) -> None:
    variant = run.name.split("_")[1] if "_" in run.name else run.name
    suite_family = run.parent.name.replace("-EUVIP", "").replace("-DGX", "")
    for row in mdwd_result_rows():
        if row.get("model_variant", "").lower() != variant.lower():
            continue
        recorded = _normalise_run_path(row.get("run_path", ""))
        if run.name not in recorded:
            continue
        if row.get("model_family", "").upper().replace("-DGX", "") != suite_family.upper():
            continue
        value = _float(row.get("test_map50_95"))
        if value is None:
            continue
        spec.predictive_metric_name = "test_mAP50_95"
        spec.predictive_metric_value = round(value, 4)
        spec.predictive_metric_source = repo_relative(MDWD_DETECTION_RESULTS)
        spec.predictive_protocol = "MDWD test split, 5-class detection, Roboflow v20 export"
        spec.extra["test_map50"] = _float(row.get("test_map50"))
        spec.extra["precision"] = _float(row.get("precision"))
        spec.extra["recall"] = _float(row.get("recall"))
        spec.extra["f1"] = _float(row.get("f1"))
        spec.extra["parameters_recorded"] = _float(row.get("parameters"))
        return
    spec.predictive_metric_source = "accuracy_source_unresolved"
    spec.predictive_protocol = (
        f"no unambiguous stored test metric for {run.name} in "
        f"{repo_relative(MDWD_DETECTION_RESULTS)}")


# ---------------------------------------------------------------------------
# MTSD supervised detection
# ---------------------------------------------------------------------------

AUGMENTATION_LABELS = {"strong": "Strong", "good": "Good", "noaug": "NoAug"}


def _mtsd_matrix_matches(family: str, scale: str, augmentation: str,
                         input_size) -> list[dict]:
    wanted_arch = FAMILY_LABELS.get(family, family).upper()
    wanted_aug = AUGMENTATION_LABELS.get(augmentation.lower(), augmentation).lower()
    rows = []
    for row in mtsd_experiment_rows():
        if (row.get("architecture") or "").upper() != wanted_arch:
            continue
        if (row.get("model_scale") or "").lower() != scale.lower():
            continue
        if (row.get("augmentation_regime") or "").lower() != wanted_aug:
            continue
        if input_size not in (None, "trained"):
            if str(row.get("image_size", "")).strip() != str(input_size):
                continue
        rows.append(row)
    return rows


def _mtsd_disk_candidates(family: str, scale: str, input_size) -> list[Path]:
    """Run directories on disk carrying a usable checkpoint for family/scale."""
    suite = {"yolo11": "YOLO11-MTSD", "yolo12": "YOLO12-MTSD",
             "yolo26": "YOLO26-MTSD", "rfdetr": "RF-DETR-MTSD"}.get(family)
    root = MTSD_RUNS / suite if suite else None
    if root is None or not root.is_dir():
        return []
    token = rf"rfdetr[-_]?{scale}" if family == "rfdetr" else rf"{family}{scale}"
    matches = []
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        if is_excluded_run(run.name):
            continue
        if not re.search(token, run.name, re.IGNORECASE):
            continue
        if input_size not in (None, "trained"):
            recorded = run_imgsz(run) or _imgsz_from_name(run.name)
            if recorded != int(input_size):
                continue
        checkpoint = rfdetr_checkpoint(run) if family == "rfdetr" else yolo_checkpoint(run)
        if checkpoint is not None:
            matches.append(run)
    return matches


def resolve_mtsd(request: dict) -> BenchmarkSpec:
    family = str(request["family"]).lower()
    scale = str(request["scale"]).lower()
    augmentation = str(request.get("augmentation", "strong")).lower()
    input_size = request.get("input_size", "trained")
    label = f"{FAMILY_LABELS.get(family, family.upper())}-{scale.upper()}"
    size_token = "trained" if input_size in (None, "trained") else str(input_size)
    spec = BenchmarkSpec(
        model_id=f"mtsd_{family}_{scale}_{augmentation}_{size_token}",
        display_name=f"MTSD {label} {augmentation} @{size_token}",
        track="mtsd", dataset="MTSD", task="detection",
        family=family, scale=scale,
        engine="rfdetr" if family == "rfdetr" else "yolo",
        augmentation=augmentation,
    )

    matrix = _mtsd_matrix_matches(family, scale, augmentation, input_size)
    completed = [r for r in matrix if (r.get("completion_status") or "").lower() == "completed"
                 and _float(r.get("test_map50_95")) is not None]
    partial = [r for r in matrix if r not in completed]

    chosen_run: Path | None = None
    if completed:
        resolved = []
        for row in completed:
            run = resolve_recorded_path(row.get("run_path", ""))
            if run is None or not run.is_dir():
                continue
            checkpoint = rfdetr_checkpoint(run) if family == "rfdetr" else yolo_checkpoint(run)
            if checkpoint is not None:
                resolved.append((run, row))
        spec.candidates = [repo_relative(run) for run, _ in resolved]
        if len(resolved) == 1:
            chosen_run, chosen_row = resolved[0]
            spec.resolution_evidence.append(
                "single completed run in the MTSD experiment matrix "
                f"({repo_relative(MTSD_EXPERIMENT_MATRIX)})")
        elif len(resolved) > 1:
            chosen = _break_mtsd_tie(resolved, input_size, family, scale, spec)
            if chosen is None:
                spec.status = "ambiguous_checkpoint"
                spec.reason = (
                    f"{len(resolved)} completed runs match {label} / {augmentation} / "
                    f"{size_token} and no documented rule separates them; refusing to guess")
                return spec
            chosen_run, chosen_row = chosen
    if chosen_run is None:
        # Nothing completed. Distinguish "never finished" from "never existed".
        disk = _mtsd_disk_candidates(family, scale, input_size)
        spec.candidates = spec.candidates or [repo_relative(d) for d in disk]
        spec.status = "missing_checkpoint"
        # A run whose augmentation regime the matrix could not classify still
        # counts as evidence that the configuration was attempted but never
        # completed; surface it rather than reporting a bare absence.
        incomplete = partial + [
            row for row in mtsd_experiment_rows()
            if (row.get("architecture") or "").upper() == FAMILY_LABELS.get(family, family).upper()
            and (row.get("model_scale") or "").lower() == scale.lower()
            and (input_size in (None, "trained")
                 or str(row.get("image_size", "")).strip() == str(input_size))
            and (row.get("completion_status") or "").lower() != "completed"
        ]
        if incomplete:
            names = sorted({Path(_normalise_run_path(r.get("run_path", ""))).name
                            for r in incomplete if r.get("run_path")})
            statuses = sorted({(r.get("completion_status") or "unknown").lower()
                               for r in incomplete})
            spec.reason = (
                f"the requested {label} / {augmentation} / {size_token} configuration exists "
                f"only as run(s) the MTSD experiment matrix records as "
                f"{'/'.join(statuses)} with no test metrics ({', '.join(names)}). "
                f"Weights may be present on disk, but no completed {size_token} "
                f"{augmentation} result exists, so the configuration is reported as "
                f"missing rather than substituted with a 640 or 960 checkpoint.")
        elif disk:
            spec.reason = (
                f"run directories exist on disk for {label} @{size_token} but none is a "
                f"completed {augmentation}-augmentation run in "
                f"{repo_relative(MTSD_EXPERIMENT_MATRIX)}")
        else:
            spec.reason = (f"no MTSD run matches {label} / {augmentation} / {size_token}")
        return spec

    checkpoint = rfdetr_checkpoint(chosen_run) if family == "rfdetr" else yolo_checkpoint(chosen_run)
    spec.run_directory = str(chosen_run)
    spec.suite = chosen_run.parent.name
    spec.checkpoint = str(checkpoint)
    recorded_size = run_imgsz(chosen_run) or _imgsz_from_name(chosen_run.name)
    matrix_size = None
    try:
        matrix_size = int(str(chosen_row.get("image_size")).strip())
    except (TypeError, ValueError):
        pass
    spec.input_resolution = recorded_size or matrix_size
    spec.resolution_evidence.append(
        f"input size {spec.input_resolution} taken from the run's own metadata"
        + (f" (matrix records {matrix_size})" if matrix_size else ""))
    if family == "rfdetr":
        spec.extra["rfdetr_default_resolution"] = RFDETR_DEFAULT_RESOLUTION.get(scale)
        spec.resolution_evidence.append(
            "RF-DETR is benchmarked at its actual trained architecture resolution, "
            "never forced to 1280")
    spec.status = "preflight_pending"

    value = _float(chosen_row.get("test_map50_95"))
    if value is not None:
        spec.predictive_metric_name = "test_mAP50_95"
        spec.predictive_metric_value = round(value, 4)
        spec.predictive_metric_source = repo_relative(MTSD_EXPERIMENT_MATRIX)
        spec.predictive_protocol = (
            f"MTSD test split; {AUGMENTATION_LABELS.get(augmentation, augmentation)} "
            f"augmentation regime; image size {spec.input_resolution}; "
            f"evaluation split {chosen_row.get('evaluation_split')}")
        for key in ("test_map50", "test_precision", "test_recall", "test_f1",
                    "val_map50_95"):
            spec.extra[key] = _float(chosen_row.get(key))
    else:
        spec.predictive_metric_source = "accuracy_source_unresolved"
        spec.predictive_protocol = (
            f"no stored test mAP50-95 for {chosen_run.name}")
    return spec


def _break_mtsd_tie(resolved: list[tuple[Path, dict]], input_size, family: str,
                    scale: str, spec: BenchmarkSpec):
    """Apply the documented tie-break rules, or return None if still ambiguous."""
    # Rule 1: an explicitly requested resolution wins over follow-up resolutions.
    if input_size not in (None, "trained"):
        exact = [(run, row) for run, row in resolved
                 if str(row.get("image_size", "")).strip() == str(input_size)]
        if len(exact) == 1:
            spec.resolution_evidence.append(
                f"selected the run whose recorded image size is {input_size}")
            return exact[0]
        resolved = exact or resolved

    # Rule 2: RF-DETR "trained" means the architecture's native resolution, which
    # is the FINAL_* run; the higher-resolution runs are documented as separate
    # resolution follow-ups in the MTSD resolution ablation.
    final_named = [(run, row) for run, row in resolved if run.name.startswith("FINAL_")]
    if len(final_named) == 1:
        spec.resolution_evidence.append(
            "selected the FINAL_* retained run over resolution follow-up runs "
            "(mtsd_resolution_ablation.md documents the others as follow-ups)")
        return final_named[0]

    # Rule 3: for RF-DETR, prefer the architecture's default/native resolution.
    if family == "rfdetr":
        native = RFDETR_DEFAULT_RESOLUTION.get(scale)
        at_native = [(run, row) for run, row in resolved
                     if str(row.get("image_size", "")).strip() == str(native)]
        if len(at_native) == 1:
            spec.resolution_evidence.append(
                f"selected the run at RF-DETR-{scale.upper()}'s native resolution {native}")
            return at_native[0]

    # Rule 4: a run that the current final inference benchmark already evaluated
    # is the more authoritative retained artifact.
    benchmarked = {row.get("run_identifier") for row in workstation_rows()}
    known = [(run, row) for run, row in resolved if run.name in benchmarked]
    if len(known) == 1:
        spec.resolution_evidence.append(
            "selected the run already present in the completed inference-benchmark "
            "provenance table")
        return known[0]
    return None


# ---------------------------------------------------------------------------
# MTSD attribute classification
# ---------------------------------------------------------------------------

def resolve_attribute(variant: str) -> BenchmarkSpec:
    variant = str(variant)
    spec = BenchmarkSpec(
        model_id=f"attr_{variant}",
        display_name=ATTRIBUTE_LABELS.get(variant, variant),
        track="attributes", dataset="MTSD", task="attribute",
        family=variant.split("_")[0],
        scale="large" if "vitl" in variant else ("base" if "vitb" in variant else ""),
        engine="attribute",
        adaptation="lora" if variant.endswith("_lora") else "unknown",
    )
    directory = ATTR_CHECKPOINTS / variant
    checkpoint = directory / "best.pt"
    spec.candidates = [repo_relative(checkpoint)]
    if not checkpoint.is_file():
        spec.status = "missing_checkpoint"
        spec.reason = (f"attribute checkpoint {repo_relative(checkpoint)} does not exist; "
                       "this benchmark never retrains a variant")
        return spec
    spec.run_directory = str(directory)
    spec.checkpoint = str(checkpoint)
    spec.suite = "AttributeClassification"
    spec.status = "preflight_pending"
    spec.resolution_evidence.append(
        "canonical best.pt selected by the existing attribute-classification "
        "selection metric (validation mean macro-F1)")
    _attach_attribute_accuracy(spec, variant)
    return spec


def _attach_attribute_accuracy(spec: BenchmarkSpec, variant: str) -> None:
    for row in _read_csv(ATTR_SIZE_ABLATION):
        if row.get("variant") != variant:
            continue
        value = _float(row.get("test_mean_macro_f1"))
        if value is None:
            break
        spec.predictive_metric_name = "test_mean_macro_f1"
        spec.predictive_metric_value = round(value, 4)
        spec.predictive_metric_source = repo_relative(ATTR_SIZE_ABLATION)
        spec.predictive_protocol = (
            "MTSD attribute test split; four-head mean macro-F1 over "
            "view_angle / mounting / condition / sign_shape on ground-truth crops")
        spec.input_resolution = int(_float(row.get("resolution")) or 0) or None
        spec.extra["parameters_total_recorded"] = _int(row.get("total_params"))
        spec.extra["parameters_trainable_recorded"] = _int(row.get("trainable_params"))
        spec.extra["test_mean_macro_f1_ci"] = [
            _float(row.get("test_mean_macro_f1_ci_lower")),
            _float(row.get("test_mean_macro_f1_ci_upper"))]
        spec.extra["architecture"] = row.get("architecture")
        spec.extra["family_label"] = row.get("family")
        return
    spec.predictive_metric_source = "accuracy_source_unresolved"
    spec.predictive_protocol = (
        f"variant {variant} not present in {repo_relative(ATTR_SIZE_ABLATION)}")


def _int(value) -> int | None:
    try:
        return int(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Zero-shot localisation (optional)
# ---------------------------------------------------------------------------

PROMPT_LABELS = {
    "sam_3": "SAM 3", "sam_3.1": "SAM 3.1", "locateanything_3b": "LocateAnything 3B",
    "cosmos_reason2_2b": "Cosmos Reason2 2B", "cosmos_reason2_8b": "Cosmos Reason2 8B",
    "cosmos_reason2_32b": "Cosmos Reason2 32B",
}


def resolve_zero_shot(slug: str, dataset: str = "MTSD") -> BenchmarkSpec:
    label = PROMPT_LABELS.get(slug, slug)
    spec = BenchmarkSpec(
        model_id=f"zs_{slug}",
        display_name=label,
        track="zero-shot", dataset=dataset, task="prompt",
        family=label.split()[0].lower(), scale="",
        engine="prompt", adaptation="zero-shot",
        status="preflight_pending",
    )
    spec.extra["backend_label"] = label
    _attach_prompt_accuracy(spec, label, dataset)
    return spec


def _attach_prompt_accuracy(spec: BenchmarkSpec, label: str, dataset: str) -> None:
    from .paths import PROMPT_SENSITIVITY_ROOT

    root = PROMPT_SENSITIVITY_ROOT / dataset.upper()
    if not root.is_dir():
        spec.predictive_metric_source = "accuracy_source_unresolved"
        return
    candidates = sorted(p for p in root.iterdir()
                        if p.is_dir() and "sensitivity" in p.name and "smoke" not in p.name)
    for run in reversed(candidates):
        summary = run / "prompt_sensitivity_per_model.csv"
        for row in _read_csv(summary):
            if row.get("model") != label:
                continue
            value = _float(row.get("macro_mean_f1"))
            if value is None:
                continue
            spec.predictive_metric_name = "macro_mean_f1"
            spec.predictive_metric_value = round(value, 4)
            spec.predictive_metric_source = repo_relative(summary)
            spec.predictive_protocol = (
                f"{dataset} prompt-localisation targeted F1, macro mean across "
                f"{row.get('n_families')} paraphrase families "
                f"(protocol {row.get('protocol_version')})")
            return
    spec.predictive_metric_source = "accuracy_source_unresolved"
    spec.predictive_protocol = (
        f"no prompt-sensitivity summary row for {label} under {repo_relative(root)}")


# ---------------------------------------------------------------------------
# Matrix resolution
# ---------------------------------------------------------------------------

def resolve_matrix(config: dict, tracks: list[str] | None = None) -> list[BenchmarkSpec]:
    """Resolve the entire configured benchmark matrix, in deterministic order."""
    models = config.get("models") or {}
    order = tracks or config.get("track_order") or ["mdwd", "mtsd", "attributes", "zero-shot"]
    specs: list[BenchmarkSpec] = []
    for track in order:
        if track == "mdwd":
            specs += [resolve_mdwd(r) for r in models.get("mdwd_detection", [])]
        elif track == "mtsd":
            specs += [resolve_mtsd(r) for r in models.get("mtsd_detection", [])]
        elif track == "attributes":
            specs += [resolve_attribute(v) for v in models.get("attribute", [])]
        elif track == "zero-shot":
            zero_shot = config.get("zero_shot") or {}
            if str(zero_shot.get("mode", "auto")).lower() == "off":
                continue
            excluded = set(zero_shot.get("excluded") or [])
            dataset = zero_shot.get("prompt_dataset", "MTSD")
            specs += [resolve_zero_shot(slug, dataset)
                      for slug in zero_shot.get("candidates", [])
                      if slug not in excluded]
    return specs


def annotate_workstation_comparability(spec: BenchmarkSpec) -> BenchmarkSpec:
    """Decide whether a Jetson row may be compared with the RTX 4090 baseline.

    A comparison is only valid with the same checkpoint, task, batch size,
    timing boundary and effective input resolution. The workstation benchmark
    passed ``imgsz=640`` to every YOLO model regardless of the resolution it was
    trained at, and constructed RF-DETR without an explicit resolution (so it
    ran at the package default for that scale). Both facts are checked here.
    """
    if spec.status in ("missing_checkpoint", "ambiguous_checkpoint"):
        spec.workstation_comparable = False
        spec.workstation_reason = "no_resolved_checkpoint"
        return spec
    if spec.task == "attribute":
        spec.workstation_comparable = True
        spec.workstation_reason = "same checkpoint, same crop sample, model-forward boundary"
        return spec
    if spec.task == "prompt":
        spec.workstation_comparable = True
        spec.workstation_reason = ("same backend and prompt protocol, in-memory pipeline "
                                   "boundary, per image-prompt pair")
        return spec

    run_name = Path(spec.run_directory).name if spec.run_directory else ""
    rows = [row for row in workstation_rows()
            if row.get("run_identifier") == run_name and row.get("dataset") == spec.dataset]
    if not rows:
        spec.workstation_comparable = False
        spec.workstation_reason = (
            f"not_protocol_comparable: run {run_name} was not evaluated by the "
            "completed workstation benchmark")
        return spec
    row = rows[0]
    if spec.engine == "yolo":
        baseline_imgsz = _int(row.get("input_size"))
        if baseline_imgsz != spec.input_resolution:
            spec.workstation_comparable = False
            spec.workstation_reason = (
                f"not_protocol_comparable: workstation measured this checkpoint at "
                f"imgsz {baseline_imgsz}, the Jetson runs it at its trained "
                f"{spec.input_resolution}")
            spec.extra["workstation_row"] = row
            return spec
    else:
        default = RFDETR_DEFAULT_RESOLUTION.get(spec.scale)
        if default != spec.input_resolution:
            spec.workstation_comparable = False
            spec.workstation_reason = (
                f"not_protocol_comparable: the workstation constructed RF-DETR without "
                f"an explicit resolution (package default {default}), the Jetson uses "
                f"the trained {spec.input_resolution}")
            spec.extra["workstation_row"] = row
            return spec
    spec.workstation_comparable = True
    spec.workstation_reason = (
        "same checkpoint, same seeded sample, batch 1, same timing boundary and "
        "same effective input resolution")
    spec.extra["workstation_row"] = row
    return spec
