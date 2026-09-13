#!/usr/bin/env python3
"""NVIDIA Jetson edge-device inference benchmark - orchestrator.

One command runs the whole experiment:

    hardware detection -> repository preflight -> checkpoint discovery
    -> dataset verification -> environment verification -> smoke tests
    -> PASS A (protocol-compatible latency) -> PASS B (sustained telemetry)
    -> optional zero-shot preflight -> consolidation -> figures -> report

Read-only over datasets, annotations, trained checkpoints, existing experiment
directories and existing evaluation results. Nothing is retrained. Every run
writes into a NEW timestamped directory under ``Results/Jetson-Benchmark/``.

Usage:
    python run_jetson_benchmark.py --dry-run
    python run_jetson_benchmark.py --profile smoke
    python run_jetson_benchmark.py                       # dissertation profile
    python run_jetson_benchmark.py --track mdwd --track mtsd
    python run_jetson_benchmark.py --resume Results/Jetson-Benchmark/<stamp>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from jetson_bench import analysis, figures as figures_module, power as power_module, report  # noqa: E402
from jetson_bench.config import ConfigError, load_config  # noqa: E402
from jetson_bench.engines import COMPARABLE_BOUNDARY, ITEM_UNIT  # noqa: E402
from jetson_bench.models import annotate_workstation_comparability, resolve_matrix  # noqa: E402
from jetson_bench.paths import (  # noqa: E402
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    RESULTS_ROOT,
    apply_benchmark_environment,
    repo_relative,
    venv_python,
)
from jetson_bench.platform_info import (  # noqa: E402
    hardware_snapshot,
    repository_snapshot,
    set_jetson_clocks,
    set_power_mode,
    software_snapshot,
    thermal_zones,
)
from jetson_bench.preflight import (  # noqa: E402
    ENGINE_ENVIRONMENT,
    build_coverage,
    dataset_reports,
    environment_reports,
    inspect_checkpoint,
)
from jetson_bench.runner import (  # noqa: E402
    PASS_A,
    PASS_B,
    BenchmarkInterrupted,
    aggregate_repeats,
    build_job,
    cooldown,
    current_max_temperature,
    pass_a_specs,
    pass_b_specs,
    run_worker,
    summarise_pass,
)
from jetson_bench.samples import resolve_prompt, resolve_sample  # noqa: E402
from jetson_bench.state import RunState, atomic_write_csv, atomic_write_json  # noqa: E402
from jetson_bench.tegrastats import TegrastatsMonitor  # noqa: E402

TRACK_CHOICES = ("mdwd", "mtsd", "attributes", "zero-shot")


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------

class Console:
    def __init__(self, log_path: Path | None = None):
        self.handle = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = log_path.open("a", encoding="utf-8")

    def __call__(self, message: str = "") -> None:
        print(message, flush=True)
        if self.handle is not None:
            self.handle.write(message + "\n")
            self.handle.flush()

    def rule(self, title: str) -> None:
        self(f"\n{'=' * 78}\n{title}\n{'=' * 78}")

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Benchmark configuration YAML.")
    parser.add_argument("--profile", default="dissertation",
                        help="Configuration profile (smoke | dissertation | "
                             "dissertation_tensorrt, or any profile in the config).")
    parser.add_argument("--track", action="append", choices=TRACK_CHOICES, default=None,
                        help="Limit to one or more tracks (repeatable).")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Limit to specific resolved model ids (debugging).")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"],
                        help="Inference device. `cpu` is for debugging only and is "
                             "never a valid dissertation result.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Inspect, resolve and plan without loading any model "
                             "or writing a run directory. Safe on a workstation.")
    parser.add_argument("--smoke-only", action="store_true",
                        help="Run preflight and smoke tests, then stop.")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume an interrupted run directory.")
    parser.add_argument("--rerun-completed", action="store_true",
                        help="With --resume, re-measure combinations already completed.")
    parser.add_argument("--power-mode", type=int, default=None,
                        help="Request an nvpmodel mode before benchmarking "
                             "(validated against the platform's own mode list).")
    parser.add_argument("--lock-clocks", action="store_true",
                        help="Run jetson_clocks before benchmarking and record it.")
    parser.add_argument("--runtime", default=None, choices=["native", "tensorrt"],
                        help="Runtime backend. TensorRT is a separate optimised-runtime "
                             "experiment, never mixed into the native comparison.")
    parser.add_argument("--allow-model-downloads", action="store_true",
                        help="Permit Hugging Face / torch.hub downloads (off by default; "
                             "the required benchmark runs from local SSD files).")
    parser.add_argument("--no-venvs", action="store_true",
                        help="Run model workers in the current interpreter instead of "
                             "the Jetson virtual environments.")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    parser.add_argument("--no-telemetry", action="store_true",
                        help="Do not start tegrastats (latency-only run).")
    parser.add_argument("--set", action="append", dest="overrides", default=[],
                        metavar="KEY=VALUE",
                        help="Override any configuration value, e.g. "
                             "--set sustained_telemetry.repeats=1 (repeatable).")
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT,
                        help="Where the timestamped run directory is created.")
    return parser.parse_args(argv)


def parse_overrides(items: list[str]) -> dict:
    from jetson_bench.config import _value

    overrides = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        overrides[key.strip()] = _value(raw)
    return overrides


def effective_config(args: argparse.Namespace) -> dict:
    overrides = parse_overrides(args.overrides)
    if args.device:
        overrides["device"] = args.device
    if args.runtime:
        overrides["runtime.native"] = args.runtime == "native"
        overrides["runtime.tensorrt_yolo"] = args.runtime == "tensorrt"
    if args.power_mode is not None:
        overrides["power.change_nvpmodel"] = True
        overrides["power.requested_mode"] = args.power_mode
    if args.lock_clocks:
        overrides["power.lock_clocks"] = True
    if args.allow_model_downloads:
        overrides["zero_shot.allow_downloads"] = True
    if args.no_figures:
        overrides["outputs.figures"] = False
    if args.no_telemetry:
        overrides["telemetry.tegrastats_enabled"] = False
    return load_config(args.config, args.profile, overrides)


def config_digest(config: dict) -> str:
    payload = json.dumps({k: v for k, v in config.items()
                          if k not in ("config_path", "available_profiles")},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Dry-run matrix
# ---------------------------------------------------------------------------

TRACK_LABEL = {"mdwd": "MDWD", "mtsd": "MTSD", "attributes": "ATTR", "zero-shot": "ZERO"}


def print_matrix(coverage: list[dict], say) -> None:
    say(f"\n{'TRACK':<6} {'MODEL':<32} {'CHECKPOINT':<14} {'DATA':<8} "
        f"{'ENV':<8} SELECTED")
    say("-" * 80)
    for row in coverage:
        track = TRACK_LABEL.get(row.get("track", ""), row.get("track", "")[:5].upper())
        if row.get("task") == "prompt":
            checkpoint = "ON-DEMAND"
            data = "--"
            selected = "AUTO" if row.get("selected") else "NO"
        else:
            checkpoint = ("OK" if row.get("checkpoint_real")
                          else (row.get("checkpoint_issue") or "MISSING")[:14].upper())
            data = "OK" if row.get("dataset_available") else "MISSING"
            selected = "YES" if row.get("selected") else "NO"
        env = "OK" if row.get("environment_usable") else "MISSING"
        say(f"{track:<6} {row.get('requested_model', '')[:31]:<32} {checkpoint:<14} "
            f"{data:<8} {env:<8} {selected}")
    blocked = [r for r in coverage if not r.get("selected")]
    if blocked:
        say("\nNot selected:")
        for row in blocked:
            say(f"  - {row.get('requested_model')}: {row.get('status')} - "
                f"{(row.get('reason') or '')[:140]}")


# ---------------------------------------------------------------------------
# Zero-shot preflight
# ---------------------------------------------------------------------------

ZERO_SHOT_REASONS = ("dependency_unavailable", "missing_local_weights", "hf_auth_required",
                     "insufficient_memory", "unsupported_on_jetson", "smoke_test_failed",
                     "manual_setup_required")

ZERO_SHOT_PROBE = r'''
import json, importlib.util, os, sys
result = {"ok": False, "reason": None, "detail": {}}
try:
    for module in ("torch", "numpy", "PIL"):
        if importlib.util.find_spec(module) is None:
            result["reason"] = "dependency_unavailable"
            result["detail"]["missing"] = module
            print(json.dumps(result)); raise SystemExit(0)
    family = sys.argv[1]
    if family == "sam3" and importlib.util.find_spec("sam3") is None:
        result["reason"] = "dependency_unavailable"
        result["detail"]["missing"] = "sam3 (Meta native package)"
        print(json.dumps(result)); raise SystemExit(0)
    if family in ("cosmos", "locate") and importlib.util.find_spec("transformers") is None:
        result["reason"] = "dependency_unavailable"
        result["detail"]["missing"] = "transformers"
        print(json.dumps(result)); raise SystemExit(0)
    if os.environ.get("UVB_ALLOW_MODEL_DOWNLOADS") != "1":
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.errors import LocalEntryNotFoundError
        repo = sys.argv[2]
        cached = None
        try:
            cached = try_to_load_from_cache(repo, "config.json")
        except Exception:
            cached = None
        if cached in (None, "_CACHED_NO_EXIST"):
            result["reason"] = "missing_local_weights"
            result["detail"]["repo"] = repo
            result["detail"]["hint"] = ("weights are not in the repository-local HF cache; "
                                        "re-run with --allow-model-downloads to fetch them")
            print(json.dumps(result)); raise SystemExit(0)
    result["ok"] = True
except SystemExit:
    raise
except Exception as exc:
    result["reason"] = "manual_setup_required"
    result["detail"]["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result))
'''

ZERO_SHOT_FAMILY = {"sam_3": ("sam3", "facebook/sam3"),
                    "sam_3.1": ("sam3", "facebook/sam3.1"),
                    "locateanything_3b": ("locate", "nvidia/LocateAnything-3B"),
                    "cosmos_reason2_2b": ("cosmos", "nvidia/Cosmos-Reason2-2B"),
                    "cosmos_reason2_8b": ("cosmos", "nvidia/Cosmos-Reason2-8B")}


def zero_shot_preflight(spec, config: dict, use_venvs: bool, say) -> dict:
    """Cheap capability check; never fails the required benchmark."""
    slug = spec.model_id.removeprefix("zs_")
    family, repo = ZERO_SHOT_FAMILY.get(slug, ("cosmos", ""))
    verdict = {"model_id": spec.model_id, "ok": False, "reason": None, "detail": {}}

    available_gb = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(errors="replace").splitlines():
            if line.startswith("MemAvailable:"):
                available_gb = int(line.split()[1]) / 1024**2
                break
    minimum = float((config.get("zero_shot") or {}).get("min_available_ram_gb", 0))
    if available_gb is not None and available_gb < minimum:
        verdict.update(reason="insufficient_memory",
                       detail={"available_gb": round(available_gb, 2),
                               "required_gb": minimum})
        return verdict

    interpreter = venv_python("prompt") if use_venvs else Path(sys.executable)
    if not Path(interpreter).is_file():
        verdict.update(reason="dependency_unavailable",
                       detail={"error": "the prompt virtual environment does not exist"})
        return verdict
    environment = dict(os.environ)
    environment["UVB_ALLOW_MODEL_DOWNLOADS"] = (
        "1" if (config.get("zero_shot") or {}).get("allow_downloads") else "0")
    try:
        completed = subprocess.run([str(interpreter), "-s", "-c", ZERO_SHOT_PROBE,
                                    family, repo],
                                   capture_output=True, text=True, timeout=300,
                                   env=environment, check=False)
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        verdict.update(payload)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        verdict.update(reason="manual_setup_required",
                       detail={"error": f"{type(exc).__name__}: {exc}"})
    return verdict


# ---------------------------------------------------------------------------
# Environment freeze
# ---------------------------------------------------------------------------

def expand_tensorrt(specs: list, config: dict) -> list:
    """Append TensorRT variants of the YOLO configurations, as separate rows.

    The native PyTorch benchmark is never replaced: each TensorRT entry is an
    additional configuration with ``engine=yolo_tensorrt`` and
    ``runtime_backend=tensorrt``, measured by its own worker process, so a
    failed export cannot affect the native result and the two runtimes are
    never averaged together. RF-DETR, attribute and prompt models are not
    exported: giving only some models an optimised runtime would make the
    comparison meaningless.
    """
    import copy

    if not (config.get("runtime") or {}).get("tensorrt_yolo"):
        return specs
    precision = ((config.get("runtime") or {}).get("tensorrt") or {}).get(
        "precision", "fp16")
    expanded = list(specs)
    for spec in specs:
        if spec.engine != "yolo" or spec.status in ("missing_checkpoint",
                                                    "ambiguous_checkpoint"):
            continue
        variant = copy.deepcopy(spec)
        variant.model_id = f"{spec.model_id}_tensorrt"
        variant.display_name = f"{spec.display_name} [TensorRT {precision}]"
        variant.engine = "yolo_tensorrt"
        variant.workstation_comparable = False
        variant.workstation_reason = (
            "not_protocol_comparable: TensorRT is a separate optimised runtime; "
            "the workstation baseline is native PyTorch")
        variant.resolution_evidence = list(spec.resolution_evidence) + [
            "TensorRT variant of the same checkpoint at the same input size"]
        expanded.append(variant)
    return expanded


def freeze_environments(environments: dict, run_dir: Path, use_venvs: bool) -> None:
    directory = run_dir / "environment_freeze"
    directory.mkdir(parents=True, exist_ok=True)
    for name in environments:
        interpreter = venv_python(name) if use_venvs else Path(sys.executable)
        target = directory / f"{name}.txt"
        if not Path(interpreter).is_file():
            target.write_text(f"# environment {name} not created\n", encoding="utf-8")
            continue
        try:
            completed = subprocess.run([str(interpreter), "-m", "pip", "freeze"],
                                       capture_output=True, text=True, timeout=300,
                                       check=False)
            target.write_text(completed.stdout or completed.stderr, encoding="utf-8")
        except (OSError, subprocess.SubprocessError) as exc:
            target.write_text(f"# pip freeze failed: {exc}\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:  # noqa: PLR0912, PLR0915 - orchestration
    args = parse_args(argv)
    try:
        config = effective_config(args)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    apply_benchmark_environment(bool((config.get("zero_shot") or {}).get("allow_downloads")))
    device = str(config.get("device", "cuda"))
    use_venvs = not args.no_venvs
    tracks = args.track or config.get("track_order") or list(TRACK_CHOICES)

    say = Console()
    say.rule("NVIDIA Jetson edge-device inference benchmark")
    say(f"Repository : {PROJECT_ROOT}")
    say(f"Profile    : {config.get('profile')}  (config {repo_relative(config['config_path'])})")
    say(f"Tracks     : {', '.join(tracks)}")
    say(f"Device     : {device}")

    # -- 1. hardware / software / repository ---------------------------------
    say.rule("1. Platform discovery")
    hardware = hardware_snapshot()
    software = software_snapshot()
    repository = repository_snapshot(args.output_root)
    say(f"Jetson      : {hardware.get('jetson_model') or 'not detected'} "
        f"({hardware.get('architecture')})")
    say(f"L4T/JetPack : {hardware.get('l4t_version') or '?'} / "
        f"{hardware.get('jetpack_version') or '?'}")
    say(f"CUDA/TRT    : {software.get('cuda_version') or '?'} / "
        f"{software.get('tensorrt_version') or '?'}")
    say(f"PyTorch     : {software.get('torch_version') or 'not importable here'} "
        f"(cuda_available={software.get('torch_cuda_available')})")
    say(f"Storage     : {(repository['storage'].get('filesystem') or '?')} on "
        f"{(repository['storage'].get('device') or '?')}, "
        f"{repository['storage'].get('free_gb')} GB free")
    say(f"Commit      : {repository.get('git_commit_short')} "
        f"({'dirty' if repository.get('git_dirty') else 'clean'})")
    for warning in repository.get("storage_warnings") or []:
        say(f"  WARNING: {warning}")
    if not repository["writable"]["ok"]:
        say(f"FATAL: repository is not writable: {repository['writable']['error']}")
        return 2

    # -- 2. matrix resolution -------------------------------------------------
    say.rule("2. Checkpoint and configuration resolution")
    specs = [annotate_workstation_comparability(spec)
             for spec in resolve_matrix(config, tracks)]
    specs = expand_tensorrt(specs, config)
    if args.models:
        wanted = set(args.models)
        specs = [s for s in specs if s.model_id in wanted]
        missing = wanted - {s.model_id for s in specs}
        if missing:
            say(f"FATAL: unknown model id(s): {sorted(missing)}")
            return 2

    hash_checkpoints = bool((config.get("execution") or {}).get("checkpoint_hashing", True))
    for spec in specs:
        if spec.checkpoint:
            record = inspect_checkpoint(spec.checkpoint, hash_checkpoints)
            spec.extra["checkpoint_mb"] = record.size_mb
            spec.extra["checkpoint_sha256"] = record.sha256
        resolved = (Path(spec.run_directory).name if spec.run_directory
                    else (spec.reason or "unresolved"))
        say(f"  {spec.display_name:<34} -> {resolved[:70]}")
        for evidence in spec.resolution_evidence:
            say(f"      evidence: {evidence}")
        if spec.predictive_metric_value is not None:
            say(f"      accuracy: {spec.predictive_metric_name}="
                f"{spec.predictive_metric_value} from {spec.predictive_metric_source}")
        elif spec.status not in ("missing_checkpoint", "ambiguous_checkpoint"):
            say(f"      accuracy: accuracy_source_unresolved "
                f"({spec.predictive_protocol})")
        say("      workstation comparison: "
            + ("valid" if spec.workstation_comparable else spec.workstation_reason[:110]))

    # -- 3. datasets and environments ----------------------------------------
    say.rule("3. Dataset and environment verification")
    datasets = dataset_reports()
    for key, record in sorted(datasets.items()):
        say(f"  {key:<18} {'OK' if record.available else 'MISSING':<8} "
            f"{record.item_count:>7} items  {record.root}"
            + (f"  ({record.note})" if record.note else ""))
    environments = environment_reports(specs, use_venvs)
    for name, record in sorted(environments.items()):
        missing = [m for m, ok in record.modules.items() if not ok]
        say(f"  env {name:<10} {'OK' if record.usable else 'NOT READY':<10} "
            f"torch={record.torch_version or '-'} cuda={record.cuda_available} "
            + (f"missing={missing}" if missing else ""))
        if record.error:
            say(f"      {record.error}")

    coverage = build_coverage(specs, datasets, environments, hash_checkpoints)
    required = [row for row in coverage if row.get("track") != "zero-shot"]
    ready = [row for row in required if row.get("selected")]
    say(f"\nRequired benchmark: {'READY' if ready else 'NOT READY'} "
        f"({len(ready)}/{len(required)} configurations selected)")
    optional = [row for row in coverage if row.get("track") == "zero-shot"]
    if optional:
        available = [row for row in optional if row.get("environment_usable")]
        say(f"Optional zero-shot environment: "
            f"{'AVAILABLE' if available else 'SKIPPED (prompt environment not ready)'}")

    # -- 4. dry run -----------------------------------------------------------
    if args.dry_run:
        say.rule("4. Execution plan (dry run - nothing loaded, nothing written)")
        print_matrix(coverage, say)
        samples_preview = {}
        for dataset, task in sorted({(s.dataset, s.task) for s in specs}):
            sample = resolve_sample(
                task, dataset, seed=int(config.get("seed", 42)),
                max_items=int((config.get(PASS_A) or {}).get("fallback_max_images", 50)),
                reuse_workstation=bool((config.get(PASS_A) or {}).get(
                    "reuse_workstation_sample", True)),
                limit=(config.get(PASS_A) or {}).get("max_images"))
            samples_preview[f"{dataset}/{task}"] = sample.provenance()
            say(f"  sample {dataset}/{task}: {sample.count} items via "
                f"{sample.source} ({sample.workstation_run or sample.pool_root})")
        say("\nPasses that would run per selected model:")
        for pass_spec in (pass_a_specs(config, specs[0]) if specs else []):
            say(f"  PASS A repeat {pass_spec['repeat']} boundary "
                f"{pass_spec['boundary']} warmup {pass_spec['warmup']}")
        for pass_spec in (pass_b_specs(config, specs[0]) if specs else []):
            say(f"  PASS B repeat {pass_spec['repeat']} boundary "
                f"{pass_spec['boundary']} >= {pass_spec['min_seconds']}s")
        say("\nDRY RUN complete - no run directory was created.")
        say.close()
        return 0

    # -- 5. run directory -----------------------------------------------------
    if args.resume:
        run_dir = Path(args.resume).resolve()
        state = RunState.load(run_dir)
        run_id = state.data["run_id"]
        say.rule(f"Resuming run {run_id}")
    else:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(args.output_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        state = RunState(run_dir, run_id, config_digest(config))
        state.save()
    for sub in ("logs", "telemetry", "figures", "environment_freeze", "worker"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    say = Console(run_dir / "logs" / "benchmark.log")
    say(f"Run directory: {run_dir}")

    atomic_write_json(run_dir / "hardware_snapshot.json", hardware)
    atomic_write_json(run_dir / "software_snapshot.json", software)
    atomic_write_json(run_dir / "repository_snapshot.json", repository)
    atomic_write_json(run_dir / "benchmark_config.json", {
        "run_id": run_id, "started_at": datetime.now().isoformat(timespec="seconds"),
        "config": config, "config_digest": config_digest(config),
        "cli": vars(args) | {"config": str(args.config), "resume": str(args.resume),
                             "output_root": str(args.output_root)},
        "tracks": tracks, "device": device, "use_venvs": use_venvs,
    })
    atomic_write_csv(run_dir / "model_manifest.csv", [spec.to_row() for spec in specs])
    atomic_write_csv(run_dir / "requested_coverage.csv", coverage)
    freeze_environments(environments, run_dir, use_venvs)

    # -- 6. power-mode policy -------------------------------------------------
    policy = {"nvpmodel_mode": (hardware.get("nvpmodel") or {}).get("nvpmodel_mode"),
              "nvpmodel_mode_name": (hardware.get("nvpmodel") or {}).get("nvpmodel_mode_name"),
              "jetson_clocks_enabled": False,
              "nvpmodel_changed": False}
    power_config = config.get("power") or {}
    if power_config.get("change_nvpmodel") and power_config.get("requested_mode") is not None:
        say.rule("Power mode")
        outcome = set_power_mode(int(power_config["requested_mode"]))
        policy["nvpmodel_change"] = outcome
        policy["nvpmodel_changed"] = outcome["applied"]
        if outcome["applied"]:
            policy["nvpmodel_mode"] = (outcome["after"] or {}).get("nvpmodel_mode")
            policy["nvpmodel_mode_name"] = (outcome["after"] or {}).get("nvpmodel_mode_name")
            say(f"  nvpmodel set to {policy['nvpmodel_mode']}")
        else:
            say(f"  nvpmodel change NOT applied: {outcome['error']}")
    if power_config.get("lock_clocks"):
        outcome = set_jetson_clocks(True)
        policy["jetson_clocks"] = outcome
        policy["jetson_clocks_enabled"] = bool(outcome.get("applied"))
        say(f"  jetson_clocks: {'enabled' if policy['jetson_clocks_enabled'] else 'not applied'}")

    # -- 7. samples -----------------------------------------------------------
    say.rule("4. Sample resolution")
    pass_a_config = config.get(PASS_A) or {}
    samples: dict[str, object] = {}
    sample_rows: list[dict] = []
    for dataset, task in sorted({(s.dataset, s.task) for s in specs}):
        key = f"{dataset}/{task}"
        max_items = int(pass_a_config.get(
            "fallback_prompt_max_images" if task == "prompt" else "fallback_max_images", 50))
        sample = resolve_sample(
            task, dataset, seed=int(config.get("seed", 42)), max_items=max_items,
            reuse_workstation=bool(pass_a_config.get("reuse_workstation_sample", True)),
            limit=pass_a_config.get("max_images"))
        samples[key] = sample
        sample_rows += sample.manifest_rows(with_hashes=hash_checkpoints)
        say(f"  {key:<18} {sample.count:>4} items via {sample.source}"
            f" ({sample.workstation_run or sample.pool_root})")
        for note in sample.notes:
            say(f"      note: {note}")
    atomic_write_csv(run_dir / "sample_manifest.csv", sample_rows)
    atomic_write_json(run_dir / "sample_provenance.json",
                      {key: s.provenance() for key, s in samples.items()})
    prompt = resolve_prompt(config)
    if prompt.get("prompt"):
        say(f"  zero-shot prompt: '{prompt['prompt']}' "
            f"({prompt['prompt_id']}, protocol {prompt['protocol_version']})")

    # -- 8. telemetry ---------------------------------------------------------
    telemetry_config = config.get("telemetry") or {}
    monitor = None
    telemetry_status = "disabled"
    if telemetry_config.get("tegrastats_enabled", True):
        monitor = TegrastatsMonitor(
            interval_ms=int(telemetry_config.get("tegrastats_interval_ms", 100)),
            keep_raw=bool(telemetry_config.get("keep_raw_lines", True)))
        telemetry_status = "running" if monitor.start() else (
            monitor.unavailable_reason or "unavailable")
        if telemetry_status != "running":
            say(f"  telemetry unavailable: {telemetry_status}")
            monitor = None

    interrupted = False

    def _handle_sigint(_signum, _frame):
        raise BenchmarkInterrupted("SIGINT")

    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_sigint)

    summary_rows: list[dict] = []
    raw_rows: list[dict] = []
    telemetry_rows: list[dict] = []
    failure_rows: list[dict] = []
    thermal_rows: list[dict] = []
    memory_rows: list[dict] = []
    zero_shot_verdicts: list[dict] = []
    power_source = power_module.UNRESOLVED
    idle = {}
    observations: dict[tuple, list[float]] = {}

    try:
        # -- 9. idle baseline ------------------------------------------------
        if monitor is not None:
            say.rule("5. Idle baseline")
            seconds = float(telemetry_config.get("idle_baseline_seconds", 10))
            monitor.clear()
            time.sleep(seconds)
            idle_samples = list(monitor.samples)
            rails = set()
            for sample in idle_samples:
                rails.update((sample.parsed.get("power_rails_mw") or {}).keys())
            power_source = power_module.resolve_power_source(
                rails, (config.get("power") or {}).get("rail_strategy", "auto"),
                (config.get("power") or {}).get("rails"))
            idle = power_module.idle_baseline(idle_samples, power_source)
            say(f"  samples {len(idle_samples)} over {seconds:.0f}s; rails {sorted(rails) or 'none'}")
            say(f"  power source: {power_source.role} ({power_source.label or 'none'}) "
                f"- {power_source.reason}")
            say(f"  idle power {idle.get('idle_power_w_mean')} W, "
                f"idle peak temperature {idle.get('idle_temperature_peak_c')} C")
            atomic_write_json(run_dir / "telemetry" / "idle_baseline.json",
                              {**idle, "samples": [s.parsed for s in idle_samples]})

        baseline_temperature = (idle.get("idle_temperature_peak_c")
                                or current_max_temperature(monitor))
        idle_power = idle.get("idle_power_w_mean")
        context = {
            "run_id": run_id,
            "git_commit": repository.get("git_commit"),
            "git_dirty": repository.get("git_dirty"),
            "jetson_model": hardware.get("jetson_model"),
            "l4t_version": hardware.get("l4t_version"),
            "jetpack_version": hardware.get("jetpack_version"),
            "cuda_version": software.get("cuda_version"),
            "cudnn_version": software.get("cudnn_version"),
            "tensorrt_version": software.get("tensorrt_version"),
            "torch_version": software.get("torch_version"),
            "python_version": software.get("python_version"),
            "seed": config.get("seed"),
            # Overridden per model below: a run can contain both runtimes, and a
            # row must state which one produced it.
            "runtime_backend": "native_pytorch",
            "precision": (config.get("runtime") or {}).get("precision", "fp32"),
            "sample_manifest": {k: (s.workstation_manifest or s.pool_root)
                                for k, s in samples.items()},
            "sample_source": {k: s.source for k, s in samples.items()},
            **policy,
        }

        # -- 10. per-model execution -----------------------------------------
        say.rule("6. Benchmark execution")
        timeout = float((config.get("execution") or {}).get("worker_timeout_seconds", 1800))
        for spec in specs:
            if spec.status not in ("preflight_ok",):
                failure_rows.append({
                    "model_id": spec.model_id, "display_name": spec.display_name,
                    "track": spec.track, "stage": "preflight", "status": spec.status,
                    "error": spec.reason})
                state.record_model(spec.model_id, spec.status, {"reason": spec.reason})
                say(f"\n[{spec.display_name}] SKIPPED ({spec.status}): "
                    f"{(spec.reason or '')[:160]}")
                continue
            if args.resume and not args.rerun_completed and state.model_status(spec.model_id) == "ok":
                say(f"\n[{spec.display_name}] already completed - resumed, not re-run")
                for payload in state.data["completed"].values():
                    if payload.get("model_id") == spec.model_id:
                        summary_rows.append(payload["summary"])
                continue

            say(f"\n[{spec.display_name}]")
            sample = samples.get(f"{spec.dataset}/{spec.task}")
            if sample is None or not sample.items:
                spec.status = "missing_dataset"
                failure_rows.append({"model_id": spec.model_id,
                                     "display_name": spec.display_name,
                                     "track": spec.track, "stage": "sample",
                                     "status": spec.status,
                                     "error": "no inputs resolved for this task"})
                continue

            if spec.track == "zero-shot":
                verdict = zero_shot_preflight(spec, config, use_venvs, say)
                zero_shot_verdicts.append(verdict)
                if not verdict.get("ok"):
                    spec.status = "skipped"
                    spec.reason = verdict.get("reason")
                    say(f"    zero-shot preflight: SKIP ({verdict.get('reason')}) "
                        f"{verdict.get('detail')}")
                    failure_rows.append({
                        "model_id": spec.model_id, "display_name": spec.display_name,
                        "track": spec.track, "stage": "zero_shot_preflight",
                        "status": "skipped", "error": verdict.get("reason"),
                        "detail": json.dumps(verdict.get("detail", {}))})
                    state.record_model(spec.model_id, "skipped", verdict)
                    continue
                say("    zero-shot preflight: OK")

            # smoke test
            start_temperature = current_max_temperature(monitor)
            smoke_job = build_job(spec, sample.items[:1], [], config, device, prompt,
                                  smoke_only=True, software=software)
            smoke = run_worker(spec, smoke_job, run_dir, "smoke", timeout, use_venvs,
                               bool((config.get("zero_shot") or {}).get("allow_downloads")),
                               run_dir / "logs")
            for index, row in enumerate(coverage):
                if row["model_id"] == spec.model_id:
                    coverage[index]["smoke_status"] = smoke.get("status")
            if smoke.get("status") != "smoke_ok":
                spec.status = smoke.get("status", "runtime_error")
                spec.reason = smoke.get("error")
                say(f"    smoke: FAILED ({spec.status}) {(spec.reason or '')[:200]}")
                failure_rows.append({
                    "model_id": spec.model_id, "display_name": spec.display_name,
                    "track": spec.track, "stage": "smoke", "status": spec.status,
                    "error": spec.reason, "log": smoke.get("log")})
                state.record_model(spec.model_id, spec.status, {"reason": spec.reason})
                continue
            say(f"    smoke: OK ({smoke['smoke']['seconds']:.2f}s) "
                f"{smoke['smoke'].get('detail')}")
            state.record_model(spec.model_id, "smoke_ok")
            if args.smoke_only:
                continue

            # measured passes
            passes = pass_a_specs(config, spec) + pass_b_specs(config, spec)
            passes = [p for p in passes
                      if args.rerun_completed or not args.resume
                      or not state.is_complete(spec.model_id, p["pass_name"], p["repeat"])]
            if not passes:
                say("    all passes already completed - nothing to re-run")
                continue
            job = build_job(spec, sample.items, passes, config, device, prompt,
                            software=software)
            if monitor is not None:
                monitor.clear()
            worker = run_worker(spec, job, run_dir, "bench", timeout, use_venvs,
                                bool((config.get("zero_shot") or {}).get("allow_downloads")),
                                run_dir / "logs")
            if worker.get("status") not in ("ok",):
                spec.status = worker.get("status", "runtime_error")
                spec.reason = worker.get("error")
                say(f"    benchmark: FAILED ({spec.status}) {(spec.reason or '')[:200]}")
                failure_rows.append({
                    "model_id": spec.model_id, "display_name": spec.display_name,
                    "track": spec.track, "stage": "benchmark", "status": spec.status,
                    "error": spec.reason, "log": worker.get("log")})
                state.record_model(spec.model_id, spec.status, {"reason": spec.reason})
                continue

            model_thermal = {"model_id": spec.model_id, "display_name": spec.display_name,
                             "temperature_start_c": start_temperature}
            for measurement in worker.get("passes", []):
                window = (monitor.window(measurement["start_monotonic"],
                                         measurement["end_monotonic"])
                          if monitor is not None else [])
                telemetry = power_module.summarise_window(
                    window, power_source, measurement.get("timed_items"), idle_power)
                row = summarise_pass(spec, measurement, worker, context, telemetry)
                summary_rows.append(row)
                key = (spec.model_id, measurement.get("pass_name"),
                       measurement.get("timing_boundary"))
                observations.setdefault(key, []).extend(measurement.get("latencies_ms", []))
                telemetry_rows.append({
                    "model_id": spec.model_id, "display_name": spec.display_name,
                    "pass_name": measurement.get("pass_name"),
                    "repeat": measurement.get("repeat"),
                    "timing_boundary": measurement.get("timing_boundary"),
                    **{k: (json.dumps(v) if isinstance(v, dict) else v)
                       for k, v in telemetry.items()}})
                for index, (latency, sample_index) in enumerate(
                        zip(measurement.get("latencies_ms", []),
                            measurement.get("sample_indices", []))):
                    item = sample.items[sample_index] if sample_index < len(sample.items) else None
                    raw_rows.append({
                        "model": spec.model_id, "pass_name": measurement.get("pass_name"),
                        "repeat": measurement.get("repeat"),
                        "timing_boundary": measurement.get("timing_boundary"),
                        "iteration": index, "sample_index": sample_index,
                        "sample": repo_relative(item) if item else None,
                        "latency_ms": latency,
                        "timestamp": round(measurement["start_monotonic"], 4)})
                state.record_pass(spec.model_id, measurement["pass_name"],
                                  measurement["repeat"],
                                  {"model_id": spec.model_id, "summary": row})
                if measurement.get("pass_name") == PASS_B:
                    for field_name in ("temperature_mean_c", "temperature_peak_c",
                                       "temperature_end_c"):
                        if telemetry.get(field_name) is not None:
                            model_thermal[field_name] = telemetry[field_name]
                say(f"    {measurement['pass_name']} r{measurement['repeat']} "
                    f"{measurement['timing_boundary']}: "
                    f"{row['latency_mean_ms']} ms mean, "
                    f"{row.get('throughput_items_s')} {ITEM_UNIT.get(spec.engine)}/s"
                    + (f", {telemetry['power_w_mean']} W" if telemetry.get('power_w_mean') else "")
                    + (f", {telemetry['energy_j_per_item']} J/item"
                       if telemetry.get('energy_j_per_item') else ""))

            # The worker released the model, collected garbage and emptied the
            # CUDA cache before exiting; check that memory actually came back
            # so a leak cannot silently inflate the next model's figures.
            baseline_memory = (worker.get("baseline_memory") or {}).get("system_ram_used_mb")
            released_memory = (worker.get("memory_after_release") or {}).get(
                "system_ram_used_mb")
            residual = (round(released_memory - baseline_memory, 2)
                        if None not in (released_memory, baseline_memory) else None)
            tolerance = float((config.get("execution") or {}).get(
                "residual_memory_tolerance_mb", 512))
            if residual is not None and residual > tolerance:
                say(f"    NOTE: {residual:.0f} MB of system RAM had not returned when the "
                    f"worker exited (tolerance {tolerance:.0f} MB); recorded with the row")

            memory_after = worker.get("memory_after_load") or {}
            memory_rows.append({
                "model": spec.model_id, "display_name": spec.display_name,
                "runtime_backend": "tensorrt" if spec.engine == "yolo_tensorrt"
                                   else "native_pytorch",
                "torch_peak_allocated_mb": memory_after.get("torch_peak_allocated_mb"),
                "torch_peak_reserved_mb": memory_after.get("torch_peak_reserved_mb"),
                "system_ram_peak_mb": max(
                    [r.get("system_ram_peak_mb") or 0 for r in telemetry_rows
                     if r["model_id"] == spec.model_id] or [0]) or None,
                "system_ram_model_delta_mb": memory_after.get("system_ram_model_delta_mb"),
                "system_ram_residual_after_release_mb": residual,
                "residual_within_tolerance": (None if residual is None
                                              else residual <= tolerance),
                "checkpoint_mb": spec.extra.get("checkpoint_mb"),
                "adapter_checkpoint_mb": (worker.get("extra") or {}).get("adapter_checkpoint_mb"),
                "parameters_total": worker.get("parameters_total"),
                "parameters_trainable": worker.get("parameters_trainable"),
            })

            abort_at = float((config.get("thermal") or {}).get("abort_temperature_c", 95))
            peak = model_thermal.get("temperature_peak_c")
            if peak is not None and peak >= abort_at:
                say(f"    THERMAL: peak {peak} C reached the configured abort threshold "
                    f"{abort_at} C; stopping this model and cooling down")
                failure_rows.append({
                    "model_id": spec.model_id, "display_name": spec.display_name,
                    "track": spec.track, "stage": "thermal", "status": "thermal_abort",
                    "error": f"peak {peak} C >= abort threshold {abort_at} C"})
                state.record_model(spec.model_id, "thermal_abort")
            else:
                spec.status = "ok"
                state.record_model(spec.model_id, "ok")
            model_thermal.update(cooldown(config, monitor, baseline_temperature, say))
            thermal_rows.append(model_thermal)
            for index, row in enumerate(coverage):
                if row["model_id"] == spec.model_id:
                    coverage[index]["benchmark_status"] = spec.status
                    coverage[index]["status"] = spec.status

    except BenchmarkInterrupted:
        interrupted = True
        say("\nInterrupted - stopping telemetry and preserving completed results.")
        state.record_event("interrupt", "operator interrupt; partial run preserved")
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        if monitor is not None:
            atomic_write_json(run_dir / "telemetry" / "tegrastats_samples.json",
                              [{"monotonic": s.monotonic, "wall_clock": s.wall_clock,
                                **s.parsed} for s in monitor.samples])
            raw_lines = [s.parsed.get("raw", "") for s in monitor.samples]
            (run_dir / "telemetry" / "tegrastats_raw.log").write_text(
                "\n".join(raw_lines) + "\n", encoding="utf-8")
            monitor.stop()

    # -- 11. consolidation ----------------------------------------------------
    say.rule("7. Consolidation")
    # Sync every configuration's final outcome back into the coverage table so a
    # reader sees one row per requested model with its real end state, whichever
    # path it took (preflight rejection, smoke failure, runtime failure, or ok).
    final_status = {spec.model_id: spec for spec in specs}
    for row in coverage:
        spec = final_status.get(row["model_id"])
        if spec is None:
            continue
        row["status"] = spec.status
        row["reason"] = spec.reason
        row["benchmark_status"] = (
            spec.status if spec.status not in ("preflight_ok", "pending") else "not_run")
        row["selected"] = spec.status in ("ok", "smoke_ok")
    summary_rows += aggregate_repeats(summary_rows, observations)
    atomic_write_csv(run_dir / "latency_summary.csv", summary_rows)
    atomic_write_csv(run_dir / "telemetry_summary.csv", telemetry_rows)
    atomic_write_csv(run_dir / "failures.csv", failure_rows)
    atomic_write_csv(run_dir / "requested_coverage.csv", coverage)
    atomic_write_csv(run_dir / "memory_summary.csv", memory_rows)
    atomic_write_csv(run_dir / "thermal_summary.csv", thermal_rows)
    atomic_write_csv(run_dir / "raw_timings.csv.gz", raw_rows,
                     compress=bool((config.get("outputs") or {}).get(
                         "compress_raw_timings", True)))
    if zero_shot_verdicts:
        atomic_write_json(run_dir / "zero_shot_preflight.json", zero_shot_verdicts)

    specs_by_id = {spec.model_id: spec for spec in specs}
    comparison = analysis.build_comparison(summary_rows, specs_by_id)
    atomic_write_csv(run_dir / "jetson_vs_workstation.csv", comparison)

    deployment = analysis.deployment_rows(
        [r for r in summary_rows if r.get("pass_name") == PASS_B or
         (r.get("pass_name") == PASS_A and r.get("repeat") == "all")])
    deployment = _merge_deployment(deployment)
    analysis.annotate_pareto(deployment)
    atomic_write_csv(run_dir / "deployment_summary.csv", deployment)
    say(f"  latency rows      : {len(summary_rows)}")
    say(f"  telemetry rows    : {len(telemetry_rows)}")
    say(f"  raw observations  : {len(raw_rows)}")
    say(f"  comparison rows   : {len(comparison)} "
        f"({sum(1 for r in comparison if r.get('comparison_valid'))} valid)")
    say(f"  failures recorded : {len(failure_rows)}")

    # -- 12. figures and report ----------------------------------------------
    figure_manifest = []
    if (config.get("outputs") or {}).get("figures", True):
        say.rule("8. Figures")
        figure_manifest = figures_module.generate(deployment, comparison, run_dir / "figures")
        atomic_write_json(run_dir / "figures" / "figure_manifest.json", figure_manifest)
        for entry in figure_manifest:
            say(f"  {'OK   ' if entry.get('generated') else 'SKIP '} {entry['figure']}"
                + (f"  ({entry.get('reason')})" if not entry.get("generated") else ""))

    if (config.get("outputs") or {}).get("report", True):
        say.rule("9. Report")
        context = {
            "run_id": run_id, "config": config, "hardware": hardware,
            "software": software, "repository": repository, "policy": policy,
            "samples": {k: s.provenance() for k, s in samples.items()},
            "idle": idle, "power_source": power_source.as_dict(),
            "coverage": coverage, "deployment": deployment, "comparison": comparison,
            "failures": failure_rows, "figures": figure_manifest,
            "memory_rows": memory_rows, "thermal_rows": thermal_rows,
            "interpretation": report.derive_interpretation(deployment, comparison),
        }
        markdown = report.build_report(context)
        (run_dir / "benchmark_report.md").write_text(markdown, encoding="utf-8")
        say(f"  wrote {repo_relative(run_dir / 'benchmark_report.md')}")

    state.record_event("complete", "interrupted" if interrupted else "finished")
    say.rule("Done")
    say(f"Results: {run_dir}")
    if interrupted:
        say("Run was interrupted; resume with:")
        say(f"  python {Path(__file__).name} --resume {run_dir}")
    say.close()
    return 1 if interrupted else 0


def _merge_deployment(rows: list[dict]) -> list[dict]:
    """One deployment row per model: Pass A latency plus Pass B telemetry."""
    merged: dict[str, dict] = {}
    for row in rows:
        model = row.get("model")
        entry = merged.setdefault(model, {})
        for key, value in row.items():
            if value is None and key in entry:
                continue
            if key in ("latency_mean_ms", "latency_p95_ms", "throughput_items_s"):
                # Latency comes from the protocol-compatible Pass A row.
                if entry.get("_latency_from_pass_a") and row.get("pass_name") == PASS_B:
                    continue
            entry[key] = value
        if row.get("pass_name") == PASS_A:
            entry["_latency_from_pass_a"] = True
    for entry in merged.values():
        entry.pop("_latency_from_pass_a", None)
    return list(merged.values())


if __name__ == "__main__":
    raise SystemExit(main())
