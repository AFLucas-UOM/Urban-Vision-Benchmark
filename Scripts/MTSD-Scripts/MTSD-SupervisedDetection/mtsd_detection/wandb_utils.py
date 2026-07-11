from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def start_run(config: dict[str, Any], name: str, group: str, mode: str, tags: list[str]):
    if mode == "disabled": return None
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(config["repo_root"]) / ".env", override=False)
    except ImportError:
        pass
    try:
        import wandb
    except ImportError:
        return None
    os.environ["WANDB_MODE"] = mode
    return wandb.init(project=config["wandb"]["project"], entity=os.getenv(config["wandb"]["entity_env"]),
                      name=name, group=group, tags=tags, config=config, reinit=True)


def finish_run(run, record: dict[str, Any], exit_code: int = 0) -> None:
    if run is None: return
    try:
        run.summary.update(record.get("unified_test_metrics", {}))
        run.summary["status"] = record.get("status")
        run.summary["checkpoint_path"] = record.get("checkpoint_best") or record.get("checkpoint_path")
        run.summary["checkpoint_source_sha256"] = record.get("checkpoint_sha256")
        run.summary["unified_eval_status"] = record.get("unified_eval_status")
        run.finish(exit_code=exit_code)
    except Exception:
        pass
