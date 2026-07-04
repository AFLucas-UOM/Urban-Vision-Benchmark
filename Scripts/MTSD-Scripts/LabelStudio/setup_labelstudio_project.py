#!/usr/bin/env python3
"""Create a local Label Studio project for a prepared GRP-* import."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = ROOT / "Datasets"
DEFAULT_GROUP = "GRP-1"
IMPORT_MODE_PROMPT = "prompt"
IMPORT_MODE_QA = "qa"
IMPORT_MODE_RAW = "raw"
IMPORT_CHUNK_SIZE = 50
IMPORT_RETRY_ATTEMPTS = 3
IMPORT_RETRY_SECONDS = 3


def request_json(method: str, url: str, token: str, **kwargs: Any) -> Any:
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Token {token}"
    response = requests.request(method, url, headers=headers, timeout=120, **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(f"{method} {url} failed: {response.status_code} {response.text[:1000]}")
    if response.text:
        return response.json()
    return None


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def wait_for_server(base_url: str, token: str, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/api/projects", headers={"Authorization": f"Token {token}"}, timeout=5)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"Label Studio did not become ready at {base_url}: {last_error}")


def start_server(args: argparse.Namespace, out_dir: Path) -> subprocess.Popen[str] | None:
    if args.no_start:
        return None
    label_studio = shutil.which("label-studio")
    if not label_studio:
        raise RuntimeError("Could not find `label-studio` on PATH. Start Label Studio manually or install it in an environment on PATH.")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "LabelStudioServer.log"
    log_file = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "DEBUG": "false",
            "LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN": "true",
            "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED": "true",
            "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT": str(ROOT),
            "LOCAL_FILES_SERVING_ENABLED": "true",
            "LOCAL_FILES_DOCUMENT_ROOT": str(ROOT),
        }
    )
    cmd = [
        label_studio,
        "start",
        "--no-browser",
        "--data-dir",
        str(args.data_dir),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--username",
        args.username,
        "--password",
        args.password,
        "--user-token",
        args.token,
        "--enable-legacy-api-token",
    ]
    print(f"Starting Label Studio at {args.base_url}; log: {log_path}")
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True)


def list_projects(base_url: str, token: str) -> list[dict[str, Any]]:
    payload = request_json("GET", f"{base_url}/api/projects", token)
    if isinstance(payload, dict) and "results" in payload:
        return payload["results"]
    if isinstance(payload, list):
        return payload
    return []


def delete_existing_project(base_url: str, token: str, title: str) -> list[dict[str, Any]]:
    deleted = []
    for project in list_projects(base_url, token):
        if project.get("title") == title:
            request_json("DELETE", f"{base_url}/api/projects/{project['id']}", token)
            deleted.append({"id": project["id"], "title": project["title"]})
    if deleted:
        wait_for_project_deletion(base_url, token, title)
    return deleted


def wait_for_project_deletion(base_url: str, token: str, title: str, timeout_seconds: int = 30) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if all(project.get("title") != title for project in list_projects(base_url, token)):
            return
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for existing Label Studio project {title!r} to be deleted.")


def import_batch_with_retries(base_url: str, token: str, project_id: int, batch: list[dict[str, Any]], start: int) -> Any:
    end = start + len(batch)
    for attempt in range(1, IMPORT_RETRY_ATTEMPTS + 1):
        try:
            return request_json("POST", f"{base_url}/api/projects/{project_id}/import", token, json=batch)
        except RuntimeError:
            if attempt == IMPORT_RETRY_ATTEMPTS:
                raise
            print(
                f"Import batch {start + 1}-{end} failed on attempt {attempt}; "
                f"retrying in {IMPORT_RETRY_SECONDS}s..."
            )
            time.sleep(IMPORT_RETRY_SECONDS)


def import_project(base_url: str, token: str, title: str, task_path: Path, out_dir: Path, import_mode: str) -> dict[str, Any]:
    config = (out_dir / "config.xml").read_text(encoding="utf-8")
    tasks = json.loads(task_path.read_text(encoding="utf-8-sig"))
    created = request_json("POST", f"{base_url}/api/projects", token, json={"title": title, "label_config": config})
    project_id = created["id"]
    import_responses = []
    try:
        for start in range(0, len(tasks), IMPORT_CHUNK_SIZE):
            batch = tasks[start : start + IMPORT_CHUNK_SIZE]
            print(f"Importing tasks {start + 1}-{start + len(batch)} of {len(tasks)}...")
            response = import_batch_with_retries(base_url, token, project_id, batch, start)
            import_responses.append(
                {
                    "start": start + 1,
                    "end": start + len(batch),
                    "response": response,
                }
            )
    except Exception:
        request_json("DELETE", f"{base_url}/api/projects/{project_id}", token)
        wait_for_project_deletion(base_url, token, title)
        raise
    project = request_json("GET", f"{base_url}/api/projects/{project_id}", token)
    return {
        "title": title,
        "project_id": project_id,
        "import_mode": import_mode,
        "task_file": display_path(task_path),
        "import_chunk_size": IMPORT_CHUNK_SIZE,
        "import_responses": import_responses,
        "project_counters": {
            "task_number": project.get("task_number"),
            "total_annotations_number": project.get("total_annotations_number"),
            "num_tasks_with_annotations": project.get("num_tasks_with_annotations"),
        },
    }


def first_existing(candidates: list[Path], *, kind: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    candidate_list = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find {kind}. Checked:\n{candidate_list}")


def resolve_dataset_dir(group: str) -> Path:
    return first_existing(
        [
            DATASETS_ROOT / group,
        ],
        kind=f"dataset directory for {group}",
    )


def resolve_images_dir(group: str) -> Path:
    dataset_dir = resolve_dataset_dir(group)
    return first_existing(
        [
            dataset_dir / "Images",
            dataset_dir / group,
            dataset_dir / "merged_images",
        ],
        kind=f"image directory for {group}",
    )


def resolve_annotations_dir(group: str) -> Path:
    dataset_dir = resolve_dataset_dir(group)
    return first_existing(
        [
            DATASETS_ROOT / "Annotations" / group,
            dataset_dir / "Annotations" / group,
            dataset_dir / "Annotations",
        ],
        kind=f"annotation directory for {group}",
    )


def compact_group_name(group: str) -> str:
    return group.replace("-", "")


def qa_annotations_available(group: str) -> bool:
    try:
        annotations_dir = resolve_annotations_dir(group)
    except FileNotFoundError:
        return False
    final_qa_dir = annotations_dir / "Final-QA"
    if not final_qa_dir.exists():
        return False
    return (
        (final_qa_dir / f"QA-{compact_group_name(group)}.json").is_file()
        or (final_qa_dir / f"QA-{group}.json").is_file()
        or any(path.is_file() for path in final_qa_dir.glob("QA-*.json"))
    )


def load_import_manifest(out_dir: Path) -> dict[str, Any]:
    manifest_path = out_dir / "import_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def available_task_files(group: str, out_dir: Path) -> dict[str, Path]:
    files = {
        IMPORT_MODE_QA: out_dir / f"{group}.{IMPORT_MODE_QA}.tasks.json",
        IMPORT_MODE_RAW: out_dir / f"{group}.{IMPORT_MODE_RAW}.tasks.json",
    }
    return {mode: path for mode, path in files.items() if path.exists()}


def expected_task_path(group: str, out_dir: Path, mode: str) -> Path:
    return out_dir / f"{group}.{mode}.tasks.json"


def choose_import_task(group: str, out_dir: Path, requested_mode: str, qa_available: bool) -> tuple[str, Path]:
    mode_files = available_task_files(group, out_dir)
    manifest = load_import_manifest(out_dir)
    legacy_task_path = out_dir / f"{group}.tasks.json"

    if requested_mode in {IMPORT_MODE_QA, IMPORT_MODE_RAW}:
        if requested_mode in mode_files:
            return requested_mode, mode_files[requested_mode]
        if manifest.get("import_mode") == requested_mode and legacy_task_path.exists():
            return requested_mode, legacy_task_path
        return requested_mode, expected_task_path(group, out_dir, requested_mode)

    if qa_available and sys.stdin.isatty():
        raw_path = mode_files.get(IMPORT_MODE_RAW, expected_task_path(group, out_dir, IMPORT_MODE_RAW))
        qa_path = mode_files.get(IMPORT_MODE_QA, expected_task_path(group, out_dir, IMPORT_MODE_QA))
        print("Choose Label Studio import mode:")
        print("  1. QA  - load pre-labelled QA JSON annotations")
        print("  2. RAW - create a fresh project from the raw dataset")
        while True:
            choice = input("Import mode [QA/raw]: ").strip().lower()
            if choice in {"", "1", "q", "qa"}:
                return IMPORT_MODE_QA, qa_path
            if choice in {"2", "r", "raw"}:
                return IMPORT_MODE_RAW, raw_path
            print("Please enter QA or RAW.")

    if len(mode_files) > 1:
        if sys.stdin.isatty():
            print("Prepared QA and RAW imports are both available.")
            print("Choose Label Studio import mode:")
            print("  1. QA  - load pre-labelled QA JSON annotations")
            print("  2. RAW - create a fresh project from the raw dataset")
            while True:
                choice = input("Import mode [QA/raw]: ").strip().lower()
                if choice in {"", "1", "q", "qa"}:
                    return IMPORT_MODE_QA, mode_files[IMPORT_MODE_QA]
                if choice in {"2", "r", "raw"}:
                    return IMPORT_MODE_RAW, mode_files[IMPORT_MODE_RAW]
                print("Please enter QA or RAW.")
        manifest_mode = manifest.get("import_mode")
        if manifest_mode in mode_files:
            print(f"Using prepared {manifest_mode.upper()} import from import_manifest.json.")
            return manifest_mode, mode_files[manifest_mode]
        print("Using prepared QA import. Pass --import-mode raw to use RAW instead.")
        return IMPORT_MODE_QA, mode_files[IMPORT_MODE_QA]

    if len(mode_files) == 1:
        mode, path = next(iter(mode_files.items()))
        print(f"Using prepared {mode.upper()} import.")
        return mode, path

    if legacy_task_path.exists():
        mode = manifest.get("import_mode") or IMPORT_MODE_RAW
        print(f"Using legacy prepared task file as {mode.upper()} import.")
        return mode, legacy_task_path

    raise SystemExit(f"Missing import task file. Run Scripts/LabelStudio/prepare_labelstudio_import.py first.")


def prepare_import_files(group: str, out_dir: Path, import_mode: str) -> None:
    script_path = ROOT / "Scripts" / "LabelStudio" / "prepare_labelstudio_import.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--group",
        group,
        "--output-dir",
        str(out_dir),
        "--import-mode",
        import_mode,
    ]
    print(f"Preparing {import_mode.upper()} import files...")
    subprocess.run(cmd, cwd=ROOT, check=True)


def default_source_mode(group: str) -> str:
    return IMPORT_MODE_QA if qa_annotations_available(group) else IMPORT_MODE_RAW


def recreate_local_file_storage(base_url: str, token: str, project_id: int, group: str) -> dict[str, Any]:
    image_path = resolve_images_dir(group)
    image_path = str(image_path.resolve())
    existing = request_json("GET", f"{base_url}/api/storages/localfiles/?project={project_id}", token)
    for storage in existing:
        if storage.get("title") == f"{group} images" or storage.get("path") == image_path:
            request_json("DELETE", f"{base_url}/api/storages/localfiles/{storage['id']}", token)

    return request_json(
        "POST",
        f"{base_url}/api/storages/localfiles/",
        token,
        json={
            "project": project_id,
            "title": f"{group} images",
            "path": image_path,
            "use_blob_urls": True,
            "recursive_scan": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default=DEFAULT_GROUP, help="Dataset group name, for example GRP-1.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to Datasets/<group>/labelstudio_output.")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--host", default="http://localhost:8080")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--username", default="sample@example.com")
    parser.add_argument("--password", default="SampleAnnotations123!")
    parser.add_argument("--token", default="sample-annotations-token")
    parser.add_argument("--title", help="Defaults to the group name.")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--no-start", action="store_true", help="Use an already running Label Studio server.")
    parser.add_argument("--keep-existing", action="store_true", help="Do not delete an existing project with the same title.")
    parser.add_argument("--reload", action="store_true", help="Regenerate import files from the current source, then recreate the project.")
    parser.add_argument("--wait-seconds", type=int, default=120)
    parser.add_argument(
        "--import-mode",
        choices=[IMPORT_MODE_PROMPT, IMPORT_MODE_QA, IMPORT_MODE_RAW],
        default=IMPORT_MODE_PROMPT,
        help="Choose a prepared QA or RAW task file.",
    )
    args = parser.parse_args()

    group = args.group
    out_dir = (args.output_dir or DATASETS_ROOT / group / "labelstudio_output").resolve()
    args.data_dir = args.data_dir or out_dir / "LabelStudioData"
    args.title = args.title or group

    qa_available = qa_annotations_available(group)
    prepared_reload_mode = None
    if args.reload:
        prepared_reload_mode = default_source_mode(group) if args.import_mode == IMPORT_MODE_PROMPT else args.import_mode
        prepare_import_files(group, out_dir, prepared_reload_mode)

    task_choice_mode = prepared_reload_mode or args.import_mode
    import_mode, task_path = choose_import_task(group, out_dir, task_choice_mode, qa_available)
    if not task_path.exists() or not (out_dir / "config.xml").exists():
        prepare_import_files(group, out_dir, import_mode)
        import_mode, task_path = choose_import_task(group, out_dir, import_mode, qa_available)
    missing = [path for path in (task_path, out_dir / "config.xml") if not path.exists()]
    if missing:
        raise SystemExit(f"Missing import files. Run Scripts/LabelStudio/prepare_labelstudio_import.py first: {missing}")

    proc = start_server(args, out_dir)
    try:
        wait_for_server(args.base_url, args.token, args.wait_seconds)
        log: dict[str, Any] = {
            "base_url": args.base_url,
            "title": args.title,
            "import_mode": import_mode,
            "task_file": display_path(task_path),
            "deleted_existing_projects": [],
        }
        if not args.keep_existing:
            log["deleted_existing_projects"] = delete_existing_project(args.base_url, args.token, args.title)
        log["created_project"] = import_project(args.base_url, args.token, args.title, task_path, out_dir, import_mode)
        log["local_file_storage"] = recreate_local_file_storage(
            args.base_url,
            args.token,
            log["created_project"]["project_id"],
            group,
        )
        log_path = out_dir / "LabelStudioImportLog.json"
        log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
        print(f"Wrote {log_path}")
    except Exception:
        if proc and proc.poll() is not None:
            print((out_dir / "LabelStudioServer.log").read_text(encoding="utf-8")[-4000:], file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
