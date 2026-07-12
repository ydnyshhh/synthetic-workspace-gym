from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from synthetic_workspace_gym.schemas import EnvironmentManifest

from .schemas import CandidateAction, CounterfactualSnapshot, stable_id


def _candidate(snapshot: CounterfactualSnapshot, kind: str, action: dict[str, Any] | None, source: str = "trajectory", *, privileged: bool = False, rationale: str | None = None) -> CandidateAction:
    group = stable_id("cf-group", snapshot.snapshot_id)
    return CandidateAction(stable_id("candidate", group, kind, action), group, snapshot.snapshot_id, kind, action, source, rationale, privileged)


def original_candidate(snapshot: CounterfactualSnapshot, manifest: EnvironmentManifest, root: Path) -> CandidateAction | None:
    return _candidate(snapshot, "original", snapshot.original_action) if snapshot.original_action else None


def submit_candidate(snapshot: CounterfactualSnapshot, manifest: EnvironmentManifest, root: Path) -> CandidateAction:
    task = _visible_task(root)
    target = task.get("required_output_path") or task.get("output_path") or task.get("target_path") or "done"
    return _candidate(snapshot, "submit", {"tool": "submit", "args": {"path_or_answer": target}}, "visible_task", rationale="submit the declared output")


def run_public_check_candidate(snapshot: CounterfactualSnapshot, manifest: EnvironmentManifest, root: Path) -> CandidateAction | None:
    entrypoint = _visible_task(root).get("entrypoint")
    if not entrypoint:
        return None
    tool = "run_python" if str(entrypoint).strip().startswith("python ") else "run_shell"
    args = {"command_or_script" if tool == "run_python" else "command": entrypoint}
    return _candidate(snapshot, "run_public_check", {"tool": tool, "args": args}, "visible_task", rationale="run the documented public check")


def read_relevant_file_candidate(snapshot: CounterfactualSnapshot, manifest: EnvironmentManifest, root: Path) -> CandidateAction | None:
    task = _visible_task(root)
    read_paths = {(event.get("action_arguments") or {}).get("path") for event in snapshot.swg_events_prefix if event.get("action_type") == "read_file"}
    options = list(task.get("input_files", []))
    for key in ("target_path", "source_path", "config_path"):
        if task.get(key): options.append(task[key])
    options.extend(["README.md", "task.json"])
    for value in options:
        path = str(value).replace("\\", "/")
        if _safe_relative(path) and path not in read_paths and (root / "visible" / path).is_file():
            return _candidate(snapshot, "read_relevant_file", {"tool": "read_file", "args": {"path": path}}, "visible_task", rationale="inspect an unexamined task-relevant file")
    return None


def skip_repeated_candidate(snapshot: CounterfactualSnapshot, manifest: EnvironmentManifest, root: Path) -> CandidateAction | None:
    if snapshot.original_action == snapshot.previous_action and snapshot.original_action:
        return _candidate(snapshot, "skip_repeated_action", None, "trajectory", rationale="omit a repeated action")
    return None


def _visible_task(root: Path) -> dict[str, Any]:
    path = root / "visible" / "task.json"
    if not path.exists(): return {}
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError): return {}


def _safe_relative(value: str) -> bool:
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts


CANDIDATE_GENERATORS: dict[str, Callable[..., CandidateAction | None]] = {
    "original": original_candidate, "submit": submit_candidate, "run_public_check": run_public_check_candidate,
    "read_relevant_file": read_relevant_file_candidate, "skip_repeated_action": skip_repeated_candidate,
}


def generate_candidates(snapshot: CounterfactualSnapshot, manifest: EnvironmentManifest, snapshot_root: Path, names: list[str], max_candidates: int = 4) -> list[CandidateAction]:
    result = []
    for name in names:
        generator = CANDIDATE_GENERATORS.get(name)
        if generator is None: raise ValueError(f"unknown candidate generator: {name}")
        candidate = generator(snapshot, manifest, snapshot_root)
        if candidate is not None: result.append(candidate)
        if len(result) >= max_candidates: break
    return result
