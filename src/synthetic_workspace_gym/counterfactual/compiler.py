from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath

from synthetic_workspace_gym.runtime.environment import load_environment
from synthetic_workspace_gym.schemas import EnvironmentManifest
from synthetic_workspace_gym.utils.io import read_json, write_json, write_jsonl

from .schemas import BranchTask, CandidateAction, CounterfactualSnapshot, stable_id

CONTINUATION_INSTRUCTION = "Continue solving the task from the current workspace state. The workspace already includes all earlier edits. Do not repeat completed work unless verification requires it. Use the remaining tool budget efficiently. Submit when the required artifact or repair is complete."


def compile_branch(snapshot: CounterfactualSnapshot, candidate: CandidateAction, snapshot_root: Path, output_root: Path, mode: str = "forced") -> BranchTask:
    manifest = EnvironmentManifest.from_dict(read_json(snapshot_root / snapshot.manifest_path))
    if candidate.snapshot_id != snapshot.snapshot_id: raise ValueError("candidate and snapshot do not match")
    task_id = stable_id("swg-cf", candidate.branch_group_id, candidate.candidate_id, mode)
    env_root = output_root / "environments" / task_id
    if env_root.exists(): shutil.rmtree(env_root)
    shutil.copytree(snapshot_root / snapshot.workspace_path, env_root / "visible")
    shutil.copytree(snapshot_root / manifest.hidden_root, env_root / "hidden")
    payload = manifest.to_dict(); payload["workspace_root"] = "visible"; payload["hidden_root"] = "hidden"
    write_json(env_root / "manifest.json", payload)
    forced = candidate.action if mode == "forced" else None
    _validate_action(forced, manifest, env_root)
    messages = list(snapshot.trajectory_prefix) + [{"role": "user", "content": CONTINUATION_INSTRUCTION}]
    task = BranchTask(task_id, candidate.branch_group_id, snapshot.snapshot_id, candidate.candidate_id, mode,
        env_root.relative_to(output_root).as_posix(), messages, forced, snapshot.remaining_steps, _remaining_time(snapshot, manifest), snapshot.family,
        snapshot.scenario_id, snapshot.difficulty, snapshot.seed,
        {"candidate_type": candidate.candidate_type, "privileged": candidate.privileged, "selector_labels": snapshot.selector_labels})
    write_json(env_root / "branch.json", task.to_dict()); write_json(env_root / "prefix_messages.json", messages)
    load_environment(env_root)
    return task


def compile_pack(items: list[tuple[CounterfactualSnapshot, CandidateAction, Path]], output_root: Path, mode: str = "forced") -> list[BranchTask]:
    tasks = [compile_branch(snapshot, candidate, root, output_root, mode) for snapshot, candidate, root in items]
    write_jsonl(output_root / "manifest.jsonl", [task.to_dict() for task in tasks])
    write_json(output_root / "metadata.json", {"format_version": "1.0", "task_count": len(tasks), "mode": mode})
    return tasks


def _remaining_time(snapshot: CounterfactualSnapshot, manifest: EnvironmentManifest) -> int:
    return max(1, int(manifest.time_limit_seconds - (snapshot.elapsed_seconds or 0)))


def _validate_action(action: dict | None, manifest: EnvironmentManifest, root: Path) -> None:
    if action is None: return
    tool = str(action.get("tool", ""))
    if tool not in manifest.tool_permissions.enabled_tools(): raise ValueError(f"forced tool is not enabled: {tool}")
    args = action.get("args", {})
    if tool in {"read_file", "write_file", "append_file", "list_directory"}:
        path = PurePosixPath(str(args.get("path", "")))
        if path.is_absolute() or ".." in path.parts: raise ValueError("candidate paths must be workspace-relative")
