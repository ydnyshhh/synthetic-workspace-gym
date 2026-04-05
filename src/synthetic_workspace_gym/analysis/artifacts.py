from __future__ import annotations

import difflib
import hashlib
import shutil
from pathlib import Path

from synthetic_workspace_gym.schemas import EpisodeSummary, EnvironmentManifest, EvaluatorResult, TrajectoryEvent
from synthetic_workspace_gym.utils.io import write_json, write_jsonl, write_text
from synthetic_workspace_gym.utils.paths import file_sha256, list_relative_files


def snapshot_hashes(root: Path) -> dict[str, str]:
    return {relative_path: file_sha256(root / relative_path) for relative_path in list_relative_files(root)}


def snapshot_texts(root: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    for relative_path in list_relative_files(root):
        path = root / relative_path
        try:
            texts[relative_path] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            texts[relative_path] = "<binary>"
    return texts


def compute_workspace_digest(root: Path) -> str:
    return compute_digest_from_hashes(snapshot_hashes(root))


def compute_digest_from_hashes(file_hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative_path in sorted(file_hashes):
        digest.update(relative_path.encode("utf-8"))
        digest.update(file_hashes[relative_path].encode("utf-8"))
    return digest.hexdigest()


def changed_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    paths = sorted(set(before) | set(after))
    return [path for path in paths if before.get(path) != after.get(path)]


def build_unified_diff(before: dict[str, str], after: dict[str, str]) -> str:
    chunks: list[str] = []
    for relative_path in sorted(set(before) | set(after)):
        before_text = before.get(relative_path, "").splitlines(keepends=True)
        after_text = after.get(relative_path, "").splitlines(keepends=True)
        if before_text == after_text:
            continue
        chunks.extend(
            difflib.unified_diff(
                before_text,
                after_text,
                fromfile=f"a/{relative_path}",
                tofile=f"b/{relative_path}",
            )
        )
    return "".join(chunks)


def export_episode_artifacts(
    artifact_root: Path,
    *,
    manifest: EnvironmentManifest,
    trajectory: list[TrajectoryEvent],
    evaluator_result: EvaluatorResult,
    summary: EpisodeSummary,
    final_workspace: Path,
    final_diff: str,
) -> None:
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_json(artifact_root / "manifest.json", manifest.to_dict())
    write_jsonl(artifact_root / "trajectory.jsonl", [event.to_dict() for event in trajectory])
    write_json(artifact_root / "evaluator_result.json", evaluator_result.to_dict())
    write_json(artifact_root / "summary.json", summary.to_dict())
    write_text(artifact_root / "final_diff.txt", final_diff)
    destination = artifact_root / "final_workspace"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(final_workspace, destination)
