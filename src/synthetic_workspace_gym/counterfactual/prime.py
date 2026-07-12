from __future__ import annotations

from pathlib import Path
from typing import Any

from synthetic_workspace_gym.counterfactual.runner import read_branch_manifest
from synthetic_workspace_gym.prime.env import SyntheticWorkspacePrimeEnv


class SyntheticWorkspacePrimeBranchEnv(SyntheticWorkspacePrimeEnv):
    """Prime-facing branch environment with packaged context and forced-action reset."""

    def __init__(self, branch_manifest_path: str | Path, branch_task_id: str | None = None,
                 branch_task_index: int = 0, branch_mode: str | None = None, **kwargs: Any) -> None:
        tasks = read_branch_manifest(Path(branch_manifest_path))
        if branch_task_id is None:
            task = tasks[branch_task_index]
        else:
            matches = [row for row in tasks if row.task_id == branch_task_id]
            if len(matches) != 1:
                raise ValueError(f"expected exactly one branch task {branch_task_id!r}, found {len(matches)}")
            task = matches[0]
        self.branch_task = task
        self.branch_mode = branch_mode or task.mode
        if self.branch_mode not in {"forced", "open"}:
            raise ValueError("branch_mode must be 'forced' or 'open'")
        kwargs.pop("family", None)
        kwargs.pop("workspace_root", None)
        super().__init__(family=task.family, scenario=task.scenario_id, difficulty=task.difficulty, seed=task.seed,
                         workspace_root=task.environment_path, max_steps=task.remaining_steps, **kwargs)

    def reset(self) -> dict[str, object]:
        observation = super().reset()
        messages = [dict(message) for message in self.branch_task.prefix_messages]
        forced_result = None
        if self.branch_mode == "forced":
            action = self.branch_task.forced_action
            if action is None:
                raise ValueError("forced branch task has no forced action")
            forced_result = self.step(action)
            messages.extend([
                {"role": "assistant", "content": str(action), "metadata": {"forced": True}},
                {"role": "tool", "content": str(forced_result["observation"]), "metadata": {"forced": True}},
            ])
        observation["messages"] = messages
        observation["prefix_messages"] = list(self.branch_task.prefix_messages)
        observation["forced_action"] = self.branch_task.forced_action if self.branch_mode == "forced" else None
        observation["forced_action_result"] = forced_result
        observation["metadata"] = {**dict(observation.get("metadata", {}) or {}),
            **self.branch_task.metadata, "counterfactual": True, "branch_task_id": self.branch_task.task_id,
            "branch_group_id": self.branch_task.branch_group_id, "snapshot_id": self.branch_task.snapshot_id,
            "candidate_id": self.branch_task.candidate_id, "branch_mode": self.branch_mode}
        return observation