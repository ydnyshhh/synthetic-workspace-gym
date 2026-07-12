from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synthetic_workspace_gym.agents.base import BaseAgent
from synthetic_workspace_gym.analysis.artifacts import build_unified_diff, snapshot_texts
from synthetic_workspace_gym.evaluators.registry import get_evaluator
from synthetic_workspace_gym.runtime.environment import load_environment
from synthetic_workspace_gym.runtime.tools import WorkspaceToolExecutor
from synthetic_workspace_gym.schemas import Action, ActionType, ToolObservation, ToolState
from synthetic_workspace_gym.utils.io import write_json, write_jsonl

from .schemas import BranchOutcome, BranchTask, stable_id


@dataclass(slots=True)
class ReplayResult:
    outcome: BranchOutcome
    messages: list[dict[str, Any]]


def replay_branch(task: BranchTask, agent: BaseAgent, output_root: Path, rollout_index: int = 0) -> ReplayResult:
    environment = load_environment(Path(task.environment_path))
    rollout_id = stable_id("rollout", task.task_id, rollout_index, agent.name)
    root = (output_root / "rollouts" / rollout_id).resolve()
    workspace = (root / "active_workspace").resolve()
    if root.exists(): shutil.rmtree(root)
    shutil.copytree(environment.visible_root, workspace)
    executor = WorkspaceToolExecutor(workspace, environment.manifest.tool_permissions)
    initial = snapshot_texts(workspace); started = time.perf_counter()
    messages = list(task.prefix_messages[:-1])
    trajectory: list[dict[str, Any]] = []
    observation: ToolObservation | dict[str, object] = {"instruction": environment.manifest.instruction, "branch": True}
    steps = 0; submitted = False
    if task.mode == "forced":
        forced = _to_action(task.forced_action)
        observation = executor.execute(forced, remaining_time_seconds=task.time_limit_seconds)
        steps += 1
        messages.extend([{"role": "assistant", "tool_call": task.forced_action, "metadata": {"forced": True}}, {"role": "tool", "name": forced.action_type.value, "content": _observation_text(observation), "metadata": {"forced": True}}])
        trajectory.append(_event(forced, observation, True))
        submitted = forced.action_type == ActionType.SUBMIT
    messages.append(task.prefix_messages[-1])
    agent.reset(environment.manifest, {"instruction": environment.manifest.instruction, "branch": True, "prefix_messages": messages})
    agent.restore_context(messages)
    while not submitted and steps < task.remaining_steps and time.perf_counter() - started < (task.time_limit_seconds or 60):
        state = ToolState(steps, task.remaining_steps - steps, environment.manifest.tool_permissions.enabled_tools(), submitted=submitted)
        action = agent.act(observation, state)
        observation = executor.execute(action, remaining_time_seconds=(task.time_limit_seconds or 60) - (time.perf_counter() - started))
        steps += 1; submitted = action.action_type == ActionType.SUBMIT
        messages.extend([{"role": "assistant", "tool_call": {"tool": action.action_type.value, "args": action.arguments}}, {"role": "tool", "name": action.action_type.value, "content": _observation_text(observation)}])
        trajectory.append(_event(action, observation, False))
    evaluator = get_evaluator(environment.manifest.family, evaluator_entrypoint=environment.manifest.evaluator_entrypoint)
    result = evaluator.evaluate(workspace, environment.manifest, environment.hidden_root)
    final_workspace = root / "final_workspace"; shutil.copytree(workspace, final_workspace)
    write_jsonl(root / "trajectory.jsonl", trajectory); write_json(root / "evaluator_result.json", result.to_dict())
    (root / "final_diff.txt").write_text(build_unified_diff(initial, snapshot_texts(workspace)), encoding="utf-8")
    outcome = BranchOutcome(rollout_id, task.task_id, task.branch_group_id, task.candidate_id, task.snapshot_id,
        agent.name, rollout_index, result.score, result.score, result.success, result.subscores, result.failure_labels,
        result.diagnostics, steps, submitted, time.perf_counter() - started, str(final_workspace), str(root / "trajectory.jsonl"),
        {**task.metadata, "mode": task.mode, "forced_action_steps": 1 if task.mode == "forced" else 0})
    write_json(root / "branch_outcome.json", outcome.to_dict())
    return ReplayResult(outcome, messages)


def _to_action(value: dict[str, Any] | None) -> Action:
    if value is None: raise ValueError("forced branch has no action")
    return Action(ActionType(value["tool"]), dict(value.get("args", {})))


def _observation_text(value: ToolObservation) -> str:
    return "\n".join(x for x in (value.message, value.content or "", value.stdout, value.stderr) if x)


def _event(action: Action, observation: ToolObservation, forced: bool) -> dict[str, Any]:
    return {"action": {"tool": action.action_type.value, "args": action.arguments}, "observation": observation.to_dict(), "forced": forced}
