from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from synthetic_workspace_gym.schemas import Action, ActionType, EnvironmentManifest, EvaluatorResult, TrajectoryEvent
from synthetic_workspace_gym.utils.io import read_json, write_json, write_jsonl

from .schemas import CounterfactualSnapshot, stable_id

MUTATIONS = {ActionType.WRITE_FILE, ActionType.APPEND_FILE}
CHECKS = {ActionType.RUN_SHELL, ActionType.RUN_PYTHON}


@dataclass(slots=True)
class SnapshotContext:
    trajectory_id: str
    episode_id: str | None
    manifest: EnvironmentManifest
    workspace: Path
    environment_root: Path
    step_index: int
    remaining_steps: int
    elapsed_seconds: float
    action: Action | None
    previous_action: Action | None
    last_observation: str | None
    trajectory_prefix: list[dict[str, Any]]
    events_prefix: list[TrajectoryEvent]
    phase: str
    evaluator_result: EvaluatorResult | None = None
    action_success: bool | None = None


@dataclass(slots=True)
class SnapshotDecision:
    selected: bool
    labels: list[str] = field(default_factory=list)
    reason: str | None = None


class SnapshotPolicy(Protocol):
    def should_snapshot(self, context: SnapshotContext) -> SnapshotDecision: ...


@dataclass(slots=True)
class NamedSnapshotPolicy:
    name: str = "none"

    def should_snapshot(self, context: SnapshotContext) -> SnapshotDecision:
        action_type = context.action.action_type if context.action else None
        selected = (
            self.name == "every_step"
            or self.name == "selected"
            or (self.name == "writes" and action_type in MUTATIONS)
            or (self.name == "checks" and action_type in CHECKS)
            or (self.name == "submits" and action_type == ActionType.SUBMIT)
            or (self.name == "writes_checks_submit" and action_type in MUTATIONS | CHECKS | {ActionType.SUBMIT})
        )
        return SnapshotDecision(selected, [f"policy:{self.name}"] if selected else [])


@dataclass(slots=True)
class SnapshotCollector:
    output_root: Path
    policy: SnapshotPolicy = field(default_factory=NamedSnapshotPolicy)
    max_snapshots: int = 3
    evaluate_intermediate: bool = False
    snapshots: list[CounterfactualSnapshot] = field(default_factory=list, init=False)
    _keys: set[tuple[str, int, str]] = field(default_factory=set, init=False)

    def maybe_capture(self, context: SnapshotContext) -> CounterfactualSnapshot | None:
        decision = self.policy.should_snapshot(context)
        if not decision.selected or len(self.snapshots) >= self.max_snapshots:
            return None
        key = (context.trajectory_id, context.step_index, context.phase)
        if key in self._keys:
            return None
        self._keys.add(key)
        return self.capture(context, decision.labels)

    def capture(self, context: SnapshotContext, labels: list[str]) -> CounterfactualSnapshot:
        snapshot_id = stable_id("snapshot", context.trajectory_id, context.step_index, context.phase)
        root = self.output_root / context.trajectory_id / f"step-{context.step_index:04d}-{context.phase}"
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(context.workspace, root / "visible")
        hidden_source = context.environment_root / context.manifest.hidden_root
        if hidden_source.exists():
            shutil.copytree(hidden_source, root / "hidden")
        manifest_payload = context.manifest.to_dict()
        manifest_payload["workspace_root"] = "visible"
        manifest_payload["hidden_root"] = "hidden"
        write_json(root / "manifest.json", manifest_payload)
        write_jsonl(root / "trajectory_prefix.jsonl", context.trajectory_prefix)
        evaluator = context.evaluator_result
        action = _action_dict(context.action)
        previous = _action_dict(context.previous_action)
        state = {
            "step_index": context.step_index,
            "remaining_steps": context.remaining_steps,
            "elapsed_seconds": context.elapsed_seconds,
            "phase": context.phase,
            "original_action": action,
            "previous_action": previous,
            "last_observation": context.last_observation,
        }
        write_json(root / "branch_state.json", state)
        snapshot = CounterfactualSnapshot(
            snapshot_id=snapshot_id, trajectory_id=context.trajectory_id, episode_id=context.episode_id,
            env_id=context.manifest.env_id, family=context.manifest.family.value,
            scenario_id=context.manifest.metadata.get("scenario_id"), difficulty=context.manifest.difficulty,
            seed=context.manifest.seed, step_index=context.step_index, remaining_steps=context.remaining_steps,
            elapsed_seconds=context.elapsed_seconds, workspace_path="visible", manifest_path="manifest.json",
            branch_state_path="branch_state.json", trajectory_prefix=context.trajectory_prefix,
            swg_events_prefix=[event.to_dict() for event in context.events_prefix], original_action=action,
            previous_action=previous, last_observation=context.last_observation,
            evaluator_score=evaluator.score if evaluator else None,
            evaluator_subscores=evaluator.subscores if evaluator else {},
            evaluator_failure_labels=evaluator.failure_labels if evaluator else [], selector_labels=labels,
            metadata={"snapshot_root": str(root), "phase": context.phase, "action_success": context.action_success},
        )
        write_json(root / "snapshot.json", snapshot.to_dict())
        self.snapshots.append(snapshot)
        return snapshot


def load_snapshot(path: Path) -> CounterfactualSnapshot:
    return CounterfactualSnapshot.from_dict(read_json(path / "snapshot.json"))


def _action_dict(action: Action | None) -> dict[str, Any] | None:
    return None if action is None else {"tool": action.action_type.value, "args": dict(action.arguments)}
