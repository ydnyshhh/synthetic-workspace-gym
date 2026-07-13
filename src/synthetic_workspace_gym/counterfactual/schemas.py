from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, TypeVar

T = TypeVar("T", bound="CounterfactualRecord")


def stable_id(prefix: str, *parts: object) -> str:
    raw = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}-{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


@dataclass(slots=True)
class CounterfactualRecord:
    format_version: str = field(default="1.0", kw_only=True)
    RECORD_TYPE: ClassVar[str] = "record"

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_dict(cls: type[T], payload: dict[str, Any]) -> T:
        names = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in names})


def _required_action_valid(action: dict[str, Any] | None) -> bool:
    return (
        isinstance(action, dict)
        and isinstance(action.get("tool"), str)
        and bool(action["tool"])
        and isinstance(action.get("args", {}), dict)
    )


def _optional_action_valid(action: dict[str, Any] | None) -> bool:
    return action is None or _required_action_valid(action)


@dataclass(slots=True)
class CounterfactualSnapshot(CounterfactualRecord):
    snapshot_id: str
    trajectory_id: str
    episode_id: str | None
    env_id: str
    family: str
    scenario_id: str | None
    difficulty: int
    seed: int
    step_index: int
    remaining_steps: int
    elapsed_seconds: float | None
    workspace_path: str
    manifest_path: str
    branch_state_path: str
    trajectory_prefix: list[dict[str, Any]] = field(default_factory=list)
    swg_events_prefix: list[dict[str, Any]] = field(default_factory=list)
    original_action: dict[str, Any] | None = None
    previous_action: dict[str, Any] | None = None
    last_observation: str | None = None
    evaluator_score: float | None = None
    evaluator_subscores: dict[str, float] = field(default_factory=dict)
    evaluator_failure_labels: list[str] = field(default_factory=list)
    selector_labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_index < 0 or self.remaining_steps < 0:
            raise ValueError("step_index and remaining_steps must be non-negative")
        if not self.snapshot_id:
            raise ValueError("snapshot_id is required")
        if not _optional_action_valid(self.original_action) or not _optional_action_valid(self.previous_action):
            raise ValueError("actions must contain tool and args")


@dataclass(slots=True)
class CandidateAction(CounterfactualRecord):
    candidate_id: str
    branch_group_id: str
    snapshot_id: str
    candidate_type: str
    action: dict[str, Any] | None
    source: str
    rationale: str | None = None
    privileged: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.candidate_type != "skip_repeated_action" and not _required_action_valid(self.action):
            raise ValueError("candidate action must contain tool and args")


@dataclass(slots=True)
class BranchTask(CounterfactualRecord):
    task_id: str
    branch_group_id: str
    snapshot_id: str
    candidate_id: str
    mode: str
    environment_path: str
    prefix_messages: list[dict[str, Any]]
    forced_action: dict[str, Any] | None
    remaining_steps: int
    time_limit_seconds: int | None
    family: str
    scenario_id: str | None
    difficulty: int
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {"forced", "open"}:
            raise ValueError("mode must be 'forced' or 'open'")
        if self.remaining_steps <= 0:
            raise ValueError("remaining_steps must be positive")
        if self.mode == "forced" and not _required_action_valid(self.forced_action):
            raise ValueError("forced mode requires a valid forced_action")
        if self.mode == "open" and self.forced_action is not None:
            raise ValueError("open mode cannot have a forced_action")


@dataclass(slots=True)
class BranchOutcome(CounterfactualRecord):
    rollout_id: str
    task_id: str
    branch_group_id: str
    candidate_id: str
    snapshot_id: str
    model: str | None
    rollout_index: int
    final_reward: float
    final_score: float
    success: bool
    subscores: dict[str, float] = field(default_factory=dict)
    failure_labels: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    step_count: int = 0
    submitted: bool = False
    duration_seconds: float = 0.0
    final_workspace_path: str | None = None
    trajectory_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    rollout_seed: int | None = None
    sampling_seed: int | None = None
    pair_id: str | None = None


@dataclass(slots=True)
class BranchComparison(CounterfactualRecord):
    branch_group_id: str
    snapshot_id: str
    original_candidate_id: str
    candidate_statistics: dict[str, dict[str, float]]
    best_candidate_id: str
    original_mean_return: float
    best_mean_return: float
    counterfactual_delta: float
    decision_regret: float
    recoverable: bool
    original_action_optimal: bool
    confidence: float | None
    labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
