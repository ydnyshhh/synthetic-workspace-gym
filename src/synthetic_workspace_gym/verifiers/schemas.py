from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class VerifiersTask:
    task_id: str
    env_id: str
    family: str
    scenario: str | None
    difficulty: int
    seed: int
    instruction: str | None = None
    environment_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "env_id": self.env_id,
            "family": self.family,
            "scenario": self.scenario,
            "difficulty": self.difficulty,
            "seed": self.seed,
            "instruction": self.instruction,
            "environment_path": self.environment_path,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class VerifiersToolCall:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": dict(self.args)}


@dataclass(slots=True)
class VerifiersReward:
    reward: float
    success: bool
    score: float
    subscores: dict[str, float] = field(default_factory=dict)
    failure_labels: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "success": self.success,
            "score": self.score,
            "subscores": dict(self.subscores),
            "failure_labels": list(self.failure_labels),
            "diagnostics": dict(self.diagnostics),
        }
