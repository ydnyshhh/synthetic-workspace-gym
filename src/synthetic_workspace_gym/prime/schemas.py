from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class PrimeTaskSpec(TypedDict):
    family: str
    scenario: str | None
    difficulty: int
    seed: int
    split: str | None
    task_id: str


class PrimeStepResult(TypedDict):
    observation: str
    done: bool
    reward: float
    info: dict[str, Any]


class PrimeRewardPayload(TypedDict):
    reward: float
    success: bool
    score: float
    subscores: dict[str, float]
    failure_labels: list[str]
    diagnostics: dict[str, Any]
    runtime_seconds: float | None
    env_id: NotRequired[str]
