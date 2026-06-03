from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from synthetic_workspace_gym.schemas import utc_timestamp

SplitName = Literal["train", "validation", "test", "heldout"]
VALID_SPLITS: set[str] = {"train", "validation", "test", "heldout"}
SPLIT_ALIASES = {
    "val": "validation",
    "valid": "validation",
    "dev": "validation",
    "eval": "validation",
}


@dataclass(slots=True)
class SplitSpec:
    name: SplitName
    families: list[str]
    scenarios: dict[str, list[str]]
    difficulties: list[int]
    seeds: list[int]
    count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = normalize_split_name(self.name)  # type: ignore[assignment]
        if self.name is None:
            raise ValueError("Invalid split name")
        self.families = [str(item) for item in self.families]
        self.scenarios = {str(key): [str(item) for item in values] for key, values in self.scenarios.items()}
        self.difficulties = [int(item) for item in self.difficulties]
        self.seeds = [int(item) for item in self.seeds]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "families": list(self.families),
            "scenarios": {family: list(values) for family, values in self.scenarios.items()},
            "difficulties": list(self.difficulties),
            "seeds": list(self.seeds),
            "count": self.count,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SplitSpec":
        return cls(
            name=payload["name"],
            families=list(payload.get("families", [])),
            scenarios={str(key): list(values) for key, values in dict(payload.get("scenarios", {})).items()},
            difficulties=list(payload.get("difficulties", [])),
            seeds=list(payload.get("seeds", [])),
            count=payload.get("count"),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass(slots=True)
class SplitAssignment:
    split: SplitName
    family: str
    scenario: str | None
    difficulty: int
    seed: int
    task_id: str
    env_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.split = normalize_split_name(self.split)  # type: ignore[assignment]
        if self.split is None:
            raise ValueError("Invalid split name")
        self.family = str(self.family)
        self.scenario = str(self.scenario) if self.scenario is not None else None
        self.difficulty = int(self.difficulty)
        self.seed = int(self.seed)
        self.task_id = str(self.task_id)
        self.env_id = str(self.env_id) if self.env_id is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "family": self.family,
            "scenario": self.scenario,
            "difficulty": self.difficulty,
            "seed": self.seed,
            "task_id": self.task_id,
            "env_id": self.env_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SplitAssignment":
        return cls(
            split=payload["split"],
            family=payload["family"],
            scenario=payload.get("scenario"),
            difficulty=payload["difficulty"],
            seed=payload["seed"],
            task_id=payload["task_id"],
            env_id=payload.get("env_id"),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass(slots=True)
class SplitManifest:
    name: str
    version: str
    created_at: str
    split_specs: dict[str, SplitSpec]
    assignments: list[SplitAssignment]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "split_specs": {name: spec.to_dict() for name, spec in self.split_specs.items()},
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SplitManifest":
        return cls(
            name=str(payload["name"]),
            version=str(payload.get("version", "v1")),
            created_at=str(payload.get("created_at") or utc_timestamp()),
            split_specs={
                str(name): SplitSpec.from_dict(spec)
                for name, spec in dict(payload.get("split_specs", {})).items()
            },
            assignments=[
                SplitAssignment.from_dict(item)
                for item in list(payload.get("assignments", []))
            ],
            metadata=dict(payload.get("metadata", {}) or {}),
        )


def normalize_split_name(value: str | None) -> SplitName | None:
    if value is None:
        return None
    normalized = SPLIT_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())
    if normalized not in VALID_SPLITS:
        raise ValueError(f"Invalid split name: {value}")
    return normalized  # type: ignore[return-value]
