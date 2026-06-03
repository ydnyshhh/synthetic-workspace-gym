from __future__ import annotations

import random
from typing import Any

from synthetic_workspace_gym.schemas import utc_timestamp

from .schemas import SplitAssignment, SplitManifest, SplitSpec, normalize_split_name


def build_split_assignments(
    split_specs: dict[str, SplitSpec],
    *,
    max_per_split: dict[str, int] | None = None,
    shuffle: bool = False,
    shuffle_seed: int = 0,
) -> list[SplitAssignment]:
    assignments: list[SplitAssignment] = []
    max_per_split = {str(key): int(value) for key, value in (max_per_split or {}).items() if value is not None}
    for split_name in ("train", "validation", "test", "heldout"):
        spec = split_specs.get(split_name)
        if spec is None:
            continue
        split_assignments: list[SplitAssignment] = []
        for family in spec.families:
            for scenario in spec.scenarios.get(family, [None]):
                for difficulty in spec.difficulties:
                    for seed in spec.seeds:
                        task_scenario = scenario or "default"
                        split_assignments.append(
                            SplitAssignment(
                                split=spec.name,
                                family=family,
                                scenario=scenario,
                                difficulty=difficulty,
                                seed=seed,
                                task_id=f"swg.{spec.name}.{family}.{task_scenario}.d{difficulty}.s{seed}",
                                metadata={"split_spec": spec.name},
                            )
                        )
        if shuffle:
            random.Random(f"{shuffle_seed}:{split_name}").shuffle(split_assignments)
        limit = max_per_split.get(split_name, spec.count)
        if limit is not None:
            split_assignments = split_assignments[: int(limit)]
        assignments.extend(split_assignments)
    return assignments


def build_split_manifest(
    name: str,
    split_specs: dict[str, SplitSpec],
    *,
    max_per_split: dict[str, int] | None = None,
    shuffle: bool = False,
    shuffle_seed: int = 0,
    metadata: dict[str, Any] | None = None,
) -> SplitManifest:
    normalized_specs = {
        normalize_split_name(key) or key: spec
        for key, spec in split_specs.items()
    }
    assignments = build_split_assignments(
        normalized_specs,
        max_per_split=max_per_split,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
    )
    return SplitManifest(
        name=name,
        version="v1",
        created_at=utc_timestamp(),
        split_specs=normalized_specs,
        assignments=assignments,
        metadata={
            "shuffle": shuffle,
            "shuffle_seed": shuffle_seed,
            **(metadata or {}),
        },
    )
