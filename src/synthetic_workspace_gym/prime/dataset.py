from __future__ import annotations

import inspect
import random
from collections.abc import Iterator, Sequence
from typing import Any

from synthetic_workspace_gym.generators.registry import get_generator


class SyntheticWorkspacePrimeDataset:
    def __init__(
        self,
        families: Sequence[str] = ("tabular", "script_repair", "pipeline", "retrieval_workspace"),
        difficulties: Sequence[int] = (1, 2, 3, 4, 5),
        scenarios: dict[str, Sequence[str]] | None = None,
        seeds: Sequence[int] = range(100),
        split: str | None = None,
    ) -> None:
        self.families = tuple(str(family) for family in families)
        self.difficulties = tuple(int(difficulty) for difficulty in difficulties)
        self.seeds = tuple(int(seed) for seed in seeds)
        self.split = split
        self.scenarios = {
            family: tuple(values)
            for family, values in (scenarios or self._discover_scenarios()).items()
        }

    def __iter__(self) -> Iterator[dict[str, object]]:
        split = self.split or "default"
        for family in self.families:
            family_scenarios = self.scenarios.get(family, (None,))
            for scenario in family_scenarios:
                for difficulty in self.difficulties:
                    for seed in self.seeds:
                        scenario_id = str(scenario) if scenario is not None else None
                        task_scenario = scenario_id or "default"
                        yield {
                            "family": family,
                            "scenario": scenario_id,
                            "difficulty": difficulty,
                            "seed": seed,
                            "split": split,
                            "task_id": f"swg.{family}.{task_scenario}.d{difficulty}.s{seed}",
                        }

    def __len__(self) -> int:
        total = 0
        for family in self.families:
            total += len(self.scenarios.get(family, (None,))) * len(self.difficulties) * len(self.seeds)
        return total

    def to_list(self) -> list[dict[str, object]]:
        return list(self)

    def _discover_scenarios(self) -> dict[str, tuple[str | None, ...]]:
        discovered: dict[str, tuple[str | None, ...]] = {}
        for family in self.families:
            discovered[family] = tuple(_discover_family_scenarios(family))
        return discovered


def _discover_family_scenarios(family: str) -> list[str | None]:
    try:
        generator = get_generator(family)
        spec = generator.sample_spec(difficulty=1, seed=1)
    except Exception:
        return [None]

    pool = _call_scenario_pool(generator, spec)
    scenario_ids = [
        str(scenario["scenario_id"])
        for scenario in pool
        if isinstance(scenario, dict) and scenario.get("scenario_id") is not None
    ]
    return scenario_ids or [None]


def _call_scenario_pool(generator: Any, spec: Any) -> list[dict[str, object]]:
    scenario_pool = getattr(generator, "scenario_pool", None)
    if scenario_pool is None:
        return []

    candidates: list[tuple[Any, ...]] = []
    try:
        parameter_count = len(inspect.signature(scenario_pool).parameters)
    except (TypeError, ValueError):
        parameter_count = -1
    rng = random.Random("prime-dataset")
    if parameter_count == 0:
        candidates.append(())
    elif parameter_count == 1:
        candidates.append((spec,))
    elif parameter_count == 2:
        candidates.append((rng, spec))
    candidates.extend(((spec,), (rng, spec), ()))

    for args in candidates:
        try:
            pool = scenario_pool(*args)
        except TypeError:
            continue
        if isinstance(pool, list):
            return pool
    return []
