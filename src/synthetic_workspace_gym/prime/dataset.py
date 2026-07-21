from __future__ import annotations

import inspect
import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.generators.d5_profiles import d5_profile_metadata_for_family
from synthetic_workspace_gym.splits.manifest import read_split_manifest
from synthetic_workspace_gym.splits.policy import default_split_policy
from synthetic_workspace_gym.splits.schemas import VALID_SPLITS, normalize_split_name


class SyntheticWorkspacePrimeDataset:
    def __init__(
        self,
        families: Sequence[str] = (
            "tabular",
            "script_repair",
            "pipeline",
            "retrieval_workspace",
        ),
        difficulties: Sequence[int] = (1, 2, 3, 4, 5),
        scenarios: dict[str, Sequence[str]] | None = None,
        seeds: Sequence[int] = range(100),
        split: str | None = None,
        split_manifest_path: str | Path | None = None,
        include_splits: Sequence[str] | None = None,
        exclude_splits: Sequence[str] | None = None,
    ) -> None:
        self.split = (
            normalize_split_name(split)
            if split
            in {"train", "validation", "test", "heldout", "val", "valid", "dev", "eval"}
            else split
        )
        self._rows: list[dict[str, object]] | None = None
        self._resolve_scenarios_per_task = False
        if split_manifest_path is not None:
            manifest = read_split_manifest(split_manifest_path)
            rows = [assignment.to_dict() for assignment in manifest.assignments]
            self._rows = _filter_split_rows(
                rows,
                split=self.split,
                include_splits=include_splits,
                exclude_splits=exclude_splits,
            )
            self.families = tuple(sorted({str(row["family"]) for row in self._rows}))
            self.difficulties = tuple(
                sorted({int(row["difficulty"]) for row in self._rows})
            )
            self.seeds = tuple(sorted({int(row["seed"]) for row in self._rows}))
            self.scenarios = {}
            return

        if (
            self.split in VALID_SPLITS
            and scenarios is None
            and tuple(difficulties) == (1, 2, 3, 4, 5)
            and tuple(seeds) == tuple(range(100))
        ):
            spec = default_split_policy(families=families)[self.split]
            self.families = tuple(spec.families)
            self.difficulties = tuple(spec.difficulties)
            self.seeds = tuple(spec.seeds)
            self.scenarios = {
                family: tuple(values) for family, values in spec.scenarios.items()
            }
        else:
            self.families = tuple(str(family) for family in families)
            self.difficulties = tuple(int(difficulty) for difficulty in difficulties)
            self.seeds = tuple(int(seed) for seed in seeds)
            self._resolve_scenarios_per_task = scenarios is None
            self.scenarios = {
                family: tuple(values) for family, values in (scenarios or {}).items()
            }

    def __iter__(self) -> Iterator[dict[str, object]]:
        if self._rows is not None:
            yield from (dict(row) for row in self._rows)
            return
        split = self.split or "default"
        if self._resolve_scenarios_per_task:
            for family in self.families:
                generator = get_generator(family)
                for difficulty in self.difficulties:
                    for seed in self.seeds:
                        scenario_id = generator.resolve_scenario_id(
                            difficulty=difficulty,
                            seed=seed,
                        )
                        profile = d5_profile_metadata_for_family(family, difficulty, seed).get("profile")
                        profile_suffix = f".{profile}" if profile else ""
                        task_id = (
                            f"swg.{split}.{family}.{scenario_id}.d{difficulty}{profile_suffix}.s{seed}"
                            if self.split in VALID_SPLITS
                            else f"swg.{family}.{scenario_id}.d{difficulty}{profile_suffix}.s{seed}"
                        )
                        yield {
                            "family": family,
                            "scenario": scenario_id,
                            "difficulty": difficulty,
                            "seed": seed,
                            "split": split,
                            "task_id": task_id,
                            "environment_path": None,
                            "metadata": d5_profile_metadata_for_family(family, difficulty, seed),
                        }
            return
        for family in self.families:
            family_scenarios = self.scenarios.get(family, (None,))
            for scenario in family_scenarios:
                for difficulty in self.difficulties:
                    for seed in self.seeds:
                        scenario_id = str(scenario) if scenario is not None else None
                        task_scenario = scenario_id or "default"
                        profile = d5_profile_metadata_for_family(family, difficulty, seed).get("profile")
                        profile_suffix = f".{profile}" if profile else ""
                        task_id = (
                            f"swg.{split}.{family}.{task_scenario}.d{difficulty}{profile_suffix}.s{seed}"
                            if self.split in VALID_SPLITS
                            else f"swg.{family}.{task_scenario}.d{difficulty}{profile_suffix}.s{seed}"
                        )
                        yield {
                            "family": family,
                            "scenario": scenario_id,
                            "difficulty": difficulty,
                            "seed": seed,
                            "split": split,
                            "task_id": task_id,
                            "environment_path": None,
                            "metadata": d5_profile_metadata_for_family(family, difficulty, seed),
                        }

    def __len__(self) -> int:
        if self._rows is not None:
            return len(self._rows)
        if self._resolve_scenarios_per_task:
            return len(self.families) * len(self.difficulties) * len(self.seeds)
        total = 0
        for family in self.families:
            total += (
                len(self.scenarios.get(family, (None,)))
                * len(self.difficulties)
                * len(self.seeds)
            )
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
        parameters = list(inspect.signature(scenario_pool).parameters.values())
    except (TypeError, ValueError):
        parameters = []
    rng = random.Random("prime-dataset")
    if not parameters:
        candidates.append(())
    elif parameters[0].name in {"rng", "random", "random_state"}:
        candidates.append((rng, spec))
    else:
        candidates.append((spec,))
    candidates.extend(((spec,), (rng, spec), ()))

    for args in candidates:
        try:
            pool = scenario_pool(*args)
        except TypeError:
            continue
        if isinstance(pool, list):
            return pool
    return []


def _filter_split_rows(
    rows: Sequence[dict[str, object]],
    *,
    split: str | None,
    include_splits: Sequence[str] | None,
    exclude_splits: Sequence[str] | None,
) -> list[dict[str, object]]:
    include = {normalize_split_name(item) for item in include_splits or []}
    exclude = {normalize_split_name(item) for item in exclude_splits or []}
    normalized_split = (
        normalize_split_name(split)
        if split
        in {"train", "validation", "test", "heldout", "val", "valid", "dev", "eval"}
        else split
    )
    filtered: list[dict[str, object]] = []
    for row in rows:
        row_split = (
            normalize_split_name(str(row.get("split")))
            if row.get("split") is not None
            else None
        )
        if normalized_split is not None and row_split != normalized_split:
            continue
        if include and row_split not in include:
            continue
        if exclude and row_split in exclude:
            continue
        payload = dict(row)
        payload["split"] = row_split
        filtered.append(payload)
    return filtered
