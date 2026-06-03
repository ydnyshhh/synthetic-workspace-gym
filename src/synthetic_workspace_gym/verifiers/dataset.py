from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from synthetic_workspace_gym.prime.dataset import SyntheticWorkspacePrimeDataset


class SWGVerifiersDataset:
    def __init__(
        self,
        families: Sequence[str] = ("tabular", "script_repair", "pipeline", "retrieval_workspace"),
        scenarios: dict[str, Sequence[str]] | None = None,
        difficulties: Sequence[int] = (1, 2, 3, 4, 5),
        seeds: Sequence[int] = range(100),
        split: str | None = None,
        rows: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self.split = split
        if rows is not None:
            self._rows = [_normalize_row(row, split=split) for row in rows]
        else:
            prime_dataset = SyntheticWorkspacePrimeDataset(
                families=families,
                scenarios=scenarios,
                difficulties=difficulties,
                seeds=seeds,
                split=split,
            )
            self._rows = [_normalize_row(row, split=split) for row in prime_dataset]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from (dict(row) for row in self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def to_list(self) -> list[dict[str, Any]]:
        return list(self)


def load_from_prime_manifest(manifest_path: str | Path) -> SWGVerifiersDataset:
    manifest_path = Path(manifest_path)
    rows: list[dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        environment_path = row.get("environment_path")
        if environment_path is not None:
            row["environment_path"] = str((manifest_path.parent / str(environment_path)).resolve())
        rows.append(row)
    return SWGVerifiersDataset(rows=rows)


def _normalize_row(row: dict[str, Any], split: str | None = None) -> dict[str, Any]:
    scenario = row.get("scenario")
    task_scenario = scenario or "default"
    family = str(row.get("family", "script_repair"))
    difficulty = int(row.get("difficulty", 3))
    seed = int(row.get("seed", 0))
    task_id = str(row.get("task_id") or f"swg.{family}.{task_scenario}.d{difficulty}.s{seed}")
    env_id = str(row.get("env_id") or task_id)
    return {
        "task_id": task_id,
        "env_id": env_id,
        "family": family,
        "scenario": scenario,
        "difficulty": difficulty,
        "seed": seed,
        "split": row.get("split", split or "default"),
        "instruction": row.get("instruction"),
        "question": row.get("question") or row.get("instruction") or task_id,
        "environment_path": row.get("environment_path"),
        "metadata": dict(row.get("metadata", {}) or {}),
    }
