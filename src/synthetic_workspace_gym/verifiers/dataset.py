from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from synthetic_workspace_gym.prime.dataset import SyntheticWorkspacePrimeDataset
from synthetic_workspace_gym.splits.manifest import read_split_manifest


class SWGVerifiersDataset:
    def __init__(
        self,
        families: Sequence[str] = ("tabular", "script_repair", "pipeline", "retrieval_workspace"),
        scenarios: dict[str, Sequence[str]] | None = None,
        difficulties: Sequence[int] = (1, 2, 3, 4, 5),
        seeds: Sequence[int] = range(100),
        split: str | None = None,
        rows: Sequence[dict[str, Any]] | None = None,
        split_manifest_path: str | Path | None = None,
        include_splits: Sequence[str] | None = None,
        exclude_splits: Sequence[str] | None = None,
    ) -> None:
        self.split = split
        if split_manifest_path is not None:
            manifest = read_split_manifest(split_manifest_path)
            self._rows = [_normalize_row(row, split=split) for row in _filter_rows(
                [assignment.to_dict() for assignment in manifest.assignments],
                split=split,
                include_splits=include_splits,
                exclude_splits=exclude_splits,
            )]
        elif rows is not None:
            self._rows = [_normalize_row(row, split=split) for row in rows]
        else:
            official_split = split if split in {"train", "validation", "test", "heldout", "val", "valid", "dev"} else None
            prime_dataset = SyntheticWorkspacePrimeDataset(
                families=families,
                scenarios=scenarios,
                difficulties=difficulties,
                seeds=seeds,
                split=official_split,
                include_splits=include_splits,
                exclude_splits=exclude_splits,
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
        row["split"] = row.get("split")
        row["task_id"] = row.get("task_id")
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
        "split": _row_split(row, split),
        "instruction": row.get("instruction"),
        "question": row.get("question") or row.get("instruction") or task_id,
        "environment_path": row.get("environment_path"),
        "metadata": dict(row.get("metadata", {}) or {}),
    }


def _row_split(row: dict[str, Any], split: str | None) -> str:
    value = row.get("split")
    if split is not None and (value is None or value == "default"):
        return split
    return str(value or split or "default")


def _filter_rows(
    rows: Sequence[dict[str, Any]],
    *,
    split: str | None,
    include_splits: Sequence[str] | None,
    exclude_splits: Sequence[str] | None,
) -> list[dict[str, Any]]:
    include = {str(item) for item in include_splits or []}
    exclude = {str(item) for item in exclude_splits or []}
    filtered = []
    for row in rows:
        row_split = str(row.get("split")) if row.get("split") is not None else None
        if split is not None and row_split != split:
            continue
        if include and row_split not in include:
            continue
        if exclude and row_split in exclude:
            continue
        filtered.append(dict(row))
    return filtered
