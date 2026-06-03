from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from synthetic_workspace_gym.utils.io import read_json, write_json

from .schemas import SplitAssignment, SplitManifest


def write_split_manifest(path: str | Path, manifest: SplitManifest) -> Path:
    target = Path(path)
    write_json(target, manifest.to_dict())
    return target


def read_split_manifest(path: str | Path) -> SplitManifest:
    return SplitManifest.from_dict(read_json(Path(path)))


def write_split_jsonl(path: str | Path, assignments: Sequence[SplitAssignment]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for assignment in assignments:
            handle.write(json.dumps(assignment.to_dict(), sort_keys=True))
            handle.write("\n")
    return target


def read_split_jsonl(path: str | Path) -> list[SplitAssignment]:
    return [
        SplitAssignment.from_dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
