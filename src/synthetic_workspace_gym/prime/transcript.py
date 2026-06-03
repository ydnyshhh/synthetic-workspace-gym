from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence, TypedDict

from synthetic_workspace_gym.schemas import utc_timestamp
from synthetic_workspace_gym.utils.io import to_json_compatible


class PrimeToolCall(TypedDict):
    tool: str
    args: dict[str, Any]


class PrimeToolObservation(TypedDict):
    observation: str
    done: bool
    reward: float
    info: dict[str, Any]


class PrimeModelMessage(TypedDict):
    role: str
    content: str


class PrimeTranscriptEvent(TypedDict):
    event_type: str
    step_index: int | None
    timestamp: str
    payload: dict[str, Any]


def make_event(event_type: str, payload: dict[str, Any], step_index: int | None = None) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "step_index": step_index,
        "timestamp": utc_timestamp(),
        "payload": payload,
    }


def write_transcript_jsonl(path: str | Path, events: Sequence[dict[str, Any]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(to_json_compatible(event), sort_keys=True))
            handle.write("\n")
    return output_path


def read_transcript_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    return [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
