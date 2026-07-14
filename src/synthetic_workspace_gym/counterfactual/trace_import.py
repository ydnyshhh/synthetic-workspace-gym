from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from synthetic_workspace_gym.agents.base import BaseAgent
from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.runtime.environment import load_environment
from synthetic_workspace_gym.runtime.runner import EpisodeRunner
from synthetic_workspace_gym.schemas import Action, ActionType, ToolState

from .snapshots import NamedSnapshotPolicy, SnapshotCollector


_TASK_FIELD = re.compile(r"^- ([a-z_]+):\s*(.+)$", re.MULTILINE)
_FAILURE_TEXT = re.compile(
    r"traceback|error:|not found|no such file|failed|exception|stderr:\s*\S",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ImportedRoot:
    example_id: int
    trace_id: str
    task_id: str
    reward: float
    failure_types: list[str]
    action_count: int
    snapshot_count: int
    replay_reward: float
    artifact_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "reward": self.reward,
            "failure_types": self.failure_types,
            "action_count": self.action_count,
            "snapshot_count": self.snapshot_count,
            "replay_reward": self.replay_reward,
            "artifact_root": self.artifact_root,
        }


class TraceReplayAgent(BaseAgent):
    name = "qwen-trace-replay"

    def __init__(self, actions: Iterable[Action]) -> None:
        super().__init__()
        self._source = [_safe_replay_action(action) for action in actions]
        self._pending: list[Action] = []

    def reset(self, manifest, initial_observation) -> None:
        super().reset(manifest, initial_observation)
        self._pending = list(self._source)

    def act(self, observation, tool_state: ToolState) -> Action:
        if not self._pending:
            return self.set_last_action(Action(ActionType.SUBMIT, {"path_or_answer": "done"}))
        return self.set_last_action(self._pending.pop(0))

def _safe_replay_action(action: Action) -> Action:
    if action.action_type not in {ActionType.READ_FILE, ActionType.WRITE_FILE, ActionType.APPEND_FILE, ActionType.LIST_DIRECTORY}:
        return action
    arguments = dict(action.arguments)
    raw = str(arguments.get("path", "")).replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        arguments["path"] = "__rejected_invalid_path__"
        return Action(action.action_type, arguments)
    return action



def load_prime_samples(path: Path) -> list[dict[str, Any]]:
    files = sorted(path.glob("samples-page-*.json")) if path.is_dir() else [path]
    samples: list[dict[str, Any]] = []
    for item in files:
        payload = json.loads(item.read_text(encoding="utf-8-sig"))
        samples.extend(dict(row) for row in payload.get("samples", []))
    return samples


def parse_task_metadata(sample: dict[str, Any]) -> dict[str, str]:
    prompt = sample.get("prompt") or []
    user = next((str(row.get("content", "")) for row in prompt if row.get("role") == "user"), "")
    return {key: value.strip() for key, value in _TASK_FIELD.findall(user)}


def extract_actions(sample: dict[str, Any]) -> list[Action]:
    actions: list[Action] = []
    for message in sample.get("completion") or []:
        if message.get("role") != "assistant":
            continue
        for raw in message.get("tool_calls") or []:
            call = json.loads(raw) if isinstance(raw, str) else raw
            name = call.get("name") or call.get("tool")
            arguments = call.get("arguments", call.get("args", {}))
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            try:
                actions.append(Action(ActionType(str(name)), dict(arguments or {})))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
    return actions


def classify_root(sample: dict[str, Any], actions: list[Action], *, max_turns: int = 25) -> list[str]:
    reward = float(sample.get("reward", sample.get("swg_reward", 0.0)) or 0.0)
    info = sample.get("info") or {}
    num_turns = int(float((info.get("metrics") or {}).get("num_turns", len(actions)) or 0))
    labels: list[str] = []
    if reward >= 0.95 and num_turns >= 10:
        labels.append("successful_but_inefficient")
    elif 0.0 < reward < 0.95:
        labels.append("partial_success")
    if reward <= 0.0:
        labels.append("failed_hidden_evaluation")
    if num_turns >= max_turns or bool(info.get("is_truncated")):
        labels.append("max_turn_termination")
    signatures = [json.dumps({"tool": a.action_type.value, "args": a.arguments}, sort_keys=True) for a in actions]
    if any(signatures.count(signature) >= 3 for signature in set(signatures)):
        labels.append("tool_loop_failure")
    submit_index = next((index for index, action in enumerate(actions, 1) if action.action_type == ActionType.SUBMIT), None)
    if submit_index is not None and submit_index <= 3 and reward < 0.95:
        labels.append("premature_submit")
    if _has_failed_public_check(sample):
        labels.append("failed_public_check")
    return labels or ["unclassified"]


def import_prime_roots(
    samples_path: Path,
    example_ids: list[int],
    output_dir: Path,
    *,
    evaluation_id: str | None = None,
    source_model: str = "Qwen/Qwen3.5-0.8B",
    max_turns: int = 25,
) -> list[ImportedRoot]:
    by_id = {int(row["example_id"]): row for row in load_prime_samples(samples_path)}
    missing = sorted(set(example_ids) - set(by_id))
    if missing:
        raise ValueError(f"Prime sample example IDs not found: {missing}")
    results: list[ImportedRoot] = []
    for example_id in example_ids:
        sample = by_id[example_id]
        task = parse_task_metadata(sample)
        required = {"task_id", "family", "scenario", "difficulty", "seed"}
        if missing_fields := sorted(required - set(task)):
            raise ValueError(f"sample {example_id} is missing task metadata: {missing_fields}")
        actions = extract_actions(sample)
        if not actions:
            raise ValueError(f"sample {example_id} contains no replayable tool actions")
        root_dir = output_dir / f"example-{example_id:04d}"
        generator = get_generator(task["family"])
        spec = generator.sample_spec(
            difficulty=int(task["difficulty"]), seed=int(task["seed"]),
            scenario_id=task["scenario"], split=task.get("split", "validation"),
            task_id=task["task_id"], max_steps=max(len(actions), max_turns), time_limit_seconds=450,
        )
        bundle = generator.generate_instance(spec, root_dir / "generated", validate=False)
        failure_types = classify_root(sample, actions, max_turns=max_turns)
        future_edits = [
            {"step_index": index, "tool": action.action_type.value, "args": dict(action.arguments)}
            for index, action in enumerate(actions)
            if action.action_type in {ActionType.WRITE_FILE, ActionType.APPEND_FILE}
        ]
        base_metadata = {
            "source_model": source_model,
            "source_evaluation_id": evaluation_id,
            "source_example_id": example_id,
            "source_trace_id": str(sample.get("trace_id", "")),
            "root_reward": float(sample.get("reward", sample.get("swg_reward", 0.0)) or 0.0),
            "root_failure_types": failure_types,
            "root_future_edits": future_edits,
        }
        collector = SnapshotCollector(
            root_dir / "snapshots", NamedSnapshotPolicy("every_step"),
            max_snapshots=max(2, len(actions) * 2), evaluate_intermediate=True,
            max_signal_snapshots=max(2, len(actions)), base_metadata=base_metadata,
        )
        summary = EpisodeRunner(root_dir / "episodes", collector).run_episode(
            load_environment(bundle.root), TraceReplayAgent(actions)
        )
        results.append(ImportedRoot(
            example_id, str(sample.get("trace_id", "")), task["task_id"],
            float(base_metadata["root_reward"]), failure_types, len(actions), len(collector.snapshots),
            summary.evaluation.score, summary.artifact_root,
        ))
    return results


def _has_failed_public_check(sample: dict[str, Any]) -> bool:
    pending_check = False
    for message in sample.get("completion") or []:
        if message.get("role") == "assistant":
            pending_check = False
            for raw in message.get("tool_calls") or []:
                call = json.loads(raw) if isinstance(raw, str) else raw
                if (call.get("name") or call.get("tool")) in {"run_shell", "run_python"}:
                    pending_check = True
        elif message.get("role") == "tool" and pending_check:
            if _FAILURE_TEXT.search(str(message.get("content", ""))):
                return True
            pending_check = False
    return False
