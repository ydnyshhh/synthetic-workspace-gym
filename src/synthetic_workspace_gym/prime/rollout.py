from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from synthetic_workspace_gym.analysis.artifacts import build_unified_diff, snapshot_texts
from synthetic_workspace_gym.prime.agents import PrimeReActAgent
from synthetic_workspace_gym.prime.clients import PrimeModelClient, ScriptedPrimeClient
from synthetic_workspace_gym.prime.env import SyntheticWorkspacePrimeEnv
from synthetic_workspace_gym.prime.transcript import write_transcript_jsonl
from synthetic_workspace_gym.schemas import utc_timestamp
from synthetic_workspace_gym.utils.io import write_json, write_text


def run_prime_rollout(
    family: str | None = None,
    scenario: str | None = None,
    difficulty: int = 3,
    seed: int = 0,
    environment_path: str | Path | None = None,
    client: PrimeModelClient | None = None,
    output_dir: str | Path = "prime_rollouts",
    max_turns: int | None = None,
    rollout_id: str | None = None,
) -> dict[str, Any]:
    runtime_root = Path(output_dir) / ".tmp" / f"runtime-{uuid4().hex[:10]}"
    env = SyntheticWorkspacePrimeEnv(
        family=family or "script_repair",
        scenario=scenario,
        difficulty=difficulty,
        seed=seed,
        workspace_root=environment_path,
        output_dir=runtime_root,
    )
    started_at = utc_timestamp()
    started = time.perf_counter()
    try:
        agent = PrimeReActAgent(client or ScriptedPrimeClient(_default_scripted_actions()), max_turns=max_turns)
        rollout = agent.run(env)
        ended_at = utc_timestamp()
        rollout.update(
            {
                "started_at": started_at,
                "ended_at": ended_at,
                "duration_seconds": time.perf_counter() - started,
                "client": client or agent.client,
            }
        )
        return write_prime_rollout_artifacts(rollout, env, output_dir=output_dir, rollout_id=rollout_id)
    finally:
        env.close()
        if runtime_root.exists():
            shutil.rmtree(runtime_root, ignore_errors=True)
        try:
            runtime_root.parent.rmdir()
        except OSError:
            pass


def write_prime_rollout_artifacts(
    rollout: dict[str, Any],
    env: SyntheticWorkspacePrimeEnv,
    output_dir: str | Path,
    rollout_id: str | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    rollout_id = rollout_id or _make_rollout_id(env.env_id)
    artifact_root = output_root / rollout_id
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    payload = build_prime_rollout_payload(rollout, env, rollout_id=rollout_id)
    write_transcript_jsonl(artifact_root / "transcript.jsonl", rollout.get("transcript_events", []))
    write_json(artifact_root / "final_reward.json", payload["reward_payload"])
    env.write_manifest_snapshot(artifact_root / "manifest.json")
    env.copy_final_workspace(artifact_root / "final_workspace")
    write_text(artifact_root / "final_diff.txt", _build_final_diff(env))

    payload.update(
        {
            "transcript_path": "transcript.jsonl",
            "final_workspace_path": "final_workspace",
            "final_reward_path": "final_reward.json",
            "manifest_path": "manifest.json",
            "final_diff_path": "final_diff.txt",
        }
    )
    write_json(artifact_root / "prime_rollout.json", payload)
    return {
        "rollout_id": rollout_id,
        "env_id": payload["env_id"],
        "success": payload["success"],
        "final_reward": payload["final_reward"],
        "artifact_dir": str(artifact_root),
        "prime_rollout_path": str(artifact_root / "prime_rollout.json"),
        "reward_payload": payload["reward_payload"],
    }


def build_prime_rollout_payload(
    rollout: dict[str, Any],
    env: SyntheticWorkspacePrimeEnv,
    rollout_id: str,
) -> dict[str, Any]:
    manifest = env.manifest
    reward_payload = dict(rollout.get("reward_payload", {}) or {})
    tool_calls = list(rollout.get("tool_calls", []))
    observations = list(rollout.get("observations", []))
    tool_counts = Counter(str(call.get("tool", "")) for call in tool_calls)
    client = rollout.get("client")
    client_type = str(getattr(client, "client_type", "external"))
    model_name = str(getattr(client, "name", client_type))
    scenario = manifest.metadata.get("scenario_id")
    task_id = f"swg.{manifest.family.value}.{scenario or 'default'}.d{manifest.difficulty}.s{manifest.seed}"
    return {
        "rollout_id": rollout_id,
        "env_id": manifest.env_id,
        "task_id": task_id,
        "family": manifest.family.value,
        "scenario": scenario,
        "difficulty": manifest.difficulty,
        "seed": manifest.seed,
        "model": {"name": model_name, "client_type": client_type},
        "started_at": rollout.get("started_at"),
        "ended_at": rollout.get("ended_at"),
        "duration_seconds": round(float(rollout.get("duration_seconds", 0.0)), 6),
        "success": bool(reward_payload.get("success", False)),
        "final_reward": float(reward_payload.get("reward", 0.0)),
        "reward_payload": reward_payload,
        "turn_count": int(rollout.get("turn_count", len(tool_calls))),
        "tool_call_count": len(tool_calls),
        "tool_counts": dict(sorted(tool_counts.items())),
        "failure_labels": list(reward_payload.get("failure_labels", []) or []),
        "subscores": dict(reward_payload.get("subscores", {}) or {}),
        "messages": list(rollout.get("messages", [])),
        "tool_calls": tool_calls,
        "observations": observations,
        "transcript_path": None,
        "final_workspace_path": None,
        "final_reward_path": None,
        "manifest_path": None,
        "final_diff_path": None,
        "metadata": {
            "max_steps": manifest.max_steps,
            "time_limit_seconds": manifest.time_limit_seconds,
            "environment_path": str(env.environment.root),
        },
    }


def run_prime_rollout_batch(
    manifest_path: str | Path,
    *,
    client_factory: Any,
    output_dir: str | Path = "prime_rollouts",
    limit: int | None = None,
    max_turns: int | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        rows = rows[:limit]

    rollouts: list[dict[str, Any]] = []
    manifest_root = manifest_path.parent
    for row in rows:
        environment_path = manifest_root / str(row["environment_path"])
        result = run_prime_rollout(
            environment_path=environment_path,
            client=client_factory(),
            output_dir=output_dir,
            max_turns=max_turns,
        )
        rollouts.append(
            {
                "rollout_id": result["rollout_id"],
                "env_id": result["env_id"],
                "reward": result["final_reward"],
                "success": result["success"],
                "path": result["artifact_dir"],
            }
        )

    summary = build_batch_summary(rollouts)
    write_json(Path(output_dir) / "batch_summary.json", summary)
    return summary


def build_batch_summary(rollouts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(rollouts)
    rewards = [float(row.get("reward", 0.0)) for row in rollouts]
    successes = [bool(row.get("success", False)) for row in rollouts]
    return {
        "count": count,
        "success_rate": (sum(1 for item in successes if item) / count) if count else 0.0,
        "mean_reward": (sum(rewards) / count) if count else 0.0,
        "rollouts": list(rollouts),
    }


def _build_final_diff(env: SyntheticWorkspacePrimeEnv) -> str:
    return build_unified_diff(snapshot_texts(env.initial_visible_root), snapshot_texts(env.active_workspace))


def _make_rollout_id(env_id: str) -> str:
    return f"{env_id}-prime-{uuid4().hex[:10]}"


def _default_scripted_actions() -> list[dict[str, Any]]:
    return [
        {"tool": "list_directory", "args": {"path": "."}},
        {"tool": "submit", "args": {"path_or_answer": "done"}},
    ]
