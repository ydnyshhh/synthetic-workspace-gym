from __future__ import annotations

from pathlib import Path
from typing import Any

from synthetic_workspace_gym.prime.verifier import verify_workspace
from synthetic_workspace_gym.sandbox.evaluator import verify_workspace_in_sandbox
from synthetic_workspace_gym.sandbox.schemas import SandboxConfig


def to_verifiers_reward(payload: dict[str, Any]) -> float:
    return float(payload.get("reward", payload.get("score", 0.0)) or 0.0)


def to_verifiers_info(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": bool(payload.get("success", False)),
        "score": float(payload.get("score", payload.get("reward", 0.0)) or 0.0),
        "subscores": dict(payload.get("subscores", {}) or {}),
        "failure_labels": list(payload.get("failure_labels", []) or []),
        "diagnostics": dict(payload.get("diagnostics", {}) or {}),
        "runtime_seconds": payload.get("runtime_seconds"),
    }


def score_workspace(
    environment_path: str | Path,
    workspace_path: str | Path,
    sandbox_config: SandboxConfig | None = None,
) -> dict[str, Any]:
    if sandbox_config is not None and sandbox_config.backend == "docker":
        payload = verify_workspace_in_sandbox(environment_path, workspace_path, sandbox_config)
    else:
        payload = verify_workspace(environment_path, workspace_path)
    return normalize_reward_payload(payload)


def normalize_reward_payload(payload: dict[str, Any]) -> dict[str, Any]:
    score = float(payload.get("score", payload.get("reward", 0.0)) or 0.0)
    reward = float(payload.get("reward", score) or 0.0)
    success = bool(payload.get("success", score >= 1.0))
    return {
        **payload,
        "reward": reward,
        "success": success,
        "score": score,
        "subscores": dict(payload.get("subscores", {}) or {}),
        "failure_labels": list(payload.get("failure_labels", []) or []),
        "diagnostics": dict(payload.get("diagnostics", {}) or {}),
    }


def compute_reward(
    payload: dict[str, Any],
    mode: str = "score",
    weights: dict[str, float] | None = None,
) -> float:
    normalized = normalize_reward_payload(payload)
    if mode == "score":
        return float(normalized["score"])
    if mode == "binary":
        return 1.0 if bool(normalized["success"]) else 0.0
    if mode.startswith("subscore:"):
        name = mode.split(":", 1)[1]
        return float(normalized["subscores"].get(name, 0.0))
    if mode == "weighted":
        subscores = normalized["subscores"]
        if not weights:
            return float(normalized["score"])
        return float(sum(float(subscores.get(name, 0.0)) * float(weight) for name, weight in weights.items()))
    return float(normalized["reward"])
