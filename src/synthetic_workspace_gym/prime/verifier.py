from __future__ import annotations

from pathlib import Path
from typing import Any

from synthetic_workspace_gym.evaluators.registry import get_evaluator
from synthetic_workspace_gym.runtime.environment import load_environment


def evaluator_result_to_prime_reward(result: Any) -> dict[str, object]:
    score = _get_field(result, "score", None)
    success = _get_field(result, "success", None)

    if score is None:
        score = 1.0 if bool(success) else 0.0
    score = float(score)

    if success is None:
        success = score >= 1.0

    return {
        "reward": score,
        "success": bool(success),
        "score": score,
        "subscores": dict(_get_field(result, "subscores", {}) or {}),
        "failure_labels": list(_get_field(result, "failure_labels", []) or []),
        "diagnostics": dict(_get_field(result, "diagnostics", {}) or {}),
        "runtime_seconds": _runtime_seconds(_get_field(result, "runtime_seconds", None)),
    }


def verify_workspace(environment_path: str | Path, workspace_path: str | Path) -> dict[str, object]:
    environment = load_environment(Path(environment_path))
    evaluator = get_evaluator(
        environment.manifest.family,
        evaluator_entrypoint=environment.manifest.evaluator_entrypoint,
    )
    result = evaluator.evaluate(Path(workspace_path).resolve(), environment.manifest, environment.hidden_root)
    payload = evaluator_result_to_prime_reward(result)
    payload["env_id"] = environment.manifest.env_id
    return payload


def _get_field(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _runtime_seconds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
