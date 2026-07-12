from __future__ import annotations

from pathlib import Path
from typing import Any

from synthetic_workspace_gym.prime.clients import PrimeModelClient, ScriptedPrimeClient, normalize_client_response
from synthetic_workspace_gym.prime.rollout import run_prime_branch_rollout, run_prime_rollout, write_prime_rollout_artifacts

from .env import SyntheticWorkspaceVerifiersEnv


def run_verifiers_rollout(
    env: SyntheticWorkspaceVerifiersEnv,
    client_or_policy: Any = None,
    output_dir: str | Path = "verifiers_rollouts",
    rollout_id: str | None = None,
    max_turns: int | None = None,
) -> dict[str, Any]:
    client = _as_prime_client(client_or_policy)
    environment_path = env.environment_path
    if env.branch_manifest_path is not None:
        result = run_prime_branch_rollout(
            env.branch_manifest_path, task_id=env.branch_task.task_id if env.branch_task else None,
            branch_mode=env.branch_mode, client=client, output_dir=output_dir, max_turns=max_turns,
            rollout_id=rollout_id, sandbox_backend=env.prime_env.sandbox_config.backend,
            sandbox_config=env.prime_env.sandbox_config, docker_image=env.prime_env.sandbox_config.image,
        )
    elif environment_path is not None:
        result = run_prime_rollout(
            environment_path=environment_path,
            client=client,
            output_dir=output_dir,
            max_turns=max_turns,
            rollout_id=rollout_id,
            sandbox_backend=env.prime_env.sandbox_config.backend,
            sandbox_config=env.prime_env.sandbox_config,
            docker_image=env.prime_env.sandbox_config.image,
        )
    else:
        result = run_prime_rollout(
            family=env.family,
            scenario=env.scenario,
            difficulty=env.difficulty,
            seed=env.seed,
            client=client,
            output_dir=output_dir,
            max_turns=max_turns,
            rollout_id=rollout_id,
            sandbox_backend=env.prime_env.sandbox_config.backend,
            sandbox_config=env.prime_env.sandbox_config,
            docker_image=env.prime_env.sandbox_config.image,
        )
    return {
        "rollout_id": result["rollout_id"],
        "success": result["success"],
        "reward": result["final_reward"],
        "artifact_dir": result["artifact_dir"],
        "verifiers_compatible": True,
        "prime_rollout_path": result["prime_rollout_path"],
    }


def write_verifiers_rollout_artifacts(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return write_prime_rollout_artifacts(*args, **kwargs)


def _as_prime_client(client_or_policy: Any) -> PrimeModelClient:
    if client_or_policy is None:
        return ScriptedPrimeClient(
            [
                {"tool": "list_directory", "args": {"path": "."}},
                {"tool": "submit", "args": {"path_or_answer": "done"}},
            ]
        )
    if hasattr(client_or_policy, "complete"):
        return client_or_policy
    return _PolicyPrimeClient(client_or_policy)


class _PolicyPrimeClient(PrimeModelClient):
    name = "verifiers-policy"
    client_type = "verifiers_policy"

    def __init__(self, policy: Any) -> None:
        self.policy = policy

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if callable(self.policy):
            action = self.policy(messages=messages, tools=tools, metadata=metadata)
        else:
            action = self.policy.complete(messages, tools, metadata)
        return normalize_client_response(action)
