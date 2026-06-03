from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from synthetic_workspace_gym.prime.env import SyntheticWorkspacePrimeEnv
from synthetic_workspace_gym.sandbox.schemas import SandboxConfig

from .compat import require_verifiers, vf
from .parser import SWGToolCallParser
from .rewards import compute_reward, normalize_reward_payload, to_verifiers_info
from .tools import get_verifiers_tools


SYSTEM_PROMPT = (
    "You are operating inside a local synthetic workspace. Use only provided tools. "
    "Inspect files before editing. Do not access hidden evaluator files. Return exactly "
    "one tool call per turn. When finished, call submit."
)


class SyntheticWorkspaceVerifiersEnv:
    def __init__(
        self,
        family: str = "script_repair",
        scenario: str | None = None,
        difficulty: int = 3,
        seed: int = 0,
        environment_path: str | Path | None = None,
        sandbox_backend: str = "local",
        sandbox_config: SandboxConfig | None = None,
        docker_image: str | None = None,
        reward_mode: str = "score",
        reward_weights: dict[str, float] | None = None,
        max_turns: int | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.family = family
        self.scenario = scenario
        self.difficulty = int(difficulty)
        self.seed = int(seed)
        self.environment_path = Path(environment_path) if environment_path is not None else None
        self.reward_mode = reward_mode
        self.reward_weights = dict(reward_weights or {})
        self.max_turns = max_turns
        self.parser = SWGToolCallParser()
        self._last_reset: dict[str, Any] | None = None
        self._prime_env = SyntheticWorkspacePrimeEnv(
            family=family,
            scenario=scenario,
            difficulty=difficulty,
            seed=seed,
            max_steps=max_turns,
            workspace_root=self.environment_path,
            output_dir=output_dir,
            sandbox_backend=sandbox_backend,
            sandbox_config=sandbox_config,
            docker_image=docker_image,
        )

    def reset(self) -> dict[str, Any]:
        observation = self._prime_env.reset()
        self._last_reset = dict(observation)
        return {
            **observation,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "task": self.task,
        }

    def step(self, action_or_completion: object) -> dict[str, Any]:
        action = self.parser.parse(action_or_completion)
        result = self._prime_env.step(action)
        info = dict(result.get("info", {}) or {})
        reward_payload = info.get("reward_payload")
        reward = float(result.get("reward", 0.0) or 0.0)
        if isinstance(reward_payload, dict):
            normalized = normalize_reward_payload(reward_payload)
            reward = compute_reward(normalized, mode=self.reward_mode, weights=self.reward_weights)
            info["reward_payload"] = normalized
            info["verifiers"] = to_verifiers_info(normalized)
        if action.get("parse_error"):
            info["parse_error"] = action["parse_error"]
        return {
            "observation": result.get("observation", ""),
            "reward": reward,
            "done": bool(result.get("done", False)),
            "info": info,
        }

    def evaluate(self) -> dict[str, Any]:
        payload = normalize_reward_payload(self._prime_env.evaluate())
        payload["reward"] = compute_reward(payload, mode=self.reward_mode, weights=self.reward_weights)
        return payload

    def close(self) -> None:
        self._prime_env.close()

    @property
    def tools(self) -> list[object]:
        permissions = None
        try:
            permissions = self._prime_env.manifest.tool_permissions.enabled_tools()
        except RuntimeError:
            pass
        return get_verifiers_tools(permissions)

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    @property
    def task(self) -> dict[str, Any]:
        if self._last_reset is not None:
            reset = self._last_reset
            return {
                "task_id": f"swg.{reset.get('family')}.{reset.get('scenario') or 'default'}.d{reset.get('difficulty')}.s{reset.get('seed')}",
                "env_id": reset.get("env_id"),
                "family": reset.get("family"),
                "scenario": reset.get("scenario"),
                "difficulty": reset.get("difficulty"),
                "seed": reset.get("seed"),
                "instruction": reset.get("instruction"),
                "environment_path": (reset.get("metadata") or {}).get("environment_path")
                if isinstance(reset.get("metadata"), dict)
                else None,
                "metadata": dict(reset.get("metadata", {}) or {}),
            }
        return {
            "task_id": f"swg.{self.family}.{self.scenario or 'default'}.d{self.difficulty}.s{self.seed}",
            "env_id": None,
            "family": self.family,
            "scenario": self.scenario,
            "difficulty": self.difficulty,
            "seed": self.seed,
            "instruction": None,
            "environment_path": str(self.environment_path) if self.environment_path else None,
            "metadata": {},
        }

    @property
    def prime_env(self) -> SyntheticWorkspacePrimeEnv:
        return self._prime_env


def make_verifiers_env(**kwargs: Any) -> object:
    require_verifiers()
    return adapt_to_verifiers(SyntheticWorkspaceVerifiersEnv(**kwargs))


def adapt_to_verifiers(base_env: SyntheticWorkspaceVerifiersEnv, vf_module: Any | None = None) -> object:
    module = vf_module if vf_module is not None else vf
    if module is None:
        return base_env

    for env_cls in _native_environment_classes(module):
        adapter = _try_construct_native(env_cls, base_env)
        if adapter is not None:
            return _NativeVerifiersAdapter(base_env, adapter)
    return base_env


def _try_construct_native(env_cls: Any, base_env: SyntheticWorkspaceVerifiersEnv) -> object | None:
    dataset = _native_dataset(base_env)
    kwargs_candidates = [
        {
            "dataset": dataset,
            "tool_defs": base_env.tools,
            "system_prompt": base_env.system_prompt,
            "env_id": base_env.task["task_id"],
        },
        {
            "dataset": dataset,
            "tools": [],
            "system_prompt": base_env.system_prompt,
            "env_id": base_env.task["task_id"],
        },
        {},
    ]
    for kwargs in kwargs_candidates:
        try:
            native = env_cls(**kwargs)
        except Exception:
            continue
        return native
    return None


def _native_environment_classes(module: Any) -> list[Any]:
    classes: list[Any] = []
    for attr in ("SingleTurnEnv", "ToolEnv", "Environment", "Env"):
        try:
            env_cls = getattr(module, attr, None)
        except Exception:
            continue
        if env_cls is not None:
            classes.append(env_cls)
    for module_name, attr in (
        ("verifiers.envs.singleturn_env", "SingleTurnEnv"),
        ("verifiers.envs.tool_env", "ToolEnv"),
        ("verifiers.envs.environment", "Environment"),
    ):
        try:
            env_cls = getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError):
            continue
        if env_cls not in classes:
            classes.append(env_cls)
    return classes


def _native_dataset(base_env: SyntheticWorkspaceVerifiersEnv) -> object:
    row = dict(base_env.task)
    row["question"] = row.get("instruction") or row.get("task_id") or "Solve the SWG workspace task."
    try:
        from datasets import Dataset  # type: ignore[import-not-found]

        return Dataset.from_list([row])
    except Exception:
        return [row]


class _NativeVerifiersAdapter:
    def __init__(self, base_env: SyntheticWorkspaceVerifiersEnv, native_env: object) -> None:
        self.base_env = base_env
        self.native_env = native_env

    def reset(self) -> dict[str, Any]:
        return self.base_env.reset()

    def step(self, action_or_completion: object) -> dict[str, Any]:
        return self.base_env.step(action_or_completion)

    def evaluate(self) -> dict[str, Any]:
        return self.base_env.evaluate()

    def close(self) -> None:
        cleanup = getattr(self.native_env, "cleanup", None)
        if callable(cleanup):
            try:
                cleanup()
            except TypeError:
                pass
        self.base_env.close()

    @property
    def tools(self) -> list[object]:
        return self.base_env.tools

    @property
    def system_prompt(self) -> str:
        return self.base_env.system_prompt

    @property
    def task(self) -> dict[str, Any]:
        return self.base_env.task

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native_env, name)
