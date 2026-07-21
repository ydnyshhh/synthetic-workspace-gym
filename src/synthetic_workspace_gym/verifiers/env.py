from __future__ import annotations

import importlib
import json
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
    "Inspect files before editing. Do not access hidden evaluator files. Use tool calls "
    "for workspace actions. When finished, call submit."
)


class SyntheticWorkspaceVerifiersEnv:
    def __init__(
        self,
        family: str = "script_repair",
        scenario: str | None = None,
        difficulty: int = 3,
        seed: int = 0,
        composition_mode: str | None = None,
        environment_path: str | Path | None = None,
        sandbox_backend: str = "local",
        sandbox_config: SandboxConfig | None = None,
        docker_image: str | None = None,
        reward_mode: str = "score",
        reward_weights: dict[str, float] | None = None,
        max_turns: int | None = None,
        time_limit_seconds: int | None = None,
        output_dir: str | Path | None = None,
        branch_manifest_path: str | Path | None = None,
        branch_task_id: str | None = None,
        branch_mode: str | None = None,
    ) -> None:
        self.branch_manifest_path = (
            Path(branch_manifest_path).resolve()
            if branch_manifest_path is not None
            else None
        )
        self.branch_task = None
        self.branch_mode = branch_mode
        if self.branch_manifest_path is not None:
            from synthetic_workspace_gym.counterfactual.runner import (
                read_branch_manifest,
            )

            tasks = read_branch_manifest(self.branch_manifest_path)
            matches = [
                task
                for task in tasks
                if branch_task_id is None or task.task_id == branch_task_id
            ]
            if not matches:
                raise ValueError(f"branch task not found: {branch_task_id!r}")
            self.branch_task = matches[0]
            family, scenario, difficulty, seed = (
                self.branch_task.family,
                self.branch_task.scenario_id,
                self.branch_task.difficulty,
                self.branch_task.seed,
            )
            environment_path = self.branch_task.environment_path
            if max_turns is None:
                max_turns = self.branch_task.remaining_steps
            if time_limit_seconds is None:
                time_limit_seconds = self.branch_task.time_limit_seconds
            self.branch_mode = branch_mode or self.branch_task.mode
        self.family = family
        self.scenario = scenario
        self.difficulty = int(difficulty)
        self.seed = int(seed)
        self.composition_mode = composition_mode
        self.environment_path = (
            Path(environment_path) if environment_path is not None else None
        )
        self.reward_mode = reward_mode
        self.reward_weights = dict(reward_weights or {})
        self.max_turns = max_turns
        self.time_limit_seconds = time_limit_seconds
        self.parser = SWGToolCallParser()
        self._last_reset: dict[str, Any] | None = None
        self._prime_env = SyntheticWorkspacePrimeEnv(
            family=family,
            scenario=scenario,
            difficulty=difficulty,
            seed=seed,
            composition_mode=composition_mode,
            max_steps=max_turns,
            workspace_root=self.environment_path,
            output_dir=output_dir,
            sandbox_backend=sandbox_backend,
            sandbox_config=sandbox_config,
            docker_image=docker_image,
            time_limit_seconds=time_limit_seconds,
        )

    def reset(self) -> dict[str, Any]:
        observation = self._prime_env.reset()
        self._last_reset = dict(observation)
        messages = self._branch_messages()
        forced_action = (
            self.branch_task.forced_action
            if self.branch_task and self.branch_mode == "forced"
            else None
        )
        forced_result = None
        if forced_action is not None:
            forced_result = self.step(forced_action)
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": json.dumps(forced_action, sort_keys=True),
                        "metadata": {"forced": True},
                    },
                    {
                        "role": "tool",
                        "content": str(forced_result["observation"]),
                        "metadata": {"forced": True},
                    },
                ]
            )
        continuation = {}
        if forced_result is not None:
            continuation = {
                "observation": forced_result["observation"],
                "reward": forced_result["reward"],
                "done": forced_result["done"],
                "info": forced_result["info"],
            }
        return {
            **observation,
            **continuation,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "task": self.task,
            "messages": messages,
            "forced_action": forced_action,
            "forced_action_result": forced_result,
            "branch_metadata": self._branch_metadata(),
        }

    def _branch_metadata(self) -> dict[str, Any]:
        if self.branch_task is None:
            return {}
        return {
            **self.branch_task.metadata,
            "counterfactual": True,
            "branch_task_id": self.branch_task.task_id,
            "branch_group_id": self.branch_task.branch_group_id,
            "snapshot_id": self.branch_task.snapshot_id,
            "candidate_id": self.branch_task.candidate_id,
            "branch_mode": self.branch_mode,
        }

    def _branch_messages(self) -> list[dict[str, Any]]:
        if self.branch_task is None:
            return [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": str((self._last_reset or {}).get("instruction", "")),
                },
            ]
        messages = [dict(message) for message in self.branch_task.prefix_messages]
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages

    def step(self, action_or_completion: object) -> dict[str, Any]:
        action = self.parser.parse(action_or_completion)
        result = self._prime_env.step(action)
        info = dict(result.get("info", {}) or {})
        reward_payload = info.get("reward_payload")
        reward = float(result.get("reward", 0.0) or 0.0)
        if isinstance(reward_payload, dict):
            normalized = normalize_reward_payload(reward_payload)
            reward = compute_reward(
                normalized, mode=self.reward_mode, weights=self.reward_weights
            )
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
        payload["reward"] = compute_reward(
            payload, mode=self.reward_mode, weights=self.reward_weights
        )
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
                "environment_path": (reset.get("metadata") or {}).get(
                    "environment_path"
                )
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
            "environment_path": str(self.environment_path)
            if self.environment_path
            else None,
            "metadata": {},
        }

    @property
    def prime_env(self) -> SyntheticWorkspacePrimeEnv:
        return self._prime_env


def make_verifiers_env(**kwargs: Any) -> object:
    require_verifiers()
    return adapt_to_verifiers(SyntheticWorkspaceVerifiersEnv(**kwargs))


def adapt_to_verifiers(
    base_env: SyntheticWorkspaceVerifiersEnv, vf_module: Any | None = None
) -> object:
    module = vf_module if vf_module is not None else vf
    if module is None:
        return base_env

    for env_cls in _native_environment_classes(module):
        adapter = _try_construct_native(env_cls, base_env)
        if adapter is not None:
            return _NativeVerifiersAdapter(base_env, adapter)
    return base_env


def _try_construct_native(
    env_cls: Any, base_env: SyntheticWorkspaceVerifiersEnv
) -> object | None:
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
    row["question"] = (
        row.get("instruction") or row.get("task_id") or "Solve the SWG workspace task."
    )
    try:
        from datasets import Dataset  # type: ignore[import-not-found]

        return Dataset.from_list([row])
    except Exception:
        return [row]


class _NativeVerifiersAdapter:
    def __init__(
        self, base_env: SyntheticWorkspaceVerifiersEnv, native_env: object
    ) -> None:
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
