from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from synthetic_workspace_gym.evaluators.registry import get_evaluator
from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.runtime.environment import LoadedEnvironment, load_environment
from synthetic_workspace_gym.runtime.tools import WorkspaceToolExecutor
from synthetic_workspace_gym.schemas import Action, ActionType, ToolObservation

from .tools import get_tool_schemas
from .verifier import evaluator_result_to_prime_reward


class SyntheticWorkspacePrimeEnv:
    def __init__(
        self,
        family: str,
        scenario: str | None = None,
        difficulty: int = 3,
        seed: int = 0,
        max_steps: int | None = None,
        workspace_root: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.family = family
        self.scenario = scenario
        self.difficulty = int(difficulty)
        self.seed = int(seed)
        self.max_steps = max_steps
        self.workspace_root = Path(workspace_root).resolve() if workspace_root is not None else None
        self.output_dir = Path(output_dir).resolve() if output_dir is not None else None

        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._environment: LoadedEnvironment | None = None
        self._active_workspace: Path | None = None
        self._executor: WorkspaceToolExecutor | None = None
        self._done = False
        self._step_count = 0
        self._started_at: float | None = None
        self._last_reward_payload: dict[str, object] | None = None

    def reset(self) -> dict[str, object]:
        self.close()
        environment = self._load_or_generate_environment()
        active_parent = self._runtime_root() / "active-workspaces"
        active_parent.mkdir(parents=True, exist_ok=True)
        active_workspace = active_parent / environment.manifest.env_id
        if active_workspace.exists():
            shutil.rmtree(active_workspace)
        shutil.copytree(environment.visible_root, active_workspace)

        self._environment = environment
        self._active_workspace = active_workspace
        self._executor = WorkspaceToolExecutor(active_workspace, environment.manifest.tool_permissions)
        self._done = False
        self._step_count = 0
        self._started_at = time.perf_counter()
        self._last_reward_payload = None

        manifest = environment.manifest
        return {
            "env_id": manifest.env_id,
            "instruction": manifest.instruction,
            "family": manifest.family.value,
            "scenario": manifest.metadata.get("scenario_id", self.scenario),
            "difficulty": manifest.difficulty,
            "seed": manifest.seed,
            "max_steps": manifest.max_steps,
            "time_limit_seconds": manifest.time_limit_seconds,
            "tool_schemas": get_tool_schemas(manifest.tool_permissions.enabled_tools()),
            "metadata": {
                **manifest.metadata,
                "environment_path": str(environment.root),
                "workspace_path": str(active_workspace),
            },
        }

    def step(self, action: dict[str, object]) -> dict[str, object]:
        environment, executor = self._require_active()
        if self._done:
            return {
                "observation": "Episode is already done.",
                "done": True,
                "reward": float((self._last_reward_payload or {}).get("reward", 0.0)),
                "info": {"reward_payload": self._last_reward_payload},
            }

        tool_name = str(action.get("tool", ""))
        arguments = dict(action.get("args", {}) or {})
        try:
            swg_action = Action(ActionType(tool_name), arguments)
        except ValueError:
            return {
                "observation": f"Unknown tool: {tool_name}",
                "done": False,
                "reward": 0.0,
                "info": {"success": False, "error": "unknown_tool"},
            }

        try:
            observation = executor.execute(swg_action, remaining_time_seconds=self._remaining_time_seconds())
        except Exception as exc:
            observation = ToolObservation(
                success=False,
                message=f"Tool execution failed: {type(exc).__name__}",
                error=str(exc),
            )
        self._step_count += 1

        self._done = (
            swg_action.action_type == ActionType.SUBMIT
            or self._step_count >= environment.manifest.max_steps
            or self._remaining_time_seconds() <= 0
        )

        reward_payload: dict[str, object] | None = None
        reward = 0.0
        if self._done:
            reward_payload = self.evaluate()
            self._last_reward_payload = reward_payload
            reward = float(reward_payload.get("reward", 0.0))

        info = self._observation_info(observation)
        info.update(
            {
                "step_index": self._step_count - 1,
                "remaining_steps": max(0, environment.manifest.max_steps - self._step_count),
                "submitted": swg_action.action_type == ActionType.SUBMIT,
            }
        )
        if reward_payload is not None:
            info["reward_payload"] = reward_payload

        return {
            "observation": self._observation_text(observation),
            "done": self._done,
            "reward": reward,
            "info": info,
        }

    def evaluate(self) -> dict[str, object]:
        environment, _ = self._require_active()
        if self._active_workspace is None:
            raise RuntimeError("Environment has no active workspace. Call reset() first.")
        evaluator = get_evaluator(
            environment.manifest.family,
            evaluator_entrypoint=environment.manifest.evaluator_entrypoint,
        )
        result = evaluator.evaluate(self._active_workspace, environment.manifest, environment.hidden_root)
        payload = evaluator_result_to_prime_reward(result)
        payload["env_id"] = environment.manifest.env_id
        return payload

    def close(self) -> None:
        self._environment = None
        self._active_workspace = None
        self._executor = None
        self._done = False
        self._step_count = 0
        self._started_at = None
        self._last_reward_payload = None
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def _load_or_generate_environment(self) -> LoadedEnvironment:
        if self.workspace_root is not None and (self.workspace_root / "manifest.json").exists():
            return load_environment(self.workspace_root)

        generator = get_generator(self.family)
        spec = generator.sample_spec(
            difficulty=self.difficulty,
            seed=self.seed,
            scenario_id=self.scenario,
            max_steps=self.max_steps or 12,
        )
        bundle = generator.generate_instance(spec, self._runtime_root() / "generated")
        return load_environment(bundle.root)

    def _runtime_root(self) -> Path:
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            return self.output_dir
        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="swg-prime-")
        return Path(self._temp_dir.name)

    def _require_active(self) -> tuple[LoadedEnvironment, WorkspaceToolExecutor]:
        if self._environment is None or self._executor is None:
            raise RuntimeError("Environment is not active. Call reset() first.")
        return self._environment, self._executor

    def _remaining_time_seconds(self) -> float:
        if self._environment is None or self._started_at is None:
            return 0.0
        elapsed = time.perf_counter() - self._started_at
        return max(0.0, float(self._environment.manifest.time_limit_seconds) - elapsed)

    def _observation_text(self, observation: ToolObservation) -> str:
        parts = [observation.message]
        if observation.content:
            parts.append(observation.content)
        if observation.stdout:
            parts.append(f"stdout:\n{observation.stdout}")
        if observation.stderr:
            parts.append(f"stderr:\n{observation.stderr}")
        if observation.error and not observation.content:
            parts.append(f"error: {observation.error}")
        return "\n".join(parts)

    def _observation_info(self, observation: ToolObservation) -> dict[str, Any]:
        return {
            "success": observation.success,
            "message": observation.message,
            "stdout": observation.stdout,
            "stderr": observation.stderr,
            "exit_code": observation.exit_code,
            "listing": observation.listing,
            "error": observation.error,
            "touched_files": observation.touched_files,
            "workspace_digest": observation.workspace_digest,
        }
