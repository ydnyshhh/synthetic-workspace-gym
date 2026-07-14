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
from synthetic_workspace_gym.sandbox.evaluator import verify_workspace_in_sandbox
from synthetic_workspace_gym.sandbox.runner import build_sandbox_backend
from synthetic_workspace_gym.sandbox.schemas import SandboxConfig
from synthetic_workspace_gym.schemas import Action, ActionType, ToolObservation
from synthetic_workspace_gym.utils.io import write_json

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
        sandbox_backend: str = "local",
        sandbox_config: SandboxConfig | None = None,
        docker_image: str | None = None,
        time_limit_seconds: int | None = None,
    ) -> None:
        self.family = family
        self.scenario = scenario
        self.difficulty = int(difficulty)
        self.seed = int(seed)
        self.max_steps = max_steps
        self.workspace_root = Path(workspace_root).resolve() if workspace_root is not None else None
        self.output_dir = Path(output_dir).resolve() if output_dir is not None else None
        self.time_limit_seconds = int(time_limit_seconds) if time_limit_seconds is not None else None
        self.sandbox_config = sandbox_config or SandboxConfig(backend=sandbox_backend)
        self.sandbox_config.backend = sandbox_backend  # type: ignore[assignment]
        if docker_image is not None:
            self.sandbox_config.image = docker_image

        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._environment: LoadedEnvironment | None = None
        self._active_workspace: Path | None = None
        self._executor: WorkspaceToolExecutor | None = None
        self._done = False
        self._step_count = 0
        self._started_at: float | None = None
        self._last_reward_payload: dict[str, object] | None = None
        self._step_limit: int | None = None

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
        runtime_home = self._runtime_root() / "runtime-home" / environment.manifest.env_id
        sandbox_tool_backend = (
            build_sandbox_backend(self.sandbox_config)
            if self.sandbox_config.backend != "local"
            else None
        )
        self._executor = WorkspaceToolExecutor(
            active_workspace,
            environment.manifest.tool_permissions,
            runtime_home=runtime_home,
            sandbox_backend=sandbox_tool_backend,
            sandbox_config=self.sandbox_config,
        )
        self._done = False
        self._step_count = 0
        self._started_at = time.perf_counter()
        self._last_reward_payload = None

        manifest = environment.manifest
        self._step_limit = int(self.max_steps or manifest.max_steps)
        return {
            "env_id": manifest.env_id,
            "instruction": manifest.instruction,
            "family": manifest.family.value,
            "scenario": manifest.metadata.get("scenario_id", self.scenario),
            "difficulty": manifest.difficulty,
            "seed": manifest.seed,
            "max_steps": self._step_limit,
            "time_limit_seconds": manifest.time_limit_seconds,
            "tool_schemas": get_tool_schemas(manifest.tool_permissions.enabled_tools()),
            "sandbox": {
                "backend": self.sandbox_config.backend,
                "image": self.sandbox_config.image,
                "network_enabled": self.sandbox_config.network_enabled,
            },
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
            message = f"Tool execution failed: {type(exc).__name__}"
            if isinstance(exc, KeyError):
                message = f"Tool execution failed: missing required argument {exc!s}"
            observation = ToolObservation(
                success=False,
                message=message,
                error="tool_execution_error",
            )
            exception_info: dict[str, object] = {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
        else:
            exception_info = {}
        self._step_count += 1

        self._done = (
            swg_action.action_type == ActionType.SUBMIT
            or self._step_count >= int(self._step_limit or environment.manifest.max_steps)
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
                "remaining_steps": max(0, int(self._step_limit or environment.manifest.max_steps) - self._step_count),
                "submitted": swg_action.action_type == ActionType.SUBMIT,
            }
        )
        if reward_payload is not None:
            info["reward_payload"] = reward_payload
        info.update(exception_info)

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
        if self.sandbox_config.backend != "local":
            payload = verify_workspace_in_sandbox(environment.root, self._active_workspace, self.sandbox_config)
            payload["env_id"] = environment.manifest.env_id
            return payload
        evaluator = get_evaluator(
            environment.manifest.family,
            evaluator_entrypoint=environment.manifest.evaluator_entrypoint,
        )
        result = evaluator.evaluate(self._active_workspace, environment.manifest, environment.hidden_root)
        payload = evaluator_result_to_prime_reward(result)
        payload["env_id"] = environment.manifest.env_id
        return payload

    def close(self) -> None:
        executor = self._executor
        if executor is not None and executor.sandbox_backend is not None:
            close = getattr(executor.sandbox_backend, "close", None)
            if callable(close):
                close()
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

    @property
    def environment(self) -> LoadedEnvironment:
        if self._environment is None:
            raise RuntimeError("Environment is not active. Call reset() first.")
        return self._environment

    @property
    def active_workspace(self) -> Path:
        if self._active_workspace is None:
            raise RuntimeError("Environment has no active workspace. Call reset() first.")
        return self._active_workspace

    @property
    def initial_visible_root(self) -> Path:
        return self.environment.visible_root

    @property
    def manifest(self):
        return self.environment.manifest

    @property
    def env_id(self) -> str:
        return self.manifest.env_id

    def get_environment(self) -> LoadedEnvironment:
        return self.environment

    def get_active_workspace(self) -> Path:
        return self.active_workspace

    def get_manifest(self):
        return self.manifest

    def copy_final_workspace(self, target_dir: str | Path) -> Path:
        target = Path(target_dir)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(self.active_workspace, target)
        return target

    def write_manifest_snapshot(self, target_path: str | Path) -> Path:
        target = Path(target_path)
        write_json(target, self.manifest.to_dict())
        return target

    def _load_or_generate_environment(self) -> LoadedEnvironment:
        if self.workspace_root is not None:
            if not (self.workspace_root / "manifest.json").exists():
                raise FileNotFoundError(f"Missing manifest.json under {self.workspace_root}")
            return load_environment(self.workspace_root)

        generator = get_generator(self.family)
        spec = generator.sample_spec(
            difficulty=self.difficulty,
            seed=self.seed,
            scenario_id=self.scenario,
            max_steps=self.max_steps or 12,
            **({"time_limit_seconds": self.time_limit_seconds} if self.time_limit_seconds is not None else {}),
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
