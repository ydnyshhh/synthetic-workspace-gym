from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from synthetic_workspace_gym.generators.common import build_complexity_profile, make_env_id
from synthetic_workspace_gym.schemas import EnvironmentManifest, EnvironmentSpec
from synthetic_workspace_gym.utils.io import write_json, write_text
from synthetic_workspace_gym.utils.paths import list_relative_files
from synthetic_workspace_gym.utils.scratch import scratch_directory


@dataclass(slots=True)
class GeneratedEnvironment:
    root: Path
    visible_root: Path
    hidden_root: Path
    manifest: EnvironmentManifest


@dataclass(slots=True)
class GeneratedPayload:
    instruction: str
    metadata: dict[str, object]
    reference_solution: dict[str, object]
    evaluator_entrypoint: str


class BaseGenerator(ABC):
    family = None

    def sample_spec(self, difficulty: int, seed: int, **overrides: object) -> EnvironmentSpec:
        difficulty = int(difficulty)
        spec = EnvironmentSpec(
            env_family=self.family,
            difficulty=difficulty,
            seed=seed,
            max_steps=int(overrides.pop("max_steps", 12)),
            time_limit_seconds=int(overrides.pop("time_limit_seconds", 60)),
            task_params=dict(overrides.pop("task_params", {})),
            evaluator_params=dict(overrides.pop("evaluator_params", {})),
            generation_params=dict(overrides.pop("generation_params", {})),
            complexity_profile=build_complexity_profile(self.family, difficulty),
        )
        if overrides:
            raise ValueError(f"Unsupported spec overrides: {sorted(overrides.keys())}")
        return spec

    def generate_instance(self, spec: EnvironmentSpec, output_dir: Path, *, validate: bool = True) -> GeneratedEnvironment:
        env_id = make_env_id(spec.env_family, spec.difficulty, spec.seed, spec.task_params)
        root = output_dir / env_id
        if root.exists():
            shutil.rmtree(root)
        visible_root = root / "visible"
        hidden_root = root / "hidden"
        visible_root.mkdir(parents=True, exist_ok=True)
        hidden_root.mkdir(parents=True, exist_ok=True)

        payload = self._build_environment(spec, root=root, visible_root=visible_root, hidden_root=hidden_root)
        manifest = EnvironmentManifest(
            env_id=env_id,
            family=spec.env_family,
            difficulty=spec.difficulty,
            seed=spec.seed,
            instruction=payload.instruction,
            workspace_root="visible",
            visible_files=list_relative_files(visible_root),
            hidden_root="hidden",
            hidden_files=list_relative_files(hidden_root),
            tool_permissions=spec.tool_permissions,
            max_steps=spec.max_steps,
            time_limit_seconds=spec.time_limit_seconds,
            metadata=payload.metadata,
            evaluator_entrypoint=payload.evaluator_entrypoint,
            reference_solution=payload.reference_solution,
        )
        write_json(root / "manifest.json", manifest.to_dict())
        bundle = GeneratedEnvironment(root=root, visible_root=visible_root, hidden_root=hidden_root, manifest=manifest)
        if validate:
            result = self.validate_instance(bundle)
            if not result.success:
                raise ValueError(
                    f"Reference solution failed validation for '{manifest.env_id}': {result.failure_labels}"
                )
        return bundle

    def validate_instance(self, instance: GeneratedEnvironment):
        from synthetic_workspace_gym.evaluators.registry import get_evaluator

        evaluator = get_evaluator(
            instance.manifest.family,
            evaluator_entrypoint=instance.manifest.evaluator_entrypoint,
        )
        scratch_root = instance.root.parent / ".validation"
        scratch_root.mkdir(parents=True, exist_ok=True)
        with scratch_directory(scratch_root, "swg-validate-") as tmp_dir:
            workspace = tmp_dir / "workspace"
            shutil.copytree(instance.visible_root, workspace)
            for relative_path, content in instance.manifest.reference_solution.get("files", {}).items():
                write_text(workspace / relative_path, str(content))
            return evaluator.evaluate(workspace, instance.manifest, instance.hidden_root)

    @abstractmethod
    def _build_environment(self, spec: EnvironmentSpec, *, root: Path, visible_root: Path, hidden_root: Path) -> GeneratedPayload:
        raise NotImplementedError
