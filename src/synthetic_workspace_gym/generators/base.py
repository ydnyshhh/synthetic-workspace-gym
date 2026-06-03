from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from synthetic_workspace_gym.generators.common import build_complexity_profile, make_env_id
from synthetic_workspace_gym.schemas import EnvironmentFamily, EnvironmentManifest, EnvironmentSpec
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
    family: EnvironmentFamily | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls is BaseGenerator:
            return
        family = getattr(cls, "family", None)
        if family is None:
            raise TypeError(f"{cls.__name__} must define a concrete EnvironmentFamily on the 'family' class attribute.")
        cls.family = EnvironmentFamily(family)

    def sample_spec(self, difficulty: int, seed: int, **overrides: object) -> EnvironmentSpec:
        difficulty = int(difficulty)
        family = self.require_family()
        generation_params = dict(overrides.pop("generation_params", {}))
        scenario_id = overrides.pop("scenario_id", generation_params.pop("scenario_id", None))
        split = overrides.pop("split", generation_params.get("split", None))
        task_id = overrides.pop("task_id", generation_params.get("task_id", None))
        if split is not None:
            generation_params["split"] = str(split)
        if task_id is not None:
            generation_params["task_id"] = str(task_id)
        spec = EnvironmentSpec(
            env_family=family,
            difficulty=difficulty,
            seed=seed,
            scenario_id=str(scenario_id) if scenario_id is not None else None,
            max_steps=int(overrides.pop("max_steps", 12)),
            time_limit_seconds=int(overrides.pop("time_limit_seconds", 60)),
            task_params=dict(overrides.pop("task_params", {})),
            evaluator_params=dict(overrides.pop("evaluator_params", {})),
            generation_params=generation_params,
            complexity_profile=build_complexity_profile(family, difficulty),
        )
        if overrides:
            raise ValueError(f"Unsupported spec overrides: {sorted(overrides.keys())}")
        return spec

    def generate_instance(self, spec: EnvironmentSpec, output_dir: Path, *, validate: bool = True) -> GeneratedEnvironment:
        env_id = make_env_id(
            spec.env_family,
            spec.difficulty,
            spec.seed,
            spec.task_params,
            scenario_id=spec.scenario_id,
            generation_params=spec.generation_params,
        )
        root = output_dir / env_id
        if root.exists():
            shutil.rmtree(root)
        visible_root = root / "visible"
        hidden_root = root / "hidden"
        visible_root.mkdir(parents=True, exist_ok=True)
        hidden_root.mkdir(parents=True, exist_ok=True)

        payload = self.build_environment(spec, root=root, visible_root=visible_root, hidden_root=hidden_root)
        metadata = dict(payload.metadata)
        split = spec.generation_params.get("split")
        task_id = spec.generation_params.get("task_id")
        if split is not None:
            metadata.update(
                {
                    "split": str(split),
                    "split_family": spec.env_family.value,
                    "split_scenario": spec.scenario_id,
                    "split_difficulty": spec.difficulty,
                    "split_seed": spec.seed,
                }
            )
        if task_id is not None:
            metadata["task_id"] = str(task_id)
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
            metadata=metadata,
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

        self.validate_reference_solution(instance)
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
    def build_environment(self, spec: EnvironmentSpec, *, root: Path, visible_root: Path, hidden_root: Path) -> GeneratedPayload:
        raise NotImplementedError

    def require_family(self) -> EnvironmentFamily:
        if self.family is None:
            raise TypeError(f"{type(self).__name__} must define a concrete EnvironmentFamily before use.")
        return EnvironmentFamily(self.family)

    def select_scenario(self, spec: EnvironmentSpec, scenarios: list[dict[str, object]]) -> dict[str, object]:
        requested = spec.scenario_id
        if requested is None:
            requested = spec.generation_params.get("scenario_id")
        if requested:
            requested = str(requested)
            for scenario in scenarios:
                if str(scenario.get("scenario_id")) == requested:
                    return scenario
            available = ", ".join(sorted(str(scenario.get("scenario_id")) for scenario in scenarios))
            raise ValueError(
                f"Unknown scenario_id '{requested}' for {self.require_family().value}. Available: {available}"
            )
        return scenarios[(spec.seed - 1) % len(scenarios)]

    def validate_reference_solution(self, instance: GeneratedEnvironment) -> None:
        solution_files = instance.manifest.reference_solution.get("files", {})
        if not solution_files:
            raise ValueError("Reference solution metadata must include at least one file artifact.")
        unchanged_paths: list[str] = []
        for relative_path, content in solution_files.items():
            visible_path = instance.visible_root / relative_path
            if visible_path.exists() and visible_path.read_text(encoding="utf-8") == str(content):
                unchanged_paths.append(relative_path)
        if unchanged_paths:
            joined = ", ".join(sorted(unchanged_paths))
            raise ValueError(
                "Reference solution did not modify one or more visible files; generation likely failed to apply a change: "
                f"{joined}"
            )
