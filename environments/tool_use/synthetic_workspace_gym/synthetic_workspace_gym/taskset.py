from __future__ import annotations

import asyncio
import hashlib
import json
import posixpath
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import verifiers.v1 as vf
from pydantic import Field, field_validator

from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.schemas import EnvironmentManifest
from synthetic_workspace_gym.utils.paths import list_relative_files

Family = Literal[
    "tabular",
    "script_repair",
    "pipeline",
    "retrieval_workspace",
    "composite_workspace",
]

DEFAULT_MANIFEST = "train-all-family-seed-42"
DEFAULT_IMAGE = "python:3.12-slim"
WORKDIR = "/workspace"

SYSTEM_PROMPT = """You are solving a self-contained workspace task in /workspace.

Use the standard terminal tools supplied by your harness. Inspect README.md and task.json first, then make the smallest changes that satisfy the task. All paths must stay inside /workspace. Do not search for hidden tests or evaluator assets. Run the visible entrypoint or a focused check when useful, and finish once the requested artifact or repair is complete.
"""


class SyntheticWorkspaceData(vf.TaskData):
    family: Family
    scenario: str
    difficulty: int
    seed: int
    split: str
    generation_fingerprint: str


class SyntheticWorkspaceState(vf.State):
    manifest: dict[str, Any] = Field(default_factory=dict)
    initial_digests: dict[str, str] = Field(default_factory=dict)
    hidden_files: dict[str, bytes] = Field(default_factory=dict)
    evaluator_result: dict[str, Any] = Field(default_factory=dict)
    changed_file_count: int = 0
    final_file_count: int = 0


class SyntheticWorkspaceTaskConfig(vf.TaskConfig):
    max_result_bytes: int = 2 * 1024 * 1024


class SyntheticWorkspaceTasksetConfig(vf.TasksetConfig):
    manifest: str = DEFAULT_MANIFEST
    families: list[Family] = Field(default_factory=list)
    difficulties: list[int] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    validate_generation: bool = False
    image: str = DEFAULT_IMAGE
    task: SyntheticWorkspaceTaskConfig = SyntheticWorkspaceTaskConfig()

    @field_validator("manifest")
    @classmethod
    def validate_manifest_name(cls, value: str) -> str:
        if not value or any(token in value for token in ("/", "\\", "..")):
            raise ValueError("manifest must be a packaged SWG manifest name")
        return value

    @field_validator("difficulties")
    @classmethod
    def validate_difficulties(cls, values: list[int]) -> list[int]:
        if any(value < 1 or value > 5 for value in values):
            raise ValueError("difficulties must be between 1 and 5")
        return values


@dataclass(frozen=True)
class MaterializedWorkspace:
    manifest: dict[str, Any]
    visible_files: dict[str, bytes]
    hidden_files: dict[str, bytes]


class SyntheticWorkspaceTask(
    vf.Task[
        SyntheticWorkspaceData,
        SyntheticWorkspaceState,
        SyntheticWorkspaceTaskConfig,
    ]
):
    NEEDS_CONTAINER = True

    def __init__(
        self,
        data: SyntheticWorkspaceData,
        config: SyntheticWorkspaceTaskConfig | None = None,
        *,
        workspace: MaterializedWorkspace | None = None,
    ) -> None:
        super().__init__(data, config)
        self._workspace = workspace
        self._workspace_lock: asyncio.Lock | None = None

    async def setup(
        self,
        trace: vf.Trace[SyntheticWorkspaceData, SyntheticWorkspaceState],
        runtime: vf.Runtime,
    ) -> None:
        workspace = await self._get_workspace()
        trace.state.manifest = workspace.manifest
        trace.state.initial_digests = {
            path: _content_digest(content)
            for path, content in workspace.visible_files.items()
        }
        trace.state.hidden_files = dict(workspace.hidden_files)

        directories = sorted(
            {
                posixpath.dirname(_runtime_path(relative_path))
                for relative_path in workspace.visible_files
            }
            | {WORKDIR}
        )
        result = await runtime.run(["mkdir", "-p", *directories], {})
        if result.exit_code != 0:
            raise RuntimeError(f"failed to create SWG workspace: {result.stderr}")

        for relative_path, content in workspace.visible_files.items():
            await runtime.write(_runtime_path(relative_path), content)

        executable_paths = [
            _runtime_path(path)
            for path in workspace.visible_files
            if path.endswith(".sh")
        ]
        if executable_paths:
            result = await runtime.run(["chmod", "+x", *executable_paths], {})
            if result.exit_code != 0:
                raise RuntimeError(f"failed to mark task scripts executable: {result.stderr}")

    async def finalize(
        self,
        trace: vf.Trace[SyntheticWorkspaceData, SyntheticWorkspaceState],
        runtime: vf.Runtime,
    ) -> None:
        grade_root = f"/tmp/swg-grade-{trace.id}"
        hidden_root = f"{grade_root}/hidden"
        library_root = f"{grade_root}/lib"
        support_files = _grader_support_files()
        directories = {
            grade_root,
            hidden_root,
            library_root,
            *(
                posixpath.dirname(_trusted_runtime_path(hidden_root, path))
                for path in trace.state.hidden_files
            ),
            *(
                posixpath.dirname(_trusted_runtime_path(library_root, path))
                for path in support_files
            ),
        }
        result = await runtime.run(["mkdir", "-p", *sorted(directories)], {})
        if result.exit_code != 0:
            raise RuntimeError(f"failed to create trusted grader directory: {result.stderr}")

        for relative_path, content in trace.state.hidden_files.items():
            await runtime.write(
                _trusted_runtime_path(hidden_root, relative_path),
                content,
            )
        for relative_path, content in support_files.items():
            await runtime.write(
                _trusted_runtime_path(library_root, relative_path),
                content,
            )

        grader_path = f"{grade_root}/trusted_grader.py"
        manifest_path = f"{grade_root}/manifest.json"
        initial_path = f"{grade_root}/initial-digests.json"
        output_path = f"{grade_root}/result.json"
        await runtime.write(
            grader_path,
            Path(__file__).with_name("trusted_grader.py").read_bytes(),
        )
        await runtime.write(
            manifest_path,
            json.dumps(trace.state.manifest, sort_keys=True).encode(),
        )
        await runtime.write(
            initial_path,
            json.dumps(trace.state.initial_digests, sort_keys=True).encode(),
        )

        result = await runtime.run(
            [
                "sh",
                "-c",
                'exec "$(command -v python3 || command -v python)" "$@"',
                "swg-grader",
                grader_path,
                WORKDIR,
                hidden_root,
                manifest_path,
                initial_path,
                output_path,
            ],
            {},
        )
        if result.exit_code != 0:
            detail = (result.stderr or result.stdout).strip()[-2000:]
            raise RuntimeError(f"trusted SWG grader failed: {detail}")
        evaluator_result = json.loads(
            (
                await runtime.read(
                    output_path,
                    max_bytes=self.config.max_result_bytes,
                )
            ).decode()
        )
        changed_files = int(evaluator_result.pop("changed_file_count", 0))
        final_files = int(evaluator_result.pop("final_file_count", 0))
        trace.state.evaluator_result = evaluator_result
        trace.state.changed_file_count = changed_files
        trace.state.final_file_count = final_files
        trace.info["synthetic_workspace_gym"] = {
            "family": self.data.family,
            "scenario": self.data.scenario,
            "difficulty": self.data.difficulty,
            "seed": self.data.seed,
            "split": self.data.split,
            "generation_fingerprint": self.data.generation_fingerprint,
            "success": bool(evaluator_result.get("success", False)),
            "failure_labels": list(evaluator_result.get("failure_labels", [])),
        }

    @vf.reward(weight=1.0)
    async def workspace_score(
        self,
        trace: vf.Trace[SyntheticWorkspaceData, SyntheticWorkspaceState],
    ) -> float:
        return float(trace.state.evaluator_result.get("score", 0.0))

    @vf.metric
    async def workspace_metrics(
        self,
        trace: vf.Trace[SyntheticWorkspaceData, SyntheticWorkspaceState],
    ) -> dict[str, float]:
        result = trace.state.evaluator_result
        metrics = {
            "success": float(bool(result.get("success", False))),
            "changed_file_count": float(trace.state.changed_file_count),
            "final_file_count": float(trace.state.final_file_count),
        }
        for name, value in dict(result.get("subscores", {})).items():
            metrics[f"subscore/{name}"] = float(value)
        return metrics

    async def apply_gold_solution(self, runtime: vf.Runtime) -> None:
        workspace = await self._get_workspace()
        manifest = EnvironmentManifest.from_dict(workspace.manifest)
        solution_files = dict(manifest.reference_solution.get("files", {}))
        for relative_path, content in solution_files.items():
            path = _runtime_path(relative_path)
            parent = posixpath.dirname(path)
            result = await runtime.run(["mkdir", "-p", parent], {})
            if result.exit_code != 0:
                raise RuntimeError(f"failed to create gold-solution directory: {result.stderr}")
            await runtime.write(path, str(content).encode())

    async def _get_workspace(self) -> MaterializedWorkspace:
        if self._workspace is not None:
            return self._workspace
        if self._workspace_lock is None:
            self._workspace_lock = asyncio.Lock()
        async with self._workspace_lock:
            if self._workspace is None:
                self._workspace = await asyncio.to_thread(
                    _materialize,
                    self.data.family,
                    self.data.scenario,
                    self.data.difficulty,
                    self.data.seed,
                    self.data.split,
                    self.data.name or "swg-task",
                    False,
                )
        return self._workspace


class SyntheticWorkspaceTaskset(
    vf.Taskset[SyntheticWorkspaceTask, SyntheticWorkspaceTasksetConfig]
):
    def load(self) -> Iterable[SyntheticWorkspaceTask]:
        assignments = _load_assignments(self.config.manifest)
        selected_families = set(self.config.families)
        selected_difficulties = set(self.config.difficulties)
        selected_tasks = set(self.config.tasks)

        matched = 0
        for assignment in assignments:
            family = str(assignment["family"])
            difficulty = int(assignment["difficulty"])
            task_id = str(assignment["task_id"])
            if selected_families and family not in selected_families:
                continue
            if selected_difficulties and difficulty not in selected_difficulties:
                continue
            if selected_tasks and task_id not in selected_tasks:
                continue

            workspace = _materialize(
                family,
                str(assignment["scenario"]),
                difficulty,
                int(assignment["seed"]),
                str(assignment["split"]),
                task_id,
                self.config.validate_generation,
            )
            manifest = EnvironmentManifest.from_dict(workspace.manifest)
            provenance = dict(manifest.metadata.get("release_provenance", {}))
            data = SyntheticWorkspaceData(
                idx=matched,
                name=task_id,
                description=f"SWG {family} workspace task",
                prompt=_task_prompt(manifest),
                system_prompt=SYSTEM_PROMPT,
                image=self.config.image,
                workdir=WORKDIR,
                network_allow=[],
                network_block=["*"],
                timeout=vf.TaskTimeout(
                    setup=300,
                    agent=float(manifest.time_limit_seconds),
                    finalize=180,
                    scoring=30,
                ),
                resources=vf.TaskResources(cpu=2, memory=4, disk=4),
                family=family,
                scenario=str(assignment["scenario"]),
                difficulty=difficulty,
                seed=int(assignment["seed"]),
                split=str(assignment["split"]),
                generation_fingerprint=str(
                    provenance.get("generation_fingerprint", "")
                ),
            )
            matched += 1
            yield SyntheticWorkspaceTask(
                data,
                self.config.task,
                workspace=workspace,
            )

        if matched == 0:
            raise ValueError("no SWG tasks matched the requested manifest filters")


def _load_assignments(manifest_name: str) -> list[dict[str, Any]]:
    resource = files("synthetic_workspace_gym.frozen_manifests").joinpath(
        f"{manifest_name}.json"
    )
    if not resource.is_file():
        raise ValueError(f"unknown packaged SWG manifest: {manifest_name}")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    assignments = payload.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError(f"invalid SWG manifest: {manifest_name}")
    return [dict(row) for row in assignments]


def _materialize(
    family: str,
    scenario: str,
    difficulty: int,
    seed: int,
    split: str,
    task_id: str,
    validate: bool,
) -> MaterializedWorkspace:
    generator = get_generator(family)
    spec = generator.sample_spec(
        difficulty=difficulty,
        seed=seed,
        scenario_id=scenario,
        generation_params={"split": split, "task_id": task_id},
    )
    with tempfile.TemporaryDirectory(prefix="swg-v1-materialize-") as tmp:
        generated = generator.generate_instance(spec, Path(tmp), validate=validate)
        return MaterializedWorkspace(
            manifest=generated.manifest.to_dict(),
            visible_files=_read_tree(generated.visible_root),
            hidden_files=_read_tree(generated.hidden_root),
        )


def _task_prompt(manifest: EnvironmentManifest) -> str:
    descriptor = dict(manifest.metadata.get("task_descriptor", {}))
    lines = [manifest.instruction]
    required_output = next(
        (
            descriptor[key]
            for key in ("required_output_path", "output_path", "target_path")
            if descriptor.get(key)
        ),
        None,
    )
    if required_output:
        lines.extend(["", f"Required final artifact: {required_output}"])
    if descriptor.get("entrypoint"):
        lines.append(f"Visible check/entrypoint: {descriptor['entrypoint']}")
    return "\n".join(lines)


def _read_tree(root: Path) -> dict[str, bytes]:
    return {
        relative_path: (root / relative_path).read_bytes()
        for relative_path in list_relative_files(root)
    }


def _runtime_path(relative_path: str) -> str:
    pure = PurePosixPath(relative_path.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe SWG workspace path: {relative_path!r}")
    return f"{WORKDIR}/{pure.as_posix()}"


def _trusted_runtime_path(root: str, relative_path: str) -> str:
    pure = PurePosixPath(relative_path.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe trusted grader path: {relative_path!r}")
    return f"{root}/{pure.as_posix()}"


@lru_cache(maxsize=1)
def _grader_support_files() -> dict[str, bytes]:
    package_root = Path(__file__).parent
    payload = {"synthetic_workspace_gym/__init__.py": b""}
    for directory_name in ("evaluators", "schemas", "utils"):
        directory = package_root / directory_name
        for path in sorted(directory.rglob("*.py")):
            relative = path.relative_to(package_root).as_posix()
            payload[f"synthetic_workspace_gym/{relative}"] = path.read_bytes()
    return payload


def _content_digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


__all__ = [
    "SyntheticWorkspaceData",
    "SyntheticWorkspaceState",
    "SyntheticWorkspaceTask",
    "SyntheticWorkspaceTaskConfig",
    "SyntheticWorkspaceTaskset",
    "SyntheticWorkspaceTasksetConfig",
]
