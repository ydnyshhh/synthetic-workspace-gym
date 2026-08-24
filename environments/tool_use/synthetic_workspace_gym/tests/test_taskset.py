from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip(
    "verifiers.v1.configs.agent",
    reason="standalone package tests require Verifiers V1",
)

from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.trace import AgentInfo, Trace, TraceTask
from verifiers.v1.utils.loaders import (
    default_harness_id,
    environment_class,
    taskset_class,
)

from synthetic_workspace_gym import SyntheticWorkspaceTaskset as ExportedTaskset
from synthetic_workspace_gym.taskset import (
    SyntheticWorkspaceData,
    SyntheticWorkspaceState,
    SyntheticWorkspaceTask,
    SyntheticWorkspaceTaskset,
    SyntheticWorkspaceTasksetConfig,
    _runtime_path,
    _trusted_runtime_path,
)


def _single_task(**overrides) -> SyntheticWorkspaceTask:
    config = SyntheticWorkspaceTasksetConfig(
        id="synthetic-workspace-gym",
        tasks=["swg.sft_train.script_repair.path_batch.d2.s53"],
        **overrides,
    )
    return next(iter(SyntheticWorkspaceTaskset(config)))


def _first_family_task(manifest: str, family: str) -> SyntheticWorkspaceTask:
    config = SyntheticWorkspaceTasksetConfig(
        id="synthetic-workspace-gym",
        manifest=manifest,
        families=[family],
    )
    return next(iter(SyntheticWorkspaceTaskset(config)))


class LocalTestRuntime:
    def __init__(self, root: Path) -> None:
        self.root = root

    def local_path(self, path: str) -> Path:
        if path == "/workspace":
            return self.root / "workspace"
        if path.startswith("/workspace/"):
            return self.root / "workspace" / path.removeprefix("/workspace/")
        if path == "/tmp":
            return self.root / "tmp"
        if path.startswith("/tmp/"):
            return self.root / "tmp" / path.removeprefix("/tmp/")
        raise ValueError(f"unexpected test-runtime path: {path}")

    async def write(self, path: str, data: bytes) -> None:
        destination = self.local_path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    async def read(self, path: str, max_bytes: int | None = None) -> bytes:
        data = self.local_path(path).read_bytes()
        if max_bytes is not None and len(data) > max_bytes:
            raise ValueError("test-runtime read exceeded limit")
        return data

    async def run(self, argv: list[str], _env: dict[str, str]) -> SimpleNamespace:
        if argv[:2] == ["mkdir", "-p"]:
            for path in argv[2:]:
                self.local_path(path).mkdir(parents=True, exist_ok=True)
        elif argv[:2] == ["chmod", "+x"]:
            pass
        elif argv[:2] == ["sh", "-c"] and argv[3] == "swg-grader":
            command = [sys.executable, *(str(self.local_path(path)) for path in argv[4:])]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            return SimpleNamespace(
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        else:
            raise AssertionError(f"unexpected test-runtime command: {argv}")
        return SimpleNamespace(exit_code=0, stdout="", stderr="")


def test_taskset_materializes_native_typed_task() -> None:
    task = _single_task()

    assert isinstance(task, SyntheticWorkspaceTask)
    assert task.NEEDS_CONTAINER is True
    assert task.data.workdir == "/workspace"
    assert task.data.network_allow == []
    assert task.data.network_block == ["*"]
    assert task.data.family == "script_repair"
    assert task.data.generation_fingerprint
    assert "hidden_files" not in SyntheticWorkspaceData.model_fields
    assert "manifest" not in SyntheticWorkspaceData.model_fields
    assert "reference_solution" not in SyntheticWorkspaceData.model_fields
    assert SyntheticWorkspaceState.model_fields["evaluator_result"] is not None


def test_package_exports_exactly_one_v1_taskset() -> None:
    assert ExportedTaskset is SyntheticWorkspaceTaskset
    assert taskset_class("synthetic-workspace-gym") is SyntheticWorkspaceTaskset
    assert default_harness_id("synthetic-workspace-gym") == "bash"
    assert environment_class("synthetic-workspace-gym").__module__.startswith(
        "verifiers.v1"
    )


def test_filters_select_requested_family_and_difficulty() -> None:
    config = SyntheticWorkspaceTasksetConfig(
        id="synthetic-workspace-gym",
        manifest="eval-composite-heldout-24",
        families=["composite_workspace"],
        difficulties=[5],
    )

    task = next(iter(SyntheticWorkspaceTaskset(config)))

    assert task.data.family == "composite_workspace"
    assert task.data.difficulty == 5


@pytest.mark.parametrize(
    ("manifest_name", "family"),
    [
        ("train-all-family-seed-42", "tabular"),
        ("train-all-family-seed-42", "script_repair"),
        ("train-all-family-seed-42", "pipeline"),
        ("train-all-family-seed-42", "retrieval_workspace"),
        ("eval-composite-heldout-24", "composite_workspace"),
    ],
)
def test_each_family_gold_solution_scores_one(
    tmp_path: Path,
    manifest_name: str,
    family: str,
) -> None:
    task = _first_family_task(manifest_name, family)
    runtime = LocalTestRuntime(tmp_path)
    trace = Trace(
        task=TraceTask(type=type(task).__name__, data=task.data),
        agent=AgentInfo(config=AgentConfig()),
        state=SyntheticWorkspaceState(),
    )

    async def run_lifecycle() -> None:
        await task.setup(trace, runtime)
        await task.apply_gold_solution(runtime)
        await task.finalize(trace, runtime)
        await task.score(trace)

    asyncio.run(run_lifecycle())

    assert trace.reward == 1.0, trace.state.evaluator_result
    assert trace.rewards["workspace_score"].score == 1.0
    assert trace.metrics["success"] == 1.0
    assert trace.metrics["changed_file_count"] >= 1.0
    assert trace.info["synthetic_workspace_gym"]["success"] is True


def test_runtime_paths_cannot_escape_workspace() -> None:
    assert _runtime_path("src/app.py") == "/workspace/src/app.py"
    assert (
        _trusted_runtime_path("/tmp/grader", "hidden/test.py")
        == "/tmp/grader/hidden/test.py"
    )

    for unsafe in ("../hidden/test.py", "/etc/passwd", "a/../../b"):
        with pytest.raises(ValueError):
            _runtime_path(unsafe)
        with pytest.raises(ValueError):
            _trusted_runtime_path("/tmp/grader", unsafe)


def test_package_contains_all_training_and_eval_manifests() -> None:
    manifest_root = Path(__file__).parents[1] / "synthetic_workspace_gym" / "frozen_manifests"
    names = {path.stem for path in manifest_root.glob("*.json")}

    assert "train-all-family-seed-42" in names
    assert "train-specialist-script_repair" in names
    assert "eval-id-d3-d5" in names
    assert "eval-scenario-heldout" in names
    assert {
        "eval-d1-d2-panel-24",
        "eval-d3-d4-panel-24",
        "eval-d5-heldout-panel-24",
    } <= names
    assert {"sft-easy-v1", "sft-validation-v1", "rl-hard-v1", "rl-eval-v1"} <= names
