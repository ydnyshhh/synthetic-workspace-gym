from __future__ import annotations

import importlib
import json
import subprocess
import sys
import shutil
import zipfile
from pathlib import Path

import pytest

from test_support import workspace_tempdir

from synthetic_workspace_gym.cli import build_parser
from synthetic_workspace_gym.counterfactual.hosted import (
    _representative_task_ids,
    hash_branch_pack,
    inspect_hosted_wheel,
    package_hosted_branch_pack,
    validate_branch_pack,
)
from synthetic_workspace_gym.runtime.environment import load_environment
from synthetic_workspace_gym.sandbox import docker_available
from synthetic_workspace_gym.sandbox.docker import DockerSandboxBackend
from synthetic_workspace_gym.sandbox.schemas import SandboxCommand, SandboxConfig
from synthetic_workspace_gym.utils.io import read_json

SWG_REF = "df0e0462de3c2c006ba4a4db69785e60ec8cccc4"


def _demo_pack() -> Path:
    return Path(__file__).parents[1] / "examples" / "counterfactual" / "demo-pack"


def _copy_pack(root: Path) -> Path:
    target = root / "branch-pack"
    shutil.copytree(_demo_pack(), target)
    return target


def test_package_hosted_generates_self_contained_smoke_package() -> None:
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        source = _copy_pack(root)
        output = root / "swg-counterfactual-pilot"
        result = package_hosted_branch_pack(
            source,
            output,
            "swg-counterfactual-pilot",
            SWG_REF,
            pack_id="swg-cf-pilot-test",
            build_wheel=False,
            smoke_test=False,
        )

        assert result.task_count == len(validate_branch_pack(source))
        assert result.pack_sha256 == hash_branch_pack(source)
        assert len(result.pack_sha256) == 64
        assert (output / "environment.py").is_file()
        assert (output / "pyproject.toml").is_file()
        copied = output / "src" / "swg_counterfactual_pilot" / "branch_pack"
        assert (copied / "manifest.jsonl").is_file()
        assert hash_branch_pack(copied) == result.pack_sha256
        metadata = read_json(output / "src" / "swg_counterfactual_pilot" / "hosted_metadata.json")
        assert metadata["pack_id"] == "swg-cf-pilot-test"
        assert metadata["pack_sha256"] == result.pack_sha256
        assert metadata["source_swg_commit"] == SWG_REF
        assert "visibility PRIVATE" in (output / "README.md").read_text(encoding="utf-8")


def test_hosted_pack_rejects_path_escape_and_missing_hidden_assets() -> None:
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        traversal_pack = _copy_pack(root / "traversal")
        manifest = traversal_pack / "manifest.jsonl"
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows[0]["environment_path"] = "../../outside"
        manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="escapes the branch pack"):
            validate_branch_pack(traversal_pack)

        missing_pack = _copy_pack(root / "missing")
        first = json.loads((missing_pack / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
        environment_root = missing_pack / Path(*Path(first["environment_path"]).parts)
        loaded = load_environment(environment_root)
        shutil.rmtree(loaded.hidden_root)
        with pytest.raises(ValueError, match="missing hidden evaluator assets"):
            validate_branch_pack(missing_pack)


def test_hosted_wheel_inspection_requires_every_pack_file() -> None:
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        source = _copy_pack(root)
        output = root / "generated"
        package_hosted_branch_pack(
            source,
            output,
            "swg-counterfactual-wheel-test",
            SWG_REF,
            build_wheel=False,
            smoke_test=False,
        )
        module_name = "swg_counterfactual_wheel_test"
        package_root = output / "src" / module_name
        wheel = root / "complete.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            for path in package_root.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(output / "src").as_posix())
        inspect_hosted_wheel(wheel, package_root / "branch_pack", module_name)

        incomplete = root / "incomplete.whl"
        with zipfile.ZipFile(incomplete, "w") as archive:
            archive.writestr(f"{module_name}/__init__.py", "")
            archive.writestr(f"{module_name}/hosted_metadata.json", "{}")
        with pytest.raises(RuntimeError, match="missing"):
            inspect_hosted_wheel(incomplete, package_root / "branch_pack", module_name)


def test_package_hosted_cli_contract() -> None:
    args = build_parser().parse_args([
        "counterfactual",
        "package-hosted",
        "--branch-pack", "artifacts/pilot-pack",
        "--output-dir", "dist/swg-counterfactual-pilot",
        "--package-name", "swg-counterfactual-pilot",
        "--swg-ref", SWG_REF,
        "--pack-id", "swg-cf-pilot-test",
    ])
    assert args.counterfactual_command == "package-hosted"
    assert args.package_name == "swg-counterfactual-pilot"
    assert args.swg_ref == SWG_REF


def test_hosted_package_refuses_nonempty_output_without_force() -> None:
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        source = _copy_pack(root)
        output = root / "existing"
        output.mkdir()
        (output / "keep.txt").write_text("user data", encoding="utf-8")
        with pytest.raises(FileExistsError, match="--force"):
            package_hosted_branch_pack(
                source,
                output,
                "swg-counterfactual-existing-test",
                SWG_REF,
                build_wheel=False,
                smoke_test=False,
            )
        assert (output / "keep.txt").read_text(encoding="utf-8") == "user data"



def test_hosted_package_rejects_nested_source_and_output_paths() -> None:
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        source = _copy_pack(root)
        with pytest.raises(ValueError, match="must not contain one another"):
            package_hosted_branch_pack(
                source,
                source / "generated",
                "swg-counterfactual-nested-test",
                SWG_REF,
                build_wheel=False,
                smoke_test=False,
            )


def test_branch_pack_hash_changes_when_content_changes() -> None:
    with workspace_tempdir() as tmp_dir:
        source = _copy_pack(Path(tmp_dir))
        original = hash_branch_pack(source)
        branch_file = next(source.glob("environments/*/branch.json"))
        branch_file.write_text(branch_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        assert hash_branch_pack(source) != original



def _import_generated_environment(output: Path, module_name: str):
    sys.path.insert(0, str(output / "src"))
    try:
        package = importlib.import_module(f"{module_name}.environment")
        return package
    finally:
        sys.path.pop(0)


def test_generated_hosted_loader_fails_closed_and_attests_wheel() -> None:
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        source = _copy_pack(root)
        output = root / "generated"
        result = package_hosted_branch_pack(
            source,
            output,
            "swg-counterfactual-secure-test",
            SWG_REF,
            build_wheel=False,
            smoke_test=False,
        )
        module = _import_generated_environment(output, result.module_name)
        wheel_sha256 = "a" * 64

        with pytest.raises(ValueError, match="isolated Docker"):
            module.load_environment(sandbox_backend="local", wheel_sha256=wheel_sha256)
        with pytest.raises(ValueError, match="wheel_sha256"):
            module.load_environment(sandbox_backend="docker")

        env = module.load_environment(sandbox_backend="docker", wheel_sha256=wheel_sha256)
        rows = [dict(row) for row in env.get_dataset()]
        assert rows
        for row in rows:
            metadata = dict(row["metadata"])
            assert metadata["pack_id"] == result.pack_id
            assert metadata["pack_sha256"] == result.pack_sha256
            assert metadata["wheel_sha256"] == wheel_sha256
            assert metadata["source_swg_commit"] == SWG_REF
            assert metadata["hosted_package_version"] == "0.1.0"
            assert Path(row["environment_path"]).is_dir()

        assert result.pack_id.endswith(f"-{result.pack_sha256[:12]}")


def test_representative_branch_selection_covers_forced_terminal_and_open() -> None:
    with workspace_tempdir() as tmp_dir:
        source = _copy_pack(Path(tmp_dir))
        tasks = validate_branch_pack(source)
        tasks[0].mode = "open"
        tasks[0].forced_action = None
        selected = _representative_task_ids(tasks)
        selected_tasks = [next(task for task in tasks if task.task_id == task_id) for task_id in selected]
        assert any(task.mode == "forced" and task.forced_action["tool"] != "submit" for task in selected_tasks)
        assert any(task.mode == "forced" and task.forced_action["tool"] == "submit" for task in selected_tasks)
        assert any(task.mode == "open" for task in selected_tasks)


def test_tool_sandbox_mount_policy_excludes_host_package_and_hidden_assets() -> None:
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        workspace = root / "visible"
        hidden = root / "installed-package" / "hidden"
        workspace.mkdir()
        hidden.mkdir(parents=True)
        backend = DockerSandboxBackend(SandboxConfig(backend="docker"))
        command = backend._docker_command(
            SandboxCommand(argv=["python", "probe.py"], mode="tool"),
            workspace.resolve(),
            hidden.resolve(),
        )
        rendered = "\n".join(command)
        assert str(workspace.resolve()) in rendered
        assert str(hidden.resolve()) not in rendered
        assert "dst=/hidden" not in rendered
        assert ",rw" not in rendered
        assert "synthetic_workspace_gym.sandbox.evaluator_entrypoint" not in rendered

        evaluator_command = backend._docker_command(
            SandboxCommand(argv=["python", "evaluator.py"], mode="evaluator"),
            workspace.resolve(),
            hidden.resolve(),
        )
        evaluator_rendered = "\n".join(evaluator_command)
        assert f"src={hidden.resolve()},dst=/hidden,readonly" in evaluator_rendered
        assert ",ro" not in evaluator_rendered


def _docker_runtime_available() -> bool:
    if not docker_available():
        return False
    completed = subprocess.run(
        ["docker", "image", "inspect", "synthetic-workspace-gym-runtime:latest"],
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


@pytest.mark.skipif(not _docker_runtime_available(), reason="Docker or SWG runtime image is unavailable")
def test_adversarial_probe_cannot_read_hidden_canary_from_tool_sandbox() -> None:
    canary = "SWG_HIDDEN_CANARY_7e6bfba1300e"
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        workspace = root / "visible"
        hidden = root / "installed-hosted-package" / "branch_pack" / "environment" / "hidden"
        workspace.mkdir()
        hidden.mkdir(parents=True)
        (hidden / "canary.txt").write_text(canary, encoding="utf-8")
        probe = workspace / "probe.py"
        probe.write_text(
            """
import importlib
import os
import pathlib
import pkgutil
import site
import subprocess
import sys

output = []
try:
    package = importlib.import_module("swg_counterfactual_pilot")
    output.extend(str(path) for path in package.PACKAGE_ROOT.rglob("*"))
except Exception as exc:
    output.append(f"import_failed:{type(exc).__name__}")
output.extend(module.name for module in pkgutil.iter_modules() if "counterfactual" in module.name)
output.extend(site.getsitepackages())
output.extend(sys.path)
for proc_path in (pathlib.Path("/proc/self/cwd"), pathlib.Path("/proc/self/environ")):
    try:
        output.append(proc_path.read_text(errors="ignore"))
    except Exception as exc:
        output.append(f"proc_failed:{type(exc).__name__}")
for root in (pathlib.Path("/workspace"), pathlib.Path("/usr/local/lib")):
    try:
        for path in root.rglob("*"):
            if "hidden" in path.parts and path.is_file():
                output.append(path.read_text(errors="ignore"))
    except Exception as exc:
        output.append(f"scan_failed:{type(exc).__name__}")
try:
    find = subprocess.run(
        ["find", "/workspace", "/usr/local/lib", "-type", "f", "-path", "*/hidden/*"],
        capture_output=True,
        text=True,
    )
    output.append(find.stdout)
except Exception as exc:
    output.append(f"find_failed:{type(exc).__name__}")
output.append(os.environ.get("PYTHONPATH", ""))
print("\\n".join(output))
""".strip()
            + "\n",
            encoding="utf-8",
        )
        backend = DockerSandboxBackend(
            SandboxConfig(backend="docker", image="synthetic-workspace-gym-runtime:latest")
        )
        result = backend.run(
            SandboxCommand(argv=["python", "probe.py"], mode="tool"),
            workspace,
            hidden_path=hidden,
        )
        assert result.success, result.stderr
        assert canary not in result.stdout
        assert str(hidden.resolve()) not in result.stdout
        assert "import_failed:" in result.stdout
