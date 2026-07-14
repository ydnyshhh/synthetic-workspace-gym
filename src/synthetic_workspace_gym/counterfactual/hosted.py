from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from synthetic_workspace_gym.runtime.environment import load_environment
from synthetic_workspace_gym.schemas import utc_timestamp
from synthetic_workspace_gym.utils.io import read_json, write_json, write_jsonl

from .schemas import BranchTask

_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*$")
_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){2}(?:[A-Za-z0-9._-]+)?$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(slots=True)
class HostedPackageResult:
    output_dir: str
    package_name: str
    module_name: str
    pack_id: str
    pack_sha256: str
    source_swg_commit: str
    hosted_package_version: str
    task_count: int
    mode: str
    wheel_path: str
    wheel_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def package_hosted_branch_pack(
    branch_pack: Path,
    output_dir: Path,
    package_name: str,
    swg_ref: str,
    *,
    pack_id: str | None = None,
    version: str = "0.1.0",
    force: bool = False,
    build_wheel: bool = True,
    smoke_test: bool = True,
) -> HostedPackageResult:
    if smoke_test and not build_wheel:
        raise ValueError("installed-wheel smoke testing requires build_wheel=True")
    source = branch_pack.resolve()
    output = output_dir.resolve()
    module_name = _validate_package_identity(package_name, version, swg_ref)
    _validate_output_location(source, output)
    tasks = validate_branch_pack(source)
    pack_sha256 = hash_branch_pack(source)
    created_at = utc_timestamp()
    resolved_pack_id = (
        pack_id
        or f"{package_name}-{created_at[:10].replace('-', '')}-{pack_sha256[:12]}"
    )
    if not resolved_pack_id.strip():
        raise ValueError("pack_id must not be empty")

    _prepare_output(output, force=force)
    package_root = output / "src" / module_name
    copied_pack = package_root / "branch_pack"
    shutil.copytree(source, copied_pack)

    modes = sorted({task.mode for task in tasks})
    mode = modes[0] if len(modes) == 1 else "mixed"
    metadata = {
        "format_version": "1.0",
        "pack_id": resolved_pack_id,
        "pack_sha256": pack_sha256,
        "pack_hash_algorithm": "sha256-path-and-content-v1",
        "source_swg_commit": swg_ref,
        "hosted_package_version": version,
        "task_count": len(tasks),
        "mode": mode,
        "created_at": created_at,
        "private_assets_warning": "Branch packs contain trusted hidden evaluator assets; publish real evaluation packages privately.",
    }
    write_json(package_root / "hosted_metadata.json", metadata)
    _write_hosted_manifest(copied_pack, package_root / "hosted_manifest.jsonl", metadata)
    _write_generated_files(
        output=output,
        package_name=package_name,
        module_name=module_name,
        version=version,
        swg_ref=swg_ref,
        metadata=metadata,
    )

    wheel = Path()
    wheel_sha256 = ""
    if build_wheel:
        wheel = _build_wheel(output)
        inspect_hosted_wheel(wheel, copied_pack, module_name)
        wheel_sha256 = _hash_file(wheel)
        if smoke_test:
            _smoke_installed_wheel(wheel, module_name, tasks)

    result = HostedPackageResult(
        output_dir=str(output),
        package_name=package_name,
        module_name=module_name,
        pack_id=resolved_pack_id,
        pack_sha256=pack_sha256,
        source_swg_commit=swg_ref,
        hosted_package_version=version,
        task_count=len(tasks),
        mode=mode,
        wheel_path=str(wheel) if wheel else "",
        wheel_sha256=wheel_sha256,
    )
    write_json(output / "package-result.json", result.to_dict())
    return result


def validate_branch_pack(branch_pack: Path) -> list[BranchTask]:
    root = branch_pack.resolve()
    manifest_path = root / "manifest.jsonl"
    metadata_path = root / "metadata.json"
    if not root.is_dir():
        raise ValueError(f"branch pack does not exist: {root}")
    if not manifest_path.is_file():
        raise ValueError("branch pack is missing manifest.jsonl")
    if not metadata_path.is_file():
        raise ValueError("branch pack is missing metadata.json")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"branch pack cannot contain symbolic links: {path}")

    tasks: list[BranchTask] = []
    task_ids: set[str] = set()
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            task = BranchTask.from_dict(payload)
        except Exception as exc:
            raise ValueError(f"invalid branch manifest row {line_number}: {exc}") from exc
        if task.task_id in task_ids:
            raise ValueError(f"duplicate branch task_id: {task.task_id}")
        task_ids.add(task.task_id)
        environment_root = _resolve_pack_environment(root, payload.get("environment_path"), line_number)
        _validate_branch_environment(environment_root, task, line_number)
        tasks.append(task)
    if not tasks:
        raise ValueError("branch pack manifest contains no tasks")

    metadata = read_json(metadata_path)
    declared_count = metadata.get("task_count") if isinstance(metadata, dict) else None
    if declared_count is not None and int(declared_count) != len(tasks):
        raise ValueError(
            f"branch pack metadata task_count={declared_count} does not match manifest count={len(tasks)}"
        )
    return tasks


def hash_branch_pack(branch_pack: Path) -> str:
    root = branch_pack.resolve()
    digest = hashlib.sha256()
    files = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda path: path.relative_to(root).as_posix())
    for path in files:
        if path.is_symlink():
            raise ValueError(f"branch pack cannot contain symbolic links: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def inspect_hosted_wheel(wheel_path: Path, copied_pack: Path, module_name: str) -> None:
    expected = {
        f"{module_name}/branch_pack/{path.relative_to(copied_pack).as_posix()}"
        for path in copied_pack.rglob("*")
        if path.is_file()
    }
    expected.update({
        f"{module_name}/__init__.py",
        f"{module_name}/hosted_metadata.json",
        f"{module_name}/hosted_manifest.jsonl",
        f"{module_name}/environment.py",
    })
    with zipfile.ZipFile(wheel_path) as archive:
        names = set(archive.namelist())
    missing = sorted(expected - names)
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"hosted wheel is missing {len(missing)} required files: {preview}")


def _validate_package_identity(package_name: str, version: str, swg_ref: str) -> str:
    if not _PACKAGE_NAME_RE.fullmatch(package_name):
        raise ValueError("package_name must start with a letter and contain only letters, digits, '.', '_', or '-'")
    if not _VERSION_RE.fullmatch(version):
        raise ValueError("version must be a three-part Python package version")
    if not _COMMIT_RE.fullmatch(swg_ref):
        raise ValueError("swg_ref must be an exact 40-character Git commit SHA")
    return re.sub(r"[-.]+", "_", package_name).lower()


def _resolve_pack_environment(root: Path, raw_path: object, line_number: int) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"branch manifest row {line_number} has no environment_path")
    if "\\" in raw_path:
        raise ValueError(f"branch manifest row {line_number} environment_path must use POSIX separators")
    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"branch manifest row {line_number} environment_path escapes the branch pack")
    resolved = (root / Path(*relative.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"branch manifest row {line_number} environment_path escapes the branch pack") from exc
    if not resolved.is_dir():
        raise ValueError(f"branch manifest row {line_number} environment does not exist: {raw_path}")
    return resolved


def _validate_branch_environment(environment_root: Path, task: BranchTask, line_number: int) -> None:
    required = ("manifest.json", "branch.json", "prefix_messages.json")
    missing = [name for name in required if not (environment_root / name).is_file()]
    if missing:
        raise ValueError(f"branch environment for row {line_number} is missing: {', '.join(missing)}")
    loaded = load_environment(environment_root)
    for label, asset_root in (("visible", loaded.visible_root), ("hidden", loaded.hidden_root)):
        try:
            asset_root.resolve().relative_to(environment_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"branch environment for row {line_number} {label} root escapes its environment"
            ) from exc
    if not loaded.visible_root.is_dir():
        raise ValueError(f"branch environment for row {line_number} is missing visible workspace assets")
    if not loaded.hidden_root.is_dir() or not any(path.is_file() for path in loaded.hidden_root.rglob("*")):
        raise ValueError(f"branch environment for row {line_number} is missing hidden evaluator assets")
    hidden_paths = [PurePosixPath(relative) for relative in loaded.manifest.hidden_files]
    if any(path.is_absolute() or ".." in path.parts for path in hidden_paths):
        raise ValueError(f"branch environment for row {line_number} declares an unsafe hidden asset path")
    missing_hidden = [
        path.as_posix() for path in hidden_paths
        if not (loaded.hidden_root / Path(*path.parts)).is_file()
    ]
    if missing_hidden:
        raise ValueError(
            f"branch environment for row {line_number} is missing declared hidden assets: {missing_hidden}"
        )
    branch_payload = read_json(environment_root / "branch.json")
    if str(branch_payload.get("task_id")) != task.task_id:
        raise ValueError(f"branch environment for row {line_number} has a mismatched branch.json task_id")
    prefix_payload = read_json(environment_root / "prefix_messages.json")
    if prefix_payload != task.prefix_messages:
        raise ValueError(f"branch environment for row {line_number} prefix_messages.json does not match manifest")


def _write_hosted_manifest(copied_pack: Path, output: Path, metadata: dict[str, Any]) -> None:
    rows = []
    for line in (copied_pack / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["environment_path"] = (
            PurePosixPath("branch_pack") / PurePosixPath(str(row["environment_path"]))
        ).as_posix()
        row["metadata"] = {
            **dict(row.get("metadata") or {}),
            "pack_id": metadata["pack_id"],
            "pack_sha256": metadata["pack_sha256"],
            "source_swg_commit": metadata["source_swg_commit"],
            "hosted_package_version": metadata["hosted_package_version"],
            "wheel_sha256": "",
        }
        rows.append(row)
    write_jsonl(output, rows)


def _validate_output_location(source: Path, output: Path) -> None:
    if output.parent == output:
        raise ValueError("output_dir cannot be a filesystem root")
    for child, parent in ((output, source), (source, output)):
        try:
            child.relative_to(parent)
        except ValueError:
            continue
        raise ValueError("output_dir and branch_pack must not contain one another")


def _prepare_output(output: Path, *, force: bool) -> None:
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {output}")
        if any(output.iterdir()):
            if not force:
                raise FileExistsError(f"output directory is not empty: {output}; pass --force to replace it")
            shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def _write_generated_files(
    *,
    output: Path,
    package_name: str,
    module_name: str,
    version: str,
    swg_ref: str,
    metadata: dict[str, Any],
) -> None:
    package_root = output / "src" / module_name
    init_source = '''from __future__ import annotations

import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
HOSTED_METADATA = json.loads((PACKAGE_ROOT / "hosted_metadata.json").read_text(encoding="utf-8"))
PACK_ID = str(HOSTED_METADATA["pack_id"])
PACK_SHA256 = str(HOSTED_METADATA["pack_sha256"])
HOSTED_PACKAGE_VERSION = str(HOSTED_METADATA["hosted_package_version"])

from .environment import load_environment

__all__ = ["load_environment"]
'''
    environment_source = f'''from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

from synthetic_workspace_gym import load_environment as load_swg_environment
from {module_name} import HOSTED_PACKAGE_VERSION, PACK_ID, PACK_SHA256, PACKAGE_ROOT

BRANCH_MANIFEST = PACKAGE_ROOT / "hosted_manifest.jsonl"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{{64}}$")


def _runtime_manifest(wheel_sha256: str) -> Path:
    rows = []
    for line in BRANCH_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        relative_environment = Path(*Path(str(row["environment_path"])).parts)
        row["environment_path"] = str((PACKAGE_ROOT / relative_environment).resolve())
        row["metadata"] = {{
            **dict(row.get("metadata") or {{}}),
            "pack_id": PACK_ID,
            "pack_sha256": PACK_SHA256,
            "wheel_sha256": wheel_sha256.lower(),
            "source_swg_commit": str(row.get("metadata", {{}}).get("source_swg_commit", "")),
            "hosted_package_version": HOSTED_PACKAGE_VERSION,
        }}
        rows.append(row)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="swg-hosted-manifest-", suffix=".jsonl", delete=False,
    )
    with handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\\n")
    return Path(handle.name)


def load_environment(
    branch_task_id: str | None = None,
    branch_mode: str | None = None,
    max_examples: int = -1,
    sandbox_backend: str = "docker",
    wheel_sha256: str | None = None,
    sample_strategy: str = "first",
    shuffle: bool = False,
    shuffle_seed: int = 0,
    **kwargs,
):
    if sandbox_backend != "docker":
        raise ValueError(
            "Hosted counterfactual packs require isolated Docker tool execution; "
            "local or unknown backends could expose trusted evaluator assets installed on the host."
        )
    if not isinstance(wheel_sha256, str) or not _SHA256_RE.fullmatch(wheel_sha256):
        raise ValueError(
            "wheel_sha256 must be the exact 64-character SHA-256 of the installed hosted wheel."
        )
    if "branch_manifest_path" in kwargs:
        raise TypeError("This generated environment owns branch_manifest_path.")
    runtime_manifest = _runtime_manifest(wheel_sha256)
    try:
        return load_swg_environment(
            branch_manifest_path=str(runtime_manifest),
            branch_task_id=branch_task_id,
            branch_mode=branch_mode,
            max_examples=max_examples,
            sandbox_backend=sandbox_backend,
            sample_strategy=sample_strategy,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            **kwargs,
        )
    finally:
        runtime_manifest.unlink(missing_ok=True)
'''
    pyproject = f'''[build-system]
requires = ["hatchling>=1.27.0"]
build-backend = "hatchling.build"

[project]
name = "{package_name}"
version = "{version}"
description = "Immutable hosted SWG counterfactual branch pack"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "synthetic-workspace-gym @ git+https://github.com/ydnyshhh/synthetic-workspace-gym.git@{swg_ref}",
]

[tool.hatch.metadata]
allow-direct-references = true

[tool.hatch.build.targets.wheel]
packages = ["src/{module_name}"]
'''
    readme = f'''# {package_name}

Generated immutable Environment Hub package for SWG counterfactual pack `{metadata["pack_id"]}`.

- Pack SHA-256: `{metadata["pack_sha256"]}`
- SWG commit: `{swg_ref}`
- Tasks: {metadata["task_count"]}
- Mode: {metadata["mode"]}

This package contains trusted files under `branch_pack/environments/*/hidden/`.
Publish serious evaluation packages privately:

```bash
prime env push --visibility PRIVATE
```

Do not update an environment version after collecting results; generate a new package version and pack ID.
'''
    (package_root / "__init__.py").write_text(init_source, encoding="utf-8")
    (package_root / "environment.py").write_text(environment_source, encoding="utf-8")
    (output / "environment.py").write_text(
        f"from {module_name} import load_environment\n\n__all__ = [\"load_environment\"]\n",
        encoding="utf-8",
    )
    (output / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    (output / "README.md").write_text(readme, encoding="utf-8")




def _representative_task_ids(tasks: list[BranchTask]) -> list[str]:
    selectors = (
        lambda task: task.mode == "forced" and bool(task.forced_action) and task.forced_action.get("tool") != "submit",
        lambda task: task.mode == "forced" and bool(task.forced_action) and task.forced_action.get("tool") == "submit",
        lambda task: task.mode == "open",
    )
    selected: list[str] = []
    for selector in selectors:
        match = next((task.task_id for task in tasks if selector(task)), None)
        if match is not None and match not in selected:
            selected.append(match)
    if not selected:
        selected.append(tasks[0].task_id)
    return selected


def _smoke_installed_wheel(wheel: Path, module_name: str, tasks: list[BranchTask]) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to smoke-test an installed hosted wheel")
    wheel_sha256 = _hash_file(wheel)
    representative_ids = _representative_task_ids(tasks)
    docker_smoke = False
    if shutil.which("docker") is not None:
        try:
            docker_smoke = subprocess.run(
                ["docker", "image", "inspect", "synthetic-workspace-gym-runtime:latest"],
                check=False, capture_output=True, text=True, timeout=10,
            ).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            docker_smoke = False
    with tempfile.TemporaryDirectory(prefix="swg-hosted-wheel-smoke-") as temp_dir:
        root = Path(temp_dir)
        venv = root / "venv"
        command = [uv, "venv", "--python", sys.executable, str(venv)]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"installed-wheel smoke setup failed: {' '.join(command)}\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
        venv_python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        install = subprocess.run(
            [uv, "pip", "install", "--python", str(venv_python), str(wheel)],
            check=False,
            capture_output=True,
            text=True,
        )
        if install.returncode != 0:
            raise RuntimeError(
                "installed-wheel dependency installation failed; the pinned SWG reference "
                f"or generated wheel is not installable:\n{install.stdout}\n{install.stderr}"
            )
        smoke_source = f"""
import asyncio
import json
from pathlib import Path
from {module_name} import load_environment

try:
    load_environment(sandbox_backend="local", wheel_sha256={wheel_sha256!r})
except ValueError as exc:
    assert "isolated Docker" in str(exc)
else:
    raise AssertionError("hosted package accepted local tool execution")

for task_id in json.loads({json.dumps(representative_ids)!r}):
    env = load_environment(
        branch_task_id=task_id,
        max_examples=1,
        sandbox_backend="docker",
        wheel_sha256={wheel_sha256!r},
    )
    rows = env.get_dataset()
    row = dict(rows[0])
    assert row["task_id"] == task_id
    assert Path(row["environment_path"]).is_dir()
    metadata = dict(row.get("metadata") or {{}})
    assert metadata["wheel_sha256"] == {wheel_sha256!r}
    assert metadata["pack_id"]
    assert metadata["pack_sha256"]
    assert metadata["source_swg_commit"]
    assert metadata["hosted_package_version"]

    if {docker_smoke!r}:
        async def exercise_installed_task():
            state = {{"input": row, "trajectory_id": f"installed-smoke-{{task_id}}"}}
            await env.setup_state(state)
            try:
                forced = row.get("branch_mode") == "forced"
                assert state["swg_env"]._step_count == (1 if forced else 0)
                assert state.get("trajectory", []) == []
                assert state["swg_policy_start_message_index"] == len(state["prompt"])
                if forced:
                    forced_result = dict(state["swg_forced_action_result"])
                    assert "Tool execution failed" not in str(forced_result.get("observation", ""))
                    terminal = row.get("forced_action", {{}}).get("tool") == "submit"
                    if not terminal:
                        assert forced_result["success"] is True
                    assert state["swg_forced_prefix_length"] == 2
                    assert state["swg_loss_mask_metadata"]["exclude_forced_messages"] is True
                    assert ("final_env_response" in state) is terminal
                    if terminal:
                        assert state["swg_reward_payload"] is not None
                else:
                    assert state["swg_forced_prefix_length"] == 0
            finally:
                state["swg_env"].close()
        asyncio.run(exercise_installed_task())
"""
        smoke = subprocess.run(
            [str(venv_python), "-c", smoke_source],
            check=False,
            capture_output=True,
            text=True,
        )
        if smoke.returncode != 0:
            raise RuntimeError(
                f"installed hosted wheel smoke test failed:\n{smoke.stdout}\n{smoke.stderr}"
            )


def _build_wheel(output: Path) -> Path:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to build a hosted package wheel")
    completed = subprocess.run(
        [uv, "build", "--wheel", "--out-dir", "dist"],
        cwd=output,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"uv build failed:\n{completed.stdout}\n{completed.stderr}")
    wheels = sorted((output / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one hosted wheel, found {len(wheels)}")
    return wheels[0]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
