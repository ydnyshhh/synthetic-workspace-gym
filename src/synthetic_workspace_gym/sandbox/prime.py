from __future__ import annotations

import os
import shlex
import shutil
import tarfile
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from synthetic_workspace_gym.sandbox.errors import SandboxExecutionError
from synthetic_workspace_gym.sandbox.schemas import SandboxCommand, SandboxConfig, SandboxResult


class PrimeSandboxBackend:
    """Run SWG commands in Prime remote sandboxes without exposing trusted assets."""

    def __init__(self, config: SandboxConfig | None = None, *, client: Any | None = None) -> None:
        self.config = config or SandboxConfig(backend="prime")
        self._client = client
        self._sandbox_id: str | None = None
        self._network_enabled: bool | None = None

    def is_available(self) -> bool:
        if self._client is not None:
            return True
        try:
            from prime_sandboxes import APIClient, SandboxClient  # noqa: F401
        except ImportError:
            return False
        return bool(APIClient().api_key)

    def close(self) -> None:
        if self._sandbox_id is None:
            return
        try:
            self._get_client().delete(self._sandbox_id)
        finally:
            self._sandbox_id = None

    def run(
        self,
        command: SandboxCommand,
        workspace_path: Path,
        hidden_path: Path | None = None,
    ) -> SandboxResult:
        workspace = Path(workspace_path).resolve()
        timeout = int(command.timeout_seconds or self.config.timeout_seconds)
        network_enabled = bool(self.config.network_enabled or command.allow_network)
        sandbox_id = self._ensure_sandbox(network_enabled)
        started = time.perf_counter()
        try:
            self._replace_remote_tree(sandbox_id, workspace, self.config.workdir, "workspace")
            if command.mode == "evaluator":
                if hidden_path is None:
                    raise SandboxExecutionError("Prime evaluator sandbox requires hidden assets")
                self._replace_remote_tree(
                    sandbox_id, Path(hidden_path).resolve(), self.config.hidden_dir, "hidden"
                )
                for source, destination in self._readonly_mounts():
                    self._replace_remote_tree(sandbox_id, source, destination, "readonly")
                self._upload_runtime(sandbox_id)

            env = dict(command.env)
            if command.mode == "evaluator":
                env["PYTHONPATH"] = "/opt/swg-runtime"
            response = self._get_client().execute_command(
                sandbox_id,
                shlex.join(command.argv),
                working_dir=command.cwd or self.config.workdir,
                env=env,
                timeout=timeout,
            )
            if command.mode == "tool":
                self._download_remote_tree(sandbox_id, self.config.workdir, workspace, timeout)
            return SandboxResult(
                success=response.exit_code == 0,
                returncode=response.exit_code,
                stdout=response.stdout,
                stderr=response.stderr,
                error=None if response.exit_code == 0 else "nonzero_exit",
                timed_out=False,
                duration_seconds=time.perf_counter() - started,
                command=list(command.argv),
            )
        except Exception as exc:
            if exc.__class__.__name__ == "CommandTimeoutError":
                return SandboxResult(
                    success=False,
                    returncode=None,
                    stdout="",
                    stderr=str(exc),
                    error="timeout",
                    timed_out=True,
                    duration_seconds=time.perf_counter() - started,
                    command=list(command.argv),
                )
            self.close()
            if isinstance(exc, SandboxExecutionError):
                raise
            raise SandboxExecutionError(f"Prime sandbox execution failed: {type(exc).__name__}: {exc}") from exc

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from prime_sandboxes import APIClient, SandboxClient
            except ImportError as exc:
                raise SandboxExecutionError(
                    "Prime sandbox backend requires the prime-sandboxes package"
                ) from exc
            self._client = SandboxClient(APIClient())
        return self._client

    def _ensure_sandbox(self, network_enabled: bool) -> str:
        if self._sandbox_id is not None:
            if self._network_enabled != network_enabled:
                self.close()
            else:
                return self._sandbox_id
        from prime_sandboxes import CreateSandboxRequest

        image = self.config.image
        if image == "synthetic-workspace-gym-runtime:latest":
            image = os.environ.get("SWG_PRIME_SANDBOX_IMAGE", "python:3.11-slim")
        request = CreateSandboxRequest(
            name=f"swg-{uuid.uuid4().hex[:12]}",
            docker_image=image,
            start_command="tail -f /dev/null",
            cpu_cores=self.config.cpus,
            memory_gb=_memory_gb(self.config.memory_limit),
            disk_size_gb=5.0,
            network_access=network_enabled,
            timeout_minutes=max(5, min(1440, (self.config.timeout_seconds + 299) // 60)),
            labels=["synthetic-workspace-gym"],
        )
        sandbox = self._get_client().create(request)
        self._get_client().wait_for_creation(sandbox.id, max_attempts=120)
        self._sandbox_id = sandbox.id
        self._network_enabled = network_enabled
        return sandbox.id

    def _replace_remote_tree(self, sandbox_id: str, source: Path, destination: str, label: str) -> None:
        if not source.is_dir():
            raise SandboxExecutionError(f"Prime sandbox {label} source is not a directory")
        with tempfile.TemporaryDirectory(prefix="swg-prime-upload-") as tmp_dir:
            archive = Path(tmp_dir) / f"{label}.tar.gz"
            _write_archive(source, archive)
            remote_archive = f"/tmp/swg-{label}-{uuid.uuid4().hex}.tar.gz"
            self._get_client().upload_file(sandbox_id, remote_archive, str(archive))
            script = (
                f"rm -rf {shlex.quote(destination)} && mkdir -p {shlex.quote(destination)} && "
                f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(destination)} && "
                f"rm -f {shlex.quote(remote_archive)}"
            )
            response = self._get_client().execute_command(sandbox_id, script, timeout=120)
            if response.exit_code != 0:
                raise SandboxExecutionError(f"Prime sandbox {label} upload extraction failed")

    def _download_remote_tree(
        self, sandbox_id: str, source: str, destination: Path, timeout: int
    ) -> None:
        remote_archive = f"/tmp/swg-workspace-out-{uuid.uuid4().hex}.tar.gz"
        response = self._get_client().execute_command(
            sandbox_id,
            f"tar -czf {shlex.quote(remote_archive)} -C {shlex.quote(source)} .",
            timeout=max(30, timeout),
        )
        if response.exit_code != 0:
            raise SandboxExecutionError("Prime sandbox workspace archive failed")
        with tempfile.TemporaryDirectory(prefix="swg-prime-download-") as tmp_dir:
            archive = Path(tmp_dir) / "workspace.tar.gz"
            extracted = Path(tmp_dir) / "extracted"
            extracted.mkdir()
            self._get_client().download_file(sandbox_id, remote_archive, str(archive))
            _extract_safe_archive(archive, extracted)
            _replace_local_tree(extracted, destination)

    def _upload_runtime(self, sandbox_id: str) -> None:
        import synthetic_workspace_gym

        package_root = Path(synthetic_workspace_gym.__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="swg-prime-runtime-") as tmp_dir:
            staging = Path(tmp_dir) / "synthetic_workspace_gym"
            shutil.copytree(package_root, staging, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (staging / "__init__.py").write_text(
                '"""Minimal evaluator runtime package."""\n', encoding="utf-8"
            )
            self._replace_remote_tree(sandbox_id, Path(tmp_dir), "/opt/swg-runtime", "runtime")

    def _readonly_mounts(self) -> list[tuple[Path, str]]:
        mounts: list[tuple[Path, str]] = []
        args = self.config.extra_docker_args
        index = 0
        while index < len(args):
            if args[index] != "--mount" or index + 1 >= len(args):
                raise SandboxExecutionError("Prime backend only accepts readonly --mount pairs")
            fields = args[index + 1].split(",")
            values = {field.split("=", 1)[0]: field.split("=", 1)[1] for field in fields if "=" in field}
            if values.get("type") != "bind" or "readonly" not in fields:
                raise SandboxExecutionError("Prime backend requires readonly bind mounts")
            destination = values.get("dst", "")
            if destination != "/environment":
                raise SandboxExecutionError("Prime backend only permits the /environment readonly mount")
            mounts.append((Path(values["src"]).resolve(), destination))
            index += 2
        return mounts


def _memory_gb(value: str) -> float:
    normalized = value.strip().lower()
    if normalized.endswith("g"):
        return float(normalized[:-1])
    if normalized.endswith("m"):
        return float(normalized[:-1]) / 1024.0
    raise ValueError(f"Unsupported memory limit: {value}")


def _write_archive(source: Path, archive: Path) -> None:
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(source.rglob("*")):
            handle.add(path, arcname=path.relative_to(source).as_posix(), recursive=False)


def _extract_safe_archive(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            member_path = PurePosixPath(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise SandboxExecutionError("Prime sandbox returned an unsafe archive path")
            if member.issym() or member.islnk() or member.isdev():
                raise SandboxExecutionError("Prime sandbox returned an unsafe archive member")
            target = (root / Path(*member_path.parts)).resolve()
            if target != root and root not in target.parents:
                raise SandboxExecutionError("Prime sandbox archive escapes the workspace")
        handle.extractall(root, members=members)


def _replace_local_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in destination.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)
