from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from synthetic_workspace_gym.analysis.artifacts import changed_files, compute_digest_from_hashes, snapshot_hashes
from synthetic_workspace_gym.runtime.policy import CommandPolicyError, resolve_python_script_command, validate_shell_command
from synthetic_workspace_gym.sandbox.base import SandboxBackend
from synthetic_workspace_gym.sandbox.schemas import SandboxCommand, SandboxConfig
from synthetic_workspace_gym.schemas import Action, ActionType, ToolObservation, ToolPermissions
from synthetic_workspace_gym.utils.paths import ensure_within_root, file_sha256


class WorkspaceToolExecutor:
    def __init__(
        self,
        workspace_root: Path,
        permissions: ToolPermissions,
        runtime_home: Path | None = None,
        sandbox_backend: SandboxBackend | None = None,
        sandbox_config: SandboxConfig | None = None,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.permissions = permissions
        self.runtime_home = (runtime_home.resolve() if runtime_home is not None else self.workspace_root / ".runtime-home")
        self.runtime_home.mkdir(parents=True, exist_ok=True)
        self.sandbox_backend = sandbox_backend
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.current_hashes = snapshot_hashes(self.workspace_root)

    @property
    def workspace_digest(self) -> str:
        return compute_digest_from_hashes(self.current_hashes)

    def execute(self, action: Action, *, remaining_time_seconds: float | None = None) -> ToolObservation:
        dispatch = {
            ActionType.READ_FILE: self.read_file,
            ActionType.WRITE_FILE: self.write_file,
            ActionType.APPEND_FILE: self.append_file,
            ActionType.LIST_DIRECTORY: self.list_directory,
            ActionType.RUN_SHELL: self.run_shell,
            ActionType.RUN_PYTHON: self.run_python,
            ActionType.SUBMIT: self.submit,
        }
        return dispatch[action.action_type](action.arguments, remaining_time_seconds=remaining_time_seconds)

    def read_file(self, arguments: dict[str, object], *, remaining_time_seconds: float | None = None) -> ToolObservation:
        if not self.permissions.read_file:
            return ToolObservation(
                success=False,
                message="read_file is disabled",
                error="permission_denied",
                workspace_digest=self.workspace_digest,
            )
        path = ensure_within_root(self.workspace_root, str(arguments["path"]))
        if not path.exists() or not path.is_file():
            return ToolObservation(
                success=False,
                message=f"File not found: {arguments['path']}",
                error="file_not_found",
                workspace_digest=self.workspace_digest,
            )
        return ToolObservation(
            success=True,
            message=f"Read {arguments['path']}",
            content=path.read_text(encoding="utf-8"),
            workspace_digest=self.workspace_digest,
        )

    def write_file(self, arguments: dict[str, object], *, remaining_time_seconds: float | None = None) -> ToolObservation:
        if not self.permissions.write_file:
            return ToolObservation(
                success=False,
                message="write_file is disabled",
                error="permission_denied",
                workspace_digest=self.workspace_digest,
            )
        relative_path = str(arguments["path"]).replace("\\", "/")
        path = ensure_within_root(self.workspace_root, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(arguments.get("content", "")), encoding="utf-8")
        self.current_hashes[relative_path] = file_sha256(path)
        return ToolObservation(
            success=True,
            message=f"Wrote {arguments['path']}",
            touched_files=[relative_path],
            workspace_digest=self.workspace_digest,
        )

    def append_file(self, arguments: dict[str, object], *, remaining_time_seconds: float | None = None) -> ToolObservation:
        if not self.permissions.append_file:
            return ToolObservation(
                success=False,
                message="append_file is disabled",
                error="permission_denied",
                workspace_digest=self.workspace_digest,
            )
        relative_path = str(arguments["path"]).replace("\\", "/")
        path = ensure_within_root(self.workspace_root, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(str(arguments.get("content", "")))
        self.current_hashes[relative_path] = file_sha256(path)
        return ToolObservation(
            success=True,
            message=f"Appended to {arguments['path']}",
            touched_files=[relative_path],
            workspace_digest=self.workspace_digest,
        )

    def list_directory(self, arguments: dict[str, object], *, remaining_time_seconds: float | None = None) -> ToolObservation:
        if not self.permissions.list_directory:
            return ToolObservation(
                success=False,
                message="list_directory is disabled",
                error="permission_denied",
                workspace_digest=self.workspace_digest,
            )
        requested = str(arguments.get("path", "."))
        path = ensure_within_root(self.workspace_root, requested)
        if not path.exists() or not path.is_dir():
            return ToolObservation(
                success=False,
                message=f"Directory not found: {requested}",
                error="directory_not_found",
                workspace_digest=self.workspace_digest,
            )
        listing = sorted(item.name + ("/" if item.is_dir() else "") for item in path.iterdir())
        return ToolObservation(
            success=True,
            message=f"Listed {requested}",
            listing=listing,
            content="\n".join(listing),
            workspace_digest=self.workspace_digest,
        )

    def run_shell(self, arguments: dict[str, object], *, remaining_time_seconds: float | None = None) -> ToolObservation:
        if not self.permissions.run_shell:
            return ToolObservation(
                success=False,
                message="run_shell is disabled",
                error="permission_denied",
                workspace_digest=self.workspace_digest,
            )
        command = str(arguments["command"])
        try:
            validate_shell_command(command)
        except CommandPolicyError as exc:
            return ToolObservation(
                success=False,
                message=str(exc),
                error="command_rejected",
                workspace_digest=self.workspace_digest,
            )
        timeout = self.remaining_timeout(self.permissions.shell_timeout_seconds, remaining_time_seconds)
        before = dict(self.current_hashes)
        if self.sandbox_backend is not None:
            result = self.sandbox_backend.run(
                SandboxCommand(
                    argv=self.sandbox_shell_command(command),
                    timeout_seconds=max(1, int(timeout)),
                    mode="tool",
                    env=self.sandbox_env(),
                ),
                self.workspace_root,
            )
            if result.timed_out:
                return ToolObservation(
                    success=False,
                    message=f"Shell command timed out: {command}",
                    stdout=result.stdout,
                    stderr=result.stderr,
                    error="timeout",
                    workspace_digest=self.workspace_digest,
                )
            stdout = result.stdout
            stderr = result.stderr
            returncode = result.returncode if result.returncode is not None else 1
        else:
            try:
                completed = subprocess.run(
                    self.shell_command(command),
                    cwd=str(self.workspace_root),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=self.subprocess_env(),
                )
            except subprocess.TimeoutExpired as exc:
                return ToolObservation(
                    success=False,
                    message=f"Shell command timed out: {command}",
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                    error="timeout",
                    workspace_digest=self.workspace_digest,
                )
            stdout = completed.stdout
            stderr = completed.stderr
            returncode = completed.returncode
        after = snapshot_hashes(self.workspace_root)
        self.current_hashes = after
        return ToolObservation(
            success=returncode == 0,
            message=f"Ran shell command: {command}",
            stdout=stdout,
            stderr=stderr,
            exit_code=returncode,
            touched_files=changed_files(before, after),
            workspace_digest=compute_digest_from_hashes(after),
        )

    def run_python(self, arguments: dict[str, object], *, remaining_time_seconds: float | None = None) -> ToolObservation:
        if not self.permissions.run_python:
            return ToolObservation(
                success=False,
                message="run_python is disabled",
                error="permission_denied",
                workspace_digest=self.workspace_digest,
            )
        payload = str(arguments.get("command_or_script") or arguments.get("command") or "")
        try:
            python_args = self.python_command(payload)
        except CommandPolicyError as exc:
            return ToolObservation(
                success=False,
                message=str(exc),
                error="command_rejected",
                workspace_digest=self.workspace_digest,
            )
        timeout = self.remaining_timeout(self.permissions.python_timeout_seconds, remaining_time_seconds)
        before = dict(self.current_hashes)
        if self.sandbox_backend is not None:
            sandbox_args = self.sandbox_python_command(python_args)
            result = self.sandbox_backend.run(
                SandboxCommand(
                    argv=sandbox_args,
                    timeout_seconds=max(1, int(timeout)),
                    mode="tool",
                    env=self.sandbox_env(),
                ),
                self.workspace_root,
            )
            if result.timed_out:
                return ToolObservation(
                    success=False,
                    message=f"Python command timed out: {payload}",
                    stdout=result.stdout,
                    stderr=result.stderr,
                    error="timeout",
                    workspace_digest=self.workspace_digest,
                )
            stdout = result.stdout
            stderr = result.stderr
            returncode = result.returncode if result.returncode is not None else 1
        else:
            try:
                completed = subprocess.run(
                    python_args,
                    cwd=str(self.workspace_root),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=self.subprocess_env(),
                )
            except subprocess.TimeoutExpired as exc:
                return ToolObservation(
                    success=False,
                    message=f"Python command timed out: {payload}",
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                    error="timeout",
                    workspace_digest=self.workspace_digest,
                )
            stdout = completed.stdout
            stderr = completed.stderr
            returncode = completed.returncode
        after = snapshot_hashes(self.workspace_root)
        self.current_hashes = after
        return ToolObservation(
            success=returncode == 0,
            message=f"Ran python command: {payload}",
            stdout=stdout,
            stderr=stderr,
            exit_code=returncode,
            touched_files=changed_files(before, after),
            workspace_digest=compute_digest_from_hashes(after),
        )

    def submit(self, arguments: dict[str, object], *, remaining_time_seconds: float | None = None) -> ToolObservation:
        if not self.permissions.submit:
            return ToolObservation(
                success=False,
                message="submit is disabled",
                error="permission_denied",
                workspace_digest=self.workspace_digest,
            )
        target = str(arguments.get("path_or_answer", ""))
        return ToolObservation(
            success=True,
            message=f"Submitted {target}",
            content=target,
            workspace_digest=self.workspace_digest,
        )

    def shell_command(self, command: str) -> list[str]:
        if os.name == "nt":
            return ["powershell", "-NoProfile", "-Command", command]
        return ["/bin/sh", "-c", command]

    def python_command(self, payload: str) -> list[str]:
        return [sys.executable, *resolve_python_script_command(self.workspace_root, payload)]

    def sandbox_shell_command(self, command: str) -> list[str]:
        if self.sandbox_config.backend == "docker":
            return ["/bin/sh", "-c", command]
        return self.shell_command(command)

    def sandbox_python_command(self, python_args: list[str]) -> list[str]:
        if self.sandbox_config.backend == "docker":
            return ["python", *python_args[1:]]
        return python_args

    def sandbox_env(self) -> dict[str, str]:
        env = self.subprocess_env()
        if self.sandbox_config.backend == "docker":
            env["HOME"] = "/home/swg"
            env["USERPROFILE"] = "/home/swg"
            env["TMP"] = "/tmp"
            env["TEMP"] = "/tmp"
            env["TMPDIR"] = "/tmp"
        return env

    def subprocess_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["HOME"] = str(self.runtime_home)
        env["USERPROFILE"] = str(self.runtime_home)
        env["TMP"] = str(self.runtime_home)
        env["TEMP"] = str(self.runtime_home)
        env["TMPDIR"] = str(self.runtime_home)
        return env

    def remaining_timeout(self, tool_timeout_seconds: int, remaining_time_seconds: float | None) -> float:
        if remaining_time_seconds is None:
            return float(tool_timeout_seconds)
        return max(0.1, min(float(tool_timeout_seconds), float(remaining_time_seconds)))
