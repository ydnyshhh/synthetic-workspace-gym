from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from synthetic_workspace_gym.analysis.artifacts import changed_files, snapshot_hashes
from synthetic_workspace_gym.schemas import Action, ActionType, ToolObservation, ToolPermissions
from synthetic_workspace_gym.utils.paths import ensure_within_root


class WorkspaceToolExecutor:
    def __init__(self, workspace_root: Path, permissions: ToolPermissions) -> None:
        self.workspace_root = workspace_root.resolve()
        self.permissions = permissions

    def execute(self, action: Action) -> ToolObservation:
        dispatch = {
            ActionType.READ_FILE: self._read_file,
            ActionType.WRITE_FILE: self._write_file,
            ActionType.APPEND_FILE: self._append_file,
            ActionType.LIST_DIRECTORY: self._list_directory,
            ActionType.RUN_SHELL: self._run_shell,
            ActionType.RUN_PYTHON: self._run_python,
            ActionType.SUBMIT: self._submit,
        }
        return dispatch[action.action_type](action.arguments)

    def _read_file(self, arguments: dict[str, object]) -> ToolObservation:
        if not self.permissions.read_file:
            return ToolObservation(success=False, message="read_file is disabled", error="permission_denied")
        path = ensure_within_root(self.workspace_root, str(arguments["path"]))
        if not path.exists() or not path.is_file():
            return ToolObservation(success=False, message=f"File not found: {arguments['path']}", error="file_not_found")
        return ToolObservation(
            success=True,
            message=f"Read {arguments['path']}",
            content=path.read_text(encoding="utf-8"),
        )

    def _write_file(self, arguments: dict[str, object]) -> ToolObservation:
        if not self.permissions.write_file:
            return ToolObservation(success=False, message="write_file is disabled", error="permission_denied")
        path = ensure_within_root(self.workspace_root, str(arguments["path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(arguments.get("content", "")), encoding="utf-8")
        return ToolObservation(
            success=True,
            message=f"Wrote {arguments['path']}",
            touched_files=[str(arguments["path"]).replace("\\", "/")],
        )

    def _append_file(self, arguments: dict[str, object]) -> ToolObservation:
        if not self.permissions.append_file:
            return ToolObservation(success=False, message="append_file is disabled", error="permission_denied")
        path = ensure_within_root(self.workspace_root, str(arguments["path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(str(arguments.get("content", "")))
        return ToolObservation(
            success=True,
            message=f"Appended to {arguments['path']}",
            touched_files=[str(arguments["path"]).replace("\\", "/")],
        )

    def _list_directory(self, arguments: dict[str, object]) -> ToolObservation:
        if not self.permissions.list_directory:
            return ToolObservation(success=False, message="list_directory is disabled", error="permission_denied")
        requested = str(arguments.get("path", "."))
        path = ensure_within_root(self.workspace_root, requested)
        if not path.exists() or not path.is_dir():
            return ToolObservation(success=False, message=f"Directory not found: {requested}", error="directory_not_found")
        listing = sorted(item.name + ("/" if item.is_dir() else "") for item in path.iterdir())
        return ToolObservation(
            success=True,
            message=f"Listed {requested}",
            listing=listing,
            content="\n".join(listing),
        )

    def _run_shell(self, arguments: dict[str, object]) -> ToolObservation:
        if not self.permissions.run_shell:
            return ToolObservation(success=False, message="run_shell is disabled", error="permission_denied")
        command = str(arguments["command"])
        before = snapshot_hashes(self.workspace_root)
        try:
            completed = subprocess.run(
                self._shell_command(command),
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=self.permissions.shell_timeout_seconds,
                env=self._subprocess_env(),
            )
        except subprocess.TimeoutExpired as exc:
            return ToolObservation(
                success=False,
                message=f"Shell command timed out: {command}",
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                error="timeout",
            )
        after = snapshot_hashes(self.workspace_root)
        return ToolObservation(
            success=completed.returncode == 0,
            message=f"Ran shell command: {command}",
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
            touched_files=changed_files(before, after),
        )

    def _run_python(self, arguments: dict[str, object]) -> ToolObservation:
        if not self.permissions.run_python:
            return ToolObservation(success=False, message="run_python is disabled", error="permission_denied")
        payload = str(arguments.get("command_or_script") or arguments.get("command") or "")
        before = snapshot_hashes(self.workspace_root)
        try:
            completed = subprocess.run(
                self._python_command(payload),
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=self.permissions.python_timeout_seconds,
                env=self._subprocess_env(),
            )
        except subprocess.TimeoutExpired as exc:
            return ToolObservation(
                success=False,
                message=f"Python command timed out: {payload}",
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                error="timeout",
            )
        after = snapshot_hashes(self.workspace_root)
        return ToolObservation(
            success=completed.returncode == 0,
            message=f"Ran python command: {payload}",
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
            touched_files=changed_files(before, after),
        )

    def _submit(self, arguments: dict[str, object]) -> ToolObservation:
        if not self.permissions.submit:
            return ToolObservation(success=False, message="submit is disabled", error="permission_denied")
        target = str(arguments.get("path_or_answer", ""))
        return ToolObservation(success=True, message=f"Submitted {target}", content=target)

    def _shell_command(self, command: str) -> list[str]:
        if os.name == "nt":
            return ["powershell", "-NoProfile", "-Command", command]
        return ["/bin/sh", "-lc", command]

    def _python_command(self, payload: str) -> list[str]:
        candidate = self.workspace_root / payload
        if payload and candidate.exists() and candidate.is_file():
            return [sys.executable, str(candidate)]
        if payload.startswith("-m "):
            return [sys.executable, *shlex.split(payload, posix=os.name != "nt")]
        return [sys.executable, "-c", payload]

    def _subprocess_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env
