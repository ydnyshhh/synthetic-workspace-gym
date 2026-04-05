from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

from synthetic_workspace_gym.utils.paths import ensure_within_root


class CommandPolicyError(ValueError):
    """Raised when a tool command violates the local runtime policy."""


_DISALLOWED_SUBSTRINGS = (
    "../",
    "..\\",
    "http://",
    "https://",
    "ftp://",
    "$HOME",
    "%USERPROFILE%",
    "~",
)
_DISALLOWED_EXECUTABLES = {
    "curl",
    "wget",
    "invoke-webrequest",
    "iwr",
    "irm",
    "scp",
    "ssh",
    "ftp",
    "nc",
    "netcat",
    "telnet",
}
_ABSOLUTE_WINDOWS_PATH = re.compile(r"(^|[\s'\"(])([A-Za-z]:[\\/])")
_ABSOLUTE_POSIX_PATH = re.compile(r"(^|[\s'\"(])/")
_PARENT_SEGMENT = re.compile(r"(^|[\s'\"(])\.\.([\\/]|$)")
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
_INLINE_PYTHON_FLAGS = {"-c", "-m"}


def validate_shell_command(command: str) -> None:
    normalized = command.lower()
    for token in _DISALLOWED_SUBSTRINGS:
        if token.lower() in normalized:
            raise CommandPolicyError(
                f"Shell command rejected because it references a disallowed path or network target: {token}"
            )
    if _PARENT_SEGMENT.search(command):
        raise CommandPolicyError("Shell command rejected because parent-directory traversal is not allowed.")
    if _ABSOLUTE_WINDOWS_PATH.search(command):
        raise CommandPolicyError("Shell command rejected because absolute filesystem paths are not allowed.")
    if _ABSOLUTE_POSIX_PATH.search(command):
        raise CommandPolicyError("Shell command rejected because absolute filesystem paths are not allowed.")
    tokens = _split_command(command)
    if tokens:
        assignment_tokens = _leading_env_assignments(tokens)
        if assignment_tokens:
            raise CommandPolicyError(
                "Shell command rejected because inline environment assignment is not allowed in the local runtime."
            )
        executable = Path(tokens[0]).name.lower()
        if executable in _DISALLOWED_EXECUTABLES:
            raise CommandPolicyError(
                f"Shell command rejected because '{tokens[0]}' is outside the allowed local-runtime policy."
            )
        if _is_python_executable(executable) and any(flag in _INLINE_PYTHON_FLAGS for flag in tokens[1:]):
            raise CommandPolicyError(
                "Shell command rejected because inline or module Python execution must use run_python with a workspace script."
            )
        if any(_looks_like_absolute_path(token) for token in tokens):
            raise CommandPolicyError("Shell command rejected because absolute filesystem paths are not allowed.")


def resolve_python_script_command(workspace_root: Path, payload: str) -> list[str]:
    stripped = payload.strip()
    if not stripped:
        raise CommandPolicyError("Python command rejected because it is empty.")
    if stripped.startswith("-m ") or stripped.startswith("-c "):
        raise CommandPolicyError("run_python only accepts workspace-relative script paths in v1.")
    if "\n" in stripped:
        raise CommandPolicyError("run_python inline code is disabled; pass a workspace-relative script path instead.")
    tokens = _split_command(stripped)
    if not tokens:
        raise CommandPolicyError("Python command rejected because it could not be parsed.")
    try:
        script_path = ensure_within_root(workspace_root, tokens[0])
    except ValueError as exc:
        raise CommandPolicyError(str(exc)) from exc
    if not script_path.exists() or not script_path.is_file():
        raise CommandPolicyError(f"Python script not found: {tokens[0]}")
    return [str(script_path), *tokens[1:]]


def _split_command(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=os.name != "nt")
    except ValueError as exc:
        raise CommandPolicyError(f"Command rejected because it could not be parsed: {exc}") from exc


def _leading_env_assignments(tokens: list[str]) -> list[str]:
    assignments: list[str] = []
    for token in tokens:
        if _ENV_ASSIGNMENT.match(token):
            assignments.append(token)
            continue
        break
    return assignments


def _looks_like_absolute_path(token: str) -> bool:
    stripped = token.strip().strip("'\"")
    return bool(stripped) and (
        bool(re.match(r"^[A-Za-z]:[\\/]", stripped))
        or stripped.startswith("/")
    )


def _is_python_executable(executable: str) -> bool:
    return executable in {"python", "python.exe", "python3", "python3.exe", "py", "py.exe"}
