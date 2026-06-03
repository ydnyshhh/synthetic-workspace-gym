from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from synthetic_workspace_gym.sandbox.schemas import SandboxCommand, SandboxConfig, SandboxResult


class LocalSandboxBackend:
    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()

    def is_available(self) -> bool:
        return True

    def run(
        self,
        command: SandboxCommand,
        workspace_path: Path,
        hidden_path: Path | None = None,
    ) -> SandboxResult:
        started = time.perf_counter()
        cwd = Path(command.cwd) if command.cwd else Path(workspace_path)
        env = dict(os.environ)
        env.update(command.env)
        timeout = command.timeout_seconds or self.config.timeout_seconds
        try:
            completed = subprocess.run(
                command.argv,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(
                success=False,
                returncode=None,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                error="timeout",
                timed_out=True,
                duration_seconds=time.perf_counter() - started,
                command=list(command.argv),
            )
        return SandboxResult(
            success=completed.returncode == 0,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            error=None if completed.returncode == 0 else "nonzero_exit",
            timed_out=False,
            duration_seconds=time.perf_counter() - started,
            command=list(command.argv),
        )
