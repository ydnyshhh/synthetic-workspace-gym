from __future__ import annotations

from pathlib import Path
from typing import Protocol

from synthetic_workspace_gym.sandbox.schemas import SandboxCommand, SandboxResult


class SandboxBackend(Protocol):
    def run(
        self,
        command: SandboxCommand,
        workspace_path: Path,
        hidden_path: Path | None = None,
    ) -> SandboxResult:
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError
