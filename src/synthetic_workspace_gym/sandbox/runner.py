from __future__ import annotations

import subprocess
from typing import Any

from synthetic_workspace_gym.sandbox.base import SandboxBackend
from synthetic_workspace_gym.sandbox.docker import DockerSandboxBackend
from synthetic_workspace_gym.sandbox.local import LocalSandboxBackend
from synthetic_workspace_gym.sandbox.prime import PrimeSandboxBackend
from synthetic_workspace_gym.sandbox.schemas import SandboxConfig


def build_sandbox_backend(config: SandboxConfig) -> SandboxBackend:
    if config.backend == "local":
        return LocalSandboxBackend(config)
    if config.backend == "docker":
        return DockerSandboxBackend(config)
    if config.backend == "prime":
        return PrimeSandboxBackend(config)
    raise ValueError(f"Unsupported sandbox backend: {config.backend}")


def docker_available() -> bool:
    try:
        completed = subprocess.run(["docker", "version"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def build_default_sandbox_config(backend: str = "local", **kwargs: Any) -> SandboxConfig:
    return SandboxConfig(backend=backend, **kwargs)
