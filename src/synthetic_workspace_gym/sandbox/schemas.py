from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class SandboxConfig:
    backend: Literal["local", "docker"] = "local"
    image: str = "synthetic-workspace-gym-runtime:latest"
    network_enabled: bool = False
    memory_limit: str = "1g"
    cpus: float = 1.0
    pids_limit: int = 256
    timeout_seconds: int = 30
    workdir: str = "/workspace"
    hidden_dir: str = "/hidden"
    run_as_user: str = "1000:1000"
    read_only_root: bool = False
    tmpfs_size: str = "256m"
    extra_docker_args: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "image": self.image,
            "network_enabled": self.network_enabled,
            "memory_limit": self.memory_limit,
            "cpus": self.cpus,
            "pids_limit": self.pids_limit,
            "timeout_seconds": self.timeout_seconds,
            "workdir": self.workdir,
            "hidden_dir": self.hidden_dir,
            "run_as_user": self.run_as_user,
            "read_only_root": self.read_only_root,
            "tmpfs_size": self.tmpfs_size,
            "extra_docker_args": list(self.extra_docker_args),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SandboxConfig":
        return cls(**payload)


@dataclass(slots=True)
class SandboxCommand:
    argv: list[str]
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int | None = None
    allow_network: bool = False
    mode: Literal["tool", "evaluator"] = "tool"

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "cwd": self.cwd,
            "env": dict(self.env),
            "timeout_seconds": self.timeout_seconds,
            "allow_network": self.allow_network,
            "mode": self.mode,
        }


@dataclass(slots=True)
class SandboxResult:
    success: bool
    returncode: int | None
    stdout: str
    stderr: str
    error: str | None
    timed_out: bool
    duration_seconds: float
    command: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "timed_out": self.timed_out,
            "duration_seconds": self.duration_seconds,
            "command": list(self.command),
        }
