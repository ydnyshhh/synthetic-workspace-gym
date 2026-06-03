from __future__ import annotations

import subprocess
import time
from pathlib import Path

from synthetic_workspace_gym.sandbox.errors import DockerUnavailableError
from synthetic_workspace_gym.sandbox.schemas import SandboxCommand, SandboxConfig, SandboxResult


class DockerSandboxBackend:
    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig(backend="docker")

    def is_available(self) -> bool:
        try:
            completed = subprocess.run(["docker", "version"], capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return completed.returncode == 0

    def run(
        self,
        command: SandboxCommand,
        workspace_path: Path,
        hidden_path: Path | None = None,
    ) -> SandboxResult:
        if not self.is_available():
            raise DockerUnavailableError("Docker is not available")

        docker_command = self._docker_command(command, Path(workspace_path).resolve(), hidden_path)
        timeout = command.timeout_seconds or self.config.timeout_seconds
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                docker_command,
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
                command=docker_command,
            )
        return SandboxResult(
            success=completed.returncode == 0,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            error=None if completed.returncode == 0 else "nonzero_exit",
            timed_out=False,
            duration_seconds=time.perf_counter() - started,
            command=docker_command,
        )

    def _docker_command(
        self,
        command: SandboxCommand,
        workspace_path: Path,
        hidden_path: Path | None,
    ) -> list[str]:
        network_enabled = self.config.network_enabled or command.allow_network
        docker_command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "bridge" if network_enabled else "none",
            "--cpus",
            str(self.config.cpus),
            "--memory",
            self.config.memory_limit,
            "--pids-limit",
            str(self.config.pids_limit),
            "--user",
            self.config.run_as_user,
            "--entrypoint",
            "",
            "--workdir",
            command.cwd or self.config.workdir,
            "--mount",
            f"type=bind,src={workspace_path},dst={self.config.workdir},rw",
            "--tmpfs",
            f"/tmp:size={self.config.tmpfs_size}",
            "--tmpfs",
            f"/home/swg:size={self.config.tmpfs_size}",
        ]
        if self.config.read_only_root:
            docker_command.append("--read-only")
            docker_command.extend(["--tmpfs", f"/var/tmp:size={self.config.tmpfs_size}"])
        if command.mode == "evaluator" and hidden_path is not None:
            docker_command.extend(
                [
                    "--mount",
                    f"type=bind,src={Path(hidden_path).resolve()},dst={self.config.hidden_dir},ro",
                ]
            )
        for key, value in command.env.items():
            docker_command.extend(["--env", f"{key}={value}"])
        docker_command.extend(self.config.extra_docker_args)
        docker_command.append(self.config.image)
        docker_command.extend(command.argv)
        return docker_command
