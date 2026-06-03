from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from test_support import workspace_tempdir

from synthetic_workspace_gym.sandbox import SandboxCommand, SandboxConfig, docker_available
from synthetic_workspace_gym.sandbox.docker import DockerSandboxBackend

IMAGE = "synthetic-workspace-gym-runtime:latest"


def docker_image_available() -> bool:
    if not docker_available():
        return False
    completed = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, text=True)
    return completed.returncode == 0


class DockerAvailabilityTests(unittest.TestCase):
    def test_docker_available_returns_bool(self) -> None:
        self.assertIsInstance(docker_available(), bool)

    def test_docker_command_disables_network_by_default(self) -> None:
        with workspace_tempdir() as tmp_dir:
            backend = DockerSandboxBackend(SandboxConfig(backend="docker", image=IMAGE))
            command = backend._docker_command(
                SandboxCommand(argv=["python", "--version"]),
                Path(tmp_dir),
                None,
            )

        self.assertIn("--network", command)
        self.assertEqual(command[command.index("--network") + 1], "none")


@unittest.skipUnless(docker_image_available(), "Docker or SWG runtime image is unavailable")
class DockerSandboxTests(unittest.TestCase):
    def test_docker_sandbox_runs_python_version(self) -> None:
        with workspace_tempdir() as tmp_dir:
            backend = DockerSandboxBackend(SandboxConfig(backend="docker", image=IMAGE, timeout_seconds=10))
            result = backend.run(SandboxCommand(argv=["python", "--version"]), Path(tmp_dir))

        self.assertTrue(result.success)
        self.assertIn("Python", result.stdout + result.stderr)

    def test_docker_sandbox_mounts_workspace_read_write(self) -> None:
        with workspace_tempdir() as tmp_dir:
            workspace = Path(tmp_dir)
            backend = DockerSandboxBackend(SandboxConfig(backend="docker", image=IMAGE, timeout_seconds=10))
            result = backend.run(
                SandboxCommand(argv=["python", "-c", "from pathlib import Path; Path('x.txt').write_text('ok')"]),
                workspace,
            )

        self.assertTrue(result.success)
        self.assertTrue((workspace / "x.txt").exists())

    def test_hidden_not_mounted_in_tool_mode_and_mounted_in_evaluator_mode(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "workspace"
            hidden = root / "hidden"
            workspace.mkdir()
            hidden.mkdir()
            backend = DockerSandboxBackend(SandboxConfig(backend="docker", image=IMAGE, timeout_seconds=10))
            tool_result = backend.run(
                SandboxCommand(argv=["/bin/sh", "-c", "test -d /hidden; echo $?"]),
                workspace,
                hidden_path=hidden,
            )
            evaluator_result = backend.run(
                SandboxCommand(argv=["/bin/sh", "-c", "test -d /hidden; echo $?"], mode="evaluator"),
                workspace,
                hidden_path=hidden,
            )

        self.assertTrue(tool_result.success)
        self.assertEqual(tool_result.stdout.strip(), "1")
        self.assertTrue(evaluator_result.success)
        self.assertEqual(evaluator_result.stdout.strip(), "0")

    def test_docker_timeout_returns_structured_failure(self) -> None:
        with workspace_tempdir() as tmp_dir:
            backend = DockerSandboxBackend(SandboxConfig(backend="docker", image=IMAGE, timeout_seconds=1))
            result = backend.run(
                SandboxCommand(argv=["python", "-c", "import time; time.sleep(5)"], timeout_seconds=1),
                Path(tmp_dir),
            )

        self.assertFalse(result.success)
        self.assertTrue(result.timed_out)


if __name__ == "__main__":
    unittest.main()
