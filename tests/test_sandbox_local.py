from __future__ import annotations

import os
import sys
import unittest
from unittest import mock
from pathlib import Path

from test_support import workspace_tempdir

from synthetic_workspace_gym.runtime.tools import WorkspaceToolExecutor
from synthetic_workspace_gym.sandbox import SandboxCommand, SandboxConfig, build_sandbox_backend
from synthetic_workspace_gym.sandbox.local import LocalSandboxBackend
from synthetic_workspace_gym.sandbox.schemas import SandboxResult
from synthetic_workspace_gym.schemas import ToolPermissions


class LocalSandboxTests(unittest.TestCase):
    def test_local_sandbox_runs_simple_command(self) -> None:
        with workspace_tempdir() as tmp_dir:
            backend = LocalSandboxBackend(SandboxConfig(timeout_seconds=5))
            result = backend.run(
                SandboxCommand(argv=[sys.executable, "-c", "print('ok')"]),
                Path(tmp_dir),
            )

        self.assertTrue(result.success)
        self.assertIn("ok", result.stdout)
        self.assertFalse(result.timed_out)

    def test_local_sandbox_timeout_returns_structured_result(self) -> None:
        with workspace_tempdir() as tmp_dir:
            backend = LocalSandboxBackend(SandboxConfig(timeout_seconds=1))
            result = backend.run(
                SandboxCommand(argv=[sys.executable, "-c", "import time; time.sleep(2)"], timeout_seconds=1),
                Path(tmp_dir),
            )

        self.assertFalse(result.success)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.error, "timeout")

    def test_sandbox_config_serializes(self) -> None:
        config = SandboxConfig(backend="docker", image="image:test", network_enabled=True)

        payload = config.to_dict()
        restored = SandboxConfig.from_dict(payload)

        self.assertEqual(restored.backend, "docker")
        self.assertEqual(restored.image, "image:test")
        self.assertTrue(restored.network_enabled)

    def test_build_sandbox_backend_local(self) -> None:
        backend = build_sandbox_backend(SandboxConfig(backend="local"))

        self.assertIsInstance(backend, LocalSandboxBackend)

    def test_docker_sandbox_env_does_not_inherit_host_environment(self) -> None:
        with workspace_tempdir() as tmp_dir:
            executor = WorkspaceToolExecutor(
                Path(tmp_dir),
                ToolPermissions(),
                sandbox_config=SandboxConfig(backend="docker"),
            )
            with mock.patch.dict(os.environ, {"SWG_SECRET_TEST_TOKEN": "secret-value"}):
                env = executor.sandbox_env()

        self.assertNotIn("SWG_SECRET_TEST_TOKEN", env)
        self.assertEqual(env["HOME"], "/home/swg")
        self.assertEqual(env["PATH"], "/usr/local/bin:/usr/bin:/bin")

    def test_public_sandbox_result_redacts_bind_mount_sources(self) -> None:
        result = SandboxResult(
            success=False,
            returncode=1,
            stdout="",
            stderr="",
            error="nonzero_exit",
            timed_out=False,
            duration_seconds=0.1,
            command=[
                "docker",
                "run",
                "--mount",
                "type=bind,src=C:/tmp/workspace,dst=/workspace,rw",
                "--mount",
                "type=bind,src=C:/tmp/hidden,dst=/hidden,ro",
            ],
        )

        payload = result.to_public_dict()

        self.assertIn("type=bind,src=<workspace_mount>,dst=/workspace,rw", payload["command"])
        self.assertIn("type=bind,src=<hidden_mount>,dst=/hidden,ro", payload["command"])
        self.assertNotIn("C:/tmp/workspace", " ".join(payload["command"]))


if __name__ == "__main__":
    unittest.main()
