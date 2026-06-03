from __future__ import annotations

import sys
import unittest
from pathlib import Path

from test_support import workspace_tempdir

from synthetic_workspace_gym.sandbox import SandboxCommand, SandboxConfig, build_sandbox_backend
from synthetic_workspace_gym.sandbox.local import LocalSandboxBackend


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


if __name__ == "__main__":
    unittest.main()
