from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from test_support import workspace_tempdir

from synthetic_workspace_gym.sandbox import docker_available
from synthetic_workspace_gym.sandbox.schemas import SandboxConfig
from synthetic_workspace_gym.verifiers.env import SyntheticWorkspaceVerifiersEnv

IMAGE = "synthetic-workspace-gym-runtime:latest"


def docker_image_available() -> bool:
    if not docker_available():
        return False
    completed = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, text=True)
    return completed.returncode == 0


class VerifiersEnvTests(unittest.TestCase):
    def test_reset_returns_instruction_tools_and_task_metadata(self) -> None:
        with workspace_tempdir() as tmp_dir:
            env = _make_env(tmp_dir)
            try:
                observation = env.reset()

                self.assertTrue(observation["instruction"])
                self.assertTrue(observation["tools"])
                self.assertEqual(observation["task"]["family"], "script_repair")
            finally:
                env.close()

    def test_step_accepts_normalized_action(self) -> None:
        with workspace_tempdir() as tmp_dir:
            env = _make_env(tmp_dir)
            try:
                env.reset()
                result = env.step({"tool": "list_directory", "args": {"path": "."}})

                self.assertFalse(result["done"])
                self.assertIn("observation", result)
                self.assertTrue(result["info"]["success"])
            finally:
                env.close()

    def test_step_accepts_raw_json_completion(self) -> None:
        with workspace_tempdir() as tmp_dir:
            env = _make_env(tmp_dir)
            try:
                env.reset()
                result = env.step('{"tool":"list_directory","args":{"path":"."}}')

                self.assertFalse(result["done"])
                self.assertTrue(result["info"]["success"])
            finally:
                env.close()

    def test_evaluate_returns_normalized_reward(self) -> None:
        with workspace_tempdir() as tmp_dir:
            env = _make_env(tmp_dir)
            try:
                env.reset()
                payload = env.evaluate()

                self.assertIn("reward", payload)
                self.assertIn("score", payload)
                self.assertIn("success", payload)
            finally:
                env.close()

    def test_local_sandbox_submit_path_works(self) -> None:
        with workspace_tempdir() as tmp_dir:
            env = _make_env(tmp_dir, sandbox_backend="local")
            try:
                env.reset()
                result = env.step("done")

                self.assertTrue(result["done"])
                self.assertIn("reward_payload", result["info"])
            finally:
                env.close()

    @unittest.skipUnless(docker_image_available(), "Docker or SWG runtime image is unavailable")
    def test_docker_sandbox_mode_smoke(self) -> None:
        with workspace_tempdir() as tmp_dir:
            env = _make_env(
                tmp_dir,
                sandbox_backend="docker",
                sandbox_config=SandboxConfig(backend="docker", image=IMAGE),
            )
            try:
                observation = env.reset()
                self.assertEqual(observation["sandbox"]["backend"], "docker")
            finally:
                env.close()


def _make_env(tmp_dir: str, **overrides: object) -> SyntheticWorkspaceVerifiersEnv:
    kwargs: dict[str, object] = {
        "family": "script_repair",
        "scenario": "csv_schema_drift",
        "difficulty": 1,
        "seed": 7,
        "output_dir": Path(tmp_dir) / "runtime",
    }
    kwargs.update(overrides)
    return SyntheticWorkspaceVerifiersEnv(**kwargs)


if __name__ == "__main__":
    unittest.main()
