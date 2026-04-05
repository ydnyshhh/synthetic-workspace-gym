from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path

from test_support import workspace_tempdir

from synthetic_workspace_gym.agents import ScriptedBaselineAgent
from synthetic_workspace_gym.agents.base import BaseAgent
from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.runtime.environment import load_environment
from synthetic_workspace_gym.runtime.runner import EpisodeRunner
from synthetic_workspace_gym.runtime.tools import WorkspaceToolExecutor
from synthetic_workspace_gym.schemas import Action, ActionType, ToolObservation, ToolPermissions, ToolState


class PermissionProbeAgent(BaseAgent):
    name = "permission-probe"

    def __init__(self) -> None:
        super().__init__()
        self.available_tools_seen: list[str] = []

    def act(self, observation: ToolObservation | dict[str, object], tool_state: ToolState) -> Action:
        self.available_tools_seen = list(tool_state.available_tools)
        return Action(ActionType.SUBMIT, {"path_or_answer": "done"})


class RuntimeTests(unittest.TestCase):
    def test_workspace_tool_executor_records_outputs_and_touched_files(self) -> None:
        with workspace_tempdir() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / "note.txt").write_text("hello\n", encoding="utf-8")
            (workspace / "make_answer.py").write_text(
                "from pathlib import Path\nPath('answer.txt').write_text('42', encoding='utf-8')\nprint('ok')\n",
                encoding="utf-8",
            )
            executor = WorkspaceToolExecutor(workspace, ToolPermissions())

            read_observation = executor.execute(Action(ActionType.READ_FILE, {"path": "note.txt"}))
            self.assertTrue(read_observation.success)
            self.assertEqual(read_observation.content, "hello\n")

            python_observation = executor.execute(
                Action(
                    ActionType.RUN_PYTHON,
                    {
                        "command_or_script": "make_answer.py"
                    },
                )
            )
            self.assertTrue(python_observation.success)
            self.assertIn("ok", python_observation.stdout)
            self.assertIn("answer.txt", python_observation.touched_files)
            self.assertTrue(python_observation.workspace_digest)

    def test_workspace_tool_executor_rejects_parent_traversal_commands(self) -> None:
        with workspace_tempdir() as tmp_dir:
            workspace = Path(tmp_dir)
            executor = WorkspaceToolExecutor(workspace, ToolPermissions())

            shell_observation = executor.execute(Action(ActionType.RUN_SHELL, {"command": "Get-ChildItem .."}))
            self.assertFalse(shell_observation.success)
            self.assertEqual(shell_observation.error, "command_rejected")

            python_observation = executor.execute(
                Action(ActionType.RUN_PYTHON, {"command_or_script": "../hidden/run_hidden_tests.py"})
            )
            self.assertFalse(python_observation.success)
            self.assertEqual(python_observation.error, "command_rejected")

    def test_workspace_tool_executor_composes_tool_timeout_with_remaining_episode_budget(self) -> None:
        with workspace_tempdir() as tmp_dir:
            workspace = Path(tmp_dir)
            executor = WorkspaceToolExecutor(
                workspace,
                ToolPermissions(run_python=False, shell_timeout_seconds=5),
            )
            sleep_command = "Start-Sleep -Seconds 1" if os.name == "nt" else "sleep 1"

            started = time.perf_counter()
            observation = executor.execute(
                Action(ActionType.RUN_SHELL, {"command": sleep_command}),
                remaining_time_seconds=0.2,
            )
            duration = time.perf_counter() - started
            self.assertFalse(observation.success)
            self.assertEqual(observation.error, "timeout")
            self.assertLess(duration, 2.0)

    def test_workspace_tool_executor_rejects_inline_python_in_shell(self) -> None:
        with workspace_tempdir() as tmp_dir:
            workspace = Path(tmp_dir)
            executor = WorkspaceToolExecutor(workspace, ToolPermissions(run_python=False))

            observation = executor.execute(
                Action(ActionType.RUN_SHELL, {"command": "python -c \"print(1)\""})
            )

            self.assertFalse(observation.success)
            self.assertEqual(observation.error, "command_rejected")

    def test_episode_runner_exports_artifacts(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            generator = get_generator("tabular")
            spec = generator.sample_spec(difficulty=3, seed=31)
            bundle = generator.generate_instance(spec, root / "generated")
            environment = load_environment(bundle.root)
            runner = EpisodeRunner(output_root=root / "episodes")
            summary = runner.run_episode(environment, ScriptedBaselineAgent())
            artifact_root = Path(summary.artifact_root)
            self.assertTrue(summary.evaluation.success)
            self.assertTrue((artifact_root / "trajectory.jsonl").exists())
            self.assertTrue((artifact_root / "evaluator_result.json").exists())
            self.assertTrue((artifact_root / "summary.json").exists())
            self.assertTrue((artifact_root / "final_diff.txt").exists())
            self.assertTrue((artifact_root / "final_workspace" / "outputs" / "report.json").exists())
            trajectory_lines = (artifact_root / "trajectory.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(trajectory_lines), summary.step_count)

    def test_episode_runner_reports_only_enabled_tools(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            generator = get_generator("tabular")
            spec = generator.sample_spec(difficulty=1, seed=9)
            spec.tool_permissions.run_shell = False
            spec.tool_permissions.run_python = False
            bundle = generator.generate_instance(spec, root / "generated")
            environment = load_environment(bundle.root)
            runner = EpisodeRunner(output_root=root / "episodes")
            agent = PermissionProbeAgent()
            summary = runner.run_episode(environment, agent)
            self.assertTrue(summary.submitted)
            self.assertNotIn("run_shell", agent.available_tools_seen)
            self.assertNotIn("run_python", agent.available_tools_seen)


if __name__ == "__main__":
    unittest.main()
