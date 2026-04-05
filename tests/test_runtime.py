from __future__ import annotations

import json
import unittest
from pathlib import Path

from test_support import workspace_tempdir

from synthetic_workspace_gym.agents import ScriptedBaselineAgent
from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.runtime.environment import load_environment
from synthetic_workspace_gym.runtime.runner import EpisodeRunner
from synthetic_workspace_gym.runtime.tools import WorkspaceToolExecutor
from synthetic_workspace_gym.schemas import Action, ActionType, ToolPermissions


class RuntimeTests(unittest.TestCase):
    def test_workspace_tool_executor_records_outputs_and_touched_files(self) -> None:
        with workspace_tempdir() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / "note.txt").write_text("hello\n", encoding="utf-8")
            executor = WorkspaceToolExecutor(workspace, ToolPermissions())

            read_observation = executor.execute(Action(ActionType.READ_FILE, {"path": "note.txt"}))
            self.assertTrue(read_observation.success)
            self.assertEqual(read_observation.content, "hello\n")

            python_observation = executor.execute(
                Action(
                    ActionType.RUN_PYTHON,
                    {
                        "command_or_script": "from pathlib import Path\nPath('answer.txt').write_text('42', encoding='utf-8')\nprint('ok')"
                    },
                )
            )
            self.assertTrue(python_observation.success)
            self.assertIn("ok", python_observation.stdout)
            self.assertIn("answer.txt", python_observation.touched_files)

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


if __name__ == "__main__":
    unittest.main()
