from __future__ import annotations

import unittest
from pathlib import Path

from test_support import workspace_tempdir

from synthetic_workspace_gym.agents import HeuristicBaselineAgent
from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.runtime.environment import load_environment
from synthetic_workspace_gym.runtime.runner import EpisodeRunner
from synthetic_workspace_gym.schemas import EnvironmentFamily


class EndToEndRolloutTests(unittest.TestCase):
    def test_heuristic_baseline_solves_one_environment_per_family(self) -> None:
        for family in EnvironmentFamily:
            with self.subTest(family=family.value):
                with workspace_tempdir() as tmp_dir:
                    root = Path(tmp_dir)
                    generator = get_generator(family)
                    spec = generator.sample_spec(difficulty=3, seed=41)
                    bundle = generator.generate_instance(spec, root / "generated")
                    environment = load_environment(bundle.root)
                    runner = EpisodeRunner(output_root=root / "episodes")
                    summary = runner.run_episode(environment, HeuristicBaselineAgent())
                    self.assertTrue(summary.evaluation.success)


if __name__ == "__main__":
    unittest.main()
