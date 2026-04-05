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
    def test_heuristic_baseline_solves_seeded_scenarios(self) -> None:
        seed_plan = {
            EnvironmentFamily.TABULAR: [41],
            EnvironmentFamily.SCRIPT_REPAIR: [1, 2, 3, 4, 5],
            EnvironmentFamily.PIPELINE: [1, 2, 3, 4],
        }
        for family, seeds in seed_plan.items():
            with workspace_tempdir() as tmp_dir:
                root = Path(tmp_dir)
                generator = get_generator(family)
                for seed in seeds:
                    with self.subTest(family=family.value, seed=seed):
                        spec = generator.sample_spec(difficulty=3, seed=seed)
                        bundle = generator.generate_instance(spec, root / "generated")
                        environment = load_environment(bundle.root)
                        runner = EpisodeRunner(output_root=root / "episodes")
                        summary = runner.run_episode(environment, HeuristicBaselineAgent())
                        self.assertTrue(summary.evaluation.success, bundle.manifest.metadata.get("scenario_id"))


if __name__ == "__main__":
    unittest.main()
