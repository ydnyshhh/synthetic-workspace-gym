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
    def test_heuristic_baseline_solves_explicit_scenarios(self) -> None:
        scenario_plan = {
            EnvironmentFamily.TABULAR: [
                "monthly_segment_report",
                "channel_status_pivot",
                "weekly_refund_rollup",
                "supplier_restock_summary",
            ],
            EnvironmentFamily.SCRIPT_REPAIR: [
                "inventory_report",
                "path_batch",
                "csv_schema_drift",
                "timestamp_normalization",
                "team_roster_export",
            ],
            EnvironmentFamily.PIPELINE: [
                "team_hours_pipeline",
                "sales_csv_pipeline",
                "artifact_stitch_pipeline",
                "quality_gate_pipeline",
            ],
        }
        for family, scenario_ids in scenario_plan.items():
            with workspace_tempdir() as tmp_dir:
                root = Path(tmp_dir)
                generator = get_generator(family)
                for scenario_id in scenario_ids:
                    with self.subTest(family=family.value, scenario_id=scenario_id):
                        spec = generator.sample_spec(difficulty=3, seed=41, scenario_id=scenario_id)
                        bundle = generator.generate_instance(spec, root / "generated")
                        environment = load_environment(bundle.root)
                        runner = EpisodeRunner(output_root=root / "episodes")
                        summary = runner.run_episode(environment, HeuristicBaselineAgent())
                        self.assertTrue(summary.evaluation.success, bundle.manifest.metadata.get("scenario_id"))


if __name__ == "__main__":
    unittest.main()
