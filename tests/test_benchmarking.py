from __future__ import annotations

import unittest
from pathlib import Path

from test_support import workspace_tempdir

from synthetic_workspace_gym.agents import HeuristicBaselineAgent
from synthetic_workspace_gym.analysis.benchmarking import build_benchmark_report, episode_to_row
from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.runtime.environment import load_environment
from synthetic_workspace_gym.runtime.runner import EpisodeRunner


class BenchmarkAnalysisTests(unittest.TestCase):
    def build_rows(self) -> list[dict[str, object]]:
        plan = [
            ("tabular", "monthly_segment_report", 2, 101),
            ("tabular", "weekly_refund_rollup", 4, 102),
            ("script_repair", "csv_schema_drift", 3, 103),
            ("pipeline", "quality_gate_pipeline", 5, 104),
            ("retrieval_workspace", "service_config_reconciliation", 4, 105),
        ]
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            runner = EpisodeRunner(output_root=root / "episodes")
            rows: list[dict[str, object]] = []
            for family, scenario_id, difficulty, seed in plan:
                generator = get_generator(family)
                spec = generator.sample_spec(difficulty=difficulty, seed=seed, scenario_id=scenario_id)
                bundle = generator.generate_instance(spec, root / "generated")
                environment = load_environment(bundle.root)
                summary = runner.run_episode(environment, HeuristicBaselineAgent())
                rows.append(episode_to_row(summary, environment.manifest))
            return rows

    def test_grouped_report_contains_expected_sections(self) -> None:
        report = build_benchmark_report(self.build_rows())
        self.assertIn("overall", report)
        self.assertIn("by_family", report)
        self.assertIn("by_difficulty", report)
        self.assertIn("by_scenario_id", report)
        self.assertIn("by_family_and_difficulty", report)
        self.assertIn("by_content_variant_id", report)
        self.assertIn("by_document_count", report)
        self.assertIn("by_retrieval_hops", report)
        self.assertIn("by_evidence_distribution", report)
        self.assertIn("by_staleness_pattern", report)
        self.assertTrue(report["by_document_count"])
        self.assertTrue(report["by_content_variant_id"])
        self.assertIn("4", report["by_retrieval_hops"])
        self.assertIn("stale_note", report["by_staleness_pattern"])

    def test_scenario_grouping_is_populated(self) -> None:
        report = build_benchmark_report(self.build_rows())
        self.assertIn("monthly_segment_report", report["by_scenario_id"])
        self.assertIn("csv_schema_drift", report["by_scenario_id"])
        self.assertIn("quality_gate_pipeline", report["by_scenario_id"])
        self.assertIn("service_config_reconciliation", report["by_scenario_id"])
        self.assertGreaterEqual(len(report["by_scenario_id"]), 5)

    def test_difficulty_grouping_works(self) -> None:
        report = build_benchmark_report(self.build_rows())
        self.assertIn("2", report["by_difficulty"])
        self.assertIn("3", report["by_difficulty"])
        self.assertIn("4", report["by_difficulty"])
        self.assertIn("5", report["by_difficulty"])

    def test_difficulty_five_realization_is_exported(self) -> None:
        row = next(item for item in self.build_rows() if item["difficulty"] == 5)
        self.assertEqual(row["difficulty_guidance"], "none")
        self.assertEqual(row["difficulty_hint_count"], 0)
        self.assertTrue(row["difficulty_discovery_required"])
        self.assertGreater(int(row["difficulty_candidate_file_count"]), 0)
        self.assertGreater(int(row["difficulty_applied_bug_count"]), 0)
        self.assertGreater(int(row["difficulty_touched_file_count"]), 0)

    def test_bucket_metrics_are_numerically_sane(self) -> None:
        report = build_benchmark_report(self.build_rows())
        buckets = [report["overall"]]
        for section_name in (
            "by_family",
            "by_difficulty",
            "by_scenario_id",
            "by_family_and_difficulty",
            "by_failure_mode",
            "by_retrieval_hops",
        ):
            buckets.extend(report[section_name].values())
        for bucket in buckets:
            self.assertGreater(bucket["count"], 0)
            self.assertGreaterEqual(bucket["success_rate"], 0.0)
            self.assertLessEqual(bucket["success_rate"], 1.0)
            self.assertGreaterEqual(bucket["mean_score"], 0.0)
            self.assertLessEqual(bucket["mean_score"], 1.0)
            self.assertGreaterEqual(bucket["perfect_rate"], 0.0)
            self.assertLessEqual(bucket["perfect_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
