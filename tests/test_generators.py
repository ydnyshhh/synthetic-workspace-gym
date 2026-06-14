from __future__ import annotations

import json
import unittest
from pathlib import Path

from test_support import workspace_tempdir

from synthetic_workspace_gym.generators.base import BaseGenerator, GeneratedPayload
from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.schemas import EnvironmentFamily, EnvironmentSpec


class GeneratorValidityTests(unittest.TestCase):
    def test_each_family_generates_structurally_valid_environment(self) -> None:
        for family in EnvironmentFamily:
            for difficulty in (1, 3, 5):
                with self.subTest(family=family.value, difficulty=difficulty):
                    with workspace_tempdir() as tmp_dir:
                        generator = get_generator(family)
                        spec = generator.sample_spec(difficulty=difficulty, seed=17)
                        bundle = generator.generate_instance(spec, Path(tmp_dir))
                        self.assertTrue((bundle.root / "manifest.json").exists())
                        self.assertTrue(bundle.manifest.visible_files)
                        self.assertTrue(bundle.manifest.hidden_files)
                        self.assertTrue(bundle.manifest.reference_solution["files"])
                        self.assertIn("complexity_profile", bundle.manifest.metadata)
                        for relative_path in bundle.manifest.visible_files:
                            self.assertTrue((bundle.visible_root / relative_path).exists())
                        for relative_path in bundle.manifest.hidden_files:
                            self.assertTrue((bundle.hidden_root / relative_path).exists())

    def test_generator_subclasses_must_define_family(self) -> None:
        with self.assertRaises(TypeError):
            class MissingFamilyGenerator(BaseGenerator):
                def build_environment(
                    self,
                    spec: EnvironmentSpec,
                    *,
                    root: Path,
                    visible_root: Path,
                    hidden_root: Path,
                ) -> GeneratedPayload:
                    raise NotImplementedError

    def test_generators_support_explicit_scenario_selection(self) -> None:
        expected = {
            "tabular": {
                "monthly_segment_report",
                "channel_status_pivot",
                "weekly_refund_rollup",
                "supplier_restock_summary",
            },
            "script_repair": {
                "inventory_report",
                "path_batch",
                "csv_schema_drift",
                "timestamp_normalization",
                "team_roster_export",
            },
            "pipeline": {
                "team_hours_pipeline",
                "sales_csv_pipeline",
                "artifact_stitch_pipeline",
                "quality_gate_pipeline",
            },
            "retrieval_workspace": {
                "service_config_reconciliation",
                "migration_plan_bundle",
                "incident_report_bundle",
                "client_adapter_sync",
            },
        }
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            for family, scenario_ids in expected.items():
                generator = get_generator(family)
                for scenario_id in scenario_ids:
                    spec = generator.sample_spec(difficulty=3, seed=99, scenario_id=scenario_id)
                    bundle = generator.generate_instance(spec, root / family)
                    self.assertEqual(str(bundle.manifest.metadata["scenario_id"]), scenario_id)
                    self.assertEqual(bundle.manifest.metadata["scenario_selection"]["selection_mode"], "explicit")

    def test_retrieval_workspace_metadata_contains_retrieval_fields(self) -> None:
        with workspace_tempdir() as tmp_dir:
            generator = get_generator("retrieval_workspace")
            spec = generator.sample_spec(
                difficulty=4,
                seed=31,
                scenario_id="service_config_reconciliation",
            )
            bundle = generator.generate_instance(spec, Path(tmp_dir))
            profile = dict(bundle.manifest.metadata["scenario_profile"])
            for key in (
                "task_type",
                "content_variant_id",
                "document_count",
                "retrieval_hops",
                "evidence_distribution",
                "distractor_count",
                "staleness_pattern",
                "output_style",
            ):
                self.assertIn(key, profile)
            document_files = [
                path
                for path in bundle.manifest.visible_files
                if path.startswith(("docs/", "notes/", "specs/", "logs/", "changelog/"))
            ]
            self.assertEqual(profile["document_count"], len(document_files))
            self.assertEqual(profile["staleness_pattern"], "stale_note")

    def test_retrieval_workspace_difficulty_increases_retrieval_complexity(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            generator = get_generator("retrieval_workspace")
            low = generator.generate_instance(
                generator.sample_spec(
                    difficulty=1,
                    seed=44,
                    scenario_id="service_config_reconciliation",
                ),
                root / "low",
            )
            high = generator.generate_instance(
                generator.sample_spec(
                    difficulty=5,
                    seed=44,
                    scenario_id="service_config_reconciliation",
                ),
                root / "high",
            )
            low_profile = dict(low.manifest.metadata["scenario_profile"])
            high_profile = dict(high.manifest.metadata["scenario_profile"])
            self.assertLess(int(low_profile["document_count"]), int(high_profile["document_count"]))
            self.assertLess(int(low_profile["retrieval_hops"]), int(high_profile["retrieval_hops"]))
            self.assertLess(int(low_profile["distractor_count"]), int(high_profile["distractor_count"]))
            self.assertEqual(low_profile["staleness_pattern"], "none")
            self.assertEqual(high_profile["staleness_pattern"], "superseded_changelog")

    def test_retrieval_workspace_seed_changes_core_fixture_outputs(self) -> None:
        scenarios = (
            "service_config_reconciliation",
            "migration_plan_bundle",
            "incident_report_bundle",
        )
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            generator = get_generator("retrieval_workspace")
            for scenario_id in scenarios:
                with self.subTest(scenario_id=scenario_id):
                    first = generator.generate_instance(
                        generator.sample_spec(difficulty=3, seed=1, scenario_id=scenario_id),
                        root / f"{scenario_id}-seed-1",
                    )
                    second = generator.generate_instance(
                        generator.sample_spec(difficulty=3, seed=2, scenario_id=scenario_id),
                        root / f"{scenario_id}-seed-2",
                    )
                    first_profile = dict(first.manifest.metadata["scenario_profile"])
                    second_profile = dict(second.manifest.metadata["scenario_profile"])
                    self.assertNotEqual(first_profile["content_variant_id"], second_profile["content_variant_id"])
                    self.assertNotEqual(
                        first.manifest.reference_solution["files"],
                        second.manifest.reference_solution["files"],
                    )

    def test_weekly_refund_contract_names_lowercase_region_normalization(self) -> None:
        with workspace_tempdir() as tmp_dir:
            generator = get_generator("tabular")
            spec = generator.sample_spec(difficulty=4, seed=102, scenario_id="weekly_refund_rollup")
            bundle = generator.generate_instance(spec, Path(tmp_dir))

            readme = (bundle.visible_root / "README.md").read_text(encoding="utf-8")
            task = (bundle.visible_root / "task.json").read_text(encoding="utf-8")
            expected_rows = json.loads(bundle.manifest.reference_solution["files"]["outputs/weekly_rollup.json"])
            expected_regions = {row["region"] for row in expected_rows}

            self.assertIn("lowercasing account lookup values", readme)
            self.assertIn("strip_lowercase", task)
            self.assertTrue(expected_regions)
            self.assertTrue(all(region == region.lower() for region in expected_regions))

    def test_migration_plan_schema_spec_path_matches_selected_schema(self) -> None:
        with workspace_tempdir() as tmp_dir:
            generator = get_generator("retrieval_workspace")
            for seed in (1, 2, 3):
                with self.subTest(seed=seed):
                    spec = generator.sample_spec(difficulty=5, seed=seed, scenario_id="migration_plan_bundle")
                    bundle = generator.generate_instance(spec, Path(tmp_dir) / f"seed-{seed}")
                    task = bundle.manifest.metadata["task_descriptor"]
                    schema_version = str(task["schema_version"])
                    schema_path = str(task["schema_spec_path"])
                    readme = (bundle.visible_root / "README.md").read_text(encoding="utf-8")

                    self.assertEqual(schema_path, f"specs/schema_{schema_version}.md")
                    self.assertTrue((bundle.visible_root / schema_path).exists())
                    self.assertIn(f"current schema `{schema_version}`", readme)
                    self.assertNotIn("current v3 schema", readme)

    def test_csv_schema_drift_readme_includes_repair_contract(self) -> None:
        with workspace_tempdir() as tmp_dir:
            generator = get_generator("script_repair")
            spec = generator.sample_spec(difficulty=2, seed=7, scenario_id="csv_schema_drift")
            bundle = generator.generate_instance(spec, Path(tmp_dir))
            readme = (bundle.visible_root / "README.md").read_text(encoding="utf-8")
            task = bundle.manifest.metadata["task_descriptor"]

            self.assertIn("## Expected behavior", readme)
            self.assertIn("account_id", readme)
            self.assertIn("sorted lexicographically by lowercase `region`", readme)
            self.assertIn("repair_contract", task)


if __name__ == "__main__":
    unittest.main()
