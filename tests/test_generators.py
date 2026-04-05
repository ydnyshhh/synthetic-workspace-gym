from __future__ import annotations

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
                def _build_environment(
                    self,
                    spec: EnvironmentSpec,
                    *,
                    root: Path,
                    visible_root: Path,
                    hidden_root: Path,
                ) -> GeneratedPayload:
                    raise NotImplementedError

    def test_script_and_pipeline_generators_cover_expected_base_scenarios(self) -> None:
        expected = {
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
        }
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            for family, scenario_ids in expected.items():
                generator = get_generator(family)
                seen: set[str] = set()
                for seed in range(1, len(scenario_ids) + 1):
                    spec = generator.sample_spec(difficulty=3, seed=seed)
                    bundle = generator.generate_instance(spec, root / family)
                    seen.add(str(bundle.manifest.metadata["scenario_id"]))
                self.assertEqual(seen, scenario_ids)


if __name__ == "__main__":
    unittest.main()
