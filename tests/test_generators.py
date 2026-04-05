from __future__ import annotations

import unittest
from pathlib import Path

from test_support import workspace_tempdir

from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.schemas import EnvironmentFamily


class GeneratorValidityTests(unittest.TestCase):
    def test_each_family_generates_structurally_valid_environment(self) -> None:
        for family in EnvironmentFamily:
            with self.subTest(family=family.value):
                with workspace_tempdir() as tmp_dir:
                    generator = get_generator(family)
                    spec = generator.sample_spec(difficulty=3, seed=17)
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


if __name__ == "__main__":
    unittest.main()
