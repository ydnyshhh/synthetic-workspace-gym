from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from test_support import workspace_tempdir

from synthetic_workspace_gym.evaluators.registry import get_evaluator
from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.schemas import EnvironmentFamily


class EvaluatorCorrectnessTests(unittest.TestCase):
    def test_unsolved_workspace_fails_for_each_family(self) -> None:
        for family in EnvironmentFamily:
            with self.subTest(family=family.value):
                with workspace_tempdir() as tmp_dir:
                    generator = get_generator(family)
                    spec = generator.sample_spec(difficulty=2, seed=21)
                    bundle = generator.generate_instance(spec, Path(tmp_dir))
                    evaluator = get_evaluator(family)
                    result = evaluator.evaluate(bundle.visible_root, bundle.manifest, bundle.hidden_root)
                    self.assertFalse(result.success)

    def test_reference_solution_passes_evaluator(self) -> None:
        for family in EnvironmentFamily:
            with self.subTest(family=family.value):
                with workspace_tempdir() as tmp_dir:
                    generator = get_generator(family)
                    spec = generator.sample_spec(difficulty=3, seed=23)
                    bundle = generator.generate_instance(spec, Path(tmp_dir))
                    evaluator = get_evaluator(family)
                    solved_workspace = Path(tmp_dir) / "solved"
                    shutil.copytree(bundle.visible_root, solved_workspace)
                    for relative_path, content in bundle.manifest.reference_solution["files"].items():
                        path = solved_workspace / relative_path
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(content, encoding="utf-8")
                    result = evaluator.evaluate(solved_workspace, bundle.manifest, bundle.hidden_root)
                    self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()
