from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from test_support import workspace_tempdir

from synthetic_workspace_gym.evaluators.registry import get_evaluator
from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.schemas import EnvironmentFamily


class EvaluatorCorrectnessTests(unittest.TestCase):
    def test_evaluator_entrypoint_is_loadable(self) -> None:
        evaluator = get_evaluator(
            EnvironmentFamily.TABULAR,
            evaluator_entrypoint="synthetic_workspace_gym.evaluators.tabular:TabularEvaluator",
        )
        self.assertEqual(type(evaluator).__name__, "TabularEvaluator")

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

    def test_tabular_evaluator_exposes_partial_credit(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            generator = get_generator("tabular")
            spec = generator.sample_spec(difficulty=3, seed=55, scenario_id="monthly_segment_report")
            bundle = generator.generate_instance(spec, root / "generated")
            evaluator = get_evaluator(bundle.manifest.family)
            expected = json.loads((bundle.hidden_root / "expected_output.json").read_text(encoding="utf-8"))
            partial_workspace = root / "partial"
            shutil.copytree(bundle.visible_root, partial_workspace)
            output_path = partial_workspace / "outputs" / "report.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(expected[:1], indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result = evaluator.evaluate(partial_workspace, bundle.manifest, bundle.hidden_root)
            self.assertFalse(result.success)
            self.assertGreater(result.score, 0.0)
            self.assertLess(result.score, 1.0)

    def test_tabular_evaluator_rejects_wrong_json_shape(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            generator = get_generator("tabular")
            spec = generator.sample_spec(difficulty=2, seed=56, scenario_id="monthly_segment_report")
            bundle = generator.generate_instance(spec, root / "generated")
            evaluator = get_evaluator(bundle.manifest.family)
            bad_workspace = root / "bad-shape"
            shutil.copytree(bundle.visible_root, bad_workspace)
            output_path = bad_workspace / "outputs" / "report.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps({"month": "2024-01"}, indent=2) + "\n", encoding="utf-8")
            result = evaluator.evaluate(bad_workspace, bundle.manifest, bundle.hidden_root)
            self.assertFalse(result.success)
            self.assertLess(result.score, 1.0)

    def test_script_repair_timeout_returns_structured_result(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            generator = get_generator("script_repair")
            spec = generator.sample_spec(difficulty=3, seed=57)
            bundle = generator.generate_instance(spec, root / "generated")
            evaluator = get_evaluator(bundle.manifest.family)
            timeout_workspace = root / "timeout-script"
            shutil.copytree(bundle.visible_root, timeout_workspace)
            target_relative_path = next(iter(bundle.manifest.reference_solution["files"]))
            target_path = timeout_workspace / target_relative_path
            target_path.write_text(self.inject_sleep(target_path.read_text(encoding="utf-8")), encoding="utf-8")
            bundle.manifest.time_limit_seconds = 0.1
            result = evaluator.evaluate(timeout_workspace, bundle.manifest, bundle.hidden_root)
            self.assertFalse(result.success)
            self.assertIn("timeout", result.failure_labels)
            self.assertEqual(result.score, 0.0)

    def test_pipeline_timeout_returns_structured_result(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            generator = get_generator("pipeline")
            spec = generator.sample_spec(difficulty=3, seed=58)
            bundle = generator.generate_instance(spec, root / "generated")
            evaluator = get_evaluator(bundle.manifest.family)
            timeout_workspace = root / "timeout-pipeline"
            shutil.copytree(bundle.visible_root, timeout_workspace)
            entrypoint = json.loads((bundle.hidden_root / "evaluator_config.json").read_text(encoding="utf-8"))["entrypoint"]
            entrypoint_path = timeout_workspace / entrypoint
            entrypoint_path.write_text(self.inject_sleep(entrypoint_path.read_text(encoding="utf-8")), encoding="utf-8")
            bundle.manifest.time_limit_seconds = 0.1
            result = evaluator.evaluate(timeout_workspace, bundle.manifest, bundle.hidden_root)
            self.assertFalse(result.success)
            self.assertIn("timeout", result.failure_labels)
            self.assertEqual(result.score, 0.0)

    def test_retrieval_workspace_evaluator_exposes_partial_credit(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            generator = get_generator("retrieval_workspace")
            spec = generator.sample_spec(
                difficulty=4,
                seed=77,
                scenario_id="service_config_reconciliation",
            )
            bundle = generator.generate_instance(spec, root / "generated")
            evaluator = get_evaluator(bundle.manifest.family)
            partial_workspace = root / "partial"
            shutil.copytree(bundle.visible_root, partial_workspace)
            output_path = partial_workspace / "config" / "service_config.json"
            output_path.write_text(
                json.dumps(
                    {
                        "base_url": "https://svc.internal/v2",
                        "cohort_limit": 250,
                        "enable_shadow_mode": True,
                        "region": "us-east-1",
                        "retry_attempts": 1,
                        "service_name": "ledger-sync",
                        "timeout_seconds": 45,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            result = evaluator.evaluate(partial_workspace, bundle.manifest, bundle.hidden_root)
            self.assertFalse(result.success)
            self.assertGreater(result.score, 0.0)
            self.assertLess(result.score, 1.0)
            self.assertGreater(result.subscores["field_f1"], 0.0)

    def inject_sleep(self, source: str) -> str:
        header = "from __future__ import annotations\n\n"
        payload = "import time\n\ntime.sleep(1)\n\n"
        if source.startswith(header):
            return header + payload + source[len(header):]
        return payload + source


if __name__ == "__main__":
    unittest.main()
