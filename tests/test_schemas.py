from __future__ import annotations

import unittest

from test_support import ROOT

from synthetic_workspace_gym.schemas import (
    ComplexityProfile,
    EnvironmentFamily,
    EnvironmentManifest,
    EnvironmentSpec,
    EvaluatorResult,
    ToolPermissions,
)
from synthetic_workspace_gym.utils.paths import ensure_within_root


class SchemaRoundTripTests(unittest.TestCase):
    def test_environment_spec_round_trip(self) -> None:
        spec = EnvironmentSpec(
            env_family=EnvironmentFamily.TABULAR,
            difficulty=3,
            seed=7,
            scenario_id="monthly_segment_report",
            tool_permissions=ToolPermissions(run_shell=False),
            task_params={"scenario": "demo"},
            complexity_profile=ComplexityProfile(
                file_count=5,
                distractor_count=1,
                dependency_depth=2,
                reasoning_hops=3,
                transformation_count=3,
                bug_subtlety=0,
                execution_required=True,
                output_constraint_strength=3,
            ),
        )
        restored = EnvironmentSpec.from_dict(spec.to_dict())
        self.assertEqual(restored.env_family, EnvironmentFamily.TABULAR)
        self.assertEqual(restored.scenario_id, "monthly_segment_report")
        self.assertEqual(restored.tool_permissions.run_shell, False)
        self.assertEqual(restored.task_params["scenario"], "demo")
        self.assertEqual(restored.complexity_profile.reasoning_hops, 3)

    def test_manifest_and_evaluator_result_round_trip(self) -> None:
        manifest = EnvironmentManifest(
            env_id="env-1",
            family=EnvironmentFamily.PIPELINE,
            difficulty=2,
            seed=11,
            instruction="repair the project",
            workspace_root="visible",
            visible_files=["README.md", "task.json"],
            hidden_root="hidden",
            hidden_files=["expected_output.json"],
            tool_permissions=ToolPermissions(),
            max_steps=12,
            time_limit_seconds=30,
            metadata={"family": "pipeline"},
            evaluator_entrypoint="synthetic_workspace_gym.evaluators.pipeline:PipelineEvaluator",
            reference_solution={"files": {"config/pipeline_config.json": "{}\n"}},
        )
        restored_manifest = EnvironmentManifest.from_dict(manifest.to_dict())
        self.assertEqual(restored_manifest.family, EnvironmentFamily.PIPELINE)
        self.assertIn("README.md", restored_manifest.visible_files)

        result = EvaluatorResult(
            success=True,
            score=1.0,
            subscores={"exact_match": 1.0},
            diagnostics={"runtime": "fast"},
            runtime_seconds=0.1,
        )
        restored_result = EvaluatorResult.from_dict(result.to_dict())
        self.assertTrue(restored_result.success)
        self.assertEqual(restored_result.subscores["exact_match"], 1.0)

    def test_ensure_within_root_rejects_parent_escape(self) -> None:
        root = ROOT / "src"
        with self.assertRaises(ValueError):
            ensure_within_root(root, "../README.md")


if __name__ == "__main__":
    unittest.main()
