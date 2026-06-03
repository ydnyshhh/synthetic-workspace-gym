from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from test_support import workspace_tempdir

from synthetic_workspace_gym.prime import (
    SWG_PRIME_TOOL_SCHEMAS,
    SyntheticWorkspacePrimeDataset,
    evaluator_result_to_prime_reward,
    get_tool_schemas,
    make_env,
)
from synthetic_workspace_gym.prime.env import SyntheticWorkspacePrimeEnv
from synthetic_workspace_gym.schemas import EvaluatorResult


class PrimeAdapterTests(unittest.TestCase):
    def test_tool_schemas_are_returned(self) -> None:
        schemas = get_tool_schemas()
        names = {schema["name"] for schema in schemas}

        self.assertEqual(len(schemas), len(SWG_PRIME_TOOL_SCHEMAS))
        self.assertIn("read_file", names)
        self.assertIn("submit", names)
        self.assertIn("parameters", schemas[0])

    def test_tool_schema_filtering(self) -> None:
        schemas = get_tool_schemas(["read_file", "submit"])

        self.assertEqual([schema["name"] for schema in schemas], ["read_file", "submit"])

    def test_evaluator_result_to_prime_reward_handles_full_result(self) -> None:
        result = EvaluatorResult(
            success=False,
            score=0.25,
            subscores={"tests_passed": 1.0},
            failure_labels=["hidden_tests_failed"],
            diagnostics={"detail": "mismatch"},
            runtime_seconds=0.5,
        )

        payload = evaluator_result_to_prime_reward(result)

        self.assertEqual(payload["reward"], 0.25)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["score"], 0.25)
        self.assertEqual(payload["subscores"], {"tests_passed": 1.0})
        self.assertEqual(payload["failure_labels"], ["hidden_tests_failed"])
        self.assertEqual(payload["diagnostics"], {"detail": "mismatch"})
        self.assertEqual(payload["runtime_seconds"], 0.5)

    def test_evaluator_result_to_prime_reward_falls_back_to_success(self) -> None:
        payload = evaluator_result_to_prime_reward(SimpleNamespace(success=True))

        self.assertEqual(payload["reward"], 1.0)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["score"], 1.0)
        self.assertEqual(payload["failure_labels"], [])

    def test_dataset_yields_task_dicts(self) -> None:
        dataset = SyntheticWorkspacePrimeDataset(
            families=("script_repair",),
            difficulties=(3,),
            scenarios={"script_repair": ("csv_schema_drift",)},
            seeds=(42,),
            split="smoke",
        )

        items = dataset.to_list()

        self.assertEqual(len(dataset), 1)
        self.assertEqual(
            items[0],
            {
                "family": "script_repair",
                "scenario": "csv_schema_drift",
                "difficulty": 3,
                "seed": 42,
                "split": "smoke",
                "task_id": "swg.script_repair.csv_schema_drift.d3.s42",
            },
        )

    def test_make_env_instantiates_environment(self) -> None:
        env = make_env(family="script_repair", scenario="csv_schema_drift", difficulty=3, seed=42)

        self.assertIsInstance(env, SyntheticWorkspacePrimeEnv)
        env.close()

    def test_reset_returns_prime_observation(self) -> None:
        with workspace_tempdir() as tmp_dir:
            env = make_env(
                family="script_repair",
                scenario="csv_schema_drift",
                difficulty=3,
                seed=42,
                output_dir=Path(tmp_dir),
            )
            try:
                observation = env.reset()
            finally:
                env.close()

        self.assertTrue(observation["env_id"])
        self.assertIn("instruction", observation)
        self.assertEqual(observation["family"], "script_repair")
        self.assertEqual(observation["scenario"], "csv_schema_drift")
        self.assertEqual(observation["max_steps"], 12)
        self.assertEqual(observation["time_limit_seconds"], 60)
        self.assertIn("metadata", observation)
        self.assertIn("tool_schemas", observation)

    def test_malformed_tool_action_returns_failed_observation(self) -> None:
        with workspace_tempdir() as tmp_dir:
            env = make_env(
                family="script_repair",
                scenario="csv_schema_drift",
                difficulty=3,
                seed=42,
                output_dir=Path(tmp_dir),
            )
            try:
                env.reset()
                result = env.step({"tool": "read_file", "args": {}})
            finally:
                env.close()

        self.assertFalse(result["done"])
        self.assertEqual(result["reward"], 0.0)
        self.assertFalse(result["info"]["success"])
        self.assertIn("Tool execution failed: missing required argument", result["observation"])
        self.assertEqual(result["info"]["error"], "tool_execution_error")
        self.assertEqual(result["info"]["exception_type"], "KeyError")
        self.assertIn("path", result["info"]["exception_message"])

    def test_submit_action_finishes_and_includes_reward_payload(self) -> None:
        with workspace_tempdir() as tmp_dir:
            env = make_env(
                family="script_repair",
                scenario="csv_schema_drift",
                difficulty=3,
                seed=42,
                output_dir=Path(tmp_dir),
            )
            try:
                env.reset()
                result = env.step({"tool": "submit", "args": {"path_or_answer": "done"}})
            finally:
                env.close()

        self.assertTrue(result["done"])
        self.assertIn("reward_payload", result["info"])
        self.assertIn("reward", result["info"]["reward_payload"])
        self.assertIsInstance(result["reward"], float)

    def test_runtime_home_is_not_visible_in_prime_workspace(self) -> None:
        with workspace_tempdir() as tmp_dir:
            env = make_env(
                family="script_repair",
                scenario="csv_schema_drift",
                difficulty=3,
                seed=42,
                output_dir=Path(tmp_dir),
            )
            try:
                env.reset()
                result = env.step({"tool": "list_directory", "args": {"path": "."}})
            finally:
                env.close()

        self.assertNotIn(".runtime-home", result["observation"])


if __name__ == "__main__":
    unittest.main()
