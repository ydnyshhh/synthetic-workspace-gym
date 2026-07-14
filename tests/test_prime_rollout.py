from __future__ import annotations

import json
import unittest
from pathlib import Path

from test_support import workspace_tempdir

from synthetic_workspace_gym.prime.agents import PrimeReActAgent
from synthetic_workspace_gym.prime.clients import HeuristicReferenceClient, JSONActionClient, ScriptedPrimeClient
from synthetic_workspace_gym.prime.env import SyntheticWorkspacePrimeEnv
from synthetic_workspace_gym.prime.rollout import build_batch_summary, run_prime_rollout, run_prime_rollout_batch
from synthetic_workspace_gym.prime.transcript import make_event, read_transcript_jsonl, write_transcript_jsonl


class PrimeRolloutTests(unittest.TestCase):
    def test_scripted_prime_client_emits_planned_actions(self) -> None:
        client = ScriptedPrimeClient(
            [
                {"tool": "list_directory", "args": {"path": "."}},
                {"tool": "submit", "args": {"path_or_answer": "done"}},
            ]
        )

        first = client.complete([], [])
        second = client.complete([], [])

        self.assertEqual(first["type"], "tool_call")
        self.assertEqual(first["tool"], "list_directory")
        self.assertEqual(second["tool"], "submit")

    def test_json_action_client_parses_json_strings(self) -> None:
        client = JSONActionClient(lambda messages, tools, metadata: '{"tool":"read_file","args":{"path":"README.md"}}')

        response = client.complete([], [])

        self.assertEqual(response["type"], "tool_call")
        self.assertEqual(response["tool"], "read_file")
        self.assertEqual(response["args"], {"path": "README.md"})

    def test_transcript_jsonl_write_read_round_trip(self) -> None:
        with workspace_tempdir() as tmp_dir:
            path = Path(tmp_dir) / "transcript.jsonl"
            events = [make_event("system", {"content": "hello"}), make_event("tool_call", {"tool": "submit"}, 0)]

            write_transcript_jsonl(path, events)
            restored = read_transcript_jsonl(path)

        self.assertEqual([event["event_type"] for event in restored], ["system", "tool_call"])
        self.assertEqual(restored[1]["step_index"], 0)

    def test_prime_react_agent_runs_scripted_rollout_to_completion(self) -> None:
        with workspace_tempdir() as tmp_dir:
            env = SyntheticWorkspacePrimeEnv(
                family="script_repair",
                scenario="csv_schema_drift",
                difficulty=1,
                seed=7,
                output_dir=Path(tmp_dir),
            )
            try:
                agent = PrimeReActAgent(
                    ScriptedPrimeClient(
                        [
                            {"tool": "list_directory", "args": {"path": "."}},
                            {"tool": "submit", "args": {"path_or_answer": "done"}},
                        ]
                    )
                )
                rollout = agent.run(env)
            finally:
                env.close()

        self.assertEqual(rollout["turn_count"], 2)
        self.assertEqual(rollout["stopped_reason"], "submit")
        self.assertIn("reward", rollout["reward_payload"])
        self.assertEqual(rollout["tool_calls"][-1]["tool"], "submit")
        self.assertTrue(any(event["event_type"] == "evaluation" for event in rollout["transcript_events"]))

    def test_run_prime_rollout_creates_artifacts(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            result = run_prime_rollout(
                family="script_repair",
                scenario="csv_schema_drift",
                difficulty=1,
                seed=7,
                client=ScriptedPrimeClient(
                    [
                        {"tool": "list_directory", "args": {"path": "."}},
                        {"tool": "submit", "args": {"path_or_answer": "done"}},
                    ]
                ),
                output_dir=root / "prime_rollouts",
                rollout_id="rollout-test",
            )
            artifact_dir = Path(result["artifact_dir"])
            payload = json.loads((artifact_dir / "prime_rollout.json").read_text(encoding="utf-8"))
            artifact_checks = {
                "prime_rollout": (artifact_dir / "prime_rollout.json").exists(),
                "transcript": (artifact_dir / "transcript.jsonl").exists(),
                "final_reward": (artifact_dir / "final_reward.json").exists(),
                "final_workspace": (artifact_dir / "final_workspace").is_dir(),
                "manifest": (artifact_dir / "manifest.json").exists(),
                "final_diff": (artifact_dir / "final_diff.txt").exists(),
                "hidden": (artifact_dir / "hidden").exists(),
            }

        self.assertTrue(artifact_checks["prime_rollout"])
        self.assertTrue(artifact_checks["transcript"])
        self.assertTrue(artifact_checks["final_reward"])
        self.assertTrue(artifact_checks["final_workspace"])
        self.assertTrue(artifact_checks["manifest"])
        self.assertTrue(artifact_checks["final_diff"])
        self.assertFalse(artifact_checks["hidden"])
        self.assertEqual(payload["rollout_id"], "rollout-test")
        self.assertEqual(payload["tool_counts"]["submit"], 1)
        self.assertEqual(payload["stopped_reason"], "submit")
        self.assertEqual(payload["model"]["privileged"], False)
        self.assertEqual(payload["messages_path"], "transcript.jsonl")
        self.assertEqual(payload["sandbox"]["backend"], "local")
        self.assertLessEqual(max(len(message["content"]) for message in payload["messages"]), 550)

    def test_run_prime_rollout_accepts_explicit_local_sandbox(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            result = run_prime_rollout(
                family="script_repair",
                scenario="csv_schema_drift",
                difficulty=1,
                seed=7,
                client=ScriptedPrimeClient([{"tool": "submit", "args": {"path_or_answer": "done"}}]),
                output_dir=root / "prime_rollouts",
                rollout_id="local-sandbox-rollout",
                sandbox_backend="local",
            )
            payload = json.loads(Path(result["prime_rollout_path"]).read_text(encoding="utf-8"))

        self.assertEqual(payload["sandbox"]["backend"], "local")

    def test_heuristic_reference_rollout_marks_model_privileged(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            result = run_prime_rollout(
                family="script_repair",
                scenario="csv_schema_drift",
                difficulty=1,
                seed=7,
                client=HeuristicReferenceClient(),
                output_dir=root / "prime_rollouts",
                rollout_id="reference-rollout",
            )
            payload = json.loads(Path(result["prime_rollout_path"]).read_text(encoding="utf-8"))

        self.assertTrue(payload["model"]["privileged"])
        self.assertEqual(payload["model"]["client_type"], "heuristic_reference")
        self.assertTrue(payload["success"])

    def test_rollout_records_max_turns_stop_reason(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            result = run_prime_rollout(
                family="script_repair",
                scenario="csv_schema_drift",
                difficulty=1,
                seed=7,
                client=ScriptedPrimeClient([{"tool": "list_directory", "args": {"path": "."}}]),
                output_dir=root / "prime_rollouts",
                max_turns=1,
                rollout_id="max-turns-rollout",
            )
            payload = json.loads(Path(result["prime_rollout_path"]).read_text(encoding="utf-8"))

        self.assertEqual(payload["stopped_reason"], "max_turns")

    def test_rollout_records_client_error_stop_reason(self) -> None:
        def raise_error(messages, tools, metadata):
            raise RuntimeError("client broke")

        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            result = run_prime_rollout(
                family="script_repair",
                scenario="csv_schema_drift",
                difficulty=1,
                seed=7,
                client=JSONActionClient(raise_error),
                output_dir=root / "prime_rollouts",
                rollout_id="client-error-rollout",
            )
            payload = json.loads(Path(result["prime_rollout_path"]).read_text(encoding="utf-8"))

        self.assertEqual(payload["stopped_reason"], "client_error")

    def test_rollout_records_tool_error_stop_reason(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            result = run_prime_rollout(
                family="script_repair",
                scenario="csv_schema_drift",
                difficulty=1,
                seed=7,
                client=ScriptedPrimeClient([{"tool": "read_file", "args": {}}]),
                output_dir=root / "prime_rollouts",
                rollout_id="tool-error-rollout",
            )
            payload = json.loads(Path(result["prime_rollout_path"]).read_text(encoding="utf-8"))

        self.assertEqual(payload["stopped_reason"], "tool_error")

    def test_malformed_tool_action_does_not_crash_env_step(self) -> None:
        with workspace_tempdir() as tmp_dir:
            env = SyntheticWorkspacePrimeEnv(
                family="script_repair",
                scenario="csv_schema_drift",
                difficulty=1,
                seed=7,
                output_dir=Path(tmp_dir),
            )
            try:
                env.reset()
                result = env.step({"tool": "read_file", "args": {}})
            finally:
                env.close()

        self.assertFalse(result["done"])
        self.assertEqual(result["info"]["error"], "tool_execution_error")
        self.assertEqual(result["info"]["exception_type"], "KeyError")

    def test_batch_summary_aggregation(self) -> None:
        summary = build_batch_summary(
            [
                {"reward": 1.0, "success": True, "rollout_id": "a"},
                {"reward": 0.0, "success": False, "rollout_id": "b"},
            ]
        )

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["mean_reward"], 0.5)

    def test_rollout_batch_catches_per_environment_failures(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps({"env_id": "missing-env", "environment_path": "environments/missing-env"}) + "\n",
                encoding="utf-8",
            )

            summary = run_prime_rollout_batch(
                manifest_path,
                client_factory=lambda: ScriptedPrimeClient(),
                output_dir=root / "rollouts",
            )

        self.assertEqual(summary["count"], 1)
        self.assertFalse(summary["rollouts"][0]["success"])
        self.assertEqual(summary["rollouts"][0]["env_id"], "missing-env")
        self.assertIn("exception_type", summary["rollouts"][0])

    def test_heuristic_reference_client_emits_solution_writes_then_submit(self) -> None:
        client = HeuristicReferenceClient()
        metadata = {"reference_solution": {"files": {"b.txt": "two", "a.txt": "one"}}}

        first = client.complete([], [], metadata)
        second = client.complete([], [], metadata)
        third = client.complete([], [], metadata)

        self.assertEqual(first["tool"], "write_file")
        self.assertEqual(first["args"]["path"], "a.txt")
        self.assertEqual(second["args"]["path"], "b.txt")
        self.assertEqual(third["tool"], "submit")




    def test_model_metadata_excludes_host_paths_and_evaluator_details(self) -> None:
        agent = PrimeReActAgent(ScriptedPrimeClient([]))
        metadata = agent._model_metadata(
            {
                "metadata": {
                    "environment_path": "/host/package/branch_pack/environment",
                    "workspace_path": "/host/runtime/active",
                    "hidden_root": "/host/package/hidden",
                    "evaluator_entrypoint": "/host/evaluator.py",
                    "pack_id": "pack-1",
                }
            },
            object(),
        )
        self.assertEqual(metadata, {"pack_id": "pack-1"})
if __name__ == "__main__":
    unittest.main()
