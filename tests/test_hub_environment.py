from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import patch
from importlib import util
from pathlib import Path
from types import SimpleNamespace

import test_support  # noqa: F401
from test_support import workspace_tempdir

import synthetic_workspace_gym as swg
from synthetic_workspace_gym.hub import (
    HUB_SYSTEM_PROMPT,
    SyntheticWorkspaceHubEnv,
    _run_blocking_swg_operation,
    _to_verifiers_branch_messages,
    _to_verifiers_tool_exchange,
    load_environment,
)
from synthetic_workspace_gym.counterfactual.runner import read_branch_manifest
from synthetic_workspace_gym.verifiers.compat import is_verifiers_available
from synthetic_workspace_gym.verifiers.env import SyntheticWorkspaceVerifiersEnv


class HubEnvironmentTests(unittest.TestCase):
    def test_package_exports_load_environment(self) -> None:
        self.assertIs(swg.load_environment, load_environment)

    def test_package_includes_verifiers_eval_defaults(self) -> None:
        spec = util.find_spec("synthetic_workspace_gym")
        self.assertIsNotNone(spec)
        package_dir = Path(next(iter(spec.submodule_search_locations or [])))
        pyproject = package_dir / "pyproject.toml"

        self.assertTrue(pyproject.exists())
        self.assertIn("[tool.verifiers.eval]", pyproject.read_text(encoding="utf-8"))

    def test_hub_system_prompt_sets_efficiency_constraints(self) -> None:
        self.assertIn("Use relative paths only", HUB_SYSTEM_PROMPT)
        self.assertIn("Do not look for or ask about hidden tests", HUB_SYSTEM_PROMPT)
        self.assertIn("After an edit, run one focused public check", HUB_SYSTEM_PROMPT)
        self.assertIn("call submit immediately", HUB_SYSTEM_PROMPT)
        self.assertIn("run_python accepts only a workspace-relative Python script path", HUB_SYSTEM_PROMPT)
        self.assertIn("include any needed output-directory creation inside that script", HUB_SYSTEM_PROMPT)
        self.assertIn("Never stop after only writing a file", HUB_SYSTEM_PROMPT)
        self.assertIn("Tabular: read README.md, task.json, and the listed input files", HUB_SYSTEM_PROMPT)
        self.assertIn("Use only the Python standard library", HUB_SYSTEM_PROMPT)

    def test_truncate_observation_marks_omitted_content(self) -> None:
        from synthetic_workspace_gym.hub import _truncate_observation

        content = _truncate_observation("abcdef", 3)
        self.assertIn("abc", content)
        self.assertIn("Observation truncated by SWG", content)
        self.assertIn("3 characters omitted", content)

    @unittest.skipUnless(is_verifiers_available(), "verifiers is unavailable")
    def test_real_branch_messages_normalize_for_native_verifiers(self) -> None:
        from verifiers.utils.message_utils import normalize_messages

        manifest = (
            Path(__file__).parents[1]
            / "examples"
            / "counterfactual"
            / "demo-pack"
            / "manifest.jsonl"
        )
        task = read_branch_manifest(manifest)[0]
        restored = normalize_messages(_to_verifiers_branch_messages(task.prefix_messages))
        tool_messages = [message for message in restored if message.role == "tool"]
        self.assertTrue(tool_messages)
        self.assertTrue(all(bool(message.tool_call_id) for message in tool_messages))

        forced = _to_verifiers_tool_exchange(
            task.forced_action or {}, "forced observation",
            "counterfactual-forced-action", metadata={"forced": True},
        )
        normalized_forced = normalize_messages(forced)
        self.assertEqual(normalized_forced[0].tool_calls[0].id, "counterfactual-forced-action")
        self.assertEqual(normalized_forced[1].tool_call_id, "counterfactual-forced-action")

        with self.assertRaisesRegex(ValueError, "no preceding tool call"):
            _to_verifiers_branch_messages([{"role": "tool", "content": "orphaned"}])
        with self.assertRaisesRegex(ValueError, "unmatched tool call"):
            _to_verifiers_branch_messages([
                {"role": "assistant", "tool_call": {"tool": "read_file", "args": {"path": "README.md"}}}
            ])

    @unittest.skipUnless(is_verifiers_available(), "verifiers is unavailable")
    def test_native_hub_setup_normalizes_restored_and_forced_messages(self) -> None:
        async def run_setup() -> None:
            manifest = (
                Path(__file__).parents[1]
                / "examples"
                / "counterfactual"
                / "demo-pack"
                / "manifest.jsonl"
            )
            task = read_branch_manifest(manifest)[0]
            with workspace_tempdir() as tmp_dir:
                env = load_environment(
                    branch_manifest_path=str(manifest),
                    branch_task_id=task.task_id,
                    branch_mode="forced",
                    output_dir=str(Path(tmp_dir) / "runtime"),
                )
                self.assertIsInstance(env, SyntheticWorkspaceHubEnv)
                row = dict(env.get_dataset()[0])
                state = {"input": row, "trajectory_id": "native-branch-message-test"}
                await env.setup_state(state)
                try:
                    tool_messages = [message for message in state["prompt"] if message.role == "tool"]
                    self.assertTrue(tool_messages)
                    self.assertTrue(all(bool(message.tool_call_id) for message in tool_messages))
                    self.assertTrue(any(
                        message.tool_call_id == "counterfactual-forced-action"
                        for message in tool_messages
                    ))
                    self.assertEqual(state["swg_forced_prefix_length"], 2)
                    self.assertTrue(state["swg_forced_action_result"]["success"])
                    self.assertTrue(state["swg_forced_action_result"]["info"]["success"])
                    self.assertEqual(state["swg_policy_start_message_index"], len(state["prompt"]))
                    self.assertEqual(
                        state["swg_loss_mask_metadata"],
                        {
                            "exclude_restored_messages": True,
                            "exclude_forced_messages": True,
                            "forced_tool_call_id": "counterfactual-forced-action",
                        },
                    )
                    self.assertEqual(state.get("trajectory", []), [])
                finally:
                    state["swg_env"].close()

        asyncio.run(run_setup())

    def test_load_environment_accepts_fixed_task_args(self) -> None:
        env = load_environment(
            split=None,
            family="script_repair",
            scenario="csv_schema_drift",
            difficulty=1,
            seed=7,
            max_examples=1,
        )
        try:
            self.assertTrue(isinstance(env, (SyntheticWorkspaceHubEnv, SyntheticWorkspaceVerifiersEnv)))
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    @unittest.skipUnless(is_verifiers_available(), "verifiers is unavailable")
    def test_hub_environment_builds_split_dataset(self) -> None:
        env = load_environment(split="heldout", family="script_repair", max_examples=2)
        try:
            rows = env.get_dataset()
            eval_rows = env.get_eval_dataset()
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(eval_rows), 2)
            self.assertEqual(list(rows["task_id"]), list(eval_rows["task_id"]))
            self.assertEqual(set(rows["split"]), {"heldout"})
            self.assertTrue(all(str(task_id).startswith("swg.heldout.") for task_id in rows["task_id"]))
            self.assertTrue(any(tool.name == "read_file" for tool in env.tool_defs))
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    @unittest.skipUnless(is_verifiers_available(), "verifiers is unavailable")
    def test_hub_environment_default_sampling_keeps_head_rows(self) -> None:
        env = load_environment(split="validation", max_examples=16)
        try:
            rows = env.get_eval_dataset()
            self.assertEqual(len(rows), 16)
            self.assertEqual(set(rows["family"]), {"tabular"})
            self.assertEqual(set(rows["scenario"]), {"monthly_segment_report"})
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    @unittest.skipUnless(is_verifiers_available(), "verifiers is unavailable")
    def test_hub_environment_balanced_sampling_covers_multiple_task_types(self) -> None:
        env = load_environment(split="validation", max_examples=16, sample_strategy="balanced")
        try:
            rows = env.get_eval_dataset()
            self.assertEqual(len(rows), 16)
            self.assertGreater(len(set(rows["family"])), 1)
            self.assertGreater(len(set(rows["scenario"])), 1)
            self.assertGreater(len(set(rows["difficulty"])), 1)
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    @unittest.skipUnless(is_verifiers_available(), "verifiers is unavailable")
    def test_hub_environment_balanced_shuffle_is_deterministic(self) -> None:
        first = load_environment(
            split="validation",
            max_examples=16,
            sample_strategy="balanced",
            shuffle=True,
            shuffle_seed=42,
        )
        second = load_environment(
            split="validation",
            max_examples=16,
            sample_strategy="balanced",
            shuffle="true",
            shuffle_seed=42,
        )
        different = load_environment(
            split="validation",
            max_examples=16,
            sample_strategy="balanced",
            shuffle=True,
            shuffle_seed=43,
        )
        try:
            first_ids = list(first.get_eval_dataset()["task_id"])
            second_ids = list(second.get_eval_dataset()["task_id"])
            different_ids = list(different.get_eval_dataset()["task_id"])
            self.assertEqual(first_ids, second_ids)
            self.assertNotEqual(first_ids, different_ids)
        finally:
            for env in (first, second, different):
                close = getattr(env, "close", None)
                if callable(close):
                    close()

    @unittest.skipUnless(is_verifiers_available(), "verifiers is unavailable")
    def test_hub_environment_executes_swg_tool_turn(self) -> None:
        async def run_turn() -> None:
            with workspace_tempdir() as tmp_dir:
                env = load_environment(
                    split=None,
                    family="script_repair",
                    scenario="csv_schema_drift",
                    difficulty=1,
                    seed=7,
                    max_examples=1,
                    output_dir=str(Path(tmp_dir) / "runtime"),
                )
                row = env.get_dataset()[0]
                state = {"input": row, "trajectory_id": "hub-test"}

                await env.setup_state(state)
                self.assertIn("swg_env", state)
                self.assertTrue(state["prompt"])

                list_call = SimpleNamespace(id="call-1", name="list_directory", arguments='{"path":"."}')
                messages = [SimpleNamespace(tool_calls=[list_call])]
                tool_messages = await env.env_response(messages, state)

                self.assertEqual(tool_messages[0].role, "tool")
                self.assertIn("README", str(tool_messages[0].content))
                state["swg_env"].close()

        asyncio.run(run_turn())

    @unittest.skipUnless(is_verifiers_available(), "verifiers is unavailable")
    def test_native_hub_offloads_blocking_operations_from_event_loop(self) -> None:
        async def run_check() -> None:
            env = load_environment(
                split=None,
                family="script_repair",
                scenario="csv_schema_drift",
                difficulty=1,
                seed=7,
                max_examples=1,
            )
            row = dict(env.get_dataset()[0])
            state = {"input": row, "trajectory_id": "hub-offload-test"}

            def slow_reset() -> dict[str, object]:
                time.sleep(0.08)
                return {"message": "ready", "metadata": {}}

            fake_runtime = SimpleNamespace(reset=slow_reset)
            with patch(
                "synthetic_workspace_gym.hub.SyntheticWorkspacePrimeEnv",
                return_value=fake_runtime,
            ):
                setup_task = asyncio.create_task(env.setup_state(state))
                await asyncio.sleep(0.01)
                self.assertFalse(setup_task.done())
                await setup_task

            def slow_action(*_args: object) -> tuple[str, bool]:
                time.sleep(0.08)
                return "slow observation", False

            tool_call = SimpleNamespace(
                id="call-offload",
                name="list_directory",
                arguments='{"path":"."}',
            )
            with patch(
                "synthetic_workspace_gym.hub._execute_swg_action",
                side_effect=slow_action,
            ):
                response_task = asyncio.create_task(
                    env.env_response([SimpleNamespace(tool_calls=[tool_call])], state)

                )
                await asyncio.sleep(0.01)
                self.assertFalse(response_task.done())
                await response_task

        asyncio.run(run_check())

    @unittest.skipUnless(is_verifiers_available(), "verifiers is unavailable")
    def test_native_hub_marks_prime_upload_timeout_as_retryable_infrastructure(self) -> None:
        from synthetic_workspace_gym.verifiers.compat import vf

        class UploadTimeoutError(RuntimeError):
            pass

        def fail_upload() -> None:
            try:
                raise UploadTimeoutError("runtime upload exceeded 300 seconds")
            except UploadTimeoutError as exc:
                raise RuntimeError("Prime sandbox execution failed") from exc

        async def run_check() -> None:
            with self.assertRaises(vf.InfraError) as raised:
                await _run_blocking_swg_operation(fail_upload)
            self.assertIsInstance(raised.exception.__cause__, RuntimeError)

        asyncio.run(run_check())

    @unittest.skipUnless(is_verifiers_available(), "verifiers is unavailable")
    def test_hub_environment_setup_uses_dataset_row_metadata(self) -> None:
        async def run_turn() -> None:
            with workspace_tempdir() as tmp_dir:
                env = load_environment(
                    split="validation",
                    family="pipeline",
                    max_examples=1,
                    max_turns=2,
                    output_dir=str(Path(tmp_dir) / "runtime"),
                )
                row = dict(env.get_eval_dataset()[0])
                state = {"input": row, "task": row, "trajectory_id": "hub-row-test"}

                await env.setup_state(state)

                self.assertEqual(state["swg_task"]["task_id"], row["task_id"])
                self.assertEqual(state["swg_task"]["family"], "pipeline")
                self.assertEqual(state["swg_task"]["difficulty"], row["difficulty"])
                self.assertEqual(state["swg_task"]["seed"], row["seed"])
                self.assertEqual(state["swg_max_observation_chars"], 20000)
                self.assertEqual(state["swg_env"].manifest.family.value, "pipeline")
                prompt_text = "\n".join(str(message.get("content", "")) for message in state["prompt"])
                self.assertIn(str(row["task_id"]), prompt_text)
                self.assertIn("- family: pipeline", prompt_text)
                self.assertIn(f"- scenario: {row['scenario']}", prompt_text)
                state["swg_env"].close()

        asyncio.run(run_turn())

    @unittest.skipUnless(is_verifiers_available(), "verifiers is unavailable")
    def test_hub_environment_falls_back_to_rows_when_hosted_state_is_empty(self) -> None:
        async def run_turn() -> None:
            with workspace_tempdir() as tmp_dir:
                env = load_environment(
                    split="validation",
                    family="tabular",
                    max_examples=2,
                    max_turns=2,
                    output_dir=str(Path(tmp_dir) / "runtime"),
                )
                first = dict(env.get_eval_dataset()[0])
                second = dict(env.get_eval_dataset()[1])

                first_state = {"trajectory_id": "hosted-empty-1"}
                await env.setup_state(first_state)
                self.assertEqual(first_state["swg_task"]["task_id"], first["task_id"])
                self.assertEqual(first_state["swg_env"].manifest.family.value, "tabular")
                first_state["swg_env"].close()

                second_state = {"trajectory_id": "hosted-empty-2"}
                await env.setup_state(second_state)
                self.assertEqual(second_state["swg_task"]["task_id"], second["task_id"])
                second_state["swg_env"].close()

        asyncio.run(run_turn())

    @unittest.skipUnless(is_verifiers_available(), "verifiers is unavailable")
    def test_hub_environment_uses_hosted_example_index_when_available(self) -> None:
        async def run_turn() -> None:
            with workspace_tempdir() as tmp_dir:
                env = load_environment(
                    split="validation",
                    family="tabular",
                    max_examples=3,
                    max_turns=2,
                    output_dir=str(Path(tmp_dir) / "runtime"),
                )
                target = dict(env.get_eval_dataset()[2])
                state = {"example_id": 2, "trajectory_id": "hosted-index-2"}

                await env.setup_state(state)

                self.assertEqual(state["swg_task"]["task_id"], target["task_id"])
                prompt_text = "\n".join(str(message.get("content", "")) for message in state["prompt"])
                self.assertIn(str(target["task_id"]), prompt_text)
                state["swg_env"].close()

        asyncio.run(run_turn())

    @unittest.skipUnless(is_verifiers_available(), "verifiers is unavailable")
    def test_hub_environment_decouples_model_turns_from_tool_steps(self) -> None:
        async def run_turn() -> None:
            with workspace_tempdir() as tmp_dir:
                env = load_environment(
                    split=None,
                    family="script_repair",
                    scenario="csv_schema_drift",
                    difficulty=1,
                    seed=7,
                    max_examples=1,
                    max_turns=2,
                    output_dir=str(Path(tmp_dir) / "runtime"),
                )
                row = env.get_dataset()[0]
                state = {"input": row, "trajectory_id": "hub-budget-test"}

                await env.setup_state(state)
                self.assertGreaterEqual(state["swg_env"].manifest.max_steps, 24)

                calls = [
                    SimpleNamespace(id=f"call-{index}", name="list_directory", arguments='{"path":"."}')
                    for index in range(3)
                ]
                tool_messages = await env.env_response([SimpleNamespace(tool_calls=calls)], state)

                self.assertEqual(len(tool_messages), 3)
                self.assertNotIn("final_env_response", state)
                state["swg_env"].close()

        asyncio.run(run_turn())

    @unittest.skipUnless(is_verifiers_available(), "verifiers is unavailable")
    def test_hub_environment_guides_after_write_and_check(self) -> None:
        async def run_turn() -> None:
            with workspace_tempdir() as tmp_dir:
                env = load_environment(
                    split=None,
                    family="script_repair",
                    scenario="csv_schema_drift",
                    difficulty=1,
                    seed=7,
                    max_examples=1,
                    output_dir=str(Path(tmp_dir) / "runtime"),
                )
                row = env.get_dataset()[0]
                state = {"input": row, "trajectory_id": "hub-guidance-test"}

                await env.setup_state(state)
                write_call = SimpleNamespace(
                    id="call-write",
                    name="write_file",
                    arguments='{"path":"scratch.txt","content":"ok"}',
                )
                write_messages = await env.env_response([SimpleNamespace(tool_calls=[write_call])], state)

                self.assertIn("You changed the repair target", str(write_messages[0].content))
                self.assertIn("python run_example.py", str(write_messages[0].content))
                self.assertTrue(state["swg_has_written"])
                self.assertNotIn("final_env_response", state)

                check_call = SimpleNamespace(
                    id="call-check",
                    name="run_shell",
                    arguments='{"command":"python --version"}',
                )
                check_messages = await env.env_response([SimpleNamespace(tool_calls=[check_call])], state)

                self.assertIn("A check ran after your edit", str(check_messages[0].content))
                self.assertTrue(state["swg_successful_check_after_write"])
                self.assertNotIn("final_env_response", state)
                state["swg_env"].close()

        asyncio.run(run_turn())

    @unittest.skipUnless(is_verifiers_available(), "verifiers is unavailable")
    def test_hub_environment_guides_tabular_script_execution(self) -> None:
        async def run_turn() -> None:
            with workspace_tempdir() as tmp_dir:
                env = load_environment(
                    split=None,
                    family="tabular",
                    scenario="weekly_refund_rollup",
                    difficulty=2,
                    seed=88,
                    max_examples=1,
                    output_dir=str(Path(tmp_dir) / "runtime"),
                )
                row = env.get_dataset()[0]
                state = {"input": row, "trajectory_id": "hub-tabular-guidance-test"}

                await env.setup_state(state)
                write_call = SimpleNamespace(
                    id="call-write",
                    name="write_file",
                    arguments='{"path":"scripts/process_events.py","content":"print(1)"}',
                )
                write_messages = await env.env_response([SimpleNamespace(tool_calls=[write_call])], state)

                content = str(write_messages[0].content)
                self.assertIn("You wrote a processing script", content)
                self.assertIn("run_python", content)
                self.assertIn("outputs/weekly_rollup.json", content)
                self.assertIn("Do not stop after writing the script", content)
                state["swg_env"].close()

        asyncio.run(run_turn())

    @unittest.skipUnless(is_verifiers_available(), "verifiers is unavailable")
    def test_hub_environment_blocks_wrong_artifact_submit(self) -> None:
        async def run_turn() -> None:
            with workspace_tempdir() as tmp_dir:
                env = load_environment(
                    split=None,
                    family="tabular",
                    scenario="weekly_refund_rollup",
                    difficulty=2,
                    seed=88,
                    max_examples=1,
                    output_dir=str(Path(tmp_dir) / "runtime"),
                )
                row = env.get_dataset()[0]
                state = {"input": row, "trajectory_id": "hub-submit-correction-test"}

                await env.setup_state(state)
                submit_call = SimpleNamespace(
                    id="call-submit",
                    name="submit",
                    arguments='{"path_or_answer":"scripts/process_events.py"}',
                )
                messages = await env.env_response([SimpleNamespace(tool_calls=[submit_call])], state)

                content = str(messages[0].content)
                self.assertIn("Submit correction", content)
                self.assertIn("outputs/weekly_rollup.json", content)
                self.assertNotIn("final_env_response", state)
                state["swg_env"].close()

        asyncio.run(run_turn())

    @unittest.skipUnless(is_verifiers_available(), "verifiers is unavailable")
    def test_hub_environment_executes_json_text_tool_turn(self) -> None:
        async def run_turn() -> None:
            with workspace_tempdir() as tmp_dir:
                env = load_environment(
                    split=None,
                    family="script_repair",
                    scenario="csv_schema_drift",
                    difficulty=1,
                    seed=7,
                    max_examples=1,
                    output_dir=str(Path(tmp_dir) / "runtime"),
                )
                row = env.get_dataset()[0]
                state = {"input": row, "trajectory_id": "hub-json-test"}

                await env.setup_state(state)
                messages = [SimpleNamespace(content='{"tool":"list_directory","args":{"path":"."}}')]
                responses = await env.env_response(messages, state)

                self.assertEqual(responses[0].role, "user")
                self.assertIn("Observation:", str(responses[0].content))
                self.assertIn("README", str(responses[0].content))
                state["swg_env"].close()

        asyncio.run(run_turn())

    @unittest.skipUnless(is_verifiers_available(), "verifiers is unavailable")
    def test_hub_environment_requests_json_when_no_tool_action(self) -> None:
        async def run_turn() -> None:
            with workspace_tempdir() as tmp_dir:
                env = load_environment(
                    split=None,
                    family="script_repair",
                    scenario="csv_schema_drift",
                    difficulty=1,
                    seed=7,
                    max_examples=1,
                    output_dir=str(Path(tmp_dir) / "runtime"),
                )
                row = env.get_dataset()[0]
                state = {"input": row, "trajectory_id": "hub-format-test"}

                await env.setup_state(state)
                messages = [SimpleNamespace(content="I need more information before I can help.")]
                responses = await env.env_response(messages, state)

                self.assertEqual(responses[0].role, "user")
                self.assertIn("Respond with exactly one JSON tool action", str(responses[0].content))
                self.assertNotIn("final_env_response", state)
                state["swg_env"].close()

        asyncio.run(run_turn())


if __name__ == "__main__":
    unittest.main()
