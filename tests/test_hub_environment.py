from __future__ import annotations

import asyncio
import unittest
from importlib import util
from pathlib import Path
from types import SimpleNamespace

import test_support  # noqa: F401
from test_support import workspace_tempdir

import synthetic_workspace_gym as swg
from synthetic_workspace_gym.hub import HUB_SYSTEM_PROMPT, SyntheticWorkspaceHubEnv, load_environment
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
            self.assertEqual(len(rows), 2)
            self.assertEqual(set(rows["split"]), {"heldout"})
            self.assertTrue(all(str(task_id).startswith("swg.heldout.") for task_id in rows["task_id"]))
            self.assertTrue(any(tool.name == "read_file" for tool in env.tool_defs))
        finally:
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

                self.assertIn("You changed the workspace", str(write_messages[0].content))
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
