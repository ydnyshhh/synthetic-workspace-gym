from __future__ import annotations

import unittest
from types import SimpleNamespace

from synthetic_workspace_gym.verifiers.parser import SWGToolCallParser, parse_completion_to_action


class VerifiersParserTests(unittest.TestCase):
    def test_parses_json_tool_call(self) -> None:
        action = parse_completion_to_action('{"tool":"read_file","args":{"path":"README.md"}}')

        self.assertEqual(action, {"tool": "read_file", "args": {"path": "README.md"}})

    def test_parses_fenced_json_tool_call(self) -> None:
        action = SWGToolCallParser().parse(
            '```json\n{"tool":"list_directory","args":{"path":"."}}\n```'
        )

        self.assertEqual(action["tool"], "list_directory")
        self.assertEqual(action["args"], {"path": "."})

    def test_parses_openai_style_function_call(self) -> None:
        action = parse_completion_to_action(
            {
                "function": {
                    "name": "write_file",
                    "arguments": '{"path":"x.txt","content":"ok"}',
                }
            }
        )

        self.assertEqual(action["tool"], "write_file")
        self.assertEqual(action["args"], {"path": "x.txt", "content": "ok"})

    def test_parses_tool_call_object(self) -> None:
        action = parse_completion_to_action(SimpleNamespace(name="submit", arguments={"path_or_answer": "done"}))

        self.assertEqual(action["tool"], "submit")
        self.assertEqual(action["args"], {"path_or_answer": "done"})

    def test_final_text_becomes_submit(self) -> None:
        action = parse_completion_to_action("the answer is done")

        self.assertEqual(action["tool"], "submit")
        self.assertEqual(action["args"], {"path_or_answer": "the answer is done"})

    def test_malformed_json_does_not_crash(self) -> None:
        action = parse_completion_to_action('{"tool":')

        self.assertEqual(action["tool"], "submit")
        self.assertIn("parse_error", action)


if __name__ == "__main__":
    unittest.main()
