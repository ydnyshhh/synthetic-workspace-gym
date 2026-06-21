from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


SWG_TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the active workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file path.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": None,
    },
    {
        "name": "write_file",
        "description": "Write UTF-8 text to a file in the active workspace, creating parent directories as needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file path.",
                },
                "content": {
                    "type": "string",
                    "description": "Complete file content to write.",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        "strict": None,
    },
    {
        "name": "append_file",
        "description": "Append UTF-8 text to a file in the active workspace, creating parent directories as needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file path.",
                },
                "content": {
                    "type": "string",
                    "description": "Content to append.",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        "strict": None,
    },
    {
        "name": "list_directory",
        "description": "List files and directories under a workspace-relative directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative directory path. Use '.' for the workspace root.",
                    "default": ".",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": None,
    },
    {
        "name": "run_shell",
        "description": "Run a shell command in the active workspace subject to SWG command policy.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        "strict": None,
    },
    {
        "name": "run_python",
        "description": "Run a workspace-relative Python script path in the active workspace. Pass only the script path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative Python script path, such as process_report.py.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": None,
    },
    {
        "name": "submit",
        "description": "Submit a final answer or artifact path and finish the episode.",
        "parameters": {
            "type": "object",
            "properties": {
                "path_or_answer": {
                    "type": "string",
                    "description": "Final artifact path or answer text.",
                }
            },
            "required": ["path_or_answer"],
            "additionalProperties": False,
        },
        "strict": None,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add SWG tool definitions to prompt/completion SFT JSONL rows."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--drop-metadata", action="store_true")
    parser.add_argument(
        "--openai-tool-calls",
        action="store_true",
        help="Convert assistant tool_calls to OpenAI function-calling shape.",
    )
    return parser.parse_args()


def convert_tool_call_to_openai(call: dict[str, Any]) -> dict[str, Any]:
    if call.get("type") == "function" and isinstance(call.get("function"), dict):
        return copy.deepcopy(call)
    name = call.get("name")
    arguments = call.get("arguments", {})
    if not isinstance(name, str) or not name:
        raise ValueError(f"Tool call is missing a string name: {call!r}")
    if not isinstance(arguments, dict):
        raise ValueError(f"Tool call arguments must be an object: {call!r}")

    converted = {
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False, sort_keys=True),
        },
    }
    if call.get("id"):
        converted["id"] = call["id"]
    return converted


def convert_message_tool_calls(message: dict[str, Any]) -> dict[str, Any]:
    converted = copy.deepcopy(message)
    tool_calls = converted.get("tool_calls")
    if tool_calls is None:
        return converted
    if not isinstance(tool_calls, list):
        raise ValueError(f"Message tool_calls must be a list: {message!r}")
    converted["tool_calls"] = [
        convert_tool_call_to_openai(call)
        for call in tool_calls
    ]
    return converted


def convert_row(
    row: dict[str, Any],
    drop_metadata: bool,
    openai_tool_calls: bool,
) -> dict[str, Any]:
    if "messages" in row:
        raise ValueError("messages key must not be present in prompt/completion input")
    if "prompt" not in row or "completion" not in row:
        raise ValueError("Input row must contain prompt and completion")

    prompt = copy.deepcopy(row["prompt"])
    completion = copy.deepcopy(row["completion"])
    if openai_tool_calls:
        if not isinstance(prompt, list):
            raise ValueError("Input prompt must be a list")
        if not isinstance(completion, dict):
            raise ValueError("Input completion must be an object")
        prompt = [
            convert_message_tool_calls(message)
            for message in prompt
        ]
        completion = convert_message_tool_calls(completion)

    out = {
        "prompt": prompt,
        "completion": completion,
        "tool_defs": SWG_TOOL_DEFS,
    }
    if not drop_metadata and "metadata" in row:
        out["metadata"] = row["metadata"]

    if "messages" in out:
        raise ValueError("messages key must not be present")
    return out


def main() -> int:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.input_jsonl.open("r", encoding="utf-8") as source:
        with args.output_jsonl.open("w", encoding="utf-8") as target:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    out = convert_row(row, args.drop_metadata, args.openai_tool_calls)
                except Exception as exc:
                    raise ValueError(f"{args.input_jsonl}:{line_number}: {exc}") from exc
                target.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1

    print(f"Wrote {count} rows to {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
