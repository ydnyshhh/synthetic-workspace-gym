from __future__ import annotations

from copy import deepcopy


SWG_PRIME_TOOL_SCHEMAS: list[dict[str, object]] = [
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the active workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative file path."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "write_file",
        "description": "Write UTF-8 text to a file in the active workspace, creating parent directories as needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative file path."},
                "content": {"type": "string", "description": "Complete file content to write."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "append_file",
        "description": "Append UTF-8 text to a file in the active workspace, creating parent directories as needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative file path."},
                "content": {"type": "string", "description": "Content to append."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
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
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_shell",
        "description": "Run a shell command in the active workspace subject to SWG command policy.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute."},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_python",
        "description": (
            "Run a workspace-relative Python script path in the active workspace. "
            "Pass only the script path, for example process_report.py; do not pass inline code, "
            "python -c, python -m, or python script.py."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command_or_script": {
                    "type": "string",
                    "description": "Workspace-relative Python script path, such as process_report.py.",
                },
            },
            "required": ["command_or_script"],
            "additionalProperties": False,
        },
    },
    {
        "name": "submit",
        "description": "Submit a final answer or artifact path and finish the episode.",
        "parameters": {
            "type": "object",
            "properties": {
                "path_or_answer": {"type": "string", "description": "Final artifact path or answer text."},
            },
            "required": ["path_or_answer"],
            "additionalProperties": False,
        },
    },
]


def get_tool_schemas(tool_permissions: list[str] | None = None) -> list[dict[str, object]]:
    schemas = deepcopy(SWG_PRIME_TOOL_SCHEMAS)
    if tool_permissions is None:
        return schemas
    allowed = set(tool_permissions)
    return [schema for schema in schemas if str(schema.get("name")) in allowed]
