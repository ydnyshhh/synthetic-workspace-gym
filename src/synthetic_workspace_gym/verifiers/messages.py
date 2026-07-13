from __future__ import annotations

import json
from typing import Any


def to_verifiers_tool_exchange(
    action: dict[str, Any],
    content: object,
    call_id: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    tool = action.get("tool")
    if not isinstance(tool, str) or not tool:
        raise ValueError("Tool action requires a non-empty tool name.")
    if not call_id:
        raise ValueError("Verifiers tool calls require a non-empty call ID.")
    message_metadata = dict(metadata or {})
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id,
                "name": tool,
                "arguments": json.dumps(dict(action.get("args", {}) or {}), sort_keys=True),
            }],
            "metadata": message_metadata,
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": str(content),
            "metadata": message_metadata,
        },
    ]


def to_verifiers_branch_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert SWG's one-tool-per-turn transcript into native Verifiers messages."""

    converted: list[dict[str, Any]] = []
    pending_tool_call_id: str | None = None
    for index, raw_message in enumerate(messages):
        message = dict(raw_message)
        role = str(message.get("role", ""))

        if pending_tool_call_id is not None and role != "tool":
            raise ValueError(
                f"Tool call before message at index {index} has no matching tool observation."
            )

        if role == "assistant" and isinstance(message.get("tool_call"), dict):
            call = dict(message["tool_call"])
            call_id = f"restored-tool-{index}"
            exchange = to_verifiers_tool_exchange(
                call, "", call_id, metadata=dict(message.get("metadata", {}) or {}),
            )
            converted.append(exchange[0])
            pending_tool_call_id = call_id
            continue

        if role == "tool":
            if pending_tool_call_id is None:
                raise ValueError(f"Tool message at index {index} has no preceding tool call.")
            converted.append({
                "role": "tool",
                "tool_call_id": pending_tool_call_id,
                "content": str(message.get("content", "")),
                "metadata": dict(message.get("metadata", {}) or {}),
            })
            pending_tool_call_id = None
            continue

        converted.append(message)

    if pending_tool_call_id is not None:
        raise ValueError("Branch prefix ends with an unmatched tool call.")
    return converted
