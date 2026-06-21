from __future__ import annotations

import copy
from typing import Any


def public_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": call["name"],
        "arguments": copy.deepcopy(call.get("arguments", {})),
    }


def sequentialize_action_window(
    history: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    following_tool_messages: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    examples: list[dict[str, Any]] = []
    warnings = 0
    working_history = copy.deepcopy(history)
    tools_by_id = {
        message.get("tool_call_id"): message
        for message in following_tool_messages
        if message.get("tool_call_id")
    }
    needs_alignment = len(calls) > 1

    for call in calls:
        call_id = call.get("id")
        example_metadata = copy.deepcopy(metadata)
        warning = None

        if needs_alignment and not call_id:
            warning = "tool_observation_alignment_uncertain"
        elif needs_alignment and call_id not in tools_by_id:
            warning = "tool_observation_alignment_uncertain"

        if warning:
            example_metadata["sequentialization_warning"] = warning
            warnings += 1

        target = {
            "role": "assistant",
            "content": "",
            "tool_calls": [public_tool_call(call)],
        }
        examples.append(
            {
                "messages": copy.deepcopy(working_history),
                "target": target,
                "metadata": example_metadata,
            }
        )

        working_history.append(copy.deepcopy(target))
        if call_id and call_id in tools_by_id:
            working_history.append(copy.deepcopy(tools_by_id[call_id]))

    return examples, warnings
