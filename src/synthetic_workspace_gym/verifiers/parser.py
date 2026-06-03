from __future__ import annotations

import json
import re
from typing import Any

from .tools import parse_verifiers_tool_call


class SWGToolCallParser:
    def parse(self, completion: object) -> dict[str, Any]:
        try:
            return _parse_completion(completion)
        except Exception as exc:
            text = _completion_text(completion)
            return {
                "tool": "submit",
                "args": {"path_or_answer": text},
                "parse_error": f"{type(exc).__name__}: {exc}",
            }


def parse_completion_to_action(completion: object) -> dict[str, Any]:
    return SWGToolCallParser().parse(completion)


def _parse_completion(completion: object) -> dict[str, Any]:
    if isinstance(completion, dict):
        if "tool_calls" in completion:
            calls = completion.get("tool_calls") or []
            if calls:
                return parse_verifiers_tool_call(calls[0])
        if "tool" in completion or "name" in completion or "function" in completion:
            return parse_verifiers_tool_call(completion)
        if "content" in completion:
            return _parse_text(str(completion.get("content") or ""))

    tool_calls = getattr(completion, "tool_calls", None)
    if tool_calls:
        return parse_verifiers_tool_call(tool_calls[0])

    if isinstance(completion, str):
        return _parse_text(completion)

    content = getattr(completion, "content", None)
    if content is not None:
        return _parse_text(str(content))

    return parse_verifiers_tool_call(completion)


def _parse_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    json_text = _extract_fenced_json(stripped) or stripped
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError:
        if stripped.startswith("{") or stripped.startswith("["):
            raise
        return {"tool": "submit", "args": {"path_or_answer": text}}
    if isinstance(payload, dict):
        return parse_verifiers_tool_call(payload)
    raise ValueError("JSON completion must be an object")


def _extract_fenced_json(text: str) -> str | None:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else None


def _completion_text(completion: object) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content") or completion)
    return str(getattr(completion, "content", completion))
