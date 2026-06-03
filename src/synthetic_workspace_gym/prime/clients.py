from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any, Protocol


class PrimeModelClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


class ScriptedPrimeClient:
    name = "scripted"
    client_type = "scripted"
    privileged = False

    def __init__(self, actions: Sequence[dict[str, Any]] | None = None) -> None:
        self.actions = list(actions or [])
        self.index = 0

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.index >= len(self.actions):
            action = {"tool": "submit", "args": {"path_or_answer": "done"}}
        else:
            action = self.actions[self.index]
            self.index += 1
        return normalize_client_response(action)


class HeuristicReferenceClient:
    name = "heuristic-reference"
    client_type = "heuristic_reference"
    privileged = True

    def __init__(self) -> None:
        self.actions: list[dict[str, Any]] | None = None
        self.index = 0

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.actions is None:
            reference_solution = dict((metadata or {}).get("reference_solution", {}) or {})
            files = dict(reference_solution.get("files", {}) or {})
            self.actions = [
                {"tool": "write_file", "args": {"path": str(path), "content": str(content)}}
                for path, content in sorted(files.items())
            ]
            self.actions.append({"tool": "submit", "args": {"path_or_answer": "reference_solution"}})

        if self.index >= len(self.actions):
            action = {"tool": "submit", "args": {"path_or_answer": "done"}}
        else:
            action = self.actions[self.index]
            self.index += 1
        return normalize_client_response(action)


class JSONActionClient:
    name = "json-action"
    client_type = "json_action"
    privileged = False

    def __init__(
        self,
        fn: Callable[[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None], dict[str, Any] | str],
        *,
        name: str | None = None,
    ) -> None:
        self.fn = fn
        if name is not None:
            self.name = name

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return normalize_client_response(self.fn(messages, tools, metadata))


def normalize_client_response(response: dict[str, Any] | str) -> dict[str, Any]:
    raw = response
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            return {"type": "message", "content": response, "raw": raw}

    if not isinstance(response, dict):
        return {"type": "message", "content": str(response), "raw": raw}

    if response.get("type") == "tool_call":
        return {
            "type": "tool_call",
            "tool": str(response.get("tool", "")),
            "args": dict(response.get("args", {}) or {}),
            "raw": response.get("raw", raw),
        }
    if response.get("type") == "message":
        return {
            "type": "message",
            "content": str(response.get("content", "")),
            "raw": response.get("raw", raw),
        }
    if "tool" in response:
        return {
            "type": "tool_call",
            "tool": str(response.get("tool", "")),
            "args": dict(response.get("args", {}) or {}),
            "raw": raw,
        }
    return {"type": "message", "content": str(response.get("content", "")), "raw": raw}
