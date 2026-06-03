from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from synthetic_workspace_gym.prime.tools import get_tool_schemas

from .compat import vf


def swg_tool_schemas() -> list[dict[str, Any]]:
    return [dict(schema) for schema in get_tool_schemas()]  # type: ignore[arg-type]


def to_verifiers_tool_schema(schema: dict[str, Any]) -> object:
    schema = deepcopy(schema)
    if vf is not None:
        for attr in ("ToolSchema", "Tool"):
            tool_cls = getattr(vf, attr, None)
            if tool_cls is not None:
                for kwargs in (
                    schema,
                    {
                        "name": schema.get("name"),
                        "description": schema.get("description", ""),
                        "parameters": schema.get("parameters", {}),
                    },
                ):
                    try:
                        return tool_cls(**kwargs)
                    except TypeError:
                        continue
        make_tool = getattr(vf, "tool", None)
        if make_tool is not None:
            try:
                return make_tool(schema)
            except TypeError:
                pass
    return {
        "type": "function",
        "function": {
            "name": schema.get("name"),
            "description": schema.get("description", ""),
            "parameters": schema.get("parameters", {}),
        },
    }


def get_verifiers_tools(tool_permissions: list[str] | None = None) -> list[object]:
    return [to_verifiers_tool_schema(schema) for schema in get_tool_schemas(tool_permissions)]  # type: ignore[arg-type]


def parse_verifiers_tool_call(payload: object) -> dict[str, Any]:
    if isinstance(payload, dict):
        if "tool" in payload:
            return {"tool": str(payload.get("tool", "")), "args": _coerce_args(payload.get("args", {}))}
        if "name" in payload:
            return {"tool": str(payload.get("name", "")), "args": _coerce_args(payload.get("arguments", {}))}
        function = payload.get("function")
        if isinstance(function, dict):
            return {
                "tool": str(function.get("name", "")),
                "args": _coerce_args(function.get("arguments", {})),
            }

    function = getattr(payload, "function", None)
    if function is not None:
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", {})
        if name is not None:
            return {"tool": str(name), "args": _coerce_args(arguments)}

    for name_attr, args_attr in (("tool", "args"), ("name", "arguments")):
        name = getattr(payload, name_attr, None)
        if name is not None:
            return {"tool": str(name), "args": _coerce_args(getattr(payload, args_attr, {}))}

    raise ValueError(f"Unsupported tool call payload: {type(payload).__name__}")


def _coerce_args(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        if not value.strip():
            return {}
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"value": value}
        return dict(parsed) if isinstance(parsed, dict) else {"value": parsed}
    return dict(value) if hasattr(value, "items") else {"value": value}
