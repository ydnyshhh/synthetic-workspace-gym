from __future__ import annotations

import re
from collections import Counter
from statistics import mean, median
from typing import Any

ALLOWED_TOOLS = {
    "read_file",
    "write_file",
    "append_file",
    "list_directory",
    "run_shell",
    "run_python",
    "submit",
}

FOCUS_SCENARIOS = {
    "migration_plan_bundle",
    "channel_status_pivot",
    "timestamp_normalization",
    "service_config_reconciliation",
}


def average(values: list[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def numeric_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "median": None, "max": None}
    return {
        "min": min(values),
        "mean": mean(values),
        "median": median(values),
        "max": max(values),
    }


def format_reward(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def counter_to_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=lambda item: str(item))}


def summarize_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    target_tool_names: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    history_lengths: list[int] = []
    target_tool_counts: list[int] = []
    trace_ids: set[str] = set()

    for example in examples:
        metadata = example.get("metadata", {})
        if metadata.get("trace_id") is not None:
            trace_ids.add(str(metadata["trace_id"]))
        family_counts[str(metadata.get("family", "unknown"))] += 1
        scenario_counts[str(metadata.get("scenario", "unknown"))] += 1
        difficulty_counts[str(metadata.get("difficulty", "unknown"))] += 1
        history_lengths.append(len(example.get("messages", [])))
        tool_calls = example.get("target", {}).get("tool_calls", [])
        target_tool_counts.append(len(tool_calls))
        for call in tool_calls:
            target_tool_names[str(call.get("name", "unknown"))] += 1

    return {
        "sft_examples": len(examples),
        "traces_used": len(trace_ids),
        "avg_examples_per_trace": average([len(examples) / len(trace_ids)]) if trace_ids else 0.0,
        "avg_history_messages": average(history_lengths),
        "avg_target_tool_calls": average(target_tool_counts),
        "max_history_messages": max(history_lengths) if history_lengths else 0,
        "max_target_tool_calls": max(target_tool_counts) if target_tool_counts else 0,
        "count_by_family": counter_to_dict(family_counts),
        "count_by_scenario": counter_to_dict(scenario_counts),
        "count_by_difficulty": counter_to_dict(difficulty_counts),
        "count_by_target_tool_name": counter_to_dict(target_tool_names),
        "submit_targets": target_tool_names["submit"],
        "write_file_targets": target_tool_names["write_file"],
        "run_shell_targets": target_tool_names["run_shell"],
        "run_python_targets": target_tool_names["run_python"],
    }


def looks_like_absolute_path(value: str) -> bool:
    text = value.strip().strip("'\"")
    if not text:
        return False
    if text.startswith("/") or text.startswith("\\\\") or text.startswith("~/"):
        return True
    if re.match(r"^[A-Za-z]:[\\/]", text):
        return True
    return bool(re.search(r"(^|\s)(/[^\s]+|[A-Za-z]:[\\/][^\s]+|\\\\[^\s]+)", text))


def find_absolute_path_entries(value: Any, path: str = "") -> list[dict[str, str]]:
    if path.endswith(".content") or path == "content":
        return []
    if isinstance(value, dict):
        found: list[dict[str, str]] = []
        for child_key, child_value in value.items():
            child_path = f"{path}.{child_key}" if path else str(child_key)
            found.extend(find_absolute_path_entries(child_value, child_path))
        return found
    if isinstance(value, list):
        found = []
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            found.extend(find_absolute_path_entries(item, child_path))
        return found
    if isinstance(value, str) and looks_like_absolute_path(value):
        return [{"argument_path": path, "value": value}]
    return []


def find_absolute_path_values(value: Any, key: str = "") -> list[str]:
    if key == "content":
        return []
    if isinstance(value, dict):
        found: list[str] = []
        for child_key, child_value in value.items():
            found.extend(find_absolute_path_values(child_value, str(child_key)))
        return found
    if isinstance(value, list):
        found = []
        for item in value:
            found.extend(find_absolute_path_values(item, key))
        return found
    if isinstance(value, str) and looks_like_absolute_path(value):
        return [value]
    return []


def invalid_run_python_path(arguments: dict[str, Any]) -> bool:
    path = arguments.get("path")
    if not isinstance(path, str):
        return True
    text = path.strip()
    lowered = text.lower()
    if not text or text != path:
        return True
    if any(char.isspace() for char in text):
        return True
    if lowered.startswith(("python ", "python3 ", "py ")):
        return True
    if lowered.startswith("-m") or lowered.startswith("-c"):
        return True
    if " -m " in lowered or " -c " in lowered:
        return True
    return False
