from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from metrics import (
    ALLOWED_TOOLS,
    FOCUS_SCENARIOS,
    counter_to_dict,
    find_absolute_path_values,
    format_reward,
    invalid_run_python_path,
    numeric_stats,
    summarize_examples,
)
from sequentialize_tools import public_tool_call, sequentialize_action_window

DEFAULT_EVAL_ID = "kxhqr8w6kxeficm93rp7s5k6"
RAW_FILENAME = "glm52_perfect_raw_actions.jsonl"
SEQUENTIAL_FILENAME = "glm52_perfect_sequential_actions.jsonl"
REPORT_JSON = "perfect_dataset_report.json"
REPORT_MD = "perfect_dataset_report.md"

REWARD_PATHS = [
    ("reward",),
    ("score",),
    ("result", "reward"),
    ("eval_result", "reward"),
    ("metrics", "reward"),
    ("swg_reward",),
    ("info", "metrics", "swg_reward"),
]

TASK_METADATA_KEYS = {
    "task_id",
    "split",
    "family",
    "scenario",
    "difficulty",
    "seed",
}


class TraceFormatError(ValueError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GLM-5.2 perfect-trace SFT action datasets.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--teacher", default="glm-5.2")
    parser.add_argument("--student", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--eval-id", default=DEFAULT_EVAL_ID)
    parser.add_argument("--reward-filter", choices=["perfect", "min", "all"], default="perfect")
    parser.add_argument("--reward-min", type=float)
    parser.add_argument("--write-raw", action="store_true")
    parser.add_argument("--write-sequential", action="store_true")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--allow-non-390", action="store_true")
    return parser.parse_args(argv)


def get_nested(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def extract_reward(sample: dict[str, Any]) -> float:
    for path in REWARD_PATHS:
        value = get_nested(sample, path)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise TraceFormatError(f"Reward value at {'.'.join(path)} is not numeric: {value!r}") from exc
    keys = ", ".join(sorted(str(key) for key in sample.keys()))
    raise TraceFormatError(
        "No reward key found in sample. "
        f"Available sample keys: {keys}. "
        "Expected one of reward, score, result.reward, eval_result.reward, metrics.reward."
    )


def load_samples(input_dir: Path, verbose: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    samples: list[dict[str, Any]] = []
    files_loaded = 0
    skipped_files: list[str] = []
    eval_ids: set[str] = set()
    json_files = sorted(input_dir.glob("*.json"))

    for path in json_files:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        page_samples = payload.get("samples") if isinstance(payload, dict) else None
        if not isinstance(page_samples, list):
            skipped_files.append(path.name)
            if verbose:
                print(f"Skipping {path.name}: no samples list")
            continue
        files_loaded += 1
        page_eval_id = payload.get("evaluation_id")
        if page_eval_id is not None:
            eval_ids.add(str(page_eval_id))
        for sample in page_samples:
            if not isinstance(sample, dict):
                raise TraceFormatError(f"{path.name} contains a non-object sample: {sample!r}")
            item = dict(sample)
            item["_source_file"] = path.name
            item["_source_evaluation_id"] = page_eval_id
            samples.append(item)

    return samples, {
        "json_files_found": len(json_files),
        "files_loaded": files_loaded,
        "skipped_files": skipped_files,
        "evaluation_ids": sorted(eval_ids),
    }


def validate_run_shape(samples: list[dict[str, Any]], load_info: dict[str, Any], args: argparse.Namespace) -> None:
    eval_ids = set(load_info["evaluation_ids"])
    if args.eval_id:
        if eval_ids and eval_ids != {args.eval_id}:
            raise TraceFormatError(f"Expected eval ID {args.eval_id}, found {sorted(eval_ids)}")
    if len(eval_ids) > 1:
        raise TraceFormatError(f"Samples span multiple evaluation IDs: {sorted(eval_ids)}")

    example_ids = [sample.get("example_id") for sample in samples]
    unique_example_ids = {item for item in example_ids if item is not None}
    if not args.allow_non_390:
        if len(samples) != 390:
            raise TraceFormatError(f"Expected exactly 390 samples, found {len(samples)}. Use --allow-non-390 to inspect partial exports.")
        if len(unique_example_ids) != 390:
            raise TraceFormatError(
                f"Expected exactly 390 unique example IDs, found {len(unique_example_ids)}. "
                "Use --allow-non-390 to inspect partial exports."
            )


def reward_matches(reward: float, args: argparse.Namespace) -> bool:
    if args.reward_min is not None:
        return reward >= args.reward_min
    if args.reward_filter == "all":
        return True
    if args.reward_filter == "min":
        raise TraceFormatError("--reward-filter min requires --reward-min")
    return math.isclose(reward, 1.0, rel_tol=0.0, abs_tol=1e-9)


def extract_task_metadata(sample: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    task = sample.get("task")
    if isinstance(task, dict):
        for key in TASK_METADATA_KEYS:
            if task.get(key) is not None:
                metadata[key] = task[key]

    prompt_text = "\n".join(
        str(message.get("content", ""))
        for message in coerce_messages(sample.get("prompt"), "prompt", sample)
        if message.get("role") == "user"
    )
    for line in prompt_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("-") or ":" not in stripped:
            continue
        key, value = stripped[1:].split(":", 1)
        key = key.strip()
        value = value.strip()
        if key not in TASK_METADATA_KEYS:
            continue
        if key in {"difficulty", "seed"}:
            try:
                metadata[key] = int(value)
            except ValueError:
                metadata[key] = value
        else:
            metadata[key] = value

    marker = "Required final artifact:"
    for line in prompt_text.splitlines():
        if marker in line:
            metadata["required_final_artifact"] = line.split(marker, 1)[1].strip()
            break
    return metadata


def coerce_messages(value: Any, field_name: str, sample: dict[str, Any]) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TraceFormatError(
            f"Sample example_id={sample.get('example_id')} has non-list {field_name}: {type(value).__name__}"
        )
    messages: list[dict[str, Any]] = []
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise TraceFormatError(
                f"Sample example_id={sample.get('example_id')} {field_name}[{index}] is not an object"
            )
        messages.append(message)
    return messages


def parse_tool_call_objects(value: Any) -> tuple[list[dict[str, Any]], int]:
    if value in (None, "", []):
        return [], 0
    if isinstance(value, dict):
        return [value], 0
    if isinstance(value, list):
        parsed: list[dict[str, Any]] = []
        malformed = 0
        for item in value:
            objects, item_malformed = parse_tool_call_objects(item)
            parsed.extend(objects)
            malformed += item_malformed
        return parsed, malformed
    if not isinstance(value, str):
        return [], 1

    text = value.strip()
    if not text:
        return [], 0
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        return parse_tool_call_objects(loaded)

    decoder = json.JSONDecoder()
    index = 0
    parsed = []
    malformed = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        try:
            item, next_index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            malformed += 1
            break
        if isinstance(item, dict):
            parsed.append(item)
        else:
            malformed += 1
        index = next_index
    return parsed, malformed


def normalize_tool_call(raw_call: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    function = raw_call.get("function")
    name = raw_call.get("name")
    arguments = raw_call.get("arguments")
    if isinstance(function, dict):
        name = name or function.get("name")
        arguments = arguments if arguments is not None else function.get("arguments")
    if name is None and raw_call.get("tool") is not None:
        name = raw_call.get("tool")
    if arguments is None and raw_call.get("args") is not None:
        arguments = raw_call.get("args")

    if not isinstance(name, str) or not name:
        return None, "missing tool name"

    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            parsed_arguments: dict[str, Any] = {}
        else:
            try:
                loaded_arguments = json.loads(text)
            except json.JSONDecodeError as exc:
                return None, f"arguments are not JSON: {exc}"
            if not isinstance(loaded_arguments, dict):
                return None, "arguments JSON is not an object"
            parsed_arguments = loaded_arguments
    elif isinstance(arguments, dict):
        parsed_arguments = copy.deepcopy(arguments)
    elif arguments is None:
        parsed_arguments = {}
    else:
        return None, "arguments are neither object nor JSON string"

    if name == "run_python" and "path" not in parsed_arguments and "command_or_script" in parsed_arguments:
        parsed_arguments["path"] = parsed_arguments.pop("command_or_script")

    call_id = raw_call.get("id") or raw_call.get("tool_call_id")
    return {"id": call_id, "name": name, "arguments": parsed_arguments}, None


def normalize_tool_calls(value: Any, quality: Counter[str]) -> list[dict[str, Any]]:
    raw_calls, malformed = parse_tool_call_objects(value)
    quality["malformed_tool_calls"] += malformed
    calls: list[dict[str, Any]] = []
    for raw_call in raw_calls:
        call, error = normalize_tool_call(raw_call)
        if error:
            quality["malformed_tool_calls"] += 1
            continue
        calls.append(call)
    return calls


def validate_target_calls(calls: list[dict[str, Any]], quality: Counter[str]) -> bool:
    valid = True
    for call in calls:
        name = call["name"]
        arguments = call.get("arguments", {})
        if name not in ALLOWED_TOOLS:
            quality["unknown_tools"] += 1
            valid = False
        absolute_paths = find_absolute_path_values(arguments)
        if absolute_paths:
            quality["absolute_path_attempts"] += len(absolute_paths)
            valid = False
        if name == "run_python" and invalid_run_python_path(arguments):
            quality["invalid_run_python_calls"] += 1
            valid = False
    return valid


def clean_message(message: dict[str, Any], quality: Counter[str], count_reasoning: bool = True) -> dict[str, Any]:
    if count_reasoning and "reasoning_content" in message:
        quality["reasoning_content_fields_stripped"] += 1
    role = str(message.get("role", ""))
    cleaned: dict[str, Any] = {"role": role}
    content = message.get("content", "")
    cleaned["content"] = "" if content is None else str(content)

    if role == "tool":
        if message.get("tool_call_id"):
            cleaned["tool_call_id"] = message.get("tool_call_id")
        return cleaned

    calls = normalize_tool_calls(message.get("tool_calls"), quality)
    if role == "assistant" and calls:
        cleaned["content"] = ""
        cleaned["tool_calls"] = [public_tool_call(call) for call in calls]
    return cleaned


def clean_prompt_messages(sample: dict[str, Any], quality: Counter[str]) -> list[dict[str, Any]]:
    return [clean_message(message, quality) for message in coerce_messages(sample.get("prompt"), "prompt", sample)]


def collect_following_tool_messages(completion: list[dict[str, Any]], start_index: int, quality: Counter[str]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    index = start_index + 1
    while index < len(completion) and completion[index].get("role") == "tool":
        messages.append(clean_message(completion[index], quality, count_reasoning=False))
        index += 1
    return messages


def base_metadata(
    sample: dict[str, Any],
    reward: float,
    task_metadata: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    metadata = {
        "teacher": args.teacher,
        "student": args.student,
        "evaluation_id": args.eval_id or sample.get("_source_evaluation_id"),
        "trace_id": sample.get("trace_id"),
        "example_id": sample.get("example_id"),
        "reward": reward,
    }
    metadata.update(task_metadata)
    return metadata


def validate_written_example(example: dict[str, Any]) -> None:
    messages = example.get("messages")
    target = example.get("target", {})
    metadata = example.get("metadata", {})
    if not messages:
        raise TraceFormatError("Written example has empty messages")
    if target.get("role") != "assistant":
        raise TraceFormatError("Written example target role is not assistant")
    if "reasoning_content" in target:
        raise TraceFormatError("Written example target contains reasoning_content")
    tool_calls = target.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        raise TraceFormatError("Written example target has no tool calls")
    for call in tool_calls:
        if call.get("name") not in ALLOWED_TOOLS:
            raise TraceFormatError(f"Written example has invalid tool name: {call.get('name')}")
    for key in ("example_id", "trace_id", "reward", "family", "scenario", "difficulty"):
        if metadata.get(key) in (None, ""):
            raise TraceFormatError(f"Written example metadata missing {key}")
    if target.get("role") == "tool":
        raise TraceFormatError("Tool response cannot be used as target")


def build_examples(samples: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    quality: Counter[str] = Counter()
    raw_examples: list[dict[str, Any]] = []
    sequential_examples: list[dict[str, Any]] = []
    total_by_scenario: Counter[str] = Counter()
    perfect_by_scenario: Counter[str] = Counter()
    raw_examples_by_scenario: Counter[str] = Counter()
    sequential_examples_by_scenario: Counter[str] = Counter()
    rewards: list[float] = []
    reward_distribution: Counter[str] = Counter()
    selected_trace_ids: set[str] = set()

    for sample in samples:
        try:
            reward = extract_reward(sample)
        except TraceFormatError:
            quality["samples_with_missing_reward"] += 1
            raise
        rewards.append(reward)
        reward_distribution[format_reward(reward)] += 1

        task_metadata = extract_task_metadata(sample)
        if any(task_metadata.get(key) in (None, "") for key in ("family", "scenario", "difficulty", "seed")):
            quality["samples_with_missing_task_metadata"] += 1
        scenario = str(task_metadata.get("scenario", "unknown"))
        total_by_scenario[scenario] += 1

        if math.isclose(reward, 1.0, rel_tol=0.0, abs_tol=1e-9):
            perfect_by_scenario[scenario] += 1
        if not reward_matches(reward, args):
            continue
        selected_trace_ids.add(str(sample.get("trace_id")))

        history = clean_prompt_messages(sample, quality)
        completion = coerce_messages(sample.get("completion"), "completion", sample)
        metadata = base_metadata(sample, reward, task_metadata, args)

        for index, message in enumerate(completion):
            role = message.get("role")
            if role == "assistant":
                if "reasoning_content" in message:
                    quality["reasoning_content_fields_stripped"] += 1
                calls = normalize_tool_calls(message.get("tool_calls"), quality)
                assistant_history_message = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [public_tool_call(call) for call in calls],
                } if calls else clean_message(message, quality, count_reasoning=False)

                if calls:
                    if validate_target_calls(calls, quality):
                        raw_metadata = copy.deepcopy(metadata)
                        raw_metadata["raw_multi_tool"] = len(calls) > 1
                        raw_example = {
                            "messages": copy.deepcopy(history),
                            "target": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [public_tool_call(call) for call in calls],
                            },
                            "metadata": raw_metadata,
                        }
                        validate_written_example(raw_example)
                        raw_examples.append(raw_example)
                        raw_examples_by_scenario[scenario] += 1

                        following_tools = collect_following_tool_messages(completion, index, quality)
                        sequential_metadata = copy.deepcopy(metadata)
                        sequential_metadata["raw_multi_tool"] = len(calls) > 1
                        split_examples, warnings = sequentialize_action_window(
                            history,
                            calls,
                            following_tools,
                            sequential_metadata,
                        )
                        quality["sequentialization_warnings"] += warnings
                        for split_example in split_examples:
                            validate_written_example(split_example)
                        sequential_examples.extend(split_examples)
                        sequential_examples_by_scenario[scenario] += len(split_examples)
                    history.append(assistant_history_message)
                else:
                    if str(message.get("content", "")).strip():
                        quality["assistant_prose_only_turns_skipped"] += 1
                    history.append(assistant_history_message)
            elif role == "tool":
                history.append(clean_message(message, quality))
            else:
                history.append(clean_message(message, quality))

    example_ids = [sample.get("example_id") for sample in samples]
    trace_ids = [sample.get("trace_id") for sample in samples]
    duplicate_example_ids = count_duplicates(example_ids)
    duplicate_trace_ids = count_duplicates(trace_ids)

    build_info = {
        "quality": quality,
        "rewards": rewards,
        "reward_distribution": reward_distribution,
        "selected_trace_ids": selected_trace_ids,
        "total_by_scenario": total_by_scenario,
        "perfect_by_scenario": perfect_by_scenario,
        "raw_examples_by_scenario": raw_examples_by_scenario,
        "sequential_examples_by_scenario": sequential_examples_by_scenario,
        "duplicate_example_ids": duplicate_example_ids,
        "duplicate_trace_ids": duplicate_trace_ids,
    }
    return raw_examples, sequential_examples, build_info


def count_duplicates(values: list[Any]) -> int:
    counter = Counter(value for value in values if value is not None)
    return sum(1 for count in counter.values() if count > 1)


def build_report(
    samples: list[dict[str, Any]],
    load_info: dict[str, Any],
    raw_examples: list[dict[str, Any]],
    sequential_examples: list[dict[str, Any]],
    build_info: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    quality = build_info["quality"]
    rewards = build_info["rewards"]
    reward_distribution = build_info["reward_distribution"]
    total_by_scenario = build_info["total_by_scenario"]
    perfect_by_scenario = build_info["perfect_by_scenario"]
    raw_by_scenario = build_info["raw_examples_by_scenario"]
    sequential_by_scenario = build_info["sequential_examples_by_scenario"]

    scenario_names = sorted(set(total_by_scenario) | set(perfect_by_scenario) | set(raw_by_scenario) | set(sequential_by_scenario))
    scenario_coverage = [
        {
            "scenario": scenario,
            "total_traces": total_by_scenario[scenario],
            "perfect_traces": perfect_by_scenario[scenario],
            "action_examples_raw": raw_by_scenario[scenario],
            "action_examples_sequential": sequential_by_scenario[scenario],
        }
        for scenario in scenario_names
    ]
    low_coverage = [
        row["scenario"]
        for row in scenario_coverage
        if row["total_traces"] > 0 and row["perfect_traces"] < max(1, math.ceil(row["total_traces"] * 0.25))
    ]
    zero_coverage = [row["scenario"] for row in scenario_coverage if row["total_traces"] > 0 and row["perfect_traces"] == 0]
    focus_warnings = [
        row
        for row in scenario_coverage
        if row["scenario"] in FOCUS_SCENARIOS and (row["perfect_traces"] == 0 or row["perfect_traces"] < 3)
    ]

    data_quality = {
        "malformed_tool_calls": quality["malformed_tool_calls"],
        "unknown_tools": quality["unknown_tools"],
        "absolute_path_attempts": quality["absolute_path_attempts"],
        "invalid_run_python_calls": quality["invalid_run_python_calls"],
        "assistant_prose_only_turns_skipped": quality["assistant_prose_only_turns_skipped"],
        "reasoning_content_fields_stripped": quality["reasoning_content_fields_stripped"],
        "samples_with_missing_reward": quality["samples_with_missing_reward"],
        "samples_with_missing_task_metadata": quality["samples_with_missing_task_metadata"],
        "duplicate_example_ids": build_info["duplicate_example_ids"],
        "duplicate_trace_ids": build_info["duplicate_trace_ids"],
        "sequentialization_warnings": quality["sequentialization_warnings"],
    }
    recommendation = make_recommendation(
        raw_examples,
        sequential_examples,
        data_quality,
        low_coverage,
        zero_coverage,
    )

    return {
        "run_level_stats": {
            "evaluation_id": args.eval_id,
            "total_files_loaded": load_info["files_loaded"],
            "total_samples": len(samples),
            "unique_example_ids": len({sample.get("example_id") for sample in samples if sample.get("example_id") is not None}),
            "reward_distribution": counter_to_dict(reward_distribution),
            "perfect_examples_count": sum(1 for reward in rewards if math.isclose(reward, 1.0, rel_tol=0.0, abs_tol=1e-9)),
            "non_perfect_count": sum(1 for reward in rewards if not math.isclose(reward, 1.0, rel_tol=0.0, abs_tol=1e-9)),
            **numeric_stats(rewards),
        },
        "dataset_stats": {
            "raw": summarize_examples(raw_examples),
            "sequential": summarize_examples(sequential_examples),
        },
        "data_quality_stats": data_quality,
        "scenario_coverage": scenario_coverage,
        "coverage_warnings": {
            "low_coverage_scenarios": low_coverage,
            "zero_coverage_scenarios": zero_coverage,
            "focus_scenario_warnings": focus_warnings,
        },
        "recommendation": recommendation,
    }


def make_recommendation(
    raw_examples: list[dict[str, Any]],
    sequential_examples: list[dict[str, Any]],
    data_quality: dict[str, Any],
    low_coverage: list[str],
    zero_coverage: list[str],
) -> dict[str, Any]:
    critical_quality = (
        data_quality["malformed_tool_calls"]
        + data_quality["unknown_tools"]
        + data_quality["absolute_path_attempts"]
        + data_quality["invalid_run_python_calls"]
    )
    ready_for_sft = bool(raw_examples and sequential_examples and critical_quality == 0)
    raw_stats = summarize_examples(raw_examples)
    sequential_preferable = (
        raw_stats["max_target_tool_calls"] > 1
        and summarize_examples(sequential_examples)["max_target_tool_calls"] == 1
    )
    sparse = bool(low_coverage or zero_coverage)
    next_variant = (
        "Build a scenario-balanced partial-trace dataset that adds high-reward non-perfect traces for sparse scenarios."
        if sparse
        else "Build recovery-state examples from Qwen failure traces after the perfect-only SFT baseline."
    )
    return {
        "ready_for_sft": ready_for_sft,
        "perfect_only_coverage_too_sparse": sparse,
        "sequentialized_variant_preferable": sequential_preferable,
        "next_dataset_variant": next_variant,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    run = report["run_level_stats"]
    quality = report["data_quality_stats"]
    raw = report["dataset_stats"]["raw"]
    sequential = report["dataset_stats"]["sequential"]
    recommendation = report["recommendation"]

    lines = [
        "# Perfect Dataset Report",
        "",
        "## Run-Level Stats",
        "",
        f"- Evaluation ID: `{run['evaluation_id']}`",
        f"- Total files loaded: {run['total_files_loaded']}",
        f"- Total samples: {run['total_samples']}",
        f"- Unique example IDs: {run['unique_example_ids']}",
        f"- Perfect examples: {run['perfect_examples_count']}",
        f"- Non-perfect examples: {run['non_perfect_count']}",
        f"- Reward min / mean / median / max: {run['min']} / {run['mean']} / {run['median']} / {run['max']}",
        "",
        "## Dataset Stats",
        "",
        "| variant | SFT examples | traces used | avg examples/trace | avg history messages | avg target tool calls | max target tool calls | submit | write_file | run_shell | run_python |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        dataset_row("raw", raw),
        dataset_row("sequential", sequential),
        "",
        "## Data Quality",
        "",
    ]
    for key, value in quality.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Scenario Coverage",
            "",
            "| scenario | total traces | perfect traces | action examples raw | action examples sequential |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["scenario_coverage"]:
        scenario = row["scenario"]
        label = f"**{scenario}**" if scenario in FOCUS_SCENARIOS and (row["perfect_traces"] == 0 or row["perfect_traces"] < 3) else scenario
        lines.append(
            f"| {label} | {row['total_traces']} | {row['perfect_traces']} | "
            f"{row['action_examples_raw']} | {row['action_examples_sequential']} |"
        )

    coverage = report["coverage_warnings"]
    lines.extend(["", "## Coverage Warnings", ""])
    if coverage["zero_coverage_scenarios"]:
        lines.append("- Zero perfect coverage: " + ", ".join(coverage["zero_coverage_scenarios"]))
    if coverage["low_coverage_scenarios"]:
        lines.append("- Low perfect coverage: " + ", ".join(coverage["low_coverage_scenarios"]))
    if not coverage["zero_coverage_scenarios"] and not coverage["low_coverage_scenarios"]:
        lines.append("- No low or zero perfect-coverage scenarios detected.")

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Ready for SFT: {recommendation['ready_for_sft']}",
            f"- Perfect-only coverage too sparse: {recommendation['perfect_only_coverage_too_sparse']}",
            f"- Sequentialized variant preferable: {recommendation['sequentialized_variant_preferable']}",
            f"- Next dataset variant: {recommendation['next_dataset_variant']}",
            "",
        ]
    )
    return "\n".join(lines)


def dataset_row(name: str, stats: dict[str, Any]) -> str:
    return (
        f"| {name} | {stats['sft_examples']} | {stats['traces_used']} | "
        f"{stats['avg_examples_per_trace']:.2f} | {stats['avg_history_messages']:.2f} | "
        f"{stats['avg_target_tool_calls']:.2f} | {stats['max_target_tool_calls']} | "
        f"{stats['submit_targets']} | {stats['write_file_targets']} | "
        f"{stats['run_shell_targets']} | {stats['run_python_targets']} |"
    )


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")


def write_report(report_dir: Path, report: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / REPORT_JSON).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    (report_dir / REPORT_MD).write_text(render_markdown_report(report), encoding="utf-8")


def print_summary(load_info: dict[str, Any], report: dict[str, Any], args: argparse.Namespace) -> None:
    run = report["run_level_stats"]
    raw = report["dataset_stats"]["raw"]
    sequential = report["dataset_stats"]["sequential"]
    quality = report["data_quality_stats"]
    invalid_tool_calls = quality["malformed_tool_calls"] + quality["unknown_tools"]
    print(f"Loaded {run['total_samples']} samples from {load_info['files_loaded']} files.")
    print(f"Evaluation ID: {run['evaluation_id']}")
    print(f"Perfect traces: {run['perfect_examples_count']} / {run['total_samples']}")
    print(f"Raw action examples: {raw['sft_examples']}")
    print(f"Sequential action examples: {sequential['sft_examples']}")
    print(f"Reasoning fields stripped: {quality['reasoning_content_fields_stripped']}")
    print(f"Invalid tool calls: {invalid_tool_calls}")
    print(f"Absolute path calls: {quality['absolute_path_attempts']}")
    print(f"Invalid run_python calls: {quality['invalid_run_python_calls']}")
    if args.dry_run:
        print("Dry run completed; no datasets or reports were written.")
    else:
        print("Wrote dataset and report successfully.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    samples, load_info = load_samples(args.input_dir, verbose=args.verbose)
    validate_run_shape(samples, load_info, args)
    if args.max_examples is not None:
        samples = samples[: args.max_examples]

    raw_examples, sequential_examples, build_info = build_examples(samples, args)
    report = build_report(samples, load_info, raw_examples, sequential_examples, build_info, args)

    if not args.dry_run:
        if args.write_raw:
            write_jsonl(args.output_dir / RAW_FILENAME, raw_examples)
        if args.write_sequential:
            write_jsonl(args.output_dir / SEQUENTIAL_FILENAME, sequential_examples)
        write_report(args.report_dir, report)

    print_summary(load_info, report, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
