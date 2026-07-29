from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


TASK_RE = re.compile(r"- (task_id|split|family|scenario|difficulty|seed):\s*([^\r\n]+)")
STRICT_TOOL_ERROR_RE = re.compile(
    r"(?im)^(?:error:|traceback|file not found:|directory not found:|"
    r"python script not found:|shell command rejected|command rejected)"
)
CODE_FAMILIES = {"pipeline", "script_repair", "tabular", "composite_workspace"}


def load_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def decode_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def task_prompt(sample: dict[str, Any]) -> str:
    task = decode_json(sample.get("task"))
    if isinstance(task, dict) and task.get("prompt"):
        return str(task["prompt"])
    prompt = decode_json(sample.get("prompt"))
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content") or "")
    if isinstance(prompt, str):
        return prompt
    raw_input = decode_json(sample.get("input"))
    if isinstance(raw_input, list):
        for message in raw_input:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content") or "")
    if isinstance(raw_input, dict):
        return str(raw_input.get("prompt") or raw_input.get("content") or "")
    return ""


def completion_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("completion", "output"):
        value = decode_json(sample.get(key))
        if isinstance(value, list):
            return [message for message in value if isinstance(message, dict)]
        if isinstance(value, dict):
            for nested in ("messages", "completion"):
                messages = decode_json(value.get(nested))
                if isinstance(messages, list):
                    return [
                        message for message in messages if isinstance(message, dict)
                    ]
    return []


def parse_metadata(prompt: str) -> dict[str, str]:
    return {
        match.group(1): match.group(2).strip() for match in TASK_RE.finditer(prompt)
    }


def parse_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        for raw in message.get("tool_calls") or []:
            call = decode_json(raw)
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                arguments = function.get("arguments")
            else:
                name = call.get("name")
                arguments = call.get("arguments")
            if name:
                calls.append(
                    {
                        "name": str(name),
                        "arguments": decode_json(arguments),
                        "message_index": message_index,
                    }
                )
    return calls


def path_argument(call: dict[str, Any]) -> str:
    arguments = call.get("arguments")
    if isinstance(arguments, dict):
        for key in ("path", "script_path", "path_or_answer"):
            if arguments.get(key):
                return str(arguments[key])
    return ""


def sample_reward(sample: dict[str, Any]) -> float:
    for key in ("reward", "swg_reward", "score"):
        value = sample.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    metrics = decode_json(sample.get("metrics"))
    if isinstance(metrics, dict) and isinstance(
        metrics.get("swg_reward"), (int, float)
    ):
        return float(metrics["swg_reward"])
    info = decode_json(sample.get("info"))
    if isinstance(info, dict):
        nested = decode_json(info.get("metrics"))
        if isinstance(nested, dict) and isinstance(
            nested.get("swg_reward"), (int, float)
        ):
            return float(nested["swg_reward"])
    return 0.0


def assistant_tail(messages: list[dict[str, Any]], limit: int = 280) -> str:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        text = str(message.get("reasoning_content") or message.get("content") or "")
        text = re.sub(r"\s+", " ", text).strip()
        return text[-limit:]
    return ""


def normalize_sample(
    sample: dict[str, Any],
    *,
    source: str,
    label: str,
    run_id: str,
    step: int | None,
) -> dict[str, Any]:
    prompt = task_prompt(sample)
    metadata = parse_metadata(prompt)
    messages = completion_messages(sample)
    calls = parse_tool_calls(messages)
    names = [call["name"] for call in calls]
    family = metadata.get("family", "unknown")
    reward = sample_reward(sample)
    tool_messages = [
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "tool"
    ]
    strict_errors = [
        text for text in tool_messages if STRICT_TOOL_ERROR_RE.search(text)
    ]
    write_indices = [
        index
        for index, name in enumerate(names)
        if name in {"write_file", "append_file"}
    ]
    run_indices = [
        index for index, name in enumerate(names) if name in {"run_shell", "run_python"}
    ]
    submit_indices = [index for index, name in enumerate(names) if name == "submit"]
    first_write = write_indices[0] if write_indices else None
    reads_before_write = sum(
        name in {"read_file", "list_directory"}
        for index, name in enumerate(names)
        if first_write is None or index < first_write
    )
    signatures = [
        f"{call['name']}:{json.dumps(call.get('arguments'), sort_keys=True, default=str)}"
        for call in calls
    ]
    repeats = sum(count - 1 for count in Counter(signatures).values() if count > 1)
    submitted = bool(submit_indices)
    verified_after_write = bool(
        write_indices and any(index > write_indices[-1] for index in run_indices)
    )
    premature_submission = bool(
        submit_indices and (not write_indices or submit_indices[0] < write_indices[0])
    )
    no_submit_perfect = reward >= 0.999999 and not submitted
    excessive_exploration = reads_before_write >= 10 or (
        names.count("read_file") + names.count("list_directory") >= 14
    )
    weak_planning = bool(strict_errors) or len(write_indices) >= 3
    missing_verification = bool(
        family in CODE_FAMILIES and write_indices and not verified_after_write
    )
    failed_recovery = bool(strict_errors and reward < 0.999999)
    context_loss = repeats >= 3
    horizon_exhaustion = bool(
        not submitted
        and (
            len(calls) >= 20
            or sum(message.get("role") == "assistant" for message in messages) >= 25
        )
    )
    wrong_evidence = bool(
        family in {"retrieval_workspace", "composite_workspace"}
        and reward < 0.9
        and len(write_indices) > 0
    )
    partial = 0.0 < reward < 0.999999
    task_id = metadata.get("task_id") or str(
        sample.get("problem_id") or sample.get("example_id") or ""
    )
    sample_id = sample.get("sample_id")
    if sample_id is None:
        sample_id = sample.get("rollout_number")
    return {
        "source": source,
        "label": label,
        "run_id": run_id,
        "step": step if step is not None else "",
        "sample_id": sample_id if sample_id is not None else "",
        "task_id": task_id,
        "split": metadata.get("split", ""),
        "family": family,
        "scenario": metadata.get("scenario", "unknown"),
        "difficulty": int(metadata.get("difficulty") or 0),
        "seed": int(metadata.get("seed") or 0),
        "reward": reward,
        "perfect": reward >= 0.999999,
        "partial": partial,
        "zero": reward <= 0.0,
        "submitted": submitted,
        "tool_count": len(calls),
        "read_count": names.count("read_file"),
        "list_count": names.count("list_directory"),
        "write_count": len(write_indices),
        "run_count": len(run_indices),
        "tool_error_count": len(strict_errors),
        "repeated_call_count": repeats,
        "reads_before_write": reads_before_write,
        "verified_after_write": verified_after_write,
        "wrong_evidence": wrong_evidence,
        "weak_planning": weak_planning,
        "excessive_exploration": excessive_exploration,
        "tool_error": bool(strict_errors),
        "context_loss": context_loss,
        "premature_submission": premature_submission,
        "missing_verification": missing_verification,
        "failed_recovery": failed_recovery,
        "evaluator_reward_mismatch": no_submit_perfect,
        "horizon_exhaustion": horizon_exhaustion,
        "tool_sequence": " -> ".join(names),
        "paths": " | ".join(filter(None, (path_argument(call) for call in calls))),
        "error_excerpt": re.sub(r"\s+", " ", strict_errors[0])[:240]
        if strict_errors
        else "",
        "assistant_tail": assistant_tail(messages),
    }


def load_rows(exports: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((exports / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for path in sorted((exports / "training").glob("*/step-*.json.gz")):
        label = path.parent.name
        match = re.search(r"step-(\d+)", path.name)
        step = int(match.group(1)) if match else None
        payload = load_gzip_json(path)
        run_id = str(payload.get("run_id") or "")
        for sample in payload.get("samples") or []:
            rows.append(
                normalize_sample(
                    sample,
                    source="training",
                    label=label,
                    run_id=run_id,
                    step=step,
                )
            )
    for path in sorted((exports / "evaluations").glob("*/page-*.json.gz")):
        label = path.parent.name
        payload = load_gzip_json(path)
        eval_id = str(payload.get("evaluation_id") or "")
        for sample in payload.get("samples") or []:
            rows.append(
                normalize_sample(
                    sample,
                    source="evaluation",
                    label=label,
                    run_id=eval_id,
                    step=None,
                )
            )
    return rows, manifest


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else 0.0


def aggregate(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result: list[dict[str, Any]] = []
    for group_key, group in sorted(
        groups.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        record = dict(zip(keys, group_key))
        record.update(
            {
                "n": len(group),
                "mean_reward": mean(float(row["reward"]) for row in group),
                "perfect_rate": mean(float(row["perfect"]) for row in group),
                "partial_rate": mean(float(row["partial"]) for row in group),
                "zero_rate": mean(float(row["zero"]) for row in group),
                "submit_rate": mean(float(row["submitted"]) for row in group),
                "mean_tool_count": mean(float(row["tool_count"]) for row in group),
                "mean_reads_before_write": mean(
                    float(row["reads_before_write"]) for row in group
                ),
                "tool_error_rate": mean(float(row["tool_error"]) for row in group),
                "excessive_exploration_rate": mean(
                    float(row["excessive_exploration"]) for row in group
                ),
                "missing_verification_rate": mean(
                    float(row["missing_verification"]) for row in group
                ),
                "evaluator_mismatch_rate": mean(
                    float(row["evaluator_reward_mismatch"]) for row in group
                ),
            }
        )
        result.append(record)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def final_training_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_step: dict[str, int] = {}
    for row in rows:
        if row["source"] != "training" or not isinstance(row["step"], int):
            continue
        max_step[row["label"]] = max(max_step.get(row["label"], -1), row["step"])
    return [
        row
        for row in rows
        if row["source"] == "training" and row["step"] == max_step.get(row["label"])
    ]


def behavior_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flags = [
        "wrong_evidence",
        "weak_planning",
        "excessive_exploration",
        "tool_error",
        "context_loss",
        "premature_submission",
        "missing_verification",
        "failed_recovery",
        "evaluator_reward_mismatch",
        "horizon_exhaustion",
    ]
    result: list[dict[str, Any]] = []
    for label, group_iter in _group_by(rows, "label"):
        group = list(group_iter)
        for flag in flags:
            flagged = [row for row in group if row[flag]]
            result.append(
                {
                    "label": label,
                    "flag": flag,
                    "count": len(flagged),
                    "rate": len(flagged) / len(group) if group else 0.0,
                    "flagged_mean_reward": mean(
                        float(row["reward"]) for row in flagged
                    ),
                }
            )
    return result


def _group_by(
    rows: list[dict[str, Any]], key: str
) -> Iterable[tuple[Any, Iterable[dict[str, Any]]]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return sorted(groups.items(), key=lambda item: str(item[0]))


def representative_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flags = [
        "wrong_evidence",
        "weak_planning",
        "excessive_exploration",
        "tool_error",
        "context_loss",
        "premature_submission",
        "missing_verification",
        "failed_recovery",
        "evaluator_reward_mismatch",
        "horizon_exhaustion",
    ]
    representatives: list[dict[str, Any]] = []
    seen: set[tuple[str, str, Any]] = set()
    for flag in flags:
        candidates = sorted(
            (row for row in rows if row[flag]),
            key=lambda row: (
                float(row["reward"]),
                -int(row["tool_count"]),
                row["task_id"],
            ),
        )
        for row in candidates[:3]:
            identity = (flag, row["label"], row["sample_id"])
            if identity in seen:
                continue
            seen.add(identity)
            representatives.append({"flag": flag, **row})
    return representatives


def load_training_usage(exports: Path) -> tuple[int, float]:
    tokens = 0
    cost = 0.0
    for path in sorted((exports / "training").glob("*/metadata.json.gz")):
        metadata = load_gzip_json(path)
        usage = metadata.get("usage") or {}
        tokens += int(usage.get("total_tokens") or 0)
        cost += float(usage.get("total_cost_usd") or 0.0)
    return tokens, cost


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    run_steps: list[dict[str, Any]],
    family: list[dict[str, Any]],
    behavior: list[dict[str, Any]],
    representatives: list[dict[str, Any]],
    tokens: int,
    cost: float,
) -> None:
    final_rows = final_training_rows(rows)
    final_metrics = aggregate(final_rows, ["label"])
    eval_metrics = aggregate(
        [row for row in rows if row["source"] == "evaluation"], ["label"]
    )
    report = [
        "# Qwen3.5-4B training-matrix offline trajectory analysis",
        "",
        f"Export timestamp: `{manifest.get('exported_at_utc', '')}`",
        "",
        "## Archive coverage",
        "",
        f"- Normalized trajectories: **{len(rows):,}**",
        f"- Training milestone trajectories: **{sum(row['source'] == 'training' for row in rows):,}**",
        f"- Hosted-evaluation trajectories: **{sum(row['source'] == 'evaluation' for row in rows):,}**",
        f"- Recorded training tokens: **{tokens:,}**",
        f"- Recorded training cost: **${cost:,.2f}**",
        "- Exporting and analysis launched no inference and consumed no training credits.",
        "",
        "Training coverage is a fixed milestone panel, not every rollout from every step. "
        "Hosted-evaluation exports include every sample the platform retained.",
        "",
        "## Final sampled training behavior",
        "",
        markdown_table(
            ["Run", "n", "Mean reward", "Perfect", "Submit", "Tools", "Explore flag"],
            [
                [
                    record["label"],
                    record["n"],
                    f"{record['mean_reward']:.3f}",
                    f"{record['perfect_rate']:.1%}",
                    f"{record['submit_rate']:.1%}",
                    f"{record['mean_tool_count']:.1f}",
                    f"{record['excessive_exploration_rate']:.1%}",
                ]
                for record in final_metrics
            ],
        ),
        "",
        "These are training-distribution samples. Near-perfect values indicate saturation and "
        "must not be interpreted as held-out generalization.",
        "",
        "## Existing hosted-evaluation results",
        "",
        markdown_table(
            ["Evaluation", "n", "Mean reward", "Perfect", "Partial", "Zero"],
            [
                [
                    record["label"],
                    record["n"],
                    f"{record['mean_reward']:.3f}",
                    f"{record['perfect_rate']:.1%}",
                    f"{record['partial_rate']:.1%}",
                    f"{record['zero_rate']:.1%}",
                ]
                for record in eval_metrics
            ],
        ),
        "",
        "## Behavioral findings",
        "",
        "1. **Specialist saturation is real.** Pipeline, script repair, and tabular converge to "
        "almost entirely perfect training batches, leaving little or no usable policy gradient.",
        "2. **Submission is not enforced by reward.** Several perfect trajectories never call "
        "`submit`, especially in the second all-family seed. This is an evaluator/reward mismatch.",
        "3. **Retrieval and composite work remain exploration-heavy.** Agents repeatedly list and "
        "read broad document trees before committing to authoritative evidence.",
        "4. **Tool errors are usually recoverable.** Missing files, rejected inline Python, and "
        "premature execution usually lead to a correction; failures persist when the correction "
        "loop reaches the horizon.",
        "5. **Verification remains uneven.** Some code trajectories submit after source edits "
        "without a successful post-edit public check. Retrieval writes are often not reread.",
        "6. **No strong context-loss signature dominates.** Most low rewards are better explained "
        "by evidence selection, weak action ordering, missing verification, or horizon exhaustion.",
        "",
        "## Family-level metrics",
        "",
        markdown_table(
            ["Source", "Run", "Family", "n", "Reward", "Perfect", "Submit"],
            [
                [
                    record["source"],
                    record["label"],
                    record["family"],
                    record["n"],
                    f"{record['mean_reward']:.3f}",
                    f"{record['perfect_rate']:.1%}",
                    f"{record['submit_rate']:.1%}",
                ]
                for record in family
            ],
        ),
        "",
        "## Interpretation constraints",
        "",
        "- Milestone batches are not matched task panels, so longitudinal reward changes are descriptive.",
        "- Behavioral flags are deterministic heuristics. They identify review candidates rather than ground truth labels.",
        "- `wrong_evidence` is only assigned to sub-0.9 retrieval/composite repairs and should be manually confirmed.",
        "- Final all-family and composition checkpoints still require the frozen held-out suite before transfer claims.",
        "",
        "## Representative review queue",
        "",
    ]
    for row in representatives[:20]:
        report.extend(
            [
                f"### {row['flag']}: `{row['label']}` / `{row['task_id']}`",
                "",
                f"- Reward: `{row['reward']:.6f}`; family: `{row['family']}`; "
                f"difficulty: `{row['difficulty']}`; submitted: `{row['submitted']}`",
                f"- Tool sequence: `{row['tool_sequence']}`",
                f"- Error: {row['error_excerpt'] or 'none'}",
                f"- Final reasoning: {row['assistant_tail'] or 'none'}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze locally exported Qwen3.5-4B matrix trajectories."
    )
    parser.add_argument(
        "--exports",
        type=Path,
        default=Path("analysis/qwen35-4b-offline/exports"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/qwen35-4b-offline/results"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/qwen35-matrix-offline-analysis.md"),
    )
    args = parser.parse_args()

    rows, manifest = load_rows(args.exports)
    if not rows:
        raise SystemExit(f"No trajectory samples found under {args.exports}")
    args.out.mkdir(parents=True, exist_ok=True)
    run_steps = aggregate(
        [row for row in rows if row["source"] == "training"], ["label", "step"]
    )
    family = aggregate(rows, ["source", "label", "family"])
    scenario = aggregate(rows, ["source", "label", "family", "scenario", "difficulty"])
    final_rows = final_training_rows(rows)
    behavior = behavior_counts(final_rows)
    representatives = representative_rows(final_rows)
    tokens, cost = load_training_usage(args.exports)

    write_csv(args.out / "normalized_samples.csv", rows)
    write_csv(args.out / "run_step_metrics.csv", run_steps)
    write_csv(args.out / "family_metrics.csv", family)
    write_csv(args.out / "scenario_difficulty_metrics.csv", scenario)
    write_csv(args.out / "behavior_flags_final.csv", behavior)
    write_csv(args.out / "representative_review_queue.csv", representatives)
    summary = {
        "schema_version": 1,
        "samples": len(rows),
        "training_samples": sum(row["source"] == "training" for row in rows),
        "evaluation_samples": sum(row["source"] == "evaluation" for row in rows),
        "training_tokens": tokens,
        "training_cost_usd": cost,
        "final_training": aggregate(final_rows, ["label"]),
        "evaluations": aggregate(
            [row for row in rows if row["source"] == "evaluation"], ["label"]
        ),
        "behavior_flags": behavior,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_report(
        args.report,
        rows,
        manifest,
        run_steps,
        family,
        behavior,
        representatives,
        tokens,
        cost,
    )
    print(f"normalized {len(rows):,} trajectories")
    print(f"wrote analysis tables to {args.out}")
    print(f"wrote report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
