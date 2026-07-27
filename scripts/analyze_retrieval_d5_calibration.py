from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from pathlib import Path
from typing import Any


EVALUATION_IDS = (
    "fok4hhnaldvoyjkk19lnctdf",
    "n2hhzed1awwrl36vciz4vt36",
    "wcbeiety5qskuncwcq88ikvj",
    "nm0h8x9pj7sxnyefopkvjvd1",
    "aizdt396t5rq9gcjihpd2icw",
    "fbfd6n2wcm457e1mo4n7w3vz",
    "y8msxuqls7d04im6slpektzd",
    "roiljd7t7bj0lpubrw34vgiv",
    "pdi5sepr8ttxr1uuicvslzu1",
)
WHEEL_SHA256 = "a7012f4c6e97fd71759cba9c3da64b9a94674f5765ee467301e54b5055052518"
PROFILE_SEEDS = {
    "A": {100, 101, 110, 111, 120},
    "B": {102, 103, 104, 105, 106},
    "C": {107, 108, 109, 117, 118},
}
PROFILE_TARGETS = {
    "A": {"mean": (0.60, 0.80), "perfect": (0.40, 0.70)},
    "B": {"mean": (0.35, 0.60), "perfect": (0.10, 0.40)},
    "C": {"mean": (0.20, 0.45), "perfect": (0.00, 0.20)},
    "combined": {"mean": (0.40, 0.65), "perfect": (0.10, 0.40)},
}
TASK_PATTERN = re.compile(r"- task_id: ([^\n]+)")
PROFILE_PATTERN = re.compile(r"\.d5_(a|b|c)\.s(\d+)")


def _tool_names(sample: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for message in sample.get("completion") or []:
        for raw in message.get("tool_calls") or []:
            try:
                call = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            name = call.get("name")
            if name:
                names.append(str(name))
    return names


def _row(evaluation_id: str, sample: dict[str, Any]) -> dict[str, Any]:
    prompt = "\n".join(str(message.get("content") or "") for message in sample.get("prompt") or [])
    task_match = TASK_PATTERN.search(prompt)
    if task_match is None:
        raise ValueError(f"{evaluation_id}: sample has no task_id")
    task_id = task_match.group(1)
    profile_match = PROFILE_PATTERN.search(task_id)
    if profile_match is None:
        raise ValueError(f"{evaluation_id}: cannot parse profile from {task_id}")
    tools = _tool_names(sample)
    token_usage = (sample.get("info") or {}).get("token_usage") or {}
    reward = sample.get("reward")
    if reward is None:
        reward = sample.get("swg_reward")
    return {
        "evaluation_id": evaluation_id,
        "example_id": sample.get("example_id"),
        "rollout_number": sample.get("rollout_number"),
        "task_id": task_id,
        "profile": profile_match.group(1).upper(),
        "seed": int(profile_match.group(2)),
        "reward": float(reward),
        "tool_steps": len(tools),
        "tool_counts": dict(collections.Counter(tools)),
        "input_tokens": float(token_usage.get("input_tokens") or 0),
        "output_tokens": float(token_usage.get("output_tokens") or 0),
        "total_tokens": float(token_usage.get("input_tokens") or 0)
        + float(token_usage.get("output_tokens") or 0),
        "total_time_seconds": float(sample.get("total_time") or 0),
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [row["reward"] for row in rows]
    return {
        "rollouts": len(rows),
        "mean_reward": statistics.mean(rewards),
        "perfect_rate": sum(reward == 1.0 for reward in rewards) / len(rows),
        "partial_rate": sum(0.0 < reward < 1.0 for reward in rewards) / len(rows),
        "mean_tool_steps": statistics.mean(row["tool_steps"] for row in rows),
        "mean_input_tokens": statistics.mean(row["input_tokens"] for row in rows),
        "mean_output_tokens": statistics.mean(row["output_tokens"] for row in rows),
        "mean_total_tokens": statistics.mean(row["total_tokens"] for row in rows),
        "reward_distribution": dict(
            sorted(collections.Counter(str(row["reward"]) for row in rows).items())
        ),
    }


def _in_range(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-root",
        type=Path,
        default=Path("analysis/retrieval-d5-calibration"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/retrieval-d5-calibration-report.json"),
    )
    parser.add_argument("--evaluation-id", action="append", default=[])
    parser.add_argument(
        "--environment",
        default="yadnyesh/synthetic-workspace-gym@0.1.27",
    )
    parser.add_argument("--wheel-sha256", default=WHEEL_SHA256)
    parser.add_argument("--manual-inspection", type=Path)
    args = parser.parse_args()
    evaluation_ids = tuple(args.evaluation_id) or EVALUATION_IDS
    manual_inspection: list[dict[str, Any]] = []
    if args.manual_inspection is not None:
        manual_inspection = list(
            json.loads(args.manual_inspection.read_text(encoding="utf-8"))
        )
    manual_unambiguous = bool(manual_inspection) and all(
        not bool(item.get("ambiguous_requirement")) for item in manual_inspection
    )

    rows: list[dict[str, Any]] = []
    for evaluation_id in evaluation_ids:
        sample_path = args.export_root / evaluation_id / "samples.json"
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        rows.extend(_row(evaluation_id, sample) for sample in payload["samples"])

    coverage = collections.Counter((row["profile"], row["seed"]) for row in rows)
    expected_cells = {
        (profile, seed): 3 for profile, seeds in PROFILE_SEEDS.items() for seed in seeds
    }
    if coverage != collections.Counter(expected_cells):
        raise ValueError(f"rollout coverage mismatch: {dict(coverage)}")

    profile_metrics = {
        profile: _metrics([row for row in rows if row["profile"] == profile])
        for profile in PROFILE_SEEDS
    }
    combined = _metrics(rows)
    all_metrics = {**profile_metrics, "combined": combined}
    reward_counts = collections.Counter(row["reward"] for row in rows)
    distinct_rewards = sorted(reward_counts)
    largest_bucket_rate = max(reward_counts.values()) / len(rows)
    b_c_delta = profile_metrics["B"]["mean_reward"] - profile_metrics["C"]["mean_reward"]
    c_b_tool_ratio = (
        profile_metrics["C"]["mean_tool_steps"] / profile_metrics["B"]["mean_tool_steps"]
    )
    c_b_token_ratio = (
        profile_metrics["C"]["mean_total_tokens"] / profile_metrics["B"]["mean_total_tokens"]
    )

    gates: dict[str, bool] = {}
    for profile, targets in PROFILE_TARGETS.items():
        gates[f"{profile}_mean_reward"] = _in_range(
            all_metrics[profile]["mean_reward"], targets["mean"]
        )
        gates[f"{profile}_perfect_rate"] = _in_range(
            all_metrics[profile]["perfect_rate"], targets["perfect"]
        )
    gates.update(
        {
            "b_mean_at_least_10_points_above_c": b_c_delta >= 0.10,
            "at_least_six_reward_values": len(distinct_rewards) >= 6,
            "at_least_half_partial_credit": combined["partial_rate"] >= 0.50,
            "no_reward_bucket_above_40_percent": largest_bucket_rate <= 0.40,
            "c_at_least_10_percent_more_tool_steps_than_b": c_b_tool_ratio >= 1.10,
            "c_at_least_10_percent_more_tokens_than_b": c_b_token_ratio >= 1.10,
            "manual_requirements_unambiguous": manual_unambiguous,
        }
    )

    report = {
        "schema_version": "retrieval-d5-qwen35-calibration-v1",
        "release_candidate": {
            "environment": args.environment,
            "wheel_sha256": args.wheel_sha256,
            "evaluator_version": "swg-capability-evaluators-v2",
            "horizon_unit": "tool_steps",
            "max_turns": 25,
            "max_tool_steps": 64,
        },
        "model": {
            "id": "Qwen/Qwen3.5-4B",
            "temperature": 0.7,
            "max_tokens": 32768,
        },
        "coverage": {
            "rollouts": len(rows),
            "cells": [
                {"profile": profile, "seed": seed, "rollouts": count}
                for (profile, seed), count in sorted(coverage.items())
            ],
            "evaluation_ids": list(evaluation_ids),
        },
        "metrics": all_metrics,
        "distribution": {
            "distinct_rewards": distinct_rewards,
            "largest_bucket_rate": largest_bucket_rate,
            "b_minus_c_mean_reward": b_c_delta,
            "c_over_b_tool_step_ratio": c_b_tool_ratio,
            "c_over_b_token_ratio": c_b_token_ratio,
        },
        "gates": gates,
        "passed": all(gates.values()),
        "manual_inspection": manual_inspection,
        "decision": {
            "publish_0_1_28": all(gates.values()),
            "reason": (
                "All calibration and manual-inspection gates passed."
                if all(gates.values())
                else "At least one calibration gate failed; do not freeze or publish 0.1.28."
            ),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "gates": gates, "metrics": all_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
