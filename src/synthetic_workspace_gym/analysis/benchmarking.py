from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median
from typing import Callable

from synthetic_workspace_gym.schemas import EnvironmentManifest, EpisodeSummary


BenchmarkRow = dict[str, object]


def episode_to_row(summary: EpisodeSummary, manifest: EnvironmentManifest) -> BenchmarkRow:
    metadata = dict(manifest.metadata)
    scenario_profile = dict(metadata.get("scenario_profile", {}))
    return {
        "episode_id": summary.episode_id,
        "env_id": summary.env_id,
        "agent_name": summary.agent_name,
        "family": manifest.family.value,
        "difficulty": manifest.difficulty,
        "seed": manifest.seed,
        "scenario_id": str(
            metadata.get(
                "scenario_id",
                manifest.reference_solution.get(
                    "scenario_id",
                    metadata.get("task_descriptor", {}).get("scenario_id", "unknown"),
                ),
            )
        ),
        "repair_surface": scenario_profile.get("repair_surface"),
        "bug_scope": scenario_profile.get("bug_scope"),
        "failure_mode": scenario_profile.get("failure_mode"),
        "smoke_test_quality": scenario_profile.get("smoke_test_quality"),
        "task_type": scenario_profile.get("task_type"),
        "content_variant_id": scenario_profile.get("content_variant_id"),
        "document_count": scenario_profile.get("document_count"),
        "retrieval_hops": scenario_profile.get("retrieval_hops"),
        "evidence_distribution": scenario_profile.get("evidence_distribution"),
        "distractor_count": scenario_profile.get("distractor_count"),
        "staleness_pattern": scenario_profile.get("staleness_pattern"),
        "input_shape": scenario_profile.get("input_shape"),
        "time_bucketing": scenario_profile.get("time_bucketing"),
        "output_style": scenario_profile.get("output_style"),
        "success": bool(summary.evaluation.success),
        "score": float(summary.evaluation.score),
        "failure_labels": list(summary.evaluation.failure_labels),
        "subscores": {str(key): float(value) for key, value in summary.evaluation.subscores.items()},
        "step_count": int(summary.step_count),
        "duration_seconds": float(summary.duration_seconds),
        "submitted": bool(summary.submitted),
        "bug_labels": [
            str(label)
            for label in metadata.get(
                "bug_labels",
                manifest.reference_solution.get("bug_labels", []),
            )
        ],
        "artifact_root": summary.artifact_root,
    }


def compute_bucket_metrics(rows: list[BenchmarkRow]) -> dict[str, object]:
    if not rows:
        return {
            "count": 0,
            "success_rate": 0.0,
            "mean_score": 0.0,
            "median_score": 0.0,
            "perfect_rate": 0.0,
            "mean_step_count": 0.0,
            "mean_duration_seconds": 0.0,
            "failure_label_counts": {},
            "mean_subscores": {},
        }

    scores = [float(row["score"]) for row in rows]
    step_counts = [int(row["step_count"]) for row in rows]
    durations = [float(row["duration_seconds"]) for row in rows]
    successes = sum(1 for row in rows if bool(row["success"]))
    perfect = sum(1 for row in rows if float(row["score"]) == 1.0)

    failure_counts: Counter[str] = Counter()
    subscore_totals: defaultdict[str, float] = defaultdict(float)
    subscore_counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        failure_counts.update(str(label) for label in row["failure_labels"])
        for key, value in dict(row["subscores"]).items():
            subscore_totals[str(key)] += float(value)
            subscore_counts[str(key)] += 1

    mean_subscores = {
        key: round(subscore_totals[key] / subscore_counts[key], 6)
        for key in sorted(subscore_totals)
        if subscore_counts[key] > 0
    }
    return {
        "count": len(rows),
        "success_rate": round(successes / len(rows), 6),
        "mean_score": round(sum(scores) / len(scores), 6),
        "median_score": round(float(median(scores)), 6),
        "perfect_rate": round(perfect / len(rows), 6),
        "mean_step_count": round(sum(step_counts) / len(step_counts), 6),
        "mean_duration_seconds": round(sum(durations) / len(durations), 6),
        "failure_label_counts": {key: failure_counts[key] for key in sorted(failure_counts)},
        "mean_subscores": mean_subscores,
    }


def group_rows(rows: list[BenchmarkRow], key_fn: Callable[[BenchmarkRow], object]) -> dict[str, list[BenchmarkRow]]:
    groups: defaultdict[str, list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        key = key_fn(row)
        if key is None:
            continue
        key_text = str(key)
        if key_text == "":
            continue
        groups[key_text].append(row)
    return {key: groups[key] for key in sorted(groups)}


def build_benchmark_report(rows: list[BenchmarkRow]) -> dict[str, object]:
    agent_names = sorted({str(row["agent_name"]) for row in rows})
    report = {
        "agent": agent_names[0] if len(agent_names) == 1 else "mixed",
        "environment_count": len(rows),
        "rows": rows,
        "overall": compute_bucket_metrics(rows),
        "by_family": summarize_groups(rows, lambda row: row["family"]),
        "by_difficulty": summarize_groups(rows, lambda row: row["difficulty"]),
        "by_scenario_id": summarize_groups(rows, lambda row: row["scenario_id"]),
        "by_family_and_difficulty": summarize_groups(
            rows,
            lambda row: f"{row['family']}|difficulty={row['difficulty']}",
        ),
        "by_bug_scope": summarize_groups(rows, lambda row: row.get("bug_scope")),
        "by_failure_mode": summarize_groups(rows, lambda row: row.get("failure_mode")),
        "by_repair_surface": summarize_groups(rows, lambda row: row.get("repair_surface")),
        "by_smoke_test_quality": summarize_groups(rows, lambda row: row.get("smoke_test_quality")),
        "by_task_type": summarize_groups(rows, lambda row: row.get("task_type")),
        "by_content_variant_id": summarize_groups(rows, lambda row: row.get("content_variant_id")),
        "by_document_count": summarize_groups(rows, lambda row: row.get("document_count")),
        "by_retrieval_hops": summarize_groups(rows, lambda row: row.get("retrieval_hops")),
        "by_evidence_distribution": summarize_groups(rows, lambda row: row.get("evidence_distribution")),
        "by_distractor_count": summarize_groups(rows, lambda row: row.get("distractor_count")),
        "by_staleness_pattern": summarize_groups(rows, lambda row: row.get("staleness_pattern")),
        "by_input_shape": summarize_groups(rows, lambda row: row.get("input_shape")),
        "by_time_bucketing": summarize_groups(rows, lambda row: row.get("time_bucketing")),
        "by_output_style": summarize_groups(rows, lambda row: row.get("output_style")),
    }
    return report


def summarize_groups(rows: list[BenchmarkRow], key_fn: Callable[[BenchmarkRow], object]) -> dict[str, dict[str, object]]:
    grouped = group_rows(rows, key_fn)
    return {key: compute_bucket_metrics(grouped_rows) for key, grouped_rows in grouped.items()}
