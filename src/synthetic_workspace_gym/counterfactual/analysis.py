from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any

from .schemas import BranchComparison, BranchOutcome


def aggregate_outcomes(outcomes: list[BranchOutcome], recoverable_threshold: float = .95, optimality_tolerance: float = .05) -> list[BranchComparison]:
    groups: dict[str, list[BranchOutcome]] = defaultdict(list)
    for outcome in outcomes:
        groups[outcome.branch_group_id].append(outcome)
    result = []
    for group_id, rows in groups.items():
        candidates: dict[str, list[BranchOutcome]] = defaultdict(list)
        for row in rows:
            candidates[row.candidate_id].append(row)
        stats = {}
        for candidate_id, values in candidates.items():
            returns = [x.final_reward for x in values]
            std = statistics.pstdev(returns)
            stats[candidate_id] = {
                "mean": statistics.fmean(returns), "std": std, "standard_error": std / math.sqrt(len(returns)),
                "min": min(returns), "max": max(returns), "count": float(len(returns)),
                "success_rate": statistics.fmean(float(x.success) for x in values),
                "mean_steps": statistics.fmean(x.step_count for x in values),
            }
        original_ids = {row.candidate_id for row in rows if row.metadata.get("candidate_type") == "original"}
        if len(original_ids) != 1:
            raise ValueError(f"branch group {group_id!r} requires exactly one original candidate; found {sorted(original_ids)}")
        original = next(iter(original_ids))
        ranked = sorted(stats, key=lambda cid: (-stats[cid]["mean"], -stats[cid]["success_rate"], stats[cid]["std"], stats[cid]["mean_steps"]))
        best = ranked[0]
        original_mean = stats[original]["mean"]
        best_mean = stats[best]["mean"]
        regret = max(0., best_mean - original_mean)
        differences = {candidate_id: _difference_statistics(candidates[original], values, group_id, candidate_id)
                       for candidate_id, values in candidates.items() if candidate_id != original}
        labels = _labels(rows, stats, original, regret, best_mean, recoverable_threshold, optimality_tolerance)
        result.append(BranchComparison(
            group_id, rows[0].snapshot_id, original, stats, best, original_mean, best_mean,
            best_mean - original_mean, regret, best_mean >= recoverable_threshold,
            original_mean >= best_mean - optimality_tolerance,
            differences.get(best, {}).get("probability_superior"), labels,
            {"candidate_ranking": ranked, "difference_statistics": differences,
             "paired_difference_statistics": differences,
             "root_trajectory_ids": sorted({str(row.metadata.get("root_trajectory_id")) for row in rows if row.metadata.get("root_trajectory_id")}),
             "confidence_note": "Explicit pair_id values use a paired bootstrap; otherwise probability uses an independent two-sample bootstrap."},
        ))
    return result


def _difference_statistics(original: list[BranchOutcome], candidate: list[BranchOutcome], group_id: str, candidate_id: str) -> dict[str, Any]:
    original_by_pair = {row.pair_id: row.final_reward for row in original if row.pair_id is not None}
    candidate_by_pair = {row.pair_id: row.final_reward for row in candidate if row.pair_id is not None}
    shared = sorted(set(original_by_pair) & set(candidate_by_pair))
    rng = random.Random(f"{group_id}:{candidate_id}:bootstrap-v2")
    completely_paired = (
        bool(shared) and len(original_by_pair) == len(original)
        and len(candidate_by_pair) == len(candidate)
        and set(original_by_pair) == set(candidate_by_pair)
    )
    if completely_paired:
        differences = [candidate_by_pair[pair_id] - original_by_pair[pair_id] for pair_id in shared]
        bootstrap = sorted(statistics.fmean(rng.choice(differences) for _ in differences) for _ in range(1000))
        std = statistics.pstdev(differences)
        return {
            "method": "paired_bootstrap", "paired_count": float(len(differences)),
            "original_count": float(len(original)), "candidate_count": float(len(candidate)),
            "mean_difference": statistics.fmean(differences),
            "standard_error": std / math.sqrt(len(differences)), "ci_low": bootstrap[24], "ci_high": bootstrap[974],
            "probability_superior": statistics.fmean(float(value > 0) for value in bootstrap),
        }
    original_returns = [row.final_reward for row in original]
    candidate_returns = [row.final_reward for row in candidate]
    bootstrap = sorted(
        statistics.fmean(rng.choice(candidate_returns) for _ in candidate_returns)
        - statistics.fmean(rng.choice(original_returns) for _ in original_returns)
        for _ in range(1000)
    )
    original_variance = statistics.pvariance(original_returns) if len(original_returns) > 1 else 0.0
    candidate_variance = statistics.pvariance(candidate_returns) if len(candidate_returns) > 1 else 0.0
    return {
        "method": "independent_bootstrap", "paired_count": 0.0,
        "original_count": float(len(original_returns)), "candidate_count": float(len(candidate_returns)),
        "mean_difference": statistics.fmean(candidate_returns) - statistics.fmean(original_returns),
        "standard_error": math.sqrt(original_variance / len(original_returns) + candidate_variance / len(candidate_returns)),
        "ci_low": bootstrap[24], "ci_high": bootstrap[974],
        "probability_superior": statistics.fmean(float(value > 0) for value in bootstrap),
    }


def _labels(rows: list[BranchOutcome], stats: dict[str, dict[str, float]], original_id: str, regret: float, best: float, threshold: float, tolerance: float) -> list[str]:
    labels = []
    if regret >= .2: labels.append("action_selection_failure")
    if best >= threshold and regret > 0: labels.append("recoverable_error")
    if best < threshold: labels.append("capability_limited")
    candidate_types = {row.candidate_id: row.metadata.get("candidate_type") for row in rows}
    original_mean = stats[original_id]["mean"]
    check_means = [stats[candidate_id]["mean"] for candidate_id, kind in candidate_types.items() if kind == "run_public_check"]
    submit_means = [stats[candidate_id]["mean"] for candidate_id, kind in candidate_types.items() if kind == "submit"]
    if check_means and max(check_means) > original_mean + tolerance: labels.append("verification_failure")
    if submit_means and max(submit_means) >= best - tolerance and max(submit_means) > original_mean + tolerance: labels.append("budget_allocation_failure")
    return labels
