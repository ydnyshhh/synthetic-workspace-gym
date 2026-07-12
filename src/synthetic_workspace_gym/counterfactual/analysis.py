from __future__ import annotations

import math
import statistics
from collections import defaultdict

from .schemas import BranchComparison, BranchOutcome


def aggregate_outcomes(outcomes: list[BranchOutcome], recoverable_threshold: float = .95, optimality_tolerance: float = .05) -> list[BranchComparison]:
    groups: dict[str, list[BranchOutcome]] = defaultdict(list)
    for outcome in outcomes: groups[outcome.branch_group_id].append(outcome)
    result = []
    for group_id, rows in groups.items():
        candidates: dict[str, list[BranchOutcome]] = defaultdict(list)
        for row in rows: candidates[row.candidate_id].append(row)
        stats = {}
        for candidate_id, values in candidates.items():
            returns = [x.final_reward for x in values]
            stats[candidate_id] = {"mean": statistics.fmean(returns), "std": statistics.pstdev(returns), "min": min(returns), "max": max(returns), "count": float(len(returns)), "success_rate": statistics.fmean(float(x.success) for x in values), "mean_steps": statistics.fmean(x.step_count for x in values)}
        original = next((x.candidate_id for x in rows if x.metadata.get("candidate_type") == "original"), rows[0].candidate_id)
        ranked = sorted(stats, key=lambda cid: (-stats[cid]["mean"], -stats[cid]["success_rate"], stats[cid]["std"], stats[cid]["mean_steps"]))
        best = ranked[0]; original_mean = stats[original]["mean"]; best_mean = stats[best]["mean"]; regret = max(0., best_mean - original_mean)
        confidence = _confidence(stats[original], stats[best], best_mean - original_mean)
        labels = _labels(rows, stats, original, regret, best_mean, recoverable_threshold, optimality_tolerance)
        result.append(BranchComparison(group_id, rows[0].snapshot_id, original, stats, best, original_mean, best_mean,
            best_mean - original_mean, regret, best_mean >= recoverable_threshold, original_mean >= best_mean - optimality_tolerance,
            confidence, labels, {"candidate_ranking": ranked}))
    return result


def _confidence(original: dict[str, float], best: dict[str, float], margin: float) -> float:
    n = min(original["count"], best["count"]); noise = original["std"] + best["std"]
    return round(min(1., (1 - math.exp(-n / 2)) * max(0., margin) / max(.05, max(0., margin) + noise)), 4)


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