from __future__ import annotations

from synthetic_workspace_gym.analysis.calibration import (
    build_monotonicity_report,
    capability_pass_rates,
)
from synthetic_workspace_gym.analysis.benchmarking import compute_bucket_metrics
from synthetic_workspace_gym.evaluators.capabilities import (
    CapabilityScore,
    capability_diagnostics,
    capability_subscores,
    weighted_capability_score,
)
from synthetic_workspace_gym.generators.d5_profiles import (
    D5_PROFILE_WEIGHTS,
    select_d5_profile,
)
from synthetic_workspace_gym.prime.dataset import SyntheticWorkspacePrimeDataset


def test_weighted_capability_score_clamps_and_preserves_names() -> None:
    capabilities = [
        CapabilityScore("normalization", 1.5, 0.25, "complete"),
        CapabilityScore("aggregation", -0.5, 0.75, "missing"),
    ]
    assert weighted_capability_score(capabilities) == 0.25
    assert capability_subscores(capabilities) == {
        "capability_normalization": 1.0,
        "capability_aggregation": 0.0,
    }
    assert capability_diagnostics(capabilities) == {
        "normalization": "complete",
        "aggregation": "missing",
    }


def test_weighted_capability_score_requires_positive_total_weight() -> None:
    try:
        weighted_capability_score([CapabilityScore("x", 1.0, 0.0)])
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("zero total weight should fail")


def test_monotonicity_report_detects_regressions_and_exact_full() -> None:
    healthy = build_monotonicity_report(
        task_id="example",
        ordered_states=[
            ("untouched", 0.1),
            ("one_fix", 0.3),
            ("all_but_one", 0.8),
            ("full", 1.0),
        ],
    )
    assert healthy["monotonic"] is True
    broken = build_monotonicity_report(
        task_id="broken",
        ordered_states=[("untouched", 0.2), ("one_fix", 0.1), ("full", 1.0)],
    )
    assert broken["monotonic"] is False
    assert broken["violations"]


def test_d5_profile_selection_is_deterministic_and_exact_over_100_seeds() -> None:
    assignments = [select_d5_profile(5, seed).profile_id for seed in range(100)]
    assert assignments == [select_d5_profile(5, seed).profile_id for seed in range(100)]
    assert {name: assignments.count(name) / 100 for name in D5_PROFILE_WEIGHTS} == (
        D5_PROFILE_WEIGHTS
    )
    assert select_d5_profile(4, 10) is None


def test_profile_is_recorded_in_d5_task_id_and_metadata() -> None:
    row = SyntheticWorkspacePrimeDataset(
        families=("retrieval_workspace",),
        difficulties=(5,),
        seeds=(104,),
        split="test",
    ).to_list()[0]
    assert row["task_id"].endswith(".d5.d5_b.s104")
    assert row["metadata"]["profile"] == "d5_b"


def test_distribution_metrics_and_capability_pass_rates() -> None:
    rows = [
        _row(0.0, 0.0),
        _row(0.5, 1.0),
        _row(1.0, 1.0),
        _row(0.5, 0.0),
    ]
    metrics = compute_bucket_metrics(rows)
    assert metrics["mean_reward"] == 0.5
    assert metrics["distinct_reward_count"] == 3
    assert metrics["zero_reward_rate"] == 0.25
    assert metrics["perfect_reward_rate"] == 0.25
    assert metrics["partial_reward_rate"] == 0.5
    assert metrics["largest_reward_bucket_rate"] == 0.5
    assert metrics["capability_pass_rates"] == {"normalization": 0.5}
    assert capability_pass_rates(rows) == {"normalization": 0.5}


def _row(score: float, capability: float) -> dict[str, object]:
    return {
        "score": score,
        "success": score == 1.0,
        "step_count": 1,
        "duration_seconds": 0.1,
        "failure_labels": [],
        "subscores": {"capability_normalization": capability},
    }
