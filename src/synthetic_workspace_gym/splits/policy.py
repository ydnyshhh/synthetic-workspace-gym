from __future__ import annotations

from collections.abc import Sequence

from synthetic_workspace_gym.generators.registry import list_generators

from .schemas import SplitSpec

IN_DISTRIBUTION_SCENARIOS: dict[str, list[str]] = {
    "tabular": ["monthly_segment_report", "channel_status_pivot", "weekly_refund_rollup"],
    "script_repair": ["inventory_report", "path_batch", "csv_schema_drift", "timestamp_normalization"],
    "pipeline": ["team_hours_pipeline", "sales_csv_pipeline", "artifact_stitch_pipeline"],
    "retrieval_workspace": ["service_config_reconciliation", "migration_plan_bundle", "incident_report_bundle"],
}

HELDOUT_SCENARIOS: dict[str, list[str]] = {
    "tabular": ["supplier_restock_summary"],
    "script_repair": ["team_roster_export"],
    "pipeline": ["quality_gate_pipeline"],
    "retrieval_workspace": ["client_adapter_sync"],
}


def default_split_policy(
    families: Sequence[str] | None = None,
    seed_offset: int = 0,
    train_seeds: range | Sequence[int] = range(0, 80),
    validation_seeds: range | Sequence[int] = range(80, 90),
    test_seeds: range | Sequence[int] = range(90, 100),
    heldout_seeds: range | Sequence[int] = range(100, 120),
) -> dict[str, SplitSpec]:
    selected = [str(family) for family in (families or list_generators())]
    return {
        "train": SplitSpec(
            name="train",
            families=selected,
            scenarios={family: in_distribution_scenarios_for_family(family) for family in selected},
            difficulties=[1, 2, 3],
            seeds=_offset(train_seeds, seed_offset),
            metadata={"policy": "default", "scenario_policy": "in_distribution"},
        ),
        "validation": SplitSpec(
            name="validation",
            families=selected,
            scenarios={family: in_distribution_scenarios_for_family(family) for family in selected},
            difficulties=[2, 3, 4],
            seeds=_offset(validation_seeds, seed_offset),
            metadata={"policy": "default", "scenario_policy": "in_distribution"},
        ),
        "test": SplitSpec(
            name="test",
            families=selected,
            scenarios={family: in_distribution_scenarios_for_family(family) for family in selected},
            difficulties=[3, 4, 5],
            seeds=_offset(test_seeds, seed_offset),
            metadata={"policy": "default", "scenario_policy": "in_distribution"},
        ),
        "heldout": SplitSpec(
            name="heldout",
            families=selected,
            scenarios={family: heldout_scenarios_for_family(family) for family in selected},
            difficulties=[3, 4, 5],
            seeds=_offset(heldout_seeds, seed_offset),
            metadata={"policy": "default", "scenario_policy": "scenario_heldout"},
        ),
    }


def scenario_pool_for_family(family: str) -> list[str]:
    return in_distribution_scenarios_for_family(family) + heldout_scenarios_for_family(family)


def heldout_scenarios_for_family(family: str) -> list[str]:
    return list(HELDOUT_SCENARIOS.get(str(family), []))


def in_distribution_scenarios_for_family(family: str) -> list[str]:
    return list(IN_DISTRIBUTION_SCENARIOS.get(str(family), []))


def _offset(values: range | Sequence[int], seed_offset: int) -> list[int]:
    return [int(value) + int(seed_offset) for value in values]
