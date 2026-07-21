from __future__ import annotations

import random

from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.generators.retrieval_profile_scenarios import (
    build_profiled_retrieval_scenario,
)


def test_retrieval_d5_profiles_have_distinct_scenarios_and_evidence_layouts() -> None:
    generator = get_generator("retrieval_workspace")
    scenarios = {}
    for seed in (100, 102, 107):
        spec = generator.sample_spec(
            difficulty=5,
            seed=seed,
            generation_params={"composition_mode": "hard_atomic"},
        )
        scenarios[seed] = build_profiled_retrieval_scenario(
            random.Random(f"{seed}:profiled_retrieval"), spec
        )

    assert scenarios[100]["scenario_id"] == "client_adapter_sync"
    assert "docs/pagination_contract.md" in scenarios[100]["files"]
    assert scenarios[102]["scenario_id"] == "client_adapter_policy_sync"
    assert {
        "docs/api_response_v3.md",
        "notes/warehouse_policy.md",
        "src/client_parser.py",
        "src/client_summary.py",
    } <= set(scenarios[102]["files"])
    assert scenarios[107]["scenario_id"] == "versioned_client_migration"
    assert {
        "release/current_manifest.json",
        "docs/api_v2.md",
        "docs/api_v3.md",
        "src/adapter.py",
        "src/serializer.py",
    } <= set(scenarios[107]["files"])
    assert {scenarios[seed]["structure"]["d5_profile"] for seed in scenarios} == {
        "d5_a",
        "d5_b",
        "d5_c",
    }
    assert len({frozenset(scenario["files"]) for scenario in scenarios.values()}) == 3