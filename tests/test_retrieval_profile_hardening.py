from __future__ import annotations

import random
import shutil
from pathlib import Path

import pytest

from synthetic_workspace_gym.evaluators.registry import get_evaluator
from synthetic_workspace_gym.evaluators.tabular_capability_program import (
    TabularCapabilityProgramEvaluator,
)
from synthetic_workspace_gym.generators.d5_profiles import (
    D5_RETRIEVAL_PROFILE_WEIGHTS,
    PIPELINE_D5_PROFILE_WEIGHTS,
    d5_profile_metadata_for_family,
    select_weighted_d5_profile,
)
from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.generators.retrieval_profile_scenarios import (
    CAPABILITY_WEIGHTS,
    build_profiled_retrieval_scenario,
)
from synthetic_workspace_gym.generators.tabular_capability_fixtures import (
    build_focused_capability_assets,
)
from synthetic_workspace_gym.prime.dataset import SyntheticWorkspacePrimeDataset
from test_support import workspace_tempdir


EXPECTED_TABULAR_WEIGHTS = {
    "script_exists": 0.03,
    "script_executes": 0.05,
    "valid_json": 0.04,
    "output_schema": 0.05,
    "active_coercion": 0.08,
    "fractional_aggregation": 0.08,
    "canonical_identity": 0.12,
    "deduplication": 0.13,
    "timestamp_normalization": 0.10,
    "temporal_status_join": 0.12,
    "hidden_end_to_end": 0.15,
    "determinism": 0.05,
}


def test_tabular_calibration_is_frozen() -> None:
    assert TabularCapabilityProgramEvaluator.WEIGHTS == EXPECTED_TABULAR_WEIGHTS
    assets = build_focused_capability_assets()
    assert {item["capability"] for item in assets["focused_fixtures"]} == {
        "active_coercion",
        "fractional_aggregation",
        "canonical_identity",
        "deduplication",
        "timestamp_normalization",
        "temporal_status_join",
    }


def test_family_specific_profile_distributions_are_exact() -> None:
    assignments = [select_weighted_d5_profile(5, seed).profile_id for seed in range(100)]
    realized = {
        profile: assignments.count(profile) / len(assignments)
        for profile in D5_RETRIEVAL_PROFILE_WEIGHTS
    }
    assert realized == D5_RETRIEVAL_PROFILE_WEIGHTS
    assert realized == PIPELINE_D5_PROFILE_WEIGHTS
    assert d5_profile_metadata_for_family("tabular", 5, 2)["profile"] == "d5_a"
    assert d5_profile_metadata_for_family("pipeline", 5, 2)["profile"] == "d5_b"
    assert d5_profile_metadata_for_family("retrieval_workspace", 5, 7)["profile"] == "d5_c"


def test_retrieval_profiles_route_to_distinct_scenarios_and_task_ids() -> None:
    rows = SyntheticWorkspacePrimeDataset(
        families=("retrieval_workspace",),
        difficulties=(5,),
        seeds=(100, 102, 107),
        split="test",
    ).to_list()
    assert [(row["scenario"], row["metadata"]["profile"]) for row in rows] == [
        ("client_adapter_sync", "d5_a"),
        ("client_adapter_policy_sync", "d5_b"),
        ("versioned_client_migration", "d5_c"),
    ]
    assert rows[1]["task_id"].endswith(
        ".client_adapter_policy_sync.d5.d5_b.s102"
    )


def test_retrieval_profiles_have_distinct_artifact_and_evidence_surfaces() -> None:
    generator = get_generator("retrieval_workspace")
    scenarios: dict[int, dict[str, object]] = {}
    for seed in (100, 102, 107):
        spec = generator.sample_spec(
            difficulty=5,
            seed=seed,
            generation_params={"composition_mode": "hard_atomic"},
        )
        scenarios[seed] = build_profiled_retrieval_scenario(
            random.Random(f"{seed}:profiled_retrieval"), spec
        )

    assert set(scenarios[100]["reference_solution_files"]) == {
        "src/client_adapter.py",
        "config/adapter_contract.json",
    }
    assert set(scenarios[102]["reference_solution_files"]) == {
        "config/client_runtime.json",
        "src/client_parser.py",
        "src/client_summary.py",
    }
    assert set(scenarios[107]["reference_solution_files"]) == {
        "config/client.json",
        "src/adapter.py",
        "src/serializer.py",
    }
    assert "release/current_manifest.json" in scenarios[107]["files"]
    assert "notes/legacy_rollout.md" in scenarios[102]["files"]
    assert len({frozenset(scenario["files"]) for scenario in scenarios.values()}) == 3


@pytest.mark.parametrize("seed", [102, 107])
def test_hard_retrieval_oracle_staircase(seed: int) -> None:
    generator = get_generator("retrieval_workspace")
    spec = generator.sample_spec(
        difficulty=5,
        seed=seed,
        generation_params={"composition_mode": "hard_atomic"},
    )
    scenario = build_profiled_retrieval_scenario(
        random.Random(f"{seed}:profiled_retrieval"), spec
    )
    labels = {str(bug["label"]) for bug in scenario["bugs"]}
    states = [
        ("untouched", set()),
        ("version_config_only", {"authority_config"}),
        ("parser_only", {"quantity_parsing", "missing_value_policy"}),
        (
            "parser_config",
            {"authority_config", "quantity_parsing", "missing_value_policy"},
        ),
        ("all_except_edge", labels - {"output_contract"}),
        ("full", labels),
    ]
    with workspace_tempdir() as tmp:
        root = Path(tmp)
        bundle = generator.generate_instance(spec, root / "generated")
        evaluator = get_evaluator(
            "retrieval_workspace",
            evaluator_entrypoint=bundle.manifest.evaluator_entrypoint,
        )
        rewards = {
            name: _evaluate_state(
                root / name,
                fixed,
                scenario,
                bundle.visible_root,
                evaluator,
                bundle.manifest,
                bundle.hidden_root,
            )
            for name, fixed in states
        }

    assert rewards["untouched"] <= 0.15
    assert 0.15 <= rewards["version_config_only"] <= 0.30
    assert 0.25 <= rewards["parser_only"] <= 0.45
    assert 0.40 <= rewards["parser_config"] <= 0.65
    assert 0.65 <= rewards["all_except_edge"] <= 0.85
    assert rewards["full"] == 1.0
    assert list(rewards.values()) == sorted(rewards.values())
    assert len(set(rewards.values())) == len(rewards)
    assert CAPABILITY_WEIGHTS == {
        "authority_resolution": 0.10,
        "schema_mapping": 0.15,
        "quantity_parsing": 0.10,
        "pagination": 0.10,
        "missing_value_policy": 0.10,
        "regional_override": 0.10,
        "deduplication": 0.10,
        "timestamp_resolution": 0.10,
        "output_contract": 0.05,
        "hidden_generalization": 0.10,
    }


def _evaluate_state(
    workspace: Path,
    fixed: set[str],
    scenario: dict[str, object],
    visible_root: Path,
    evaluator,
    manifest,
    hidden_root: Path,
) -> float:
    shutil.copytree(visible_root, workspace)
    files = dict(scenario["correct_files"])
    for bug in scenario["bugs"]:
        if str(bug["label"]) not in fixed:
            target = str(bug["target_path"])
            files[target] = bug["apply"](files[target])
    for relative_path, content in files.items():
        target = workspace / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
    return evaluator.evaluate(workspace, manifest, hidden_root).score